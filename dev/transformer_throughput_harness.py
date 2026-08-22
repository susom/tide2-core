"""Developer throughput and GPU-memory measurement harness for TransformerCore.

This is a **dev-only** tool. It is deliberately placed under ``dev/`` and is
**not** imported by any product module and **not** run in CI. It exists to give a
repeatable way to measure, on a real GPU box, how the transformer inference hot
path (:meth:`tide2.transformers.core.TransformerCore.infer_raw_direct`) behaves
for a given combination of (model, batch size, padding strategy, compile mode).

It reports, per combination:
  * steady-state throughput (texts/sec), measured after warmup passes and
    averaged over several timed passes;
  * an estimated tokenize-vs-forward split (tokenize timed separately, forward
    inferred as full-pass minus tokenize);
  * all four GPU-memory views that matter for the OOM-recovery investigation:
      - ``torch.cuda.memory_allocated``   (live tensor bytes),
      - ``torch.cuda.max_memory_allocated`` (peak live bytes in the timed region),
      - ``torch.cuda.memory_reserved``    (the caching-allocator / CUDA-graph pool
        that the OOM leak actually grows),
      - ``torch.cuda.mem_get_info``       (free / total device footprint).

Why this lives outside product code:
  * ``TransformerCore._forward_batch_direct`` hardcodes ``padding=True`` (longest)
    and the product batch pipeline no longer supports ``torch.compile`` at all —
    it runs eager (compile was measured to add nothing at batch scale and its only
    wired mode leaked reserved VRAM). To sweep padding and compile modes for
    measurement anyway, this harness applies two clearly-scoped local hacks (a
    tokenizer call-proxy and a direct ``torch.compile`` on ``core._model``) that
    must never leak into product code.

Running (must be a GPU box):
    python dev/transformer_throughput_harness.py \
        --model StanfordAIMI/stanford-deidentifier-v2 \
        --batch-size 8,16,32 --num-texts 256 --seq-len 512 --max-chars 2000 \
        --padding longest --compile-mode eager --warmup 2 --iters 5

    # real notes from a parquet source, max_length padding, dynamic compile:
    python dev/transformer_throughput_harness.py \
        --input /data/notes.parquet --batch-size 16 --num-texts 512 \
        --padding max_length --compile-mode dynamic

If CUDA is unavailable and a CUDA device was requested, the harness prints a
clear message and self-skips (exit 0). Pass ``--device cpu`` to force a CPU run
(GPU-memory columns then read ``n/a``).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Pre-import pyarrow before the deep presidio/transformers import chain to avoid a
# native-init segfault (same workaround the docs build uses; see CLAUDE.md).
import pyarrow  # noqa: F401, ICN001
import torch

from tide2.transformers.core import TransformerCore

# A clinical-sounding, PHI-dense filler so the NER model actually emits entities.
_CLINICAL_LOREM = (
    "The patient John Q. Doe (MRN 1234567, DOB 03/14/1975) was admitted on "
    "02/08/2021 to Stanford Hospital under Dr. Emily Carter for evaluation of "
    "chest pain radiating to the left arm. He lives at 42 Elm Street, Palo Alto, "
    "CA 94301 and can be reached at (650) 555-0142 or johndoe@example.com. Home "
    "medications include metoprolol 25 mg twice daily and atorvastatin 40 mg "
    "nightly. Vitals on arrival: BP 148/92, HR 96, SpO2 97% on room air. Troponin "
    "was mildly elevated; serial ECGs were obtained and cardiology was consulted. "
)

# Candidate note-text column names, checked case-insensitively in this order.
_TEXT_COLUMN_HINTS = ("note_text", "text", "note", "body", "report")

# torch.OutOfMemoryError exists on newer torch; older builds raise RuntimeError.
_OOM_EXCEPTIONS: tuple[type[BaseException], ...] = (
    (torch.OutOfMemoryError, RuntimeError) if hasattr(torch, "OutOfMemoryError") else (RuntimeError,)
)

_BYTES_PER_MB = 1024 * 1024

_ROW_FMT = (
    "{tag:<7}{n:>7}{batch:>7}{seqmax:>15}{padding:>11}{compile:>16}"
    "{texts_s:>10}{tok:>12}{fwd:>12}{alloc:>13}{resv:>13}{peak:>10}{free:>10}{total:>10}"
)


class _PaddingTokenizerProxy:
    """Wrap a tokenizer to force a padding strategy on every ``__call__``.

    ``TransformerCore._forward_batch_direct`` always calls the tokenizer with
    ``padding=True`` (pad to the longest sequence in the batch). To benchmark
    ``max_length`` padding without touching product code, this proxy rewrites the
    padding-related kwargs on each call and delegates everything else to the real
    tokenizer.

    Attributes:
        wrapped: The underlying HuggingFace tokenizer.
        padding: Either ``"longest"`` or ``"max_length"``.
        max_length: Target length used when ``padding == "max_length"``.
    """

    def __init__(self, wrapped: Any, padding: str, max_length: int) -> None:
        """Store the wrapped tokenizer and the desired padding strategy.

        Args:
            wrapped: The real tokenizer to delegate to.
            padding: ``"longest"`` or ``"max_length"``.
            max_length: Sequence length used for ``max_length`` padding.
        """
        self.wrapped = wrapped
        self.padding = padding
        self.max_length = max_length

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Force the configured padding strategy, then delegate to the tokenizer.

        Returns:
            The tokenizer's batch-encoding output.
        """
        if self.padding == "max_length":
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = self.max_length
            kwargs["truncation"] = True
        else:
            kwargs["padding"] = "longest"
        return self.wrapped(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped tokenizer.

        Args:
            name: Attribute name not found on the proxy itself.

        Returns:
            The corresponding attribute of the wrapped tokenizer.
        """
        return getattr(self.wrapped, name)


def _is_oom_error(exc: BaseException) -> bool:
    """Return True if the exception represents a CUDA out-of-memory condition.

    Args:
        exc: The caught exception.

    Returns:
        True for ``torch.OutOfMemoryError`` or a RuntimeError whose message
        mentions running out of memory; False otherwise.
    """
    if hasattr(torch, "OutOfMemoryError") and isinstance(exc, torch.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _device_index(device: str) -> int:
    """Resolve the integer CUDA device index from a device string.

    Args:
        device: A device string such as ``"cuda:0"`` or ``"cuda"``.

    Returns:
        The CUDA device index.
    """
    if ":" in device:
        return int(device.split(":")[1])
    return torch.cuda.current_device()


def _pick_text_column(columns: list[str]) -> str:
    """Choose the most note-text-like column from a set of column names.

    Args:
        columns: Available column names.

    Returns:
        The chosen column name.

    Raises:
        ValueError: If no plausible text column is found.
    """
    lowered = {c.lower(): c for c in columns}
    for hint in _TEXT_COLUMN_HINTS:
        for low, original in lowered.items():
            if hint in low:
                return original
    raise ValueError(
        f"Could not find a note-text-like column in {list(columns)!r}; "
        f"expected one containing any of {_TEXT_COLUMN_HINTS}."
    )


def _cycle_to_length(texts: list[str], num_texts: int) -> list[str]:
    """Repeat/trim a list of texts to exactly ``num_texts`` items.

    Args:
        texts: Source texts (must be non-empty).
        num_texts: Desired total number of texts.

    Returns:
        A list of exactly ``num_texts`` texts.
    """
    if not texts:
        raise ValueError("No input texts available to cycle from.")
    if len(texts) >= num_texts:
        return texts[:num_texts]
    out: list[str] = []
    while len(out) < num_texts:
        out.extend(texts)
    return out[:num_texts]


def synthesize_texts(num_texts: int, max_chars: int) -> list[str]:
    """Build synthetic clinical-sounding texts of a fixed character length.

    SHIELD and similar corpora have no short notes, so short inputs must be made
    by truncation. This repeats a PHI-dense filler string and truncates to
    ``max_chars``.

    Args:
        num_texts: Number of texts to produce.
        max_chars: Character length of each text.

    Returns:
        A list of ``num_texts`` identical texts, each ``max_chars`` characters.
    """
    unit = _CLINICAL_LOREM
    reps = (max_chars // len(unit)) + 1
    filled = (unit * reps)[:max_chars]
    return [filled for _ in range(num_texts)]


def load_input_texts(path: str, num_texts: int, max_chars: int) -> list[str]:
    """Load real notes from a parquet or plain-text file, truncated to length.

    Args:
        path: Path to a ``.parquet`` file (a note-text-like column is auto-picked)
            or a ``.txt`` file (one note per line).
        num_texts: Number of texts to return (cycled if the source is shorter).
        max_chars: Maximum character length per text.

    Returns:
        A list of exactly ``num_texts`` truncated texts.
    """
    if path.endswith(".txt"):
        with Path(path).open(encoding="utf-8") as handle:
            texts = [line.rstrip("\n") for line in handle if line.strip()]
    else:
        import pandas as pd

        frame = pd.read_parquet(path)
        column = _pick_text_column(list(frame.columns))
        texts = frame[column].astype(str).tolist()

    texts = [t[:max_chars] for t in texts if t]
    return _cycle_to_length(texts, num_texts)


def _dtype_from_name(name: str) -> torch.dtype:
    """Map a dtype name to a torch dtype.

    Args:
        name: One of ``"float16"``, ``"bfloat16"``, ``"float32"``.

    Returns:
        The corresponding ``torch.dtype``.
    """
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_core(args: argparse.Namespace) -> TransformerCore:
    """Construct and load a TransformerCore for the requested configuration.

    Compile modes are applied here:
        * ``eager``: load the model as-is (no torch.compile) — this is what the
          product pipeline runs.
        * ``dynamic``: load uncompiled, then wrap ``core._model`` with
          ``torch.compile(..., mode="default", dynamic=True)``.
        * ``reduce-overhead``: load uncompiled, then wrap ``core._model`` with
          ``torch.compile(..., mode="reduce-overhead", dynamic=False)``.

    The dynamic / reduce-overhead paths patch ``core._model`` directly. The
    product batch pipeline no longer supports ``torch.compile`` at all (it runs
    eager); these are harness-only shortcuts for measurement, not a supported API.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A loaded ``TransformerCore`` with the requested compile mode applied.
    """
    core = TransformerCore(
        model_name=args.model,
        model_path=args.model_path,
        device=args.device,
        dtype=_dtype_from_name(args.dtype),
        load_immediately=True,
        local_files_only=args.local_files_only,
        allow_huggingface_download=not args.local_files_only,
    )

    if args.compile_mode == "dynamic":
        core._model = torch.compile(core._model, mode="default", dynamic=True)
    elif args.compile_mode == "reduce-overhead":
        core._model = torch.compile(core._model, mode="reduce-overhead", dynamic=False)

    return core


def _read_memory(is_cuda: bool, index: int) -> dict[str, float | None]:
    """Read the full set of GPU-memory metrics for the timed region.

    Args:
        is_cuda: Whether the run is on a CUDA device.
        index: CUDA device index (ignored when ``is_cuda`` is False).

    Returns:
        A dict with ``allocated_MB``, ``reserved_MB``, ``peak_MB``, ``free_MB``,
        and ``total_MB`` (all ``None`` for CPU runs).
    """
    if not is_cuda:
        return {
            "allocated_MB": None,
            "reserved_MB": None,
            "peak_MB": None,
            "free_MB": None,
            "total_MB": None,
        }

    torch.cuda.synchronize(index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    return {
        "allocated_MB": torch.cuda.memory_allocated(index) / _BYTES_PER_MB,
        "reserved_MB": torch.cuda.memory_reserved(index) / _BYTES_PER_MB,
        "peak_MB": torch.cuda.max_memory_allocated(index) / _BYTES_PER_MB,
        "free_MB": free_bytes / _BYTES_PER_MB,
        "total_MB": total_bytes / _BYTES_PER_MB,
    }


def _time_tokenize_pass(
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
    padding: str,
    seq_len: int,
) -> float:
    """Time a single tokenize-only pass over all texts (CPU-side).

    Uses the same tokenizer kwargs as ``_forward_batch_direct`` so the split
    estimate is comparable to the real hot path.

    Args:
        tokenizer: The real (unwrapped) tokenizer.
        texts: Texts to tokenize.
        batch_size: Texts per tokenize call.
        padding: ``"longest"`` or ``"max_length"``.
        seq_len: Max length used for ``max_length`` padding.

    Returns:
        Elapsed wall-clock seconds for the pass.
    """
    if padding == "max_length":
        pad_kwargs: dict[str, Any] = {"padding": "max_length", "max_length": seq_len}
    else:
        pad_kwargs = {"padding": "longest"}

    start = time.perf_counter()
    for offset in range(0, len(texts), batch_size):
        sub = texts[offset : offset + batch_size]
        tokenizer(
            sub,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            **pad_kwargs,
        )
    return time.perf_counter() - start


def measure_combo(
    core: TransformerCore,
    real_tokenizer: Any,
    texts: list[str],
    batch_size: int,
    args: argparse.Namespace,
    is_cuda: bool,
    index: int,
) -> dict[str, Any]:
    """Measure throughput, timing split, and memory for one batch size.

    Args:
        core: The loaded (and padding-proxied) TransformerCore.
        real_tokenizer: The unwrapped tokenizer, for tokenize-only timing.
        texts: The full list of texts fed per pass.
        batch_size: Texts per GPU forward pass.
        args: Parsed CLI arguments.
        is_cuda: Whether the run is on CUDA.
        index: CUDA device index.

    Returns:
        A metrics dict for the combination, or a dict with ``status="OOM"`` if an
        out-of-memory condition was hit.
    """
    num_texts = len(texts)

    try:
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(index)

        # Warmup (also triggers torch.compile specialization for this shape).
        for _ in range(args.warmup):
            core.infer_raw_direct(texts, batch_size=batch_size)
            _time_tokenize_pass(real_tokenizer, texts, batch_size, args.padding, args.seq_len)

        if is_cuda:
            torch.cuda.synchronize(index)
            torch.cuda.reset_peak_memory_stats(index)

        pass_times: list[float] = []
        tokenize_times: list[float] = []
        for _ in range(args.iters):
            if is_cuda:
                torch.cuda.synchronize(index)
            start = time.perf_counter()
            core.infer_raw_direct(texts, batch_size=batch_size)
            if is_cuda:
                torch.cuda.synchronize(index)
            pass_times.append(time.perf_counter() - start)
            tokenize_times.append(_time_tokenize_pass(real_tokenizer, texts, batch_size, args.padding, args.seq_len))
    except _OOM_EXCEPTIONS as exc:
        if not _is_oom_error(exc):
            raise
        if is_cuda:
            torch.cuda.empty_cache()
        return {
            "n": num_texts,
            "batch": batch_size,
            "seq_len": args.seq_len,
            "max_chars": args.max_chars,
            "padding": args.padding,
            "compile": args.compile_mode,
            "status": "OOM",
        }

    memory = _read_memory(is_cuda, index)

    mean_pass = statistics.mean(pass_times)
    mean_tokenize = statistics.mean(tokenize_times)
    forward_seconds = max(mean_pass - mean_tokenize, 0.0)
    texts_per_second = num_texts / mean_pass if mean_pass > 0 else float("nan")

    return {
        "n": num_texts,
        "batch": batch_size,
        "seq_len": args.seq_len,
        "max_chars": args.max_chars,
        "padding": args.padding,
        "compile": args.compile_mode,
        "status": "ok",
        "texts_per_s": texts_per_second,
        "total_ms": mean_pass * 1000.0,
        "tokenize_ms": mean_tokenize * 1000.0,
        "forward_ms": forward_seconds * 1000.0,
        **memory,
    }


def _fmt(value: float | None, spec: str) -> str:
    """Format a possibly-None numeric value, using ``n/a`` for None.

    Args:
        value: The number to format, or None.
        spec: A format spec (e.g. ``".1f"``).

    Returns:
        The formatted string.
    """
    if value is None:
        return "n/a"
    return format(value, spec)


def _print_header() -> None:
    """Print the greppable results-table header line."""
    print(
        _ROW_FMT.format(
            tag="HEADER",
            n="n",
            batch="batch",
            seqmax="seq/maxchars",
            padding="padding",
            compile="compile",
            texts_s="texts_s",
            tok="tokenize_ms",
            fwd="forward_ms",
            alloc="alloc_MB",
            resv="resv_MB",
            peak="peak_MB",
            free="free_MB",
            total="total_MB",
        )
    )


def _print_row(row: dict[str, Any]) -> None:
    """Print one greppable ``RESULT`` line for a measured combination.

    Args:
        row: A metrics dict from :func:`measure_combo`.
    """
    seqmax = f"{row['seq_len']}/{row['max_chars']}"
    if row.get("status") == "OOM":
        oom = "OOM"
        print(
            _ROW_FMT.format(
                tag="RESULT",
                n=row["n"],
                batch=row["batch"],
                seqmax=seqmax,
                padding=row["padding"],
                compile=row["compile"],
                texts_s=oom,
                tok=oom,
                fwd=oom,
                alloc=oom,
                resv=oom,
                peak=oom,
                free=oom,
                total=oom,
            )
        )
        return

    print(
        _ROW_FMT.format(
            tag="RESULT",
            n=row["n"],
            batch=row["batch"],
            seqmax=seqmax,
            padding=row["padding"],
            compile=row["compile"],
            texts_s=_fmt(row["texts_per_s"], ".1f"),
            tok=_fmt(row["tokenize_ms"], ".2f"),
            fwd=_fmt(row["forward_ms"], ".2f"),
            alloc=_fmt(row["allocated_MB"], ".1f"),
            resv=_fmt(row["reserved_MB"], ".1f"),
            peak=_fmt(row["peak_MB"], ".1f"),
            free=_fmt(row["free_MB"], ".1f"),
            total=_fmt(row["total_MB"], ".1f"),
        )
    )


def _parse_batch_sizes(values: list[str] | None) -> list[int]:
    """Parse ``--batch-size`` (repeatable and/or comma-separated) into ints.

    Args:
        values: Raw ``--batch-size`` values as collected by argparse.

    Returns:
        A list of positive integer batch sizes.
    """
    if not values:
        return [8]
    sizes: list[int] = []
    for value in values:
        for part in value.split(","):
            stripped = part.strip()
            if stripped:
                sizes.append(int(stripped))
    return sizes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the harness.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Dev-only throughput / GPU-memory harness for TransformerCore.",
    )
    parser.add_argument(
        "--model",
        default="StanfordAIMI/stanford-deidentifier-v2",
        help="Model config name or HuggingFace repo id.",
    )
    parser.add_argument("--model-path", default=None, help="Explicit local model directory override.")
    parser.add_argument("--device", default="cuda:0", help="Device: cuda:N or cpu.")
    parser.add_argument(
        "--batch-size",
        action="append",
        default=None,
        help="Texts per forward pass. Repeatable and/or comma-separated (e.g. 8,16,32).",
    )
    parser.add_argument("--num-texts", type=int, default=256, help="Total texts fed per pass.")
    parser.add_argument("--seq-len", type=int, default=512, help="Token length for max_length padding.")
    parser.add_argument("--max-chars", type=int, default=2000, help="Char length to synthesize/truncate texts to.")
    parser.add_argument(
        "--padding",
        choices=("longest", "max_length"),
        default="longest",
        help="Padding strategy applied to the tokenizer.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("eager", "dynamic", "reduce-overhead"),
        default="eager",
        help="Compile mode: eager (no compile), dynamic, or reduce-overhead.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Untimed warmup passes.")
    parser.add_argument("--iters", type=int, default=5, help="Timed passes to average.")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Model dtype (may be overridden by the model config).",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Never download from HuggingFace Hub.")
    parser.add_argument(
        "--input",
        default=None,
        help="Optional parquet/txt source of real notes (else texts are synthesized).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the throughput / memory sweep and print results.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success or clean self-skip).
    """
    args = parse_args(argv)
    is_cuda = args.device.startswith("cuda")

    if is_cuda and not torch.cuda.is_available():
        print(
            "CUDA is not available but a CUDA device was requested "
            f"({args.device!r}). This harness must run on a real GPU box. "
            "Skipping. Pass --device cpu to force a CPU run.",
            file=sys.stderr,
        )
        return 0

    index = _device_index(args.device) if is_cuda else -1
    batch_sizes = _parse_batch_sizes(args.batch_size)

    if args.input:
        texts = load_input_texts(args.input, args.num_texts, args.max_chars)
        source = f"input:{args.input}"
    else:
        texts = synthesize_texts(args.num_texts, args.max_chars)
        source = "synthesized"

    print(
        f"# harness: model={args.model} device={args.device} dtype={args.dtype} "
        f"compile={args.compile_mode} padding={args.padding} source={source} "
        f"num_texts={len(texts)} warmup={args.warmup} iters={args.iters}"
    )

    core = build_core(args)

    # Install the padding proxy so infer_raw_direct honors --padding, and keep a
    # handle to the real tokenizer for the separate tokenize-only timing.
    real_tokenizer = core._tokenizer
    core._tokenizer = _PaddingTokenizerProxy(real_tokenizer, args.padding, args.seq_len)

    _print_header()

    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        row = measure_combo(core, real_tokenizer, texts, batch_size, args, is_cuda, index)
        results.append(row)
        _print_row(row)

    summary = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "compile_mode": args.compile_mode,
        "padding": args.padding,
        "source": source,
        "num_texts": len(texts),
        "seq_len": args.seq_len,
        "max_chars": args.max_chars,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    print("=== JSON SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
