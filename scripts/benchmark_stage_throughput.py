#!/usr/bin/env python3
"""Single-stage throughput benchmark for a ``tide2-runner run <stage>`` job.

Runs **one** distributed de-identification stage (``transformer`` | ``recognizer``
| ``anonymizer``) through the real CLI path, times its wall-clock, and reports
throughput in **content-tokens/second** — ``tokens/s = T / wall_seconds`` — where
``T`` is the corpus's total model-tokenizer content-token count (the same ``T`` for
every stage, so per-stage numbers are directly comparable and the end-to-end total
is ``T / Σ(stage walls)``).

Why the CLI subprocess (not the runner API): it faithfully exercises the same
``argparse → cmd_run → LocalJobRunner.run_<stage>`` path an operator uses, and
isolates each run in its own Ray process so runs don't contaminate each other's
allocator/actor state. This script does **one** run of **one** stage; a sweep is
just this script called repeatedly with different knobs (see the throwaway
``tune.sh`` in the benchmark plan).

``T`` comes from ``--tokens`` when given (precomputed once per corpus and reused,
which is the fast path), otherwise it is counted from the input parquet's
``note_text`` with the model's fast tokenizer (``add_special_tokens=False``).

For the transformer stage it additionally samples whole-device GPU utilization and
peak VRAM (via ``nvidia-smi`` and any ``TIDE2_LOG_GPU_MEM`` peak lines in the run
log) and, when ``--forwarded-tokens`` is supplied, reports **forwarded tokens/s**
(windows x padded length, incl. overlap) to show raw GPU work vs. useful content.

Outputs: one line printed to stdout, one JSON record appended to ``--json-out``
(JSONL), and one markdown table row appended to ``--md-out``.

Examples
--------
    # Transformer on the L4, T precomputed by prep:
    python scripts/benchmark_stage_throughput.py transformer \
        -i in.parquet -o out --model 20260211_debertav3_finetuned \
        --tokens 402317 --forwarded-tokens 511488 --gpu-batch-size 128 \
        --label "worst/transformer(L4)" --json-out res.jsonl --md-out res.md

    # Same transformer, CPU-only:
    python scripts/benchmark_stage_throughput.py transformer \
        -i in.parquet -o out --model 20260211_debertav3_finetuned \
        --tokens 402317 --device cpu --transformer-cpus 15 \
        --label "worst/transformer(CPU)" --json-out res.jsonl --md-out res.md

    # Recognizer (T counted from the input if --tokens omitted):
    python scripts/benchmark_stage_throughput.py recognizer \
        -i rec_in.parquet -o rec_out --model 20260211_debertav3_finetuned \
        --num-actors 15 --json-out res.jsonl --md-out res.md
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess  # nosec B404 # only used with operator-built arg lists, no shell
import sys
import threading
import time
from pathlib import Path

DEFAULT_MODEL = "20260211_debertav3_finetuned"


# --------------------------------------------------------------------------- #
# Token counting (the throughput denominator T)
# --------------------------------------------------------------------------- #
def count_content_tokens(input_path: str, model_name: str, text_column: str = "note_text") -> int:
    """Total model-tokenizer content-tokens across ``text_column`` of the input.

    Content-tokens means ``add_special_tokens=False`` (no ``[CLS]``/``[SEP]``): the
    raw information content the stage must process, independent of windowing. Loads
    the same fast tokenizer the transformer stage uses so ``T`` matches the model
    that produced the entities.
    """
    import pandas as pd
    from transformers import AutoTokenizer

    from tide2.utils.gcs_resource_manager import resolve_model_path

    model_path = resolve_model_path(model_name=model_name)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    files = _resolve_parquet_files(input_path)
    total = 0
    for f in files:
        df = pd.read_parquet(f, columns=[text_column])
        texts = [t for t in df[text_column].tolist() if isinstance(t, str) and t]
        # batch encode without special tokens; sum per-note lengths
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        total += sum(len(ids) for ids in enc)
    return total


def _resolve_parquet_files(path: str) -> list[Path]:
    """Return parquet files from a file, directory, or ``*.parquet`` glob."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.parquet"))
    return sorted(
        Path(p.anchor if p.is_absolute() else ".").glob(str(p.relative_to(p.anchor) if p.is_absolute() else p))
    )


# --------------------------------------------------------------------------- #
# Resource sampler (whole-device GPU + whole-box CPU/RAM)
# --------------------------------------------------------------------------- #
class ResourceSampler:
    """Background thread sampling whole-box GPU and CPU/RAM at a fixed cadence.

    GPU is read via ``nvidia-smi`` (utilization %, memory used MiB, power W); CPU
    and RAM via ``psutil``. Samples are whole-device / whole-box because the actor
    runs in a Ray worker, so per-process counters would miss it.
    """

    def __init__(self, interval: float = 1.0, gpu_index: int = 0) -> None:
        self.interval = interval
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, float]] = []

    def _query_gpu(self) -> tuple[float, float, float]:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)  # noqa: S603 # nosec B603 # fixed nvidia-smi arg list, no shell
            util, mem, power = (x.strip() for x in out.stdout.strip().split(","))
            return float(util), float(mem), float(power)
        except Exception:
            return 0.0, 0.0, 0.0

    def _loop(self) -> None:
        import psutil

        psutil.cpu_percent(interval=None)  # prime the counter
        t0 = time.time()
        while not self._stop.is_set():
            gpu_util, gpu_mem, gpu_power = self._query_gpu()
            self.samples.append(
                {
                    "t": round(time.time() - t0, 2),
                    "gpu_util": gpu_util,
                    "gpu_mem_mib": gpu_mem,
                    "gpu_power_w": gpu_power,
                    "cpu_pct": psutil.cpu_percent(interval=None),
                    "ram_used_gb": round(psutil.virtual_memory().used / 1024**3, 2),
                }
            )
            self._stop.wait(self.interval)

    def __enter__(self) -> ResourceSampler:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)

    def summary(self) -> dict[str, float]:
        """Mean / p95 / max of each sampled metric (empty -> ``{}``)."""
        if not self.samples:
            return {}

        def agg(key: str) -> dict[str, float]:
            vals = [s[key] for s in self.samples]
            return {
                f"{key}_mean": round(statistics.mean(vals), 1),
                f"{key}_p95": round(sorted(vals)[min(len(vals) - 1, int(0.95 * len(vals)))], 1),
                f"{key}_max": round(max(vals), 1),
            }

        result: dict[str, float] = {"n_samples": len(self.samples)}
        for key in ("gpu_util", "gpu_mem_mib", "gpu_power_w", "cpu_pct", "ram_used_gb"):
            result.update(agg(key))
        return result

    def write_timeseries(self, path: Path) -> None:
        if not self.samples:
            return
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.samples[0].keys()))
            w.writeheader()
            w.writerows(self.samples)


# --------------------------------------------------------------------------- #
# CLI assembly for one stage
# --------------------------------------------------------------------------- #
def build_cli(args: argparse.Namespace, runner_cmd: list[str]) -> list[str]:
    """Assemble the ``run <stage>`` CLI invocation from the benchmark args."""
    cmd = [*runner_cmd, "run", args.stage, "-i", args.input, "-o", args.output, "--log-level", "INFO"]

    if not args.checkpoint:
        cmd += ["--no-checkpoint"]

    # Knobs shared / stage-specific — only append when explicitly set.
    def add(flag: str, val: object) -> None:
        if val is not None:
            cmd.extend([flag, str(val)])

    if args.stage == "transformer":
        add("--model", args.model)
        cmd += ["--num-gpus", "1" if args.device == "gpu" else "0"]
        add("--gpu-batch-size", args.gpu_batch_size)
        add("--batch-size", args.batch_size)
        add("--num-agg-actors", args.num_agg_actors)
        add("--chunk-overlap", args.chunk_overlap)
        add("--transformer-cpus", args.transformer_cpus)
        add("--agg-num-cpus", args.agg_num_cpus)
        add("--read-cpus", args.read_cpus)
        add("--write-cpus", args.write_cpus)
    else:  # recognizer / anonymizer (CPU stages)
        add("--num-actors", args.num_actors)
        add("--batch-size", args.batch_size)
        add("--cpus-per-actor", args.cpus_per_actor)
        add("--worker-num-cpus", args.worker_num_cpus)
        add("--read-cpus", args.read_cpus)
        add("--write-cpus", args.write_cpus)
        if args.stage == "anonymizer":
            add("--salt", args.salt)
            add("--key", args.key)

    cmd.extend(args.extra)
    return cmd


def _count_handled_ooms(log_path: Path) -> int:
    """How many times the transformer actor logged a halve-on-OOM recovery."""
    if not log_path.exists():
        return 0
    return sum(1 for line in log_path.read_text(errors="ignore").splitlines() if "CUDA OOM; halving" in line)


def _peak_vram_gb_from_log(log_path: Path) -> float:
    """Max ``peak=<N>MB`` reported by the actor's ``TIDE2_LOG_GPU_MEM`` lines (GB)."""
    if not log_path.exists():
        return 0.0
    peak_mb = 0.0
    for line in log_path.read_text(errors="ignore").splitlines():
        if "GPU mem [" in line and "peak=" in line:
            try:
                peak_mb = max(peak_mb, float(line.split("peak=")[1].split("MB")[0]))
            except (IndexError, ValueError):
                continue
    return round(peak_mb / 1024, 2)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["transformer", "recognizer", "anonymizer"], help="Stage to benchmark.")
    p.add_argument("-i", "--input", required=True, help="Input parquet (file/dir/glob).")
    p.add_argument("-o", "--output", required=True, help="Output directory.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name (transformer stage + tokenizer for T).")
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help="Transformer device (gpu=L4, cpu=CPU-only).")
    p.add_argument("--tokens", type=int, default=None, help="Precomputed content-token total T (else counted from -i).")
    p.add_argument(
        "--forwarded-tokens",
        type=int,
        default=None,
        help="Transformer only: total forwarded tokens (windows x padded length) for forwarded tokens/s.",
    )
    # Knobs (only forwarded to the CLI when set).
    p.add_argument("--gpu-batch-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-agg-actors", type=int, default=None)
    p.add_argument("--num-actors", type=int, default=None)
    p.add_argument("--chunk-overlap", type=int, default=None)
    p.add_argument("--transformer-cpus", type=float, default=None)
    p.add_argument("--agg-num-cpus", type=float, default=None)
    p.add_argument("--cpus-per-actor", type=int, default=None)
    p.add_argument("--worker-num-cpus", type=float, default=None)
    p.add_argument("--read-cpus", type=float, default=None)
    p.add_argument("--write-cpus", type=float, default=None)
    p.add_argument("--salt", default=None, help="Salt file (anonymizer).")
    p.add_argument("--key", default=None, help="Key file (anonymizer).")
    p.add_argument(
        "--checkpoint",
        action="store_true",
        help="Enable Ray checkpointing (off by default here for clean throughput).",
    )
    # Reporting.
    p.add_argument("--label", default=None, help="Row label, e.g. 'worst/transformer(L4)'. Default: '<stage>'.")
    p.add_argument("--json-out", default=None, help="Append the JSON record here (JSONL).")
    p.add_argument("--md-out", default=None, help="Append the markdown table row here.")
    p.add_argument("--sample-interval", type=float, default=1.0, help="Resource sampling cadence (seconds).")
    p.add_argument("--keep-output", action="store_true", help="Do not delete a prior output dir before running.")
    p.add_argument("--dry-run", action="store_true", help="Print the CLI command without running.")
    p.add_argument(
        "--runner-cmd",
        default=f"{sys.executable} -m tide2.runner.cli",
        help="Command that launches the runner CLI (default: current interpreter -m tide2.runner.cli).",
    )
    return p


def main() -> int:
    args, extra = _build_parser().parse_known_args()
    # Any unrecognized flags after a literal '--' are forwarded to the runner CLI.
    args.extra = extra[1:] if extra and extra[0] == "--" else extra

    label = args.label or args.stage
    runner_cmd = args.runner_cmd.split()

    cmd = build_cli(args, runner_cmd)
    print(f"[bench] {label} ({args.stage}, device={args.device})")
    print(f"        {' '.join(cmd)}")
    if args.dry_run:
        return 0

    # Resolve T (the throughput denominator) — precomputed is the fast path.
    tokens = args.tokens
    if tokens is None:
        print("[bench] --tokens not given; counting content-tokens from input (slow) ...")
        tokens = count_content_tokens(args.input, args.model)
        print(f"[bench] T = {tokens:,} content-tokens")

    out_path = Path(args.output)
    if out_path.exists() and not args.keep_output:
        shutil.rmtree(out_path)
    log_path = out_path.parent / f"log_{_slug(label)}.txt"
    ts_path = out_path.parent / f"resources_{_slug(label)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    with ResourceSampler(interval=args.sample_interval) as sampler, log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)  # noqa: S603 # nosec B603 # operator-built CLI, no shell
    wall = time.time() - start
    ok = proc.returncode == 0

    sampler.write_timeseries(ts_path)
    res = sampler.summary()
    is_transformer = args.stage == "transformer"
    peak_vram_gb = max(_peak_vram_gb_from_log(log_path), round(res.get("gpu_mem_mib_max", 0) / 1024, 2))

    record = {
        "label": label,
        "stage": args.stage,
        "device": args.device if is_transformer else "cpu",
        "ok": ok,
        "tokens": tokens,
        "wall_s": round(wall, 1),
        "tokens_per_s": round(tokens / wall, 1) if wall else 0.0,
        "forwarded_tokens": args.forwarded_tokens,
        "forwarded_tokens_per_s": (
            round(args.forwarded_tokens / wall, 1) if is_transformer and args.forwarded_tokens and wall else None
        ),
        "gpu_util_mean": res.get("gpu_util_mean") if is_transformer and args.device == "gpu" else None,
        "gpu_util_max": res.get("gpu_util_max") if is_transformer and args.device == "gpu" else None,
        "peak_vram_gb": peak_vram_gb if is_transformer and args.device == "gpu" else None,
        "cpu_pct_mean": res.get("cpu_pct_mean"),
        "handled_ooms": _count_handled_ooms(log_path) if is_transformer else 0,
        "log": str(log_path),
    }

    _print_line(record)
    if not ok:
        print(f"        FAILED (exit {proc.returncode}); see {log_path}")
    if args.json_out:
        with Path(args.json_out).open("a") as f:
            f.write(json.dumps(record) + "\n")
    if args.md_out:
        _append_md_row(Path(args.md_out), record)
    return 0 if ok else 1


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def _print_line(r: dict) -> None:
    fwd = f"{r['forwarded_tokens_per_s']:.0f} fwd-tok/s | " if r.get("forwarded_tokens_per_s") else ""
    gpu = (
        f"GPU {r['gpu_util_mean']:.0f}%/{r['gpu_util_max']:.0f}%, {r['peak_vram_gb']:.1f}GB | "
        if r.get("gpu_util_mean") is not None
        else ""
    )
    print(
        f"        -> {r['tokens_per_s']:.0f} tok/s | {fwd}{gpu}"
        f"wall {r['wall_s']:.1f}s | T={r['tokens']:,} | OOMs={r['handled_ooms']} | ok={r['ok']}"
    )


_MD_HEADER = (
    "| stage | device | tokens/s | forwarded tok/s | peak VRAM (GB) | GPU util mean/max | wall (s) | T (tokens) | OOMs |\n"
    "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
)


def _append_md_row(path: Path, r: dict) -> None:
    """Append one markdown row, writing the header first if the file is new."""
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(_MD_HEADER)

    def num(x: object, fmt: str = "{:,.0f}") -> str:
        return fmt.format(x) if isinstance(x, (int, float)) else "—"

    gpu_util = f"{r['gpu_util_mean']:.0f}% / {r['gpu_util_max']:.0f}%" if r.get("gpu_util_mean") is not None else "—"
    row = (
        f"| {r['label']} | {r['device']} | {num(r['tokens_per_s'])} | {num(r['forwarded_tokens_per_s'])} | "
        f"{num(r['peak_vram_gb'], '{:.1f}')} | {gpu_util} | {num(r['wall_s'], '{:.1f}')} | "
        f"{num(r['tokens'])} | {r['handled_ooms']} |\n"
    )
    with path.open("a") as f:
        f.write(row)


if __name__ == "__main__":
    sys.exit(main())
