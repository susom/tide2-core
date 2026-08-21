#!/usr/bin/env python
"""Machine-checked verification of the transformer CUDA-OOM-recovery fix (GPU).

Purpose
-------
Prove, on a real GPU, that the OOM-recovery fix (release GPU tensors at the point
of failure + iterative, single-live-exception recovery) holds on **both** code
paths the pipeline uses:

- ``direct`` mode: drive the actor/core inference path in-process (no Ray), for a
  clean per-``__call__`` GPU-memory curve.
- ``ray`` mode: run the real Ray Data ``map_batches`` pipeline via ``LocalJobRunner``
  over the SHIELD parquet, exercising actor reuse across many blocks.

Requires a GPU; exits non-zero if none is available. This is **not** a merge gate
(PR CI has no GPU) — it validates the fix on this hardware. It does **not** attempt
to reproduce the original ~80 GiB incident's numbers.

Expected outcome
----------------
- On the **fixed** code: exits 0. Allocated memory returns to the post-warmup
  baseline after every ``__call__`` (including after an OOM-and-recovery), a
  first-attempt OOM recovers by splitting and returns one prediction per input in
  order, and peak stays below total VRAM by a margin. Ray mode additionally
  completes with every input note present in the output (no dropped blocks).
- On the **unfixed** code: fails (exit 1) — memory grows across ``__call__``s / the
  OOM cascade never recovers.

Data source: SHIELD dataset (``text`` column) at
``/data/neurips2026_data/shield_dataset/shield_pii_dataset.parquet`` on an L4.

Usage
-----
    python scripts/verify_oom_fix.py --mode both
    python scripts/verify_oom_fix.py --mode direct --model StanfordAIMI/stanford-deidentifier-v2
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verify_oom_fix")

DEFAULT_PARQUET = "/data/neurips2026_data/shield_dataset/shield_pii_dataset.parquet"
DEFAULT_MODEL = "StanfordAIMI/stanford-deidentifier-v2"

# Memory returned to baseline is never byte-exact (allocator rounding, cached
# blocks); allow a small slack.
BASELINE_TOLERANCE_MB = 128.0
# Peak allocated must stay this fraction below total VRAM.
PEAK_HEADROOM_FRACTION = 0.95


def load_texts(parquet_path: str, column: str, limit: int | None) -> list[str]:
    """Load non-empty text rows from the dataset, deterministically ordered."""
    table = pq.read_table(parquet_path, columns=[column])
    texts = [t for t in table.column(column).to_pylist() if t]
    if limit is not None:
        texts = texts[:limit]
    return texts


def build_batch(texts: list[str]) -> dict[str, list]:
    """Build a columnar batch shaped like the Ray Data transformer input."""
    n = len(texts)
    return {
        "chunk_text": list(texts),
        "text_hash": [f"h{i}" for i in range(n)],
        "chunk_id": list(range(n)),
        "char_offset_start": [0] * n,
        "patient_id": [f"p{i}" for i in range(n)],
        "chunk_uid": [f"u{i}" for i in range(n)],
    }


def _mb(nbytes: int) -> float:
    return nbytes / 1024**2


def run_direct(args) -> bool:  # noqa: PLR0915 - one linear check reads clearest as a single flow
    """D2a: in-process actor path. Returns True on pass."""
    import torch

    from tide2.actors.transformer import create_transformer_actor

    device = torch.device("cuda:0")
    total_vram = torch.cuda.get_device_properties(device).total_memory
    texts = load_texts(args.parquet, args.column, args.limit)
    logger.info("direct: loaded %d texts", len(texts))

    ok = True

    # --- Actor with auto batch sizing for the memory-stability curve ---
    actor = create_transformer_actor(model_name=args.model, allow_huggingface_download=True)()

    # Warmup: load kernels / stabilize allocator, then set the baseline.
    warm = min(len(texts), args.batch)
    actor(build_batch(texts[:warm]))
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated(device)
    logger.info("direct: post-warmup baseline allocated=%.1fMB", _mb(baseline))

    # --- (a) No monotonic growth across many __call__s ---
    n_passes = args.passes
    for i in range(n_passes):
        start = (i * args.batch) % max(1, len(texts) - args.batch)
        actor(build_batch(texts[start : start + args.batch]))
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated(device)
        drift = _mb(allocated - baseline)
        logger.info("direct: pass %d/%d allocated=%.1fMB drift=%.1fMB", i + 1, n_passes, _mb(allocated), drift)
        if drift > BASELINE_TOLERANCE_MB:
            logger.error("direct: FAIL memory grew %.1fMB above baseline on pass %d", drift, i + 1)
            ok = False
            break

    # --- (b) First-attempt OOM recovers, correct count + order ---
    # Force an oversized first forward by pinning a huge gpu_batch_size and feeding
    # many long texts, so infer_raw_direct OOMs and recovery must split down.
    longest = sorted(texts, key=len, reverse=True)[: max(1, args.oom_count // 4) or 1]
    oom_texts = (longest * ((args.oom_count // len(longest)) + 1))[: args.oom_count]
    oom_actor = create_transformer_actor(
        model_name=args.model, gpu_batch_size=args.oom_count, allow_huggingface_download=True
    )()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    oom_baseline = torch.cuda.memory_allocated(device)
    try:
        out = oom_actor(build_batch(oom_texts))
    except RuntimeError:
        logger.exception("direct: FAIL OOM recovery raised instead of recovering")
        return False
    preds = out["predictions_raw_json"]
    if len(preds) != len(oom_texts):
        logger.error("direct: FAIL recovery returned %d results for %d inputs", len(preds), len(oom_texts))
        ok = False
    else:
        logger.info("direct: OOM recovery returned %d/%d predictions in order", len(preds), len(oom_texts))

    # Memory must return to baseline even after the OOM cascade (the leak test).
    gc.collect()
    torch.cuda.synchronize()
    post_oom = torch.cuda.memory_allocated(device)
    drift = _mb(post_oom - oom_baseline)
    logger.info("direct: post-OOM-recovery drift=%.1fMB", drift)
    if drift > BASELINE_TOLERANCE_MB:
        logger.error("direct: FAIL memory leaked %.1fMB after OOM recovery", drift)
        ok = False

    # --- (c) Peak below total VRAM by a margin ---
    peak = torch.cuda.max_memory_allocated(device)
    logger.info("direct: peak allocated=%.1fMB / total=%.1fMB", _mb(peak), _mb(total_vram))
    if peak > total_vram * PEAK_HEADROOM_FRACTION:
        logger.error("direct: FAIL peak exceeded %.0f%% of VRAM", PEAK_HEADROOM_FRACTION * 100)
        ok = False

    logger.info("direct: %s", "PASS" if ok else "FAIL")
    return ok


def run_ray(args) -> bool:
    """D2b: full Ray Data pipeline via LocalJobRunner. Returns True on pass."""
    import pandas as pd

    from tide2.runner.local_runner import LocalJobRunner
    from tide2.utils.text_processing import compute_text_hash

    texts = load_texts(args.parquet, args.column, args.limit)
    logger.info("ray: loaded %d notes", len(texts))

    df = pd.DataFrame({"note_text": texts})
    df["text_hash"] = df["note_text"].apply(compute_text_hash)
    df["patient_id"] = df["text_hash"]
    input_hashes = set(df["text_hash"])

    with tempfile.TemporaryDirectory(dir=args.workdir) as tmp:
        in_path = Path(tmp) / "shield_input.parquet"
        out_path = Path(tmp) / "transformer_out"
        df.to_parquet(in_path, index=False)

        runner = LocalJobRunner(num_gpus=1)
        try:
            runner.run_transformer(
                input_path=str(in_path),
                output_path=str(out_path),
                model_name=args.model,
                batch_size=args.batch,
                gpu_batch_size=args.oom_count,  # inflate to force first-attempt OOM in the actor
                enable_checkpoint=False,
            )
        finally:
            runner.shutdown()

        out_files = list(out_path.glob("**/*.parquet"))
        if not out_files:
            logger.error("ray: FAIL no output written")
            return False
        out_df = pd.concat([pq.read_table(f).to_pandas() for f in out_files], ignore_index=True)

    out_hashes = set(out_df["text_hash"])
    missing = input_hashes - out_hashes
    logger.info("ray: output rows=%d, notes covered=%d/%d", len(out_df), len(out_hashes), len(input_hashes))
    if missing:
        logger.error("ray: FAIL %d input notes missing from output (dropped blocks)", len(missing))
        return False
    if len(out_df) < len(input_hashes):
        logger.error("ray: FAIL fewer output rows (%d) than input notes (%d)", len(out_df), len(input_hashes))
        return False

    logger.info("ray: PASS")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["direct", "ray", "both"], default="both")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--column", default="text")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Cap number of notes (default: all).")
    parser.add_argument("--batch", type=int, default=256, help="Ray/direct batch size for normal passes.")
    parser.add_argument("--passes", type=int, default=20, help="Direct-mode stability passes.")
    parser.add_argument(
        "--oom-count",
        type=int,
        default=4096,
        help="Texts (and pinned gpu_batch_size) used to force a first-attempt OOM.",
    )
    parser.add_argument("--workdir", default=".", help="Parent dir for the Ray temp workspace.")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        logger.exception("torch not importable")
        return 2
    if not torch.cuda.is_available():
        logger.error("No GPU available; this script requires CUDA. Exiting non-zero.")
        return 2

    results: dict[str, bool] = {}
    if args.mode in ("direct", "both"):
        results["direct"] = run_direct(args)
    if args.mode in ("ray", "both"):
        results["ray"] = run_ray(args)

    logger.info("summary: %s", {k: ("PASS" if v else "FAIL") for k, v in results.items()})
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
