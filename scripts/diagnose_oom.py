#!/usr/bin/env python
"""Diagnose the transformer CUDA-OOM mechanism: leak (A) vs CUDA-graph pool (B).

Runs against the current (typically **unfixed**) code on a real GPU and prints a
quantitative verdict distinguishing the two root-cause hypotheses from the plan:

- **A — recovery leak:** failed-forward GPU tensors stay pinned across OOM
  recovery, so ``empty_cache()`` reclaims nothing and memory grows monotonically
  across ``__call__``s (independent of input shape), especially after an
  OOM-and-recovery cascade.
- **B — CUDA-graph pool growth:** ``torch.compile(mode="reduce-overhead")``
  captures/retains a static memory pool per unique ``(batch, seq_len)`` shape, so
  growth tracks the number of *new shapes*, not repeated work.

Requires a GPU; exits non-zero if none. Read-only w.r.t. any deployed config (no
writes, no model mutation). Not a merge gate — a diagnostic aid.

Data source: SHIELD dataset (``text`` column). This VM: NVIDIA L4 (23 GiB).

Usage
-----
    python scripts/diagnose_oom.py
    python scripts/diagnose_oom.py --compile-cache /path/to/compiled_cache.bin --limit 1381
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys

import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("diagnose_oom")

DEFAULT_PARQUET = "/data/neurips2026_data/shield_dataset/shield_pii_dataset.parquet"
DEFAULT_MODEL = "StanfordAIMI/stanford-deidentifier-v2"

# Per-batch allocated growth (MB) above which we call it "growth".
GROWTH_MB = 32.0


def load_texts(parquet_path: str, column: str, limit: int | None) -> list[str]:
    table = pq.read_table(parquet_path, columns=[column])
    texts = [t for t in table.column(column).to_pylist() if t]
    return texts[:limit] if limit is not None else texts


def build_batch(texts: list[str]) -> dict[str, list]:
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


def main() -> int:  # noqa: PLR0915 - a linear diagnostic script reads clearest top-to-bottom
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--column", default="text")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--compile-model", action="store_true", help="Mirror a deployment that enables torch.compile.")
    parser.add_argument("--compile-cache", default=None, help="Path to compiled_cache.bin (enables compile).")
    parser.add_argument("--repeats", type=int, default=10, help="Same-shape repeat count.")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        logger.exception("torch not importable")
        return 2
    if not torch.cuda.is_available():
        logger.error("No GPU available; this script requires CUDA. Exiting non-zero.")
        return 2

    from tide2.actors.transformer import create_transformer_actor

    device = torch.device("cuda:0")
    texts = load_texts(args.parquet, args.column, args.limit)
    logger.info("loaded %d texts", len(texts))

    compile_model = True if (args.compile_model or args.compile_cache) else None
    actor = create_transformer_actor(
        model_name=args.model,
        compile_model=compile_model,
        compile_cache_path=args.compile_cache,
        allow_huggingface_download=True,
    )()

    compiled = type(getattr(actor._core, "_model", None)).__name__
    logger.info("model wrapper class: %s (compile likely %s)", compiled, "ON" if "Optimized" in compiled else "OFF")

    def snapshot(tag: str, n: int, seq: int) -> int:
        gc.collect()
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        free, total = torch.cuda.mem_get_info(device)
        logger.info(
            "[%s] batch=%d seq~=%d | allocated=%.1fMB reserved=%.1fMB free=%.1fMB/%.1fMB",
            tag,
            n,
            seq,
            _mb(alloc),
            _mb(reserved),
            _mb(free),
            _mb(total),
        )
        return alloc

    # Warmup + baseline
    actor(build_batch(texts[:16]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = snapshot("baseline", 0, 0)

    tokenizer = actor._core._tokenizer

    def seq_of(batch_texts: list[str]) -> int:
        return int(tokenizer(batch_texts, truncation=True, return_tensors="pt")["input_ids"].shape[1])

    # --- Phase 1: identical shape repeated -> isolates leak (A) from shape (B) ---
    same = texts[:16]
    same_seq = seq_of(same)
    prev = base
    same_growth = 0.0
    for i in range(args.repeats):
        actor(build_batch(same))
        cur = snapshot(f"same-shape#{i + 1}", len(same), same_seq)
        same_growth += max(0.0, _mb(cur - prev))
        prev = cur

    # --- Phase 2: new shapes each time -> exposes CUDA-graph pool growth (B) ---
    prev = snapshot("pre-newshape", 0, 0)
    shape_growth = 0.0
    for i, size in enumerate([8, 24, 40, 56, 72, 88]):
        chunk = texts[:size]
        actor(build_batch(chunk))
        cur = snapshot(f"new-shape#{i + 1}", size, seq_of(chunk))
        shape_growth += max(0.0, _mb(cur - prev))
        prev = cur

    # --- Phase 3: force OOM + recovery -> growth here implicates A ---
    prev = snapshot("pre-oom", 0, 0)
    longest = sorted(texts, key=len, reverse=True)[:64] or texts[:64]
    oom_texts = (longest * 64)[:4096]
    oom_actor = create_transformer_actor(model_name=args.model, gpu_batch_size=4096, allow_huggingface_download=True)()
    try:
        oom_actor(build_batch(oom_texts))
        logger.info("OOM recovery completed")
    except RuntimeError as e:
        logger.info("OOM path raised (expected on unfixed code sometimes): %s", str(e)[:120])
    del oom_actor
    post_oom = snapshot("post-oom", 0, 0)
    oom_growth = max(0.0, _mb(post_oom - prev))

    # --- Verdict ---
    logger.info("=" * 60)
    logger.info("same-shape cumulative growth: %.1fMB", same_growth)
    logger.info("new-shape cumulative growth : %.1fMB", shape_growth)
    logger.info("post-OOM-recovery growth    : %.1fMB", oom_growth)
    compile_on = "Optimized" in compiled

    if compile_on and shape_growth > GROWTH_MB and shape_growth > same_growth:
        verdict = "B (CUDA-graph pool grows per input shape; compile enabled)"
    elif oom_growth > GROWTH_MB or same_growth > GROWTH_MB:
        verdict = "A (memory not released across __call__s / OOM recovery)"
    else:
        verdict = "No significant growth observed on this run (fix may already be present)"
    logger.info("VERDICT: %s", verdict)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
