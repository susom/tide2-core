"""Importable GPU verification harness for the transformer OOM-recovery fix.

This module is imported by ``tests/test_transformer_oom_recovery_gpu.py`` and is
also runnable manually as a CLI (``python tests/oom_verification.py --mode all``).
It has **no** ``test_`` prefix, so pytest does not collect it directly.

Why it lives here (not under ``scripts/``): the GPU regression test previously
imported a ``scripts/`` CLI script via ``importlib``, coupling the test suite to a
command-line tool (review Finding #4). The reusable logic now lives in this
importable module and the pure-diagnostic script was dropped.

What it proves on a real GPU:

- ``direct`` / ``ray``: the OOM-recovery fix holds — memory returns to baseline
  after every ``__call__`` (including after an OOM-and-recovery), a forced
  first-attempt OOM recovers with one prediction per input in order, and peak
  stays below total VRAM. These now assert on **both** ``allocated`` **and**
  ``reserved`` plus the total device footprint (``mem_get_info``), not just
  ``allocated`` — ``reserved`` (CUDA-graph pools) is the metric that actually
  drives the production OOM, so the old allocated-only checks were blind to it.

- ``reserved-bounded``: drives many *distinct* input shapes through the actor and
  asserts ``reserved`` stays bounded. With the default (eager) path this passes.
  Passing ``--compile-churn`` force-applies ``reduce-overhead`` compile to
  reproduce the blocker (unbounded ``reserved`` growth under shape churn) as a
  manual diagnostic — this is the case the honest verification exists to catch.
  The product batch pipeline no longer supports ``torch.compile`` (it runs
  eager); this flag force-compiles the model directly, only to keep the
  historical blocker reproducible.

Requires CUDA; the CLI exits non-zero if no GPU is present.
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
logger = logging.getLogger("oom_verification")

DEFAULT_PARQUET = "/data/neurips2026_data/shield_dataset/shield_pii_dataset.parquet"
DEFAULT_MODEL = "StanfordAIMI/stanford-deidentifier-v2"

# Memory returned to baseline is never byte-exact (allocator rounding, cached
# blocks); allow a small slack.
BASELINE_TOLERANCE_MB = 128.0
# Reserved (caching allocator / CUDA-graph pools) grows in larger, coarser steps
# than allocated, so it needs a looser return-to-baseline slack.
RESERVED_TOLERANCE_MB = 512.0
# Whole-device footprint (``total - free`` from ``mem_get_info``) must return to
# within this slack of the pre-OOM footprint once ``empty_cache`` has run. Looser
# than the per-process tolerances because ``mem_get_info`` also sees other
# processes sharing the GPU (concurrent jobs), so a little cross-process noise is
# expected.
FOOTPRINT_TOLERANCE_MB = 1024.0
# Peak allocated / reserved must stay this fraction below total VRAM.
PEAK_HEADROOM_FRACTION = 0.95
# reserved-bounded scenario: total reserved growth across all distinct shapes
# must stay under this bound. Eager stays flat (~0); reduce-overhead + churn
# blows past it (§6d in the review measured ~+900MB over six shapes, then GBs).
RESERVED_GROWTH_BOUND_MB = 1024.0


def load_texts(parquet_path: str, column: str, limit: int | None) -> list[str]:
    """Load non-empty text rows from the dataset, deterministically ordered."""
    table = pq.read_table(parquet_path, columns=[column])
    texts = [t for t in table.column(column).to_pylist() if t]
    if limit is not None:
        texts = texts[:limit]
    return texts


def build_batch(texts: list[str]) -> dict[str, list]:
    """Build a columnar batch shaped like the Ray Data transformer input (whole notes)."""
    n = len(texts)
    return {
        "note_text": list(texts),
        "text_hash": [f"h{i}" for i in range(n)],
        "patient_id": [f"p{i}" for i in range(n)],
    }


def _mb(nbytes: int) -> float:
    return nbytes / 1024**2


def read_gpu_memory(device) -> dict[str, float]:
    """Snapshot every GPU-memory metric that matters for the OOM investigation.

    Returns allocated / reserved / peak-allocated / peak-reserved (via torch's
    per-process counters) plus the whole-device free / total footprint (via
    ``torch.cuda.mem_get_info``, which sees other processes and fragmentation).
    All values are in MB.
    """
    import torch

    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_MB": _mb(torch.cuda.memory_allocated(device)),
        "reserved_MB": _mb(torch.cuda.memory_reserved(device)),
        "peak_allocated_MB": _mb(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_MB": _mb(torch.cuda.max_memory_reserved(device)),
        "free_MB": _mb(free_bytes),
        "total_MB": _mb(total_bytes),
    }


def run_direct(args) -> bool:
    """In-process actor path. Asserts allocated AND reserved stability. True on pass."""
    import torch

    from tide2.actors.transformer import create_transformer_actor

    device = torch.device("cuda:0")
    total_vram = torch.cuda.get_device_properties(device).total_memory
    texts = load_texts(args.parquet, args.column, args.limit)
    logger.info("direct: loaded %d texts", len(texts))

    ok = True
    actor = None
    oom_actor = None

    try:
        actor = create_transformer_actor(model_name=args.model, allow_huggingface_download=True)()

        # Warmup: load kernels / stabilize allocator, then set the baseline.
        warm = min(len(texts), args.batch)
        actor(build_batch(texts[:warm]))
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = read_gpu_memory(device)
        logger.info(
            "direct: post-warmup baseline allocated=%.1fMB reserved=%.1fMB", base["allocated_MB"], base["reserved_MB"]
        )

        # --- (a) No monotonic ALLOCATED growth across many __call__s (the A1 leak
        # test). ``reserved`` is only logged here: in eager mode the caching
        # allocator legitimately grows its pool to the working set's largest shape
        # and keeps it (it drops only on empty_cache), so reserved does NOT return
        # to baseline per-pass — that is normal, not a leak. Unbounded per-shape
        # reserved growth (the compile blocker) is covered by run_reserved_bounded
        # instead. ---
        n_passes = args.passes
        for i in range(n_passes):
            start = (i * args.batch) % max(1, len(texts) - args.batch)
            actor(build_batch(texts[start : start + args.batch]))
            gc.collect()
            cur = read_gpu_memory(device)
            alloc_drift = cur["allocated_MB"] - base["allocated_MB"]
            resv_drift = cur["reserved_MB"] - base["reserved_MB"]
            logger.info(
                "direct: pass %d/%d allocated=%.1fMB (drift=%.1f) reserved=%.1fMB (drift=%.1f)",
                i + 1,
                n_passes,
                cur["allocated_MB"],
                alloc_drift,
                cur["reserved_MB"],
                resv_drift,
            )
            if alloc_drift > BASELINE_TOLERANCE_MB:
                logger.error("direct: FAIL allocated grew %.1fMB above baseline on pass %d", alloc_drift, i + 1)
                ok = False
                break

        # --- (b) First-attempt OOM recovers, correct count + order ---
        longest = sorted(texts, key=len, reverse=True)[: max(1, args.oom_count // 4) or 1]
        oom_texts = (longest * ((args.oom_count // len(longest)) + 1))[: args.oom_count]
        oom_actor = create_transformer_actor(
            model_name=args.model, gpu_batch_size=args.oom_count, allow_huggingface_download=True
        )()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        oom_base = read_gpu_memory(device)
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

        # The handled-OOM path must actually have fired — otherwise this whole
        # scenario is a vacuous pass (the batch was small enough to never OOM).
        handled = oom_actor._handled_oom_count
        logger.info("direct: actor handled %d OOM(s) during recovery", handled)
        if handled <= 0:
            logger.error("direct: FAIL forced batch did not trigger a handled OOM (test is vacuous)")
            ok = False

        # ``allocated`` must return to baseline after the OOM cascade — the core
        # leak test (A1). Checked before empty_cache so a real leak cannot be
        # masked.
        gc.collect()
        post = read_gpu_memory(device)
        alloc_drift = post["allocated_MB"] - oom_base["allocated_MB"]
        logger.info(
            "direct: post-OOM-recovery allocated drift=%.1fMB reserved=%.1fMB",
            alloc_drift,
            post["reserved_MB"],
        )
        if alloc_drift > BASELINE_TOLERANCE_MB:
            logger.error("direct: FAIL allocated leaked %.1fMB after OOM recovery", alloc_drift)
            ok = False

        # ``reserved`` legitimately spikes during the deliberately-forced OOM, but
        # in eager mode empty_cache() must be able to RECLAIM it back near baseline
        # — the honest reserved leak test (under the compile blocker, graph pools
        # cannot be reclaimed and this would fail).
        torch.cuda.empty_cache()
        reclaimed = read_gpu_memory(device)
        resv_drift = reclaimed["reserved_MB"] - oom_base["reserved_MB"]
        logger.info("direct: post-empty_cache reserved=%.1fMB (drift=%.1f)", reclaimed["reserved_MB"], resv_drift)
        if resv_drift > RESERVED_TOLERANCE_MB:
            logger.error("direct: FAIL reserved not reclaimed by empty_cache (drift %.1fMB) after OOM", resv_drift)
            ok = False

        # Whole-device footprint (``total - free`` via ``mem_get_info``) must also
        # return to baseline once the pool is reclaimed. This is the metric the
        # harness advertises but previously never asserted; it catches leaks that
        # escape the per-process counters (e.g. child allocations). Equivalent to
        # ``oom_base.free - reclaimed.free`` since total is constant.
        footprint_drift = oom_base["free_MB"] - reclaimed["free_MB"]
        logger.info(
            "direct: post-empty_cache device footprint free=%.1fMB (drift vs baseline=%.1f)",
            reclaimed["free_MB"],
            footprint_drift,
        )
        if footprint_drift > FOOTPRINT_TOLERANCE_MB:
            logger.error("direct: FAIL device footprint grew %.1fMB after OOM recovery", footprint_drift)
            ok = False

        # --- (c) Peak ALLOCATED below total VRAM by a margin. (Peak reserved is
        # not asserted: the forced-OOM probe deliberately drives it toward VRAM.)
        # ---
        peak_alloc = post["peak_allocated_MB"]
        total_mb = _mb(total_vram)
        logger.info("direct: peak allocated=%.1fMB / total=%.1fMB", peak_alloc, total_mb)
        if peak_alloc > total_mb * PEAK_HEADROOM_FRACTION:
            logger.error("direct: FAIL peak allocated exceeded %.0f%% of VRAM", PEAK_HEADROOM_FRACTION * 100)
            ok = False

        logger.info("direct: %s", "PASS" if ok else "FAIL")
        return ok
    finally:
        # Release the in-process actors (each pins a model on the GPU) so they do
        # not leak VRAM into any subsequent scenario in the same process.
        actor = None
        oom_actor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_reserved_bounded(args) -> bool:
    """Drive many distinct input shapes and assert reserved growth stays bounded.

    This is the compile-aware scenario. With the default (eager) actor, distinct
    shapes add no CUDA-graph pools, so ``reserved`` is flat and the check passes.
    With ``--compile-churn`` the model is force-compiled with ``reduce-overhead``
    (the leak-prone mode) so each new shape mints a graph pool and ``reserved``
    grows without bound — reproducing the blocker the honest check exists to
    detect. Returns True iff reserved growth stayed under the bound.
    """
    import torch

    from tide2.actors.transformer import create_transformer_actor

    device = torch.device("cuda:0")
    texts = load_texts(args.parquet, args.column, args.limit)

    actor = create_transformer_actor(model_name=args.model, allow_huggingface_download=True)()

    if args.compile_churn:
        # Force the leak-prone path directly (product code no longer auto-enables
        # it). This is a deliberate blocker reproduction, not a supported config.
        actor._core._model = torch.compile(actor._core._model, mode="reduce-overhead", fullgraph=True)
        logger.warning("reserved-bounded: FORCED reduce-overhead compile (blocker reproduction)")

    # A ladder of distinct (batch, seq) shapes: vary both the row count and the
    # per-batch max length (by truncating texts) so every batch is a new shape.
    shape_specs = [(8, 64), (12, 96), (16, 128), (20, 160), (24, 192), (28, 224), (32, 256), (36, 320)]

    # Warmup one shape, then baseline reserved.
    actor(build_batch([t[:128] for t in texts[:8]]))
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = read_gpu_memory(device)
    logger.info("reserved-bounded: baseline reserved=%.1fMB", base["reserved_MB"])

    max_growth = 0.0
    for n_rows, max_chars in shape_specs:
        rows = [t[:max_chars] for t in texts[:n_rows] if t]
        if not rows:
            continue
        actor(build_batch(rows))
        gc.collect()
        cur = read_gpu_memory(device)
        growth = cur["reserved_MB"] - base["reserved_MB"]
        max_growth = max(max_growth, growth)
        logger.info(
            "reserved-bounded: shape n=%d max_chars=%d reserved=%.1fMB growth=%.1fMB",
            n_rows,
            max_chars,
            cur["reserved_MB"],
            growth,
        )

    bounded = max_growth <= RESERVED_GROWTH_BOUND_MB
    logger.info(
        "reserved-bounded: %s (max reserved growth=%.1fMB, bound=%.1fMB)",
        "PASS" if bounded else "FAIL",
        max_growth,
        RESERVED_GROWTH_BOUND_MB,
    )
    return bounded


def run_ray(args) -> bool:
    """Full Ray Data pipeline via LocalJobRunner. Returns True on pass."""
    import os

    import pandas as pd

    from tide2.runner.local_runner import LocalJobRunner
    from tide2.utils.text_processing import compute_text_hash

    source = load_texts(args.parquet, args.column, args.limit)
    logger.info("ray: loaded %d source notes", len(source))

    # Build a genuinely oversized ACTOR batch. ``map_batches``'s ``batch_size``
    # caps the rows an actor gets per ``__call__``, so pinning ``gpu_batch_size``
    # alone never OOMs — and the raw SHIELD notes are mostly short (one sub-max
    # chunk each), so even a large row count fits. We therefore synthesize
    # ``oom_count`` *unique* near-max-length notes (one full chunk each) from the
    # longest source text. With ``batch_size == gpu_batch_size == oom_count`` the
    # actor's first forward is ``oom_count`` full-length sequences — large enough
    # to OOM and exercise recovery — while unique hashes keep the coverage check
    # meaningful. ~1900 chars ≈ 475 tokens keeps each note a single near-max
    # chunk (just under the 512 chunk size, avoiding a tiny second chunk). The
    # model runs in fp16, so ``oom_count`` must be large enough that this many
    # half-precision full-length sequences overflow VRAM.
    base = max(source, key=len) if source else "word "
    per_note = (base * ((1900 // max(1, len(base))) + 1))[:1900]
    texts = [f"{i} {per_note}" for i in range(args.oom_count)]

    df = pd.DataFrame({"note_text": texts})
    df["text_hash"] = df["note_text"].apply(compute_text_hash)
    df["patient_id"] = df["text_hash"]
    input_hashes = set(df["text_hash"])
    logger.info("ray: built %d near-max-length notes (%d chars each)", len(texts), len(per_note))

    ok = True
    with tempfile.TemporaryDirectory(dir=args.workdir) as tmp:
        in_path = Path(tmp) / "shield_input.parquet"
        out_path = Path(tmp) / "transformer_out"
        df.to_parquet(in_path, index=False)

        # Cross-process handled-OOM counter: the actor runs in a Ray worker, so we
        # cannot read its in-process ``_handled_oom_count``. It appends to this
        # file (env var inherited by the local Ray workers) each time it recovers
        # from a CUDA OOM; we count the lines afterwards to prove the recovery
        # path actually fired.
        count_file = Path(tmp) / "handled_oom.count"
        os.environ["TIDE2_OOM_COUNT_FILE"] = str(count_file)

        # Force an oversized ACTOR batch: ``map_batches``'s ``batch_size`` caps how
        # many rows the actor receives per ``__call__``, so pinning
        # ``gpu_batch_size`` alone never builds an oversized forward — the batch fed
        # to the actor must itself be large. Drive both to ``oom_count`` so the
        # actor gets one huge batch and attempts a single oversized forward that
        # OOMs, exercising recovery.
        runner = LocalJobRunner(num_gpus=1)
        try:
            runner.run_transformer(
                input_path=str(in_path),
                output_path=str(out_path),
                model_name=args.model,
                batch_size=args.oom_count,  # oversized actor batch (see above)
                gpu_batch_size=args.oom_count,  # inflate to force a first-attempt OOM in the actor
                enable_checkpoint=False,
            )
        finally:
            runner.shutdown()
            os.environ.pop("TIDE2_OOM_COUNT_FILE", None)

        handled = 0
        if count_file.exists():
            handled = sum(1 for _ in count_file.open())
        logger.info("ray: actor handled %d OOM(s) during recovery", handled)
        if handled <= 0:
            logger.error("ray: FAIL oversized batch did not trigger a handled OOM (test is vacuous)")
            ok = False

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

    logger.info("ray: %s", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    """CLI entry point for manual GPU verification runs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["direct", "ray", "reserved-bounded", "both", "all"], default="all")
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
    parser.add_argument(
        "--compile-churn",
        action="store_true",
        help="reserved-bounded: force reduce-overhead compile to reproduce the blocker.",
    )
    parser.add_argument("--workdir", default=".", help="Parent dir for the Ray temp workspace.")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        logger.exception("torch not importable")
        return 2
    if not torch.cuda.is_available():
        logger.error("No GPU available; this harness requires CUDA. Exiting non-zero.")
        return 2

    results: dict[str, bool] = {}
    if args.mode in ("direct", "both", "all"):
        results["direct"] = run_direct(args)
    if args.mode in ("ray", "both", "all"):
        results["ray"] = run_ray(args)
    if args.mode in ("reserved-bounded", "all"):
        results["reserved-bounded"] = run_reserved_bounded(args)

    logger.info("summary: %s", {k: ("PASS" if v else "FAIL") for k, v in results.items()})
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
