#!/usr/bin/env python3
"""Throughput benchmark + knob sweep for the ``run transformer`` Ray stage.

Drives the **real CLI path** (``python -m tide2.runner.cli run transformer``) over
the long-notes corpus, sampling GPU and CPU utilization for the duration of each
run, and reports throughput (notes/s and output-chunks/s) alongside the resource
profile. Use it to find the ``--gpu-batch-size`` / ``--batch-size`` /
``--num-agg-actors`` combination that maximizes throughput on this box.

Why the CLI subprocess (not the runner API): it faithfully exercises the same
argparse → ``cmd_run`` → ``LocalJobRunner.run_transformer`` path an operator uses,
and isolates each config in its own Ray process so runs don't contaminate each
other's allocator/actor state.

The long-notes corpus has no ``patient_id`` column (``run_transformer`` requires
one), so the input is prepped once into ``--workdir`` and cached per ``--limit``.

Examples
--------
    # Quick sweep on 25 notes to find the knee, monitoring GPU/CPU:
    python scripts/benchmark_transformer_throughput.py --sweep --limit 25

    # Single full run of all 1000 notes with a chosen config:
    python scripts/benchmark_transformer_throughput.py \
        --limit 0 --gpu-batch-size 128 --batch-size 1024 --num-agg-actors 8

    # Show the commands without running them:
    python scripts/benchmark_transformer_throughput.py --sweep --dry-run

Outputs (under ``--workdir``): per-run stdout logs, a per-run GPU/CPU time-series
CSV, and a ``results.json`` / ``results.csv`` summary sorted by throughput.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import statistics
import subprocess  # nosec B404 # only used with hardcoded/operator-built arg lists, no shell
import sys
import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

DEFAULT_CORPUS = "/data/tmp/long_notes-000000000000.parquet"
DEFAULT_MODEL = "20260211_debertav3_finetuned"
DEFAULT_WORKDIR = "/data/tmp/tide2_bench"

# Mean chars/note of the full long-notes corpus (measured). Used to scale a
# truncated-note sweep's throughput to a real full-run ETA (chunk count, hence GPU
# time, scales ~linearly with note length).
FULL_CORPUS_MEAN_CHARS = 439_184


# --------------------------------------------------------------------------- #
# Input prep
# --------------------------------------------------------------------------- #
def prepare_input(corpus_glob: str, workdir: Path, limit: int, max_note_chars: int = 0) -> tuple[Path, int]:
    """Materialize a ``run_transformer``-ready parquet.

    Reads the long-notes corpus and keeps ``note_text`` + ``text_hash``, renaming
    ``stanford_patient_uid`` → ``patient_id`` (the column ``run_transformer``
    requires). ``max_note_chars`` > 0 truncates each note so a small suite still
    exercises multi-chunk chunking + per-note merge but finishes fast. Cached per
    ``(limit, max_note_chars)`` so repeated sweeps skip the re-write.

    Returns ``(input_path, num_notes)``.
    """
    import pandas as pd

    tag = "all" if limit <= 0 else str(limit)
    chartag = "" if max_note_chars <= 0 else f"_c{max_note_chars}"
    out = workdir / f"bench_input_{tag}{chartag}.parquet"
    files = sorted(glob.glob(corpus_glob))  # noqa: PTH207 # user-supplied glob may span dirs
    if not files:
        raise FileNotFoundError(f"no parquet files matched {corpus_glob!r}")

    if out.exists():
        n = pd.read_parquet(out, columns=["text_hash"]).shape[0]
        print(f"[prep] reusing cached input {out} ({n} notes)")
        return out, n

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if limit > 0:
        df = df.head(limit)

    for required in ("note_text", "text_hash", "stanford_patient_uid"):
        if required not in df.columns:
            raise ValueError(f"corpus missing {required!r} column; found {list(df.columns)}")

    prepped = df[["note_text", "text_hash", "stanford_patient_uid"]].rename(
        columns={"stanford_patient_uid": "patient_id"}
    )
    prepped = prepped.dropna(subset=["note_text"])
    if max_note_chars > 0:
        prepped["note_text"] = prepped["note_text"].str.slice(0, max_note_chars)
    workdir.mkdir(parents=True, exist_ok=True)
    prepped.to_parquet(out, index=False)
    mean_chars = int(prepped["note_text"].str.len().mean()) if len(prepped) else 0
    print(f"[prep] wrote {out} ({len(prepped)} notes, mean {mean_chars} chars/note)")
    return out, len(prepped)


# --------------------------------------------------------------------------- #
# Resource sampler
# --------------------------------------------------------------------------- #
class ResourceSampler:
    """Background thread sampling whole-box GPU and CPU/RAM at a fixed cadence.

    GPU is read via ``nvidia-smi`` (utilization %, memory used MiB, power W); CPU
    and RAM via ``psutil``. Samples are whole-device / whole-box (the actor runs in
    a Ray worker, so per-process counters would miss it).
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
        """Mean / p50 / p95 / max of each sampled metric (empty -> zeros)."""
        if not self.samples:
            return {}

        def agg(key: str) -> dict[str, float]:
            vals = [s[key] for s in self.samples]
            return {
                f"{key}_mean": round(statistics.mean(vals), 1),
                f"{key}_p50": round(statistics.median(vals), 1),
                f"{key}_p95": round(sorted(vals)[min(len(vals) - 1, int(0.95 * len(vals)))], 1),
                f"{key}_max": round(max(vals), 1),
            }

        result: dict[str, float] = {"n_samples": len(self.samples)}
        for key in ("gpu_util", "gpu_mem_mib", "gpu_power_w", "cpu_pct", "ram_used_gb"):
            result.update(agg(key))
        return result

    def gpu_busy_seconds(self, util_threshold: float = 20.0) -> float:
        """Approx seconds the GPU was actively working (samples at/above threshold).

        This excludes model-load / Ray-init / write idle time, so chunks divided by
        this is a far better steady-state throughput estimate on small runs than
        chunks / wall-clock.
        """
        busy = sum(1 for s in self.samples if s["gpu_util"] >= util_threshold)
        return round(busy * self.interval, 1)

    def write_timeseries(self, path: Path) -> None:
        if not self.samples:
            return
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.samples[0].keys()))
            w.writeheader()
            w.writerows(self.samples)


# --------------------------------------------------------------------------- #
# One config run
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    """One point in the knob space to benchmark."""

    gpu_batch_size: int
    batch_size: int
    num_agg_actors: int | None = None
    num_transformer_actors: int = 1
    chunk_overlap: int | None = None
    enable_checkpoint: bool = False  # off for clean throughput (no resume skip)

    def label(self) -> str:
        agg = self.num_agg_actors if self.num_agg_actors is not None else "auto"
        ck = "on" if self.enable_checkpoint else "off"
        co = self.chunk_overlap if self.chunk_overlap is not None else "def"
        return f"gbs{self.gpu_batch_size}_bs{self.batch_size}_co{co}_agg{agg}_ck{ck}"


@dataclass
class RunResult:
    label: str
    config: dict
    ok: bool
    elapsed_s: float
    notes: int
    out_chunks: int
    notes_per_s: float
    chunks_per_s: float  # end-to-end: chunks / wall (dominated by startup on tiny runs)
    forward_passes: int  # approx ceil(out_chunks / gpu_batch_size)
    gpu_busy_s: float  # seconds the GPU was actively working (util >= threshold)
    chunks_per_gpu_s: float  # chunks / gpu_busy_s (compare only when chunking is fixed)
    notes_per_gpu_s: float  # documents / gpu_busy_s — north-star across chunk_size/overlap changes
    handled_ooms: int
    resources: dict = field(default_factory=dict)


def build_cli(runner_cmd: list[str], cfg: RunConfig, model: str, in_path: Path, out_path: Path) -> list[str]:
    """Assemble the ``run transformer`` CLI invocation for one config."""
    cmd = [
        *runner_cmd,
        "run",
        "transformer",
        "-i",
        str(in_path),
        "-o",
        str(out_path),
        "--model",
        model,
        "--num-gpus",
        "1",
        "--gpu-batch-size",
        str(cfg.gpu_batch_size),
        "--batch-size",
        str(cfg.batch_size),
        "--log-level",
        "INFO",
    ]
    if cfg.num_agg_actors is not None:
        cmd += ["--num-agg-actors", str(cfg.num_agg_actors)]
    if cfg.chunk_overlap is not None:
        cmd += ["--chunk-overlap", str(cfg.chunk_overlap)]
    if not cfg.enable_checkpoint:
        cmd += ["--no-checkpoint"]
    return cmd


def count_output_rows(out_path: Path) -> int:
    """Total rows written across the output parquet shards.

    The transformer stage now emits one document-level row per note (no per-chunk
    rows), so this equals the note count on a clean run.
    """
    import pyarrow.parquet as pq

    files = list(out_path.glob("**/*.parquet"))
    return sum(pq.read_metadata(f).num_rows for f in files)


def run_one(
    cfg: RunConfig,
    *,
    runner_cmd: list[str],
    model: str,
    in_path: Path,
    notes: int,
    workdir: Path,
    interval: float,
    dry_run: bool,
) -> RunResult:
    """Run one config end-to-end while sampling resources; return its metrics."""
    out_path = workdir / f"out_{cfg.label()}"
    log_path = workdir / f"log_{cfg.label()}.txt"
    ts_path = workdir / f"resources_{cfg.label()}.csv"

    # Clean prior output + its checkpoint sibling so throughput isn't skewed by
    # resume-skipping already-processed rows.
    for p in (out_path, out_path.parent / (out_path.name + "_ray_checkpoint")):
        if p.exists():
            shutil.rmtree(p)

    cmd = build_cli(runner_cmd, cfg, model, in_path, out_path)
    print(f"\n[run] {cfg.label()}\n      {' '.join(cmd)}")
    if dry_run:
        return RunResult(cfg.label(), asdict(cfg), True, 0.0, notes, 0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, {})

    start = time.time()
    with ResourceSampler(interval=interval) as sampler, log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)  # noqa: S603 # nosec B603 # operator-built CLI, no shell
    elapsed = time.time() - start
    ok = proc.returncode == 0

    sampler.write_timeseries(ts_path)
    out_chunks = count_output_rows(out_path) if ok and out_path.exists() else 0
    handled = _count_handled_ooms(log_path)
    gpu_busy = sampler.gpu_busy_seconds()
    # Output is one row per note now (windowing happens inside the actor and is not
    # observable here), so this is a coarse lower-bound forward estimate from the
    # note count, not the true window count. notes_per_gpu_s is the reliable metric.
    forwards = -(-out_chunks // cfg.gpu_batch_size) if out_chunks else 0

    if not ok:
        print(f"      FAILED (exit {proc.returncode}); see {log_path}")

    result = RunResult(
        label=cfg.label(),
        config=asdict(cfg),
        ok=ok,
        elapsed_s=round(elapsed, 1),
        notes=notes,
        out_chunks=out_chunks,
        notes_per_s=round(notes / elapsed, 2) if elapsed else 0.0,
        chunks_per_s=round(out_chunks / elapsed, 1) if elapsed else 0.0,
        forward_passes=forwards,
        gpu_busy_s=gpu_busy,
        chunks_per_gpu_s=round(out_chunks / gpu_busy, 1) if gpu_busy else 0.0,
        notes_per_gpu_s=round(notes / gpu_busy, 3) if gpu_busy else 0.0,
        handled_ooms=handled,
        resources=sampler.summary(),
    )
    _print_run_line(result)
    return result


def _count_handled_ooms(log_path: Path) -> int:
    """How many times the actor logged a halve-on-OOM recovery (gbs too large)."""
    if not log_path.exists():
        return 0
    return sum(1 for line in log_path.read_text(errors="ignore").splitlines() if "CUDA OOM; halving" in line)


def _print_run_line(r: RunResult) -> None:
    res = r.resources
    print(
        f"      -> {r.notes_per_gpu_s:>6.3f} notes/gpu-s | {r.chunks_per_gpu_s:>6.1f} chunks/gpu-s | "
        f"{r.out_chunks:>5d} chunks | {r.forward_passes:>3d} fwd | "
        f"gpu_busy {r.gpu_busy_s:>5.1f}s / wall {r.elapsed_s:>6.1f}s | "
        f"GPU {res.get('gpu_util_mean', 0):>4.0f}%/{res.get('gpu_util_max', 0):>4.0f}%, "
        f"{res.get('gpu_mem_mib_max', 0) / 1024:>4.1f}GB | OOMs={r.handled_ooms}"
    )


# --------------------------------------------------------------------------- #
# Sweep grid + reporting
# --------------------------------------------------------------------------- #
def sweep_grid(args: argparse.Namespace) -> list[RunConfig]:
    """Nested grid over the throughput-relevant knobs for this box.

    Each dim collapses to its single-run value when its ``--sweep-*`` list is
    omitted, so the same function serves a focused GPU-knee sweep or a full
    gpu_batch by batch by agg by checkpoint cross-product.
    """
    aggs = args.sweep_agg if args.sweep_agg else [args.num_agg_actors]
    ckpts = [c == "on" for c in args.sweep_checkpoint] if args.sweep_checkpoint else [args.enable_checkpoint]
    overlaps = args.sweep_chunk_overlap if args.sweep_chunk_overlap is not None else [args.chunk_overlap]
    grid: list[RunConfig] = []
    for gbs in args.sweep_gpu_batch:
        for bs in args.sweep_batch:
            for co in overlaps:
                for agg in aggs:
                    for ckpt in ckpts:
                        grid.append(
                            RunConfig(
                                gpu_batch_size=gbs,
                                batch_size=bs,
                                num_agg_actors=(None if agg in (0, None) else agg),
                                chunk_overlap=co,
                                enable_checkpoint=ckpt,
                            )
                        )
    return grid


def persist_results(results: list[RunResult], workdir: Path) -> None:
    """Write results.json/csv (ranked by documents/GPU-second) after each config.

    Called incrementally so a mid-sweep kill still leaves the completed configs on
    disk. ``notes_per_gpu_s`` is the ranking key: it is the true document
    throughput and stays comparable even when chunk_size/overlap change the chunk
    count (chunks/wall is startup-dominated on small suites and misleads).
    """
    ranked = sorted(results, key=lambda r: r.notes_per_gpu_s, reverse=True)
    (workdir / "results.json").write_text(json.dumps([asdict(r) for r in ranked], indent=2))

    with (workdir / "results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "label",
                "ok",
                "notes_per_gpu_s",
                "chunks_per_gpu_s",
                "out_chunks",
                "forward_passes",
                "gpu_busy_s",
                "elapsed_s",
                "gpu_util_mean",
                "gpu_mem_gb_max",
                "cpu_pct_mean",
                "handled_ooms",
            ]
        )
        for r in ranked:
            res = r.resources
            w.writerow(
                [
                    r.label,
                    r.ok,
                    r.notes_per_gpu_s,
                    r.chunks_per_gpu_s,
                    r.out_chunks,
                    r.forward_passes,
                    r.gpu_busy_s,
                    r.elapsed_s,
                    res.get("gpu_util_mean", 0),
                    round(res.get("gpu_mem_mib_max", 0) / 1024, 2),
                    res.get("cpu_pct_mean", 0),
                    r.handled_ooms,
                ]
            )


def print_ranking(results: list[RunResult], workdir: Path, eta_scale: float = 1.0) -> None:
    """Print the final ranking and a full-run ETA for the best config.

    ``eta_scale`` scales the ETA from the (possibly truncated) test note length up
    to the real corpus mean, so a fast truncated sweep still yields an honest
    full-run estimate.
    """
    ranked = sorted(results, key=lambda r: r.notes_per_gpu_s, reverse=True)
    print("\n" + "=" * 116)
    print("RANKING (by documents / GPU-busy-second — the real throughput, chunk-count-invariant)")
    print("=" * 116)
    for i, r in enumerate(ranked, 1):
        _print_run_line(r)
        print(f"   #{i} {r.label}")
    print(f"\nSummary written to {workdir / 'results.json'} and {workdir / 'results.csv'}")
    if ranked and ranked[0].ok and ranked[0].notes_per_gpu_s:
        best = ranked[0]
        eta_h = 1000 * eta_scale / best.notes_per_gpu_s / 3600  # full corpus = 1000 long notes
        scale_note = "" if eta_scale == 1.0 else f", scaled x{eta_scale:.1f} to full note length"
        print(
            f"Best: {best.label} at {best.notes_per_gpu_s} notes/gpu-s "
            f"(~{eta_h:.1f}h GPU time for the full 1000-note run{scale_note})"
        )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=DEFAULT_CORPUS, help="Input parquet glob (default: long-notes corpus).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Transformer model name.")
    p.add_argument("--workdir", default=DEFAULT_WORKDIR, help="Scratch dir for input/output/logs.")
    p.add_argument("--limit", type=int, default=25, help="Notes to use (0 = all 1000). Use a small value for sweeps.")
    p.add_argument(
        "--max-note-chars",
        type=int,
        default=0,
        help="Truncate each note to this many chars (0 = full). Use e.g. 6000 for a fast "
        "multi-chunk suite that still exercises chunking across notes + per-note merge.",
    )
    p.add_argument("--sample-interval", type=float, default=1.0, help="Resource sampling cadence (seconds).")
    p.add_argument("--dry-run", action="store_true", help="Print the CLI commands without running.")
    p.add_argument(
        "--runner-cmd",
        default=f"{sys.executable} -m tide2.runner.cli",
        help="Command that launches the runner CLI (default: current interpreter -m tide2.runner.cli).",
    )

    # Single-run knobs (ignored in --sweep mode).
    p.add_argument("--gpu-batch-size", type=int, default=64, help="Windows per GPU forward (single run).")
    p.add_argument("--batch-size", type=int, default=8, help="Whole notes per actor __call__ (single run).")
    p.add_argument("--num-agg-actors", type=int, default=None, help="BIO aggregation actors (None = auto ~30%% CPUs).")
    p.add_argument("--chunk-overlap", type=int, default=None, help="Window token overlap (None = model config).")
    p.add_argument("--enable-checkpoint", action="store_true", help="Enable Ray checkpointing (off by default here).")

    # Sweep mode + grid.
    p.add_argument("--sweep", action="store_true", help="Sweep the grid instead of a single run.")
    p.add_argument("--sweep-gpu-batch", type=int, nargs="+", default=[32, 64, 128, 256], help="gpu_batch_size grid.")
    p.add_argument("--sweep-batch", type=int, nargs="+", default=[512, 1024], help="batch_size grid.")
    p.add_argument(
        "--sweep-agg", type=int, nargs="+", default=None, help="num_agg_actors grid (0 = auto). Omit = single value."
    )
    p.add_argument(
        "--sweep-checkpoint", nargs="+", choices=["off", "on"], default=None, help="checkpoint modes. Omit = single."
    )
    p.add_argument(
        "--sweep-chunk-overlap",
        type=int,
        nargs="+",
        default=None,
        help="window overlap grid in tokens (lower = fewer redundant windows). Omit = single.",
    )
    args = p.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    in_path, notes = prepare_input(args.corpus, workdir, args.limit, args.max_note_chars)
    print(f"[bench] {notes} notes | model={args.model} | workdir={workdir}")

    runner_cmd = args.runner_cmd.split()
    if args.sweep:
        configs = sweep_grid(args)
        print(f"[bench] sweeping {len(configs)} configs on {notes} notes")
    else:
        configs = [
            RunConfig(
                gpu_batch_size=args.gpu_batch_size,
                batch_size=args.batch_size,
                num_agg_actors=args.num_agg_actors,
                chunk_overlap=args.chunk_overlap,
                enable_checkpoint=args.enable_checkpoint,
            )
        ]

    results: list[RunResult] = []
    for cfg in configs:
        results.append(
            run_one(
                cfg,
                runner_cmd=runner_cmd,
                model=args.model,
                in_path=in_path,
                notes=notes,
                workdir=workdir,
                interval=args.sample_interval,
                dry_run=args.dry_run,
            )
        )
        if not args.dry_run:
            persist_results(results, workdir)  # durable after every config

    if not args.dry_run:
        # Scale the ETA from the (possibly truncated) test note length to the real
        # corpus mean, so a fast truncated sweep still gives an honest full-run ETA.
        test_chars = (
            min(args.max_note_chars, FULL_CORPUS_MEAN_CHARS) if args.max_note_chars > 0 else FULL_CORPUS_MEAN_CHARS
        )
        eta_scale = FULL_CORPUS_MEAN_CHARS / test_chars
        print_ranking(results, workdir, eta_scale=eta_scale)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
