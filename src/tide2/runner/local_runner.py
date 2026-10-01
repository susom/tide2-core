"""
Ray-based job runner for TIDE 2.0.

Execution Environments:
    - Local machine or single GCP VM, using Ray for parallelism.
    - Any host or orchestration task that instantiates LocalJobRunner to
      execute recognition, anonymization, or transformer stages.

Examples:
    # Local development
    runner = LocalJobRunner()
    runner.run_recognition("./data/input", "./data/output")

    # Single VM with GCS
    runner = LocalJobRunner(num_cpus=224, object_store_gb=100)
    runner.run_recognition("gs://bucket/input", "gs://bucket/output")

    # GPU transformer stage, with explicit shutdown
    runner = LocalJobRunner(num_gpus=1)
    try:
        runner.run_transformer(input_path, output_path, model_name=model)
    finally:
        runner.shutdown()
"""

import hashlib
import json
import logging
import os
import shutil
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq
import ray
import ray.data
import ray.data.exceptions
from ray.data.checkpoint import CheckpointConfig
from ray.data.dataset import Dataset

from .fault_tolerance import GracefulShutdown
from .fault_tolerance import configure_data_context
from .fault_tolerance import get_ray_remote_args_cpu
from .fault_tolerance import get_ray_remote_args_gpu
from .hardware import alive_node_cpus
from .hardware import apply_recommendations
from .hardware import detect_hardware
from .hardware import recommend_object_store_gb
from .hardware import recommend_settings
from .hardware import render_settings_table
from .utils import DEFAULT_DASHBOARD_HOST
from .utils import detect_columns
from .utils import gpu_worker_runtime_env
from .utils import log_ray_cluster_info
from .utils import resolve_input_files

logger = logging.getLogger(__name__)

KEY_SIZE_BYTES = 32
DEFAULT_RECOGNITION_BATCH_TIMEOUT = 120
DEFAULT_LLM_BATCH_TIMEOUT = 300


def _resolve_slot_cpus(
    num_cpus: int | float | None,
    worker_num_cpus: int | float | None,
    stage_name: str = "",
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Resolve pool slot CPU reservation, preserving pre-collapse accounting.

    Before removing the supervisor tier, each pool slot reserved ``num_cpus`` for
    the supervisor actor plus ``worker_num_cpus`` (defaulting to 1.0) for the
    spawned worker actor. To maintain exact cluster resource allocation across
    all existing callers and configurations, this helper adds the two values together:
        slot_cpus = float(num_cpus or 0.0) + (1.0 if worker_num_cpus is None else float(worker_num_cpus))

    Args:
        num_cpus: CPUs per actor/slot (deprecated).
        worker_num_cpus: CPUs reserved for the worker actor.
        stage_name: Name of the pipeline stage for logging/warnings.

    Returns:
        Tuple of (slot_cpus, ray_remote_args, resolved_config_dict).
    """
    if num_cpus is not None:
        import warnings

        warnings.warn(
            f"{stage_name + ': ' if stage_name else ''}`num_cpus` (or `--cpus-per-actor` / YAML `cpus_per_actor`) "
            "for stage actors is deprecated in favor of `worker_num_cpus`. "
            "Total reserved CPUs per pool slot remain identical via additive resolution. "
            "(Note: runner cluster-level `LocalJobRunner(num_cpus=...)` is unchanged.)",
            DeprecationWarning,
            stacklevel=3,
        )

    slot_cpus = float(num_cpus or 0.0) + (1.0 if worker_num_cpus is None else float(worker_num_cpus))
    ray_remote_args = get_ray_remote_args_cpu(num_cpus=slot_cpus)
    resolved = {
        "stage": stage_name,
        "input_num_cpus": num_cpus,
        "input_worker_num_cpus": worker_num_cpus,
        "resolved_slot_cpus": slot_cpus,
        "ray_remote_args": ray_remote_args,
    }
    return slot_cpus, ray_remote_args, resolved


DEPRECATED_TRANSFORMER_KWARGS: frozenset[str] = frozenset(
    {
        "chunk_size",
        "flat_map_cpus",
        "compile_model",
        "compile_cache_path",
        "pre_chunked",
        "short_seq_budget",
    }
)


def _check_deprecated_transformer_kwargs(kwargs: dict[str, Any], caller_name: str) -> None:
    """Validate keyword arguments against deprecated transformer parameters.

    Args:
        kwargs: Keyword arguments passed to run_transformer.
        caller_name: Name of the calling function/method for error messages.

    Raises:
        ValueError: If any deprecated parameter is supplied.
        TypeError: If an unrecognized parameter is supplied.
    """
    for param in (
        "chunk_size",
        "flat_map_cpus",
        "compile_model",
        "compile_cache_path",
        "pre_chunked",
        "short_seq_budget",
    ):
        if param in kwargs:
            msg = f"The parameter '{param}' is deprecated and no longer supported. Please remove it from your call."
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
            raise ValueError(msg)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"{caller_name}() got an unexpected keyword argument '{unexpected}'")


def _configure_checkpoint(
    ctx: "ray.data.DataContext",
    *,
    enable: bool,
    output_dir: Path,
    id_column: str,
) -> None:
    """Configure (or clear) Ray Data row-level checkpointing on ``ctx``.

    When ``enable`` is True, points the checkpoint at a sibling
    ``<output_dir>_ray_checkpoint`` directory keyed on ``id_column``. When False,
    clears any checkpoint config so it does not leak into the stage.

    CRITICAL on tiny clusters (≲4 CPUs, e.g. 2-CPU Colab): enabling checkpointing
    injects a sort + repartition shuffle whose per-operator CPU reservations,
    under Ray 2.55's ReservationOpResourceAllocator, sum to more than the cluster
    has — so nothing schedules and the stage hangs at 0/1
    (``backpressured:tasks(ResourceBudget)``). Pass ``enable=False`` on such boxes;
    the cost is loss of row-level resume, not correctness.

    Note: ``ctx.op_resource_reservation_enabled = False`` does NOT resolve this —
    the per-operator reservation floors still exceed a ≲4-CPU cluster. The
    fractional-CPU knobs + ``enable=False`` are the validated fix.
    """
    if enable:
        checkpoint_dir = output_dir.parent / (output_dir.name + "_ray_checkpoint")
        ctx.checkpoint_config = CheckpointConfig(
            id_column=id_column,
            checkpoint_path=str(checkpoint_dir),
            delete_checkpoint_on_success=False,
        )
    else:
        ctx.checkpoint_config = None


@dataclass(frozen=True)
class StageColumns:
    """Static column contract for one pipeline stage.

    Replaces ``detect_columns``' Parquet-footer sniffing on the streamed path,
    where a chained stage has no file to read. ``requires`` must be present on
    the stage's input, ``optional`` is passed through when present, and
    ``produces`` is what the stage adds to (or replaces in) its output.
    """

    requires: frozenset[str]
    optional: frozenset[str]
    produces: frozenset[str]

    def available_after(self, upstream: frozenset[str]) -> frozenset[str]:
        """Columns available downstream of this stage given ``upstream`` columns."""
        return (upstream & (self.requires | self.optional)) | self.produces


#: Column contracts, mirroring the actor implementations in ``tide2.actors``.
TRANSFORMER_STAGE_COLUMNS = StageColumns(
    requires=frozenset({"text_hash", "note_text"}),
    optional=frozenset({"patient_id", "patient_identifiers", "patient_uid", "jitter", "row_id"}),
    produces=frozenset({"text_hash", "patient_id", "note_text", "recognizer_results_json"}),
)
RECOGNIZER_STAGE_COLUMNS = StageColumns(
    requires=frozenset({"text_hash", "note_text"}),
    optional=frozenset(
        {"patient_identifiers", "recognizer_results_json", "patient_id", "patient_uid", "jitter", "row_id"}
    ),
    produces=frozenset(
        {
            "text_hash",
            "note_text",
            "patient_uid",
            "row_id",
            "recognizer_results_json",
            "entity_count",
            "processing_timestamp",
            "processing_status",
            "error_message",
        }
    ),
)
LLM_RECOGNIZER_STAGE_COLUMNS = StageColumns(
    requires=frozenset({"text_hash", "note_text"}),
    optional=frozenset(),
    produces=frozenset(
        {"text_hash", "note_text", "recognizer_results_json", "entity_count", "processing_status", "error_message"}
    ),
)
ANONYMIZER_STAGE_COLUMNS = StageColumns(
    requires=frozenset({"text_hash", "note_text", "recognizer_results_json"}),
    optional=frozenset({"patient_uid", "patient_id", "jitter", "row_id"}),
    produces=frozenset(
        {
            "text_hash",
            "patient_uid",
            "anonymized_note_text",
            "anonymizer_results_json",
            "entity_count",
            "processing_timestamp",
            "processing_status",
            "error_message",
        }
    ),
)

#: Columns the final sink is allowed to carry. Raw ``note_text`` is deliberately
#: absent — see ``run_pipeline``'s invariants.
FINAL_OUTPUT_COLUMNS = frozenset(
    {
        "text_hash",
        "patient_uid",
        "row_id",
        "anonymized_note_text",
        "anonymizer_results_json",
        "entity_count",
        "processing_timestamp",
        "processing_status",
        "error_message",
    }
)

#: A node with at most this many CPUs cannot hold three concurrent actor pools.
MIN_STREAMED_NODE_CPUS = 4

#: Fraction of input rows that may be dropped before the streamed path warns.
STREAMED_DROP_WARN_FRACTION = 0.01


def _streamed_source_columns(contracts: list[tuple[str, "StageColumns"]]) -> frozenset[str]:
    """Columns the source dataset must carry for a chained plan.

    Only the first stage reads from the source; everything after it reads that
    stage's output. Projecting the source to this set is both a memory control
    (``note_text`` is the bulk of every block, and blocks now live in the object
    store across all three operators) and a leakage control.
    """
    if not contracts:
        return frozenset()
    first = contracts[0][1]
    return first.requires | first.optional


def validate_stage_columns(
    source_columns: Iterable[str],
    stages: list[tuple[str, StageColumns]],
) -> frozenset[str]:
    """Validate a chained plan's column contracts against the source columns.

    Walks the ``stages`` in order, threading each stage's ``available_after``
    set into the next, and raises before a single operator is built. This runs
    entirely on the static contracts plus the *source* column list, so it never
    calls ``Dataset.schema()`` on a lazy mid-plan dataset (which can execute
    upstream operators and silently defeat pipelining).

    Args:
        source_columns: Columns present on the source dataset.
        stages: ``(stage_name, contract)`` pairs in execution order.

    Returns:
        The columns available after the last stage, including pass-through
        columns a stage forwards but does not itself produce (``row_id``,
        ``patient_uid``, ``jitter``).

    Raises:
        ValueError: If a stage's required columns are not available, naming the
            stage and the missing columns.
    """
    available = frozenset(source_columns)
    for name, contract in stages:
        missing = contract.requires - available
        if missing:
            raise ValueError(
                f"Stage {name!r} requires column(s) {sorted(missing)} which are not available at that point "
                f"in the chain. Available: {sorted(available)}."
            )
        available = contract.available_after(available)
    return available


def _alive_node_cpus() -> list[float]:
    """CPU capacity of each alive node in the cluster.

    Goes through :mod:`tide2.runner.hardware`, which is the only module that
    reads ``ray.nodes()`` directly.
    """
    return alive_node_cpus()


def _log_execution_timeout(stage: str, ctx: Any, extra: str = "") -> None:
    """Log a Ray Data no-progress timeout uniformly for every stage.

    Args:
        stage: Human-readable stage name for the message.
        ctx: The Ray ``DataContext`` the stage ran under.
        extra: Optional sentence appended for stage-specific guidance.
    """
    logger.exception(
        "%s failed due to execution timeout (no_progress_timeout_s=%s).%s",
        stage,
        getattr(ctx, "execution_no_progress_timeout_s", None),
        f" {extra}" if extra else "",
    )


def check_streamed_admission(pool_minimums: dict[str, float]) -> float:
    """Fail fast instead of deadlocking when the chained plan cannot be scheduled.

    A chained plan holds every operator's pool resident at once. Under Ray
    2.55+'s ``ReservationOpResourceAllocator`` the per-operator reservations must
    all fit on a single node or nothing schedules and the run hangs at ``0/1``.
    This converts that silent hang into an error.

    This is a **heuristic**: the allocator reserves via
    ``op_resource_reservation_ratio`` rather than literally summing pool
    minimums, so it can pass and still be tight. The real backstop is the
    execution-level no-progress guard (``no_progress_timeout_s``), which fires on
    a ``0/1`` deadlock because that is pure no-progress.

    Args:
        pool_minimums: Operator name → minimum CPUs it reserves.

    Returns:
        The CPU capacity of the largest alive node.

    Raises:
        ValueError: If the cluster is multi-node, the largest node has
            ≤ ``MIN_STREAMED_NODE_CPUS`` CPUs, or the minimums do not fit.
    """
    node_cpus = _alive_node_cpus()
    if not node_cpus:
        raise ValueError("No alive Ray nodes found; cannot admit a streamed plan.")
    if len(node_cpus) > 1:
        raise ValueError(
            f"execution_mode='streamed' is single-node only; found {len(node_cpus)} alive nodes "
            f"({node_cpus} CPUs). Use execution_mode='discrete', which is the multi-node path."
        )
    largest = max(node_cpus)
    if largest <= MIN_STREAMED_NODE_CPUS:
        raise ValueError(
            f"execution_mode='streamed' needs more than {MIN_STREAMED_NODE_CPUS} CPUs on a single node "
            f"(largest node has {largest}). Use execution_mode='discrete' with the small-box recipe "
            "(fractional CPU knobs AND enable_checkpoint=False) — see README, 'Why small boxes deadlock'."
        )

    total_min = sum(pool_minimums.values())
    budget = largest - 1.0
    if total_min > budget:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(pool_minimums.items()))
        raise ValueError(
            f"Streamed plan reserves a minimum of {total_min} CPUs ({breakdown}) but the node has "
            f"{largest} CPUs (usable budget {budget}). Lower the per-operator minimums "
            "(read_cpus, write_cpus, transformer_cpus, num_cpus/worker_num_cpus) or use "
            "execution_mode='discrete'."
        )
    return largest


class LocalJobRunner:
    """
    Ray-based job runner for single-node execution (local machine or VM).

    Features:
    - Ray Data row-level checkpointing for resume capability
    - Graceful shutdown handling
    - Native Ray fault tolerance
    """

    def __init__(
        self,
        num_cpus: int | None = None,
        num_gpus: int | float | None = None,
        object_store_gb: int | None = None,
        dashboard_host: str = DEFAULT_DASHBOARD_HOST,
        include_dashboard: bool = False,
        no_progress_timeout_s: float | None = None,
    ):
        """
        Initialize local job runner.

        Args:
            num_cpus: CPU count override
            num_gpus: GPU count override (supports fractional e.g. 0.33)
            object_store_gb: Object store size in GB (default: ~30% of system RAM)
            dashboard_host: Dashboard host
            include_dashboard: Enable Ray dashboard
            no_progress_timeout_s: Ray Data hang-detection timeout applied to every
                stage this runner launches. None = Ray Data's default. Stages reset
                the DataContext per job, so the value is re-applied on each one.
        """
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.object_store_gb = object_store_gb
        self.dashboard_host = dashboard_host
        self.include_dashboard = include_dashboard
        self.no_progress_timeout_s = no_progress_timeout_s
        self._initialized = False

    def _data_context_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """Build ``configure_data_context`` kwargs, carrying the sticky hang timeout.

        Each stage reconfigures the DataContext with its own streaming params, which
        would otherwise reset ``no_progress_timeout_s`` to the library default and
        silently discard both the CLI flag and the hardware recommendation.
        """
        kwargs: dict[str, Any] = dict(overrides)
        if self.no_progress_timeout_s is not None:
            kwargs["no_progress_timeout_s"] = self.no_progress_timeout_s
        return kwargs

    def _init_ray(self) -> None:
        """Initialize Ray."""
        if self._initialized:
            return

        # Disable uv runtime_env isolation hook so local workers directly inherit
        # the active environment without redundant per-worker venv creation
        os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

        # When dashboard is enabled, bind to 0.0.0.0 so it's accessible
        # from outside Docker containers
        dashboard_host = self.dashboard_host
        if self.include_dashboard and dashboard_host == DEFAULT_DASHBOARD_HOST:
            dashboard_host = "0.0.0.0"  # noqa: S104 # nosec B104 # we run in docker

        kwargs: dict[str, Any] = {
            "ignore_reinit_error": True,
            "include_dashboard": self.include_dashboard,
            "dashboard_host": dashboard_host,
            # Configure the CUDA allocator (expandable_segments) for GPU actors
            # before CUDA initializes in the worker. Fragmentation mitigation, not
            # a leak fix. See runner.utils.gpu_worker_runtime_env.
            "runtime_env": gpu_worker_runtime_env(),
        }

        if self.num_cpus:
            kwargs["num_cpus"] = self.num_cpus
        if self.num_gpus is not None:
            kwargs["num_gpus"] = self.num_gpus
        if self.object_store_gb:
            kwargs["object_store_memory"] = self.object_store_gb * 1024**3
        else:
            rec_gb = recommend_object_store_gb(detect_hardware())
            if rec_gb is not None:
                kwargs["object_store_memory"] = int(rec_gb * 1024**3)

        ray.init(**kwargs)
        logger.info("Ray initialized")

        # Log resources
        log_ray_cluster_info()

        # Configure Ray Data context
        configure_data_context(**self._data_context_kwargs(verbose_progress=True))

        self._initialized = True

    def _auto_num_actors(self, fraction: float | None = None) -> int:
        """Auto-detect number of actors from cluster resources.

        Args:
            fraction: Optional fraction override of cluster CPUs. If None,
                derived from the hardware profile (e.g. ~0.875 on gpu-workstation).
        """
        hw = detect_hardware()
        if fraction is not None:
            return max(1, int(hw.cluster_cpu * fraction))
        rec = recommend_settings(hw)
        rec_actors = rec.recognizer.get("num_actors")
        if rec_actors is not None:
            return int(rec_actors)
        return max(1, int(hw.cluster_cpu * 0.45))

    def _resolve_transformer_resources(
        self,
        num_gpus: int | float | None,
        num_transformer_actors: int | None,
        num_agg_actors: int | None,
    ) -> tuple[int | float, bool, int, int]:
        """Resolve GPU/CPU resources and actor counts for transformer jobs.

        Returns:
            Tuple of (num_gpus, cpu_only_mode, num_transformer_actors, num_agg_actors)
        """
        hw = detect_hardware()
        rec = recommend_settings(hw)

        if num_gpus is None:
            rec_gpus = rec.transformer.get("num_gpus")
            num_gpus = rec_gpus if rec_gpus is not None else float(hw.cluster_gpu)

        cpu_only_mode = num_gpus == 0

        if num_transformer_actors is None:
            # Each transformer actor is memory-intensive (~500MB+ for model)
            # CPU mode: limit actors; GPU mode: scale with GPUs
            rec_actors = rec.transformer.get("num_transformer_actors")
            if rec_actors is not None:
                num_transformer_actors = rec_actors
            elif cpu_only_mode:
                num_transformer_actors = max(1, int(hw.cluster_cpu * 0.25))
            elif num_gpus < 1.0 and num_gpus > 0:
                num_transformer_actors = max(1, round(1.0 / num_gpus))
            else:
                num_transformer_actors = int(num_gpus)

        if cpu_only_mode:
            logger.warning(
                "No GPUs available, running transformer inference on CPU. "
                "This will be significantly slower than GPU inference."
            )

        if num_agg_actors is None:
            rec_agg = rec.transformer.get("num_agg_actors")
            if rec_agg is not None:
                num_agg_actors = rec_agg
            else:
                num_agg_actors = 0 if not cpu_only_mode and num_gpus < 1.0 else max(1, int(hw.cluster_cpu * 0.3))

        return num_gpus, cpu_only_mode, num_transformer_actors, num_agg_actors

    # ------------------------------------------------------------------
    # Stage builders
    #
    # Each builder resolves the actor class / pool / resources, calls
    # ``map_batches``, and returns the new lazy ``Dataset``. They deliberately
    # touch nothing else — no DataContext, no checkpoints, no filesystem, no
    # timing — so the same call can be used by a discrete stage (read →
    # build → write) and by the streamed path (build → build → build → write).
    # ------------------------------------------------------------------

    @staticmethod
    def _actor_pool(num_actors: int, pool_min_size: int | None) -> "ray.data.ActorPoolStrategy":
        """Fixed pool for discrete stages, autoscaling pool for chained plans.

        ``pool_min_size=None`` reproduces the discrete path exactly
        (``size=num_actors``). Passing a minimum switches to
        ``min_size``/``max_size`` so idle pools release CPUs to the busy one as
        the bottleneck shifts from GPU to CPU — required when three pools are
        resident at once.
        """
        if pool_min_size is None:
            return ray.data.ActorPoolStrategy(size=num_actors)
        return ray.data.ActorPoolStrategy(min_size=min(pool_min_size, num_actors), max_size=num_actors)

    def build_recognizer_stage(
        self,
        ds: Dataset,
        *,
        batch_size: int,
        num_actors: int,
        ray_remote_args: dict[str, Any],
        pool_min_size: int | None = None,
    ) -> Dataset:
        """Append the regex/rule-based recognizer operator to ``ds``."""
        from tide2.actors import RecognizerActor

        return ds.map_batches(
            RecognizerActor,
            batch_size=batch_size,
            compute=self._actor_pool(num_actors, pool_min_size),
            **ray_remote_args,
        )

    def build_llm_recognizer_stage(
        self,
        ds: Dataset,
        *,
        batch_size: int,
        num_actors: int,
        ray_remote_args: dict[str, Any],
        fn_constructor_kwargs: dict[str, Any],
        pool_min_size: int | None = None,
    ) -> Dataset:
        """Append the LLM recognizer operator to ``ds``.

        This stage is network-bound with its own retry behaviour: it *reserves*
        CPU but barely uses any, so in a chained plan keep its slot reservation
        small (e.g. ``num_cpus=0.25``) and size its pool independently of the
        CPU budget.
        """
        from tide2.actors import LlmRecognizerActor

        return ds.map_batches(
            LlmRecognizerActor,
            batch_size=batch_size,
            compute=self._actor_pool(num_actors, pool_min_size),
            fn_constructor_kwargs=fn_constructor_kwargs,
            **ray_remote_args,
        )

    def build_anonymizer_stage(
        self,
        ds: Dataset,
        *,
        actor_cls: type,
        batch_size: int,
        num_actors: int,
        ray_remote_args: dict[str, Any],
        pool_min_size: int | None = None,
    ) -> Dataset:
        """Append the anonymizer operator to ``ds``.

        ``actor_cls`` comes from ``create_anonymizer_actor_class`` (it closes
        over the salt/key), so the builder never handles key material.
        """
        return ds.map_batches(
            actor_cls,
            batch_size=batch_size,
            compute=self._actor_pool(num_actors, pool_min_size),
            **ray_remote_args,
        )

    def build_transformer_stage(
        self,
        ds: Dataset,
        *,
        transformer_actor: type,
        model_name: str,
        batch_size: int,
        num_transformer_actors: int,
        ray_remote_args_transformer: dict[str, Any],
        num_agg_actors: int,
        agg_num_cpus: float,
        pool_min_size: int = 1,
    ) -> Dataset:
        """Append the transformer inference (and optional BIO aggregation) operators.

        When ``num_agg_actors == 0`` the actor aggregates BIO tokens in place and
        a single operator is appended; otherwise a second CPU operator is chained
        on, which streams concurrently with the GPU one.
        """
        ds_raw = ds.map_batches(
            transformer_actor,
            batch_size=batch_size,
            batch_format="numpy",
            compute=ray.data.ActorPoolStrategy(min_size=pool_min_size, max_size=num_transformer_actors),
            **ray_remote_args_transformer,
        )
        if num_agg_actors == 0:
            return ds_raw

        from tide2.actors import BIOAggregationActor

        return ds_raw.map_batches(
            BIOAggregationActor,
            batch_size=batch_size,
            batch_format="numpy",
            compute=ray.data.ActorPoolStrategy(size=num_agg_actors),
            fn_constructor_kwargs={"model_name": model_name},
            **get_ray_remote_args_cpu(num_cpus=agg_num_cpus),
        )

    def run_recognition(  # noqa: PLR0915
        self,
        input_path: str | list[str],
        output_path: str,
        num_actors: int | None = None,
        batch_size: int = 150,
        batch_timeout: int = 120,
        num_cpus: int | float = 2,
        read_parallelism: int | None = None,
        read_cpus: float = 0.25,
        read_op_min_num_blocks: int = 200,
        target_max_block_size_mb: int = 128,
        target_min_block_size_mb: int = 1,
        worker_num_cpus: int | float | None = None,
        write_cpus: float = 1.0,
        enable_checkpoint: bool = True,
        override_num_blocks: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Run recognition job with Ray Data checkpointing for resume.

        Args:
            input_path: Input parquet files (local path or GCS URI)
            output_path: Output directory
            num_actors: Actor count (auto-detect if None)
            batch_size: Batch size per actor
            num_cpus: CPUs per actor (affects streaming executor scheduling)
            read_parallelism: Number of read output blocks (default: num input files)
            read_cpus: CPUs per read task (lower = more concurrent reads)
            read_op_min_num_blocks: Minimum read output blocks for DataContext
            target_max_block_size_mb: Max block size in MB for DataContext
            target_min_block_size_mb: Min block size in MB for DataContext
            worker_num_cpus: CPUs to reserve for each supervisor's worker actor.
                None = Ray default (1). Each pool slot needs supervisor
                (num_cpus) + worker (worker_num_cpus) CPUs; lower both to fit
                small boxes.
            write_cpus: CPUs to reserve for each write_parquet task. Default 1.0
                reproduces Ray's default task reservation.
            enable_checkpoint: If True (default), enable Ray Data row-level
                checkpointing for resume. MUST be set to False on tiny clusters
                (≲4 CPUs, e.g. Google Colab): the checkpoint pipeline adds a
                sort+repartition shuffle whose per-operator CPU reservations
                exceed the cluster, deadlocking the stage at 0/1. Disabling it
                trades resume capability (not correctness) for the ability to run.
            override_num_blocks: Explicit number of Ray Data blocks to split the
                input into (e.g. 32 to fix single-block starvation on multicore nodes).
            dry_run: If True, validate setup and show plan without processing

        Returns:
            Processing statistics dictionary
        """
        self._init_ray()

        # Override DataContext with job-specific streaming params
        configure_data_context(
            **self._data_context_kwargs(
                verbose_progress=True,
                target_max_block_size_mb=target_max_block_size_mb,
                target_min_block_size_mb=target_min_block_size_mb,
                read_op_min_num_blocks=read_op_min_num_blocks,
            )
        )

        start_time = time.time()

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        shutdown = GracefulShutdown()

        # Warn if batch_timeout was passed
        if batch_timeout != DEFAULT_RECOGNITION_BATCH_TIMEOUT:
            import warnings

            warnings.warn(
                "`batch_timeout` is deprecated and ignored; Ray Data's execution-level "
                "no-progress timeout now guards against hangs.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Resolve slot CPU reservation
        _slot_cpus, ray_remote_args, resolved_cpus = _resolve_slot_cpus(
            num_cpus=num_cpus,
            worker_num_cpus=worker_num_cpus,
            stage_name="run_recognition",
        )

        # Resolve input files
        input_files = resolve_input_files(input_path)
        if not input_files:
            raise FileNotFoundError(f"No files found matching: {input_path}")

        logger.info(f"Found {len(input_files)} input file(s)")

        # Auto-detect actors
        if num_actors is None:
            num_actors = self._auto_num_actors()

        # Detect columns
        required_cols = ["text_hash", "note_text"]
        optional_cols = [
            "patient_identifiers",
            "recognizer_results_json",
            "patient_id",
            "patient_uid",
            "jitter",
            "row_id",
        ]
        columns = detect_columns(input_files[0], required_cols, optional_cols)

        ctx = ray.data.DataContext.get_current()
        logger.info("Recognition job starting")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(
            f"  Actors: {num_actors}, Batch size: {batch_size}, "
            f"Slot CPUs: {_slot_cpus} (from num_cpus={num_cpus}, worker_num_cpus={worker_num_cpus}), "
            f"no_progress_timeout_s={getattr(ctx, 'execution_no_progress_timeout_s', None)}"
        )
        logger.info(f"  Resolved CPU config: {resolved_cpus}")

        try:
            logger.info(f"Processing {len(input_files)} files in single streaming pipeline")

            # Dry-run: validate setup and show plan without processing
            if dry_run:
                logger.info("DRY RUN - validation complete, no processing performed")
                return {
                    "dry_run": True,
                    "input_path": input_path,
                    "output_path": output_path,
                    "num_files": len(input_files),
                    "num_actors": num_actors,
                    "batch_size": batch_size,
                    "columns_detected": columns,
                }

            # Configure Ray Data checkpointing for row-level resume. See
            # _configure_checkpoint for why enable_checkpoint=False is REQUIRED on
            # tiny clusters (≲4 CPUs, e.g. 2-CPU Colab) and why disabling
            # op_resource_reservation_enabled does NOT help.
            _configure_checkpoint(ctx, enable=enable_checkpoint, output_dir=output_dir, id_column="text_hash")

            # Single streaming pipeline — no repartition, no segment loop.
            num_blocks = read_parallelism if read_parallelism is not None else len(input_files)
            # Ensure enough blocks to utilize all actors
            num_blocks = max(num_blocks, num_actors)
            if override_num_blocks is not None:
                num_blocks = override_num_blocks
            ds = ray.data.read_parquet(
                input_files,
                columns=columns,
                override_num_blocks=num_blocks,
                ray_remote_args={"num_cpus": read_cpus},
            )
            processed = self.build_recognizer_stage(
                ds,
                batch_size=batch_size,
                num_actors=num_actors,
                ray_remote_args=ray_remote_args,
            )
            processed.write_parquet(str(output_dir), compression="zstd", ray_remote_args={"num_cpus": write_cpus})

            # Clear checkpoint config to avoid leaking to subsequent pipelines
            ctx.checkpoint_config = None

            processing_time = time.time() - start_time
            logger.info(f"Recognition complete in {processing_time:.2f}s")

            return {
                "processing_time_seconds": processing_time,
                "num_files": len(input_files),
                "num_actors": num_actors,
                "batch_size": batch_size,
            }

        except ray.data.exceptions.ExecutionTimeoutError:
            _log_execution_timeout("Recognition", ctx)
            raise
        except Exception:
            logger.exception("Recognition failed")
            raise

        finally:
            shutdown.restore_handlers()

    def run_llm_recognition(
        self,
        input_path: str,
        output_path: str,
        project_id: str,
        model_name: str = "gemini-2.5-flash",
        prompt_name: str = "phi_detection",
        provider_type: str = "google",
        context_length: int = 1_048_576,
        max_tokens: int = 16384,
        temperature: float = 0.0,
        region: str = "us-central1",
        endpoint_id: int | None = None,
        max_retries: int = 3,
        num_actors: int | None = None,
        batch_size: int = 10,
        batch_timeout: int = 300,
        num_cpus: int | float = 1,
        worker_num_cpus: int | float | None = None,
        read_parallelism: int | None = None,
        read_cpus: float = 0.25,
        read_op_min_num_blocks: int = 200,
        target_max_block_size_mb: int = 128,
        target_min_block_size_mb: int = 1,
        write_cpus: float = 1.0,
        enable_checkpoint: bool = True,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Run LLM-based recognition job with Ray Data checkpointing for resume.

        Uses LlmRecognizerActor (LLM-based entity detection) instead of the
        regex/rule-based RecognizerActor. Each actor makes LLM API calls to
        detect PHI entities in clinical text.

        Args:
            input_path: Input parquet files (local path or GCS URI).
                Required columns: text_hash, note_text.
            output_path: Output directory
            project_id: GCP project ID for LLM API access
            model_name: LLM model name (default: "gemini-2.5-flash")
            prompt_name: Name of the prompt config in resources/llm_prompts/ (default: "phi_detection")
            provider_type: LLM provider type (default: "google")
            context_length: Model context window in tokens (default: 1_048_576)
            max_tokens: Maximum tokens for LLM output (default: 16384)
            temperature: Model temperature for response generation (default: 0.0)
            region: Cloud region for the LLM API (default: "us-central1")
            endpoint_id: Optional Vertex AI endpoint ID
            max_retries: Maximum retry attempts for failed LLM requests (default: 3)
            num_actors: Actor count (auto-detect if None)
            batch_size: Batch size per actor (default: 10, lower than regex recognizer
                because each note requires an LLM API call)
            batch_timeout: Seconds before a batch is killed (default: 300)
            num_cpus: CPUs per supervisor actor
            worker_num_cpus: CPUs per worker actor (None = Ray default of 1).
                Set to 0 along with num_cpus=0 for I/O-bound oversubscription.
            read_parallelism: Number of read output blocks (default: num input files)
            read_cpus: CPUs per read task (lower = more concurrent reads)
            read_op_min_num_blocks: Minimum read output blocks for DataContext
            target_max_block_size_mb: Max block size in MB for DataContext
            target_min_block_size_mb: Min block size in MB for DataContext
            write_cpus: CPUs to reserve for each write_parquet task. Default 1.0
                reproduces Ray's default task reservation. Lower (e.g. 0.25) to
                fit small boxes where concurrent operators contend for CPUs.
            enable_checkpoint: If True (default), enable Ray Data row-level
                checkpointing for resume. Set False on tiny clusters (≲4 CPUs):
                the checkpoint sort+repartition shuffle deadlocks Ray 2.55's
                reservation allocator. See run_recognition for details.
            dry_run: If True, validate setup and show plan without processing

        Returns:
            Processing statistics dictionary
        """
        self._init_ray()

        # Override DataContext with job-specific streaming params
        configure_data_context(
            **self._data_context_kwargs(
                verbose_progress=True,
                target_max_block_size_mb=target_max_block_size_mb,
                target_min_block_size_mb=target_min_block_size_mb,
                read_op_min_num_blocks=read_op_min_num_blocks,
            )
        )

        start_time = time.time()

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        shutdown = GracefulShutdown()

        # Warn if batch_timeout was passed
        if batch_timeout != DEFAULT_LLM_BATCH_TIMEOUT:
            import warnings

            warnings.warn(
                "`batch_timeout` is deprecated and ignored; Ray Data's execution-level "
                "no-progress timeout now guards against hangs.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Resolve slot CPU reservation
        _slot_cpus, ray_remote_args, resolved_cpus = _resolve_slot_cpus(
            num_cpus=num_cpus,
            worker_num_cpus=worker_num_cpus,
            stage_name="run_llm_recognition",
        )

        # Resolve input files
        input_files = resolve_input_files(input_path)
        if not input_files:
            raise FileNotFoundError(f"No files found matching: {input_path}")

        logger.info(f"Found {len(input_files)} input file(s)")

        # Auto-detect actors
        if num_actors is None:
            num_actors = self._auto_num_actors()

        # Detect columns — LLM recognizer only needs text_hash and note_text
        required_cols = ["text_hash", "note_text"]
        columns = detect_columns(input_files[0], required_cols, [])

        ctx = ray.data.DataContext.get_current()
        logger.info("LLM Recognition job starting")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Model: {model_name} (provider: {provider_type}, prompt: {prompt_name})")
        logger.info(
            f"  Actors: {num_actors}, Batch size: {batch_size}, "
            f"Slot CPUs: {_slot_cpus} (from num_cpus={num_cpus}, worker_num_cpus={worker_num_cpus}), "
            f"no_progress_timeout_s={getattr(ctx, 'execution_no_progress_timeout_s', None)}"
        )
        logger.info(f"  Resolved CPU config: {resolved_cpus}")

        try:
            logger.info(f"Processing {len(input_files)} files in single streaming pipeline")

            # Dry-run: validate setup and show plan without processing
            if dry_run:
                logger.info("DRY RUN - validation complete, no processing performed")
                return {
                    "dry_run": True,
                    "input_path": input_path,
                    "output_path": output_path,
                    "num_files": len(input_files),
                    "num_actors": num_actors,
                    "batch_size": batch_size,
                    "columns_detected": columns,
                    "model_name": model_name,
                    "prompt_name": prompt_name,
                    "provider_type": provider_type,
                }

            # Configure Ray Data checkpointing for row-level resume. See
            # _configure_checkpoint for why this must be disabled on tiny clusters
            # (the checkpoint shuffle deadlocks Ray 2.55's reservation allocator, and
            # disabling op_resource_reservation_enabled does NOT help).
            _configure_checkpoint(ctx, enable=enable_checkpoint, output_dir=output_dir, id_column="text_hash")

            # Single streaming pipeline
            # Default to num_actors blocks so data is distributed across all actors
            num_blocks = read_parallelism if read_parallelism is not None else max(num_actors, len(input_files))
            ds = ray.data.read_parquet(
                input_files,
                columns=columns,
                override_num_blocks=num_blocks,
                ray_remote_args={"num_cpus": read_cpus},
            )
            processed = self.build_llm_recognizer_stage(
                ds,
                batch_size=batch_size,
                num_actors=num_actors,
                ray_remote_args=ray_remote_args,
                fn_constructor_kwargs={
                    "project_id": project_id,
                    "provider_type": provider_type,
                    "model_name": model_name,
                    "prompt_name": prompt_name,
                    "context_length": context_length,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "region": region,
                    "endpoint_id": endpoint_id,
                    "max_retries": max_retries,
                },
            )
            processed.write_parquet(str(output_dir), compression="zstd", ray_remote_args={"num_cpus": write_cpus})

            # Clear checkpoint config to avoid leaking to subsequent pipelines
            ctx.checkpoint_config = None

            processing_time = time.time() - start_time
            logger.info(f"LLM Recognition complete in {processing_time:.2f}s")

            return {
                "processing_time_seconds": processing_time,
                "num_files": len(input_files),
                "num_actors": num_actors,
                "batch_size": batch_size,
                "model_name": model_name,
                "prompt_name": prompt_name,
                "provider_type": provider_type,
            }

        except ray.data.exceptions.ExecutionTimeoutError:
            _log_execution_timeout("LLM Recognition", ctx)
            raise
        except Exception:
            logger.exception("LLM Recognition failed")
            raise

        finally:
            shutdown.restore_handlers()

    def run_anonymization(  # noqa: PLR0915
        self,
        input_path: str | list[str],
        output_path: str,
        salt_path: str,
        key_path: str,
        num_actors: int | None = None,
        batch_size: int = 200,
        num_cpus: int | float = 2,
        read_parallelism: int | None = None,
        read_cpus: float = 0.25,
        read_op_min_num_blocks: int = 200,
        target_max_block_size_mb: int = 128,
        target_min_block_size_mb: int = 1,
        acc_num_salt: str | None = None,
        acc_num_study_id: str | None = None,
        jitter_required: bool = False,
        worker_num_cpus: int | float | None = None,
        write_cpus: float = 1.0,
        enable_checkpoint: bool = True,
        override_num_blocks: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Run anonymization job with Ray Data checkpointing for resume.

        Args:
            input_path: Input parquet files with recognizer results
            output_path: Output directory
            salt_path: Path to FPE salt file
            key_path: Path to FPE key file
            num_actors: Actor count (auto-detect if None)
            batch_size: Batch size per actor
            num_cpus: CPUs per actor (affects streaming executor scheduling)
            read_parallelism: Number of read output blocks (default: num input files)
            read_cpus: CPUs per read task (lower = more concurrent reads)
            read_op_min_num_blocks: Minimum read output blocks for DataContext
            target_max_block_size_mb: Max block size in MB for DataContext
            target_min_block_size_mb: Min block size in MB for DataContext
            acc_num_salt: Salt for accession number hashing
            acc_num_study_id: Study ID for accession number hashing
            jitter_required: If True, notes without a jitter value fail instead
                of computing one automatically
            worker_num_cpus: CPUs to reserve for each supervisor's worker actor.
                None = Ray default (1). Each pool slot needs supervisor
                (num_cpus) + worker (worker_num_cpus) CPUs; lower both to fit
                small boxes.
            write_cpus: CPUs to reserve for each write_parquet task. Default 1.0
                reproduces Ray's default task reservation. Note the follow-up
                zero-row guard ``processed.count()`` re-executes the read +
                map_batches plan (not the write), so it is unaffected by this knob.
            enable_checkpoint: If True (default), enable Ray Data row-level
                checkpointing for resume. MUST be set to False on tiny clusters
                (≲4 CPUs, e.g. Google Colab): the checkpoint pipeline adds a
                sort+repartition shuffle whose per-operator CPU reservations
                exceed the cluster, deadlocking the stage at 0/1. Disabling it
                trades resume capability (not correctness) for the ability to run.
            override_num_blocks: Explicit number of Ray Data blocks to split the
                input into (e.g. 32 to fix single-block starvation on multicore nodes).
            dry_run: If True, validate setup and show plan without processing

        Returns:
            Processing statistics dictionary, including ``output_rows`` (the number
            of rows written) so a successful run is observable.

        Raises:
            RuntimeError: If a non-empty input produces zero output rows. Ray's
                ``max_errored_blocks`` and the supervisor's ``_failed_batch``
                fallback can otherwise turn a total failure into a successful-looking
                0-row write; this guard surfaces it as a hard error instead.
        """
        from tide2.actors import create_anonymizer_actor_class

        self._init_ray()

        # Override DataContext with job-specific streaming params
        configure_data_context(
            **self._data_context_kwargs(
                verbose_progress=True,
                target_max_block_size_mb=target_max_block_size_mb,
                target_min_block_size_mb=target_min_block_size_mb,
                read_op_min_num_blocks=read_op_min_num_blocks,
            )
        )

        start_time = time.time()

        # Load keys
        salt = self._load_key(salt_path)
        key = self._load_key(key_path)

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        shutdown = GracefulShutdown()

        # Resolve slot CPU reservation
        _slot_cpus, ray_remote_args, resolved_cpus = _resolve_slot_cpus(
            num_cpus=num_cpus,
            worker_num_cpus=worker_num_cpus,
            stage_name="run_anonymization",
        )

        # Resolve input files
        input_files = resolve_input_files(input_path)
        if not input_files:
            raise FileNotFoundError(f"No files found matching: {input_path}")

        logger.info(f"Found {len(input_files)} input file(s)")

        if num_actors is None:
            num_actors = self._auto_num_actors()

        # Detect columns
        required_cols = ["text_hash", "note_text", "recognizer_results_json"]
        optional_cols = ["patient_uid", "patient_id", "jitter", "row_id"]
        columns = detect_columns(input_files[0], required_cols, optional_cols)

        ctx = ray.data.DataContext.get_current()
        logger.info("Anonymization job starting")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(
            f"  Actors: {num_actors}, Batch size: {batch_size}, "
            f"Slot CPUs: {_slot_cpus} (from num_cpus={num_cpus}, worker_num_cpus={worker_num_cpus}), "
            f"no_progress_timeout_s={getattr(ctx, 'execution_no_progress_timeout_s', None)}"
        )
        logger.info(f"  Resolved CPU config: {resolved_cpus}")

        # Create actor class with keys
        AnonymizerActor = create_anonymizer_actor_class(  # noqa: N806 # its a type
            salt=salt,
            key=key,
            acc_num_salt=acc_num_salt,
            acc_num_study_id=acc_num_study_id,
            jitter_required=jitter_required,
        )

        try:
            logger.info(f"Processing {len(input_files)} files in single streaming pipeline")

            # Dry-run: validate setup and show plan without processing
            if dry_run:
                logger.info("DRY RUN - validation complete, no processing performed")
                return {
                    "dry_run": True,
                    "input_path": input_path,
                    "output_path": output_path,
                    "num_files": len(input_files),
                    "num_actors": num_actors,
                    "batch_size": batch_size,
                    "columns_detected": columns,
                    "keys_loaded": True,
                }

            # Configure Ray Data checkpointing for row-level resume. See
            # _configure_checkpoint for why this must be disabled on tiny clusters
            # (the checkpoint shuffle deadlocks Ray 2.55's reservation allocator, and
            # disabling op_resource_reservation_enabled does NOT help).
            id_col = "row_id" if "row_id" in columns else "text_hash"
            _configure_checkpoint(ctx, enable=enable_checkpoint, output_dir=output_dir, id_column=id_col)

            # Single streaming pipeline — no repartition, no segment loop.
            num_blocks = read_parallelism if read_parallelism is not None else len(input_files)
            # Ensure enough blocks to utilize all actors
            num_blocks = max(num_blocks, num_actors)
            if override_num_blocks is not None:
                num_blocks = override_num_blocks
            ds = ray.data.read_parquet(
                input_files,
                columns=columns,
                override_num_blocks=num_blocks,
                ray_remote_args={"num_cpus": read_cpus},
            )
            processed = self.build_anonymizer_stage(
                ds,
                actor_cls=AnonymizerActor,
                batch_size=batch_size,
                num_actors=num_actors,
                ray_remote_args=ray_remote_args,
            )
            processed.write_parquet(str(output_dir), compression="zstd", ray_remote_args={"num_cpus": write_cpus})

            # Guard against silent total failure: Ray's max_errored_blocks can turn
            # every dropped batch into a successful-looking 0-row write.
            # Surface that as a hard error instead.
            import pyarrow.dataset as _pad

            try:
                output_files = list(output_dir.glob("*.parquet"))
                if output_files:
                    output_rows = _pad.dataset(output_files).count_rows()
                elif hasattr(processed, "count"):
                    output_rows = processed.count()
                else:
                    output_rows = 0
            except Exception:
                output_rows = processed.count() if hasattr(processed, "count") else 0

            if output_rows == 0 and len(input_files) > 0:
                raise RuntimeError(
                    "Anonymizer wrote 0 rows from non-empty input — all batches failed. "
                    "Check worker logs for the underlying error."
                )

            # Clear checkpoint config to avoid leaking to subsequent pipelines
            ctx.checkpoint_config = None

            processing_time = time.time() - start_time
            logger.info(f"Anonymization complete in {processing_time:.2f}s")

            return {
                "processing_time_seconds": processing_time,
                "num_files": len(input_files),
                "num_actors": num_actors,
                "batch_size": batch_size,
                "output_rows": output_rows,
            }

        except ray.data.exceptions.ExecutionTimeoutError:
            _log_execution_timeout("Anonymization", ctx)
            raise
        except Exception:
            logger.exception("Anonymization failed")
            raise

        finally:
            shutdown.restore_handlers()

    def run_transformer(  # noqa: PLR0915
        self,
        input_path: str | list[str],
        output_path: str,
        model_name: str,
        model_path: str | None = None,
        bucket_name: str | None = None,
        project_id: str | None = None,
        num_gpus: int | None = None,
        num_transformer_actors: int | None = None,
        batch_size: int = 8,
        gpu_batch_size: int | None = None,
        chunk_overlap: int | None = None,
        num_agg_actors: int | None = None,
        read_cpus: float = 1.0,
        write_cpus: float = 1.0,
        agg_num_cpus: float = 1.0,
        transformer_cpus: float | None = None,
        enable_checkpoint: bool = True,
        override_num_blocks: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run transformer NER job with token-accurate windowing.

        Whole notes flow to the GPU actor, which tokenizes each note once and
        token-windows it against the model's real context window before inference;
        the downstream aggregation actor produces document-level entities directly.
        There is no separate char-chunking stage or reassembly stage.

        Hardware sizing (CPU/actor knobs)
        ---------------------------------
        Ray Data runs every operator of this stage concurrently
        (read -> transformer actor -> aggregation actor -> write) and, under
        Ray 2.55's ReservationOpResourceAllocator, must reserve a minimum CPU
        slice for each *eligible* operator at once. If those minimums sum to more
        than the cluster's CPUs, NOTHING schedules and the stage hangs at 0/1
        (``backpressured:tasks(ResourceBudget)``). Two independent levers avoid
        this on small boxes:

        1. **Fractional CPU knobs** (``read_cpus``, ``write_cpus``,
           ``agg_num_cpus``, ``transformer_cpus``) shrink each operator's
           reservation so the concurrent sum fits.
        2. **``enable_checkpoint=False``** removes the checkpoint shuffle
           (sort + repartition), which otherwise adds several more eligible
           operators and re-triggers the deadlock *even with* fractional CPUs.
           Both levers are required together on ≲4-CPU boxes.

        Note: ``ctx.op_resource_reservation_enabled = False`` does NOT resolve this
        — the per-operator reservation floors still exceed a ≲4-CPU cluster. The
        fractional-CPU knobs + ``enable_checkpoint=False`` are the validated fix.

        Recommended settings by hardware (C = total CPUs, G = total GPUs):

        - **Big VM (C≳16), GPU or CPU**: use defaults (all knobs 1.0,
          ``transformer_cpus=None``, ``enable_checkpoint=True``). The library
          auto-scales actor counts; reservations fit comfortably.
        - **Small GPU box (e.g. C=2, G=1 — Colab T4)**: the transformer actor is
          GPU-pinned (0 CPU), so budget the CPU operators fractionally:
          ``read_cpus=write_cpus=0.25``, ``agg_num_cpus=0.5``,
          ``transformer_cpus=0.25`` (small optional floor),
          ``num_transformer_actors=1``, ``num_agg_actors=1``,
          ``enable_checkpoint=False``.
        - **Small CPU box (e.g. C=2, G=0 — Colab CPU)**: the actor needs ~1 CPU
          and ``transformer_cpus`` also caps its torch thread count
          (``transformer.py``: ``int(transformer_cpus) or 1``), so give it most
          of the box: ``transformer_cpus=max(0.5, C-1.0)``,
          ``read_cpus=write_cpus=0.25``, ``agg_num_cpus=0.25``,
          ``num_transformer_actors=1``, ``num_agg_actors=1``,
          ``enable_checkpoint=False``. Expect SLOW single-threaded inference —
          this restores correctness, not speed.

        Args:
            input_path: Input parquet files
            output_path: Output directory
            model_name: Name of transformer model configuration
            model_path: Optional explicit model path
            bucket_name: Optional GCS bucket for model loading
            project_id: Optional GCP project ID
            num_gpus: Number of GPUs (or fractional GPUs, e.g. 0.33) per actor.
            num_transformer_actors: Number of transformer inference actors.
                If None, defaults to scaling with available GPUs, or ~25% of
                available CPUs in CPU-only mode.
            batch_size: Batch size for map_batches (whole notes per actor call).
                This is the host-memory knob: a batch of long notes tokenizes to
                many ragged ids at once, so keep it modest. GPU memory stays
                bounded by ``gpu_batch_size``, not this.
            gpu_batch_size: Number of token windows per GPU forward. Size it for
                the load; if a batch OOMs, the actor halves and retries. None = a
                nominal default that only needs to run out of the box.
            chunk_overlap: Token overlap between adjacent windows when a note
                exceeds the model's per-window token budget (default: from model
                config's ``CHUNK_OVERLAP_SIZE``).
            num_agg_actors: Number of CPU actors for BIO aggregation.
                If 0, aggregates within the transformer actor directly, emitting
                document-ready entities without a separate aggregation actor pool.
                If None, auto-computed (0 in fractional-GPU mode; ~30% CPUs otherwise).
            read_cpus: CPUs to reserve for each read_parquet task. Default 1.0
                reproduces Ray's default reservation. Lower (e.g. 0.25) to fit
                small boxes where concurrent operators contend for CPUs.
            write_cpus: CPUs to reserve for each write_parquet task. Default 1.0
                reproduces Ray's default reservation.
            agg_num_cpus: CPUs to reserve for each BIO aggregation actor.
                Default 1.0 reproduces Ray's default actor reservation.
            transformer_cpus: CPU floor for the transformer actor. None leaves
                the Ray default (0 CPU in GPU mode, since the actor is GPU-pinned;
                1 CPU in CPU mode). In CPU mode this also caps torch threads, so
                set it to ~(total CPUs - 1) on small CPU boxes.
            enable_checkpoint: If True (default), enable Ray Data row-level
                checkpointing for resume. MUST be set to False on tiny clusters
                (≲4 CPUs): the checkpoint sort+repartition shuffle deadlocks the
                stage regardless of the fractional CPU knobs above (see the
                "Hardware sizing" section). Disabling it loses resume capability,
                not correctness.
            override_num_blocks: Explicit number of Ray Data blocks to split the
                input into.

        Returns:
            Processing statistics dictionary
        """
        _check_deprecated_transformer_kwargs(kwargs, "LocalJobRunner.run_transformer")
        from tide2.actors import create_transformer_actor
        from tide2.transformers.config import load_model_config

        self._init_ray()
        start_time = time.time()

        # Resolve the window overlap from the model config if not provided. This is
        # the token overlap between adjacent windows of an over-budget note; the
        # per-window token budget itself is the model's real context window,
        # resolved inside the actor from MODEL_MAX_LENGTH (no CHUNK_SIZE here).
        model_config = load_model_config(model_name)

        # Fail fast on the driver if the single length authority is missing/invalid.
        # MODEL_MAX_LENGTH is a hard requirement (the transformer actor resolves the
        # per-window token budget from it); validating here surfaces a clear
        # driver-side error instead of a harder-to-diagnose failure deep inside Ray
        # worker actor init.
        _mml = model_config.get("MODEL_MAX_LENGTH")
        if not isinstance(_mml, int) or _mml <= 0:
            raise ValueError(
                f"Model {model_name!r} config is missing a valid 'MODEL_MAX_LENGTH' "
                f"(got {_mml!r}). It must be a positive integer equal to the model's real "
                f"tokenized context window (e.g. 512 for BERT/RoBERTa, 8192 for ModernBERT)."
            )

        if chunk_overlap is None:
            chunk_overlap = model_config.get("CHUNK_OVERLAP_SIZE", 40)

        num_gpus, cpu_only_mode, num_transformer_actors, num_agg_actors = self._resolve_transformer_resources(
            num_gpus, num_transformer_actors, num_agg_actors
        )

        logger.info("Transformer NER job starting")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Model: {model_name}")
        if cpu_only_mode:
            logger.info(f"  Device: CPU (no GPUs), Actors: {num_transformer_actors}, Batch size: {batch_size}")
        else:
            logger.info(f"  GPUs: {num_gpus}, Batch size: {batch_size}, GPU batch size: {gpu_batch_size}")
        logger.info(f"  BIO aggregation actors: {num_agg_actors}")
        logger.info(f"  Window overlap: {chunk_overlap} tokens")

        # Resolve model path on driver (downloads if needed, validates weights)
        if model_path is None:
            from tide2.utils.gcs_resource_manager import resolve_model_path

            model_path = resolve_model_path(
                model_name=model_name,
                bucket_name=bucket_name,
                project_id=project_id,
            )
            logger.info(f"Model resolved on driver: {model_path}")

        # Create transformer actor class. The actor tokenizes + token-windows whole
        # notes against the model's real budget; chunk_overlap is the window overlap.
        aggregate_in_actor = num_agg_actors == 0
        transformer_actor = create_transformer_actor(
            model_name=model_name,
            model_path=model_path,
            bucket_name=bucket_name,
            project_id=project_id,
            gpu_batch_size=gpu_batch_size,
            window_overlap=chunk_overlap,
            aggregate_bio=aggregate_in_actor,
        )

        input_files = resolve_input_files(input_path)
        required_cols = ["text_hash", "note_text"]
        optional_cols = ["patient_id", "patient_identifiers", "patient_uid", "jitter", "row_id"]
        if input_files:
            try:
                columns = detect_columns(input_files[0], required_cols, optional_cols)
            except (FileNotFoundError, OSError):
                columns = ["text_hash", "note_text", "patient_id"]
            read_target = input_files
        else:
            columns = ["text_hash", "note_text", "patient_id"]
            read_target = self._resolve_input_pattern(input_path)

        self._ensure_output_dir(output_path)

        # Configure Ray Data checkpointing for row-level resume, keyed on text_hash
        # (one row per note through the whole stage now — no chunk_uid).
        ctx = ray.data.DataContext.get_current()
        _configure_checkpoint(
            ctx, enable=enable_checkpoint, output_dir=Path(output_path).resolve(), id_column="text_hash"
        )

        read_kwargs: dict[str, Any] = {
            "columns": columns,
            "ray_remote_args": {"num_cpus": read_cpus},
        }
        if override_num_blocks is not None:
            read_kwargs["override_num_blocks"] = override_num_blocks

        # Phase 1: Read whole notes (the actor's token-windowing is the sole chunker)
        ds: Dataset = ray.data.read_parquet(
            read_target,
            **read_kwargs,
        )

        # Phase 2: Transformer inference (tokenize -> window -> forward, per note).
        # When aggregate_in_actor is True, also aggregates BIO tokens directly.
        ray_remote_args_transformer = get_ray_remote_args_gpu(num_gpus=0 if cpu_only_mode else num_gpus)
        if transformer_cpus is not None:
            # Set a CPU floor for the transformer actor. Never add num_gpus here
            # in CPU mode; GPU pinning (num_gpus=1 or fractional) is preserved in GPU mode.
            ray_remote_args_transformer["num_cpus"] = transformer_cpus

        # Phase 3 (unless aggregate_in_actor): CPU aggregation -> dedup -> Presidio
        # format, producing document-ready recognizer_results_json (runs concurrently
        # with the GPU via streaming). Folds in the old separate reassembly stage.
        ds_predictions = self.build_transformer_stage(
            ds,
            transformer_actor=transformer_actor,
            model_name=model_name,
            batch_size=batch_size,
            num_transformer_actors=num_transformer_actors,
            ray_remote_args_transformer=ray_remote_args_transformer,
            num_agg_actors=num_agg_actors,
            agg_num_cpus=agg_num_cpus,
        )

        # Phase 4: Write document-level recognizer results (fully streaming, no groupby)
        ds_predictions.write_parquet(output_path, compression="zstd", ray_remote_args={"num_cpus": write_cpus})

        # Clear checkpoint config to avoid leaking to subsequent pipelines
        ctx.checkpoint_config = None

        elapsed = time.time() - start_time
        logger.info(f"Transformer NER complete in {elapsed:.1f}s")

        return {
            "elapsed_seconds": round(elapsed, 2),
            "model_name": model_name,
            "chunk_overlap": chunk_overlap,
            "num_gpus": num_gpus,
            "cpu_only_mode": cpu_only_mode,
            "num_transformer_actors": num_transformer_actors,
            "num_agg_actors": num_agg_actors,
            "batch_size": batch_size,
        }

    def run_pipeline(  # noqa: PLR0915 # its a long function but its the main pipeline runner
        self,
        input_data: str | pd.DataFrame,
        output_dir: str,
        model_name: str,
        *,
        run_transformer: bool = True,
        run_recognizer: bool = True,
        run_anonymizer: bool = True,
        produce_visualizer_json: bool = False,
        salt_hex: str = "00" * 32,
        key_hex: str = "11" * 32,
        transformer_kwargs: dict[str, Any] | None = None,
        recognizer_kwargs: dict[str, Any] | None = None,
        anonymizer_kwargs: dict[str, Any] | None = None,
        llm_recognizer_mode: str = "off",
        llm_recognizer_kwargs: dict[str, Any] | None = None,
        hardware_autotune: bool = True,
        execution_mode: Literal["discrete", "streamed"] = "discrete",
    ) -> dict[str, Any]:
        """
        Run the full de-identification pipeline (transformer → recognizer → anonymizer).

        Designed for small datasets. Each stage can be toggled independently.
        When a stage is skipped, its output from a previous run is expected on disk.

        Execution modes
        ---------------
        ``execution_mode="discrete"`` (default) runs each stage as its own Ray
        Data execution, writing its output to Parquet before the next stage
        reads it. That per-stage boundary is what lets the GPU stage run on one
        machine and the CPU stages elsewhere as independent jobs, so **discrete
        is the mode for production and for anything multi-machine**, and it is
        the only mode supported on ≲4-CPU boxes.

        ``execution_mode="streamed"`` chains the stages into a single Ray Data
        execution: blocks cross the object store instead of Parquet, and the GPU
        stage overlaps the CPU stages. It is a **single-cluster optimization**
        for development, benchmarks, and single-box batches. Trade-offs:

        - **No row-level resume.** Row-level checkpointing is keyed to one
          ``id_column`` and one sink, which a chained plan does not have, and
          its sort+repartition re-triggers the small-box deadlock. A mid-run
          failure re-runs GPU inference. Long-running or unattended jobs should
          stay on discrete.
        - Only ``06_anonymizer_output`` is written; the ``01_``/``02_``/``04_``
          intermediates are not.
        - ``note_text`` stays resident in the object store across all three
          operators and may spill (in plaintext) to Ray's local spill directory.
          Size ``object_store_gb`` and host disk accordingly, and dispose of the
          host's storage under the same rules as the output directory.
        - The return dict has a different shape — discriminate on the
          ``execution_mode`` key, which both modes now set.

        Streamed **falls back to discrete, with a warning and a corrected
        ``execution_mode`` in the returned manifest**, when ``enable_checkpoint``
        is explicitly requested, when ``llm_recognizer_mode="merge"``, when
        ``produce_visualizer_json=True``, or when there is nothing to chain. It
        **raises** on multi-node clusters and on nodes with ≤4 CPUs, where a
        chained plan cannot be scheduled (see ``check_streamed_admission``).

        Args:
            input_data: Path to input parquet file, or a DataFrame.
                Required column: note_text.
                Optional columns: text_hash, patient_identifiers (JSON string),
                patient_id, recognizer_results_json, jitter.
            output_dir: Output directory for all intermediate and final files.
            model_name: Transformer model name (e.g. "StanfordAIMI/stanford-deidentifier-base").
            run_transformer: Run GPU transformer NER stage.
            run_recognizer: Run CPU recognizer stage.
            run_anonymizer: Run CPU anonymizer stage.
            produce_visualizer_json: Write JSON files for tide2-visualizer.
            salt_hex: Hex-encoded 32-byte FPE salt.
            key_hex: Hex-encoded 32-byte FPE key.
            transformer_kwargs: Extra kwargs passed to self.run_transformer().
                e.g. bucket_name, project_id, num_gpus, batch_size,
                gpu_batch_size, chunk_overlap.
            recognizer_kwargs: Extra kwargs passed to self.run_recognition().
                e.g. num_actors, batch_size, batch_timeout, num_cpus.
            anonymizer_kwargs: Extra kwargs passed to self.run_anonymization().
                e.g. num_actors, batch_size, acc_num_salt, acc_num_study_id.
            llm_recognizer_mode: LLM recognizer mode. One of:
                - "off" (default): No LLM recognition.
                - "only": LLM replaces transformer + regex recognizer entirely.
                    Pipeline: LLM recognizer → anonymizer.
                - "merge": LLM runs alongside existing recognizer, results are
                    merged per note using resolve_recognizer_results().
            llm_recognizer_kwargs: Extra kwargs passed to self.run_llm_recognition().
                e.g. project_id, model_name, provider_type, context_length,
                max_tokens, num_actors, batch_size.
            hardware_autotune: Enable hardware autotuning of per-stage settings.
                When False, today's defaults run and recommendations are not applied.
            execution_mode: "discrete" (default) or "streamed" — see above.

        Returns:
            Dictionary with per-stage statistics and output paths in discrete
            mode; in streamed mode, ``{"execution_mode", "total_elapsed_seconds",
            "input_rows", "output_rows", "dropped_rows", "output_dir",
            "operator_stats"}``. Both carry ``execution_mode``.
        """
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # --- Intermediate paths ---
        transformer_input_path = output_path / "01_transformer_input.parquet"
        transformer_output_path = output_path / "02_transformer_output"
        recognizer_output_path = output_path / "04_recognizer_output"
        anonymizer_output_path = output_path / "06_anonymizer_output"

        llm_recognizer_output_path = output_path / "03b_llm_recognizer_output"

        # Validate llm_recognizer_mode
        valid_llm_modes = ("off", "only", "merge")
        if llm_recognizer_mode not in valid_llm_modes:
            raise ValueError(f"llm_recognizer_mode must be one of {valid_llm_modes}, got '{llm_recognizer_mode}'")

        results: dict[str, Any] = {"output_dir": str(output_path)}
        # Copied, not aliased: apply_recommendations fills these in place, and a
        # caller's dict must not come back carrying this run's resolved values (a
        # second call would then read them as explicitly-supplied USER settings).
        t_kw = dict(transformer_kwargs or {})
        r_kw = dict(recognizer_kwargs or {})
        a_kw = dict(anonymizer_kwargs or {})
        llm_kw = llm_recognizer_kwargs or {}

        # ------------------------------------------------------------------
        # Prepare input DataFrame
        # ------------------------------------------------------------------
        df_input = self._prepare_pipeline_input(input_data)

        # ------------------------------------------------------------------
        # Execution mode: resolve fallbacks before anything is written
        # ------------------------------------------------------------------
        if execution_mode not in ("discrete", "streamed"):
            raise ValueError(f"execution_mode must be 'discrete' or 'streamed', got {execution_mode!r}")

        if execution_mode == "streamed":
            reason = self._streamed_fallback_reason(
                llm_recognizer_mode=llm_recognizer_mode,
                produce_visualizer_json=produce_visualizer_json,
                stage_kwargs=(t_kw, r_kw, a_kw, dict(llm_kw)),
                run_anonymizer=run_anonymizer,
                run_transformer=run_transformer,
                run_recognizer=run_recognizer,
            )
            if reason is not None:
                logger.warning(
                    "%s Falling back to discrete execution: running the three stages sequentially, each "
                    "serialized to Parquet before the next begins. This is slower than streamed mode "
                    "(expect roughly the discrete baseline) because stage output is written to and re-read "
                    "from disk instead of staying in memory. Pass enable_checkpoint=False (and drop the "
                    "other listed options) for streamed execution.",
                    reason,
                )
                execution_mode = "discrete"

        if execution_mode == "streamed":
            return self._run_pipeline_streamed(
                df_input=df_input,
                output_path=output_path,
                model_name=model_name,
                run_transformer=run_transformer,
                run_recognizer=run_recognizer,
                run_anonymizer=run_anonymizer,
                salt_hex=salt_hex,
                key_hex=key_hex,
                transformer_kwargs=t_kw,
                recognizer_kwargs=r_kw,
                anonymizer_kwargs=a_kw,
                llm_recognizer_mode=llm_recognizer_mode,
                llm_recognizer_kwargs=dict(llm_kw),
                hardware_autotune=hardware_autotune,
                start_time=start_time,
            )

        results["execution_mode"] = "discrete"

        # Write transformer input (with all columns needed across downstream stages)
        df_input.to_parquet(transformer_input_path, index=False)
        logger.info(f"Pipeline input: {len(df_input)} notes")

        self._init_ray()

        # Resolve settings via hardware recommender
        self._apply_pipeline_recommendations(model_name, t_kw, r_kw, a_kw, hardware_autotune)

        # ------------------------------------------------------------------
        # Phase 1: Transformer NER
        # ------------------------------------------------------------------
        t_kwargs: dict[str, Any] = dict(t_kw)

        if llm_recognizer_mode == "only":
            # In "only" mode, skip transformer entirely — LLM replaces it
            if run_transformer:
                logger.warning("llm_recognizer_mode='only' overrides run_transformer=True; skipping transformer stage")
            logger.info("Pipeline phase 1/3: Transformer NER (SKIPPED — LLM-only mode)")
        elif run_transformer:
            logger.info("Pipeline phase 1/3: Transformer NER")
            transformer_manifest = self.run_transformer(
                input_path=str(transformer_input_path),
                output_path=str(transformer_output_path),
                model_name=model_name,
                **t_kwargs,
            )
            results["transformer"] = transformer_manifest
        else:
            logger.info("Pipeline phase 1/3: Transformer NER (SKIPPED)")

        # ------------------------------------------------------------------
        # Phase 2: Recognizer (+ optional LLM recognizer)
        # ------------------------------------------------------------------
        r_kwargs: dict[str, Any] = dict(r_kw)

        rec_input_path = transformer_output_path if run_transformer else transformer_input_path

        if llm_recognizer_mode == "only":
            # LLM replaces both transformer and regex recognizer
            logger.info("Pipeline phase 2/3: LLM Recognizer (only mode)")

            # Write LLM recognizer input (just note_text + text_hash)
            llm_input_path = output_path / "03_llm_recognizer_input.parquet"
            df_input[["note_text", "text_hash"]].to_parquet(llm_input_path, index=False)

            llm_manifest = self.run_llm_recognition(
                input_path=str(llm_input_path),
                output_path=str(recognizer_output_path),
                **llm_kw,
            )
            results["llm_recognizer"] = llm_manifest

        elif llm_recognizer_mode == "merge":
            # Run both regex recognizer and LLM recognizer, then merge
            logger.info("Pipeline phase 2/3: Recognizer + LLM Recognizer (merge mode)")

            # --- Standard regex recognizer ---
            if run_recognizer:
                recognizer_manifest = self.run_recognition(
                    input_path=str(rec_input_path),
                    output_path=str(recognizer_output_path),
                    **r_kwargs,
                )
                results["recognizer"] = recognizer_manifest
            else:
                logger.info("Regex recognizer stage skipped (run_recognizer=False)")

            # --- LLM recognizer ---
            llm_input_path = output_path / "03_llm_recognizer_input.parquet"
            df_input[["note_text", "text_hash"]].to_parquet(llm_input_path, index=False)

            llm_manifest = self.run_llm_recognition(
                input_path=str(llm_input_path),
                output_path=str(llm_recognizer_output_path),
                **llm_kw,
            )
            results["llm_recognizer"] = llm_manifest

            # --- Merge results ---
            logger.info("Merging regex and LLM recognizer results")
            from presidio_anonymizer.entities import RecognizerResult

            from tide2.utils.span_metrics import resolve_recognizer_results

            # Read regex recognizer output
            rec_files = list(recognizer_output_path.glob("**/*.parquet"))
            if rec_files:
                dfs_regex = [pq.read_table(f).to_pandas() for f in rec_files]
                df_regex = pd.concat(dfs_regex, ignore_index=True)
            else:
                df_regex = pd.DataFrame(columns=["text_hash", "recognizer_results_json"])

            # Read LLM recognizer output
            llm_files = list(llm_recognizer_output_path.glob("**/*.parquet"))
            if llm_files:
                dfs_llm = [pq.read_table(f).to_pandas() for f in llm_files]
                df_llm = pd.concat(dfs_llm, ignore_index=True)
            else:
                df_llm = pd.DataFrame(columns=["text_hash", "recognizer_results_json"])

            # Merge on text_hash
            df_merged = df_regex[["text_hash", "recognizer_results_json"]].merge(
                df_llm[["text_hash", "recognizer_results_json"]],
                on="text_hash",
                how="outer",
                suffixes=("_regex", "_llm"),
            )

            merged_rows = []
            for _, row in df_merged.iterrows():
                regex_json = row.get("recognizer_results_json_regex", "[]")
                llm_json = row.get("recognizer_results_json_llm", "[]")
                if pd.isna(regex_json):
                    regex_json = "[]"
                if pd.isna(llm_json):
                    llm_json = "[]"

                regex_results = [
                    RecognizerResult(
                        entity_type=r["entity_type"],
                        start=r["start"],
                        end=r["end"],
                        score=r["score"],
                    )
                    for r in json.loads(regex_json)
                ]
                llm_results = [
                    RecognizerResult(
                        entity_type=r["entity_type"],
                        start=r["start"],
                        end=r["end"],
                        score=r["score"],
                    )
                    for r in json.loads(llm_json)
                ]

                combined = regex_results + llm_results
                resolved = resolve_recognizer_results(combined, strategy="longest_wins") if combined else []

                merged_rows.append(
                    {
                        "text_hash": row["text_hash"],
                        "recognizer_results_json": json.dumps(
                            [
                                {
                                    "entity_type": r.entity_type,
                                    "start": r.start,
                                    "end": r.end,
                                    "score": r.score,
                                }
                                for r in resolved
                            ]
                        ),
                    }
                )

            df_merged_results = pd.DataFrame(merged_rows)
            cols_to_keep = [c for c in ["note_text", "patient_uid", "row_id", "jitter"] if c in df_regex.columns]
            if cols_to_keep:
                df_merged_results = df_merged_results.merge(
                    df_regex[["text_hash", *cols_to_keep]].drop_duplicates(subset=["text_hash"]),
                    on="text_hash",
                    how="left",
                )

            # Overwrite recognizer output with merged results
            if recognizer_output_path.exists():
                shutil.rmtree(recognizer_output_path)
            recognizer_output_path.mkdir(parents=True, exist_ok=True)
            df_merged_results.to_parquet(recognizer_output_path / "merged_results.parquet", index=False)
            logger.info(f"Merged {len(df_merged_results)} notes from regex + LLM recognizers")

        elif run_recognizer:
            # Standard recognizer path (no LLM)
            logger.info("Pipeline phase 2/3: Recognizer")
            recognizer_manifest = self.run_recognition(
                input_path=str(rec_input_path),
                output_path=str(recognizer_output_path),
                **r_kwargs,
            )
            results["recognizer"] = recognizer_manifest
        else:
            logger.info("Pipeline phase 2/3: Recognizer (SKIPPED)")

        # ------------------------------------------------------------------
        # Phase 3: Anonymizer
        # ------------------------------------------------------------------
        if run_anonymizer:
            logger.info("Pipeline phase 3/3: Anonymizer")

            # Write hex keys to temp files
            salt_file = output_path / "salt.bin"
            key_file = output_path / "key.bin"
            salt_file.write_text(salt_hex)
            key_file.write_text(key_hex)

            a_kwargs: dict[str, Any] = dict(a_kw)

            anon_input_path = (
                recognizer_output_path
                if run_recognizer or llm_recognizer_mode != "off"
                else (transformer_output_path if run_transformer else transformer_input_path)
            )

            anonymizer_manifest = self.run_anonymization(
                input_path=str(anon_input_path),
                output_path=str(anonymizer_output_path),
                salt_path=str(salt_file),
                key_path=str(key_file),
                **a_kwargs,
            )
            results["anonymizer"] = anonymizer_manifest
        else:
            logger.info("Pipeline phase 3/3: Anonymizer (SKIPPED)")

        # ------------------------------------------------------------------
        # Visualizer JSON output
        # ------------------------------------------------------------------
        if produce_visualizer_json:
            logger.info("Creating visualizer JSON files")
            self._write_visualizer_json(
                output_path=output_path,
                transformer_output_path=transformer_output_path,
                recognizer_output_path=recognizer_output_path,
                anonymizer_output_path=anonymizer_output_path,
                df_input=df_input,
            )
            results["visualizer_json"] = True

        results["total_elapsed_seconds"] = round(time.time() - start_time, 2)
        logger.info(f"Pipeline complete in {results['total_elapsed_seconds']:.1f}s")
        return results

    # ------------------------------------------------------------------
    # Streamed execution
    # ------------------------------------------------------------------

    @staticmethod
    def _streamed_fallback_reason(
        *,
        llm_recognizer_mode: str,
        produce_visualizer_json: bool,
        stage_kwargs: tuple[dict[str, Any], ...],
        run_transformer: bool,
        run_recognizer: bool,
        run_anonymizer: bool,
    ) -> str | None:
        """Return why streamed cannot be used here, or None if it can.

        These are *fallbacks*, not errors: refusing to run because the caller
        asked for resume (or for the visualizer) would be the wrong trade. The
        hard refusals — multi-node and ≤4-CPU nodes — live in
        ``check_streamed_admission``, because they cannot be satisfied at all.
        """
        if llm_recognizer_mode == "merge":
            return (
                "llm_recognizer_mode='merge' requires discrete execution: chaining it would consume the "
                "shared transformer prefix twice (re-running GPU inference) and needs a row_id join that "
                "streamed mode does not yet implement."
            )
        if produce_visualizer_json:
            return (
                "produce_visualizer_json=True requires discrete execution: the visualizer's recognizer JSON "
                "needs the pre-anonymization recognizer_results_json, which the anonymizer does not carry "
                "into the final output and which streamed mode writes no intermediate for."
            )
        if any(kw.get("enable_checkpoint") is True for kw in stage_kwargs):
            return "enable_checkpoint=True requires discrete execution: a chained plan has no per-stage sink to key row-level resume on."
        if not (run_transformer or run_recognizer or run_anonymizer) and llm_recognizer_mode == "off":
            return "No stages are enabled, so there is nothing to chain."
        return None

    def _apply_pipeline_recommendations(
        self,
        model_name: str,
        t_kw: dict[str, Any],
        r_kw: dict[str, Any],
        a_kw: dict[str, Any],
        hardware_autotune: bool,
    ) -> None:
        """Fill per-stage kwargs in place from the hardware recommender."""
        hw = detect_hardware()
        rec = recommend_settings(hw, model_name=model_name)
        runner_kw: dict[str, Any] = {
            # Seeded from the constructor so an explicitly supplied timeout is
            # reported as USER and never overridden by a recommendation.
            "no_progress_timeout_s": self.no_progress_timeout_s,
            "object_store_gb": self.object_store_gb,
        }
        applied = apply_recommendations(
            rec,
            transformer=t_kw,
            recognizer=r_kw,
            anonymizer=a_kw,
            runner=runner_kw,
            hardware_autotune=hardware_autotune,
        )
        logger.info("\n" + render_settings_table(applied))

        # Make the resolved timeout sticky: every stage below reconfigures the
        # DataContext and would otherwise reset it to the library default.
        self.no_progress_timeout_s = runner_kw.get("no_progress_timeout_s")

    def _resolve_streamed_transformer_actor(
        self,
        model_name: str,
        t_kw: dict[str, Any],
    ) -> tuple[type, int, int, dict[str, Any]]:
        """Resolve the transformer actor class and its pool/resource settings.

        Mirrors ``run_transformer``'s driver-side resolution (model config
        validation, window overlap, GPU/actor counts, model download) without
        touching the DataContext, checkpoints, or the filesystem.

        Returns:
            ``(actor_class, num_transformer_actors, num_agg_actors, ray_remote_args)``
        """
        from tide2.actors import create_transformer_actor
        from tide2.transformers.config import load_model_config

        model_config = load_model_config(model_name)
        _mml = model_config.get("MODEL_MAX_LENGTH")
        if not isinstance(_mml, int) or _mml <= 0:
            raise ValueError(
                f"Model {model_name!r} config is missing a valid 'MODEL_MAX_LENGTH' (got {_mml!r}). "
                "It must be a positive integer equal to the model's real tokenized context window."
            )
        chunk_overlap = t_kw.get("chunk_overlap")
        if chunk_overlap is None:
            chunk_overlap = model_config.get("CHUNK_OVERLAP_SIZE", 40)

        num_gpus, cpu_only_mode, num_transformer_actors, num_agg_actors = self._resolve_transformer_resources(
            t_kw.get("num_gpus"), t_kw.get("num_transformer_actors"), t_kw.get("num_agg_actors")
        )

        model_path = t_kw.get("model_path")
        if model_path is None:
            from tide2.utils.gcs_resource_manager import resolve_model_path

            model_path = resolve_model_path(
                model_name=model_name,
                bucket_name=t_kw.get("bucket_name"),
                project_id=t_kw.get("project_id"),
            )

        actor = create_transformer_actor(
            model_name=model_name,
            model_path=model_path,
            bucket_name=t_kw.get("bucket_name"),
            project_id=t_kw.get("project_id"),
            gpu_batch_size=t_kw.get("gpu_batch_size"),
            window_overlap=chunk_overlap,
            aggregate_bio=num_agg_actors == 0,
        )

        ray_remote_args = get_ray_remote_args_gpu(num_gpus=0 if cpu_only_mode else num_gpus)
        # ``transformer_cpus`` is a *reservation*, not a measured need; the
        # discrete default starves the CPU pools in a chained plan. In GPU mode
        # it does not cap torch threads (that path is CPU-mode only), so a low
        # floor only risks throughput, never correctness.
        transformer_cpus = t_kw.get("transformer_cpus")
        if transformer_cpus is None and not cpu_only_mode:
            transformer_cpus = 1.0
        if transformer_cpus is not None:
            ray_remote_args["num_cpus"] = transformer_cpus

        return actor, num_transformer_actors, num_agg_actors, ray_remote_args

    def _run_pipeline_streamed(  # noqa: PLR0915 # one linear plan; splitting it hides the chain
        self,
        *,
        df_input: pd.DataFrame,
        output_path: Path,
        model_name: str,
        run_transformer: bool,
        run_recognizer: bool,
        run_anonymizer: bool,
        salt_hex: str,
        key_hex: str,
        transformer_kwargs: dict[str, Any],
        recognizer_kwargs: dict[str, Any],
        anonymizer_kwargs: dict[str, Any],
        llm_recognizer_mode: str,
        llm_recognizer_kwargs: dict[str, Any],
        hardware_autotune: bool,
        start_time: float,
    ) -> dict[str, Any]:
        """Chain the enabled stages into a single Ray Data execution.

        See ``run_pipeline``'s docstring for the mode's semantics and
        trade-offs. This method assumes the fallback checks have already run.
        """
        from tide2.actors import create_anonymizer_actor_class

        logger.info(
            "streamed mode: no row-level resume; a mid-run failure re-runs GPU inference. "
            "Use execution_mode='discrete' for long-running or unattended jobs."
        )

        self._init_ray()
        self._apply_pipeline_recommendations(
            model_name, transformer_kwargs, recognizer_kwargs, anonymizer_kwargs, hardware_autotune
        )

        anonymizer_output_path = output_path / "06_anonymizer_output"
        recognizer_output_path = output_path / "04_recognizer_output"

        use_llm = llm_recognizer_mode == "only"
        if use_llm and run_transformer:
            logger.warning("llm_recognizer_mode='only' overrides run_transformer=True; skipping transformer stage")
        chain_transformer = run_transformer and not use_llm
        chain_recognizer = run_recognizer and not use_llm

        # --- Column contracts, validated on the source before any operator ---
        contracts: list[tuple[str, StageColumns]] = []
        if use_llm:
            contracts.append(("llm_recognizer", LLM_RECOGNIZER_STAGE_COLUMNS))
        else:
            if chain_transformer:
                contracts.append(("transformer", TRANSFORMER_STAGE_COLUMNS))
            if chain_recognizer:
                contracts.append(("recognizer", RECOGNIZER_STAGE_COLUMNS))
        if run_anonymizer:
            contracts.append(("anonymizer", ANONYMIZER_STAGE_COLUMNS))
        final_columns = validate_stage_columns(df_input.columns, contracts)

        # --- Resolve pools and CPU budget, then admit or fail fast ---
        pool_minimums: dict[str, float] = {}
        read_cpus = float(transformer_kwargs.get("read_cpus") or recognizer_kwargs.get("read_cpus") or 0.25)
        write_cpus = float(anonymizer_kwargs.get("write_cpus") or 0.5)
        pool_minimums["read"] = read_cpus
        pool_minimums["write"] = write_cpus

        transformer_actor = None
        if chain_transformer:
            (
                transformer_actor,
                num_transformer_actors,
                num_agg_actors,
                t_remote_args,
            ) = self._resolve_streamed_transformer_actor(model_name, transformer_kwargs)
            pool_minimums["transformer"] = float(t_remote_args.get("num_cpus", 0.0))
            if num_agg_actors:
                pool_minimums["bio_aggregation"] = float(transformer_kwargs.get("agg_num_cpus") or 1.0)

        node_cpus = max(_alive_node_cpus() or [0.0])

        llm_remote_args: dict[str, Any] = {}
        if use_llm:
            llm_actors = int(llm_recognizer_kwargs.get("num_actors") or 4)
            # Network-bound: it reserves CPU but barely uses any, so keep the
            # slot reservation small and the pool size independent of the budget.
            llm_remote_args = get_ray_remote_args_cpu(num_cpus=float(llm_recognizer_kwargs.get("num_cpus") or 0.25))
            pool_minimums["llm_recognizer"] = float(llm_remote_args["num_cpus"])

        if chain_recognizer:
            r_slot, r_remote_args, _ = _resolve_slot_cpus(
                num_cpus=recognizer_kwargs.get("num_cpus"),
                worker_num_cpus=recognizer_kwargs.get("worker_num_cpus"),
                stage_name="streamed recognizer",
            )
            rec_max = int(recognizer_kwargs.get("num_actors") or max(2, int(node_cpus * 0.625)))
            rec_min = min(2, rec_max)
            pool_minimums["recognizer"] = r_slot * rec_min

        if run_anonymizer:
            a_slot, a_remote_args, _ = _resolve_slot_cpus(
                num_cpus=anonymizer_kwargs.get("num_cpus"),
                worker_num_cpus=anonymizer_kwargs.get("worker_num_cpus"),
                stage_name="streamed anonymizer",
            )
            anon_max = int(anonymizer_kwargs.get("num_actors") or max(1, int(node_cpus * 0.5)))
            anon_min = 1
            pool_minimums["anonymizer"] = a_slot * anon_min

        node_cpus = check_streamed_admission(pool_minimums)

        # --- One DataContext for the whole plan ---
        # Today each stage method calls configure_data_context with its own
        # arguments; the context is global and last-call-wins, so in a chained
        # plan those calls would race. Blocks carry note_text through every
        # operator here, so keep target_max_block_size modest.
        target_max_block_size_mb = int(anonymizer_kwargs.get("target_max_block_size_mb") or 64)
        configure_data_context(
            **self._data_context_kwargs(
                verbose_progress=True,
                target_max_block_size_mb=target_max_block_size_mb,
                target_min_block_size_mb=int(anonymizer_kwargs.get("target_min_block_size_mb") or 1),
                read_op_min_num_blocks=int(anonymizer_kwargs.get("read_op_min_num_blocks") or 200),
            )
        )
        ctx = ray.data.DataContext.get_current()
        ctx.checkpoint_config = None

        num_blocks = next(
            (
                int(kw["override_num_blocks"])
                for kw in (transformer_kwargs, recognizer_kwargs, anonymizer_kwargs)
                if kw.get("override_num_blocks") is not None
            ),
            max(32, int(node_cpus * 2)),
        )
        num_blocks = min(num_blocks, max(1, len(df_input)))

        logger.info("Streamed pipeline plan:")
        logger.info(f"  Stages: {[name for name, _ in contracts]}")
        logger.info(f"  Pool CPU minimums: {pool_minimums} (Σ={sum(pool_minimums.values())}, node CPUs={node_cpus})")
        logger.info(f"  Source blocks: {num_blocks}, target_max_block_size={target_max_block_size_mb}MB")
        logger.info(
            f"  no_progress_timeout_s={getattr(ctx, 'execution_no_progress_timeout_s', None)}, "
            f"object_store_gb={self.object_store_gb}, checkpointing=disabled"
        )

        shutdown = GracefulShutdown()
        input_rows = len(df_input)
        try:
            # ``from_pandas`` defaults to very few blocks and the first pool
            # starves, so split the source at creation — the same starvation the
            # discrete path fixes with override_num_blocks. Passing the block
            # count here rather than calling ``.repartition()`` avoids an
            # all-to-all pass that would push the whole corpus (``note_text``
            # included) through the object store before the first stage starts.
            source_cols = [c for c in df_input.columns if c in _streamed_source_columns(contracts)]
            ds: Dataset = ray.data.from_pandas(df_input[source_cols], override_num_blocks=num_blocks)

            if use_llm:
                llm_kwargs = {k: v for k, v in llm_recognizer_kwargs.items() if k not in ("num_actors", "num_cpus")}
                llm_kwargs.setdefault("provider_type", "google")
                llm_kwargs.setdefault("model_name", "gemini-2.5-flash")
                llm_kwargs.setdefault("prompt_name", "phi_detection")
                ds = self.build_llm_recognizer_stage(
                    ds,
                    batch_size=int(llm_recognizer_kwargs.get("batch_size") or 10),
                    num_actors=llm_actors,
                    ray_remote_args=llm_remote_args,
                    fn_constructor_kwargs=llm_kwargs,
                    pool_min_size=1,
                )
            else:
                if chain_transformer:
                    assert transformer_actor is not None
                    ds = self.build_transformer_stage(
                        ds,
                        transformer_actor=transformer_actor,
                        model_name=model_name,
                        batch_size=int(transformer_kwargs.get("batch_size") or 8),
                        num_transformer_actors=num_transformer_actors,
                        ray_remote_args_transformer=t_remote_args,
                        num_agg_actors=num_agg_actors,
                        agg_num_cpus=float(transformer_kwargs.get("agg_num_cpus") or 1.0),
                    )
                if chain_recognizer:
                    ds = self.build_recognizer_stage(
                        ds,
                        batch_size=int(recognizer_kwargs.get("batch_size") or 150),
                        num_actors=rec_max,
                        ray_remote_args=r_remote_args,
                        pool_min_size=rec_min,
                    )

            if run_anonymizer:
                salt_file = output_path / "salt.bin"
                key_file = output_path / "key.bin"
                salt_file.write_text(salt_hex)
                key_file.write_text(key_hex)
                anonymizer_actor = create_anonymizer_actor_class(
                    salt=self._load_key(str(salt_file)),
                    key=self._load_key(str(key_file)),
                    acc_num_salt=anonymizer_kwargs.get("acc_num_salt"),
                    acc_num_study_id=anonymizer_kwargs.get("acc_num_study_id"),
                    jitter_required=bool(anonymizer_kwargs.get("jitter_required", False)),
                )
                ds = self.build_anonymizer_stage(
                    ds,
                    actor_cls=anonymizer_actor,
                    batch_size=int(anonymizer_kwargs.get("batch_size") or 200),
                    num_actors=anon_max,
                    ray_remote_args=a_remote_args,
                    pool_min_size=anon_min,
                )
                sink_dir = anonymizer_output_path
                # The anonymizer already emits exactly the final column set, so
                # this projection is a guard, not a transformation: it is what
                # keeps raw note_text out of the sink. ``final_columns`` comes
                # from the contract walk, so it includes pass-throughs such as
                # ``row_id`` that no stage lists under ``produces``.
                sink_columns = sorted(FINAL_OUTPUT_COLUMNS & final_columns)
            else:
                sink_dir = recognizer_output_path
                sink_columns = None

            sink_dir.mkdir(parents=True, exist_ok=True)
            if sink_columns:
                ds = ds.select_columns(sink_columns)
            ds.write_parquet(str(sink_dir), compression="zstd", ray_remote_args={"num_cpus": write_cpus})

            # Row reconciliation. Count with pyarrow, never ds.count() — that
            # re-executes the whole chained plan.
            import pyarrow.dataset as _pad

            output_files = list(sink_dir.glob("**/*.parquet"))
            output_rows = _pad.dataset(output_files).count_rows() if output_files else 0
            if output_rows == 0 and input_rows > 0:
                raise RuntimeError(
                    f"Streamed pipeline wrote 0 rows from {input_rows} input rows — all batches failed. "
                    "Check worker logs for the underlying error."
                )
            dropped = input_rows - output_rows
            if dropped > 0 and dropped / input_rows > STREAMED_DROP_WARN_FRACTION:
                logger.warning(
                    "Streamed pipeline dropped %d of %d rows (%.2f%%)",
                    dropped,
                    input_rows,
                    100.0 * dropped / input_rows,
                )

            try:
                operator_stats: dict[str, Any] = {"summary": ds.stats()}
            except Exception:  # pragma: no cover - stats are diagnostic only
                logger.debug("Could not collect Ray Data stats", exc_info=True)
                operator_stats = {}

            elapsed = round(time.time() - start_time, 2)
            logger.info(f"Streamed pipeline complete in {elapsed:.1f}s ({output_rows} rows)")
            return {
                "execution_mode": "streamed",
                "total_elapsed_seconds": elapsed,
                "input_rows": input_rows,
                "output_rows": output_rows,
                "dropped_rows": dropped,
                "output_dir": str(output_path),
                "operator_stats": operator_stats,
            }
        except ray.data.exceptions.ExecutionTimeoutError:
            _log_execution_timeout(
                "Streamed pipeline",
                ctx,
                "A 0/1 deadlock surfaces here: the named operator is the one that never started.",
            )
            raise
        except Exception:
            logger.exception("Streamed pipeline failed")
            raise
        finally:
            shutdown.restore_handlers()

    def shutdown(self) -> None:
        """Shutdown Ray after job completion."""
        if self._initialized:
            ray.shutdown()
            self._initialized = False
            logger.info("Ray shutdown complete")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_pipeline_input(input_data: str | pd.DataFrame) -> pd.DataFrame:
        """Load and normalize the pipeline input DataFrame.

        Lowercases column names (so e.g. ``JITTER`` from BigQuery works) and
        derives ``text_hash`` / ``patient_id`` / ``patient_uid`` / ``row_id``
        when absent. Shared verbatim by both execution modes so the normalized
        frame is identical.
        """
        from tide2.utils.text_processing import compute_text_hash

        df_input = pd.read_parquet(input_data) if isinstance(input_data, str) else input_data.copy()

        # Normalize column names to lowercase so that e.g. "JITTER" from BQ works
        df_input.columns = df_input.columns.str.lower()

        if "note_text" not in df_input.columns:
            raise ValueError("Input data must contain a 'note_text' column")

        if "text_hash" not in df_input.columns:
            df_input["text_hash"] = df_input["note_text"].apply(compute_text_hash)

        # Ensure patient_id exists (use text_hash as fallback)
        if "patient_id" not in df_input.columns:
            df_input["patient_id"] = df_input["text_hash"]

        if "patient_uid" not in df_input.columns:
            df_input["patient_uid"] = df_input["patient_id"]

        if "row_id" not in df_input.columns:
            df_input["row_id"] = (
                df_input["text_hash"] + ":" + df_input["patient_uid"].fillna("None").astype(str)
            ).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

        return df_input

    def _load_key(self, key_path: str) -> bytes:
        """Load a 32-byte key from a file (hex-encoded)."""
        with Path(key_path).open("r", encoding="utf-8") as f:
            hex_key = f.read().strip()
        key = bytes.fromhex(hex_key)
        if len(key) != KEY_SIZE_BYTES:
            raise ValueError(f"Key must be exactly {KEY_SIZE_BYTES} bytes, got {len(key)}")
        return key

    def _resolve_input_pattern(self, input_path: str | list[str]) -> str | list[str]:
        """Resolve input path to glob pattern, directory, or file list.

        Returns directory paths as-is for local filesystems so Ray Data
        handles listing directly (pyarrow glob fails on FUSE mounts).
        When input_path is a list of file paths, returns it unchanged
        (ray.data.read_parquet accepts List[str] natively).
        """
        if isinstance(input_path, list):
            return input_path
        if "*" in input_path or input_path.endswith(".parquet"):
            return input_path
        if not input_path.startswith("gs://") and Path(input_path).is_dir():
            return input_path
        return f"{input_path}/*.parquet"

    def _ensure_output_dir(self, output_path: str) -> None:
        """Ensure output directory exists."""
        if not output_path.startswith("gs://"):
            Path(output_path).mkdir(parents=True, exist_ok=True)

    def _build_recognizer_input_from_transformer(
        self,
        transformer_output_path: Path,
        df_input: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assemble the recognizer input DataFrame from transformer-stage output.

        The transformer stage now emits document-level rows directly
        (text_hash, patient_id, note_text, recognizer_results_json) — there is no
        chunk→reassembly step. This reads that output, preserves the passthrough
        columns (patient_identifiers, patient_uid, jitter, row_id), and attaches
        patient_identifiers from df_input using row_id if missing. Falls back to
        the raw input when no transformer output exists (e.g. the transformer
        stage was skipped).
        """
        trans_files = list(transformer_output_path.glob("**/*.parquet"))
        if trans_files:
            dfs_trans = [pq.read_table(f).to_pandas() for f in trans_files]
            df_rec_in = pd.concat(dfs_trans, ignore_index=True)
            keep = [
                c
                for c in [
                    "text_hash",
                    "patient_id",
                    "note_text",
                    "recognizer_results_json",
                    "patient_identifiers",
                    "patient_uid",
                    "jitter",
                    "row_id",
                ]
                if c in df_rec_in.columns
            ]
            df_rec_in = df_rec_in[keep].copy()
        else:
            logger.info("No transformer output found; using input data for recognizer")
            df_rec_in = df_input.copy()
            if "recognizer_results_json" not in df_rec_in.columns:
                df_rec_in["recognizer_results_json"] = "[]"

        if "patient_identifiers" in df_input.columns and "patient_identifiers" not in df_rec_in.columns:
            # Join by row_id when present in both frames; otherwise fall back to text_hash.
            if "row_id" in df_input.columns and "row_id" in df_rec_in.columns:
                id_map = df_input.drop_duplicates(subset="row_id").set_index("row_id")["patient_identifiers"]
                df_rec_in["patient_identifiers"] = df_rec_in["row_id"].map(id_map).fillna("{}")
            else:
                id_map = df_input.drop_duplicates(subset="text_hash").set_index("text_hash")["patient_identifiers"]
                df_rec_in["patient_identifiers"] = df_rec_in["text_hash"].map(id_map).fillna("{}")
        elif "patient_identifiers" not in df_rec_in.columns:
            df_rec_in["patient_identifiers"] = "{}"
        return df_rec_in

    def _get_text_source(
        self,
        trans_files: list[Path],
        df_input: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a text_hash/note_text DataFrame, preferring transformer output."""
        if trans_files:
            dfs_trans = [pq.read_table(f).to_pandas() for f in trans_files]
            df_trans = pd.concat(dfs_trans, ignore_index=True)
            col_source = df_trans if "note_text" in df_trans.columns else df_input
            return col_source[["text_hash", "note_text"]].drop_duplicates(subset=["text_hash"])
        return df_input[["text_hash", "note_text"]]

    def _write_recognizer_json_files(
        self,
        cli_recognizer_dir: Path,
        recognizer_output_path: Path,
        transformer_output_path: Path,
        df_input: pd.DataFrame,
    ) -> None:
        """Write per-sample recognizer JSON files for the visualizer."""
        rec_files = list(recognizer_output_path.glob("**/*.parquet"))
        if not rec_files:
            return
        dfs_rec = [pq.read_table(f).to_pandas() for f in rec_files]
        df_rec = pd.concat(dfs_rec, ignore_index=True)
        trans_files = list(transformer_output_path.glob("**/*.parquet"))
        text_source = self._get_text_source(trans_files, df_input)
        df_merged = df_rec.merge(text_source, on="text_hash", how="left")
        count = 0
        for _, row in df_merged.iterrows():
            sample_id = str(row.get("text_hash", "unknown"))
            note_text = row.get("note_text", "") if pd.notna(row.get("note_text")) else ""
            results_json = (
                row.get("recognizer_results_json", "[]") if pd.notna(row.get("recognizer_results_json")) else "[]"
            )
            recognizer_results = json.loads(results_json) if results_json else []
            cli_data = {"key": sample_id, "value": note_text, "recognizer_results": recognizer_results}
            with (cli_recognizer_dir / f"{sample_id}.json").open("w", encoding="utf-8") as f:
                json.dump(cli_data, f, ensure_ascii=False, indent=2)
            count += 1
        logger.info(f"Created {count} recognizer JSON files in {cli_recognizer_dir}")

    def _write_anonymizer_json_files(
        self,
        cli_anonymizer_dir: Path,
        anonymizer_output_path: Path,
    ) -> None:
        """Write per-sample anonymizer JSON files for the visualizer."""
        anon_files = list(anonymizer_output_path.glob("**/*.parquet"))
        if not anon_files:
            return
        dfs_anon = [pq.read_table(f).to_pandas() for f in anon_files]
        df_anon = pd.concat(dfs_anon, ignore_index=True)
        count = 0
        for _, row in df_anon.iterrows():
            sample_id = next(
                (str(row[col]) for col in ["text_hash"] if col in row.index and pd.notna(row[col])),
                "unknown",
            )
            anonymized_text = next(
                (
                    row[col]
                    for col in ["anonymized_note_text", "deid_note_text", "anonymized_text", "note_text"]
                    if col in row.index and pd.notna(row[col])
                ),
                "",
            )
            items: list[Any] = []
            for col in ["anonymizer_results_json", "items", "anonymizer_results"]:
                if col in row.index and pd.notna(row[col]):
                    items_data = row[col]
                    items = json.loads(items_data) if isinstance(items_data, str) else list(items_data)
                    break
            cli_data = {"text": anonymized_text, "items": items}
            with (cli_anonymizer_dir / f"{sample_id}.json").open("w", encoding="utf-8") as f:
                json.dump(cli_data, f, ensure_ascii=False, indent=2)
            count += 1
        logger.info(f"Created {count} anonymizer JSON files in {cli_anonymizer_dir}")

    def _write_visualizer_json(
        self,
        output_path: Path,
        transformer_output_path: Path,
        recognizer_output_path: Path,
        anonymizer_output_path: Path,
        df_input: pd.DataFrame,
    ) -> None:
        """Write JSON files for tide2-visualizer (unified_interface.py)."""
        cli_recognizer_dir = output_path / "cli_recognizer_json"
        cli_anonymizer_dir = output_path / "cli_anonymizer_json"
        cli_recognizer_dir.mkdir(parents=True, exist_ok=True)
        cli_anonymizer_dir.mkdir(parents=True, exist_ok=True)
        self._write_recognizer_json_files(cli_recognizer_dir, recognizer_output_path, transformer_output_path, df_input)
        self._write_anonymizer_json_files(cli_anonymizer_dir, anonymizer_output_path)


def run_recognition_simple(
    input_path: str,
    output_path: str,
    num_actors: int | None = None,
    batch_size: int = 150,
    num_cpus: int | None = None,
    object_store_gb: int | None = None,
) -> dict[str, Any]:
    """
    Simple function to run recognition job.

    Args:
        input_path: Input parquet files
        output_path: Output directory
        num_actors: Number of actors
        batch_size: Batch size
        num_cpus: CPUs
        object_store_gb: Object store size GB

    Returns:
        Processing statistics
    """
    runner = LocalJobRunner(
        num_cpus=num_cpus,
        object_store_gb=object_store_gb,
    )

    try:
        return runner.run_recognition(
            input_path=input_path,
            output_path=output_path,
            num_actors=num_actors,
            batch_size=batch_size,
        )
    finally:
        runner.shutdown()


def run_anonymization_simple(
    input_path: str,
    output_path: str,
    salt_path: str,
    key_path: str,
    num_actors: int | None = None,
    batch_size: int = 200,
    num_cpus: int | None = None,
    object_store_gb: int | None = None,
) -> dict[str, Any]:
    """
    Simple function to run anonymization job.

    Args:
        input_path: Input parquet files
        output_path: Output directory
        salt_path: Path to salt file
        key_path: Path to key file
        num_actors: Number of actors
        batch_size: Batch size
        num_cpus: CPUs
        object_store_gb: Object store size GB

    Returns:
        Processing statistics
    """
    runner = LocalJobRunner(
        num_cpus=num_cpus,
        object_store_gb=object_store_gb,
    )

    try:
        return runner.run_anonymization(
            input_path=input_path,
            output_path=output_path,
            salt_path=salt_path,
            key_path=key_path,
            num_actors=num_actors,
            batch_size=batch_size,
        )
    finally:
        runner.shutdown()


def run_transformer_simple(
    input_path: str | list[str],
    output_path: str,
    model_name: str,
    bucket_name: str | None = None,
    project_id: str | None = None,
    num_gpus: int | None = None,
    batch_size: int = 8,
    chunk_overlap: int | None = None,
    num_cpus: int | None = None,
    object_store_gb: int | None = None,
    num_agg_actors: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Simple function to run transformer NER job.

    Args:
        input_path: Input parquet files
        output_path: Output directory
        model_name: Name of transformer model configuration
        bucket_name: Optional GCS bucket for model loading
        project_id: Optional GCP project ID
        num_gpus: Number of GPU actors
        batch_size: Batch size for map_batches (whole notes per actor call)
        chunk_overlap: Token overlap between adjacent windows (default: from model config)
        num_cpus: CPUs
        object_store_gb: Object store size GB
        num_agg_actors: Number of CPU actors for BIO aggregation (auto if None)
        **kwargs: Deprecated parameters. Passing any deprecated argument
            will raise a ValueError with a deprecation warning.

    Returns:
        Processing statistics
    """
    _check_deprecated_transformer_kwargs(kwargs, "run_transformer_simple")
    runner = LocalJobRunner(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        object_store_gb=object_store_gb,
    )

    try:
        return runner.run_transformer(
            input_path=input_path,
            output_path=output_path,
            model_name=model_name,
            bucket_name=bucket_name,
            project_id=project_id,
            num_gpus=num_gpus,
            batch_size=batch_size,
            chunk_overlap=chunk_overlap,
            num_agg_actors=num_agg_actors,
        )
    finally:
        runner.shutdown()


def run_pipeline_simple(
    input_data: str | pd.DataFrame,
    output_dir: str,
    model_name: str,
    *,
    run_transformer: bool = True,
    run_recognizer: bool = True,
    run_anonymizer: bool = True,
    produce_visualizer_json: bool = False,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    object_store_gb: int | None = None,
    transformer_kwargs: dict[str, Any] | None = None,
    recognizer_kwargs: dict[str, Any] | None = None,
    anonymizer_kwargs: dict[str, Any] | None = None,
    llm_recognizer_mode: str = "off",
    llm_recognizer_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Simple function to run the full de-identification pipeline.

    Args:
        input_data: Parquet path or DataFrame with at least a 'note_text' column.
        output_dir: Output directory for all intermediate and final files.
        model_name: Transformer model name.
        run_transformer: Run GPU transformer NER stage.
        run_recognizer: Run CPU recognizer stage.
        run_anonymizer: Run CPU anonymizer stage.
        produce_visualizer_json: Write JSON files for tide2-visualizer.
        num_cpus: CPUs for Ray.
        num_gpus: GPUs for Ray.
        object_store_gb: Object store size in GB.
        transformer_kwargs: Extra kwargs passed to run_transformer().
        recognizer_kwargs: Extra kwargs passed to run_recognition().
        anonymizer_kwargs: Extra kwargs passed to run_anonymization().
        llm_recognizer_mode: "off", "only", or "merge".
        llm_recognizer_kwargs: Extra kwargs passed to run_llm_recognition().

    Returns:
        Pipeline statistics dictionary.
    """
    runner = LocalJobRunner(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        object_store_gb=object_store_gb,
    )

    try:
        return runner.run_pipeline(
            input_data=input_data,
            output_dir=output_dir,
            model_name=model_name,
            run_transformer=run_transformer,
            run_recognizer=run_recognizer,
            run_anonymizer=run_anonymizer,
            produce_visualizer_json=produce_visualizer_json,
            transformer_kwargs=transformer_kwargs,
            recognizer_kwargs=recognizer_kwargs,
            anonymizer_kwargs=anonymizer_kwargs,
            llm_recognizer_mode=llm_recognizer_mode,
            llm_recognizer_kwargs=llm_recognizer_kwargs,
        )
    finally:
        runner.shutdown()
