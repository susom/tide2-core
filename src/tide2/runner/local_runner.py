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
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import ray
import ray.data
from ray.data.checkpoint import CheckpointConfig
from ray.data.dataset import Dataset

from .fault_tolerance import GracefulShutdown
from .fault_tolerance import configure_data_context
from .fault_tolerance import get_ray_remote_args_cpu
from .fault_tolerance import get_ray_remote_args_gpu
from .utils import DEFAULT_DASHBOARD_HOST
from .utils import detect_columns
from .utils import log_ray_cluster_info
from .utils import resolve_input_files

logger = logging.getLogger(__name__)

KEY_SIZE_BYTES = 32


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
        num_gpus: int | None = None,
        object_store_gb: int | None = None,
        dashboard_host: str = DEFAULT_DASHBOARD_HOST,
        include_dashboard: bool = False,
    ):
        """
        Initialize local job runner.

        Args:
            num_cpus: CPU count override
            num_gpus: GPU count override
            object_store_gb: Object store size in GB (default: ~30% of system RAM)
            dashboard_host: Dashboard host
            include_dashboard: Enable Ray dashboard
        """
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.object_store_gb = object_store_gb
        self.dashboard_host = dashboard_host
        self.include_dashboard = include_dashboard
        self._initialized = False

    def _init_ray(self) -> None:
        """Initialize Ray."""
        if self._initialized:
            return

        # When dashboard is enabled, bind to 0.0.0.0 so it's accessible
        # from outside Docker containers
        dashboard_host = self.dashboard_host
        if self.include_dashboard and dashboard_host == DEFAULT_DASHBOARD_HOST:
            dashboard_host = "0.0.0.0"  # noqa: S104 # nosec B104 # we run in docker

        kwargs: dict[str, Any] = {
            "ignore_reinit_error": True,
            "include_dashboard": self.include_dashboard,
            "dashboard_host": dashboard_host,
        }

        if self.num_cpus:
            kwargs["num_cpus"] = self.num_cpus
        if self.num_gpus:
            kwargs["num_gpus"] = self.num_gpus
        if self.object_store_gb:
            kwargs["object_store_memory"] = self.object_store_gb * 1024**3
        else:
            # Auto-tune: ~30% of system RAM for object store
            try:
                import psutil

                total_ram = psutil.virtual_memory().total
                kwargs["object_store_memory"] = int(total_ram * 0.3)
            except ImportError:
                pass  # Let Ray use its default

        ray.init(**kwargs)
        logger.info("Ray initialized")

        # Log resources
        log_ray_cluster_info()

        # Configure Ray Data context
        configure_data_context(verbose_progress=True)

        self._initialized = True

    def _auto_num_actors(self, fraction: float = 0.45) -> int:
        """Auto-detect number of actors from cluster resources.

        Uses 0.45 of available CPUs by default because the supervisor/worker
        pattern doubles the actual process count (each supervisor spawns a
        remote worker actor). On a 224-CPU machine this yields ~100 supervisors
        + ~100 workers = ~200 processes.
        """
        cpus = ray.cluster_resources().get("CPU", 4)
        return max(1, int(cpus * fraction))

    def _resolve_transformer_resources(
        self,
        num_gpus: int | None,
        num_transformer_actors: int | None,
        num_agg_actors: int | None,
    ) -> tuple[int, bool, int, int]:
        """Resolve GPU/CPU resources and actor counts for transformer jobs.

        Returns:
            Tuple of (num_gpus, cpu_only_mode, num_transformer_actors, num_agg_actors)
        """
        if num_gpus is None:
            num_gpus = int(ray.cluster_resources().get("GPU", 0))

        cpu_only_mode = num_gpus == 0
        available_cpus = ray.cluster_resources().get("CPU", 4)

        if num_transformer_actors is None:
            # Each transformer actor is memory-intensive (~500MB+ for model)
            # CPU mode: limit actors; GPU mode: one actor per GPU
            num_transformer_actors = max(1, int(available_cpus * 0.25)) if cpu_only_mode else num_gpus

        if cpu_only_mode:
            logger.warning(
                "No GPUs available, running transformer inference on CPU. "
                "This will be significantly slower than GPU inference."
            )

        if num_agg_actors is None:
            num_agg_actors = max(1, int(available_cpus * 0.3))

        return num_gpus, cpu_only_mode, num_transformer_actors, num_agg_actors

    def run_recognition(
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
            dry_run: If True, validate setup and show plan without processing

        Returns:
            Processing statistics dictionary
        """
        from tide2.actors import RecognizerActor

        self._init_ray()

        # Override DataContext with job-specific streaming params
        configure_data_context(
            verbose_progress=True,
            target_max_block_size_mb=target_max_block_size_mb,
            target_min_block_size_mb=target_min_block_size_mb,
            read_op_min_num_blocks=read_op_min_num_blocks,
        )

        start_time = time.time()

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        shutdown = GracefulShutdown()

        # Resolve input files
        input_files = resolve_input_files(input_path)
        if not input_files:
            raise FileNotFoundError(f"No files found matching: {input_path}")

        logger.info(f"Found {len(input_files)} input file(s)")

        # Auto-detect actors
        if num_actors is None:
            num_actors = self._auto_num_actors()

        # Detect columns
        required_cols = ["text_hash", "note_text", "patient_identifiers"]
        optional_cols = ["recognizer_results_json"]
        columns = detect_columns(input_files[0], required_cols, optional_cols)

        logger.info("Recognition job starting")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Actors: {num_actors}, Batch size: {batch_size}, CPUs/actor: {num_cpus}")

        ray_remote_args = get_ray_remote_args_cpu(num_cpus=num_cpus)

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
            ctx = ray.data.DataContext.get_current()
            _configure_checkpoint(ctx, enable=enable_checkpoint, output_dir=output_dir, id_column="text_hash")

            # Single streaming pipeline — no repartition, no segment loop.
            num_blocks = read_parallelism if read_parallelism is not None else len(input_files)
            # Ensure enough blocks to utilize all actors
            num_blocks = max(num_blocks, num_actors)
            ds = ray.data.read_parquet(
                input_files,
                columns=columns,
                override_num_blocks=num_blocks,
                ray_remote_args={"num_cpus": read_cpus},
            )
            processed = ds.map_batches(
                RecognizerActor,
                batch_size=batch_size,
                compute=ray.data.ActorPoolStrategy(size=num_actors),
                fn_constructor_kwargs={
                    "batch_timeout": batch_timeout,
                    "worker_num_cpus": worker_num_cpus,
                },
                **ray_remote_args,
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
        from tide2.actors import LlmRecognizerActor

        self._init_ray()

        # Override DataContext with job-specific streaming params
        configure_data_context(
            verbose_progress=True,
            target_max_block_size_mb=target_max_block_size_mb,
            target_min_block_size_mb=target_min_block_size_mb,
            read_op_min_num_blocks=read_op_min_num_blocks,
        )

        start_time = time.time()

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        shutdown = GracefulShutdown()

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

        logger.info("LLM Recognition job starting")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Model: {model_name} (provider: {provider_type}, prompt: {prompt_name})")
        logger.info(f"  Actors: {num_actors}, Batch size: {batch_size}, CPUs/actor: {num_cpus}")

        ray_remote_args = get_ray_remote_args_cpu(num_cpus=num_cpus)

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
            ctx = ray.data.DataContext.get_current()
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
            processed = ds.map_batches(
                LlmRecognizerActor,
                batch_size=batch_size,
                compute=ray.data.ActorPoolStrategy(size=num_actors),
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
                    "batch_timeout": batch_timeout,
                    "worker_num_cpus": worker_num_cpus,
                },
                **ray_remote_args,
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

        except Exception:
            logger.exception("LLM Recognition failed")
            raise

        finally:
            shutdown.restore_handlers()

    def run_anonymization(
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
            verbose_progress=True,
            target_max_block_size_mb=target_max_block_size_mb,
            target_min_block_size_mb=target_min_block_size_mb,
            read_op_min_num_blocks=read_op_min_num_blocks,
        )

        start_time = time.time()

        # Load keys
        salt = self._load_key(salt_path)
        key = self._load_key(key_path)

        output_dir = Path(output_path).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        shutdown = GracefulShutdown()

        # Resolve input files
        input_files = resolve_input_files(input_path)
        if not input_files:
            raise FileNotFoundError(f"No files found matching: {input_path}")

        logger.info(f"Found {len(input_files)} input file(s)")

        if num_actors is None:
            num_actors = self._auto_num_actors()

        # Detect columns
        required_cols = ["text_hash", "note_text", "recognizer_results_json", "patient_uid"]
        optional_cols = ["jitter", "row_id"]
        columns = detect_columns(input_files[0], required_cols, optional_cols)

        logger.info("Anonymization job starting")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Actors: {num_actors}, Batch size: {batch_size}, CPUs/actor: {num_cpus}")

        # Create actor class with keys
        AnonymizerActor = create_anonymizer_actor_class(  # noqa: N806 # its a type
            salt=salt,
            key=key,
            acc_num_salt=acc_num_salt,
            acc_num_study_id=acc_num_study_id,
            jitter_required=jitter_required,
            worker_num_cpus=worker_num_cpus,
        )

        ray_remote_args = get_ray_remote_args_cpu(num_cpus=num_cpus)

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
            ctx = ray.data.DataContext.get_current()
            id_col = "row_id" if "row_id" in columns else "text_hash"
            _configure_checkpoint(ctx, enable=enable_checkpoint, output_dir=output_dir, id_column=id_col)

            # Single streaming pipeline — no repartition, no segment loop.
            num_blocks = read_parallelism if read_parallelism is not None else len(input_files)
            # Ensure enough blocks to utilize all actors
            num_blocks = max(num_blocks, num_actors)
            ds = ray.data.read_parquet(
                input_files,
                columns=columns,
                override_num_blocks=num_blocks,
                ray_remote_args={"num_cpus": read_cpus},
            )
            processed = ds.map_batches(
                AnonymizerActor,
                batch_size=batch_size,
                compute=ray.data.ActorPoolStrategy(size=num_actors),
                **ray_remote_args,
            )
            processed.write_parquet(str(output_dir), compression="zstd", ray_remote_args={"num_cpus": write_cpus})

            # Guard against silent total failure: Ray's max_errored_blocks and the
            # supervisor's _failed_batch fallback can turn every dropped batch into a
            # successful-looking 0-row write. Surface that as a hard error instead.
            output_rows = processed.count()
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

        except Exception:
            logger.exception("Anonymization failed")
            raise

        finally:
            shutdown.restore_handlers()

    def run_transformer(
        self,
        input_path: str | list[str],
        output_path: str,
        model_name: str,
        model_path: str | None = None,
        bucket_name: str | None = None,
        project_id: str | None = None,
        num_gpus: int | None = None,
        num_transformer_actors: int | None = None,
        batch_size: int = 512,
        gpu_batch_size: int | None = None,
        short_seq_budget: float | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        compile_model: bool | None = None,
        compile_cache_path: str | None = None,
        num_agg_actors: int | None = None,
        pre_chunked: bool = False,
        read_cpus: float = 1.0,
        flat_map_cpus: float = 1.0,
        write_cpus: float = 1.0,
        agg_num_cpus: float = 1.0,
        transformer_cpus: float | None = None,
        enable_checkpoint: bool = True,
    ) -> dict[str, Any]:
        """
        Run transformer NER job with proper document chunking.

        Documents are chunked into overlapping windows before inference,
        then predictions are aggregated back to document-level entities.

        Hardware sizing (CPU/actor knobs)
        ---------------------------------
        Ray Data runs every operator of this stage concurrently
        (read -> flat_map -> transformer actor -> BIO actor -> write) and, under
        Ray 2.55's ReservationOpResourceAllocator, must reserve a minimum CPU
        slice for each *eligible* operator at once. If those minimums sum to more
        than the cluster's CPUs, NOTHING schedules and the stage hangs at 0/1
        (``backpressured:tasks(ResourceBudget)``). Two independent levers avoid
        this on small boxes:

        1. **Fractional CPU knobs** (``read_cpus``, ``flat_map_cpus``,
           ``write_cpus``, ``agg_num_cpus``, ``transformer_cpus``) shrink each
           operator's reservation so the concurrent sum fits.
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
          ``read_cpus=flat_map_cpus=write_cpus=0.25``, ``agg_num_cpus=0.5``,
          ``transformer_cpus=0.25`` (small optional floor),
          ``num_transformer_actors=1``, ``num_agg_actors=1``,
          ``enable_checkpoint=False``.
        - **Small CPU box (e.g. C=2, G=0 — Colab CPU)**: the actor needs ~1 CPU
          and ``transformer_cpus`` also caps its torch thread count
          (``transformer.py``: ``int(transformer_cpus) or 1``), so give it most
          of the box: ``transformer_cpus=max(0.5, C-1.0)``,
          ``read_cpus=flat_map_cpus=write_cpus=0.25``, ``agg_num_cpus=0.25``,
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
            num_gpus: Number of GPU actors (auto-detect if None). Used when GPUs
                are available to determine actor pool size.
            num_transformer_actors: Number of transformer inference actors.
                If None, defaults to num_gpus when GPUs available, or ~25% of
                available CPUs in CPU-only mode.
            batch_size: Batch size for map_batches (chunks per actor call).
                Larger values reduce Ray Data dispatch overhead.
            gpu_batch_size: Batch size for HuggingFace pipeline inference
                (texts fed to GPU at once). None = auto-compute from model
                config and available GPU memory.
            chunk_size: Maximum chunk size in tokens (default: from model config)
            chunk_overlap: Overlap between chunks in tokens (default: from model config)
            compile_model: If True, apply torch.compile with mega-cache
            compile_cache_path: Path to compiled cache .bin file
            num_agg_actors: Number of CPU actors for BIO aggregation.
                If None, auto-computed as ~30% of available CPUs.
            pre_chunked: If True, input is already chunked (skip flat_map).
                Expects columns: chunk_text, text_hash, chunk_id,
                char_offset_start, patient_id.
            read_cpus: CPUs to reserve for each read_parquet task. Default 1.0
                reproduces Ray's default reservation. Lower (e.g. 0.25) to fit
                small boxes where concurrent operators contend for CPUs.
            flat_map_cpus: CPUs to reserve for each chunking flat_map task.
                Default 1.0 reproduces Ray's default reservation.
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

        Returns:
            Processing statistics dictionary
        """
        from tide2.actors import BIOAggregationActor
        from tide2.actors import create_transformer_actor
        from tide2.transformers.config import load_model_config

        from .transformer import chunk_document_row

        self._init_ray()
        start_time = time.time()

        # Resolve chunk_size/chunk_overlap from model config if not provided
        model_config = load_model_config(model_name)
        if chunk_size is None:
            chunk_size = model_config.get("CHUNK_SIZE", 512)
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
        logger.info(f"  Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

        # Resolve model path on driver (downloads if needed, validates weights)
        if model_path is None:
            from tide2.utils.gcs_resource_manager import resolve_model_path

            model_path = resolve_model_path(
                model_name=model_name,
                bucket_name=bucket_name,
                project_id=project_id,
            )
            logger.info(f"Model resolved on driver: {model_path}")

        # Create transformer actor class
        transformer_actor = create_transformer_actor(
            model_name=model_name,
            model_path=model_path,
            bucket_name=bucket_name,
            project_id=project_id,
            compile_model=compile_model,
            compile_cache_path=compile_cache_path,
            gpu_batch_size=gpu_batch_size,
            short_seq_budget=short_seq_budget,
        )

        input_pattern = self._resolve_input_pattern(input_path)
        self._ensure_output_dir(output_path)

        # Configure Ray Data checkpointing for row-level resume.
        # text_hash is always present in the raw input parquet. chunk_uid is
        # produced by chunk_document_row after ReadParquet, so it does not
        # exist in the input schema when the checkpoint filter runs.
        #
        # CRITICAL on tiny clusters (≲4 CPUs): this is THE transformer-stage
        # deadlock — fractional CPU knobs alone do NOT fix it, and disabling
        # op_resource_reservation_enabled does NOT help either. See
        # _configure_checkpoint for the full explanation; pass
        # enable_checkpoint=False on small boxes.
        ctx = ray.data.DataContext.get_current()
        _configure_checkpoint(
            ctx, enable=enable_checkpoint, output_dir=Path(output_path).resolve(), id_column="text_hash"
        )

        # Phase 1: Read documents (only columns needed for chunking + inference)
        ds: Dataset = ray.data.read_parquet(
            input_pattern,
            columns=["text_hash", "note_text", "patient_id"],
            ray_remote_args={"num_cpus": read_cpus},
        )

        # Phase 2: Chunk documents (or skip if pre-chunked)
        if pre_chunked:
            ds_chunks: Dataset = ds
            logger.info("Using pre-chunked input (skipping chunking step)")
        else:
            logger.info(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})")

            def chunk_fn(row):
                return chunk_document_row(row, chunk_size, chunk_overlap)

            ds_chunks = ds.flat_map(chunk_fn, num_cpus=flat_map_cpus)

        # Phase 3: Transformer inference (raw BIO tokens only, no aggregation)
        # Use GPU remote args with num_gpus=0 for CPU-only mode (no GPU resource request)
        ray_remote_args_transformer = get_ray_remote_args_gpu(num_gpus=0 if cpu_only_mode else 1)
        if transformer_cpus is not None:
            # Set a CPU floor for the transformer actor. Never add num_gpus here
            # in CPU mode; GPU pinning (num_gpus=1) is preserved in GPU mode.
            ray_remote_args_transformer["num_cpus"] = transformer_cpus

        ds_raw = ds_chunks.map_batches(
            transformer_actor,
            batch_size=batch_size,
            batch_format="numpy",
            compute=ray.data.ActorPoolStrategy(size=num_transformer_actors),
            **ray_remote_args_transformer,
        )

        # Phase 4: CPU BIO aggregation (runs concurrently with GPU via streaming)
        ray_remote_args_cpu = get_ray_remote_args_cpu(num_cpus=agg_num_cpus)

        ds_predictions = ds_raw.map_batches(
            BIOAggregationActor,
            batch_size=batch_size,
            batch_format="numpy",
            compute=ray.data.ActorPoolStrategy(size=num_agg_actors),
            **ray_remote_args_cpu,
        )

        # Phase 5: Write chunk-level predictions (fully streaming, no groupby)
        ds_predictions.write_parquet(output_path, compression="zstd", ray_remote_args={"num_cpus": write_cpus})

        # Clear checkpoint config to avoid leaking to subsequent pipelines
        ctx.checkpoint_config = None

        elapsed = time.time() - start_time
        logger.info(f"Transformer NER complete in {elapsed:.1f}s")

        return {
            "elapsed_seconds": round(elapsed, 2),
            "model_name": model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_gpus": num_gpus,
            "cpu_only_mode": cpu_only_mode,
            "num_transformer_actors": num_transformer_actors,
            "num_agg_actors": num_agg_actors,
            "batch_size": batch_size,
        }

    def run_reassembly(
        self,
        input_path: str | list[str],
        output_path: str,
        model_name: str,
        num_actors: int | None = None,
        batch_size: int = 500,
        num_cpus: int = 1,
    ) -> dict[str, Any]:
        """
        Run chunk-to-document reassembly via map_batches.

        Input rows are one-per-document with chunks pre-grouped into a
        chunks_json column (e.g. via BigQuery ARRAY_AGG). Required columns:
        text_hash, patient_id, note_text, chunks_json.

        Args:
            input_path: Path to parquet files with pre-grouped chunks.
            output_path: Output directory for document-level results.
            model_name: Transformer model name for recognition_metadata.
            num_actors: Number of ReassemblyActor instances (auto-detect if None).
            batch_size: Rows per batch.
            num_cpus: CPUs per actor.

        Returns:
            Processing statistics dictionary.
        """
        from tide2.actors import ReassemblyActor

        self._init_ray()
        start_time = time.time()

        if num_actors is None:
            num_actors = self._auto_num_actors()

        self._ensure_output_dir(output_path)

        logger.info("Reassembly job starting")
        logger.info(f"  Input: {input_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Actors: {num_actors}, Batch size: {batch_size}")

        input_pattern = self._resolve_input_pattern(input_path)

        ds = ray.data.read_parquet(input_pattern)
        ds = ds.map_batches(
            ReassemblyActor,
            batch_size=batch_size,
            compute=ray.data.ActorPoolStrategy(size=num_actors),
            fn_constructor_kwargs={"model_name": model_name},
            num_cpus=num_cpus,
        )
        ds.write_parquet(output_path, compression="zstd")

        elapsed = time.time() - start_time
        logger.info(f"Reassembly complete in {elapsed:.1f}s")

        return {
            "elapsed_seconds": round(elapsed, 2),
            "model_name": model_name,
            "num_actors": num_actors,
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
    ) -> dict[str, Any]:
        """
        Run the full de-identification pipeline (transformer → recognizer → anonymizer).

        Designed for small datasets. Each stage can be toggled independently.
        When a stage is skipped, its output from a previous run is expected on disk.

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
                chunk_size, chunk_overlap.
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

        Returns:
            Dictionary with per-stage statistics and output paths.
        """
        from tide2.utils.text_processing import compute_text_hash

        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # --- Intermediate paths ---
        transformer_input_path = output_path / "01_transformer_input.parquet"
        transformer_output_path = output_path / "02_transformer_output"
        recognizer_input_path = output_path / "03_recognizer_input.parquet"
        recognizer_output_path = output_path / "04_recognizer_output"
        anonymizer_input_path = output_path / "05_anonymizer_input.parquet"
        anonymizer_output_path = output_path / "06_anonymizer_output"

        llm_recognizer_output_path = output_path / "03b_llm_recognizer_output"

        # Validate llm_recognizer_mode
        valid_llm_modes = ("off", "only", "merge")
        if llm_recognizer_mode not in valid_llm_modes:
            raise ValueError(f"llm_recognizer_mode must be one of {valid_llm_modes}, got '{llm_recognizer_mode}'")

        results: dict[str, Any] = {"output_dir": str(output_path)}
        t_kw = transformer_kwargs or {}
        r_kw = recognizer_kwargs or {}
        a_kw = anonymizer_kwargs or {}
        llm_kw = llm_recognizer_kwargs or {}

        # ------------------------------------------------------------------
        # Prepare input DataFrame
        # ------------------------------------------------------------------
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

        # Write transformer input (only needs note_text, patient_id, text_hash)
        df_input[["note_text", "patient_id", "text_hash"]].to_parquet(transformer_input_path, index=False)
        logger.info(f"Pipeline input: {len(df_input)} notes")

        # ------------------------------------------------------------------
        # Phase 1: Transformer NER
        # ------------------------------------------------------------------
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
                **t_kw,
            )
            results["transformer"] = transformer_manifest
        else:
            logger.info("Pipeline phase 1/3: Transformer NER (SKIPPED)")

        # ------------------------------------------------------------------
        # Phase 2: Reassembly + Recognizer (+ optional LLM recognizer)
        # ------------------------------------------------------------------
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
                trans_files = list(transformer_output_path.glob("**/*.parquet"))
                if trans_files:
                    from .transformer import reassemble_document_predictions

                    dfs_trans = [pq.read_table(f).to_pandas() for f in trans_files]
                    df_chunks = pd.concat(dfs_trans, ignore_index=True)
                    df_rec_in = reassemble_document_predictions(
                        df_chunks=df_chunks,
                        df_notes=df_input[["text_hash", "note_text", "patient_id"]],
                        model_name=model_name,
                    )
                    if "patient_identifiers" in df_input.columns:
                        id_map = df_input.drop_duplicates(subset="text_hash").set_index("text_hash")[
                            "patient_identifiers"
                        ]
                        df_rec_in["patient_identifiers"] = df_rec_in["text_hash"].map(id_map).fillna("{}")
                    else:
                        df_rec_in["patient_identifiers"] = "{}"
                else:
                    logger.info("No transformer output found; using input data for recognizer")
                    df_rec_in = df_input.copy()
                    if "patient_identifiers" not in df_rec_in.columns:
                        df_rec_in["patient_identifiers"] = "{}"
                    if "recognizer_results_json" not in df_rec_in.columns:
                        df_rec_in["recognizer_results_json"] = "[]"

                df_rec_in.to_parquet(recognizer_input_path, index=False)
                recognizer_manifest = self.run_recognition(
                    input_path=str(recognizer_input_path),
                    output_path=str(recognizer_output_path),
                    **r_kw,
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

            # Overwrite recognizer output with merged results
            if recognizer_output_path.exists():
                shutil.rmtree(recognizer_output_path)
            recognizer_output_path.mkdir(parents=True, exist_ok=True)
            df_merged_results.to_parquet(recognizer_output_path / "merged_results.parquet", index=False)
            logger.info(f"Merged {len(df_merged_results)} notes from regex + LLM recognizers")

        elif run_recognizer:
            # Standard recognizer path (no LLM)
            logger.info("Pipeline phase 2/3: Recognizer")

            trans_files = list(transformer_output_path.glob("**/*.parquet"))
            if trans_files:
                from .transformer import reassemble_document_predictions

                dfs_trans = [pq.read_table(f).to_pandas() for f in trans_files]
                df_chunks = pd.concat(dfs_trans, ignore_index=True)
                df_rec_in = reassemble_document_predictions(
                    df_chunks=df_chunks,
                    df_notes=df_input[["text_hash", "note_text", "patient_id"]],
                    model_name=model_name,
                )
                if "patient_identifiers" in df_input.columns:
                    id_map = df_input.drop_duplicates(subset="text_hash").set_index("text_hash")["patient_identifiers"]
                    df_rec_in["patient_identifiers"] = df_rec_in["text_hash"].map(id_map).fillna("{}")
                else:
                    df_rec_in["patient_identifiers"] = "{}"
            else:
                logger.info("No transformer output found; using input data for recognizer")
                df_rec_in = df_input.copy()
                if "patient_identifiers" not in df_rec_in.columns:
                    df_rec_in["patient_identifiers"] = "{}"
                if "recognizer_results_json" not in df_rec_in.columns:
                    df_rec_in["recognizer_results_json"] = "[]"

            df_rec_in.to_parquet(recognizer_input_path, index=False)

            recognizer_manifest = self.run_recognition(
                input_path=str(recognizer_input_path),
                output_path=str(recognizer_output_path),
                **r_kw,
            )
            results["recognizer"] = recognizer_manifest
        else:
            logger.info("Pipeline phase 2/3: Recognizer (SKIPPED)")

        # ------------------------------------------------------------------
        # Phase 3: Anonymizer
        # ------------------------------------------------------------------
        if run_anonymizer:
            logger.info("Pipeline phase 3/3: Anonymizer")

            # Read recognizer output
            rec_files = list(recognizer_output_path.glob("**/*.parquet"))
            if not rec_files:
                raise FileNotFoundError(
                    f"No recognizer output found in {recognizer_output_path}. Run with run_recognizer=True first."
                )
            dfs_rec = [pq.read_table(f).to_pandas() for f in rec_files]
            df_rec = pd.concat(dfs_rec, ignore_index=True)
            df_rec.columns = df_rec.columns.str.lower()

            # Merge columns from input that the recognizer doesn't produce
            cols_to_add = [
                c
                for c in ["note_text", "patient_id", "patient_uid", "jitter"]
                if c in df_input.columns and c not in df_rec.columns
            ]
            if cols_to_add:
                df_rec = df_rec.merge(
                    df_input[["text_hash", *cols_to_add]],
                    on="text_hash",
                    how="left",
                )

            # patient_uid required by anonymizer
            if "patient_uid" not in df_rec.columns:
                df_rec["patient_uid"] = df_rec.get("patient_id", df_rec["text_hash"])

            # Always generate unique row_id for anonymizer checkpointing
            # Use fillna("None") to match SQL COALESCE(..., 'None') behavior
            df_rec["row_id"] = (df_rec["text_hash"] + ":" + df_rec["patient_uid"].fillna("None").astype(str)).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest()
            )

            df_rec.to_parquet(anonymizer_input_path, index=False)

            # Write hex keys to temp files
            salt_file = output_path / "salt.bin"
            key_file = output_path / "key.bin"
            salt_file.write_text(salt_hex)
            key_file.write_text(key_hex)

            anonymizer_manifest = self.run_anonymization(
                input_path=str(anonymizer_input_path),
                output_path=str(anonymizer_output_path),
                salt_path=str(salt_file),
                key_path=str(key_file),
                **a_kw,
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

    def shutdown(self) -> None:
        """Shutdown Ray after job completion."""
        if self._initialized:
            ray.shutdown()
            self._initialized = False
            logger.info("Ray shutdown complete")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

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
    batch_size: int = 512,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    num_cpus: int | None = None,
    object_store_gb: int | None = None,
    compile_model: bool | None = None,
    compile_cache_path: str | None = None,
    num_agg_actors: int | None = None,
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
        batch_size: Batch size for map_batches (chunks per GPU call)
        chunk_size: Maximum chunk size in tokens (default: from model config)
        chunk_overlap: Overlap between chunks in tokens (default: from model config)
        num_cpus: CPUs
        object_store_gb: Object store size GB
        compile_model: If True, apply torch.compile with mega-cache
        compile_cache_path: Path to compiled cache .bin file
        num_agg_actors: Number of CPU actors for BIO aggregation (auto if None)

    Returns:
        Processing statistics
    """
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
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            compile_model=compile_model,
            compile_cache_path=compile_cache_path,
            num_agg_actors=num_agg_actors,
        )
    finally:
        runner.shutdown()


def run_reassembly_simple(
    input_path: str,
    output_path: str,
    model_name: str,
    num_actors: int | None = None,
    batch_size: int = 500,
    num_cpus: int | None = None,
    object_store_gb: int | None = None,
) -> dict[str, Any]:
    """
    Simple function to run chunk-to-document reassembly job.

    Input parquet files must have one row per document with chunks_json
    column (pre-grouped, e.g. via BigQuery ARRAY_AGG or
    reassemble_document_predictions_local).

    Args:
        input_path: Path to parquet files with pre-grouped chunks.
        output_path: Output directory for document-level results.
        model_name: Transformer model name.
        num_actors: Number of actors.
        batch_size: Batch size.
        num_cpus: CPUs for Ray.
        object_store_gb: Object store size GB.

    Returns:
        Processing statistics.
    """
    runner = LocalJobRunner(
        num_cpus=num_cpus,
        object_store_gb=object_store_gb,
    )

    try:
        return runner.run_reassembly(
            input_path=input_path,
            output_path=output_path,
            model_name=model_name,
            num_actors=num_actors,
            batch_size=batch_size,
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
