#!/usr/bin/env python3
"""
TIDE 2.0 Runner CLI.

Commands:
    run             Run a processing job (recognition, anonymization, transformer)

Usage Examples:
    # Run recognition locally
    tide2-runner run recognizer -i ./data/input -o ./data/output

    # Run with more resources
    tide2-runner run recognizer -i gs://bucket/input -o gs://bucket/output \
        --num-cpus 224 --num-actors 200
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_run(args: argparse.Namespace) -> None:
    """Run a job."""
    from tide2.runner.local_runner import LocalJobRunner

    # Validate required fields (may come from CLI or config)
    if not args.input:
        print("Error: --input is required (via CLI or config file)")
        sys.exit(1)
    if not args.output:
        print("Error: --output is required (via CLI or config file)")
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    runner = LocalJobRunner(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        object_store_gb=args.object_store_gb,
        include_dashboard=getattr(args, "include_dashboard", False),
    )

    dry_run = getattr(args, "dry_run", False)

    # Collect optional kwargs — only pass if explicitly set so runner uses its defaults
    optional_kwargs: dict = {}
    for attr, key in [
        ("batch_size", "batch_size"),
        ("batch_timeout", "batch_timeout"),
        ("cpus_per_actor", "num_cpus"),
        ("read_parallelism", "read_parallelism"),
        ("read_cpus", "read_cpus"),
        ("read_op_min_num_blocks", "read_op_min_num_blocks"),
        ("target_max_block_size_mb", "target_max_block_size_mb"),
        ("target_min_block_size_mb", "target_min_block_size_mb"),
        ("worker_num_cpus", "worker_num_cpus"),
        ("write_cpus", "write_cpus"),
        ("enable_checkpoint", "enable_checkpoint"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            optional_kwargs[key] = val

    try:
        if args.job_type == "recognizer":
            result = runner.run_recognition(
                input_path=args.input,
                output_path=args.output,
                num_actors=args.num_actors,
                dry_run=dry_run,
                **optional_kwargs,
            )
        elif args.job_type == "anonymizer":
            if not args.salt or not args.key:
                print("Error: --salt and --key are required for anonymizer jobs")
                sys.exit(1)

            result = runner.run_anonymization(
                input_path=args.input,
                output_path=args.output,
                salt_path=args.salt,
                key_path=args.key,
                num_actors=args.num_actors,
                acc_num_salt=args.acc_num_salt,
                acc_num_study_id=args.acc_num_study_id,
                jitter_required=getattr(args, "jitter_required", False),
                dry_run=dry_run,
                **optional_kwargs,
            )
        elif args.job_type == "transformer":
            if not args.model:
                print("Error: --model is required for transformer jobs")
                sys.exit(1)
            transformer_kwargs: dict = {}
            for attr, key in [
                ("num_gpus", "num_gpus"),
                ("batch_size", "batch_size"),
                ("gpu_batch_size", "gpu_batch_size"),
                ("model_path", "model_path"),
                ("bucket_name", "bucket_name"),
                ("project_id", "project_id"),
                ("chunk_size", "chunk_size"),
                ("chunk_overlap", "chunk_overlap"),
                ("compile_cache_path", "compile_cache_path"),
                ("num_agg_actors", "num_agg_actors"),
                ("short_seq_budget", "short_seq_budget"),
                ("read_cpus", "read_cpus"),
                ("flat_map_cpus", "flat_map_cpus"),
                ("write_cpus", "write_cpus"),
                ("agg_num_cpus", "agg_num_cpus"),
                ("transformer_cpus", "transformer_cpus"),
                ("enable_checkpoint", "enable_checkpoint"),
            ]:
                val = getattr(args, attr, None)
                if val is not None:
                    transformer_kwargs[key] = val
            if getattr(args, "compile_model", False):
                transformer_kwargs["compile_model"] = True
            if getattr(args, "pre_chunked", False):
                transformer_kwargs["pre_chunked"] = True
            result = runner.run_transformer(
                input_path=args.input,
                output_path=args.output,
                model_name=args.model,
                **transformer_kwargs,
            )
        elif args.job_type == "reassembly":
            if not args.model:
                print("Error: --model is required for reassembly jobs")
                sys.exit(1)

            reassembly_kwargs: dict = {}
            for attr, key in [
                ("num_actors", "num_actors"),
                ("batch_size", "batch_size"),
                ("cpus_per_actor", "num_cpus"),
            ]:
                val = getattr(args, attr, None)
                if val is not None:
                    reassembly_kwargs[key] = val

            result = runner.run_reassembly(
                input_path=args.input,
                output_path=args.output,
                model_name=args.model,
                **reassembly_kwargs,
            )
        elif args.job_type == "llm-recognizer":
            if not args.project_id:
                print("Error: --project-id is required for llm-recognizer jobs")
                sys.exit(1)

            llm_kwargs: dict = {}
            for attr, key in [
                ("model", "model_name"),
                ("provider_type", "provider_type"),
                ("context_length", "context_length"),
                ("max_tokens", "max_tokens"),
                ("temperature", "temperature"),
                ("region", "region"),
                ("endpoint_id", "endpoint_id"),
                ("max_retries", "max_retries"),
                ("num_actors", "num_actors"),
                ("batch_size", "batch_size"),
                ("batch_timeout", "batch_timeout"),
                ("cpus_per_actor", "num_cpus"),
                ("read_parallelism", "read_parallelism"),
                ("read_cpus", "read_cpus"),
                ("read_op_min_num_blocks", "read_op_min_num_blocks"),
                ("target_max_block_size_mb", "target_max_block_size_mb"),
                ("target_min_block_size_mb", "target_min_block_size_mb"),
                ("write_cpus", "write_cpus"),
                ("prompt_name", "prompt_name"),
            ]:
                val = getattr(args, attr, None)
                if val is not None:
                    llm_kwargs[key] = val

            result = runner.run_llm_recognition(
                input_path=args.input,
                output_path=args.output,
                project_id=args.project_id,
                dry_run=dry_run,
                **llm_kwargs,
            )
        elif args.job_type == "pipeline":
            if not args.model:
                print("Error: --model is required for pipeline jobs")
                sys.exit(1)

            # Build per-stage kwargs from CLI flags
            t_kw: dict = {}
            for attr, key in [
                ("num_gpus", "num_gpus"),
                ("bucket_name", "bucket_name"),
                ("project_id", "project_id"),
                ("chunk_size", "chunk_size"),
                ("chunk_overlap", "chunk_overlap"),
                ("batch_size", "batch_size"),
                ("model_path", "model_path"),
                ("compile_cache_path", "compile_cache_path"),
                ("num_agg_actors", "num_agg_actors"),
                ("short_seq_budget", "short_seq_budget"),
                ("read_cpus", "read_cpus"),
                ("flat_map_cpus", "flat_map_cpus"),
                ("write_cpus", "write_cpus"),
                ("agg_num_cpus", "agg_num_cpus"),
                ("transformer_cpus", "transformer_cpus"),
                ("enable_checkpoint", "enable_checkpoint"),
            ]:
                val = getattr(args, attr, None)
                if val is not None:
                    t_kw[key] = val
            if getattr(args, "compile_model", False):
                t_kw["compile_model"] = True

            r_kw: dict = {}
            for attr, key in [
                ("num_actors", "num_actors"),
                ("batch_size", "batch_size"),
                ("batch_timeout", "batch_timeout"),
                ("cpus_per_actor", "num_cpus"),
                ("read_cpus", "read_cpus"),
                ("worker_num_cpus", "worker_num_cpus"),
                ("write_cpus", "write_cpus"),
                ("enable_checkpoint", "enable_checkpoint"),
            ]:
                val = getattr(args, attr, None)
                if val is not None:
                    r_kw[key] = val
            cpu_bs = getattr(args, "cpu_batch_size", None)
            if cpu_bs is not None:
                r_kw["batch_size"] = cpu_bs

            a_kw: dict = {}
            if getattr(args, "jitter_required", False):
                a_kw["jitter_required"] = True
            for attr, key in [
                ("num_actors", "num_actors"),
                ("batch_size", "batch_size"),
                ("cpus_per_actor", "num_cpus"),
                ("acc_num_salt", "acc_num_salt"),
                ("acc_num_study_id", "acc_num_study_id"),
                ("read_cpus", "read_cpus"),
                ("worker_num_cpus", "worker_num_cpus"),
                ("write_cpus", "write_cpus"),
                ("enable_checkpoint", "enable_checkpoint"),
            ]:
                val = getattr(args, attr, None)
                if val is not None:
                    a_kw[key] = val
            if cpu_bs is not None:
                a_kw["batch_size"] = cpu_bs

            # Build LLM recognizer kwargs for pipeline
            llm_mode = getattr(args, "llm_recognizer_mode", "off")
            llm_kw: dict = {}
            if llm_mode != "off":
                # project_id: prefer --llm-project-id, fall back to --project-id
                llm_project_id = getattr(args, "llm_project_id", None) or args.project_id
                if not llm_project_id:
                    print("Error: --llm-project-id or --project-id is required when using --llm-recognizer-mode")
                    sys.exit(1)
                llm_kw["project_id"] = llm_project_id

                # model_name: prefer --llm-model, fall back to default
                llm_model = getattr(args, "llm_model", None)
                if llm_model:
                    llm_kw["model_name"] = llm_model

                for attr, key in [
                    ("provider_type", "provider_type"),
                    ("context_length", "context_length"),
                    ("max_tokens", "max_tokens"),
                    ("temperature", "temperature"),
                    ("region", "region"),
                    ("endpoint_id", "endpoint_id"),
                    ("max_retries", "max_retries"),
                    ("num_actors", "num_actors"),
                    ("batch_size", "batch_size"),
                    ("batch_timeout", "batch_timeout"),
                    ("prompt_name", "prompt_name"),
                ]:
                    val = getattr(args, attr, None)
                    if val is not None:
                        llm_kw[key] = val

            result = runner.run_pipeline(
                input_data=args.input,
                output_dir=args.output,
                model_name=args.model,
                run_transformer=getattr(args, "run_transformer", True),
                run_recognizer=getattr(args, "run_recognizer", True),
                run_anonymizer=getattr(args, "run_anonymizer", True),
                produce_visualizer_json=getattr(args, "produce_visualizer_json", False),
                salt_hex=getattr(args, "salt_hex", "00" * 32),
                key_hex=getattr(args, "key_hex", "11" * 32),
                transformer_kwargs=t_kw,
                recognizer_kwargs=r_kw,
                anonymizer_kwargs=a_kw,
                llm_recognizer_mode=llm_mode,
                llm_recognizer_kwargs=llm_kw if llm_kw else None,
            )
        else:
            print(f"Unknown job type: {args.job_type}")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        if result.get("dry_run"):
            print("DRY RUN COMPLETE - No processing performed")
        else:
            print("JOB COMPLETE")
        print(f"{'=' * 60}")
        for k, v in result.items():
            if isinstance(v, int):
                print(f"  {k}: {v:,}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            elif isinstance(v, list):
                print(f"  {k}: {', '.join(str(x) for x in v)}")
            else:
                print(f"  {k}: {v}")

    finally:
        runner.shutdown()


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="tide2-runner",
        description="TIDE 2.0 Runner - Run recognition/anonymization jobs on a single node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run recognition locally
  tide2-runner run recognizer -i ./data/input -o ./data/output

  # Run with GCS I/O and more resources
  tide2-runner run recognizer -i gs://bucket/input -o gs://bucket/output \\
      --num-cpus 224 --num-actors 200
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # =========================================================================
    # run command
    # =========================================================================
    run_p = subparsers.add_parser(
        "run",
        help="Run a processing job",
        description="Run recognition, anonymization, or transformer job",
    )
    run_p.add_argument(
        "job_type",
        choices=["recognizer", "anonymizer", "transformer", "reassembly", "pipeline", "llm-recognizer"],
        help="Type of job to run",
    )
    run_p.add_argument("--config", "-c", help="Path to YAML config file (CLI flags override config values)")
    run_p.add_argument("--input", "-i", help="Input path (local dir or gs://)")
    run_p.add_argument("--output", "-o", help="Output path")
    run_p.add_argument("--num-actors", type=int, help="Number of actors (auto-detect if not set)")
    run_p.add_argument("--batch-size", type=int, help="Batch size per actor (default: 150 recognizer, 200 anonymizer)")
    run_p.add_argument("--batch-timeout", type=int, help="Batch timeout in seconds (default: 120, recognizer only)")
    run_p.add_argument("--num-cpus", type=int, help="Total CPUs for Ray cluster")
    run_p.add_argument("--num-gpus", type=int, help="Number of GPUs (transformer jobs)")
    run_p.add_argument(
        "--gpu-batch-size",
        type=int,
        help="GPU batch size for HF pipeline inference (transformer jobs, auto-computed if not set)",
    )
    run_p.add_argument("--object-store-gb", type=int, help="Object store memory in GB")
    run_p.add_argument("--cpus-per-actor", type=int, help="CPUs per actor (default: 2)")
    run_p.add_argument("--read-parallelism", type=int, help="Number of read output blocks")
    run_p.add_argument("--read-cpus", type=float, help="CPUs per read task (default: 0.25)")
    run_p.add_argument("--read-op-min-num-blocks", type=int, help="Min read output blocks (default: 200)")
    run_p.add_argument("--target-max-block-size-mb", type=int, help="Max block size in MB (default: 128)")
    run_p.add_argument("--target-min-block-size-mb", type=int, help="Min block size in MB (default: 1)")
    run_p.add_argument(
        "--worker-num-cpus",
        type=float,
        help="CPUs per supervisor worker actor (recognizer/anonymizer/pipeline). "
        "Lower (with --cpus-per-actor) to fit small boxes; default: Ray default (1)",
    )
    run_p.add_argument(
        "--write-cpus",
        type=float,
        help="CPUs per write_parquet task (recognizer/anonymizer/transformer/llm-recognizer/pipeline, default: 1.0)",
    )
    run_p.add_argument(
        "--flat-map-cpus",
        type=float,
        help="CPUs per chunking flat_map task (transformer/pipeline jobs, default: 1.0)",
    )
    run_p.add_argument(
        "--agg-num-cpus",
        type=float,
        help="CPUs per BIO aggregation actor (transformer/pipeline jobs, default: 1.0)",
    )
    run_p.add_argument(
        "--transformer-cpus",
        type=float,
        help="CPU floor for the transformer actor (transformer/pipeline jobs). On CPU-only "
        "small boxes set ~(total CPUs - 1); default: Ray default (0 GPU mode, 1 CPU mode)",
    )
    run_p.add_argument(
        "--no-checkpoint",
        dest="enable_checkpoint",
        action="store_false",
        default=None,
        help="Disable Ray Data row-level checkpointing (recognizer/anonymizer/transformer/"
        "pipeline). REQUIRED on tiny clusters (≲4 CPUs, e.g. Colab): the checkpoint shuffle "
        "deadlocks Ray's reservation allocator. Trades resume capability, not correctness.",
    )
    run_p.add_argument("--model", help="Model name (required for transformer jobs)")
    run_p.add_argument("--model-path", help="Explicit local path to model (transformer jobs)")
    run_p.add_argument("--bucket-name", help="GCS bucket for model loading (transformer jobs)")
    run_p.add_argument("--project-id", help="GCP project ID for model loading (transformer jobs)")
    run_p.add_argument("--chunk-size", type=int, help="Max chunk size in tokens (transformer jobs, default: 512)")
    run_p.add_argument(
        "--chunk-overlap", type=int, help="Overlap between chunks in tokens (transformer jobs, default: 40)"
    )
    run_p.add_argument(
        "--compile-model", action="store_true", help="Apply torch.compile with mega-cache (transformer jobs)"
    )
    run_p.add_argument("--compile-cache-path", help="Path to compiled cache .bin file (transformer jobs)")
    run_p.add_argument(
        "--pre-chunked", action="store_true", help="Input is pre-chunked, skip chunking step (transformer jobs only)"
    )
    run_p.add_argument(
        "--num-agg-actors",
        type=int,
        help="Number of CPU actors for BIO aggregation (transformer jobs, auto-computed if not set)",
    )
    run_p.add_argument(
        "--short-seq-budget",
        type=float,
        help="Memory budget fraction for short sequences (transformer jobs, auto-computed from GPU VRAM if not set)",
    )
    run_p.add_argument("--salt", help="Path to salt file (required for anonymizer jobs)")
    run_p.add_argument("--key", help="Path to key file (required for anonymizer jobs)")
    run_p.add_argument("--acc-num-salt", help="Salt for accession number hashing (anonymizer jobs)")
    run_p.add_argument("--acc-num-study-id", help="Study ID for accession number hashing (anonymizer jobs)")
    run_p.add_argument(
        "--jitter-required",
        action="store_true",
        help="Fail notes that have no jitter value instead of computing one (anonymizer/pipeline jobs)",
    )
    # Pipeline-specific arguments
    run_p.add_argument(
        "--no-transformer",
        dest="run_transformer",
        action="store_false",
        help="Skip transformer stage (pipeline jobs, default: run)",
    )
    run_p.add_argument(
        "--no-recognizer",
        dest="run_recognizer",
        action="store_false",
        help="Skip recognizer stage (pipeline jobs, default: run)",
    )
    run_p.add_argument(
        "--no-anonymizer",
        dest="run_anonymizer",
        action="store_false",
        help="Skip anonymizer stage (pipeline jobs, default: run)",
    )
    run_p.add_argument(
        "--produce-visualizer-json",
        action="store_true",
        help="Write JSON files for tide2-visualizer (pipeline jobs)",
    )
    run_p.add_argument("--salt-hex", default="00" * 32, help="Hex-encoded FPE salt (pipeline jobs)")
    run_p.add_argument("--key-hex", default="11" * 32, help="Hex-encoded FPE key (pipeline jobs)")
    run_p.add_argument("--cpu-batch-size", type=int, help="Batch size for CPU actors (pipeline jobs, default: 100)")
    # LLM recognizer arguments (standalone and pipeline)
    run_p.add_argument(
        "--provider-type",
        default=None,
        help="LLM provider type (default: google) — llm-recognizer and pipeline jobs",
    )
    run_p.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="LLM context window in tokens (default: 1048576) — llm-recognizer and pipeline jobs",
    )
    run_p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum LLM output tokens (default: 16384) — llm-recognizer and pipeline jobs",
    )
    run_p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature for response generation (default: 0.0) — llm-recognizer and pipeline jobs",
    )
    run_p.add_argument(
        "--region",
        default=None,
        help="Cloud region for LLM API (default: us-central1) — llm-recognizer and pipeline jobs",
    )
    run_p.add_argument(
        "--endpoint-id",
        type=int,
        default=None,
        help="Vertex AI endpoint ID — llm-recognizer and pipeline jobs",
    )
    run_p.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum retry attempts for failed LLM requests (default: 3) — llm-recognizer and pipeline jobs",
    )
    # Pipeline LLM recognizer arguments
    run_p.add_argument(
        "--llm-recognizer-mode",
        choices=["off", "only", "merge"],
        default="off",
        help="LLM recognizer mode for pipeline: off, only (replaces transformer+recognizer), "
        "merge (combine with regex recognizer) (default: off)",
    )
    run_p.add_argument(
        "--llm-project-id",
        default=None,
        help="GCP project ID for LLM API (pipeline jobs, defaults to --project-id)",
    )
    run_p.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name for pipeline jobs (default: gemini-2.5-flash)",
    )
    run_p.add_argument(
        "--prompt-name",
        default=None,
        help="LLM prompt name in resources/llm_prompts/ or path to a prompt directory (default: phi_detection)",
    )
    run_p.add_argument("--dry-run", action="store_true", help="Validate setup without processing")
    run_p.add_argument("--include-dashboard", action="store_true", help="Enable Ray dashboard (port 8265)")
    run_p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )
    run_p.set_defaults(func=cmd_run)

    # Parse and execute
    args = parser.parse_args()

    # If --config provided on the 'run' command, load YAML and backfill unset args
    if args.command == "run" and getattr(args, "config", None):
        _apply_config(args)

    args.func(args)


def _apply_config(args: argparse.Namespace) -> None:
    """Load YAML config and set any arg that wasn't provided on the command line."""
    import yaml

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    with config_path.open() as f:
        config = yaml.safe_load(f) or {}

    # Map YAML keys (underscore) to argparse dest names
    # YAML uses the same names as argparse dest (e.g. num_actors, batch_size)
    for key, value in config.items():
        current = getattr(args, key, None)
        # Only backfill if the CLI didn't set it (None for optional args, False for flags)
        if current is None or (isinstance(current, bool) and not current and isinstance(value, bool)):
            setattr(args, key, value)


if __name__ == "__main__":
    main()
