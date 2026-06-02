"""
Shared utilities for cluster processing modules.

This module provides common functions used across all execution modes
(local, VM, cluster) to reduce code duplication.
"""

import logging
from pathlib import Path
from typing import Any

import ray

logger = logging.getLogger(__name__)

# Threshold for small dataset where count() is acceptable
SMALL_DATASET_FILE_THRESHOLD = 10

# Default segment size (files per segment) for bounded failure scope
DEFAULT_SEGMENT_SIZE = 20

# Default dashboard host (localhost for security)
DEFAULT_DASHBOARD_HOST = "127.0.0.1"


def resolve_input_files(input_glob: str | list[str]) -> list[str]:
    """
    Resolve glob pattern to list of files.

    Handles five cases:
    0. List of file paths (returned as-is)
    1. Single file path
    2. Directory path (returns all .parquet files recursively)
    3. Recursive glob pattern (e.g., "dir/**/*.parquet")
    4. Simple glob pattern (e.g., "dir/*.parquet")

    Args:
        input_glob: File path, directory path, glob pattern, or list of file paths.

    Returns:
        List of resolved file paths as strings.
    """
    if isinstance(input_glob, list):
        return [str(f) for f in input_glob]

    input_path = Path(input_glob)

    if input_path.exists() and input_path.is_file():
        return [str(input_path)]
    if input_path.exists() and input_path.is_dir():
        # Use rglob for recursive search in directories
        return sorted([str(f) for f in input_path.rglob("*.parquet")])

    # Handle glob patterns including **
    if "**" in input_glob:
        # Find the base directory (everything before **)
        base_idx = input_glob.index("**")
        base_dir = Path(input_glob[:base_idx].rstrip("/"))
        pattern = input_glob[base_idx:]  # e.g., "**/*.parquet"
        if base_dir.exists():
            return sorted([str(f) for f in base_dir.glob(pattern)])
        return []

    parent = Path(input_glob).parent
    pattern = Path(input_glob).name
    if parent.exists():
        return sorted([str(f) for f in parent.glob(pattern)])
    return []


def detect_columns(sample_file: str, required: list[str], optional: list[str]) -> list[str]:
    """
    Detect available columns from sample file.

    Matching is case-insensitive: if the file has "JITTER" and required/optional
    lists request "jitter", the actual file column name ("JITTER") is returned.

    Args:
        sample_file: Path to sample parquet file.
        required: List of required column names.
        optional: List of optional column names.

    Returns:
        List of actual column names to read from the file.

    Raises:
        ValueError: If any required columns are missing.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(sample_file)
    available_columns = set(pf.schema_arrow.names)
    lower_to_actual = {c.lower(): c for c in available_columns}

    columns = []
    for c in required:
        actual = lower_to_actual.get(c.lower())
        if actual:
            columns.append(actual)

    for c in optional:
        actual = lower_to_actual.get(c.lower())
        if actual and actual not in columns:
            columns.append(actual)

    missing = [c for c in required if c.lower() not in lower_to_actual]
    if missing:
        raise ValueError(f"Required columns not found in {sample_file}: {missing}")

    return columns


def init_ray_local(
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    object_store_memory_gb: int | None = None,
    dashboard_host: str = DEFAULT_DASHBOARD_HOST,
    metrics_port: int = 9090,
) -> None:
    """
    Initialize Ray for local/VM mode with standard configuration.

    Args:
        num_cpus: Total CPUs for Ray (default: auto-detect).
        num_gpus: Total GPUs for Ray (default: auto-detect).
        object_store_memory_gb: Object store memory in GB.
        dashboard_host: Host for Ray Dashboard.
        metrics_port: Port for Prometheus metrics.
    """
    ray_init_kwargs: dict[str, Any] = {
        "dashboard_host": dashboard_host,
        "ignore_reinit_error": True,
        "include_dashboard": True,
        "_metrics_export_port": metrics_port,
    }

    if num_cpus:
        ray_init_kwargs["num_cpus"] = num_cpus
    if num_gpus:
        ray_init_kwargs["num_gpus"] = num_gpus
    if object_store_memory_gb:
        ray_init_kwargs["object_store_memory"] = object_store_memory_gb * 1024**3

    ray.init(**ray_init_kwargs)

    logger.info(f"Ray Dashboard: http://{dashboard_host}:8265")
    logger.info(f"Prometheus metrics: http://{dashboard_host}:{metrics_port}")


def log_ray_cluster_info() -> None:
    """Log information about the Ray cluster."""
    if not ray.is_initialized():
        logger.warning("Ray is not initialized")
        return

    try:
        resources = ray.cluster_resources()
        available = ray.available_resources()

        logger.info("Ray cluster resources:")
        logger.info(f"  CPUs: {resources.get('CPU', 0):.0f} total, {available.get('CPU', 0):.0f} available")
        logger.info(f"  GPUs: {resources.get('GPU', 0):.0f} total, {available.get('GPU', 0):.0f} available")
        logger.info(f"  Memory: {resources.get('memory', 0) / 1e9:.1f} GB total")
        logger.info(f"  Object Store: {resources.get('object_store_memory', 0) / 1e9:.1f} GB")
    except Exception as e:
        logger.warning(f"Could not get Ray cluster info: {e}")
