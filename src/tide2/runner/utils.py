"""
Shared utilities for cluster processing modules.

This module provides common functions used across all execution modes
(local, VM, cluster) to reduce code duplication.
"""

import logging
import os
from pathlib import Path
from typing import Any

import ray

from tide2.utils.batch_columns import _check_deprecated_patient_uid

logger = logging.getLogger(__name__)

# Threshold for small dataset where count() is acceptable
SMALL_DATASET_FILE_THRESHOLD = 10

# Default segment size (files per segment) for bounded failure scope
DEFAULT_SEGMENT_SIZE = 20

# Default dashboard host (localhost for security)
DEFAULT_DASHBOARD_HOST = "127.0.0.1"

# CUDA allocator hint for GPU actors. expandable_segments lets PyTorch's caching
# allocator grow/shrink segments instead of pre-carving fixed blocks, reducing
# fragmentation-driven OOM headroom loss. NOTE: this is a fragmentation mitigation,
# NOT a leak fix (see the OOM-recovery fix in transformers/core.py and
# actors/transformer.py). It must be set BEFORE CUDA initializes in the worker
# process, which is why it is injected via Ray's runtime_env rather than in the
# actor __init__ (too late — CUDA is already up).
_PYTORCH_CUDA_ALLOC_CONF_DEFAULT = "expandable_segments:True"


def gpu_worker_runtime_env() -> dict[str, Any]:
    """Ray ``runtime_env`` that configures the CUDA allocator for GPU workers.

    Returns a ``{"env_vars": {...}}`` dict setting ``PYTORCH_CUDA_ALLOC_CONF`` so
    Ray applies it to every worker process before CUDA initializes. Respects an
    operator-provided ``PYTORCH_CUDA_ALLOC_CONF`` in the driver environment
    (passed through verbatim) instead of overriding it.
    """
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", _PYTORCH_CUDA_ALLOC_CONF_DEFAULT)
    return {"env_vars": {"PYTORCH_CUDA_ALLOC_CONF": alloc_conf}}


def is_uri(path: str | os.PathLike[str]) -> bool:
    """Return True for object-store paths such as ``gs://bucket/dir``."""
    return "://" in str(path)


def _filesystem_and_path(uri: str) -> tuple[Any, str]:
    from pyarrow import fs as pafs

    return pafs.FileSystem.from_uri(uri)


def output_location(path: str | os.PathLike[str], *, resolve: bool = False) -> Path | str:
    """Return an output location: the URI as a string, or a local ``Path``.

    ``Path`` mangles ``gs://bucket/out`` into ``gs:/bucket/out``, so URIs are
    kept as strings. Local paths are made absolute only when ``resolve`` is True.
    """
    if is_uri(path):
        return str(path).rstrip("/")
    return Path(path).resolve() if resolve else Path(path)


def make_output_dir(location: Path | str) -> None:
    """Create a local output directory. Object stores have no directories to create."""
    if isinstance(location, Path):
        location.mkdir(parents=True, exist_ok=True)


def join_location(location: Path | str, name: str) -> Path | str:
    """Append ``name`` to a local ``Path`` or a URI string."""
    if isinstance(location, Path):
        return location / name
    return f"{location}/{name}"


def list_parquet_files(location: Path | str, *, recursive: bool = True) -> list[str]:
    """List ``*.parquet`` files under a local directory or URI, sorted."""
    if not is_uri(location):
        base = Path(location)
        return sorted(str(f) for f in (base.rglob("*.parquet") if recursive else base.glob("*.parquet")))
    from pyarrow import fs as pafs

    filesystem, path = _filesystem_and_path(str(location))
    scheme = str(location).split("://", 1)[0]
    selector = pafs.FileSelector(path, recursive=recursive, allow_not_found=True)
    return sorted(
        f"{scheme}://{info.path}"
        for info in filesystem.get_file_info(selector)
        if info.type == pafs.FileType.File and info.path.endswith(".parquet")
    )


def read_parquet_schema(file: str | os.PathLike[str]) -> Any:
    """Read a Parquet file's Arrow schema from a local path or URI."""
    import pyarrow.parquet as pq

    if not is_uri(file):
        return pq.read_schema(file)
    filesystem, path = _filesystem_and_path(str(file))
    return pq.read_schema(path, filesystem=filesystem)


def read_text_file(location: Path | str) -> str:
    """Read a UTF-8 text file from a local path or URI."""
    if not is_uri(location):
        return Path(location).read_text(encoding="utf-8")
    filesystem, path = _filesystem_and_path(str(location))
    with filesystem.open_input_stream(path) as stream:
        return stream.read().decode("utf-8")


def write_text_file(location: Path | str, text: str) -> None:
    """Write a UTF-8 text file to a local path or URI."""
    if not is_uri(location):
        Path(location).write_text(text, encoding="utf-8")
        return
    filesystem, path = _filesystem_and_path(str(location))
    with filesystem.open_output_stream(path) as stream:
        stream.write(text.encode("utf-8"))


def _resolve_uri_input(uri: str) -> list[str]:
    """Resolve a URI to a single file or the Parquet files under a directory prefix."""
    from pyarrow import fs as pafs

    filesystem, path = _filesystem_and_path(uri)
    if filesystem.get_file_info(path).type == pafs.FileType.File:
        return [uri]
    return list_parquet_files(uri)


def resolve_input_files(input_glob: str | list[str]) -> list[str]:  # noqa: PLR0911 - one return per input kind
    """
    Resolve glob pattern to list of files.

    Handles six cases:
    0. List of file paths (returned as-is)
    1. Single file path
    2. Directory path (returns all .parquet files recursively)
    3. Recursive glob pattern (e.g., "dir/**/*.parquet")
    4. Simple glob pattern (e.g., "dir/*.parquet")
    5. Object-store URI (e.g., "gs://bucket/dir"): a file, or the .parquet files
       under a directory prefix. Glob patterns are not supported for URIs.

    Args:
        input_glob: File path, directory path, glob pattern, URI, or list of file paths.

    Returns:
        List of resolved file paths as strings.
    """
    if isinstance(input_glob, list):
        return [str(f) for f in input_glob]

    if is_uri(input_glob):
        return _resolve_uri_input(input_glob.rstrip("/"))

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

    if is_uri(sample_file):
        available_columns = set(read_parquet_schema(sample_file).names)
    else:
        available_columns = set(pq.ParquetFile(sample_file).schema_arrow.names)
    lower_to_actual = {c.lower(): c for c in available_columns}

    _check_deprecated_patient_uid(lower_to_actual, location=f"schema of {sample_file}")

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
        # Configure the CUDA allocator for GPU actors before CUDA initializes.
        "runtime_env": gpu_worker_runtime_env(),
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
