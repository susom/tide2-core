"""
Shared utilities for cluster processing modules.

This module provides common functions used across all execution modes
(local, VM, cluster) to reduce code duplication.
"""

import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import ray

from tide2.utils.batch_columns import check_no_patient_uid

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


def remove_stage_dir(location: Path | str, expected_name: str) -> None:
    """Delete one stage folder (local path or URI) so that it can be rebuilt; a missing folder is fine.

    Raises:
        ValueError: If the last path part is not ``expected_name``, the folder is a symlink,
            or a URI has no bucket part. Nothing is deleted in these cases.
    """
    text = str(location).rstrip("/")
    if text.rsplit("/", 1)[-1] != expected_name:
        raise ValueError(f"Refusing to delete {text!r}: the folder is not named {expected_name!r}.")
    if not is_uri(text):
        folder = Path(text)
        if folder.is_symlink():
            raise ValueError(f"Refusing to delete {text!r}: it is a symlink.")
        if folder.is_dir():
            shutil.rmtree(folder)
        return
    from pyarrow import fs as pafs

    filesystem, path = _filesystem_and_path(text)
    if "/" not in path.strip("/"):
        raise ValueError(f"Refusing to delete {text!r}: it has no bucket part.")
    if filesystem.get_file_info(path).type == pafs.FileType.Directory:
        filesystem.delete_dir(path)


def open_parquet_dataset(files: list[str]) -> Any:
    """Open local Parquet files, or files of one object store (``gs://``), as one pyarrow dataset.

    ``pads.dataset`` cannot resolve a URI from a list of strings, so the filesystem is
    resolved here and the scheme is removed from each path.

    Raises:
        ValueError: If the files mix a local path with a URI, or use different URI schemes.
    """
    import pyarrow.dataset as pads

    schemes = {str(f).split("://", 1)[0] if is_uri(f) else "" for f in files}
    if len(schemes) > 1:
        raise ValueError(f"Input files must all be on one filesystem, got: {sorted(s or 'local' for s in schemes)}")
    if "" in schemes or not files:
        return pads.dataset(files, format="parquet")
    filesystem, _ = _filesystem_and_path(str(files[0]))
    paths = [str(f).split("://", 1)[1] for f in files]
    return pads.dataset(paths, format="parquet", filesystem=filesystem)


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

    check_no_patient_uid(lower_to_actual, f"schema of {sample_file}")

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


_SAFE_FILE_STEM = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _file_stem(sample_id: str) -> str:
    """Return ``sample_id`` as a file name part; an id that is not a plain token becomes its sha256."""
    if _SAFE_FILE_STEM.fullmatch(sample_id):
        return sample_id
    return hashlib.sha256(sample_id.encode()).hexdigest()


def _sample_id(row: dict[str, Any]) -> str:
    """Return the file-name-safe id of a row: its ``row_id``, or the legacy ``text_hash``."""
    for column in ("row_id", "text_hash"):
        if row.get(column) is not None:
            return _file_stem(str(row[column]))
    return "unknown"


def _parquet_files_under(folder: Path) -> list[Path]:
    return sorted(folder.glob("**/*.parquet")) if folder.is_dir() else []


def _text_by_hash(transformer_files: list[Path]) -> dict[str, str]:
    """Map ``text_hash`` to ``note_text`` from the transformer output, when it has both columns."""
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq

    if not transformer_files:
        return {}
    names = pq.read_schema(transformer_files[0]).names
    if "note_text" not in names or "text_hash" not in names:
        return {}
    table = pads.dataset(transformer_files).to_table(columns=["text_hash", "note_text"])
    text_map: dict[str, str] = {}
    for row in table.to_pylist():
        if row["text_hash"] is not None and row["text_hash"] not in text_map:
            text_map[row["text_hash"]] = row["note_text"] or ""
    return text_map


def _write_recognizer_json(files: list[Path], text_map: dict[str, str], out_dir: Path) -> int:
    import pyarrow.dataset as pads

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in pads.dataset(files).to_table().to_pylist():
        sample_id = _sample_id(row)
        text_hash = row.get("text_hash")
        results_json = row.get("recognizer_results_json") or "[]"
        data = {
            "key": sample_id,
            "text_hash": text_hash,
            "row_id": row.get("row_id"),
            "value": row.get("note_text") or text_map.get(str(text_hash), ""),
            "recognizer_results": json.loads(results_json),
        }
        (out_dir / f"{sample_id}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
    return count


def _write_anonymizer_json(files: list[Path], out_dir: Path) -> int:
    import pyarrow.dataset as pads

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in pads.dataset(files).to_table().to_pylist():
        sample_id = _sample_id(row)
        text = next(
            (
                str(row[col])
                for col in ("anonymized_note_text", "deid_note_text", "anonymized_text", "note_text")
                if row.get(col) is not None
            ),
            "",
        )
        items: list[Any] = []
        for col in ("anonymizer_results_json", "items", "anonymizer_results"):
            if row.get(col) is not None:
                items = json.loads(row[col]) if isinstance(row[col], str) else list(row[col])
                break
        data = {"text": text, "items": items}
        (out_dir / f"{sample_id}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
    return count


def write_visualizer_json(
    output_dir: str | os.PathLike[str], json_dir: str | os.PathLike[str] | None = None
) -> dict[str, int]:
    """Write the JSON files that ``tide2-visualizer`` reads, from a finished pipeline run.

    Reads the stage folders under ``output_dir`` (a local directory) and writes
    ``cli_recognizer_json/`` and ``cli_anonymizer_json/`` into ``json_dir``
    (default: ``output_dir``). The recognizer JSON comes from the folder the anonymizer
    read: ``merged_recognizer_output``, else ``llm_recognizer_output``, else
    ``recognizer_output``. A streamed run has no recognizer folder, so only the
    anonymizer JSON is written; each skipped part is logged.

    Args:
        output_dir: The ``output_dir`` that was given to ``run_pipeline``.
        json_dir: Where to create the two JSON folders. Defaults to ``output_dir``.

    Returns:
        The number of files written: ``{"recognizer_json": n, "anonymizer_json": m}``.

    Raises:
        ValueError: If ``output_dir`` is a URI or is not a directory.
    """
    # Deferred: local_runner imports this module.
    from tide2.runner.local_runner import STAGE_DIRS

    if is_uri(output_dir):
        raise ValueError(f"write_visualizer_json needs a local output_dir, got {output_dir!r}.")
    root = Path(output_dir)
    if not root.is_dir():
        raise ValueError(f"output_dir is not a directory: {root}")
    target = root if json_dir is None else Path(json_dir)

    counts = {"recognizer_json": 0, "anonymizer_json": 0}

    recognizer_files: list[Path] = []
    for stage in ("merged_recognizer", "llm_recognizer", "recognizer"):
        recognizer_files = _parquet_files_under(root / STAGE_DIRS[stage])
        if recognizer_files:
            break
    if recognizer_files:
        text_map = _text_by_hash(_parquet_files_under(root / STAGE_DIRS["transformer"]))
        counts["recognizer_json"] = _write_recognizer_json(recognizer_files, text_map, target / "cli_recognizer_json")
    else:
        logger.info("No recognizer output under %s: recognizer JSON skipped.", root)

    anonymizer_files = _parquet_files_under(root / STAGE_DIRS["anonymizer"])
    if anonymizer_files:
        counts["anonymizer_json"] = _write_anonymizer_json(anonymizer_files, target / "cli_anonymizer_json")
    else:
        logger.info("No anonymizer output under %s: anonymizer JSON skipped.", root)

    logger.info("Visualizer JSON written to %s: %s", target, counts)
    return counts
