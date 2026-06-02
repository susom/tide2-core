"""
GCS Resource Path Resolver for TIDE.

This module provides utilities for resolving resource paths, including automatic
download and local caching of GCS resources (models, patient vaults).

Environment Variables:
    TIDE_CACHE_DIR: Custom cache directory location (default: $HOME/.cache/tide2)
    GCP_PROJECT_ID: GCP project ID for GCS operations
"""

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Default cache directory: $HOME/.cache/tide2
# Can be overridden with TIDE_CACHE_DIR environment variable
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tide2"


def get_cache_dir() -> Path:
    """
    Get the TIDE cache directory path.

    Returns cache directory from TIDE_CACHE_DIR env var, or default location.
    Creates the directory if it doesn't exist.

    Returns:
        Path to the TIDE cache directory
    """
    cache_dir_str = os.getenv("TIDE_CACHE_DIR")

    if cache_dir_str:
        cache_dir = Path(cache_dir_str).expanduser().resolve()
    else:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS bucket path (starts with gs://)."""
    return path.startswith("gs://")


def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """
    Parse GCS path into bucket name and prefix.

    Args:
        gcs_path: GCS path like 'gs://bucket-name/path/to/resource'

    Returns:
        Tuple of (bucket_name, blob_prefix)
    """
    if not is_gcs_path(gcs_path):
        raise ValueError(f"Not a valid GCS path: {gcs_path}")

    path_without_scheme = gcs_path[5:]  # Remove 'gs://'
    parts = path_without_scheme.split("/", 1)
    bucket_name = parts[0]
    blob_prefix = parts[1] if len(parts) > 1 else ""

    return bucket_name, blob_prefix


def download_from_gcs(
    gcs_path: str, resource_type: str, project_id: str | None, local_target_path: Path | None = None
) -> Path:
    """
    Download a resource from GCS to local cache.

    Args:
        gcs_path: GCS path like 'gs://bucket/path/to/resource'
        resource_type: Resource type ('models', 'vaults', 'data') for cache organization
        project_id: GCP project ID
        local_target_path: Optional specific local path to download to

    Returns:
        Path to the cached resource
    """
    from tide2.utils.gcs_connector import GCSConnector

    bucket_name, blob_prefix = parse_gcs_path(gcs_path)

    # Determine local cache path
    if local_target_path is None:
        cache_dir = get_cache_dir()
        local_path = cache_dir / resource_type / bucket_name / blob_prefix
    else:
        local_path = local_target_path

    # Check if already cached
    if local_path.exists():
        # For files, check if it exists and has content
        if local_path.is_file() and local_path.stat().st_size > 0:
            logger.info(f"Using cached file: {local_path}")
            return local_path
        # For directories, check if it exists and has content
        if local_path.is_dir():
            try:
                # Check if directory has any files (recursive check)
                files_in_dir = list(local_path.rglob("*"))
                if files_in_dir and any(f.is_file() for f in files_in_dir):
                    logger.info(
                        f"Using cached directory: {local_path} (contains {len([f for f in files_in_dir if f.is_file()])} files)"
                    )
                    return local_path
                logger.info(f"Cached directory is empty, re-downloading: {local_path}")
            except Exception as e:
                logger.warning(f"Error checking cached directory {local_path}: {e}")

    # Download from GCS
    logger.info(f"Downloading {gcs_path} to cache...")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure project_id is available
    if project_id is None:
        project_id = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT"))
        if project_id is None:
            raise ValueError(
                "GCP project ID required for GCS download. "
                "Set GCP_PROJECT_ID environment variable or pass project_id parameter."
            )

    connector = GCSConnector.create_with_optimal_settings(project_id=project_id, max_retries=3)

    # Use a temporary directory for downloading to avoid path structure issues
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # First, try to check if it's a single file
            bucket = connector.client.bucket(bucket_name)
            blob = bucket.blob(blob_prefix)

            # Check if blob exists as a single file
            if blob.exists():
                logger.info(f"Downloading single file: {blob_prefix}")
                temp_file_path = temp_path / Path(blob_prefix).name
                success = connector.download_single_file(
                    bucket_name=bucket_name, blob_name=blob_prefix, local_path=str(temp_file_path)
                )

                if not success:
                    raise Exception(f"Failed to download single file from {gcs_path}")

                # Move the single file to the final location
                shutil.move(str(temp_file_path), str(local_path))
                logger.info(f"Moved single file to: {local_path}")

            else:
                # If not a single file, treat as directory
                logger.info(f"Downloading directory with prefix: {blob_prefix}")
                results = connector.download_directory(
                    bucket_name=bucket_name, prefix=blob_prefix, local_directory=str(temp_path)
                )

                if not results or all(not success for success in results.values()):
                    raise Exception(f"Failed to download from {gcs_path}")

                # Find all downloaded files in the temp directory and move them to final location
                downloaded_files = list(temp_path.rglob("*"))
                actual_files = [f for f in downloaded_files if f.is_file()]

                if not actual_files:
                    raise Exception(f"No files were downloaded from {gcs_path}")

                # Create the target directory if it doesn't exist
                local_path.mkdir(parents=True, exist_ok=True)

                # Move all files to the controlled final location, preserving directory structure
                for file_path in actual_files:
                    # Get relative path from temp directory to preserve structure
                    rel_path = file_path.relative_to(temp_path)
                    target_file = local_path / rel_path

                    # Create subdirectories if needed
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file_path), str(target_file))
                    logger.debug(f"Moved {rel_path} to {target_file}")

                logger.info(f"Moved {len(actual_files)} files to controlled location: {local_path}")

        except Exception as e:
            logger.error(f"Error downloading from GCS: {e}")
            # Clean up incomplete download
            if local_path.exists():
                if local_path.is_dir():
                    shutil.rmtree(local_path)
                else:
                    local_path.unlink()
            raise Exception(f"Failed to download from {gcs_path}")

    logger.info(f"Successfully cached resource at: {local_path}")
    return local_path


def resolve_resource_path(path: str | None, resource_type: str = "data", project_id: str | None = None) -> str | None:
    """
    Resolve a resource path, downloading from GCS if needed.

    - If path is None, returns None
    - If path starts with 'gs://', downloads to cache and returns local path
    - Otherwise, returns the path as-is (local path)

    Args:
        path: Resource path (local, GCS, or None)
        resource_type: Resource type for cache organization ('models', 'vaults', 'data')
        project_id: GCP project ID (auto-detected from env if not provided)

    Returns:
        Local path string, or None if input was None
    """
    if path is None:
        return None

    if not is_gcs_path(path):
        # Local path, return as-is
        return path

    # GCS path - download to cache
    if project_id is None:
        project_id = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT"))
        if project_id is None:
            raise ValueError(
                "GCP project ID required for GCS paths. "
                "Set GCP_PROJECT_ID environment variable or pass project_id parameter."
            )

    local_path = download_from_gcs(path, resource_type, project_id)
    return str(local_path)


_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def validate_model_directory(model_dir: Path) -> bool:
    """Check that a model directory has config.json and at least one weight file."""
    if not (model_dir / "config.json").is_file():
        return False
    return any((model_dir / f).is_file() for f in _WEIGHT_FILES)


def resolve_model_path(
    model_name: str,
    bucket_name: str | None = None,
    project_id: str | None = None,
    allow_huggingface_download: bool = True,
    local_dir: str | None = None,
) -> str:
    """
    Resolve model path using standardized bucket structure.

    First tries to find model locally, then attempts GCS download, then
    optionally falls back to HuggingFace Hub.

    Expected GCS structure: gs://<bucket_name>/tide2/resources/models/<model_name>
    Local structure: $HOME/.cache/tide2/resources/models/<model_name>/

    Args:
        model_name: Name of the model (e.g., "BioClinical-ModernBERT-large")
        bucket_name: GCS bucket name (optional)
        project_id: GCP project ID for GCS operations
        allow_huggingface_download: If True, attempt to download from HuggingFace Hub
            when the model is not found locally or via GCS. Uses model_name as the
            HuggingFace repo ID. Default: True.
        local_dir: If provided, download the model directly to this directory
            instead of the default cache (~/.cache/tide2/). Useful when the home
            directory has limited disk space and a shared volume is available.

    Returns:
        Local path to the model directory

    Raises:
        ValueError: If model cannot be found locally, via GCS, or via HuggingFace
    """
    if local_dir is not None:
        local_model_path = Path(local_dir)
    else:
        cache_dir = get_cache_dir()
        local_model_path = cache_dir / "resources" / "models" / model_name

    # First, check if model exists locally in cache
    if local_model_path.exists() and local_model_path.is_dir():
        if validate_model_directory(local_model_path):
            logger.info(f"Found model locally: {local_model_path}")
            return str(local_model_path)
        logger.warning(
            f"Cached model directory {local_model_path} is incomplete "
            f"(missing weight files or config.json). Re-downloading..."
        )
        shutil.rmtree(local_model_path)

    # If bucket_name provided, try to download from GCS
    if bucket_name:
        gcs_model_path = f"gs://{bucket_name}/tide2/resources/models/{model_name}"
        logger.info(f"Attempting to download model from: {gcs_model_path}")

        try:
            resolved_path = download_from_gcs(gcs_model_path, "models", project_id, local_model_path)
            if resolved_path and resolved_path.exists():
                if not validate_model_directory(resolved_path):
                    raise ValueError(
                        f"Downloaded model from GCS is incomplete at {resolved_path}. "
                        f"Missing weight files (model.safetensors or pytorch_model.bin) or config.json."
                    )
                logger.info(f"Successfully downloaded model to: {resolved_path}")
                return str(resolved_path)
        except Exception as e:
            logger.warning(f"Failed to download model from GCS: {e}")

    # Try HuggingFace Hub as fallback
    if allow_huggingface_download:
        logger.info(f"Attempting to download model '{model_name}' from HuggingFace Hub...")
        try:
            from huggingface_hub import snapshot_download

            local_model_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=model_name, local_dir=str(local_model_path))
            if validate_model_directory(local_model_path):
                logger.info(f"Successfully downloaded model from HuggingFace Hub to: {local_model_path}")
                return str(local_model_path)
            logger.warning(
                f"HuggingFace download incomplete at {local_model_path}. "
                f"Missing weight files (model.safetensors or pytorch_model.bin) or config.json. "
                f"Check your network connection or try: huggingface-cli login"
            )
            shutil.rmtree(local_model_path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to download model from HuggingFace Hub: {e}")
            # Clean up incomplete directory to avoid false cache hits on next run
            if local_model_path.exists():
                shutil.rmtree(local_model_path, ignore_errors=True)

    # Model not found locally, via GCS, or via HuggingFace
    cache_dir = get_cache_dir()
    models_dir = cache_dir / "resources" / "models"
    available_local = list(models_dir.glob("*")) if models_dir.exists() else []
    error_msg = f"Model '{model_name}' not found. "

    if bucket_name:
        error_msg += (
            f"Checked locally and attempted download from gs://{bucket_name}/tide2/resources/models/{model_name}. "
        )
    else:
        error_msg += "No bucket_name provided for GCS download. "

    if allow_huggingface_download:
        error_msg += f"Also attempted HuggingFace Hub download with repo_id='{model_name}'. "

    if available_local:
        error_msg += f"Available local models: {[p.name for p in available_local]}"
    else:
        error_msg += "No models found in local cache."

    raise ValueError(error_msg)


def clear_cache(resource_type: str | None = None):
    """
    Clear the TIDE cache directory.

    Args:
        resource_type: Specific resource type to clear ('models', 'vaults', 'data'),
                      or None to clear entire cache
    """
    import shutil

    cache_dir = get_cache_dir()

    if resource_type:
        target_dir = cache_dir / resource_type
        if target_dir.exists():
            shutil.rmtree(target_dir)
            logger.info(f"Cleared cache for: {resource_type}")
            target_dir.mkdir(parents=True, exist_ok=True)
    else:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared entire cache: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)
