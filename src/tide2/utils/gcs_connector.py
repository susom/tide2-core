"""
Google Cloud Storage utility with simplified retry support and concurrent operations.

This module provides reliable strategies for both downloading and uploading files to/from GCS:

Download features:
- Single file download with tenacity retry decorators
- Concurrent downloads for multiple files
- Directory downloads with prefix filtering
- Pattern-based file filtering

Upload features:
- Single file upload with tenacity retry decorators
- Concurrent uploads for multiple files
- Folder uploads with pattern filtering
- Automatic content type detection

Retry features:
- Exponential backoff retry using tenacity decorators
- Configurable maximum retry attempts and backoff multiplier
- Direct retry decorators on single file operations

Requirements:
- tenacity library is required and assumed to be available
"""

import fnmatch
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

from google.cloud import storage
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential


def gcs_retry_decorator(func):
    """
    Method decorator that applies retry logic using tenacity.
    Uses the instance's retry configuration.

    Args:
        func: Function to decorate with retry logic

    Returns:
        Decorated function with retry support
    """

    def wrapper(self, *args, **kwargs):
        # Create retry decorator with instance parameters
        retrier = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.retry_multiplier, min=1, max=10),
            retry=retry_if_exception_type((Exception,)),
        )

        # Apply retry to the original function
        retried_func = retrier(func)
        return retried_func(self, *args, **kwargs)

    return wrapper


class GCSConnector:
    """Reliable GCS connector with tenacity retry decorators for single file operations."""

    @classmethod
    def create_with_optimal_settings(cls, project_id: str, max_retries: int = 3) -> "GCSConnector":
        """
        Create a GCSConnector instance with optimal settings for the current environment.

        Args:
            project_id: Google Cloud project ID
            max_retries: Maximum number of retry attempts

        Returns:
            GCSConnector instance configured with optimal settings
        """
        # Determine optimal number of workers (reduced for large file stability)
        optimal_workers = min(4, os.cpu_count() or 2)  # Cap at 4, default to 2 if unknown

        connector = cls(project_id=project_id, max_workers=optimal_workers, max_retries=max_retries)

        # Print configuration information
        print(f"GCS Connector initialized with {optimal_workers} workers, {max_retries} max retries")
        print("Retry support: Available")

        return connector

    def __init__(self, project_id: str, max_workers: int = 8, max_retries: int = 3, retry_multiplier: float = 2.0):
        """
        Initialize the GCS connector.

        Args:
            project_id: Google Cloud project ID
            max_workers: Maximum number of worker threads for parallel operations
            max_retries: Maximum number of retry attempts for failed operations
            retry_multiplier: Exponential backoff multiplier for retries
        """
        self.project_id = project_id
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_multiplier = retry_multiplier
        self.client = storage.Client(project=project_id)

    def get_connector_status(self) -> dict[str, Any]:
        """
        Get information about GCS connector configuration and features.

        Returns:
            Dictionary containing connector status and configuration
        """
        return {
            "tenacity_available": True,
            "max_workers": self.max_workers,
            "max_retries": self.max_retries,
            "retry_multiplier": self.retry_multiplier,
            "retry_status": f"Retry support enabled with {self.max_retries} max attempts",
        }

    def get_recommended_settings(self) -> dict[str, Any]:
        """
        Get recommended settings based on current environment and available features.

        Returns:
            Dictionary containing recommended configuration settings
        """
        recommendations = {
            "tenacity_available": True,
            "current_max_workers": self.max_workers,
            "recommended_max_workers": min(self.max_workers, 8),  # Cap at 8 for stability
            "current_max_retries": self.max_retries,
            "recommended_max_retries": 3,  # Good default for most use cases
        }

        installation_commands = []
        notes = []

        notes.append("Using simplified GCS upload/download methods with tenacity decorators for reliable transfers")
        notes.append(f"Retry support enabled with {self.max_retries} max attempts and exponential backoff")

        if installation_commands:
            recommendations["installation_commands"] = installation_commands

        recommendations["notes"] = notes

        return recommendations

    @gcs_retry_decorator
    def _download_file_with_retry(self, bucket_name: str, blob_name: str, local_path: str) -> None:
        """Internal method to download a file with retry decorator."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Get blob size for reporting
        blob.reload()
        blob_size_mb = blob.size / (1024 * 1024) if blob.size else 0

        print(f"Downloading {blob_name} ({blob_size_mb:.2f} MB)...")
        start_time = time.time()

        # For very large files (>1GB), add timeout and chunked download
        if blob_size_mb > 1024:  # 1GB threshold
            print(f"Large file detected ({blob_size_mb:.2f} MB). Using chunked download...")

            # Set a longer timeout for large files (30 minutes)
            import google.cloud.storage.constants

            original_timeout = getattr(google.cloud.storage.constants, "_DEFAULT_TIMEOUT", 60)

            try:
                # Download with custom timeout for large files
                blob.download_to_filename(local_path, timeout=1800)  # 30 minutes
            except Exception as e:
                print(f"Large file download failed: {e}")
                # Try with even longer timeout
                print("Retrying with extended timeout...")
                blob.download_to_filename(local_path, timeout=3600)  # 60 minutes
        else:
            # Download the file normally for smaller files
            blob.download_to_filename(local_path)

        elapsed_time = time.time() - start_time
        download_speed = blob_size_mb / elapsed_time if elapsed_time > 0 else 0
        print(f"✓ Downloaded {blob_name} in {elapsed_time:.2f}s ({download_speed:.2f} MB/s)")

    def download_single_file(self, bucket_name: str, blob_name: str, local_path: str) -> bool:
        """
        Download a single file from GCS with retry support.

        Args:
            bucket_name: Name of the GCS bucket
            blob_name: Name of the blob to download
            local_path: Local file path to save the download

        Returns:
            True if download successful, False otherwise
        """
        try:
            self._download_file_with_retry(bucket_name, blob_name, local_path)
            return True
        except Exception:
            # Cache miss - file doesn't exist yet
            return False

    def download_multiple_files(self, download_specs: list[tuple[str, str, str]]) -> dict[str, bool]:
        """
        Download multiple files concurrently from GCS.

        Args:
            download_specs: List of tuples (bucket_name, blob_name, local_path)

        Returns:
            Dictionary mapping blob names to success status
        """
        results = {}

        print(f"Starting concurrent download of {len(download_specs)} files...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_blob = {
                executor.submit(self.download_single_file, bucket_name, blob_name, local_path): blob_name
                for bucket_name, blob_name, local_path in download_specs
            }

            # Collect results
            for future in as_completed(future_to_blob):
                blob_name = future_to_blob[future]
                try:
                    success = future.result()
                    results[blob_name] = success
                except Exception as e:
                    print(f"✗ Error in concurrent download of {blob_name}: {e!s}")
                    results[blob_name] = False

        successful_downloads = sum(1 for success in results.values() if success)
        print(f"Completed: {successful_downloads}/{len(download_specs)} files downloaded successfully")

        return results

    def download_directory(self, bucket_name: str, prefix: str, local_directory: str) -> dict[str, bool]:
        """
        Download all files with a given prefix from GCS bucket.

        Args:
            bucket_name: Name of the GCS bucket
            prefix: Prefix to filter blobs (like a directory path)
            local_directory: Local directory to save downloads

        Returns:
            Dictionary mapping blob names to success status
        """
        bucket = self.client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        print(f"Found {len(blobs)} files with prefix '{prefix}'")

        # Create local directory if it doesn't exist
        Path(local_directory).mkdir(parents=True, exist_ok=True)

        # Prepare download specifications
        download_specs = []
        for blob in blobs:
            if not blob.name.endswith("/"):  # Skip directory markers
                # Maintain directory structure
                relative_path = blob.name[len(prefix) :].lstrip("/")
                local_path = os.path.join(local_directory, relative_path)

                # Create subdirectories if needed
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                download_specs.append((bucket_name, blob.name, local_path))

        return self.download_multiple_files(download_specs)

    def list_files_by_pattern(
        self, bucket_name: str, pattern: str = "*", prefix: str = "", max_results: int | None = None
    ) -> list[str]:
        """
        List files in a GCS bucket according to a pattern.

        Args:
            bucket_name: Name of the GCS bucket
            pattern: Pattern to match file names (supports wildcards like *, ?, [])
                    Examples: "*.txt", "data_*.json", "report_[0-9]*.csv"
            prefix: Prefix to filter blobs before pattern matching (for efficiency)
            max_results: Maximum number of results to return (None for all)

        Returns:
            List of blob names that match the pattern
        """
        try:
            bucket = self.client.bucket(bucket_name)

            # List blobs with prefix for efficiency
            blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)

            # Convert pattern to regex for more flexible matching
            # Handle common glob patterns
            regex_pattern = pattern.replace("*", ".*").replace("?", ".")
            # Handle character classes [abc] or [0-9]
            regex_pattern = re.sub(r"\[([^\]]+)\]", r"[\1]", regex_pattern)
            regex_pattern = f"^{regex_pattern}$"

            compiled_pattern = re.compile(regex_pattern)

            matching_files = []
            for blob in blobs:
                # Skip directory markers
                if blob.name.endswith("/"):
                    continue

                # Extract just the filename for pattern matching
                filename = os.path.basename(blob.name)

                # Check if filename matches the pattern
                if compiled_pattern.match(filename):
                    matching_files.append(blob.name)

            print(f"Found {len(matching_files)} files matching pattern '{pattern}' in bucket '{bucket_name}'")
            return matching_files

        except Exception as e:
            print(f"✗ Error listing files with pattern '{pattern}': {e!s}")
            return []

    def list_files_by_glob_pattern(
        self, bucket_name: str, glob_pattern: str, prefix: str = "", max_results: int | None = None
    ) -> list[str]:
        """
        List files in a GCS bucket using Unix shell-style glob patterns.

        Args:
            bucket_name: Name of the GCS bucket
            glob_pattern: Glob pattern to match file names
                         Examples: "*.txt", "data_*.json", "report_[0-9]*.csv"
            prefix: Prefix to filter blobs before pattern matching (for efficiency)
            max_results: Maximum number of results to return (None for all)

        Returns:
            List of blob names that match the glob pattern
        """
        try:
            bucket = self.client.bucket(bucket_name)

            # List blobs with prefix for efficiency
            blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)

            matching_files = []
            for blob in blobs:
                # Skip directory markers
                if blob.name.endswith("/"):
                    continue

                # Extract just the filename for pattern matching
                filename = os.path.basename(blob.name)

                # Use fnmatch for Unix shell-style wildcards
                if fnmatch.fnmatch(filename, glob_pattern):
                    matching_files.append(blob.name)

            print(f"Found {len(matching_files)} files matching glob pattern '{glob_pattern}' in bucket '{bucket_name}'")
            return matching_files

        except Exception as e:
            print(f"✗ Error listing files with glob pattern '{glob_pattern}': {e!s}")
            return []

    @gcs_retry_decorator
    def _upload_file_with_retry(
        self, bucket_name: str, local_file_path: str, blob_name: str, content_type: str | None
    ) -> None:
        """Internal method to upload a file with retry decorator."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Get file size for reporting
        file_size = os.path.getsize(local_file_path)
        file_size_mb = file_size / (1024 * 1024)

        print(f"Uploading {local_file_path} to {blob_name} ({file_size_mb:.2f} MB)...")
        start_time = time.time()

        # Set content type if provided
        if content_type:
            blob.content_type = content_type

        # Upload the file
        blob.upload_from_filename(local_file_path)

        elapsed_time = time.time() - start_time
        upload_speed = file_size_mb / elapsed_time if elapsed_time > 0 else 0
        print(f"✓ Uploaded {blob_name} in {elapsed_time:.2f}s ({upload_speed:.2f} MB/s)")

    def upload_single_file(
        self,
        bucket_name: str,
        local_file_path: str,
        blob_name: str | None = None,
        content_type: str | None = None,
    ) -> bool:
        """
        Upload a single file to GCS with retry support.

        Args:
            bucket_name: Name of the GCS bucket
            local_file_path: Path to the local file to upload
            blob_name: Name for the blob in GCS (if None, uses filename)
            content_type: MIME type of the file (if None, auto-detected)

        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Check if local file exists
            if not os.path.exists(local_file_path):
                print(f"✗ Local file not found: {local_file_path}")
                return False

            # Use filename if blob_name not provided
            if blob_name is None:
                blob_name = os.path.basename(local_file_path)

            self._upload_file_with_retry(bucket_name, local_file_path, blob_name, content_type)
            return True

        except Exception as e:
            print(f"✗ Error uploading {local_file_path}: {e!s}")
            return False

    @gcs_retry_decorator
    def _upload_string_with_retry(
        self, bucket_name: str, blob_name: str, content: str, content_type: str = "application/json"
    ) -> None:
        """Internal method to upload string content with retry decorator."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        content_size_kb = len(content.encode("utf-8")) / 1024
        print(f"Uploading {blob_name} ({content_size_kb:.2f} KB)...")
        start_time = time.time()

        blob.upload_from_string(content, content_type=content_type)

        elapsed_time = time.time() - start_time
        upload_speed = content_size_kb / elapsed_time if elapsed_time > 0 else 0
        print(f"✓ Uploaded {blob_name} in {elapsed_time:.2f}s ({upload_speed:.2f} KB/s)")

    def upload_from_string(
        self, bucket_name: str, blob_name: str, content: str, content_type: str = "application/json"
    ) -> bool:
        """
        Upload string content to GCS with retry support.

        Args:
            bucket_name: Name of the GCS bucket
            blob_name: Name for the blob in GCS
            content: String content to upload
            content_type: MIME type (default: application/json)

        Returns:
            True if upload successful, False otherwise
        """
        try:
            self._upload_string_with_retry(bucket_name, blob_name, content, content_type)
            return True
        except Exception as e:
            print(f"✗ Error uploading string to {blob_name}: {e!s}")
            return False

    def upload_multiple_files(self, upload_specs: list[tuple[str, str, str]]) -> dict[str, bool]:
        """
        Upload multiple files concurrently to GCS.

        Args:
            upload_specs: List of tuples (bucket_name, local_file_path, blob_name)

        Returns:
            Dictionary mapping local file paths to success status
        """
        results = {}

        print(f"Starting concurrent upload of {len(upload_specs)} files...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all upload tasks
            future_to_file = {
                executor.submit(
                    self.upload_single_file,
                    bucket_name,
                    local_file_path,
                    blob_name,
                    None,  # content_type (auto-detect)
                ): local_file_path
                for bucket_name, local_file_path, blob_name in upload_specs
            }

            # Collect results
            for future in as_completed(future_to_file):
                local_file_path = future_to_file[future]
                try:
                    success = future.result()
                    results[local_file_path] = success
                except Exception as e:
                    print(f"✗ Error in concurrent upload of {local_file_path}: {e!s}")
                    results[local_file_path] = False

        successful_uploads = sum(1 for success in results.values() if success)
        print(f"Completed: {successful_uploads}/{len(upload_specs)} files uploaded successfully")

        return results

    def upload_folder(
        self,
        bucket_name: str,
        local_folder_path: str,
        prefix: str = "",
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, bool]:
        """
        Upload an entire folder to GCS, maintaining directory structure.

        Args:
            bucket_name: Name of the GCS bucket
            local_folder_path: Path to the local folder to upload
            prefix: Prefix to add to blob names (like a directory path in GCS)
            include_patterns: List of glob patterns for files to include (e.g., ["*.txt", "*.json"])
            exclude_patterns: List of glob patterns for files to exclude (e.g., ["*.tmp", "__pycache__/*"])

        Returns:
            Dictionary mapping local file paths to success status
        """
        try:
            if not os.path.exists(local_folder_path):
                print(f"✗ Local folder not found: {local_folder_path}")
                return {}

            if not os.path.isdir(local_folder_path):
                print(f"✗ Path is not a directory: {local_folder_path}")
                return {}

            print(f"Scanning folder: {local_folder_path}")

            # Collect all files to upload
            files_to_upload = []

            for root, dirs, files in os.walk(local_folder_path):
                for file in files:
                    local_file_path = os.path.join(root, file)

                    # Check include patterns
                    if include_patterns:
                        include_match = any(fnmatch.fnmatch(file, pattern) for pattern in include_patterns)
                        if not include_match:
                            continue

                    # Check exclude patterns
                    if exclude_patterns:
                        exclude_match = any(
                            fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(local_file_path, pattern)
                            for pattern in exclude_patterns
                        )
                        if exclude_match:
                            continue

                    # Calculate relative path from the base folder
                    relative_path = os.path.relpath(local_file_path, local_folder_path)

                    # Create blob name with prefix
                    blob_name = (
                        os.path.join(prefix, relative_path).replace(os.sep, "/")
                        if prefix
                        else relative_path.replace(os.sep, "/")
                    )

                    files_to_upload.append((local_file_path, blob_name))

            if not files_to_upload:
                print("No files found to upload")
                return {}

            print(f"Found {len(files_to_upload)} files to upload")

            # Upload files concurrently
            results = {}

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all upload tasks
                future_to_file = {
                    executor.submit(
                        self.upload_single_file,
                        bucket_name,
                        local_file_path,
                        blob_name,
                        None,  # content_type (auto-detect)
                    ): local_file_path
                    for local_file_path, blob_name in files_to_upload
                }

                # Collect results
                for future in as_completed(future_to_file):
                    local_file_path = future_to_file[future]
                    try:
                        success = future.result()
                        results[local_file_path] = success
                    except Exception as e:
                        print(f"✗ Error in concurrent upload of {local_file_path}: {e!s}")
                        results[local_file_path] = False

            successful_uploads = sum(1 for success in results.values() if success)
            print(f"Completed: {successful_uploads}/{len(files_to_upload)} files uploaded successfully")

            return results

        except Exception as e:
            print(f"✗ Error uploading folder {local_folder_path}: {e!s}")
            return {}

    @gcs_retry_decorator
    def _delete_file_with_retry(self, bucket_name: str, blob_name: str) -> None:
        """Internal method to delete a file with retry decorator."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        print(f"Deleting {blob_name}...")
        start_time = time.time()

        # Delete the file
        blob.delete()

        elapsed_time = time.time() - start_time
        print(f"✓ Deleted {blob_name} in {elapsed_time:.2f}s")

    def delete_single_file(self, bucket_name: str, blob_name: str) -> bool:
        """
        Delete a single file from GCS with retry support.

        Args:
            bucket_name: Name of the GCS bucket
            blob_name: Name of the blob to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Check if blob exists
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                print(f"✗ File not found in bucket: {blob_name}")
                return False

            self._delete_file_with_retry(bucket_name, blob_name)
            return True

        except Exception as e:
            print(f"✗ Error deleting {blob_name}: {e!s}")
            return False

    def delete_multiple_files(self, delete_specs: list[tuple[str, str]]) -> dict[str, bool]:
        """
        Delete multiple files concurrently from GCS.

        Args:
            delete_specs: List of tuples (bucket_name, blob_name)

        Returns:
            Dictionary mapping blob names to success status
        """
        results = {}

        print(f"Starting concurrent deletion of {len(delete_specs)} files...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all delete tasks
            future_to_blob = {
                executor.submit(self.delete_single_file, bucket_name, blob_name): blob_name
                for bucket_name, blob_name in delete_specs
            }

            # Collect results
            for future in as_completed(future_to_blob):
                blob_name = future_to_blob[future]
                try:
                    success = future.result()
                    results[blob_name] = success
                except Exception as e:
                    print(f"✗ Error in concurrent deletion of {blob_name}: {e!s}")
                    results[blob_name] = False

        successful_deletions = sum(1 for success in results.values() if success)
        print(f"Completed: {successful_deletions}/{len(delete_specs)} files deleted successfully")

        return results

    def delete_folder(
        self,
        bucket_name: str,
        prefix: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, bool]:
        """
        Delete all files with a given prefix from GCS bucket (like deleting a folder).

        Args:
            bucket_name: Name of the GCS bucket
            prefix: Prefix to filter blobs for deletion (like a directory path)
            include_patterns: List of glob patterns for files to include (e.g., ["*.txt", "*.json"])
            exclude_patterns: List of glob patterns for files to exclude (e.g., ["*.tmp", "*.log"])
            dry_run: If True, only list files that would be deleted without actually deleting them

        Returns:
            Dictionary mapping blob names to success status (or True for dry run listings)
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))

            print(f"Found {len(blobs)} files with prefix '{prefix}'")

            # Filter blobs based on patterns
            files_to_delete = []
            for blob in blobs:
                if blob.name.endswith("/"):  # Skip directory markers
                    continue

                # Extract just the filename for pattern matching
                filename = os.path.basename(blob.name)

                # Check include patterns
                if include_patterns:
                    include_match = any(fnmatch.fnmatch(filename, pattern) for pattern in include_patterns)
                    if not include_match:
                        continue

                # Check exclude patterns
                if exclude_patterns:
                    exclude_match = any(
                        fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(blob.name, pattern)
                        for pattern in exclude_patterns
                    )
                    if exclude_match:
                        continue

                files_to_delete.append(blob.name)

            if not files_to_delete:
                print("No files found to delete")
                return {}

            print(f"Selected {len(files_to_delete)} files for deletion")

            if dry_run:
                print("DRY RUN - Files that would be deleted:")
                for blob_name in files_to_delete:
                    print(f"  - {blob_name}")
                return dict.fromkeys(files_to_delete, True)

            # Prepare delete specifications
            delete_specs = [(bucket_name, blob_name) for blob_name in files_to_delete]

            return self.delete_multiple_files(delete_specs)

        except Exception as e:
            print(f"✗ Error deleting folder with prefix '{prefix}': {e!s}")
            return {}

    def delete_files_by_pattern(
        self,
        bucket_name: str,
        pattern: str = "*",
        prefix: str = "",
        dry_run: bool = False,
        max_results: int | None = None,
    ) -> dict[str, bool]:
        """
        Delete files in a GCS bucket according to a pattern.

        Args:
            bucket_name: Name of the GCS bucket
            pattern: Pattern to match file names (supports wildcards like *, ?, [])
                    Examples: "*.txt", "data_*.json", "report_[0-9]*.csv"
            prefix: Prefix to filter blobs before pattern matching (for efficiency)
            dry_run: If True, only list files that would be deleted without actually deleting them
            max_results: Maximum number of results to process (None for all)

        Returns:
            Dictionary mapping blob names to success status (or True for dry run listings)
        """
        try:
            # First, list files matching the pattern
            matching_files = self.list_files_by_pattern(
                bucket_name=bucket_name, pattern=pattern, prefix=prefix, max_results=max_results
            )

            if not matching_files:
                print(f"No files found matching pattern '{pattern}'")
                return {}

            print(f"Selected {len(matching_files)} files for deletion")

            if dry_run:
                print("DRY RUN - Files that would be deleted:")
                for blob_name in matching_files:
                    print(f"  - {blob_name}")
                return dict.fromkeys(matching_files, True)

            # Prepare delete specifications
            delete_specs = [(bucket_name, blob_name) for blob_name in matching_files]

            return self.delete_multiple_files(delete_specs)

        except Exception as e:
            print(f"✗ Error deleting files with pattern '{pattern}': {e!s}")
            return {}
