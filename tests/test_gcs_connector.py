"""
Unit tests for the GCS Connector module.

This module tests the functionality for Google Cloud Storage operations including
upload, download, delete, and list operations with retry logic and concurrent processing.
"""

from unittest.mock import Mock
from unittest.mock import patch

import pytest

from tide2.utils.gcs_connector import GCSConnector
from tide2.utils.gcs_connector import gcs_retry_decorator


class TestGCSRetryDecorator:
    """Test cases for the gcs_retry_decorator function."""

    def test_decorator_success(self):
        """Test that decorator allows successful function calls."""
        # Create a mock connector with retry settings
        mock_connector = Mock()
        mock_connector.max_retries = 3
        mock_connector.retry_multiplier = 2.0

        @gcs_retry_decorator
        def dummy_function(self, arg1, arg2):
            return f"{arg1}-{arg2}"

        result = dummy_function(mock_connector, "test", "value")
        assert result == "test-value"

    def test_decorator_retry_on_failure(self):
        """Test that decorator retries on failure."""
        mock_connector = Mock()
        mock_connector.max_retries = 3
        mock_connector.retry_multiplier = 0

        call_count = 0

        @gcs_retry_decorator
        def failing_function(self):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        result = failing_function(mock_connector)
        assert result == "success"
        assert call_count == 3


class TestGCSConnectorInitialization:
    """Test cases for GCSConnector initialization and configuration."""

    @patch("tide2.utils.gcs_connector.storage.Client")
    def test_init_default_values(self, mock_client):
        """Test initialization with default values."""
        connector = GCSConnector(project_id="test-project")

        assert connector.project_id == "test-project"
        assert connector.max_workers == 8
        assert connector.max_retries == 3
        assert connector.retry_multiplier == 2.0
        mock_client.assert_called_once_with(project="test-project")

    @patch("tide2.utils.gcs_connector.storage.Client")
    def test_init_custom_values(self, mock_client):
        """Test initialization with custom values."""
        connector = GCSConnector(project_id="custom-project", max_workers=4, max_retries=5, retry_multiplier=3.0)

        assert connector.project_id == "custom-project"
        assert connector.max_workers == 4
        assert connector.max_retries == 5
        assert connector.retry_multiplier == 3.0

    @patch("tide2.utils.gcs_connector.storage.Client")
    @patch("os.cpu_count", return_value=16)
    def test_create_with_optimal_settings(self, mock_cpu_count, mock_client):
        """Test create_with_optimal_settings method."""
        connector = GCSConnector.create_with_optimal_settings("test-project", max_retries=5)

        assert connector.project_id == "test-project"
        assert connector.max_workers == 4  # Should be capped at 4
        assert connector.max_retries == 5

    @patch("tide2.utils.gcs_connector.storage.Client")
    @patch("os.cpu_count", return_value=2)
    def test_create_with_optimal_settings_low_cpu(self, mock_cpu_count, mock_client):
        """Test create_with_optimal_settings with low CPU count."""
        connector = GCSConnector.create_with_optimal_settings("test-project")

        assert connector.max_workers == 2

    @patch("tide2.utils.gcs_connector.storage.Client")
    def test_get_connector_status(self, mock_client):
        """Test get_connector_status method."""
        connector = GCSConnector("test-project", max_workers=4, max_retries=2)
        status = connector.get_connector_status()

        expected_status = {
            "tenacity_available": True,
            "max_workers": 4,
            "max_retries": 2,
            "retry_multiplier": 2.0,
            "retry_status": "Retry support enabled with 2 max attempts",
        }
        assert status == expected_status

    @patch("tide2.utils.gcs_connector.storage.Client")
    def test_get_recommended_settings(self, mock_client):
        """Test get_recommended_settings method."""
        connector = GCSConnector("test-project", max_workers=12, max_retries=5)
        settings = connector.get_recommended_settings()

        assert settings["tenacity_available"] is True
        assert settings["current_max_workers"] == 12
        assert settings["recommended_max_workers"] == 8  # Capped at 8
        assert settings["current_max_retries"] == 5
        assert settings["recommended_max_retries"] == 3
        assert "notes" in settings


class TestGCSConnectorDownload:
    """Test cases for download operations."""

    @patch("tide2.utils.gcs_connector.storage.Client")
    def setup_method(self, method, mock_client):
        """Set up test fixtures."""
        self.connector = GCSConnector("test-project")
        self.connector.retry_multiplier = 0
        self.mock_client = mock_client.return_value
        self.mock_bucket = Mock()
        self.mock_blob = Mock()

        self.mock_client.bucket.return_value = self.mock_bucket
        self.mock_bucket.blob.return_value = self.mock_blob

    def test_download_single_file_success(self):
        """Test successful single file download."""
        # Setup mocks
        self.mock_blob.reload.return_value = None
        self.mock_blob.size = 1024 * 1024  # 1MB
        self.mock_blob.download_to_filename.return_value = None

        result = self.connector.download_single_file("test-bucket", "test-blob", "/tmp/test-file")

        assert result is True
        self.mock_client.bucket.assert_called_with("test-bucket")
        self.mock_bucket.blob.assert_called_with("test-blob")
        self.mock_blob.download_to_filename.assert_called_with("/tmp/test-file")

    def test_download_single_file_failure(self):
        """Test single file download failure."""
        # Setup mock to raise exception
        self.mock_blob.reload.side_effect = Exception("Download failed")

        result = self.connector.download_single_file("test-bucket", "test-blob", "/tmp/test-file")

        assert result is False

    @patch("tide2.utils.gcs_connector.ThreadPoolExecutor")
    def test_download_multiple_files(self, mock_executor):
        """Test multiple file download."""
        # Setup mock executor
        mock_future1 = Mock()
        mock_future1.result.return_value = True
        mock_future2 = Mock()
        mock_future2.result.return_value = False

        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

        # Mock as_completed
        with patch("tide2.utils.gcs_connector.as_completed", return_value=[mock_future1, mock_future2]):
            download_specs = [("bucket1", "blob1", "/tmp/file1"), ("bucket2", "blob2", "/tmp/file2")]

            results = self.connector.download_multiple_files(download_specs)

            assert len(results) == 2
            assert mock_executor_instance.submit.call_count == 2

    def test_download_directory(self):
        """Test directory download."""
        # Setup mock blobs
        mock_blob1 = Mock()
        mock_blob1.name = "prefix/file1.txt"
        mock_blob2 = Mock()
        mock_blob2.name = "prefix/file2.txt"
        mock_blob3 = Mock()
        mock_blob3.name = "prefix/subdir/"  # Directory marker

        self.mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]

        with patch.object(self.connector, "download_multiple_files") as mock_download_multiple:
            mock_download_multiple.return_value = {"prefix/file1.txt": True, "prefix/file2.txt": True}

            results = self.connector.download_directory("test-bucket", "prefix/", "/tmp/downloads")

            assert len(results) == 2
            # Verify download_multiple_files was called with correct specs
            mock_download_multiple.assert_called_once()
            call_args = mock_download_multiple.call_args[0][0]
            assert len(call_args) == 2  # Two files (excluding directory marker)


class TestGCSConnectorList:
    """Test cases for list operations."""

    @patch("tide2.utils.gcs_connector.storage.Client")
    def setup_method(self, method, mock_client):
        """Set up test fixtures."""
        self.connector = GCSConnector("test-project")
        self.mock_client = mock_client.return_value
        self.mock_bucket = Mock()
        self.mock_client.bucket.return_value = self.mock_bucket

    def test_list_files_by_pattern(self):
        """Test listing files by pattern."""
        # Setup mock blobs
        mock_blob1 = Mock()
        mock_blob1.name = "data/file1.txt"
        mock_blob2 = Mock()
        mock_blob2.name = "data/file2.json"
        mock_blob3 = Mock()
        mock_blob3.name = "data/file3.txt"
        mock_blob4 = Mock()
        mock_blob4.name = "data/subdir/"  # Directory marker

        self.mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3, mock_blob4]

        results = self.connector.list_files_by_pattern("test-bucket", "*.txt", "data/")

        assert len(results) == 2
        assert "data/file1.txt" in results
        assert "data/file3.txt" in results
        assert "data/file2.json" not in results

    def test_list_files_by_pattern_with_character_class(self):
        """Test listing files by pattern with character classes."""
        mock_blob1 = Mock()
        mock_blob1.name = "data/file1.txt"
        mock_blob2 = Mock()
        mock_blob2.name = "data/file9.txt"
        mock_blob3 = Mock()
        mock_blob3.name = "data/filea.txt"

        self.mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]

        results = self.connector.list_files_by_pattern("test-bucket", "file[0-9].txt", "data/")

        assert len(results) == 2
        assert "data/file1.txt" in results
        assert "data/file9.txt" in results
        assert "data/filea.txt" not in results

    def test_list_files_by_pattern_error(self):
        """Test list files by pattern with error."""
        self.mock_bucket.list_blobs.side_effect = Exception("List failed")

        results = self.connector.list_files_by_pattern("test-bucket", "*.txt")

        assert results == []

    def test_list_files_by_glob_pattern(self):
        """Test listing files by glob pattern."""
        mock_blob1 = Mock()
        mock_blob1.name = "logs/app.log"
        mock_blob2 = Mock()
        mock_blob2.name = "logs/error.log"
        mock_blob3 = Mock()
        mock_blob3.name = "logs/access.txt"

        self.mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]

        results = self.connector.list_files_by_glob_pattern("test-bucket", "*.log", "logs/")

        assert len(results) == 2
        assert "logs/app.log" in results
        assert "logs/error.log" in results
        assert "logs/access.txt" not in results


class TestGCSConnectorUpload:
    """Test cases for upload operations."""

    @patch("tide2.utils.gcs_connector.storage.Client")
    def setup_method(self, method, mock_client):
        """Set up test fixtures."""
        self.connector = GCSConnector("test-project")
        self.mock_client = mock_client.return_value
        self.mock_bucket = Mock()
        self.mock_blob = Mock()

        self.mock_client.bucket.return_value = self.mock_bucket
        self.mock_bucket.blob.return_value = self.mock_blob

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=1024)
    def test_upload_single_file_success(self, mock_getsize, mock_exists):
        """Test successful single file upload."""
        self.mock_blob.upload_from_filename.return_value = None

        result = self.connector.upload_single_file("test-bucket", "/tmp/test-file", "test-blob")

        assert result is True
        self.mock_client.bucket.assert_called_with("test-bucket")
        self.mock_bucket.blob.assert_called_with("test-blob")
        self.mock_blob.upload_from_filename.assert_called_with("/tmp/test-file")

    @patch("os.path.exists", return_value=False)
    def test_upload_single_file_not_found(self, mock_exists):
        """Test upload when local file doesn't exist."""
        result = self.connector.upload_single_file("test-bucket", "/tmp/nonexistent", "test-blob")

        assert result is False

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=1024)
    def test_upload_single_file_with_auto_blob_name(self, mock_getsize, mock_exists):
        """Test upload with automatic blob name generation."""
        self.mock_blob.upload_from_filename.return_value = None

        result = self.connector.upload_single_file("test-bucket", "/tmp/test-file.txt")

        assert result is True
        self.mock_bucket.blob.assert_called_with("test-file.txt")

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=1024)
    def test_upload_single_file_with_content_type(self, mock_getsize, mock_exists):
        """Test upload with content type specification."""
        self.mock_blob.upload_from_filename.return_value = None

        result = self.connector.upload_single_file("test-bucket", "/tmp/test.json", "test.json", "application/json")

        assert result is True
        assert self.mock_blob.content_type == "application/json"

    @patch("tide2.utils.gcs_connector.ThreadPoolExecutor")
    def test_upload_multiple_files(self, mock_executor):
        """Test multiple file upload."""
        mock_future1 = Mock()
        mock_future1.result.return_value = True
        mock_future2 = Mock()
        mock_future2.result.return_value = False

        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

        with patch("tide2.utils.gcs_connector.as_completed", return_value=[mock_future1, mock_future2]):
            upload_specs = [("bucket1", "/tmp/file1", "blob1"), ("bucket2", "/tmp/file2", "blob2")]

            results = self.connector.upload_multiple_files(upload_specs)

            assert len(results) == 2
            assert mock_executor_instance.submit.call_count == 2

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isdir", return_value=True)
    @patch("os.walk")
    @patch("os.makedirs")  # Mock makedirs to prevent actual directory creation
    def test_upload_folder(self, mock_makedirs, mock_walk, mock_isdir, mock_exists):
        """Test folder upload."""
        # Mock os.walk to return some files
        mock_walk.return_value = [
            ("/tmp/folder", ["subdir"], ["file1.txt", "file2.json"]),
            ("/tmp/folder/subdir", [], ["file3.txt"]),
        ]

        # Mock the ThreadPoolExecutor and upload process to avoid actual file operations
        with patch("tide2.utils.gcs_connector.ThreadPoolExecutor") as mock_executor:
            mock_future1 = Mock()
            mock_future1.result.return_value = True
            mock_future2 = Mock()
            mock_future2.result.return_value = True
            mock_future3 = Mock()
            mock_future3.result.return_value = True

            mock_executor_instance = Mock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            mock_executor_instance.submit.side_effect = [mock_future1, mock_future2, mock_future3]

            with patch(
                "tide2.utils.gcs_connector.as_completed", return_value=[mock_future1, mock_future2, mock_future3]
            ):
                results = self.connector.upload_folder("test-bucket", "/tmp/folder", "uploads/")

                # Verify ThreadPoolExecutor was used
                assert mock_executor_instance.submit.call_count == 3
                # Verify results contain the expected files
                assert len(results) == 3

    @patch("os.path.exists", return_value=False)
    def test_upload_folder_not_found(self, mock_exists):
        """Test upload folder when folder doesn't exist."""
        results = self.connector.upload_folder("test-bucket", "/tmp/nonexistent")

        assert results == {}

    @patch("os.path.exists", return_value=True)
    @patch("os.path.isdir", return_value=False)
    def test_upload_folder_not_directory(self, mock_isdir, mock_exists):
        """Test upload folder when path is not a directory."""
        results = self.connector.upload_folder("test-bucket", "/tmp/file.txt")

        assert results == {}


class TestGCSConnectorDelete:
    """Test cases for delete operations."""

    @patch("tide2.utils.gcs_connector.storage.Client")
    def setup_method(self, method, mock_client):
        """Set up test fixtures."""
        self.connector = GCSConnector("test-project")
        self.connector.retry_multiplier = 0
        self.mock_client = mock_client.return_value
        self.mock_bucket = Mock()
        self.mock_blob = Mock()

        self.mock_client.bucket.return_value = self.mock_bucket
        self.mock_bucket.blob.return_value = self.mock_blob

    def test_delete_single_file_success(self):
        """Test successful single file deletion."""
        self.mock_blob.exists.return_value = True
        self.mock_blob.delete.return_value = None

        result = self.connector.delete_single_file("test-bucket", "test-blob")

        assert result is True
        self.mock_client.bucket.assert_called_with("test-bucket")
        self.mock_bucket.blob.assert_called_with("test-blob")
        self.mock_blob.delete.assert_called_once()

    def test_delete_single_file_not_found(self):
        """Test delete when file doesn't exist."""
        self.mock_blob.exists.return_value = False

        result = self.connector.delete_single_file("test-bucket", "nonexistent-blob")

        assert result is False
        self.mock_blob.delete.assert_not_called()

    def test_delete_single_file_failure(self):
        """Test delete single file failure."""
        self.mock_blob.exists.return_value = True
        self.mock_blob.delete.side_effect = Exception("Delete failed")

        result = self.connector.delete_single_file("test-bucket", "test-blob")

        assert result is False

    @patch("tide2.utils.gcs_connector.ThreadPoolExecutor")
    def test_delete_multiple_files(self, mock_executor):
        """Test multiple file deletion."""
        mock_future1 = Mock()
        mock_future1.result.return_value = True
        mock_future2 = Mock()
        mock_future2.result.return_value = False

        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

        with patch("tide2.utils.gcs_connector.as_completed", return_value=[mock_future1, mock_future2]):
            delete_specs = [("bucket1", "blob1"), ("bucket2", "blob2")]

            results = self.connector.delete_multiple_files(delete_specs)

            assert len(results) == 2
            assert mock_executor_instance.submit.call_count == 2

    def test_delete_folder(self):
        """Test folder deletion."""
        # Setup mock blobs
        mock_blob1 = Mock()
        mock_blob1.name = "folder/file1.txt"
        mock_blob2 = Mock()
        mock_blob2.name = "folder/file2.txt"
        mock_blob3 = Mock()
        mock_blob3.name = "folder/subdir/"  # Directory marker

        self.mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]

        with patch.object(self.connector, "delete_multiple_files") as mock_delete_multiple:
            mock_delete_multiple.return_value = {"folder/file1.txt": True, "folder/file2.txt": True}

            results = self.connector.delete_folder("test-bucket", "folder/")

            assert len(results) == 2
            mock_delete_multiple.assert_called_once()
            # Verify only files (not directory markers) were included
            call_args = mock_delete_multiple.call_args[0][0]
            assert len(call_args) == 2

    def test_delete_folder_with_patterns(self):
        """Test folder deletion with include/exclude patterns."""
        mock_blob1 = Mock()
        mock_blob1.name = "logs/app.log"
        mock_blob2 = Mock()
        mock_blob2.name = "logs/error.log"
        mock_blob3 = Mock()
        mock_blob3.name = "logs/access.txt"
        mock_blob4 = Mock()
        mock_blob4.name = "logs/temp.tmp"

        self.mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3, mock_blob4]

        with patch.object(self.connector, "delete_multiple_files") as mock_delete_multiple:
            mock_delete_multiple.return_value = {"logs/app.log": True, "logs/error.log": True}

            self.connector.delete_folder("test-bucket", "logs/", include_patterns=["*.log"], exclude_patterns=["*.tmp"])

            mock_delete_multiple.assert_called_once()
            # Should only include .log files, excluding .txt and .tmp
            call_args = mock_delete_multiple.call_args[0][0]
            blob_names = [spec[1] for spec in call_args]
            assert "logs/app.log" in blob_names
            assert "logs/error.log" in blob_names
            assert "logs/access.txt" not in blob_names
            assert "logs/temp.tmp" not in blob_names

    def test_delete_folder_dry_run(self):
        """Test folder deletion with dry run."""
        mock_blob1 = Mock()
        mock_blob1.name = "temp/file1.txt"
        mock_blob2 = Mock()
        mock_blob2.name = "temp/file2.txt"

        self.mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

        results = self.connector.delete_folder("test-bucket", "temp/", dry_run=True)

        assert len(results) == 2
        assert all(success for success in results.values())  # All should be True for dry run

    def test_delete_folder_no_files(self):
        """Test folder deletion when no files match."""
        self.mock_bucket.list_blobs.return_value = []

        results = self.connector.delete_folder("test-bucket", "empty/")

        assert results == {}

    def test_delete_folder_error(self):
        """Test folder deletion with error."""
        self.mock_bucket.list_blobs.side_effect = Exception("List failed")

        results = self.connector.delete_folder("test-bucket", "folder/")

        assert results == {}

    def test_delete_files_by_pattern(self):
        """Test deleting files by pattern."""
        with patch.object(self.connector, "list_files_by_pattern") as mock_list:
            with patch.object(self.connector, "delete_multiple_files") as mock_delete_multiple:
                mock_list.return_value = ["data/file1.txt", "data/file2.txt"]
                mock_delete_multiple.return_value = {"data/file1.txt": True, "data/file2.txt": True}

                results = self.connector.delete_files_by_pattern("test-bucket", "*.txt", "data/")

                mock_list.assert_called_once_with(
                    bucket_name="test-bucket", pattern="*.txt", prefix="data/", max_results=None
                )
                mock_delete_multiple.assert_called_once()
                assert len(results) == 2

    def test_delete_files_by_pattern_dry_run(self):
        """Test deleting files by pattern with dry run."""
        with patch.object(self.connector, "list_files_by_pattern") as mock_list:
            mock_list.return_value = ["temp/file1.tmp", "temp/file2.tmp"]

            results = self.connector.delete_files_by_pattern("test-bucket", "*.tmp", "temp/", dry_run=True)

            assert len(results) == 2
            assert all(success for success in results.values())  # All should be True for dry run

    def test_delete_files_by_pattern_no_files(self):
        """Test delete by pattern when no files match."""
        with patch.object(self.connector, "list_files_by_pattern") as mock_list:
            mock_list.return_value = []

            results = self.connector.delete_files_by_pattern("test-bucket", "*.nonexistent")

            assert results == {}

    def test_delete_files_by_pattern_error(self):
        """Test delete by pattern with error."""
        with patch.object(self.connector, "list_files_by_pattern") as mock_list:
            mock_list.side_effect = Exception("Pattern matching failed")

            results = self.connector.delete_files_by_pattern("test-bucket", "*.txt")

            assert results == {}


class TestGCSConnectorIntegration:
    """Integration test cases that test multiple methods working together."""

    @patch("tide2.utils.gcs_connector.storage.Client")
    def setup_method(self, method, mock_client):
        """Set up test fixtures."""
        self.connector = GCSConnector("test-project", max_workers=2)
        self.mock_client = mock_client.return_value

    def test_upload_then_download_workflow(self):
        """Test a complete upload then download workflow."""
        # This would test the interaction between upload and download methods
        # In a real scenario, you might upload files then download them back
        pass

    def test_list_then_delete_workflow(self):
        """Test listing files then deleting them."""
        # This would test the interaction between list and delete methods
        pass

    def test_concurrent_operations_limit(self):
        """Test that concurrent operations respect the max_workers limit."""
        # This would test that the ThreadPoolExecutor is properly configured
        pass


if __name__ == "__main__":
    pytest.main([__file__])
