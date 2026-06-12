"""
Tests for GCS resource manager.

These tests verify the resource path resolution and caching functionality.
"""

import os
from unittest.mock import patch

import pytest

from tide2.utils.gcs_resource_manager import get_cache_dir
from tide2.utils.gcs_resource_manager import is_gcs_path
from tide2.utils.gcs_resource_manager import parse_gcs_path
from tide2.utils.gcs_resource_manager import resolve_resource_path


def test_get_cache_dir_default(tmp_path, monkeypatch):
    """Test that default cache directory is created correctly."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("TIDE_CACHE_DIR", raising=False)

    # DEFAULT_CACHE_DIR is computed at import time, so patch it directly
    expected = fake_home / ".cache" / "tide2"
    monkeypatch.setattr("tide2.utils.gcs_resource_manager.DEFAULT_CACHE_DIR", expected)

    cache_dir = get_cache_dir()

    assert cache_dir == expected
    assert cache_dir.exists()


def test_get_cache_dir_custom(tmp_path, monkeypatch):
    """Test that custom cache directory from env var is used."""
    custom_cache = tmp_path / "custom_cache"
    monkeypatch.setenv("TIDE_CACHE_DIR", str(custom_cache))

    cache_dir = get_cache_dir()

    assert cache_dir == custom_cache
    assert cache_dir.exists()


def test_is_gcs_path():
    """Test GCS path detection."""
    assert is_gcs_path("gs://my-bucket/path/to/file")
    assert is_gcs_path("gs://bucket/")
    assert is_gcs_path("gs://bucket")

    assert not is_gcs_path("/local/path")
    assert not is_gcs_path("s3://bucket/path")
    assert not is_gcs_path("http://example.com")


def test_parse_gcs_path():
    """Test GCS path parsing."""
    bucket, prefix = parse_gcs_path("gs://my-bucket/path/to/directory")
    assert bucket == "my-bucket"
    assert prefix == "path/to/directory"

    bucket, prefix = parse_gcs_path("gs://my-bucket")
    assert bucket == "my-bucket"
    assert prefix == ""

    bucket, prefix = parse_gcs_path("gs://my-bucket/path/")
    assert bucket == "my-bucket"
    assert prefix == "path/"

    with pytest.raises(ValueError):
        parse_gcs_path("/local/path")


def test_resolve_resource_path_none():
    """Test that None path returns None."""
    result = resolve_resource_path(None)
    assert result is None


def test_resolve_resource_path_local():
    """Test that local path is returned as-is."""
    local_path = "/local/path/to/file"
    result = resolve_resource_path(local_path)
    assert result == local_path


@patch("tide2.utils.gcs_resource_manager.download_from_gcs")
def test_resolve_resource_path_gcs(mock_download, tmp_path):
    """Test that GCS path triggers download."""
    gcs_path = "gs://my-bucket/models/my-model"
    mock_download.return_value = tmp_path / "cached"

    result = resolve_resource_path(gcs_path, resource_type="models", project_id="test-project")

    mock_download.assert_called_once_with(gcs_path, "models", "test-project")
    assert result == str(tmp_path / "cached")


def test_resolve_resource_path_gcs_no_project(monkeypatch):
    """Test that GCS path without project ID raises error."""
    gcs_path = "gs://my-bucket/models/my-model"

    # Clear env vars that resolve_resource_path falls back to
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    with pytest.raises(ValueError, match="GCP project ID required"):
        resolve_resource_path(gcs_path, resource_type="models", project_id=None)


def test_cache_dir_expansion(tmp_path, monkeypatch):
    """Test that ~ is expanded in TIDE_CACHE_DIR."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("TIDE_CACHE_DIR", "~/my_cache")

    cache_dir = get_cache_dir()

    assert cache_dir == fake_home / "my_cache"
    assert cache_dir.exists()


def test_cache_dir_relative_path(tmp_path, monkeypatch):
    """Test that relative paths in TIDE_CACHE_DIR are resolved."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        monkeypatch.setenv("TIDE_CACHE_DIR", "relative/cache")
        cache_dir = get_cache_dir()
        assert cache_dir.is_absolute()
        assert cache_dir.exists()
    finally:
        os.chdir(original_cwd)
