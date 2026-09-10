"""Unit tests for ``scripts/parquet_utils._resolve_parquet_files``.

These tests pin the behavior of :func:`_resolve_parquet_files`, with particular
focus on the glob-resolution rewrite that replaced the historical
``glob.glob(..., recursive=True)`` call with :meth:`pathlib.Path.glob`.  That
change was committed without a test even though it altered how absolute and
relative glob patterns are anchored, so the cases below lock in the expected
matches, sorting, and file-vs-directory filtering.

``scripts`` is not an importable package in this repo (no ``scripts/__init__``,
no ``pythonpath`` entry in ``pyproject.toml``, and no test currently imports
from it), so the module is loaded directly from disk via ``importlib`` rather
than with ``from scripts.parquet_utils import ...``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parent.parent / "scripts" / "parquet_utils.py"
_spec = importlib.util.spec_from_file_location("parquet_utils_under_test", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
parquet_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parquet_utils)

_resolve_parquet_files = parquet_utils._resolve_parquet_files


def _touch(path: Path) -> Path:
    """Create *path* (and parents) as an empty file and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


class TestResolveSingleFile:
    """A path pointing at an existing file returns just that file."""

    def test_single_parquet_file(self, tmp_path: Path) -> None:
        """A single existing ``.parquet`` file resolves to ``[that path]``."""
        f = _touch(tmp_path / "notes.parquet")

        result = _resolve_parquet_files(f)

        assert result == [f]

    def test_single_file_accepts_str_input(self, tmp_path: Path) -> None:
        """A file path passed as ``str`` resolves the same as a ``Path``."""
        f = _touch(tmp_path / "notes.parquet")

        result = _resolve_parquet_files(str(f))

        assert result == [Path(str(f))]

    def test_single_non_parquet_file_still_returned(self, tmp_path: Path) -> None:
        """The file branch returns any existing file regardless of extension."""
        f = _touch(tmp_path / "notes.txt")

        result = _resolve_parquet_files(f)

        assert result == [f]


class TestResolveDirectory:
    """A directory is searched recursively via ``rglob`` for parquet files."""

    def test_directory_returns_nested_parquet_sorted(self, tmp_path: Path) -> None:
        """Nested ``.parquet`` files are returned sorted; others excluded."""
        top = _touch(tmp_path / "a.parquet")
        nested = _touch(tmp_path / "sub" / "b.parquet")
        deeper = _touch(tmp_path / "sub" / "deep" / "c.parquet")
        _touch(tmp_path / "sub" / "notes.txt")
        _touch(tmp_path / "readme.md")

        result = _resolve_parquet_files(tmp_path)

        assert result == sorted([top, nested, deeper])

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """A directory with no parquet files yields an empty list."""
        (tmp_path / "empty").mkdir()

        result = _resolve_parquet_files(tmp_path / "empty")

        assert result == []

    def test_directory_excludes_non_parquet(self, tmp_path: Path) -> None:
        """Non-parquet files in a directory are never returned."""
        _touch(tmp_path / "data.csv")
        _touch(tmp_path / "data.json")
        keep = _touch(tmp_path / "data.parquet")

        result = _resolve_parquet_files(tmp_path)

        assert result == [keep]


class TestResolveRelativeGlob:
    """Relative glob patterns are anchored at the current working directory."""

    def test_relative_flat_glob(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``"*.parquet"`` matches files in cwd, sorted, non-parquet excluded."""
        a = _touch(tmp_path / "a.parquet")
        b = _touch(tmp_path / "b.parquet")
        _touch(tmp_path / "c.txt")
        # A nested file must NOT match a non-recursive flat pattern.
        _touch(tmp_path / "sub" / "d.parquet")
        monkeypatch.chdir(tmp_path)

        result = _resolve_parquet_files("*.parquet")

        assert result == sorted([Path("a.parquet"), Path("b.parquet")])
        assert Path("a.parquet") == a.relative_to(tmp_path)
        assert Path("b.parquet") == b.relative_to(tmp_path)

    def test_relative_recursive_glob(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``"**/*.parquet"`` matches nested files recursively, sorted."""
        top = _touch(tmp_path / "top.parquet")
        nested = _touch(tmp_path / "sub" / "n.parquet")
        deeper = _touch(tmp_path / "sub" / "deep" / "dd.parquet")
        _touch(tmp_path / "sub" / "note.txt")
        monkeypatch.chdir(tmp_path)

        result = _resolve_parquet_files("**/*.parquet")

        expected = sorted(
            [
                top.relative_to(tmp_path),
                nested.relative_to(tmp_path),
                deeper.relative_to(tmp_path),
            ]
        )
        assert result == expected

    def test_relative_glob_excludes_directories(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A relative pattern matching a directory drops it (``is_file`` only)."""
        keep = _touch(tmp_path / "keep.parquet")
        # A directory whose name also ends in ``.parquet`` must be filtered out.
        (tmp_path / "dir.parquet").mkdir()
        monkeypatch.chdir(tmp_path)

        result = _resolve_parquet_files("*.parquet")

        assert result == [keep.relative_to(tmp_path)]


class TestResolveAbsoluteGlob:
    """Absolute glob patterns are anchored at the filesystem root."""

    def test_absolute_recursive_glob(self, tmp_path: Path) -> None:
        """An absolute ``**`` pattern matches nested files across subdirs."""
        top = _touch(tmp_path / "top.parquet")
        nested = _touch(tmp_path / "sub" / "n.parquet")
        deeper = _touch(tmp_path / "sub" / "deep" / "dd.parquet")
        _touch(tmp_path / "sub" / "ignore.txt")

        pattern = str(tmp_path / "**" / "*.parquet")
        result = _resolve_parquet_files(pattern)

        assert result == sorted([top, nested, deeper])
        # Results are absolute, matching the absolute pattern input.
        assert all(m.is_absolute() for m in result)

    def test_absolute_flat_glob(self, tmp_path: Path) -> None:
        """An absolute flat pattern matches only the anchored directory level."""
        a = _touch(tmp_path / "a.parquet")
        b = _touch(tmp_path / "b.parquet")
        _touch(tmp_path / "sub" / "deep.parquet")

        pattern = str(tmp_path / "*.parquet")
        result = _resolve_parquet_files(pattern)

        assert result == sorted([a, b])

    def test_absolute_glob_no_match_returns_empty(self, tmp_path: Path) -> None:
        """An absolute pattern with no matches returns an empty list."""
        _touch(tmp_path / "a.txt")

        pattern = str(tmp_path / "*.parquet")
        result = _resolve_parquet_files(pattern)

        assert result == []


class TestResolveFiltersDirectories:
    """Glob patterns that match directories return only real files."""

    def test_absolute_glob_filters_directories(self, tmp_path: Path) -> None:
        """A ``*`` pattern matching both dirs and files keeps only files."""
        f1 = _touch(tmp_path / "one.parquet")
        f2 = _touch(tmp_path / "two.parquet")
        # Directories that also match the ``*`` glob must be excluded.
        (tmp_path / "a_subdir").mkdir()
        (tmp_path / "z_subdir").mkdir()

        pattern = str(tmp_path / "*")
        result = _resolve_parquet_files(pattern)

        assert result == sorted([f1, f2])
        assert all(m.is_file() for m in result)


class TestResolveNonStandaloneDoubleStar:
    """Patterns where ``**`` is not a whole path component still resolve.

    ``pathlib.Path.glob`` raises ``ValueError`` when ``**`` is embedded in a
    larger component (e.g. ``part-**.parquet``); the resolver must fall back to
    ``glob.glob(..., recursive=True)`` for those patterns rather than crashing.
    """

    def test_relative_partial_double_star_glob(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A relative ``part-**.parquet`` pattern matches ``part-*`` files."""
        a = _touch(tmp_path / "part-0.parquet")
        b = _touch(tmp_path / "part-1.parquet")
        _touch(tmp_path / "other.parquet")
        _touch(tmp_path / "part-2.txt")
        monkeypatch.chdir(tmp_path)

        result = _resolve_parquet_files("part-**.parquet")

        assert result == sorted([Path("part-0.parquet"), Path("part-1.parquet")])
        assert Path("part-0.parquet") == a.relative_to(tmp_path)
        assert Path("part-1.parquet") == b.relative_to(tmp_path)

    def test_absolute_partial_double_star_glob(self, tmp_path: Path) -> None:
        """An absolute ``part-**.parquet`` pattern resolves via the fallback."""
        a = _touch(tmp_path / "part-0.parquet")
        b = _touch(tmp_path / "part-1.parquet")
        _touch(tmp_path / "other.parquet")

        pattern = str(tmp_path / "part-**.parquet")
        result = _resolve_parquet_files(pattern)

        assert result == sorted([a, b])
        assert all(m.is_absolute() for m in result)

    def test_partial_double_star_filters_directories(self, tmp_path: Path) -> None:
        """The fallback still drops directories that match the pattern."""
        keep = _touch(tmp_path / "part-0.parquet")
        (tmp_path / "part-dir.parquet").mkdir()

        pattern = str(tmp_path / "part-**.parquet")
        result = _resolve_parquet_files(pattern)

        assert result == [keep]

    def test_partial_double_star_no_match_returns_empty(self, tmp_path: Path) -> None:
        """A non-standalone ``**`` pattern with no matches returns ``[]``."""
        _touch(tmp_path / "part-0.txt")

        pattern = str(tmp_path / "part-**.parquet")
        result = _resolve_parquet_files(pattern)

        assert result == []


class TestResolveOrdering:
    """Resolution order is deterministic (lexicographically sorted)."""

    def test_sorted_ordering_is_deterministic(self, tmp_path: Path) -> None:
        """Files created out of order are returned in sorted order."""
        created = [
            _touch(tmp_path / "z.parquet"),
            _touch(tmp_path / "m.parquet"),
            _touch(tmp_path / "a.parquet"),
            _touch(tmp_path / "sub" / "b.parquet"),
        ]

        result = _resolve_parquet_files(tmp_path)

        assert result == sorted(created)
        assert result == sorted(result)
