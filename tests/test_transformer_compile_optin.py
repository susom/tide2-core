"""Unit tests for strictly-opt-in torch.compile resolution (CPU-only, in CI).

These pin the behavior of
:meth:`tide2.transformers.core.TransformerCore._resolve_compile_cache_path`
after Workstream A: compilation is entered **only** on an explicit
``compile_model=True`` opt-in. The mere presence of a ``compiled_cache.bin``
beside the weights must never turn compilation on (that silent auto-enable was
the production leak footgun — review Finding #0).

No model is loaded: the tests build a bare ``TransformerCore`` via ``__new__``
and set only the attributes the resolver reads, matching the existing
mock-based test style in ``test_transformer_oom_recovery.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tide2.transformers.core import TransformerCore


def _make_core(model_path: Path, compile_model, compile_cache_path=None) -> TransformerCore:
    """Build a TransformerCore with only the fields the resolver reads."""
    core = TransformerCore.__new__(TransformerCore)
    core.model_path = str(model_path)
    core.compile_model = compile_model
    core.compile_cache_path = compile_cache_path
    return core


def _write_cache(model_path: Path) -> Path:
    """Create a stand-in compiled_cache.bin beside the weights."""
    cache = model_path / "compiled_cache.bin"
    cache.write_bytes(b"stub")
    return cache


class TestCompileOptIn:
    """compile_model gates compilation; cache presence alone never does."""

    def test_default_none_never_compiles_even_with_cache(self, tmp_path: Path) -> None:
        # The key regression: cache present but no opt-in => OFF (was auto-enabled).
        _write_cache(tmp_path)
        core = _make_core(tmp_path, compile_model=None)

        assert core._resolve_compile_cache_path() is None

    def test_default_none_no_cache_off(self, tmp_path: Path) -> None:
        core = _make_core(tmp_path, compile_model=None)

        assert core._resolve_compile_cache_path() is None

    def test_explicit_false_overrides_present_cache(self, tmp_path: Path) -> None:
        _write_cache(tmp_path)
        core = _make_core(tmp_path, compile_model=False)

        assert core._resolve_compile_cache_path() is None

    def test_opt_in_with_cache_returns_path(self, tmp_path: Path) -> None:
        cache = _write_cache(tmp_path)
        core = _make_core(tmp_path, compile_model=True)

        assert core._resolve_compile_cache_path() == cache

    def test_opt_in_missing_cache_raises(self, tmp_path: Path) -> None:
        core = _make_core(tmp_path, compile_model=True)

        with pytest.raises(FileNotFoundError, match="compile_model=True"):
            core._resolve_compile_cache_path()

    def test_opt_in_honors_cache_path_override(self, tmp_path: Path) -> None:
        override = tmp_path / "elsewhere" / "custom_cache.bin"
        override.parent.mkdir()
        override.write_bytes(b"stub")
        # No compiled_cache.bin beside the (empty) model dir; override wins.
        core = _make_core(tmp_path, compile_model=True, compile_cache_path=str(override))

        assert core._resolve_compile_cache_path() == override

    def test_override_missing_still_raises_on_opt_in(self, tmp_path: Path) -> None:
        override = tmp_path / "missing_cache.bin"
        core = _make_core(tmp_path, compile_model=True, compile_cache_path=str(override))

        with pytest.raises(FileNotFoundError, match=r"missing_cache\.bin"):
            core._resolve_compile_cache_path()
