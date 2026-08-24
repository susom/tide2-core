"""Unit tests for the dev-only transformer throughput harness (CPU-only, in CI).

The harness itself is not imported by product code, but its argument parsing is
pure Python and worth guarding: ``_parse_batch_sizes`` must reject non-positive
batch sizes (which would divide-by-zero or loop forever downstream) at parse
time rather than deep inside a GPU pass.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

# The harness lives under dev/ (outside the importable package), so load it by path.
_HARNESS_PATH = Path(__file__).resolve().parents[1] / "dev" / "transformer_throughput_harness.py"
_spec = importlib.util.spec_from_file_location("transformer_throughput_harness", _HARNESS_PATH)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


class TestParseBatchSizes:
    """_parse_batch_sizes splits/validates the repeatable --batch-size flag."""

    def test_default_when_empty(self) -> None:
        assert harness._parse_batch_sizes(None) == [8]
        assert harness._parse_batch_sizes([]) == [8]

    def test_comma_separated_and_repeated(self) -> None:
        assert harness._parse_batch_sizes(["8,16,32"]) == [8, 16, 32]
        assert harness._parse_batch_sizes(["8", "16"]) == [8, 16]
        assert harness._parse_batch_sizes(["8, 16 ", "32"]) == [8, 16, 32]

    @pytest.mark.parametrize("bad", ["0", "-1", "8,0", "4,-2,8"])
    def test_rejects_non_positive(self, bad: str) -> None:
        with pytest.raises(argparse.ArgumentTypeError):
            harness._parse_batch_sizes([bad])
