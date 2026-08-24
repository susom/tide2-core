"""Unit tests for transformer-actor configuration plumbing (CPU-only, in CI).

Covers the tokenizer-parallelism helper added in Workstream B: it must pin the
rayon thread pool and enable HuggingFace tokenizer parallelism via ``setdefault``
semantics (operator-set env always wins) and be a no-op for non-positive worker
counts.
"""

from __future__ import annotations

import pytest

from tide2.actors.transformer import TransformerInferenceActor


class TestConfigureTokenizerParallelism:
    """_configure_tokenizer_parallelism pins rayon + tokenizer env safely."""

    def test_sets_env_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
        monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

        TransformerInferenceActor._configure_tokenizer_parallelism(6)

        import os

        assert os.environ["RAYON_NUM_THREADS"] == "6"
        assert os.environ["TOKENIZERS_PARALLELISM"] == "true"

    def test_operator_set_env_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # setdefault must not clobber an operator-provided value.
        monkeypatch.setenv("RAYON_NUM_THREADS", "3")
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")

        TransformerInferenceActor._configure_tokenizer_parallelism(16)

        import os

        assert os.environ["RAYON_NUM_THREADS"] == "3"
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"

    @pytest.mark.parametrize("workers", [0, -1])
    def test_non_positive_is_noop(self, workers: int, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)

        TransformerInferenceActor._configure_tokenizer_parallelism(workers)

        import os

        assert "RAYON_NUM_THREADS" not in os.environ


class TestCpuFloor:
    """_cpu_floor rounds fractional Ray CPU reservations up to a worker floor."""

    @pytest.mark.parametrize(
        ("assigned", "expected"),
        [
            (0, 0),
            (0.0, 0),
            (None, 0),
            (0.25, 1),
            (0.5, 1),
            (0.99, 1),
            (1, 1),
            (1.0, 1),
            (1.5, 2),
            (2, 2),
            (2.0, 2),
            (4, 4),
        ],
    )
    def test_ceils_positive_fractions(self, assigned: float | None, expected: int) -> None:
        # A fractional pin (e.g. transformer_cpus=0.25) must count as >= 1 worker,
        # never truncate to 0 (which would leave the rayon pool at all cores).
        assert TransformerInferenceActor._cpu_floor(assigned) == expected
