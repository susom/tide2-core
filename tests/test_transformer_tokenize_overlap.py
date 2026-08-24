"""Unit tests for in-actor tokenize/forward overlap (CPU-only, in CI).

Workstream G double-buffers tokenization against the GPU forward: while group N
runs its forward, group N+1 is tokenized on a background thread. These tests pin
correctness/ordering of the overlap path (and its OOM fallback) with a fake core
exposing the split ``tokenize_batch`` / ``forward_tokenized`` API — no model, no
CUDA — so they run in PR CI.
"""

from __future__ import annotations

import inspect

from tide2.actors.transformer import TransformerInferenceActor
from tide2.actors.transformer import create_transformer_actor
from tide2.runner.local_runner import LocalJobRunner


class _OverlapFakeCore:
    """Fake core with the split tokenize/forward API used by the overlap path.

    ``forward_tokenized`` raises a CUDA-OOM ``RuntimeError`` for any group whose
    text tuple is in ``oom_on`` (exercising the shrink fallback), otherwise
    returns per-text markers. All three entry points record their calls.
    """

    def __init__(self, oom_on: set[tuple[str, ...]] | None = None) -> None:
        self.oom_on = oom_on or set()
        self.tokenize_calls: list[list[str]] = []
        self.forward_calls: list[list[str]] = []
        self.infer_calls: list[list[str]] = []

    def tokenize_batch(self, texts):
        self.tokenize_calls.append(list(texts))
        return {"texts": list(texts)}  # stand-in encoding

    def forward_tokenized(self, encoded, texts):
        self.forward_calls.append(list(texts))
        if tuple(texts) in self.oom_on:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB.")
        return [[{"marker": t}] for t in texts]

    def infer_raw_direct(self, texts, batch_size=None):
        self.infer_calls.append(list(texts))
        return [[{"marker": t}] for t in texts]


def _make_actor(core: _OverlapFakeCore, cap) -> TransformerInferenceActor:
    actor = TransformerInferenceActor.__new__(TransformerInferenceActor)
    actor._core = core
    actor._tokenize_overlap = True
    actor._batch_cap_for_seq = cap
    return actor


def _markers(results):
    return [r[0]["marker"] for r in results]


class TestTokenizeOverlap:
    """The overlap path preserves order/completeness and prefetches tokenization."""

    def test_overlap_matches_sequential_output(self):
        core = _OverlapFakeCore()
        actor = _make_actor(core, cap=lambda _c: 2)
        texts = ["a" * n + f"#{i}" for i, n in enumerate([3, 1, 4, 1, 5, 9, 2, 6])]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _markers(results) == texts  # original order preserved
        # Every group was tokenized (prefetched) and forwarded exactly once.
        assert len(core.tokenize_calls) == len(core.forward_calls)
        processed = [t for call in core.forward_calls for t in call]
        assert sorted(processed) == sorted(texts)
        # No shrink fallback needed when nothing OOMs.
        assert core.infer_calls == []

    def test_overlap_oom_falls_back_to_shrink(self):
        # Force the first (shortest) group to OOM in forward_tokenized.
        core_probe = _OverlapFakeCore()
        actor_probe = _make_actor(core_probe, cap=lambda _c: 2)
        texts = ["a" * n for n in [1, 1, 2, 2, 3, 3]]
        groups = actor_probe._bucket_groups(texts)
        first_group_texts = tuple(texts[i] for i in groups[0])

        core = _OverlapFakeCore(oom_on={first_group_texts})
        actor = _make_actor(core, cap=lambda _c: 2)

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _markers(results) == texts  # still complete and ordered
        # The OOM group was retried via the shrink path (infer_raw_direct).
        assert core.infer_calls, "expected shrink fallback to run infer_raw_direct"

    def test_single_group_skips_overlap(self):
        # One group => overlap gives no benefit and the sequential path runs.
        core = _OverlapFakeCore()
        actor = _make_actor(core, cap=lambda _c: 10**9)
        texts = ["x", "y", "z"]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _markers(results) == texts
        # Sequential path uses the batch-shrink loop (infer_raw_direct), not the
        # overlap forward_tokenized path.
        assert core.forward_calls == []
        assert core.infer_calls == [["x", "y", "z"]]

    def test_overlap_oom_fallback_starts_at_half_group_size(self):
        # M9: the whole-group forward just OOMed, so the shrink fallback must
        # start at HALF the failed group size and skip the redundant oversized
        # retry. Two equal-length groups of 4; force the first to OOM.
        texts = [f"aa{i}" for i in range(8)]  # all length 3 -> stable order
        core_probe = _OverlapFakeCore()
        actor_probe = _make_actor(core_probe, cap=lambda _c: 4)
        groups = actor_probe._bucket_groups(texts)
        assert len(groups) == 2 and len(groups[0]) == 4  # precondition
        first_group_texts = tuple(texts[i] for i in groups[0])

        core = _OverlapFakeCore(oom_on={first_group_texts})
        actor = _make_actor(core, cap=lambda _c: 4)

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _markers(results) == texts  # complete and ordered
        # First shrink forward is half the group (2), not the full 4 that OOMed.
        assert core.infer_calls, "expected shrink fallback to run infer_raw_direct"
        assert len(core.infer_calls[0]) == 2
        assert core.infer_calls[0] == list(first_group_texts[:2])
        # No infer call ever runs at the failed (full) group size.
        assert all(len(call) <= 2 for call in core.infer_calls)


class _ShrinkFakeCore:
    """Fake core whose ``infer_raw_direct`` OOMs on chunks of >= 2 texts that
    contain a poison text, so the shrink loop's retry sizing can be observed.
    """

    def __init__(self, poison: str) -> None:
        self.poison = poison
        self.infer_calls: list[list[str]] = []

    def infer_raw_direct(self, texts, batch_size=None):
        self.infer_calls.append(list(texts))
        if self.poison in texts and len(texts) >= 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 MiB.")
        return [[{"marker": t}] for t in texts]


class TestBatchShrinkProgress:
    """The shrink loop derives the retry size from the failed chunk, not the
    stored batch size, so a tail chunk smaller than batch_size still shrinks."""

    def test_tail_oom_retry_derives_from_chunk_not_batch_size(self):
        texts = [f"t{i}" for i in range(6)]
        core = _ShrinkFakeCore(poison="t5")
        actor = TransformerInferenceActor.__new__(TransformerInferenceActor)
        actor._core = core

        # initial_batch_size=4 > the 2-text tail (t4,t5): the tail OOMs, and the
        # retry must drop to len(chunk)//2 == 1 rather than batch_size//2 == 2
        # (which would re-run the identical failing 2-text slice).
        results = actor._infer_group_with_batch_shrink(texts, initial_batch_size=4)

        assert [r[0]["marker"] for r in results] == texts  # complete and ordered
        sizes = [len(c) for c in core.infer_calls]
        assert sizes == [4, 2, 1, 1]  # the failed 2-text tail is never retried at size 2
        assert sizes.count(2) == 1


class TestTokenizeOverlapSignature:
    """H6: ``tokenize_overlap`` is keyword-only and last on every public entry."""

    def _tokenize_overlap_param(self, func):
        params = list(inspect.signature(func).parameters.values())
        by_name = {p.name: p for p in params}
        assert "tokenize_overlap" in by_name, f"tokenize_overlap missing from {func.__qualname__}"
        return by_name["tokenize_overlap"], params

    def test_constructor_keyword_only(self):
        param, _ = self._tokenize_overlap_param(TransformerInferenceActor.__init__)
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_factory_keyword_only(self):
        param, _ = self._tokenize_overlap_param(create_transformer_actor)
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_run_transformer_keyword_only(self):
        param, _ = self._tokenize_overlap_param(LocalJobRunner.run_transformer)
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_factory_rejects_extra_positional(self):
        # 8 positionals are allowed; a 9th (the old tokenize_overlap slot) must
        # not be accepted positionally now that it is keyword-only.
        import pytest

        with pytest.raises(TypeError):
            create_transformer_actor("m", None, None, None, None, None, None, True, True)  # ty: ignore[too-many-positional-arguments]
