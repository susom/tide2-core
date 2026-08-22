"""Unit tests for in-actor tokenize/forward overlap (CPU-only, in CI).

Workstream G double-buffers tokenization against the GPU forward: while group N
runs its forward, group N+1 is tokenized on a background thread. These tests pin
correctness/ordering of the overlap path (and its OOM fallback) with a fake core
exposing the split ``tokenize_batch`` / ``forward_tokenized`` API — no model, no
CUDA — so they run in PR CI.
"""

from __future__ import annotations

from tide2.actors.transformer import TransformerInferenceActor


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
