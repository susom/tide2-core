"""Unit tests for in-actor length bucketing + reassembly (CPU-only, in CI).

Workstream C groups incoming texts by length before each GPU forward (to cut
padding waste) and reassembles outputs into the original input order. These tests
pin that contract with a recording fake core and a stubbed batch cap — no model,
no CUDA — so they run in PR CI. OOM/batch-shrink behavior lives in
``test_transformer_oom_recovery.py``.
"""

from __future__ import annotations

from tide2.actors.transformer import _GROUP_SPAN_MIN_ANCHOR_CHARS
from tide2.actors.transformer import _MAX_GROUP_LENGTH_SPAN
from tide2.actors.transformer import TransformerInferenceActor


class _RecordingCore:
    """Fake core that records the texts of each forward and returns markers."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def infer_raw_direct(self, texts, batch_size=None):
        self.calls.append(list(texts))
        return [[{"marker": t}] for t in texts]


def _make_actor(core: _RecordingCore, cap) -> TransformerInferenceActor:
    """Build a model-less actor wired to a fake core and a stubbed length cap.

    ``cap`` is a callable ``max_chars -> int`` used as ``_batch_cap_for_seq`` so
    grouping is driven entirely by the test, not the real memory model.
    """
    actor = TransformerInferenceActor.__new__(TransformerInferenceActor)
    actor._core = core
    actor._batch_cap_for_seq = cap
    return actor


def _markers(results: list[list[dict]]) -> list[str]:
    return [r[0]["marker"] for r in results]


class TestBucketingReassembly:
    """Output order + completeness across the bucketing/reassembly contract."""

    def test_order_preserved_single_group(self):
        # Huge cap => one group, but bucketing still length-sorts internally;
        # reassembly must restore the ORIGINAL input order.
        core = _RecordingCore()
        actor = _make_actor(core, cap=lambda _c: 10**9)
        texts = ["aaaa", "a", "aaaaaaaa", "aa", "aaa"]  # lengths 4,1,8,2,3

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _markers(results) == texts  # original order, not length order
        # The single forward received the texts length-sorted (padding win).
        assert len(core.calls) == 1
        assert core.calls[0] == ["a", "aa", "aaa", "aaaa", "aaaaaaaa"]

    def test_multiple_groups_cover_every_text_once(self):
        # Fixed small cap => several groups; nothing dropped or duplicated.
        core = _RecordingCore()
        actor = _make_actor(core, cap=lambda _c: 2)
        texts = [f"{'x' * (i % 5 + 1)}#{i}" for i in range(11)]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _markers(results) == texts  # original order preserved
        processed = [t for call in core.calls for t in call]
        assert sorted(processed) == sorted(texts)  # exact cover, no dupes/drops
        # Each group respects the cap of 2.
        assert all(len(call) <= 2 for call in core.calls)

    def test_groups_are_length_homogeneous(self):
        # Each forward's texts must be contiguous in the global length ordering.
        core = _RecordingCore()
        actor = _make_actor(core, cap=lambda _c: 3)
        texts = ["a" * n for n in [7, 1, 3, 9, 2, 5, 4, 8, 6]]

        actor._run_inference_raw_with_oom_recovery(texts)

        # Concatenating the groups reproduces the fully length-sorted order.
        flat = [t for call in core.calls for t in call]
        assert flat == sorted(texts, key=len)
        # Within every group, lengths are non-decreasing (homogeneous slice).
        for call in core.calls:
            lengths = [len(t) for t in call]
            assert lengths == sorted(lengths)

    def test_all_same_length_single_group(self):
        core = _RecordingCore()
        actor = _make_actor(core, cap=lambda _c: 10**9)
        texts = ["abc", "def", "ghi", "jkl"]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _markers(results) == texts
        assert len(core.calls) == 1

    def test_single_row(self):
        core = _RecordingCore()
        actor = _make_actor(core, cap=lambda _c: 5)

        results = actor._run_inference_raw_with_oom_recovery(["only"])

        assert _markers(results) == ["only"]
        assert core.calls == [["only"]]

    def test_empty_input(self):
        core = _RecordingCore()
        actor = _make_actor(core, cap=lambda _c: 5)

        results = actor._run_inference_raw_with_oom_recovery([])

        assert results == []
        assert core.calls == []

    def test_duplicate_texts_not_deduplicated(self):
        # Identical texts must each get their own aligned result (no collapsing).
        core = _RecordingCore()
        actor = _make_actor(core, cap=lambda _c: 2)
        texts = ["same", "same", "same"]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _markers(results) == texts
        assert len(results) == 3


class TestLengthSpanBoundary:
    """The span bound splits wide-length batches even when the memory cap fits all.

    This is the M8 fix: when ``_batch_cap_for_seq`` already admits the whole batch
    (cap >= batch size) the memory model alone leaves one big group, padding short
    texts up to the batch's single longest. ``_MAX_GROUP_LENGTH_SPAN`` caps that
    in-group padding waste.
    """

    def _text(self, chars: int) -> str:
        return "x" * chars

    def test_wide_span_splits_under_infinite_cap(self):
        # Cap admits everything, so ONLY the span bound can force a split.
        core = _RecordingCore()
        actor = _make_actor(core, cap=lambda _c: 10**9)
        # Anchor for the first (shortest) group is max(200, floor); 200*1.5 = 300,
        # so 200/250 group together and 400/500 form a second group.
        assert _MAX_GROUP_LENGTH_SPAN == 1.5
        assert _GROUP_SPAN_MIN_ANCHOR_CHARS <= 200
        texts = [self._text(n) for n in (250, 200, 500, 400)]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _markers(results) == texts  # original order preserved
        assert len(core.calls) == 2
        assert [len(t) for t in core.calls[0]] == [200, 250]
        assert [len(t) for t in core.calls[1]] == [400, 500]

    def test_within_span_stays_one_group(self):
        # Longest within 1.5x of shortest (and above the anchor) => single forward.
        core = _RecordingCore()
        actor = _make_actor(core, cap=lambda _c: 10**9)
        texts = [self._text(n) for n in (200, 260, 300)]  # 300/200 = 1.5x exactly

        actor._run_inference_raw_with_oom_recovery(texts)

        assert len(core.calls) == 1

    def test_short_texts_ignore_span(self):
        # Below the anchor, a wide ratio must NOT split (padding waste is trivial).
        core = _RecordingCore()
        actor = _make_actor(core, cap=lambda _c: 10**9)
        small = _GROUP_SPAN_MIN_ANCHOR_CHARS // 8 or 1
        texts = [self._text(small), self._text(small * 6)]  # 6x ratio, both tiny

        actor._run_inference_raw_with_oom_recovery(texts)

        assert len(core.calls) == 1

    def test_span_and_cap_compose(self):
        # A tight cap can still split a span-homogeneous run into cap-sized forwards.
        core = _RecordingCore()
        actor = _make_actor(core, cap=lambda _c: 2)
        texts = [self._text(n) for n in (200, 210, 220, 230)]  # all within span

        actor._run_inference_raw_with_oom_recovery(texts)

        # Span keeps them together, but cap=2 forces 2-row forwards.
        assert all(len(call) <= 2 for call in core.calls)
        flat = [t for call in core.calls for t in call]
        assert sorted(flat, key=len) == sorted(texts, key=len)
