"""Unit tests for transformer CUDA-OOM recovery control flow (CPU-only, in CI).

These tests guard the *control flow and result merging* of the actor's windowed
inference path and the *cleanup mechanism* of the core forward pass. They
deliberately do **not** measure GPU memory — they run on CPU with no CUDA and use
fakes/mocks, so they are safe in PR CI. Real GPU-memory reclamation is validated
separately by the GPU test (``tests/test_transformer_oom_recovery_gpu.py``, backed
by ``tests/oom_verification.py``).

Covered:
    1. Order + completeness across several window-batch shrink levels.
    2. Terminal single-window OOM raises without unbounded recursion/looping.
    3. Non-OOM RuntimeErrors propagate unchanged.
    4. CPU guard: works with torch.cuda.is_available() mocked False.
    5. The forward is retried over the **same** tokenized windows — the tokenizer
       is called exactly once even across an OOM cascade (no re-tokenize).
    6. A1 mechanism: an exception after the forward inside ``forward_windows``
       releases the GPU tensor locals (verified via weakref to the fake logits).

Windowing/coverage/offset correctness lives in ``test_transformer_windowing.py``.
"""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tide2.actors.transformer import TransformerInferenceActor
from tide2.transformers.core import TransformerCore

_OOM_MESSAGE = "CUDA out of memory. Tried to allocate 2.00 MiB."


class _FakeCore:
    """Stand-in for TransformerCore driving the actor's windowing path.

    ``tokenize_ragged`` emits a single content token per text spanning the whole
    string (offset ``(0, len)``); with a large token budget each text becomes one
    window, so window count tracks text count for control-flow assertions.
    ``forward_windows`` raises a CUDA-OOM ``RuntimeError`` when the window-batch is
    larger than ``oom_threshold`` and otherwise returns one marker prediction per
    window (``word`` == the original text). It records tokenize and forward calls.
    """

    num_special_tokens = 2

    def __init__(self, oom_threshold: int = 10**9, non_oom_error: bool = False) -> None:
        self.oom_threshold = oom_threshold
        self.non_oom_error = non_oom_error
        self.tokenize_calls = 0
        self.forward_sizes: list[int] = []

    def tokenize_ragged(self, texts):
        self.tokenize_calls += 1
        input_ids = [[0] for _ in texts]  # one content token per text
        offset_mapping = [[(0, len(t))] for t in texts]
        return {"input_ids": input_ids, "offset_mapping": offset_mapping}

    def forward_windows(self, windows):
        self.forward_sizes.append(len(windows))
        if self.non_oom_error:
            raise RuntimeError("some other non-memory error")
        if len(windows) > self.oom_threshold:
            raise RuntimeError(_OOM_MESSAGE)
        return [
            [
                {"entity": "B-X", "score": 1.0, "start": s, "end": e, "word": text[s:e], "index": i}
                for i, (s, e) in enumerate(offsets)
            ]
            for _content_ids, offsets, text in windows
        ]


def _make_actor(
    core: _FakeCore, *, budget: int = 10**9, overlap: int = 0, gpu_batch_size: int = 10**9
) -> TransformerInferenceActor:
    """Build a model-less actor wired to a fake core and a fixed GPU batch size.

    A huge ``budget`` puts each text in a single window (window count == text
    count), so ``gpu_batch_size`` alone drives slicing/shrink and the fake core
    alone decides OOM — no memory model is involved.
    """
    actor = TransformerInferenceActor.__new__(TransformerInferenceActor)
    actor._core = core
    actor._token_budget = budget
    actor._window_overlap = overlap
    actor._num_special_tokens = core.num_special_tokens
    actor._gpu_batch_size = gpu_batch_size
    actor._handled_oom_count = 0
    return actor


def _words(results: list[list[dict]]) -> list[str]:
    return [r[0]["word"] for r in results]


class TestOomRecoveryControlFlow:
    """Order, completeness, termination and error propagation."""

    def test_order_and_completeness_across_shrink_levels(self):
        # threshold=1 forces shrinking all the way down to single windows.
        core = _FakeCore(oom_threshold=1)
        actor = _make_actor(core)
        texts = [f"t{i}" for i in range(8)]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert len(results) == len(texts)
        assert _words(results) == texts  # original order preserved

    def test_order_preserved_with_mixed_shrink_levels(self):
        # threshold=2 => some 2-batches succeed, larger batches shrink further,
        # exercising merge-back across uneven shrink depths.
        core = _FakeCore(oom_threshold=2)
        actor = _make_actor(core)
        texts = [f"t{i}" for i in range(7)]  # odd count -> uneven halves

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _words(results) == texts

    def test_terminal_single_window_oom_raises(self):
        # threshold=0 => even a single window OOMs; recovery must give up with a
        # clear terminal error and must not recurse/loop forever.
        core = _FakeCore(oom_threshold=0)
        actor = _make_actor(core)

        with pytest.raises(RuntimeError, match="single token window"):
            actor._run_inference_raw_with_oom_recovery(["a", "b"])

    def test_non_oom_error_propagates_unchanged(self):
        core = _FakeCore(oom_threshold=100, non_oom_error=True)
        actor = _make_actor(core)

        with pytest.raises(RuntimeError, match="some other non-memory error"):
            actor._run_inference_raw_with_oom_recovery(["a", "b", "c"])
        # It failed on the first (whole-batch) forward without shrinking.
        assert core.forward_sizes == [3]

    @patch("tide2.actors.transformer.torch.cuda.is_available", return_value=False)
    def test_cpu_guard_no_cuda(self, mock_is_available):
        # With no CUDA, empty_cache() must be skipped but shrinking still works.
        core = _FakeCore(oom_threshold=1)
        actor = _make_actor(core)
        texts = [f"t{i}" for i in range(4)]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _words(results) == texts


class TestWindowBatchShrink:
    """OOM recovery shrinks the window-batch and reclaims cache between tries."""

    def test_batch_size_shrinks_on_oom(self):
        # threshold=1 forces shrinking down to single-window forwards.
        core = _FakeCore(oom_threshold=1)
        actor = _make_actor(core)
        texts = [f"t{i}" for i in range(8)]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert _words(results) == texts
        assert len(core.forward_sizes) > 1  # oversized attempt + shrink retries
        assert core.forward_sizes[-1] == 1  # final successful forwards are size 1
        assert actor._handled_oom_count > 0

    def test_tokenizer_called_once_across_oom_cascade(self):
        # The whole point of the rewrite: the forward retries over the SAME
        # tokenized windows, so the tokenizer runs exactly once per __call__.
        core = _FakeCore(oom_threshold=1)
        actor = _make_actor(core)

        actor._run_inference_raw_with_oom_recovery([f"t{i}" for i in range(8)])

        assert core.tokenize_calls == 1

    @patch("tide2.actors.transformer.torch.cuda.empty_cache")
    @patch("tide2.actors.transformer.torch.cuda.is_available", return_value=True)
    def test_empty_cache_called_only_after_handled_oom(self, mock_is_available, mock_empty_cache):
        # No OOM (threshold high) => empty_cache never called (off the success path).
        core = _FakeCore(oom_threshold=100)
        actor = _make_actor(core)
        actor._run_inference_raw_with_oom_recovery([f"t{i}" for i in range(4)])
        assert mock_empty_cache.call_count == 0

        # With OOMs, it is called once per handled OOM (before each retry).
        mock_empty_cache.reset_mock()
        core = _FakeCore(oom_threshold=1)
        actor = _make_actor(core)
        actor._run_inference_raw_with_oom_recovery([f"t{i}" for i in range(4)])
        assert mock_empty_cache.call_count >= 1


class TestA1Mechanism:
    """Mechanism check: forward_windows releases GPU tensor locals on error.

    Not a proof of GPU reclamation (that needs a real device) — it verifies that
    when a step after the forward raises, the tensor references held by the
    method's frame (here the model's ``logits``) are dropped in the ``finally``,
    so no exception traceback can pin them.
    """

    def test_logits_released_on_post_forward_exception(self):
        logits_holder: dict = {}

        class _FakeTokenizer:
            pad_token_id = 0

        class _FakeModel:
            def parameters(self):
                return iter([SimpleNamespace(device=torch.device("cpu"))])

            def __call__(self, **kwargs):
                logits = torch.zeros(1, 4, 2)  # weakref-able stand-in for GPU logits
                logits_holder["ref"] = weakref.ref(logits)
                return SimpleNamespace(logits=logits)

        core = TransformerCore.__new__(TransformerCore)
        core._model = _FakeModel()
        core._tokenizer = _FakeTokenizer()
        core._id2label = {0: "O", 1: "B-X"}
        core._ignore_labels_set = {"O"}
        # Pre-seed the special-token affixes so the probe encode is skipped (the
        # fake tokenizer only needs pad_token_id): [CLS] ... [SEP].
        core._special_affixes = ([101], [102])

        # The forward succeeds and binds ``logits``, then softmax raises OOM, so
        # the finally-block must drop the logits reference.
        with patch("torch.softmax", side_effect=RuntimeError(_OOM_MESSAGE)):
            with pytest.raises(RuntimeError, match="out of memory"):
                core.forward_windows([([5, 6], [(0, 1), (1, 2)], "ab")])

        gc.collect()
        assert logits_holder["ref"]() is None, "logits tensor was not released by the finally block"
