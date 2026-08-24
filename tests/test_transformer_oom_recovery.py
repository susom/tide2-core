"""Unit tests for transformer CUDA-OOM recovery control flow (CPU-only, in CI).

These tests guard the *control flow and result reassembly* of the OOM-recovery
path and the *cleanup mechanism* of the direct forward pass. They deliberately do
**not** measure GPU memory — they run on CPU with no CUDA and use fakes/mocks, so
they are safe in PR CI. Real GPU-memory reclamation is validated separately by the
GPU test (``tests/test_transformer_oom_recovery_gpu.py``, backed by
``tests/oom_verification.py``).

Covered:
    1. Order + completeness across several batch-shrink levels.
    2. Terminal single-text OOM raises without unbounded recursion/looping.
    3. Non-OOM RuntimeErrors propagate unchanged.
    4. CPU guard: works with torch.cuda.is_available() mocked False.
    5. A1 mechanism: an exception inside _forward_batch_direct releases the GPU
       tensor locals (verified via weakref to a stand-in tensor).

Note: recovery now shrinks the effective batch size (single owner of
sub-batching) instead of recursively splitting text slices, so the previous
"at most one live exception" / "empty_cache outside the handler" mechanism
tests are gone — Fix A1 (source-level tensor release) makes those properties
irrelevant.
"""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from tide2.actors.transformer import TransformerInferenceActor
from tide2.transformers.core import TransformerCore

_OOM_MESSAGE = "CUDA out of memory. Tried to allocate 2.00 MiB."


class _FakeCore:
    """Stand-in for TransformerCore.

    ``infer_raw_direct`` raises a CUDA-OOM ``RuntimeError`` when the chunk is
    larger than ``oom_threshold`` and otherwise returns a deterministic per-text
    marker. It records the size of each call for control-flow assertions.
    """

    def __init__(self, oom_threshold: int, non_oom_error: bool = False) -> None:
        self.oom_threshold = oom_threshold
        self.non_oom_error = non_oom_error
        self.call_sizes: list[int] = []

    def infer_raw_direct(self, texts, batch_size=None):
        self.call_sizes.append(len(texts))
        if self.non_oom_error:
            raise RuntimeError("some other non-memory error")
        if len(texts) > self.oom_threshold:
            raise RuntimeError(_OOM_MESSAGE)
        return [[{"marker": t}] for t in texts]


def _make_actor(core: _FakeCore) -> TransformerInferenceActor:
    """Build an actor without loading a model, wired to a fake core.

    ``_effective_batch_size`` is stubbed so recovery does not touch the real
    (model-config-dependent) sizing logic — the fake core alone decides OOM.
    """
    actor = TransformerInferenceActor.__new__(TransformerInferenceActor)
    actor._core = core
    # Recovery only needs a per-slice batch size; feed the whole slice each time so
    # the fake core alone decides OOM. (len(slice) == effective batch here.)
    actor._effective_batch_size = len
    return actor


class TestOomRecoveryControlFlow:
    """Order, completeness, termination and error propagation."""

    def test_order_and_completeness_across_split_levels(self):
        # threshold=1 forces splitting all the way down to single texts.
        core = _FakeCore(oom_threshold=1)
        actor = _make_actor(core)
        texts = [f"t{i}" for i in range(8)]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert len(results) == len(texts)
        assert [r[0]["marker"] for r in results] == texts  # original order preserved

    def test_order_preserved_with_mixed_split_levels(self):
        # threshold=2 => some 2-slices succeed, some larger slices split further,
        # exercising reassembly across uneven split depths.
        core = _FakeCore(oom_threshold=2)
        actor = _make_actor(core)
        texts = [f"t{i}" for i in range(7)]  # odd count -> uneven halves

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert [r[0]["marker"] for r in results] == texts

    def test_terminal_single_text_oom_raises(self):
        # threshold=0 => even a single text OOMs; recovery must give up with a
        # clear terminal error and must not recurse/loop forever.
        core = _FakeCore(oom_threshold=0)
        actor = _make_actor(core)

        with pytest.raises(RuntimeError, match="single text chunk"):
            actor._run_inference_raw_with_oom_recovery(["a", "b"])

    def test_non_oom_error_propagates_unchanged(self):
        core = _FakeCore(oom_threshold=100, non_oom_error=True)
        actor = _make_actor(core)

        with pytest.raises(RuntimeError, match="some other non-memory error"):
            actor._run_inference_raw_with_oom_recovery(["a", "b", "c"])
        # It failed on the first (whole-batch) attempt without splitting.
        assert core.call_sizes == [3]

    @patch("tide2.actors.transformer.torch.cuda.is_available", return_value=False)
    def test_cpu_guard_no_cuda(self, mock_is_available):
        # With no CUDA, empty_cache() must be skipped but splitting still works.
        core = _FakeCore(oom_threshold=1)
        actor = _make_actor(core)
        texts = [f"t{i}" for i in range(4)]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        assert [r[0]["marker"] for r in results] == texts


class TestBatchShrinkRecovery:
    """OOM recovery halves the GPU batch size and reclaims cache between tries."""

    def test_batch_size_shrinks_on_oom(self):
        # threshold=1 forces shrinking down to single-text chunks.
        core = _FakeCore(oom_threshold=1)
        actor = _make_actor(core)
        texts = [f"t{i}" for i in range(8)]

        results = actor._run_inference_raw_with_oom_recovery(texts)

        # Completeness/order preserved, and more than one forward happened
        # (the initial oversized chunk plus shrink retries).
        assert [r[0]["marker"] for r in results] == texts
        assert len(core.call_sizes) > 1
        # The final successful chunks are size 1 (shrunk from the full batch).
        assert core.call_sizes[-1] == 1
        # Every shrink is a handled OOM, so the counter must have advanced.
        assert actor._handled_oom_count > 0

    def test_handled_oom_counter_stays_zero_without_oom(self):
        # No OOM (threshold high) => the recovery path never fires, so the
        # handled-OOM counter must stay at zero.
        core = _FakeCore(oom_threshold=100)
        actor = _make_actor(core)
        actor._run_inference_raw_with_oom_recovery([f"t{i}" for i in range(4)])
        assert getattr(actor, "_handled_oom_count", 0) == 0

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


class _MovedTensor:
    """Weakref-able stand-in for a device-resident tensor."""


class _StandInTensor:
    """Minimal CPU stand-in whose ``.to()`` yields a fresh, weakref-able object."""

    def __init__(self, moved_refs: list) -> None:
        self._moved_refs = moved_refs

    def to(self, _device):
        moved = _MovedTensor()
        self._moved_refs.append(weakref.ref(moved))
        return moved


class TestA1Mechanism:
    """Mechanism check: _forward_batch_direct releases GPU tensor locals on error.

    Not a proof of GPU reclamation (that needs a real device) — it verifies that
    when the forward pass raises, the moved-to-device tensor references held by
    the method's frame are dropped, so no exception traceback can pin them.
    """

    def test_moved_tensors_released_on_forward_exception(self):
        moved_refs: list = []

        class _FakeEncoding(dict):
            pass

        def fake_tokenizer(texts, **kwargs):
            return _FakeEncoding(
                {
                    "input_ids": _StandInTensor(moved_refs),
                    "attention_mask": _StandInTensor(moved_refs),
                    "offset_mapping": Mock(),
                    "special_tokens_mask": Mock(),
                }
            )

        class _FakeModel:
            def parameters(self):
                return iter([SimpleNamespace(device=torch.device("cpu"))])

            def __call__(self, **kwargs):
                raise RuntimeError(_OOM_MESSAGE)

        core = TransformerCore.__new__(TransformerCore)
        core._model = _FakeModel()
        core._tokenizer = fake_tokenizer
        core._id2label = {0: "O"}
        core._ignore_labels_set = {"O"}

        with pytest.raises(RuntimeError, match="out of memory"):
            core._forward_batch_direct(["some text"])

        # The forward moved two tensors to the device before raising.
        assert len(moved_refs) == 2
        # Drop the traceback we might hold, then collect: the finally-block del of
        # the frame locals must leave nothing referencing the moved tensors.
        gc.collect()
        assert all(ref() is None for ref in moved_refs), "moved GPU tensors were not released"
