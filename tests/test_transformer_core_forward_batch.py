"""Regression test for TransformerCore._forward_batch_direct's vectorized postprocessing.

Verifies the numpy-vectorized implementation produces identical output to the
original per-token Python double-loop it replaced, using a fully mocked
model/tokenizer so no real weights are needed.
"""

from types import SimpleNamespace

import torch

from tide2.transformers.core import TransformerCore


class _FakeModel(torch.nn.Module):
    """Minimal nn.Module stand-in: returns fixed logits, ignores its inputs."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **_kwargs) -> SimpleNamespace:
        return SimpleNamespace(logits=self._logits)


class _FakeTokenizer:
    """Returns a fixed, pre-built encoding regardless of the input texts."""

    def __init__(self, encoded: dict):
        self._encoded = encoded

    def __call__(self, *_args, **_kwargs) -> dict:
        return dict(self._encoded)


def _naive_forward_batch_direct(
    texts: list[str],
    id2label: dict[int, str],
    ignore: set[str],
    scores_np,
    label_ids_np,
    offset_np,
    special_np,
) -> list[list[dict]]:
    """Pre-vectorization reference implementation, used to check for regressions."""
    results: list[list[dict]] = []
    for i, text in enumerate(texts):
        preds: list[dict] = []
        for j in range(scores_np.shape[1]):
            if special_np[i, j]:
                continue
            label = id2label[label_ids_np[i, j]]
            if label in ignore:
                continue
            s, e = int(offset_np[i, j, 0]), int(offset_np[i, j, 1])
            preds.append(
                {
                    "entity": label,
                    "score": float(scores_np[i, j]),
                    "start": s,
                    "end": e,
                    "word": text[s:e],
                    "index": j,
                }
            )
        results.append(preds)
    return results


def _build_core(model: _FakeModel, tokenizer: _FakeTokenizer, id2label: dict[int, str], ignore: set[str]):
    core = TransformerCore.__new__(TransformerCore)
    core._model = model
    core._tokenizer = tokenizer
    core._id2label = id2label
    core._ignore_labels_set = ignore
    core._pipeline = object()  # non-None: skips _ensure_pipeline_loaded's real model-loading path
    return core


class TestForwardBatchDirectVectorization:
    """_forward_batch_direct must match the original nested-loop semantics exactly."""

    def test_matches_naive_reference_implementation(self):
        texts = ["John lives", "NYC"]
        id2label = {0: "O", 1: "PERSON", 2: "LOCATION"}
        ignore = {"O"}

        offset_mapping = torch.tensor(
            [
                [[0, 0], [0, 4], [5, 10], [0, 0]],
                [[0, 0], [0, 3], [0, 0], [0, 0]],
            ]
        )
        special_tokens_mask = torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1]])

        logits = torch.zeros(2, 4, 3)
        logits[0, 1] = torch.tensor([0.0, 5.0, 0.0])  # -> PERSON
        logits[0, 2] = torch.tensor([5.0, 0.0, 0.0])  # -> O (ignored, not special)
        logits[1, 1] = torch.tensor([0.0, 0.0, 5.0])  # -> LOCATION

        encoded = {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "offset_mapping": offset_mapping,
            "special_tokens_mask": special_tokens_mask,
        }

        core = _build_core(_FakeModel(logits), _FakeTokenizer(encoded), id2label, ignore)
        actual = core._forward_batch_direct(texts)

        probs = torch.softmax(logits, dim=-1)
        scores_np = probs.max(dim=-1).values.numpy()
        label_ids_np = probs.max(dim=-1).indices.numpy()
        expected = _naive_forward_batch_direct(
            texts,
            id2label,
            ignore,
            scores_np,
            label_ids_np,
            offset_mapping.numpy(),
            special_tokens_mask.numpy(),
        )

        assert actual == expected

    def test_all_tokens_special_or_ignored_yields_empty_predictions(self):
        texts = ["hi"]
        id2label = {0: "O", 1: "PERSON"}
        ignore = {"O"}

        offset_mapping = torch.tensor([[[0, 0], [0, 2], [0, 0]]])
        special_tokens_mask = torch.tensor([[1, 0, 1]])
        logits = torch.zeros(1, 3, 2)
        logits[0, 1] = torch.tensor([5.0, 0.0])  # -> O, ignored

        encoded = {
            "input_ids": torch.zeros(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "offset_mapping": offset_mapping,
            "special_tokens_mask": special_tokens_mask,
        }

        core = _build_core(_FakeModel(logits), _FakeTokenizer(encoded), id2label, ignore)
        assert core._forward_batch_direct(texts) == [[]]


class _CharTokenizer:
    """Character-level fake tokenizer: pads to the longest text in the sub-batch,

    like a real tokenizer would, so different sub-batches naturally produce
    differently-shaped encodings.
    """

    def __call__(self, texts: list[str], **_kwargs) -> dict:
        max_len = max(len(t) for t in texts)
        batch = len(texts)
        offset_mapping = torch.zeros(batch, max_len, 2, dtype=torch.long)
        special_tokens_mask = torch.zeros(batch, max_len, dtype=torch.long)
        for i, text in enumerate(texts):
            for j in range(max_len):
                if j < len(text):
                    offset_mapping[i, j, 0] = j
                    offset_mapping[i, j, 1] = j + 1
                else:
                    special_tokens_mask[i, j] = 1  # padding
        return {
            "input_ids": torch.zeros(batch, max_len, dtype=torch.long),
            "attention_mask": torch.ones(batch, max_len, dtype=torch.long),
            "offset_mapping": offset_mapping,
            "special_tokens_mask": special_tokens_mask,
        }


class _AlwaysPersonModel(torch.nn.Module):
    """Fake model: every non-special token gets argmax label id=1 ("PERSON")."""

    def __init__(self):
        super().__init__()
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, **_kwargs) -> SimpleNamespace:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 2)
        logits[:, :, 1] = 5.0
        return SimpleNamespace(logits=logits)


class TestInferRawDirectSubBatching:
    """infer_raw_direct must not lose, reorder, or mix up sub-batches."""

    def test_multi_subbatch_matches_per_text_expectations(self):
        # Deliberately different lengths so each sub-batch pads to a different
        # width - if the pipeline ever paired the wrong encoding with the wrong
        # texts, the "word" slices below would come out wrong or raise.
        texts = ["a", "bb", "ccc", "dddd", "e"]
        core = _build_core(_AlwaysPersonModel(), _CharTokenizer(), {0: "O", 1: "PERSON"}, {"O"})

        results = core.infer_raw_direct(texts, batch_size=2)

        assert len(results) == len(texts)
        for text, preds in zip(texts, results, strict=True):
            assert [p["word"] for p in preds] == list(text)
            assert all(p["entity"] == "PERSON" for p in preds)

    def test_multi_subbatch_result_matches_sequential_calls(self):
        texts = ["alpha", "b", "gamma", "delta"]
        core = _build_core(_AlwaysPersonModel(), _CharTokenizer(), {0: "O", 1: "PERSON"}, {"O"})

        batched = core.infer_raw_direct(texts, batch_size=2)
        sequential = [pred for i in range(0, len(texts), 2) for pred in core._forward_batch_direct(texts[i : i + 2])]

        assert batched == sequential
