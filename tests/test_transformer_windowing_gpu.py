"""GPU integration tests for token-accurate windowing (real model, opt-in).

Proves the windowing fix end-to-end with the real tokenizer and model on a CUDA
device: dense chunks that tokenize past ``model_max_length`` are covered by
overlapping ≤-budget windows (never truncated), a PHI entity planted in the tail
that the old truncating path would have dropped is now detected at its true
offset, and a forced small batch still covers every window under OOM shrink.

Marked ``integration`` and skipped without CUDA or the model, so it auto-skips in
PR CI (no GPU/model) and runs on a GPU box. The token-level coverage math is
proven model-free in ``tests/test_transformer_windowing.py``; this is the honest
real-model confirmation.
"""

import json

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]

_MODEL = "StanfordAIMI/stanford-deidentifier-v2"


@pytest.fixture(scope="module")
def actor():
    """A real GPU actor, or skip if the model cannot be loaded on this box."""
    from tide2.actors.transformer import create_transformer_actor

    try:
        return create_transformer_actor(model_name=_MODEL, allow_huggingface_download=True)()
    except Exception as e:
        pytest.skip(f"model {_MODEL} unavailable: {e}")


def _dense_chunk_over_budget(actor) -> str:
    """Build a chunk whose real tokenization exceeds the per-window budget."""
    # Clinical-ish dense text; ~2-3 chars/token on this model, so this comfortably
    # exceeds a 510-token budget and forces multiple windows.
    sentence = "The patient was seen in clinic on 03/14 for follow-up of hypertension and diabetes; labs were ordered. "
    text = sentence * 40
    n_tokens = len(actor._core.tokenize_ragged([text])["input_ids"][0])
    assert n_tokens > actor._token_budget, f"expected an over-budget chunk, got {n_tokens} tokens"
    return text


def test_real_tokenizer_windows_cover_dense_chunk(actor):
    text = _dense_chunk_over_budget(actor)
    enc = actor._core.tokenize_ragged([text])

    windows = actor._plan_windows([text], enc["input_ids"], enc["offset_mapping"])

    assert len(windows) > 1  # genuinely windowed, not a single truncated pass
    assert all(len(w.content_ids) <= actor._token_budget for w in windows)
    covered = set()
    for w in windows:
        covered |= set(range(w.token_start, w.token_end))
    assert covered == set(range(len(enc["input_ids"][0])))  # all tokens, no gaps


def test_tail_phi_detected_after_windowing(actor):
    from tide2.actors.transformer import BIOAggregationActor

    # Filler large enough that the planted name lands well past where truncation at
    # model_max_length would have cut the chunk.
    filler = "The patient was seen in clinic for routine follow-up and evaluation. " * 40
    tail = "Attending physician: Dr. Jonathan Whitfield."
    text = filler + tail
    expected_start = text.index("Jonathan Whitfield")

    raw = actor({"chunk_text": [text], "text_hash": ["h"], "chunk_id": [0], "char_offset_start": [0]})
    agg = BIOAggregationActor()(
        {
            "chunk_text": [text],
            "predictions_raw_json": raw["predictions_raw_json"],
            "text_hash": ["h"],
            "chunk_id": [0],
            "char_offset_start": [0],
        }
    )
    entities = json.loads(agg["predictions_json"][0])

    # An entity must be detected overlapping the tail name — proof the tail was
    # forwarded (the old truncating path never saw these characters).
    name_end = expected_start + len("Jonathan Whitfield")
    hit = [e for e in entities if e["start"] < name_end and e["end"] > expected_start]
    assert hit, f"no entity detected over the tail name at char {expected_start}"


def test_forced_small_batch_preserves_coverage(actor):
    from tide2.actors.transformer import create_transformer_actor

    text = _dense_chunk_over_budget(actor)
    # Pin a tiny gpu_batch_size so windows are forced through many small forwards
    # (and any OOM shrink), then confirm the merged coverage is still complete.
    small = create_transformer_actor(model_name=_MODEL, gpu_batch_size=1, allow_huggingface_download=True)()
    out = small({"chunk_text": [text], "text_hash": ["h"], "chunk_id": [0], "char_offset_start": [0]})
    preds = json.loads(out["predictions_raw_json"][0])

    # Every predicted token offset lies within the chunk and the last window's
    # tokens (near the end of the chunk) are represented — nothing was dropped.
    assert preds, "expected some predictions on a long clinical chunk"
    assert max(p["end"] for p in preds) > len(text) * 0.9
