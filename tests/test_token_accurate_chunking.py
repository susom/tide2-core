"""Tests for the unified token-accurate chunking path.

Covers the three pillars of the "one length authority, one chunker" rework:

1. **Length authority** — every shipped model config pins ``MODEL_MAX_LENGTH`` and
   resolves to a finite, positive ``token_budget``; a config missing it *raises*
   instead of silently falling back to the tokenizer sentinel (the bug that let
   dense chunks run over-length).
2. **Shared windowing primitive** — ``plan_windows`` keeps every window within
   budget, covers all tokens with no gaps, carries exact char offsets, honors the
   overlap, and splits an over-budget note into ≥2 windows — for both dense
   (~1.9 char/tok) and sparse (~5 char/tok) inputs.
3. **Reassembly fold** — the per-note aggregation stage (aggregate → dedup →
   format) emits the document-level ``recognizer_results_json`` shape the old
   chunk→reassembly path produced, collapsing the duplicates window overlap
   creates.
"""

import json
from itertools import pairwise

import pytest

from tide2.transformers.config import get_available_models
from tide2.transformers.config import load_model_config
from tide2.transformers.core import TransformerCore
from tide2.transformers.core import plan_windows

# ---------------------------------------------------------------------------
# 1. Single length authority: MODEL_MAX_LENGTH -> token_budget, or raise
# ---------------------------------------------------------------------------


class TestLengthAuthority:
    def test_every_shipped_model_pins_positive_model_max_length(self):
        models = get_available_models()
        assert models  # the config is not empty
        for name in models:
            cfg = load_model_config(name)
            mml = cfg.get("MODEL_MAX_LENGTH")
            assert isinstance(mml, int) and mml > 0, f"{name} has bad MODEL_MAX_LENGTH={mml!r}"

    def test_missing_model_max_length_raises(self):
        # Build a core without touching the network/model, then drive the resolver.
        core = TransformerCore.__new__(TransformerCore)
        core.model_name = "no-mml-model"
        core._config = {"PRESIDIO_SUPPORTED_ENTITIES": []}  # no MODEL_MAX_LENGTH
        with pytest.raises(ValueError, match="MODEL_MAX_LENGTH"):
            _ = core.model_max_length

    def test_nonpositive_model_max_length_raises(self):
        core = TransformerCore.__new__(TransformerCore)
        core.model_name = "bad-mml-model"
        core._config = {"MODEL_MAX_LENGTH": 0}
        with pytest.raises(ValueError, match="non-positive"):
            _ = core.model_max_length

    def test_token_budget_subtracts_special_tokens(self):
        core = TransformerCore.__new__(TransformerCore)
        core.model_name = "m"
        core._config = {"MODEL_MAX_LENGTH": 512}
        # Stub the tokenizer path num_special_tokens uses, and mark loaded so
        # _ensure_pipeline_loaded is a no-op.
        core._pipeline = object()

        class _Tok:
            def num_special_tokens_to_add(self, pair=False):
                return 2

        core._tokenizer = _Tok()
        assert core.model_max_length == 512
        assert core.num_special_tokens == 2
        assert core.token_budget == 510  # 512 - 2

    def test_invalid_dtype_raises_actionable_error(self, monkeypatch):
        # A typo'd DTYPE must fail with a clear ValueError naming the config key,
        # not a bare AttributeError from getattr(torch, ...). Raised in __init__
        # before any model/network access, so no model load is needed.
        import tide2.transformers.core as core_mod

        monkeypatch.setattr(core_mod, "load_model_config", lambda _n: {"MODEL_MAX_LENGTH": 512, "DTYPE": "flaot16"})
        with pytest.raises(ValueError, match="DTYPE"):
            core_mod.TransformerCore(model_name="typo-dtype-model", load_immediately=False)


# ---------------------------------------------------------------------------
# 2. Shared windowing primitive: plan_windows
# ---------------------------------------------------------------------------


def _fake_tokenize(text: str, chars_per_token: float) -> tuple[list[int], list[tuple[int, int]]]:
    """Tile ``text`` into contiguous tokens of ~chars_per_token chars each.

    Returns (input_ids, offsets) exactly as a fast tokenizer's ragged output
    would, so plan_windows can be exercised without a model. ``chars_per_token``
    may be fractional to hit dense (1.9) / sparse (5) densities.
    """
    ids: list[int] = []
    offs: list[tuple[int, int]] = []
    k = 0
    tid = 0
    n = len(text)
    while k < n:
        end = min(k + max(1, round(chars_per_token)), n)
        ids.append(tid)
        offs.append((k, end))
        tid += 1
        k = end
    return ids, offs


def _covered_tokens(windows) -> set[int]:
    covered: set[int] = set()
    for w in windows:
        covered |= set(range(w.token_start, w.token_end))
    return covered


class TestPlanWindowsPrimitive:
    @pytest.mark.parametrize("chars_per_token", [1, 2, 5])
    def test_windows_respect_budget_and_cover_all_tokens(self, chars_per_token):
        budget, overlap = 100, 20
        text = "x" * 4000
        ids, offs = _fake_tokenize(text, chars_per_token)
        n_tokens = len(ids)

        windows = plan_windows([text], [ids], [offs], budget, overlap)

        assert n_tokens > budget  # genuinely over budget for every density here
        assert len(windows) >= 2  # so it must split
        assert all(len(w.content_ids) <= budget for w in windows)
        assert _covered_tokens(windows) == set(range(n_tokens))  # no gaps

    def test_char_offsets_are_exact(self):
        budget, overlap = 50, 10
        text = "abcdefghij" * 40  # 400 chars
        ids, offs = _fake_tokenize(text, 2)
        windows = plan_windows([text], [ids], [offs], budget, overlap)
        for w in windows:
            for tok_off in w.offsets:
                s, e = tok_off
                # The window's text is the whole note, offsets index into it.
                assert w.text[s:e] == text[s:e]

    def test_adjacent_windows_overlap_by_step(self):
        budget, overlap = 100, 20
        ids, offs = _fake_tokenize("y" * 1000, 1)  # 1000 tokens
        windows = plan_windows(["y" * 1000], [ids], [offs], budget, overlap)
        step = budget - overlap
        for prev, curr in pairwise(windows):
            assert curr.token_start - prev.token_start == step

    def test_within_budget_single_window(self):
        ids, offs = _fake_tokenize("z" * 30, 1)  # 30 tokens < budget
        windows = plan_windows(["z" * 30], [ids], [offs], 100, 20)
        assert len(windows) == 1
        assert (windows[0].token_start, windows[0].token_end) == (0, 30)

    def test_zero_tokens_no_window(self):
        assert plan_windows([""], [[]], [[]], 100, 20) == []

    def test_overlap_clamped_below_budget(self):
        # overlap >= budget would deadlock the step; plan_windows must clamp it.
        ids, offs = _fake_tokenize("q" * 500, 1)
        windows = plan_windows(["q" * 500], [ids], [offs], 100, 999)
        assert all(len(w.content_ids) <= 100 for w in windows)
        assert _covered_tokens(windows) == set(range(500))  # still full coverage


# ---------------------------------------------------------------------------
# 3. Reassembly fold: per-note aggregation -> recognizer_results_json
# ---------------------------------------------------------------------------


class TestAggregationFold:
    def _actor(self):
        from tide2.actors.transformer import BIOAggregationActor

        return BIOAggregationActor("MODEL_X")

    def test_emits_presidio_recognizer_result_shape(self):
        note_text = "John lives in Seattle"
        raw = json.dumps([{"entity": "B-PERSON", "score": 0.9, "start": 0, "end": 4, "word": "John", "index": 1}])
        results_json, count = self._actor()._format_note(raw, note_text)
        assert count == 1
        (e,) = json.loads(results_json)
        # Exact contract the anonymizer consumes (same shape reassembly produced).
        assert e["entity_type"] == "PERSON"
        assert (e["start"], e["end"]) == (0, 4)
        assert e["score"] == pytest.approx(0.9)
        assert e["analysis_explanation"] is None
        meta = e["recognition_metadata"]
        assert meta["recognizer_name"] == "TransformersRecognizer[MODEL_X]"
        assert meta["matched_pattern"] == "John"
        assert meta["recognizer_identifier"].startswith("TransformersRecognizer[MODEL_X]_")

    def test_overlap_region_duplicates_are_deduped(self):
        # The same PHI span surfaces from two overlapping windows with different
        # token ``index`` values (so the raw-token tuple dedup can't collapse it).
        # The span-level IoU dedup in aggregation must leave exactly one entity.
        note_text = "Patient Jane Doe was seen."
        start = note_text.index("Jane Doe")
        end = start + len("Jane Doe")
        mid = start + len("Jane")
        raw = json.dumps(
            [
                {"entity": "B-PATIENT", "score": 0.95, "start": start, "end": mid, "word": "Jane", "index": 3},
                {"entity": "I-PATIENT", "score": 0.90, "start": mid + 1, "end": end, "word": "Doe", "index": 4},
                # Duplicate copies from the overlapping next window (different index).
                {"entity": "B-PATIENT", "score": 0.95, "start": start, "end": mid, "word": "Jane", "index": 1},
                {"entity": "I-PATIENT", "score": 0.90, "start": mid + 1, "end": end, "word": "Doe", "index": 2},
            ]
        )
        results_json, count = self._actor()._format_note(raw, note_text)
        entities = json.loads(results_json)
        assert count == 1, f"overlap duplicates not collapsed: {entities}"
        assert (entities[0]["start"], entities[0]["end"]) == (start, end)
        assert entities[0]["entity_type"] == "PATIENT"
        assert entities[0]["recognition_metadata"]["matched_pattern"] == "Jane Doe"

    def test_empty_and_missing_inputs(self):
        actor = self._actor()
        assert actor._format_note("[]", "some text") == ("[]", 0)
        assert actor._format_note(json.dumps([{"entity": "B-X", "score": 1.0, "start": 0, "end": 1}]), "") == ("[]", 0)

    def test_call_batch_contract(self):
        actor = self._actor()
        note_text = "Call Dr. Smith."
        raw = json.dumps([{"entity": "B-DOCTOR", "score": 0.8, "start": 9, "end": 14, "word": "Smith", "index": 3}])
        out = actor(
            {
                "note_text": [note_text],
                "predictions_raw_json": [raw],
                "text_hash": ["h0"],
                "patient_id": ["p0"],
            }
        )
        assert out["text_hash"] == ["h0"]
        assert out["patient_id"] == ["p0"]
        assert out["entity_count"] == [1]
        assert out["note_text"] == [note_text]
        assert len(out["processing_timestamp"]) == 1
        (e,) = json.loads(out["recognizer_results_json"][0])
        assert e["entity_type"] == "DOCTOR"
        assert e["recognition_metadata"]["matched_pattern"] == "Smith"

    def test_openmed_label_mapping_and_ignore_labels(self):
        """OpenMed models map lower-case labels to Presidio entities and drop ignore labels."""
        from tide2.actors.transformer import BIOAggregationActor

        actor = BIOAggregationActor("OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1")
        note_text = "Jane was born on 1990-01-01 with blood type O positive."
        raw = json.dumps(
            [
                {"entity": "B-first_name", "score": 0.99, "start": 0, "end": 4, "word": "Jane", "index": 1},
                {
                    "entity": "B-date_of_birth",
                    "score": 0.95,
                    "start": 17,
                    "end": 27,
                    "word": "1990-01-01",
                    "index": 4,
                },
                {"entity": "B-blood_type", "score": 0.90, "start": 33, "end": 43, "word": "blood type", "index": 6},
            ]
        )
        results_json, count = actor._format_note(raw, note_text)
        entities = json.loads(results_json)

        assert count == 2
        entity_types = {e["entity_type"] for e in entities}
        assert entity_types == {"PERSON", "DATE_TIME"}
        assert "blood_type" not in entity_types


def test_dense_text_parity_transformers_recognizer_and_actor():
    """Exact parity between TransformersRecognizer and TransformerInferenceActor on token-dense text."""
    from tide2.actors.transformer import TransformerInferenceActor
    from tide2.recognizers.transformers_recognizer import TransformersRecognizer

    dense_text = "Patient #9876-5432: BP 120/80, HR 72, DOCTOR Smith seen on 2026-05-12."

    class _TunableCore:
        model_name = "StanfordAIMI/stanford-deidentifier-v2"
        model_max_length = 20
        num_special_tokens = 2
        token_budget = 18

        def __init__(self):
            self.model_path = None
            self._config = {
                "MODEL_MAX_LENGTH": 20,
                "PRESIDIO_SUPPORTED_ENTITIES": ["PATIENT", "DOCTOR", "DATE", "ID"],
                "LABELS_TO_IGNORE": ["O"],
                "MODEL_TO_PRESIDIO_MAPPING": {"DOCTOR": "DOCTOR", "PATIENT": "PATIENT"},
            }

        def tokenize_ragged(self, texts):
            return {
                "input_ids": [list(range(len(t))) for t in texts],
                "offset_mapping": [[(i, i + 1) for i in range(len(t))] for t in texts],
            }

        def forward_windows(self, windows):
            preds = []
            for _ids, offs, text in windows:
                w_preds = []
                for idx, (s, e) in enumerate(offs):
                    sub = text[s:e]
                    if text[s:].startswith("Smith") and e <= s + len("Smith"):
                        w_preds.append(
                            {"entity": "B-DOCTOR", "score": 0.99, "start": s, "end": e, "word": sub, "index": idx}
                        )
                preds.append(w_preds)
            return preds

    core = _TunableCore()

    # Direct actor flow
    actor = TransformerInferenceActor.__new__(TransformerInferenceActor)
    actor.model_name = core.model_name
    actor._core = core
    actor._token_budget = 18
    actor._window_overlap = 5
    actor._num_special_tokens = 2
    actor._gpu_batch_size = 10
    actor._handled_oom_count = 0
    actor._aggregate_bio = True
    actor._recognizer_name = "TransformersRecognizer[test]"
    actor._model_to_presidio_mapping = {"DOCTOR": "DOCTOR"}
    actor._ignore_labels = {"O"}

    actor_raw = actor._run_inference_raw_with_oom_recovery([dense_text])[0]
    actor_json, actor_count = actor._format_note(actor_raw, dense_text)
    actor_entities = json.loads(actor_json)

    # Standalone recognizer flow
    recognizer = TransformersRecognizer.__new__(TransformersRecognizer)
    recognizer.model_name = core.model_name
    recognizer.name = "TransformersRecognizer[test]"
    recognizer._core = core
    recognizer.supported_entities = ["DOCTOR", "PATIENT"]
    recognizer.ignore_labels = ["O"]
    recognizer.model_to_presidio_mapping = {"DOCTOR": "DOCTOR"}
    recognizer.default_explanation = "Identified as {}"
    recognizer.text_overlap_length = 5
    recognizer.id_entity_name = "ID"
    recognizer.id_score_reduction = 0.5

    rec_results = recognizer._get_ner_results_for_text(dense_text)

    assert len(rec_results) == actor_count == 1
    assert rec_results[0]["start"] == actor_entities[0]["start"]
    assert rec_results[0]["end"] == actor_entities[0]["end"]
    assert rec_results[0]["entity_group"] == actor_entities[0]["entity_type"]
