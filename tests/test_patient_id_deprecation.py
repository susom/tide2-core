"""Unit tests verifying deprecation warnings and errors for patient_uid."""

import pandas as pd
import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.llm_recognizer import LlmRecognizerWorker
from tide2.actors.recognizer import RecognizerWorker
from tide2.actors.transformer import BIOAggregationActor
from tide2.actors.transformer import TransformerInferenceActor
from tide2.anonymizers.accession_number_hash import AccessionNumberHashAnonymizer
from tide2.runner.utils import detect_columns
from tide2.utils.batch_columns import copy_passthrough


def test_inspect_pipeline_input_raises_for_patient_uid(tmp_path):
    """_inspect_pipeline_input raises ValueError on patient_uid column."""
    from tide2.runner.local_runner import _inspect_pipeline_input

    file_path = tmp_path / "sample.parquet"
    df = pd.DataFrame({"note_text": ["Clinical note."], "patient_uid": ["P1"]})
    df.to_parquet(file_path)

    with pytest.deprecated_call(), pytest.raises(ValueError, match=r"`patient_uid`.*is deprecated"):
        _inspect_pipeline_input([str(file_path)])


def test_detect_columns_raises_for_patient_uid(tmp_path):
    """detect_columns raises ValueError when parquet file has patient_uid."""
    file_path = tmp_path / "sample.parquet"
    df = pd.DataFrame({"text_hash": ["h1"], "note_text": ["text"], "patient_uid": ["p1"]})
    df.to_parquet(file_path)

    with pytest.deprecated_call(), pytest.raises(ValueError, match=r"`patient_uid`.*is deprecated"):
        detect_columns(str(file_path), required=["text_hash", "note_text"], optional=["patient_id"])


def test_copy_passthrough_raises_for_patient_uid():
    """copy_passthrough raises ValueError when patient_uid is in batch."""
    batch = {"patient_uid": ["p1"], "patient_id": ["p1"]}
    res = {}
    with pytest.deprecated_call(), pytest.raises(ValueError, match=r"`patient_uid`.*is deprecated"):
        copy_passthrough(batch, res)


def test_copy_passthrough_case_insensitive():
    """copy_passthrough handles case-insensitive headers like PATIENT_ID, ROW_ID, JITTER."""
    batch = {
        "PATIENT_ID": ["p1", "p2"],
        "ROW_ID": ["r1", "r2"],
        "JITTER": [10, -5],
    }
    res = {}
    copy_passthrough(batch, res)
    assert res["patient_id"] == ["p1", "p2"]
    assert res["row_id"] == ["r1", "r2"]
    assert res["jitter"] == [10, -5]


def test_recognizer_worker_raises_for_patient_uid():
    """RecognizerWorker.process_batch raises ValueError when patient_uid is in batch."""
    worker_cls = getattr(RecognizerWorker, "__ray_actor_class__", RecognizerWorker)
    worker = worker_cls.__new__(worker_cls)
    batch = {
        "text_hash": ["h1"],
        "note_text": ["text"],
        "patient_uid": ["p1"],
    }
    with pytest.deprecated_call(), pytest.raises(ValueError, match=r"`patient_uid`.*is deprecated"):
        worker.process_batch(batch)


def test_llm_recognizer_worker_raises_for_patient_uid():
    """LlmRecognizerWorker.process_batch raises ValueError when patient_uid is in batch."""
    worker = LlmRecognizerWorker.__new__(LlmRecognizerWorker)
    batch = {
        "text_hash": ["h1"],
        "note_text": ["text"],
        "patient_uid": ["p1"],
    }
    with pytest.deprecated_call(), pytest.raises(ValueError, match=r"`patient_uid`.*is deprecated"):
        worker.process_batch(batch)


def test_transformer_inference_worker_raises_for_patient_uid():
    """TransformerInferenceActor raises ValueError when patient_uid is in batch."""
    actor = TransformerInferenceActor.__new__(TransformerInferenceActor)
    batch = {
        "text_hash": ["h1"],
        "note_text": ["text"],
        "patient_uid": ["p1"],
    }
    with pytest.deprecated_call(), pytest.raises(ValueError, match=r"`patient_uid`.*is deprecated"):
        actor(batch)


def test_bio_aggregation_actor_raises_for_patient_uid():
    """BIOAggregationActor raises ValueError when patient_uid is in batch."""
    actor = BIOAggregationActor.__new__(BIOAggregationActor)
    batch = {
        "text_hash": ["h1"],
        "note_text": ["text"],
        "predictions_raw_json": ["[]"],
        "patient_uid": ["p1"],
    }
    with pytest.deprecated_call(), pytest.raises(ValueError, match=r"`patient_uid`.*is deprecated"):
        actor(batch)


def test_anonymizer_worker_raises_for_patient_uid():
    """AnonymizerWorker.process_batch raises ValueError when patient_uid is in batch."""
    worker_cls = getattr(AnonymizerWorker, "__ray_actor_class__", AnonymizerWorker)
    worker = worker_cls.__new__(worker_cls)
    batch = {
        "text_hash": ["h1"],
        "note_text": ["text"],
        "patient_uid": ["p1"],
    }
    with pytest.deprecated_call(), pytest.raises(ValueError, match=r"`patient_uid`.*is deprecated"):
        worker.process_batch(batch)


def test_accession_number_hash_operate_and_validate():
    """AccessionNumberHashAnonymizer raises ValueError on patient_uid parameter."""
    anon = AccessionNumberHashAnonymizer()
    with pytest.deprecated_call(), pytest.raises(ValueError, match=r"`patient_uid`.*is deprecated"):
        anon.operate("ACC123", {"patient_uid": "p1"})

    with pytest.deprecated_call(), pytest.raises(ValueError, match=r"`patient_uid`.*is deprecated"):
        anon.validate({"patient_uid": "p1"})


def test_null_note_text_passes_through_recognizer_to_anonymizer():
    """Null note_text in RecognizerWorker is normalized and processed by AnonymizerWorker without crashing."""
    rec_worker_cls = getattr(RecognizerWorker, "__ray_actor_class__", RecognizerWorker)
    rec_worker = rec_worker_cls.__new__(rec_worker_cls)
    rec_worker.analyzer = None

    def dummy_process_note(note_text, text_hash, cached_results=None, patient_identifiers=None):
        return {
            "text_hash": text_hash,
            "recognizer_results_json": "[]",
            "entity_count": 0,
        }

    rec_worker.process_note = dummy_process_note

    batch = {
        "text_hash": ["h1", "h2"],
        "note_text": [None, "hello world"],
        "patient_id": ["p1", "p2"],
    }
    rec_output = rec_worker.process_batch(batch)
    assert rec_output["note_text"] == ["", "hello world"]

    anon_worker_cls = getattr(AnonymizerWorker, "__ray_actor_class__", AnonymizerWorker)
    anon_worker = anon_worker_cls.__new__(anon_worker_cls)

    def dummy_anon_process_note(
        note_text, original_text_hash, recognizer_results_json=None, patient_id=None, jitter=None
    ):
        return {
            "text_hash": original_text_hash,
            "patient_id": patient_id,
            "anonymized_note_text": note_text,
            "anonymizer_results_json": "[]",
            "entity_count": 0,
            "stage_status": "success",
            "stage_reason": None,
        }

    anon_worker.process_note = dummy_anon_process_note
    anon_output = anon_worker.process_batch(rec_output)
    assert anon_output["anonymized_note_text"] == ["", "hello world"]


def test_anonymizer_worker_preserves_uppercase_row_id():
    """AnonymizerWorker.process_batch preserves row_id when provided as uppercase ROW_ID."""
    anon_worker_cls = getattr(AnonymizerWorker, "__ray_actor_class__", AnonymizerWorker)
    anon_worker = anon_worker_cls.__new__(anon_worker_cls)

    def dummy_anon_process_note(
        note_text, original_text_hash, recognizer_results_json=None, patient_id=None, jitter=None
    ):
        return {
            "text_hash": original_text_hash,
            "patient_id": patient_id,
            "anonymized_note_text": note_text,
            "anonymizer_results_json": "[]",
            "entity_count": 0,
            "stage_status": "success",
            "stage_reason": None,
        }

    anon_worker.process_note = dummy_anon_process_note
    batch = {
        "text_hash": ["h1"],
        "note_text": ["hello"],
        "ROW_ID": ["custom_row_123"],
    }
    output = anon_worker.process_batch(batch)
    assert "row_id" in output
    assert output["row_id"] == ["custom_row_123"]


def test_transformer_worker_preserves_missing_patient_id_as_none():
    """TransformerInferenceActor and BIOAggregationActor preserve missing patient_id as None."""
    from tide2.actors.transformer import BIOAggregationActor
    from tide2.actors.transformer import TransformerInferenceActor

    worker_cls = getattr(TransformerInferenceActor, "__ray_actor_class__", TransformerInferenceActor)
    worker = worker_cls.__new__(worker_cls)
    worker._aggregate_bio = True
    worker._recognizer_name = "test_recognizer"
    worker._model_to_presidio_mapping = {}
    worker._ignore_labels = set()
    worker._log_gpu_mem = lambda _: None
    worker._run_inference_raw_with_oom_recovery = lambda _: [[]]
    worker._format_note = lambda _p, _t: ("[]", 0)

    batch = {
        "text_hash": ["h1"],
        "note_text": ["sample text"],
        "ROW_ID": ["row_abc"],
    }
    out = worker(batch)
    assert out["patient_id"] == [None]
    assert out["row_id"] == ["row_abc"]

    agg_cls = getattr(BIOAggregationActor, "__ray_actor_class__", BIOAggregationActor)
    agg = agg_cls.__new__(agg_cls)
    agg._recognizer_name = "test_recognizer"
    agg._model_to_presidio_mapping = {}
    agg._ignore_labels = set()
    agg_batch = {
        "text_hash": ["h1"],
        "note_text": ["sample text"],
        "predictions_raw_json": ["[]"],
        "PATIENT_ID": ["pid_123"],
        "JITTER": [5],
    }
    agg_out = agg(agg_batch)
    assert agg_out["patient_id"] == ["pid_123"]
    assert agg_out["jitter"] == [5]
