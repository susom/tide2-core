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
