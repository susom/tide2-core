"""Tests for LlmRecognizerWorker batch processing and passthrough preservation."""

from unittest.mock import Mock
from unittest.mock import patch

from presidio_analyzer import RecognizerResult

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.llm_recognizer import LlmRecognizerWorker

MODULE = "tide2.recognizers.llm_json_recognizer"

PHI_PROMPT_CONFIG = {
    "prompt_template": "Detect PHI in: {clinical_text}",
    "supported_entities": ["PATIENT", "DATE"],
}


def test_llm_recognizer_worker_preserves_note_text_and_passthrough():
    """LlmRecognizerWorker preserves note_text and copies passthrough columns."""
    with (
        patch(f"{MODULE}.load_llm_prompt", return_value=PHI_PROMPT_CONFIG),
        patch(f"{MODULE}.LlmModel"),
    ):
        worker = LlmRecognizerWorker(
            project_id="test-proj",
            provider_type="google",
            model_name="gemini-2.5-flash",
        )

    # Mock internal recognizer.analyze
    worker.recognizer = Mock()
    worker.recognizer.analyze.return_value = [
        RecognizerResult(entity_type="PATIENT", start=0, end=4, score=0.95),
    ]

    batch = {
        "text_hash": ["hash_1", "hash_2"],
        "note_text": ["John has an appointment.", "No phi here."],
        "patient_id": ["p_001", "p_002"],
        "row_id": ["row_1", "row_2"],
        "jitter": [15, -20],
        "patient_identifiers": ["id1", "id2"],
    }

    out = worker.process_batch(batch)

    assert out["text_hash"] == ["hash_1", "hash_2"]
    assert out["note_text"] == ["John has an appointment.", "No phi here."]
    assert out["patient_id"] == ["p_001", "p_002"]
    assert out["row_id"] == ["row_1", "row_2"]
    assert out["jitter"] == [15, -20]
    assert out["patient_identifiers"] == ["id1", "id2"]
    assert out["entity_count"] == [1, 1]

    # Verify downstream AnonymizerWorker accepts this output directly
    anon_worker = AnonymizerWorker(
        salt=b"\x00" * 32,
        key=b"\x11" * 32,
    )
    anon_out = anon_worker.process_batch(out)
    assert len(anon_out["anonymized_note_text"]) == 2
    assert anon_out["patient_id"] == ["p_001", "p_002"]
    assert anon_out["row_id"] == ["row_1", "row_2"]


def test_llm_recognizer_worker_handles_pd_na_and_nulls():
    """LlmRecognizerWorker treats pd.NA, None, and empty text as successful empty notes."""
    import numpy as np
    import pandas as pd

    with (
        patch(f"{MODULE}.load_llm_prompt", return_value=PHI_PROMPT_CONFIG),
        patch(f"{MODULE}.LlmModel"),
    ):
        worker = LlmRecognizerWorker(
            project_id="test-proj",
            provider_type="google",
            model_name="gemini-2.5-flash",
        )

    worker.recognizer = Mock()

    batch = {
        "text_hash": ["h_na", "h_none", "h_nan", "h_empty"],
        "note_text": [pd.NA, None, np.nan, ""],
    }

    out = worker.process_batch(batch)

    if out["processing_status"] != ["success", "success", "success", "success"]:
        raise ValueError(f"Expected all success, got {out['processing_status']}")
    if out["entity_count"] != [0, 0, 0, 0]:
        raise ValueError(f"Expected all 0 entity_count, got {out['entity_count']}")
    if any(err is not None for err in out["error_message"]):
        raise ValueError(f"Expected no errors, got {out['error_message']}")
    if worker.recognizer.analyze.called:
        raise ValueError("analyze should not have been called for null/empty notes")
