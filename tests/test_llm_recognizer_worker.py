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
        "patient_uid": ["p_001", "p_002"],
        "row_id": ["row_1", "row_2"],
        "jitter": [15, -20],
        "patient_identifiers": ["id1", "id2"],
    }

    out = worker.process_batch(batch)

    assert out["text_hash"] == ["hash_1", "hash_2"]
    assert out["note_text"] == ["John has an appointment.", "No phi here."]
    assert out["patient_uid"] == ["p_001", "p_002"]
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
    assert anon_out["patient_uid"] == ["p_001", "p_002"]
    assert anon_out["row_id"] == ["row_1", "row_2"]
