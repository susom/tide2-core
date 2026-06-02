"""
Unit tests for LlmJsonRecognizer.

Tests cover constructor validation, JSON entity extraction, and end-to-end
analyze flow with mocked LLM and prompt loading.
"""

import json
from unittest.mock import patch

import pytest

MODULE = "tide2.recognizers.llm_json_recognizer"

PHI_PROMPT_CONFIG = {
    "prompt_template": "Detect PHI in: {clinical_text}",
    "supported_entities": [
        "PATIENT",
        "LOCATION",
        "PHONE",
        "WEB",
        "DATE",
        "ID",
        "HOSPITAL",
        "DOCTOR",
        "AGE",
        "OTHER",
    ],
}

CUSTOM_PROMPT_CONFIG = {
    "prompt_template": "Extract entities from: {clinical_text}",
    "supported_entities": ["MEDICATION", "PROCEDURE", "DIAGNOSIS"],
}


def _make_recognizer(prompt_config, **kwargs):
    """Helper to create a recognizer with mocked dependencies."""
    defaults = {
        "project_id": "test-project",
        "provider_type": "google",
        "model_name": "test-model",
    }
    defaults.update(kwargs)
    with (
        patch(f"{MODULE}.load_llm_prompt", return_value=prompt_config),
        patch(f"{MODULE}.LlmModel"),
    ):
        from tide2.recognizers.llm_json_recognizer import LlmJsonRecognizer

        return LlmJsonRecognizer(**defaults)


class TestLlmJsonRecognizerInit:
    """Constructor validation tests."""

    def test_init_with_default_phi_entities(self):
        recognizer = _make_recognizer(PHI_PROMPT_CONFIG)
        assert set(recognizer.get_supported_entities()) == set(PHI_PROMPT_CONFIG["supported_entities"])

    def test_init_with_custom_entity_types(self):
        recognizer = _make_recognizer(CUSTOM_PROMPT_CONFIG)
        assert set(recognizer.get_supported_entities()) == {"MEDICATION", "PROCEDURE", "DIAGNOSIS"}

    def test_init_supported_entities_subset(self):
        recognizer = _make_recognizer(PHI_PROMPT_CONFIG, supported_entities=["PATIENT", "DATE"])
        assert set(recognizer.get_supported_entities()) == {"PATIENT", "DATE"}

    def test_init_supported_entities_not_in_prompt_raises(self):
        with pytest.raises(ValueError, match="not supported by prompt"):
            _make_recognizer(PHI_PROMPT_CONFIG, supported_entities=["NONEXISTENT"])

    def test_init_empty_prompt_entities_raises(self):
        config = {**CUSTOM_PROMPT_CONFIG, "supported_entities": []}
        with pytest.raises(ValueError, match="invalid supported_entities"):
            _make_recognizer(config)

    def test_init_non_string_prompt_entities_raises(self):
        config = {**CUSTOM_PROMPT_CONFIG, "supported_entities": [123, None]}
        with pytest.raises(ValueError, match="invalid supported_entities"):
            _make_recognizer(config)

    def test_init_whitespace_only_entity_raises(self):
        config = {**CUSTOM_PROMPT_CONFIG, "supported_entities": ["   "]}
        with pytest.raises(ValueError, match="invalid supported_entities"):
            _make_recognizer(config)


class TestEntityExtraction:
    """JSON parsing and entity extraction tests."""

    def setup_method(self):
        self.phi_recognizer = _make_recognizer(PHI_PROMPT_CONFIG)
        self.custom_recognizer = _make_recognizer(CUSTOM_PROMPT_CONFIG)

    def test_extract_standard_phi_entities(self):
        text = "Patient John Smith was seen on 01/15/2024"
        llm_json = json.dumps(
            {
                "PATIENT": [{"text": "John Smith", "confidence": 0.95}],
                "DATE": [{"text": "01/15/2024", "confidence": 0.99}],
            }
        )
        results = self.phi_recognizer._parse_json_response_internal(llm_json, text, ["PATIENT", "DATE"])
        assert len(results) == 2
        types = {r.entity_type for r in results}
        assert types == {"PATIENT", "DATE"}
        for r in results:
            assert text[r.start : r.end] in ("John Smith", "01/15/2024")

    def test_extract_custom_entity_types(self):
        text = "Prescribed Metformin 500mg before the appendectomy"
        llm_json = json.dumps(
            {
                "MEDICATION": [{"text": "Metformin 500mg", "confidence": 0.90}],
                "PROCEDURE": [{"text": "appendectomy", "confidence": 0.88}],
            }
        )
        results = self.custom_recognizer._parse_json_response_internal(llm_json, text, ["MEDICATION", "PROCEDURE"])
        assert len(results) == 2
        types = {r.entity_type for r in results}
        assert types == {"MEDICATION", "PROCEDURE"}

    def test_extract_filters_unknown_types(self):
        text = "John Smith takes Metformin"
        llm_json = json.dumps(
            {
                "MEDICATION": [{"text": "Metformin", "confidence": 0.90}],
                "UNKNOWN_TYPE": [{"text": "John Smith", "confidence": 0.80}],
            }
        )
        results = self.custom_recognizer._parse_json_response_internal(llm_json, text, ["MEDICATION", "UNKNOWN_TYPE"])
        assert len(results) == 1
        assert results[0].entity_type == "MEDICATION"

    def test_extract_filters_unrequested_types(self):
        text = "Prescribed Metformin before the appendectomy"
        llm_json = json.dumps(
            {
                "MEDICATION": [{"text": "Metformin", "confidence": 0.90}],
                "PROCEDURE": [{"text": "appendectomy", "confidence": 0.88}],
            }
        )
        results = self.custom_recognizer._parse_json_response_internal(llm_json, text, ["MEDICATION"])
        assert len(results) == 1
        assert results[0].entity_type == "MEDICATION"

    def test_extract_multiple_occurrences(self):
        text = "Dr. Smith referred to Dr. Smith again"
        llm_json = json.dumps(
            {
                "DOCTOR": [{"text": "Dr. Smith", "confidence": 0.95}],
            }
        )
        results = self.phi_recognizer._parse_json_response_internal(llm_json, text, ["DOCTOR"])
        assert len(results) == 2
        assert all(r.entity_type == "DOCTOR" for r in results)
        positions = sorted((r.start, r.end) for r in results)
        assert text[positions[0][0] : positions[0][1]] == "Dr. Smith"
        assert text[positions[1][0] : positions[1][1]] == "Dr. Smith"

    def test_extract_empty_json(self):
        results = self.phi_recognizer._parse_json_response_internal("{}", "some text", ["PATIENT"])
        assert results == []


class TestAnalyze:
    """End-to-end analyze flow with mocked LLM."""

    def setup_method(self):
        self.recognizer = _make_recognizer(CUSTOM_PROMPT_CONFIG)
        self.mock_llm = self.recognizer._llm_model

    def test_analyze_with_custom_entities_end_to_end(self):
        text = "Patient received Metformin for diabetes"
        self.mock_llm.get_response.return_value = json.dumps(
            {
                "MEDICATION": [{"text": "Metformin", "confidence": 0.95}],
                "DIAGNOSIS": [{"text": "diabetes", "confidence": 0.90}],
            }
        )
        self.mock_llm.estimate_tokens_from_characters.return_value = 50

        results = self.recognizer.analyze(text, ["MEDICATION", "DIAGNOSIS"])

        assert len(results) == 2
        types = {r.entity_type for r in results}
        assert types == {"MEDICATION", "DIAGNOSIS"}

    def test_analyze_empty_text(self):
        results = self.recognizer.analyze("   ", ["MEDICATION"])
        assert results == []
        self.mock_llm.get_response.assert_not_called()

    def test_analyze_no_matching_entities(self):
        results = self.recognizer.analyze("some text", ["NONEXISTENT_TYPE"])
        assert results == []
        self.mock_llm.get_response.assert_not_called()
