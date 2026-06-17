"""
Test for the PassthroughRecognizer to ensure it behaves correctly for ablation studies.
"""

import pytest

from tide2.recognizers.passthrough_recognizer import PassthroughRecognizer


def test_passthrough_recognizer_initialization():
    """Test that PassthroughRecognizer can be initialized properly."""
    recognizer = PassthroughRecognizer()

    # Check default values
    assert recognizer.supported_entities == ["PASSTHROUGH"]
    assert recognizer.supported_language == "en"
    assert recognizer.name == "PassthroughRecognizer"


def test_passthrough_recognizer_custom_initialization():
    """Test PassthroughRecognizer with custom parameters."""
    recognizer = PassthroughRecognizer(supported_entity="CUSTOM", supported_language="es")

    assert recognizer.supported_entities == ["CUSTOM"]
    assert recognizer.supported_language == "es"


def test_passthrough_recognizer_analyze_returns_empty():
    """Test that analyze method always returns empty list."""
    recognizer = PassthroughRecognizer()

    # Test with various inputs - should always return empty list
    test_cases = [
        "This is a test with John Smith born on 01/01/1990",
        "Patient ID: 12345, SSN: 123-45-6789",
        "Contact: john.doe@email.com, Phone: (555) 123-4567",
        "",  # empty string
        "A" * 1000,  # long string
    ]

    for text in test_cases:
        results = recognizer.analyze(text, ["PERSON", "DATE", "PHONE_NUMBER"])
        assert results == [], f"Expected empty list for text: {text[:50]}..."


def test_passthrough_recognizer_load():
    """Test that load method works without errors."""
    recognizer = PassthroughRecognizer()

    # Should not raise any exceptions
    recognizer.load()


def test_passthrough_recognizer_get_supported_entities():
    """Test get_supported_entities method."""
    recognizer = PassthroughRecognizer()

    entities = recognizer.get_supported_entities()
    assert entities == ["PASSTHROUGH"]


def test_passthrough_recognizer_integration():
    """Test that PassthroughRecognizer can be used in a typical workflow."""
    recognizer = PassthroughRecognizer()

    # Simulate typical usage pattern
    recognizer.load()

    # Analyze some text
    text = "Patient John Doe, MRN: 123456, DOB: 1990-01-01"
    entities_to_find = ["PERSON", "MEDICAL_RECORD_NUMBER", "DATE_TIME"]

    results = recognizer.analyze(text, entities_to_find)

    # Should return empty results
    assert len(results) == 0
    assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__])
