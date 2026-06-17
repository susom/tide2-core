"""
Comprehensive unit tests for KnownValuesRecognizer.

Tests cover name permutation generation, entity merging functionality,
and the main KnownValuesRecognizer with temporary SQLite database setup
for testing known patient identifiers.
"""

import contextlib
import sqlite3
import tempfile
from pathlib import Path

import pytest
from presidio_analyzer.recognizer_result import RecognizerResult

from tide2.recognizers.known_values import KnownValuesRecognizer
from tide2.recognizers.known_values import _are_entities_continuous
from tide2.recognizers.known_values import _create_merged_result
from tide2.recognizers.known_values import generate_person_name_permutations
from tide2.recognizers.known_values import merge_continuous_entities


class TestGeneratePersonNamePermutations:
    """Test the generate_person_name_permutations function."""

    def test_empty_input(self):
        """Test handling of empty inputs."""
        assert generate_person_name_permutations("") == []
        assert generate_person_name_permutations("   ") == []

    def test_single_word_name(self):
        """Test that single-word names return only the original name."""
        result = generate_person_name_permutations("John")
        assert result == ["John"]

        result = generate_person_name_permutations("Smith")
        assert result == ["Smith"]

    def test_two_word_name(self):
        """Test individual components for a simple two-word name."""
        result = generate_person_name_permutations("John Doe")
        # New optimized behavior: only individual words (no permutations)
        expected = {"John", "Doe"}
        assert set(result) == expected
        assert len(result) == 2

    def test_three_word_name(self):
        """Test individual components for a three-word name."""
        result = generate_person_name_permutations("John Michael Smith")

        # New optimized behavior: only individual words
        expected = {"John", "Michael", "Smith"}
        assert set(result) == expected
        assert len(result) == 3

    def test_name_with_existing_comma(self):
        """Test handling of names that already contain commas."""
        result = generate_person_name_permutations("Smith, John A.")

        # Should clean comma and generate permutations
        assert len(result) > 1
        assert "Smith" in result
        assert "John" in result
        assert "A." in result

    def test_name_with_extra_whitespace(self):
        """Test handling of names with extra whitespace."""
        result = generate_person_name_permutations("  John   Michael   Smith  ")

        # Should normalize whitespace and return individual words
        expected = {"John", "Michael", "Smith"}
        assert set(result) == expected


class TestEntityMerging:
    """Test the entity merging utilities."""

    def test_merge_continuous_entities_basic(self):
        """Test basic merging of two adjacent entities."""
        result1 = RecognizerResult(
            entity_type="PATIENT",
            start=0,
            end=4,  # "John"
            score=0.9,
        )
        result2 = RecognizerResult(
            entity_type="PATIENT",
            start=5,
            end=8,  # "Doe"
            score=0.9,
        )

        text = "John Doe called yesterday"
        merged = merge_continuous_entities([result1, result2], text)

        assert len(merged) == 1
        assert merged[0].entity_type == "PATIENT"
        assert merged[0].start == 0
        assert merged[0].end == 8
        assert merged[0].score == 0.9

    def test_merge_continuous_entities_no_merge_different_types(self):
        """Test that entities of different types are not merged."""
        result1 = RecognizerResult(
            entity_type="PATIENT",
            start=0,
            end=4,  # "John"
            score=0.9,
        )
        result2 = RecognizerResult(
            entity_type="LOCATION",
            start=5,
            end=12,  # "Seattle"
            score=0.9,
        )

        text = "John Seattle"
        merged = merge_continuous_entities([result1, result2], text)

        # Should not merge different entity types
        assert len(merged) == 2

    def test_merge_continuous_entities_with_gap(self):
        """Test merging entities separated by small gap."""
        result1 = RecognizerResult(
            entity_type="PATIENT",
            start=0,
            end=4,  # "John"
            score=0.9,
        )
        result2 = RecognizerResult(
            entity_type="PATIENT",
            start=6,
            end=9,  # "Doe"
            score=0.8,
        )

        text = "John, Doe called"
        merged = merge_continuous_entities([result1, result2], text, max_gap_chars=3)

        assert len(merged) == 1
        assert merged[0].start == 0
        assert merged[0].end == 9
        assert merged[0].score == 0.9  # Should use max score

    def test_merge_continuous_entities_gap_too_large(self):
        """Test that entities with large gaps are not merged."""
        result1 = RecognizerResult(
            entity_type="PATIENT",
            start=0,
            end=4,  # "John"
            score=0.9,
        )
        result2 = RecognizerResult(
            entity_type="PATIENT",
            start=20,
            end=23,  # "Doe"
            score=0.8,
        )

        text = "John is a patient and Doe is another"
        merged = merge_continuous_entities([result1, result2], text, max_gap_chars=3)

        # Gap is too large, should not merge
        assert len(merged) == 2

    def test_are_entities_continuous(self):
        """Test the _are_entities_continuous utility function."""
        result1 = RecognizerResult(entity_type="PATIENT", start=0, end=4, score=0.9)
        result2 = RecognizerResult(entity_type="PATIENT", start=5, end=8, score=0.9)

        # Adjacent entities of same type
        assert _are_entities_continuous(result1, result2, "John Doe", max_gap_chars=1)

        # Different types - should return False
        result3 = RecognizerResult(entity_type="LOCATION", start=5, end=8, score=0.9)
        assert not _are_entities_continuous(result1, result3, "John Doe", max_gap_chars=1)

        # Gap too large
        result4 = RecognizerResult(entity_type="PATIENT", start=10, end=13, score=0.9)
        assert not _are_entities_continuous(result1, result4, "John and Sam", max_gap_chars=1)

    def test_create_merged_result(self):
        """Test the _create_merged_result utility function."""
        result1 = RecognizerResult(entity_type="PATIENT", start=0, end=4, score=0.9)
        result2 = RecognizerResult(entity_type="PATIENT", start=5, end=8, score=0.7)

        merged = _create_merged_result([result1, result2])

        assert merged.entity_type == "PATIENT"
        assert merged.start == 0
        assert merged.end == 8
        assert merged.score == 0.9  # Should use max score


class TestKnownValuesRecognizer:
    """Test KnownValuesRecognizer functionality with SQLite database."""

    def setup_method(self):
        """Set up test fixtures with temporary SQLite database."""
        # Create temporary SQLite database
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        # Close the file descriptor immediately since we'll use sqlite3 to access it
        import os

        os.close(fd)

        # Create database schema and insert test data
        self._setup_test_database()

        # Sample known values for testing
        self.test_known_values = {
            "phone_number": ["555-1234", "(617) 555-5678", "617.555.9012"],
            "person": ["John Doe", "Johnny Doe", "J. Doe", "Dr. John"],
            "email_address": ["john@email.com", "johnny@work.com"],
            "location": ["123 Main St", "Boston", "Cambridge, MA"],
            "mrn": ["MRN123456", "MRN123400"],
            "har": ["HAR789012", "HAR789001"],
            "acc_num": ["ACC123ABC", "ACC456DEF"],
            "csn_id": ["CSN001122", "CSN334455"],
            "us_ssn": ["123-45-6789", "987-65-4321"],
            "url": ["https://example.com", "www.test.org"],
            "medical_license": ["LIC123456", "LIC789012"],
            "ip_address": ["192.168.1.1", "10.0.0.1"],
        }

    def teardown_method(self):
        """Clean up temporary database."""
        with contextlib.suppress(OSError):
            Path(self.db_path).unlink()

    def _setup_test_database(self):
        """Create test database with patient identifiers."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create patient identifiers table
        cursor.execute("""
            CREATE TABLE patient_identifiers (
                patient_id TEXT,
                identifier_type TEXT,
                identifier_value TEXT,
                PRIMARY KEY (patient_id, identifier_type, identifier_value)
            )
        """)

        # Insert test data
        test_data = [
            ("patient_123", "phone_number", "555-1234"),
            ("patient_123", "phone_number", "(617) 555-5678"),
            ("patient_123", "person", "John Doe"),
            ("patient_123", "person", "Johnny Doe"),
            ("patient_123", "person", "J. Doe"),
            ("patient_123", "email_address", "john@email.com"),
            ("patient_123", "location", "123 Main St"),
            ("patient_123", "location", "Boston"),
            ("patient_123", "mrn", "MRN123456"),
            ("patient_123", "har_number", "HAR789012"),
            ("patient_456", "phone_number", "(555) 987-6543"),
            ("patient_456", "person", "Jane Smith"),
            ("patient_456", "email_address", "jane@test.com"),
        ]

        cursor.executemany("INSERT INTO patient_identifiers VALUES (?, ?, ?)", test_data)

        conn.commit()
        conn.close()

    def _get_patient_known_values(self, patient_id: str) -> dict:
        """Retrieve known values from database for a patient."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT identifier_type, identifier_value FROM patient_identifiers WHERE patient_id = ?", (patient_id,)
        )

        known_values = {}
        for identifier_type, identifier_value in cursor.fetchall():
            if identifier_type not in known_values:
                known_values[identifier_type] = []
            known_values[identifier_type].append(identifier_value)

        conn.close()
        return known_values

    def test_initialization(self):
        """Test KnownValuesRecognizer initialization."""
        recognizer = KnownValuesRecognizer(known_values=self.test_known_values, entity_type="person")

        assert "PATIENT" in recognizer.supported_entities
        assert recognizer.supported_language == "en"
        assert recognizer.name == "KnownValuesRecognizer"

    def test_phone_number_recognition(self):
        """Test recognition of known phone numbers."""
        known_values = {"phone_number": ["555-1234", "(617) 555-5678"]}
        recognizer = KnownValuesRecognizer(known_values=known_values, entity_type="phone_number")

        text = "Call me at 555-1234 or (617) 555-5678"
        results = recognizer.analyze(text, ["PHONE"])

        assert len(results) == 2
        assert results[0].entity_type == "PHONE"
        assert results[1].entity_type == "PHONE"
        assert results[0].score == 0.95
        assert results[1].score == 0.95

    def test_person_name_recognition(self):
        """Test recognition of known person names."""
        known_values = {"person": ["John Doe", "Jane Smith"]}
        recognizer = KnownValuesRecognizer(known_values=known_values, entity_type="person")

        text = "Patient John Doe and Jane Smith arrived"
        results = recognizer.analyze(text, ["PATIENT"])

        assert len(results) == 2
        for result in results:
            assert result.entity_type == "PATIENT"
            assert result.score == 0.95

    def test_person_name_permutations(self):
        """Test that person name permutations are recognized."""
        known_values = {"person": ["John Doe"]}
        recognizer = KnownValuesRecognizer(known_values=known_values, entity_type="person")

        # Test reversed name format
        text = "Patient Doe, John was seen today"
        results = recognizer.analyze(text, ["PATIENT"])

        assert len(results) >= 1
        assert any(result.entity_type == "PATIENT" for result in results)

    def test_case_insensitive_matching(self):
        """Test case-insensitive matching of known values."""
        known_values = {"person": ["John Doe"]}
        recognizer = KnownValuesRecognizer(known_values=known_values, entity_type="person")

        text = "Patient JOHN DOE and john doe arrived"
        results = recognizer.analyze(text, ["PATIENT"])

        assert len(results) == 2
        for result in results:
            assert result.entity_type == "PATIENT"

    def test_spurious_value_filtering(self):
        """Test that spurious values are filtered out."""
        # Include some spurious values that should be ignored
        known_values = {"person": ["John Doe", "A", "the", "", "  ", "1", "Dr. Smith"]}
        recognizer = KnownValuesRecognizer(known_values=known_values, entity_type="person")

        # Should only match meaningful names, not spurious values
        text = "Patient John Doe and Dr. Smith saw the patient"
        results = recognizer.analyze(text, ["PATIENT"])

        # Should find "John Doe" and "Dr. Smith" but not "the", "A", etc.
        assert len(results) >= 2
        for result in results:
            matched_text = text[result.start : result.end]
            assert matched_text not in ["A", "the", "1", ""]

    def test_multiple_entity_types(self):
        """Test recognition of multiple entity types."""
        known_values = {"phone_number": ["555-1234"], "person": ["John Doe"], "email_address": ["john@email.com"]}
        recognizer = KnownValuesRecognizer(known_values=known_values, entity_type="person")

        text = "John Doe's phone is 555-1234 and email is john@email.com"
        results = recognizer.analyze(text, ["PATIENT"])

        # Should only find PATIENT entities (John Doe) since that's what we're analyzing for
        person_results = [r for r in results if r.entity_type == "PATIENT"]
        assert len(person_results) >= 1

    def test_entity_merging_person(self):
        """Test entity merging for PATIENT entities."""
        known_values = {"person": ["John", "Doe"]}
        recognizer = KnownValuesRecognizer(
            known_values=known_values, entity_type="person", merge_person_max_gap_chars=1
        )

        text = "Patient John Doe arrived"
        results = recognizer.analyze(text, ["PATIENT"])

        # Should merge "John" and "Doe" into single entity
        assert len(results) >= 1
        # Look for a merged result that spans both words
        merged_found = any(r for r in results if text[r.start : r.end] == "John Doe")
        assert merged_found or len(results) == 1  # Either merged or found as one entity

    def test_entity_merging_location(self):
        """Test entity merging for LOCATION entities."""
        known_values = {"location": ["Main", "Street"]}
        recognizer = KnownValuesRecognizer(
            known_values=known_values, entity_type="location", merge_location_max_gap_chars=3
        )

        text = "Address is Main Street"
        results = recognizer.analyze(text, ["LOCATION"])

        assert len(results) >= 1

    def test_database_integration(self):
        """Test integration with SQLite database."""
        # Get known values from database
        known_values = self._get_patient_known_values("patient_123")

        recognizer = KnownValuesRecognizer(known_values=known_values, entity_type="person")

        text = "Patient John Doe called from 555-1234"
        results = recognizer.analyze(text, ["PATIENT"])

        assert len(results) >= 1
        person_result = next(r for r in results if r.entity_type == "PATIENT")
        assert text[person_result.start : person_result.end] in ["John Doe", "John", "Doe"]

    def test_no_known_values(self):
        """Test behavior when no known values are provided."""
        with pytest.raises(ValueError, match="No known values"):
            KnownValuesRecognizer(known_values={}, entity_type="person")

    def test_empty_text(self):
        """Test analysis of empty text."""
        recognizer = KnownValuesRecognizer(known_values={"person": ["John Doe"]}, entity_type="person")

        results = recognizer.analyze("", ["PATIENT"])
        assert len(results) == 0

        results = recognizer.analyze("   ", ["PATIENT"])
        assert len(results) == 0

    def test_unsupported_entity_type(self):
        """Test analysis with unsupported entity type - should return no results."""
        recognizer = KnownValuesRecognizer(known_values={"person": ["John Doe"]}, entity_type="person")

        text = "Patient John Doe arrived"
        results = recognizer.analyze(text, ["UNSUPPORTED_TYPE"])

        # Should return no results since PATIENT is not in the requested entity types
        assert len(results) == 0

        # But should find results when requesting the correct entity type
        results = recognizer.analyze(text, ["PATIENT"])
        assert len(results) >= 1

    def test_medical_record_numbers(self):
        """Test recognition of medical record numbers."""
        known_values = {"mrn": ["MRN123456", "987654321"]}
        recognizer = KnownValuesRecognizer(known_values=known_values, entity_type="mrn")

        text = "Patient MRN123456 and backup 987654321"
        results = recognizer.analyze(text, ["MRN"])

        assert len(results) == 2
        for result in results:
            assert result.entity_type == "MRN"
            assert result.score == 0.95

    def test_har_numbers(self):
        """Test recognition of HAR numbers."""
        known_values = {"har": ["HAR789012", "HAR111222"]}
        recognizer = KnownValuesRecognizer(known_values=known_values, entity_type="har")

        text = "Check HAR789012 and HAR111222 records"
        results = recognizer.analyze(text, ["HAR"])

        assert len(results) == 2
        for result in results:
            assert result.entity_type == "HAR"

    def test_complex_clinical_scenario(self):
        """Test complex clinical scenario with multiple entity types from database."""
        # Simulate a full patient record lookup
        patient_known_values = self._get_patient_known_values("patient_123")

        # Create recognizers for different entity types
        if "person" in patient_known_values:
            person_recognizer = KnownValuesRecognizer(known_values=patient_known_values, entity_type="person")

        if "phone_number" in patient_known_values:
            phone_recognizer = KnownValuesRecognizer(known_values=patient_known_values, entity_type="phone_number")

        text = "Patient John Doe can be reached at 555-1234 or johnny@work.com"

        # Test person recognition if person data exists. The fixture DB stores
        # "John Doe" for patient_123, which appears verbatim in the text.
        if "person" in patient_known_values:
            person_results = person_recognizer.analyze(text, ["PATIENT"])
            assert len(person_results) >= 1
            assert all(r.entity_type == "PATIENT" for r in person_results)

        # Test phone recognition if phone data exists. The fixture DB stores
        # "555-1234" for patient_123, which appears verbatim in the text.
        if "phone_number" in patient_known_values:
            phone_results = phone_recognizer.analyze(text, ["PHONE"])
            assert len(phone_results) >= 1
            assert all(r.entity_type == "PHONE" for r in phone_results)

    def test_all_spurious_values_empty_automaton(self):
        """Test that recognizer handles all values being filtered as spurious.

        When all provided values are filtered out (single chars, stopwords, etc.),
        the automaton is None and analyze() should return empty list without error.
        """
        # All values are spurious - single chars and stopwords
        known_values = {"person": ["a", "I", "the", "an", "is", ""]}
        recognizer = KnownValuesRecognizer(known_values=known_values, entity_type="person")

        # Automaton should be None since all values were filtered
        assert recognizer.automaton is None

        # analyze() should return empty list without raising Aho-Corasick error
        text = "This is a test with some text"
        results = recognizer.analyze(text, ["PATIENT"])
        assert results == []
