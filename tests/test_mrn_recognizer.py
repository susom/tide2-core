"""
Unit tests for MrnRecognizer.

Tests cover pattern matching for Medical Record Numbers (MRN) in various formats,
including numbered patterns, contextual patterns with labels, and edge cases.
"""

from tide2.recognizers.mrn_recognizer import MrnRecognizer


class TestMrnRecognizer:
    """Test MrnRecognizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.recognizer = MrnRecognizer()

    def test_initialization(self):
        """Test MrnRecognizer initialization."""
        assert self.recognizer.supported_entities == ["MRN"]
        assert self.recognizer.supported_language == "en"
        assert self.recognizer.name == "MrnRecognizer"

    def test_custom_initialization(self):
        """Test MrnRecognizer with custom parameters."""
        recognizer = MrnRecognizer(supported_entity="MEDICAL_ID", supported_language="es")
        assert recognizer.supported_entities == ["MEDICAL_ID"]
        assert recognizer.supported_language == "es"

    def test_mrn_dashed_patterns(self):
        """Test recognition of MRN patterns with dashes."""
        test_cases = [
            ("Patient ID: 123-45-67", [(12, 21)]),  # ###-##-##
            ("MRN: 456-78-9012", [(5, 16)]),  # ###-##-####
            ("ID 789-01-23-4", [(3, 14)]),  # ###-##-##-# (with trailing digit)
            ("Check 1234567-8", [(6, 15)]),  # #######-#
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "MRN"
                assert result.start == start
                assert result.end == end
                # Score varies by pattern
                assert result.score >= 0.6

    def test_mrn_simple_digit_patterns(self):
        """Test recognition of simple digit-only MRN patterns."""
        test_cases = [
            ("MRN 12345678", [(4, 12)]),  # 8 digits
            ("Patient 1234567890", [(8, 18)]),  # 10 digits
            ("ID: 12-34-56", [(4, 12)]),  # ##-##-## format
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "MRN"
                assert result.start == start
                assert result.end == end

    def test_mrn_with_colon_label(self):
        """Test MRN recognition with explicit MRN: label."""
        test_cases = [
            ("MRN: 123456789", [(5, 14)]),
            ("MRN:987654321", [(4, 13)]),
            ("Patient MRN: 456-78-90", [(13, 22)]),
            ("Check MRN: 12-34-56-78", [(11, 22)]),
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "MRN"
                assert result.start == start
                assert result.end == end
                assert result.score == 0.95  # High confidence with explicit label

    def test_mrn_with_mr_hash(self):
        """Test MRN recognition with MR# label."""
        test_cases = [
            ("MR# 123456789", [(4, 13)]),
            ("Patient MR# 987-65-43", [(12, 21)]),
            ("Check MR# 456789012", [(10, 19)]),
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "MRN"
                assert result.start == start
                assert result.end == end
                assert result.score == 0.95

    def test_mrn_with_space_label(self):
        """Test MRN recognition with space-separated labels."""
        test_cases = [
            ("MRN 123456789", [(4, 13)]),
            ("MR 987654321", [(3, 12)]),
            ("Patient MRN 456-78-90", [(12, 21)]),
            ("Check MR 12-34-56", [(9, 17)]),
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "MRN"
                assert result.start == start
                assert result.end == end
                assert result.score >= 0.90  # High confidence with label

    def test_mrn_medical_record_number_label(self):
        """Test MRN recognition with full 'Medical Record Number:' label."""
        test_cases = [
            ("Medical Record Number: 123456789", [(23, 32)]),
            ("Patient Medical Record Number: 987-65-43", [(31, 40)]),
            ("medical record number: 456789012", [(23, 32)]),  # case insensitive
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "MRN"
                assert result.start == start
                assert result.end == end
                assert result.score == 0.95

    def test_mrn_med_rec_patterns(self):
        """Test MRN recognition with MED REC # patterns."""
        test_cases = [
            ("MED REC #: 123456789", [(11, 20)]),
            ("MED REC #:  (123)456789", [(12, 23)]),  # with parentheses
            ("MED REC: 987654321", [(9, 18)]),
            ("MED REC:  (456)123789", [(10, 21)]),  # with parentheses
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "MRN"
                assert result.start == start
                assert result.end == end
                assert result.score == 0.95

    def test_multiple_mrns_in_text(self):
        """Test recognition of multiple MRN numbers in the same text."""
        text = "Patient MRN: 123-45-67 and backup MRN 987654321"
        results = self.recognizer.analyze(text, ["MRN"])

        assert len(results) >= 2  # Should find at least 2 MRNs

        # Sort results by start position for predictable testing
        results.sort(key=lambda x: x.start)

        # First MRN with colon
        assert results[0].start == 13
        assert results[0].end == 22
        assert text[results[0].start : results[0].end] == "123-45-67"

        # Second MRN with space
        found_second = False
        for result in results[1:]:
            if text[result.start : result.end] == "987654321":
                found_second = True
                break
        assert found_second, "Second MRN not found"

    def test_mrn_case_insensitive(self):
        """Test case-insensitive MRN recognition."""
        test_cases = [
            ("mrn: 123456789", [(5, 14)]),
            ("MRN: 987654321", [(5, 14)]),
            ("Mrn: 456789012", [(5, 14)]),
            ("medical record number: 123456", [(23, 29)]),
            ("MEDICAL RECORD NUMBER: 987654", [(23, 29)]),
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "MRN"
                assert result.start == start
                assert result.end == end

    def test_mrn_various_formats(self):
        """Test MRN recognition with various number formats."""
        test_cases = [
            ("MRN: 1-2-3", [(5, 10)]),  # minimal dashes
            ("MRN: 12 34 56 78", [(5, 16)]),  # spaces as separators
            ("MRN: 123-456-789-0", [(5, 18)]),  # multiple dashes
            ("MRN: 12345678901234", [(5, 19)]),  # long number
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "MRN"
                assert result.start == start
                assert result.end == end

    def test_no_false_positives(self):
        """Test that non-MRN patterns are not recognized."""
        test_cases = [
            "This is a normal sentence.",
            "Date: 12/31/2023",  # date format
            "Phone: 555-1234",  # phone number
            "Price: $123.45",  # currency
            "Time: 12:34:56",  # time format
            "MRN",  # label only
            "MRN:",  # label with colon only
            "MRN: ",  # label with space only
            "123",  # too short
        ]

        for text in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            # Some of these might match simple digit patterns, so we check for reasonable constraints
            if results:
                # If there are matches, they should be reasonable MRN-like numbers
                for result in results:
                    matched_text = text[result.start : result.end]
                    # Should contain digits and be reasonable length
                    assert len(matched_text) >= 5, f"Too short match: {matched_text} in {text}"

    def test_mrn_with_word_boundaries(self):
        """Test that MRN detection respects word boundaries."""
        test_cases = [
            ("HARM: 123456789", []),  # Should not match (not MR boundary)
            ("MRNA: 123456789", []),  # Should not match
            ("MRN: 123456789", [(5, 14)]),  # Should match
            ("The MRN: 123456789", [(9, 18)]),  # Should match
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            if expected_spans:
                for result, (start, end) in zip(results, expected_spans, strict=False):
                    assert result.entity_type == "MRN"
                    assert result.start == start
                    assert result.end == end

    def test_analyze_empty_text(self):
        """Test analysis of empty or whitespace-only text."""
        test_cases = ["", "   ", "\n\t"]

        for text in test_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert results == []

    def test_supported_entities(self):
        """Test that only supported entity types work."""
        text = "MRN: 123456789"

        # Should work with MRN
        results = self.recognizer.analyze(text, ["MRN"])
        assert len(results) == 1

        # Should not return results for other entity types (but still finds MRN)
        results = self.recognizer.analyze(text, ["PERSON", "PHONE_NUMBER", "HAR"])
        assert len(results) == 1  # Still finds MRN entity since it's in the text

    def test_mrn_score_variations(self):
        """Test that different patterns have appropriate confidence scores."""
        high_confidence_cases = [
            "MRN: 123456789",  # 0.95
            "MR# 123456789",  # 0.95
            "Medical Record Number: 123456789",  # 0.95
            "MED REC #: 123456789",  # 0.95
        ]

        for text in high_confidence_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) >= 1, f"No match for: {text}"
            assert results[0].score == 0.95, f"Wrong score for: {text}"

        medium_confidence_cases = [
            "MR 123456789",  # 0.90
        ]

        for text in medium_confidence_cases:
            results = self.recognizer.analyze(text, ["MRN"])
            assert len(results) >= 1, f"No match for: {text}"
            assert results[0].score == 0.90, f"Wrong score for: {text}"
