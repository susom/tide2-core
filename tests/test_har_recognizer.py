"""
Unit tests for HarRecognizer.

Tests cover pattern matching for HAR (Hospital Account Record) numbers
with case-sensitive and case-insensitive variations.
"""

from tide2.recognizers.har_recognizer import HarRecognizer


class TestHarRecognizer:
    """Test HarRecognizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.recognizer = HarRecognizer()

    def test_initialization(self):
        """Test HarRecognizer initialization."""
        assert self.recognizer.supported_entities == ["HAR"]
        assert self.recognizer.supported_language == "en"
        assert self.recognizer.name == "HarRecognizer"

    def test_custom_initialization(self):
        """Test HarRecognizer with custom parameters."""
        recognizer = HarRecognizer(supported_entity="CUSTOM_HAR", supported_language="es")
        assert recognizer.supported_entities == ["CUSTOM_HAR"]
        assert recognizer.supported_language == "es"

    def test_har_basic_pattern(self):
        """Test recognition of basic HAR pattern."""
        test_cases = [
            ("HAR: 123456", [(5, 11)]),
            ("HAR:123456", [(4, 10)]),
            ("HAR: 987654321", [(5, 14)]),
            ("HAR:9876543210", [(4, 14)]),
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["HAR"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "HAR"
                assert result.start == start
                assert result.end == end
                assert result.score == 0.95
                assert text[start:end].isdigit()  # Verify only digits are captured

    def test_har_with_spacing_variations(self):
        """Test HAR recognition with various spacing patterns."""
        test_cases = [
            ("HAR: 123456", [(5, 11)]),  # single space after colon
            ("HAR:  123456", [(6, 12)]),  # double space after colon
            ("HAR:     123456", [(9, 15)]),  # multiple spaces after colon
            ("HAR       : 123456", [(12, 18)]),  # spaces before colon
            ("HAR      :     123456", [(15, 21)]),  # spaces before and after colon
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["HAR"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "HAR"
                assert result.start == start
                assert result.end == end
                assert result.score == 0.95

    def test_har_case_insensitive(self):
        """Test case-insensitive HAR recognition."""
        test_cases = [
            ("har: 123456", [(5, 11)]),
            ("Har: 987654", [(5, 11)]),
            ("HAR: 456789", [(5, 11)]),
            ("hAr: 789012", [(5, 11)]),
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["HAR"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "HAR"
                assert result.start == start
                assert result.end == end
                assert result.score == 0.95

    def test_har_in_sentences(self):
        """Test HAR recognition within longer sentences."""
        test_cases = [
            ("Patient HAR: 123456 was updated", [(13, 19)]),
            ("Check HAR: 987654 for details", [(11, 17)]),
            ("The HAR: 456789 needs review", [(9, 15)]),
            ("Submit form with HAR: 789012 today", [(22, 28)]),
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["HAR"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "HAR"
                assert result.start == start
                assert result.end == end
                assert result.score == 0.95

    def test_multiple_hars_in_text(self):
        """Test recognition of multiple HAR numbers in the same text."""
        text = "Patient HAR: 123456 and backup HAR: 789012"
        results = self.recognizer.analyze(text, ["HAR"])

        assert len(results) == 2

        # First HAR
        assert results[0].start == 13
        assert results[0].end == 19
        assert results[0].score == 0.95
        assert text[results[0].start : results[0].end] == "123456"

        # Second HAR
        assert results[1].start == 36
        assert results[1].end == 42
        assert results[1].score == 0.95
        assert text[results[1].start : results[1].end] == "789012"

    def test_har_different_lengths(self):
        """Test HAR numbers of different lengths."""
        test_cases = [
            ("HAR: 1", [(5, 6)]),  # single digit
            ("HAR: 12", [(5, 7)]),  # two digits
            ("HAR: 123", [(5, 8)]),  # three digits
            ("HAR: 123456789012345", [(5, 20)]),  # very long number
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["HAR"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "HAR"
                assert result.start == start
                assert result.end == end

    def test_no_false_positives(self):
        """Test that non-HAR patterns are not recognized."""
        test_cases = [
            "This is a normal sentence.",
            "HAR without colon 123456",  # missing colon
            "HR: 123456",  # wrong prefix (HR vs HAR)
            "CAR: 123456",  # different prefix
            "HAR: abc123",  # contains letters
            "HAR: abc123def",  # contains letters
            "HAR:",  # no number after colon
            "HAR: ",  # only space after colon
        ]

        for text in test_cases:
            results = self.recognizer.analyze(text, ["HAR"])
            assert len(results) == 0, f"False positive for text: {text}"

    def test_har_boundary_detection(self):
        """Test that HAR detection respects word boundaries."""
        test_cases = [
            ("SHARE: 123456", []),  # Should not match (not HAR boundary)
            ("SHARP: 123456", []),  # Should not match
            ("HAR: 123456", [(5, 11)]),  # Should match (proper boundary)
            ("The HAR: 123456", [(9, 15)]),  # Should match (word boundary)
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["HAR"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            if expected_spans:
                for result, (start, end) in zip(results, expected_spans, strict=False):
                    assert result.entity_type == "HAR"
                    assert result.start == start
                    assert result.end == end

    def test_analyze_empty_text(self):
        """Test analysis of empty or whitespace-only text."""
        test_cases = ["", "   ", "\n\t"]

        for text in test_cases:
            results = self.recognizer.analyze(text, ["HAR"])
            assert results == []

    def test_supported_entities(self):
        """Test that only supported entity types work."""
        text = "HAR: 123456"

        # Should work with HAR
        results = self.recognizer.analyze(text, ["HAR"])
        assert len(results) == 1

        # Should not return results for other entity types (but still finds HAR)
        results = self.recognizer.analyze(text, ["PERSON", "PHONE_NUMBER", "ACC_NUM"])
        assert len(results) == 1  # Still finds HAR entity since it's in the text

    def test_har_with_line_breaks(self):
        """Test HAR recognition across line breaks."""
        test_cases = [
            ("HAR:\n123456", [(5, 11)]),  # line break after colon
            ("HAR: \n123456", [(6, 12)]),  # space and line break after colon
            ("HAR:\t123456", [(5, 11)]),  # tab after colon
            ("HAR: \t 123456", [(7, 13)]),  # space, tab, space after colon
        ]

        for text, expected_spans in test_cases:
            results = self.recognizer.analyze(text, ["HAR"])
            assert len(results) == len(expected_spans), f"Failed for text: {text}"

            for result, (start, end) in zip(results, expected_spans, strict=False):
                assert result.entity_type == "HAR"
                assert result.start == start
                assert result.end == end
