"""
Tests for AccessionRecognizer.
"""

import pytest

from tide2.recognizers import AccessionRecognizer


class TestAccessionRecognizer:
    """Tests for AccessionRecognizer."""

    @pytest.fixture
    def recognizer(self):
        """Create accession recognizer."""
        return AccessionRecognizer()

    def test_initialization(self, recognizer):
        """Test that optimized recognizer initializes correctly."""
        assert recognizer.name == "AccessionRecognizer"
        assert "ACC_NUM" in recognizer.get_supported_entities()

    def test_accession_with_keyword(self, recognizer):
        """Test accession number with ACCESSION keyword."""
        text = "ACCESSION NUMBER: RAD-12345678"
        results = recognizer.analyze(text, ["ACC_NUM"])
        assert len(results) >= 1
        # Should capture the actual accession value
        matched_texts = [text[r.start : r.end] for r in results]
        assert any("RAD" in m or "12345678" in m for m in matched_texts)

    def test_accession_misspelling(self, recognizer):
        """Test accession with common misspelling."""
        text = "ACESSION: MR-123456789"
        results = recognizer.analyze(text, ["ACC_NUM"])
        assert len(results) >= 1

    def test_accession_abbreviation(self, recognizer):
        """Test accession with abbreviated keyword."""
        text = "ACC. NO: CT-987654321"
        results = recognizer.analyze(text, ["ACC_NUM"])
        assert len(results) >= 1

    def test_standalone_accession_format(self, recognizer):
        """Test standalone accession format without keyword."""
        text = "Report RAD1234567890"
        results = recognizer.analyze(text, ["ACC_NUM"])
        # Should match but with lower confidence
        if results:
            assert results[0].score <= 0.5  # Standalone has lower confidence

    def test_no_match_on_plain_text(self, recognizer):
        """Test that plain text doesn't match."""
        text = "The patient was seen today for a follow-up."
        results = recognizer.analyze(text, ["ACC_NUM"])
        assert len(results) == 0

    def test_multiple_accession_numbers(self, recognizer):
        """Test multiple accession numbers in text."""
        text = "ACCESSION: RAD-12345678 and CT-98765432"
        results = recognizer.analyze(text, ["ACC_NUM"])
        assert len(results) >= 1

    def test_pattern_specific_scores(self, recognizer):
        """Test that contextual matches have higher scores."""
        # With keyword context
        text1 = "ACCESSION NUMBER: RAD-12345678"
        results1 = recognizer.analyze(text1, ["ACC_NUM"])

        # Without keyword context
        text2 = "Code RAD1234567890"
        results2 = recognizer.analyze(text2, ["ACC_NUM"])

        if results1 and results2:
            # Context should give higher confidence
            assert results1[0].score > results2[0].score
