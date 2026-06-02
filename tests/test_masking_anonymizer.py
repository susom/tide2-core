"""
Unit tests for MaskingAnonymizer.

Tests cover entity type label replacement including:
- Basic masking for various entity types
- Missing entity_type fallback
- Output depends only on entity_type, not input text
"""

from presidio_anonymizer.operators import OperatorType

from tide2.anonymizers.masking import MaskingAnonymizer


class TestMaskingAnonymizer:
    """Test MaskingAnonymizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.anonymizer = MaskingAnonymizer()

    def test_operator_name(self):
        """Test operator_name method."""
        assert self.anonymizer.operator_name() == "masking"

    def test_operator_type(self):
        """Test operator_type method."""
        assert self.anonymizer.operator_type() == OperatorType.Anonymize

    def test_operate_person(self):
        """Test masking a PERSON entity."""
        result = self.anonymizer.operate("John Smith", {"entity_type": "PERSON"})
        assert result == "[PERSON]"

    def test_operate_date(self):
        """Test masking a DATE entity."""
        result = self.anonymizer.operate("01/01/2000", {"entity_type": "DATE"})
        assert result == "[DATE]"

    def test_operate_missing_entity_type(self):
        """Test masking when entity_type is not provided."""
        result = self.anonymizer.operate("some text", {})
        assert result == "[UNKNOWN]"

    def test_operate_ignores_input_text(self):
        """Test that result depends only on entity_type, not input text."""
        params = {"entity_type": "PERSON"}
        result_a = self.anonymizer.operate("John Smith", params)
        result_b = self.anonymizer.operate("Jane Doe", params)
        assert result_a == result_b == "[PERSON]"

    def test_validate_does_not_raise(self):
        """Test that validate accepts any params without raising."""
        self.anonymizer.validate({"entity_type": "PERSON"})
        self.anonymizer.validate({"entity_type": "ANYTHING"})
        self.anonymizer.validate({})
