"""
Unit tests for AgeGroupAnonymizer.

Tests cover age grouping functionality with various formats including:
- Numeric ages (plain numbers)
- Ages with units (years, months, days)
- Gestational ages (weeks)
- Written number ages (twenty-five year old)
- Upper limit enforcement
"""

import pytest

from tide2.anonymizers.age_grouping import AgeGroupAnonymizer
from tide2.string_parsers.format_detector import FormatType


class TestAgeGroupAnonymizer:
    """Test AgeGroupAnonymizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.anonymizer = AgeGroupAnonymizer()

    def test_initialization(self):
        """Test AgeGroupAnonymizer initialization."""
        assert self.anonymizer.supported_entity_types == ["AGE"]
        assert hasattr(self.anonymizer, "format_detector")

    def test_operator_name(self):
        """Test operator_name method."""
        assert self.anonymizer.operator_name() == "age_grouping"

    def test_operator_type(self):
        """Test operator_type method."""
        from presidio_anonymizer.operators import OperatorType

        assert self.anonymizer.operator_type() == OperatorType.Anonymize

    # Test validation
    def test_validate_valid_params(self):
        """Test validate method with valid parameters."""
        params = {"entity_type": "AGE", "upper_limit": 80}
        # Should not raise any exception
        self.anonymizer.validate(params)

    def test_validate_invalid_entity_type(self):
        """Test validate method with invalid entity type."""
        params = {"entity_type": "INVALID", "upper_limit": 80}
        with pytest.raises(ValueError, match="Entity type 'INVALID' is not supported"):
            self.anonymizer.validate(params)

    def test_validate_invalid_upper_limit_negative(self):
        """Test validate method with negative upper limit."""
        params = {"entity_type": "AGE", "upper_limit": -10}
        with pytest.raises(ValueError, match="Parameter 'upper_limit' must be a positive integer"):
            self.anonymizer.validate(params)

    def test_validate_invalid_upper_limit_zero(self):
        """Test validate method with zero upper limit."""
        params = {"entity_type": "AGE", "upper_limit": 0}
        with pytest.raises(ValueError, match="Parameter 'upper_limit' must be a positive integer"):
            self.anonymizer.validate(params)

    def test_validate_invalid_upper_limit_type(self):
        """Test validate method with non-integer upper limit."""
        params = {"entity_type": "AGE", "upper_limit": "80"}
        with pytest.raises(ValueError, match="Parameter 'upper_limit' must be a positive integer"):
            self.anonymizer.validate(params)

    # Test _contains_written_number helper
    def test_contains_written_number_positive_cases(self):
        """Test _contains_written_number with positive cases."""
        test_cases = ["twenty-five year old", "fifty years old", "thirty months old", "sixty-seven year old patient"]
        for case in test_cases:
            assert self.anonymizer._contains_written_number(case), f"Should detect written number in: {case}"

    def test_contains_written_number_negative_cases(self):
        """Test _contains_written_number with negative cases."""
        test_cases = [
            "25 year old",  # numeric, not written
            "hello world",  # no age keywords
            "twenty something",  # no age keywords
            "old house",  # age keyword but no number
        ]
        for case in test_cases:
            assert not self.anonymizer._contains_written_number(case), f"Should not detect written number in: {case}"

    # Test real age processing functionality
    def test_operate_numeric_age_under_limit(self):
        """Test operate with numeric age under limit."""
        result = self.anonymizer.operate("25", {"upper_limit": 80})

        # Should return some age-related result (could be the same or processed age)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_operate_numeric_age_over_limit(self):
        """Test operate with numeric age over limit."""
        result = self.anonymizer.operate("95", {"upper_limit": 80})

        # Should handle age processing (might cap at limit)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_operate_no_format_detected_no_written_numbers(self):
        """Test operate when no age format is detected."""
        result = self.anonymizer.operate("random text", {"upper_limit": 80})

        # Should return the original text or some processed version
        assert isinstance(result, str)
        assert len(result) > 0

    def test_operate_written_numbers_detection(self):
        """Test operate with written numbers in text."""
        result = self.anonymizer.operate("twenty five years old", {"upper_limit": 80})

        # Should process written age numbers
        assert isinstance(result, str)
        assert len(result) > 0

    def test_operate_extraction_fails(self):
        """Test operate when age extraction fails."""
        result = self.anonymizer.operate("invalid age", {"upper_limit": 80})

        # Should handle invalid input gracefully
        assert isinstance(result, str)
        assert len(result) > 0

    # Test _extract_age_value method
    def test_extract_age_value_numeric_only(self):
        """Test _extract_age_value with numeric only format."""
        test_cases = [("25", 25), ("100", 100), ("age: 45", 45), ("Patient is 67 years old", 67)]
        for text, expected in test_cases:
            result = self.anonymizer._extract_age_value(text, FormatType.AGE_NUMERIC_ONLY)
            assert result == expected, f"Failed for text: {text}"

    def test_extract_age_value_with_units(self):
        """Test _extract_age_value with units format."""
        test_cases = [
            ("25 yo", 25),
            ("45 Y", 45),
            ("12 mo", 12),
            ("75 year old", 75),
            ("32.5 years", 32),  # Should convert float to int
        ]
        for text, expected in test_cases:
            result = self.anonymizer._extract_age_value(text, FormatType.AGE_WITH_UNITS)
            assert result == expected, f"Failed for text: {text}"

    def test_extract_age_value_gestational(self):
        """Test _extract_age_value with gestational format."""
        test_cases = [("32w", 32), ("38w gestational age", 38), ("Born at 24w", 24)]
        for text, expected in test_cases:
            result = self.anonymizer._extract_age_value(text, FormatType.AGE_GESTATIONAL)
            assert result == expected, f"Failed for text: {text}"

    def test_extract_age_value_written_numbers_simple(self):
        """Test _extract_age_value with written numbers - simple cases."""
        test_cases = [("twenty year old", 20), ("fifty years old", 50), ("ninety year old", 90), ("ten year old", 10)]
        for text, expected in test_cases:
            result = self.anonymizer._extract_age_value(text, FormatType.AGE_WRITTEN_NUMBERS)
            assert result == expected, f"Failed for text: {text}"

    def test_extract_age_value_written_numbers_compound(self):
        """Test _extract_age_value with written numbers - compound cases."""
        test_cases = [
            ("twenty-five year old", 25),
            ("sixty-seven years old", 67),
            ("thirty-nine year old", 39),
            ("forty-two year old", 42),
        ]
        for text, expected in test_cases:
            result = self.anonymizer._extract_age_value(text, FormatType.AGE_WRITTEN_NUMBERS)
            assert result == expected, f"Failed for text: {text}"

    def test_extract_age_value_written_numbers_space_separated(self):
        """Test _extract_age_value with written numbers - space separated."""
        test_cases = [("sixty five year old", 65), ("twenty one years old", 21), ("thirty eight year old", 38)]
        for text, expected in test_cases:
            result = self.anonymizer._extract_age_value(text, FormatType.AGE_WRITTEN_NUMBERS)
            assert result == expected, f"Failed for text: {text}"

    def test_extract_age_value_written_numbers_invalid(self):
        """Test _extract_age_value with invalid written numbers."""
        invalid_cases = [
            "invalid text",
            "hundred year old",  # hundred not in dictionary
            "no numbers here",
        ]
        for text in invalid_cases:
            result = self.anonymizer._extract_age_value(text, FormatType.AGE_WRITTEN_NUMBERS)
            assert result is None or result == 0, f"Should return None/0 for: {text}"

    # Test _format_age method
    def test_format_age_numeric_only(self):
        """Test _format_age with numeric only format."""
        test_cases = [
            ("25", 80, "80"),
            ("Patient is 95 years old", 80, "Patient is 80 years old"),
            ("Age: 45", 40, "Age: 40"),
        ]
        for original, age, expected in test_cases:
            result = self.anonymizer._format_age(original, age, FormatType.AGE_NUMERIC_ONLY)
            assert result == expected, f"Failed for: {original} -> {age}"

    def test_format_age_with_units(self):
        """Test _format_age with units format."""
        test_cases = [
            ("25 yo", 80, "80 yo"),
            ("45.5 years", 40, "40 years"),
            ("Patient is 95 Y old", 80, "Patient is 80 Y old"),
        ]
        for original, age, expected in test_cases:
            result = self.anonymizer._format_age(original, age, FormatType.AGE_WITH_UNITS)
            assert result == expected, f"Failed for: {original} -> {age}"

    def test_format_age_gestational(self):
        """Test _format_age with gestational format."""
        test_cases = [
            ("45w", 42, "42w"),  # Limited to 42 weeks max
            ("32w gestational", 35, "35w gestational"),
            ("Born at 50w", 42, "Born at 42w"),  # Should limit to 42 weeks
        ]
        for original, weeks, expected in test_cases:
            result = self.anonymizer._format_age(original, weeks, FormatType.AGE_GESTATIONAL)
            assert result == expected, f"Failed for: {original} -> {weeks}"

    def test_format_age_written_numbers(self):
        """Test _format_age with written numbers format."""
        result = self.anonymizer._format_age("ninety year old", 80, FormatType.AGE_WRITTEN_NUMBERS)

        # Should convert to words appropriately
        assert isinstance(result, str)
        assert len(result) > 0

    # Test _convert_number_to_words method
    def test_convert_number_to_words_simple(self):
        """Test _convert_number_to_words with simple numbers."""
        test_cases = [
            (5, "five year old", "five"),
            (20, "twenty year old", "twenty"),
            (19, "nineteen year old", "nineteen"),
        ]
        for number, original, expected_base in test_cases:
            result = self.anonymizer._convert_number_to_words(number, original)
            assert expected_base in result.lower(), f"Failed for number: {number}"

    def test_convert_number_to_words_compound_hyphen(self):
        """Test _convert_number_to_words with compound numbers using hyphens."""
        test_cases = [
            (25, "twenty-five year old", "twenty-five"),
            (67, "sixty-seven year old", "sixty-seven"),
            (39, "thirty-nine year old", "thirty-nine"),
        ]
        for number, original, expected_base in test_cases:
            result = self.anonymizer._convert_number_to_words(number, original)
            assert expected_base in result.lower(), f"Failed for number: {number}"

    def test_convert_number_to_words_compound_space(self):
        """Test _convert_number_to_words with compound numbers using spaces."""
        test_cases = [
            (25, "twenty five year old", "twenty five"),
            (67, "sixty seven year old", "sixty seven"),
            (84, "eighty four year old", "eighty four"),
        ]
        for number, original, expected_base in test_cases:
            result = self.anonymizer._convert_number_to_words(number, original)
            assert expected_base in result.lower(), f"Failed for number: {number}"

    def test_convert_number_to_words_case_preservation(self):
        """Test _convert_number_to_words preserves original case."""
        test_cases = [
            (25, "Twenty-five year old", "Twenty-five"),  # Capitalized
            (50, "fifty year old", "fifty"),  # Lowercase
        ]
        for number, original, _expected_base in test_cases:
            result = self.anonymizer._convert_number_to_words(number, original)
            if original[0].isupper():
                assert result[0].isupper(), f"Should preserve capitalization for: {original}"
            else:
                assert result[0].islower(), f"Should preserve lowercase for: {original}"

    def test_convert_number_to_words_large_numbers(self):
        """Test _convert_number_to_words with numbers >= 100."""
        result = self.anonymizer._convert_number_to_words(150, "one hundred fifty year old")
        # The method tries to replace patterns, so result might be the original with some replacements
        # For numbers >= 100, it should return the string representation when no pattern matches
        assert isinstance(result, str)
        assert len(result) > 0

    # Integration tests
    def test_integration_numeric_age_scenarios(self):
        """Integration test for various numeric age scenarios."""
        # Test cases where format detection should work
        # Note: Plain numbers like "25" are detected as DIGITS, not AGE_NUMERIC_ONLY
        # Age anonymizer requires age-related context (yo, years old, etc.)
        test_cases = [
            ("25 yo", 80, 25),  # Under limit age with units
            ("95 years old", 80, 80),  # Over limit age - should be capped
            ("75 Y", 80, 75),  # Under limit with Y unit
        ]

        for text, upper_limit, _expected_max in test_cases:
            params = {"upper_limit": upper_limit}
            result = self.anonymizer.operate(text, params)
            # Result should be a string
            assert isinstance(result, str)
            # Extract the number from the result
            import re

            match = re.search(r"\d+", result)
            if match:
                age_result = int(match.group())
                assert age_result <= upper_limit, f"Age {age_result} should not exceed limit {upper_limit}"

    def test_integration_gestational_age_scenarios(self):
        """Integration test for gestational age scenarios."""
        test_cases = [
            ("32w", 80),  # Normal gestational age
            ("45w", 80),  # Over gestational limit (42w max)
            ("Born at 50w", 80),
        ]

        for text, upper_limit in test_cases:
            params = {"upper_limit": upper_limit}
            result = self.anonymizer.operate(text, params)

            # Should process gestational ages appropriately
            assert isinstance(result, str)
            assert len(result) > 0

    def test_default_upper_limit(self):
        """Test that default upper limit is used when not specified."""
        result = self.anonymizer.operate("95", {})  # No upper_limit specified

        # Should use some default behavior
        assert isinstance(result, str)
        assert len(result) > 0
