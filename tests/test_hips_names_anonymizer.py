"""
Unit tests for HipsNamesAnonymizer.

Tests cover HIPS name anonymization including:
- Unified name list loading
- Name parsing and component replacement
- Cryptographic string selection for names
- Key validation
- Consistent replacement regardless of context
"""

import string

import pytest

from tide2.anonymizers.hips_names import HipsNamesAnonymizer
from tide2.cryptographic.keys_utils import derive_key
from tide2.cryptographic.keys_utils import generate_salt


class TestHipsNamesAnonymizer:
    """Test HipsNamesAnonymizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create the anonymizer with real implementation (no mocking of repository methods)
        self.anonymizer = HipsNamesAnonymizer()

        # Use internal key generation functions to create proper cryptographic keys
        self.valid_salt = generate_salt()
        self.valid_key = derive_key("test_input_for_deterministic_key")

    def test_initialization(self):
        """Test HipsNamesAnonymizer initialization."""
        assert hasattr(self.anonymizer, "unified_names_list")
        assert hasattr(self.anonymizer, "initials_list")
        assert hasattr(self.anonymizer, "name_parser")
        assert self.anonymizer.supported_entity_types == ["PERSON", "DOCTOR", "PATIENT", "HCW"]

    def test_operator_name(self):
        """Test operator_name method."""
        assert self.anonymizer.operator_name() == "hips_names"

    def test_operator_type(self):
        """Test operator_type method."""
        from presidio_anonymizer.operators import OperatorType

        assert self.anonymizer.operator_type() == OperatorType.Anonymize

    def test_unified_names_list(self):
        """Test that unified names list is correctly loaded."""
        assert isinstance(self.anonymizer.unified_names_list, list)
        assert len(self.anonymizer.unified_names_list) > 0
        # Names should be lowercase
        for name in self.anonymizer.unified_names_list[:10]:
            assert name == name.lower()

    def test_initials_list(self):
        """Test that initials list contains uppercase letters."""
        assert isinstance(self.anonymizer.initials_list, list)
        assert len(self.anonymizer.initials_list) == len(string.ascii_uppercase)
        assert "A" in self.anonymizer.initials_list
        assert "Z" in self.anonymizer.initials_list

    # Test validation
    def test_validate_valid_params(self):
        """Test validate method with valid parameters."""
        params = {"entity_type": "PERSON", "salt": self.valid_salt, "key": self.valid_key}
        # Should not raise any exception
        self.anonymizer.validate(params)

    def test_validate_all_supported_entity_types(self):
        """Test validate method with all supported entity types."""
        for entity_type in self.anonymizer.supported_entity_types:
            params = {
                "entity_type": entity_type,
                "salt": self.valid_salt,
                "key": self.valid_key,
            }
            # Should not raise any exception
            self.anonymizer.validate(params)

    def test_validate_invalid_entity_type(self):
        """Test validate method with invalid entity type."""
        params = {
            "entity_type": "INVALID_TYPE",
            "salt": self.valid_salt,
            "key": self.valid_key,
        }
        with pytest.raises(ValueError, match="Entity type 'INVALID_TYPE' is not supported"):
            self.anonymizer.validate(params)

    def test_validate_missing_salt(self):
        """Test validate method with missing salt."""
        params = {"entity_type": "PERSON", "key": self.valid_key}
        with pytest.raises(ValueError, match="Both 'salt' and 'key' must be provided"):
            self.anonymizer.validate(params)

    def test_validate_missing_key(self):
        """Test validate method with missing key."""
        params = {"entity_type": "PERSON", "salt": self.valid_salt}
        with pytest.raises(ValueError, match="Both 'salt' and 'key' must be provided"):
            self.anonymizer.validate(params)

    # Test operate method
    def test_operate_full_name_parsing(self):
        """Test operate method with full name that gets parsed completely."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        # Test with a simple name format that our parser can handle
        result = self.anonymizer.operate("John Michael Smith", params)

        # Result should be a string (the actual formatting depends on the name parser implementation)
        assert isinstance(result, str)
        assert len(result) > 0

        # Test that the result is different from the input (should be anonymized)
        assert result != "John Michael Smith"

    def test_operate_with_middle_initial(self):
        """Test operate method with middle initial instead of full middle name."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        # Test with a name that has middle initial
        result = self.anonymizer.operate("Jane M. Doe", params)

        # Result should be a string
        assert isinstance(result, str)
        assert len(result) > 0

        # Test that the result is different from the input (should be anonymized)
        assert result != "Jane M. Doe"

    def test_operate_partial_name_components(self):
        """Test operate method with partial name components (single name)."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        # Test with just a first name
        result = self.anonymizer.operate("John", params)

        # Result should be a string
        assert isinstance(result, str)
        assert len(result) > 0

        # Test that the result is different from the input (should be anonymized)
        assert result != "John"

    def test_operate_without_sex_parameter(self):
        """Test operate method works without sex parameter."""
        params = {
            "salt": self.valid_salt,
            "key": self.valid_key,
        }

        result = self.anonymizer.operate("Alex Smith", params)

        # Result should be a string
        assert isinstance(result, str)
        assert len(result) > 0

        # Test that the result is different from the input (should be anonymized)
        assert result != "Alex Smith"

    # Test that anonymization works with complex names
    def test_complex_name_anonymization(self):
        """Test that complex names with multiple components are anonymized."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        result = self.anonymizer.operate("John Michael R. Smith", params)

        # Result should be a string and different from input
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "John Michael R. Smith"

    # Test edge cases
    def test_operate_empty_string(self):
        """Test operate method with empty string."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        result = self.anonymizer.operate("", params)

        # Empty input should return empty result
        assert result == ""

    def test_operate_whitespace_handling(self):
        """Test operate method handles whitespace in name components correctly."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        result = self.anonymizer.operate("  John  \tSmith\n", params)

        # Result should be a clean string without extra whitespace
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "  John  \tSmith\n"

        # Should handle the whitespace correctly
        assert result.strip() == result  # No leading/trailing whitespace

    def test_operate_case_handling(self):
        """Test operate method handles different cases correctly."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        result = self.anonymizer.operate("JOHN mIcHaEl smith", params)

        # Result should be a string and different from input
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "JOHN mIcHaEl smith"

    # Integration tests
    def test_integration_complete_workflow(self):
        """Integration test for complete name anonymization workflow."""
        params = {
            "entity_type": "DOCTOR",
            "salt": self.valid_salt,
            "key": self.valid_key,
        }

        # Test validation first
        self.anonymizer.validate(params)

        # Test operation
        result = self.anonymizer.operate("Dr. Jane Elizabeth Doe MD", params)

        # Verify complete workflow works
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != "Dr. Jane Elizabeth Doe MD"

    def test_integration_multiple_names_consistency(self):
        """Integration test that same input with same keys produces same output."""
        params = {
            "entity_type": "PERSON",
            "salt": self.valid_salt,
            "key": self.valid_key,
        }

        # Multiple calls with same parameters
        results = []
        for _ in range(3):
            result = self.anonymizer.operate("John Smith", params)
            results.append(result)

        # All results should be the same due to deterministic anonymization
        assert all(r == results[0] for r in results)
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_integration_different_entity_types(self):
        """Integration test with different supported entity types."""
        base_params = {"salt": self.valid_salt, "key": self.valid_key}

        for entity_type in self.anonymizer.supported_entity_types:
            params = {**base_params, "entity_type": entity_type}

            # Validation should pass
            self.anonymizer.validate(params)

            # Operation should work
            result = self.anonymizer.operate("Test Name", params)
            assert isinstance(result, str)
            assert len(result) > 0
            assert result != "Test Name"

    def test_same_name_consistent_replacement(self):
        """Test that the same name always maps to the same replacement regardless of context."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        # "Harrison" appearing as a standalone name
        result_standalone = self.anonymizer.operate("Harrison", params)

        # "Harrison" appearing as first name in a multi-token name
        result_first = self.anonymizer.operate("Harrison Ford", params)

        # "Harrison" appearing as last name in a multi-token name
        result_last = self.anonymizer.operate("John Harrison", params)

        # Extract the replacement for "Harrison" from each result
        # In result_first, the first token is the replacement for "harrison"
        first_token_in_multi = result_first.split()[0]
        # In result_last, the last token is the replacement for "harrison"
        last_token_in_multi = result_last.split()[-1]

        # All should produce the same replacement for "harrison"
        assert result_standalone.lower() == first_token_in_multi.lower()
        assert result_standalone.lower() == last_token_in_multi.lower()

    def test_entity_type_does_not_affect_replacement(self):
        """Test that entity type hints don't change the replacement (unified list)."""
        base_params = {"salt": self.valid_salt, "key": self.valid_key}

        results = []
        for entity_type in ["PERSON", "DOCTOR", "PATIENT", "HCW"]:
            params = {**base_params, "entity_type": entity_type}
            result = self.anonymizer.operate("Harrison", params)
            results.append(result)

        # All entity types should produce the same replacement
        assert all(r == results[0] for r in results)
