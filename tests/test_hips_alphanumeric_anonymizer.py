"""
Unit tests for HipsAlphaNumericAnonymizer.

Tests cover HIPS alphanumeric anonymization including:
- Format-preserving encryption
- Supported entity types
- Key validation
- Encryption consistency
- Error handling
"""

import pytest

from tide2.anonymizers.hips_alphanumeric import HipsAlphaNumericAnonymizer
from tide2.cryptographic.keys_utils import derive_key
from tide2.cryptographic.keys_utils import generate_salt


class TestHipsAlphaNumericAnonymizer:
    """Test HipsAlphaNumericAnonymizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.anonymizer = HipsAlphaNumericAnonymizer()
        # Generate proper bytes keys for testing
        self.valid_salt = generate_salt()
        self.valid_key = derive_key("test_input_for_key")

    def test_initialization(self):
        """Test HipsAlphaNumericAnonymizer initialization."""
        assert hasattr(self.anonymizer, "entities_supported")
        expected_entities = {
            "DEFAULT",
            "PHONE",
            "PHONE_NUMBER",
            "US_SSN",
            "MEDICAL_LICENSE",
            "MRN",
            "HAR",
            "ACC_NUM",
            "ID",
            "CSN_ID",
        }
        assert self.anonymizer.entities_supported == expected_entities

    def test_operator_name(self):
        """Test operator_name method."""
        assert self.anonymizer.operator_name() == "hips_alphanumeric"

    def test_operator_type(self):
        """Test operator_type method."""
        from presidio_anonymizer.operators import OperatorType

        assert self.anonymizer.operator_type() == OperatorType.Anonymize

    # Test validation
    def test_validate_valid_params(self):
        """Test validate method with valid parameters."""
        params = {"entity_type": "MRN", "salt": self.valid_salt, "key": self.valid_key}
        # Should not raise any exception
        self.anonymizer.validate(params)

    def test_validate_default_entity_type(self):
        """Test validate method with default entity type."""
        params = {"salt": self.valid_salt, "key": self.valid_key}
        # Should use DEFAULT entity type and not raise exception
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
        params = {"entity_type": "MRN", "key": self.valid_key}
        with pytest.raises(ValueError, match="Both 'salt' and 'key' must be provided"):
            self.anonymizer.validate(params)

    def test_validate_missing_key(self):
        """Test validate method with missing key."""
        params = {"entity_type": "MRN", "salt": self.valid_salt}
        with pytest.raises(ValueError, match="Both 'salt' and 'key' must be provided"):
            self.anonymizer.validate(params)

    def test_validate_empty_salt(self):
        """Test validate method with empty salt."""
        params = {"entity_type": "MRN", "salt": "", "key": self.valid_key}
        with pytest.raises(ValueError, match="Both 'salt' and 'key' must be provided"):
            self.anonymizer.validate(params)

    def test_validate_none_salt(self):
        """Test validate method with None salt."""
        params = {"entity_type": "MRN", "salt": None, "key": self.valid_key}
        with pytest.raises(ValueError, match="Both 'salt' and 'key' must be provided"):
            self.anonymizer.validate(params)

    def test_validate_empty_key(self):
        """Test validate method with empty key."""
        params = {"entity_type": "MRN", "salt": self.valid_salt, "key": ""}
        with pytest.raises(ValueError, match="Both 'salt' and 'key' must be provided"):
            self.anonymizer.validate(params)

    def test_validate_none_key(self):
        """Test validate method with None key."""
        params = {"entity_type": "MRN", "salt": self.valid_salt, "key": None}
        with pytest.raises(ValueError, match="Both 'salt' and 'key' must be provided"):
            self.anonymizer.validate(params)

    # Test operate method
    def test_operate_basic_functionality(self):
        """Test basic operate functionality."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        result = self.anonymizer.operate("test_input", params)

        # Verify result is a string and different from input (encrypted)
        assert isinstance(result, str)
        assert result != "test_input"  # Should be encrypted/anonymized

    def test_operate_different_inputs(self):
        """Test operate method with different input texts."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        test_inputs = ["123456789", "ABC-123-DEF", "555-123-4567", "MRN123456"]

        for input_text in test_inputs:
            result = self.anonymizer.operate(input_text, params)

            # Verify result is a string
            assert isinstance(result, str)
            # Verify deterministic behavior - same input produces same output
            result2 = self.anonymizer.operate(input_text, params)
            assert result == result2

    def test_operate_different_keys(self):
        """Test operate method with different key pairs."""
        # Generate different key pairs
        key_pairs = []
        for i in range(3):
            salt = generate_salt()
            key = derive_key(f"test_input_{i}")
            key_pairs.append((salt, key))

        input_text = "test"
        results = []

        for salt, key in key_pairs:
            params = {"salt": salt, "key": key}

            result = self.anonymizer.operate(input_text, params)
            results.append(result)

            # Verify result is a string
            assert isinstance(result, str)

        # Verify different keys produce different results
        assert len(set(results)) == len(results)  # All results should be unique

    # Test all supported entity types
    def test_operate_all_supported_entities(self):
        """Test operate method with all supported entity types."""
        for entity_type in self.anonymizer.entities_supported:
            params = {
                "entity_type": entity_type,
                "salt": self.valid_salt,
                "key": self.valid_key,
            }

            # Should not raise validation error
            self.anonymizer.validate(params)

            # Should work with operate
            result = self.anonymizer.operate(f"test_{entity_type}", params)
            assert isinstance(result, str)

    def test_fpe_integration(self):
        """Test that FormatPreservingEncryption.encrypt works correctly."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        input_text = "123-45-6789"
        result = self.anonymizer.operate(input_text, params)

        # Verify we get a string result
        assert isinstance(result, str)
        # Verify result is different from input (encrypted)
        assert result != input_text

    def test_fpe_exception_handling(self):
        """Test handling of invalid key lengths."""
        # Test with invalid salt length (not 32 bytes)
        invalid_salt = b"short"
        params = {"salt": invalid_salt, "key": self.valid_key}

        with pytest.raises(ValueError, match="Salt must be exactly 32 bytes"):
            self.anonymizer.operate("test_input", params)

    def test_operate_empty_string(self):
        """Test operate method with empty string."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        result = self.anonymizer.operate("", params)

        # Verify result is a string
        assert isinstance(result, str)

    def test_operate_special_characters(self):
        """Test operate method with special characters."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        special_input = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = self.anonymizer.operate(special_input, params)

        # Verify result is a string
        assert isinstance(result, str)

    def test_operate_unicode_characters(self):
        """Test operate method with Unicode characters."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        unicode_input = "测试123αβγ"
        result = self.anonymizer.operate(unicode_input, params)

        # Verify result is a string
        assert isinstance(result, str)

    # Integration tests
    def test_integration_complete_workflow(self):
        """Integration test for complete anonymization workflow."""
        # Test complete workflow: validate -> operate
        params = {"entity_type": "MRN", "salt": self.valid_salt, "key": self.valid_key}

        # First validate
        self.anonymizer.validate(params)

        # Then operate
        result = self.anonymizer.operate("MRN123456", params)

        # Verify result is a string
        assert isinstance(result, str)

    def test_integration_multiple_operations_same_keys(self):
        """Integration test for multiple operations with same keys."""
        params = {"entity_type": "MRN", "salt": self.valid_salt, "key": self.valid_key}

        inputs = ["MRN123", "MRN456", "MRN789"]
        results = []

        for input_text in inputs:
            result = self.anonymizer.operate(input_text, params)
            results.append(result)
            assert isinstance(result, str)

        # Verify results are all strings and deterministic
        for i, input_text in enumerate(inputs):
            result2 = self.anonymizer.operate(input_text, params)
            assert results[i] == result2

    def test_integration_different_entity_types_same_keys(self):
        """Integration test for different entity types with same keys."""
        base_params = {"salt": self.valid_salt, "key": self.valid_key}

        test_cases = [("MRN", "MRN123456"), ("HAR", "HAR789012"), ("US_SSN", "123-45-6789"), ("PHONE", "555-123-4567")]

        for entity_type, input_text in test_cases:
            params = {**base_params, "entity_type": entity_type}

            # Validate first
            self.anonymizer.validate(params)

            # Then operate
            result = self.anonymizer.operate(input_text, params)
            assert isinstance(result, str)

    def test_param_extraction(self):
        """Test that parameters are correctly extracted from params dict."""
        # Test with extra parameters that should be ignored
        params = {
            "entity_type": "MRN",
            "salt": self.valid_salt,
            "key": self.valid_key,
            "extra_param1": "ignored",
            "extra_param2": 123,
            "extra_param3": True,
        }

        result = self.anonymizer.operate("test", params)

        # Should work despite extra parameters
        assert isinstance(result, str)

    def test_consistency_across_calls(self):
        """Test that same inputs with same keys always produce same outputs."""
        params = {"salt": self.valid_salt, "key": self.valid_key}

        input_text = "consistent_test"

        # Call multiple times
        results = []
        for _ in range(5):
            result = self.anonymizer.operate(input_text, params)
            results.append(result)

        # All results should be identical (deterministic)
        assert all(r == results[0] for r in results)

        # But should be different from input
        assert results[0] != input_text
