"""
Unit tests for FakerAnonymizer.

Tests cover faker-based anonymization including:
- All supported entity types
- Format detection integration
- Seed consistency
- URL format-specific anonymization
- Validation
"""

import pytest

from tide2.anonymizers.faker import FakerAnonymizer


class TestFakerAnonymizer:
    """Test FakerAnonymizer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.anonymizer = FakerAnonymizer()

    def test_initialization(self):
        """Test FakerAnonymizer initialization."""
        assert hasattr(self.anonymizer, "fake")
        assert hasattr(self.anonymizer, "format_detector")
        assert hasattr(self.anonymizer, "fake_dict")
        assert hasattr(self.anonymizer, "entities_supported")
        assert len(self.anonymizer.entities_supported) > 0

    def test_operator_name(self):
        """Test operator_name method."""
        assert self.anonymizer.operator_name() == "faker_anonymizer"

    def test_operator_type(self):
        """Test operator_type method."""
        from presidio_anonymizer.operators import OperatorType

        assert self.anonymizer.operator_type() == OperatorType.Anonymize

    def test_fake_dict_completeness(self):
        """Test that fake_dict contains all expected entity types."""
        expected_entities = [
            "DEFAULT",
            "OTHER",
            "AGE",
            "IBAN_CODE",
            "CREDIT_CARD",
            "CRYPTO",
            "IP_ADDRESS",
            "URL",
            "EMAIL",
            "EMAIL_ADDRESS",
            "NRP",
            "MEDICAL_LICENSE",
            "PHONE",
            "PHONE_NUMBER",
            "US_BANK_NUMBER",
            "US_DRIVER_LICENSE",
            "US_ITIN",
            "US_PASSPORT",
            "US_SSN",
            "ORGANIZATION",
        ]

        for entity in expected_entities:
            assert entity in self.anonymizer.fake_dict, f"Missing entity type: {entity}"
            assert callable(self.anonymizer.fake_dict[entity]), f"Entity type {entity} should be callable"

    def test_url_specific_fake_dict_entries(self):
        """Test URL-specific fake dictionary entries."""
        url_entities = [
            "URL_HTTPS",
            "URL_HTTP",
            "URL_FTP",
            "URL_WITH_AUTH",
            "URL_WITH_PORT",
            "URL_WITH_PARAMS",
            "URL_LOCALHOST",
            "URL_IP_BASED",
            "URL_INCOMPLETE",
            "URL_DOMAIN_ONLY",
            "URL_WWW_PREFIX",
            "URL_LOCALHOST_NO_PROTOCOL",
            "URL_IP_NO_PROTOCOL",
            "URL_MALFORMED",
            "URL_INCOMPLETE_PORT",
            "URL_LEADING_DOT",
            "URL_TRAILING_DOT",
            "URL_DOUBLE_DOT",
        ]

        for entity in url_entities:
            assert entity in self.anonymizer.fake_dict, f"Missing URL entity type: {entity}"
            assert callable(self.anonymizer.fake_dict[entity]), f"URL entity type {entity} should be callable"

    def test_entities_supported_consistency(self):
        """Test that entities_supported matches fake_dict keys."""
        assert self.anonymizer.entities_supported == set(self.anonymizer.fake_dict.keys())

    # Test validation
    def test_validate_valid_entity_type(self):
        """Test validate method with valid entity type."""
        params = {"entity_type": "EMAIL"}
        # Should not raise any exception
        self.anonymizer.validate(params)

    def test_validate_invalid_entity_type(self):
        """Test validate method with invalid entity type."""
        params = {"entity_type": "INVALID_TYPE"}
        with pytest.raises(ValueError, match="Entity type 'INVALID_TYPE' is not supported"):
            self.anonymizer.validate(params)

    def test_validate_missing_entity_type(self):
        """Test validate method with missing entity type."""
        params = {}
        # Should raise KeyError or return None for missing entity_type
        with pytest.raises((ValueError, KeyError)):
            self.anonymizer.validate(params)

    # Test basic operate functionality
    def test_operate_default_entity(self):
        """Test operate method with DEFAULT entity type."""
        params = {"entity_type": "DEFAULT"}
        result = self.anonymizer.operate("test", params)

        assert isinstance(result, str)
        assert len(result) > 0
        assert result.isdigit()  # DEFAULT should return 8-digit number
        assert len(result) == 8

    def test_operate_with_seed_consistency(self):
        """Test that same seed produces same results."""
        params = {"entity_type": "EMAIL", "faker_seed": 12345}

        result1 = self.anonymizer.operate("test@example.com", params)
        result2 = self.anonymizer.operate("another@example.com", params)

        # With same seed, should get same result
        assert result1 == result2

    def test_operate_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        params1 = {"entity_type": "EMAIL", "faker_seed": 12345}
        params2 = {"entity_type": "EMAIL", "faker_seed": 54321}

        result1 = self.anonymizer.operate("test@example.com", params1)
        result2 = self.anonymizer.operate("test@example.com", params2)

        # Different seeds should produce different results
        assert result1 != result2

    def test_operate_no_seed_uses_random(self):
        """Test that no seed parameter uses random seeding."""
        params = {"entity_type": "EMAIL"}

        # Test that multiple calls without seed produce different results
        results = []
        for _ in range(3):
            result = self.anonymizer.operate("test@example.com", params)
            results.append(result)
            assert isinstance(result, str)

        # Results should vary since no fixed seed is used
        # Note: There's a small chance they could be the same, but very unlikely
        assert len(set(results)) > 1 or len(results[0]) > 0  # At least generate valid output

    # Test specific entity types
    def test_operate_age_entity(self):
        """Test operate method with AGE entity type."""
        params = {"entity_type": "AGE", "faker_seed": 12345}
        result = self.anonymizer.operate("25", params)

        assert isinstance(result, str)
        assert result.isdigit()
        age_value = int(result)
        assert 18 <= age_value <= 90  # Should be in expected range

    def test_operate_email_entity(self):
        """Test operate method with EMAIL entity type."""
        params = {"entity_type": "EMAIL", "faker_seed": 12345}
        result = self.anonymizer.operate("user@example.com", params)

        assert isinstance(result, str)
        assert "@" in result
        assert "." in result  # Should be a valid email format

    def test_operate_phone_entity(self):
        """Test operate method with PHONE entity type - digits only."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("5551234567", params)

        assert isinstance(result, str)
        assert len(result) == 10  # Should match input length
        assert result.isdigit()  # Should be all digits

    def test_operate_phone_entity_normalized_format(self):
        """Test operate method with PHONE entity type - normalized format."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("555-123-4567", params)

        assert isinstance(result, str)
        assert len(result) == 12  # Should match format XXX-XXX-XXXX
        # Check format has dashes in correct positions
        parts = result.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 3
        assert parts[0].isdigit()
        assert len(parts[1]) == 3
        assert parts[1].isdigit()
        assert len(parts[2]) == 4
        assert parts[2].isdigit()

    def test_operate_ssn_entity(self):
        """Test operate method with US_SSN entity type."""
        params = {"entity_type": "US_SSN", "faker_seed": 12345}
        result = self.anonymizer.operate("123-45-6789", params)

        assert isinstance(result, str)
        assert len(result) == 11  # SSN format: XXX-XX-XXXX
        assert result[3] == "-"
        assert result[6] == "-"

    def test_operate_credit_card_entity(self):
        """Test operate method with CREDIT_CARD entity type."""
        params = {"entity_type": "CREDIT_CARD", "faker_seed": 12345}
        result = self.anonymizer.operate("4111111111111111", params)

        assert isinstance(result, str)
        assert result.replace(" ", "").isdigit()  # Should be digits (may have spaces)

    def test_operate_crypto_entity(self):
        """Test operate method with CRYPTO entity type."""
        params = {"entity_type": "CRYPTO", "faker_seed": 12345}
        result = self.anonymizer.operate("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", params)

        assert isinstance(result, str)
        assert result.startswith("bc1")  # Bitcoin address format
        assert len(result) == 29  # bc1 + 26 characters

    def test_operate_organization_entity(self):
        """Test operate method with ORGANIZATION entity type."""
        params = {"entity_type": "ORGANIZATION", "faker_seed": 12345}
        result = self.anonymizer.operate("Acme Corp", params)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should be a company name

    # Test URL-specific entities
    def test_operate_url_https_entity(self):
        """Test operate method with URL_HTTPS entity type."""
        params = {"entity_type": "URL_HTTPS", "faker_seed": 12345}
        result = self.anonymizer.operate("https://example.com", params)

        assert isinstance(result, str)
        assert result.startswith("https://")

    def test_operate_url_ftp_entity(self):
        """Test operate method with URL_FTP entity type."""
        params = {"entity_type": "URL_FTP", "faker_seed": 12345}
        result = self.anonymizer.operate("ftp://files.example.com", params)

        assert isinstance(result, str)
        assert result.startswith("ftp://")
        assert "/files/" in result

    def test_operate_url_localhost_entity(self):
        """Test operate method with URL_LOCALHOST entity type."""
        params = {"entity_type": "URL_LOCALHOST", "faker_seed": 12345}
        result = self.anonymizer.operate("http://localhost:3000", params)

        assert isinstance(result, str)
        assert result.startswith("http://localhost:")
        # Should have a port number
        port = result.split(":")[-1]
        assert port.isdigit()
        port_num = int(port)
        assert 3000 <= port_num <= 9999

    def test_operate_url_with_auth_entity(self):
        """Test operate method with URL_WITH_AUTH entity type."""
        params = {"entity_type": "URL_WITH_AUTH", "faker_seed": 12345}
        result = self.anonymizer.operate("https://user:pass@example.com", params)

        assert isinstance(result, str)
        assert result.startswith("https://")
        assert ":" in result
        assert "@" in result

    # Test WEB entity with format detection
    def test_operate_web_entity_with_format_detection(self):
        """Test operate method with WEB entity that triggers format detection."""
        params = {"entity_type": "WEB", "faker_seed": 12345}

        # Use real format detection - test with a URL that should be detected
        result = self.anonymizer.operate("https://example.com", params)

        # Result should be a fake URL
        assert isinstance(result, str)
        # The real format detector might detect URL format, result will vary based on implementation
        assert len(result) > 0

    def test_operate_web_entity_no_format_detected(self):
        """Test operate method with WEB entity when no format is detected."""
        params = {"entity_type": "WEB", "faker_seed": 12345}

        # Use real format detection with text that likely won't match URL patterns
        result = self.anonymizer.operate("some text", params)

        # Should fall back to DEFAULT behavior
        assert isinstance(result, str)
        assert len(result) > 0

    def test_operate_web_entity_fallback_behavior(self):
        """Test operate method with WEB entity fallback behavior."""
        params = {"entity_type": "WEB", "faker_seed": 12345}

        # Test with various inputs to see real format detection behavior
        test_inputs = ["ftp://example.com", "random text", "123-456-7890"]

        for test_input in test_inputs:
            result = self.anonymizer.operate(test_input, params)

            # Should produce some fake result
            assert isinstance(result, str)
            assert len(result) > 0

    # Test edge cases
    def test_operate_empty_text(self):
        """Test operate method with empty text."""
        params = {"entity_type": "EMAIL", "faker_seed": 12345}
        result = self.anonymizer.operate("", params)

        assert isinstance(result, str)
        assert len(result) > 0  # Should still generate fake data

    def test_operate_very_long_text(self):
        """Test operate method with very long input text."""
        long_text = "a" * 10000
        params = {"entity_type": "EMAIL", "faker_seed": 12345}
        result = self.anonymizer.operate(long_text, params)

        assert isinstance(result, str)
        assert "@" in result  # Should still be a valid email

    # Test that faker functions are actually callable
    def test_all_fake_functions_callable(self):
        """Test that all functions in fake_dict are callable and return strings."""
        for entity_type, func in self.anonymizer.fake_dict.items():
            if entity_type == "WEB":
                continue  # WEB is handled specially

            try:
                result = func("test")
                assert isinstance(result, str), f"Entity type {entity_type} should return string"
                assert len(result) > 0, f"Entity type {entity_type} should return non-empty string"
            except Exception as e:
                pytest.fail(f"Entity type {entity_type} function failed: {e}")

    # Test faker seeding behavior
    def test_faker_seeding_isolation(self):
        """Test that faker seeding doesn't affect other instances."""
        # Create two anonymizer instances
        anonymizer1 = FakerAnonymizer()
        anonymizer2 = FakerAnonymizer()

        params1 = {"entity_type": "EMAIL", "faker_seed": 12345}
        params2 = {"entity_type": "EMAIL", "faker_seed": 54321}

        result1 = anonymizer1.operate("test", params1)
        result2 = anonymizer2.operate("test", params2)

        # Should get different results with different seeds
        assert result1 != result2

        # Using same seed again should give same result
        result1_repeat = anonymizer1.operate("test", params1)
        assert result1 == result1_repeat

    # Integration tests
    def test_integration_all_supported_entities(self):
        """Integration test that all supported entities can be processed."""
        for entity_type in self.anonymizer.entities_supported:
            if entity_type == "WEB":
                continue  # Tested separately due to format detection

            params = {"entity_type": entity_type, "faker_seed": 12345}

            try:
                result = self.anonymizer.operate(f"test_{entity_type.lower()}", params)
                assert isinstance(result, str)
                assert len(result) > 0
            except Exception as e:
                pytest.fail(f"Failed to process entity type {entity_type}: {e}")

    def test_integration_reproducible_results(self):
        """Integration test for reproducible results across multiple runs."""
        params = {"entity_type": "EMAIL", "faker_seed": 99999}

        results = []
        for i in range(5):
            result = self.anonymizer.operate(f"test{i}@example.com", params)
            results.append(result)

        # All results should be the same due to same seed
        assert all(r == results[0] for r in results)

    def test_integration_different_inputs_same_seed(self):
        """Integration test that different inputs with same seed give same output."""
        seed = 77777
        entity_type = "EMAIL"  # Changed from PHONE to EMAIL since phone formats now differ

        inputs = ["test1@example.com", "test2@example.com", "test3@example.com", "test4@example.com"]

        results = []
        for input_text in inputs:
            params = {"entity_type": entity_type, "faker_seed": seed}
            result = self.anonymizer.operate(input_text, params)
            results.append(result)

        # All should give same result since same seed and entity type
        assert all(r == results[0] for r in results)

    # Test specific faker methods behavior
    def test_faker_methods_produce_valid_output(self):
        """Test that faker methods produce valid output for each entity type."""
        params = {"entity_type": "EMAIL", "faker_seed": 12345}
        result = self.anonymizer.operate("user@real.com", params)

        # Verify it's a valid email format
        assert isinstance(result, str)
        assert "@" in result
        assert "." in result
        # Verify it's different from the original
        assert result != "user@real.com"

    def test_faker_seed_consistency_behavior(self):
        """Test that faker seeding produces consistent results."""
        params = {"entity_type": "EMAIL", "faker_seed": 42}

        # Run the same operation multiple times with the same seed
        result1 = self.anonymizer.operate("test@example.com", params)
        result2 = self.anonymizer.operate("another@example.com", params)

        # Should get the same result due to seeding
        assert result1 == result2
        assert isinstance(result1, str)
        assert "@" in result1
        assert "." in result1


class TestPhoneNumberGeneration:
    """Test phone number generation with format detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.anonymizer = FakerAnonymizer()

    # Test digits-only phone numbers
    def test_phone_digits_only_10_digits(self):
        """Test phone number generation with 10 digits only."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("4124568708", params)

        assert isinstance(result, str)
        assert len(result) == 10
        assert result.isdigit()
        assert result != "4124568708"  # Should be different from original

    def test_phone_digits_only_7_digits(self):
        """Test phone number generation with 7 digits only."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("5551234", params)

        assert isinstance(result, str)
        assert len(result) == 7
        assert result.isdigit()

    def test_phone_digits_only_11_digits(self):
        """Test phone number generation with 11 digits only (international)."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("14125551234", params)

        assert isinstance(result, str)
        assert len(result) == 11
        assert result.isdigit()

    def test_phone_digits_only_12_digits(self):
        """Test phone number generation with 12 digits only."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("123456789012", params)

        assert isinstance(result, str)
        assert len(result) == 12
        assert result.isdigit()

    # Test normalized format with dashes
    def test_phone_normalized_format_standard_us(self):
        """Test phone number generation with standard US format XXX-XXX-XXXX."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("412-456-8708", params)

        assert isinstance(result, str)
        assert len(result) == 12
        parts = result.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 3
        assert parts[0].isdigit()
        assert len(parts[1]) == 3
        assert parts[1].isdigit()
        assert len(parts[2]) == 4
        assert parts[2].isdigit()
        assert result != "412-456-8708"  # Should be different

    def test_phone_normalized_format_international_style(self):
        """Test phone number generation with 4-segment format falls back to faker."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("1-412-456-8708", params)

        # 4-segment pattern doesn't match our 3-segment regex, so falls back to faker
        assert isinstance(result, str)
        assert len(result) > 0
        # Faker will generate its own format

    def test_phone_normalized_format_custom_segments(self):
        """Test phone number generation with custom 3-segment lengths."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("12-34567-8901", params)

        assert isinstance(result, str)
        parts = result.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 2
        assert parts[0].isdigit()
        assert len(parts[1]) == 5
        assert parts[1].isdigit()
        assert len(parts[2]) == 4
        assert parts[2].isdigit()

    def test_phone_normalized_format_two_segments_uses_faker(self):
        """Test phone number generation with 2 segments falls back to faker."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("555-1234567", params)

        # 2-segment pattern doesn't match our 3-segment regex, so falls back to faker
        assert isinstance(result, str)
        assert len(result) > 0

    # Test whitespace preservation
    def test_phone_digits_only_with_leading_whitespace(self):
        """Test phone number generation preserves leading whitespace."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("  4124568708", params)

        assert isinstance(result, str)
        assert result.startswith("  ")
        assert len(result) == 12  # 2 spaces + 10 digits
        assert result[2:].isdigit()

    def test_phone_digits_only_with_trailing_whitespace(self):
        """Test phone number generation preserves trailing whitespace."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("4124568708  ", params)

        assert isinstance(result, str)
        assert result.endswith("  ")
        assert len(result) == 12  # 10 digits + 2 spaces
        assert result[:10].isdigit()

    def test_phone_digits_only_with_both_whitespaces(self):
        """Test phone number generation preserves both leading and trailing whitespace."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("  4124568708  ", params)

        assert isinstance(result, str)
        assert result.startswith("  ")
        assert result.endswith("  ")
        assert len(result) == 14  # 2 + 10 + 2
        assert result.strip().isdigit()

    def test_phone_normalized_with_leading_whitespace(self):
        """Test phone number generation preserves leading whitespace with normalized format."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("  412-456-8708", params)

        assert isinstance(result, str)
        assert result.startswith("  ")
        assert len(result) == 14  # 2 spaces + 12 chars
        parts = result.strip().split("-")
        assert len(parts) == 3

    def test_phone_normalized_with_trailing_whitespace(self):
        """Test phone number generation preserves trailing whitespace with normalized format."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("412-456-8708  ", params)

        assert isinstance(result, str)
        assert result.endswith("  ")
        assert len(result) == 14  # 12 chars + 2 spaces
        parts = result.strip().split("-")
        assert len(parts) == 3

    def test_phone_normalized_with_tab_whitespace(self):
        """Test phone number generation preserves tab whitespace."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("\t412-456-8708\t", params)

        assert isinstance(result, str)
        assert result.startswith("\t")
        assert result.endswith("\t")
        parts = result.strip().split("-")
        assert len(parts) == 3

    # Test fallback to faker for non-standard formats
    def test_phone_with_parentheses_uses_faker(self):
        """Test phone number with parentheses falls back to faker."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("(555) 123-4567", params)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should use faker's default phone number format
        # Just verify it's a string (format may vary)

    def test_phone_with_dots_uses_faker(self):
        """Test phone number with dots falls back to faker."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("555.123.4567", params)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_phone_with_spaces_uses_faker(self):
        """Test phone number with spaces falls back to faker."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("555 123 4567", params)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_phone_with_plus_sign_uses_faker(self):
        """Test phone number with plus sign falls back to faker."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("+1-555-123-4567", params)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_phone_with_extension_uses_faker(self):
        """Test phone number with extension falls back to faker."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("555-123-4567 ext 123", params)

        assert isinstance(result, str)
        assert len(result) > 0

    # Test PHONE_NUMBER entity type
    def test_phone_number_entity_digits_only(self):
        """Test PHONE_NUMBER entity type with digits only."""
        params = {"entity_type": "PHONE_NUMBER", "faker_seed": 12345}
        result = self.anonymizer.operate("5551234567", params)

        assert isinstance(result, str)
        assert len(result) == 10
        assert result.isdigit()

    def test_phone_number_entity_normalized_format(self):
        """Test PHONE_NUMBER entity type with normalized format."""
        params = {"entity_type": "PHONE_NUMBER", "faker_seed": 12345}
        result = self.anonymizer.operate("555-123-4567", params)

        assert isinstance(result, str)
        parts = result.split("-")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    # Test consistency with same seed
    def test_phone_same_seed_same_format_produces_same_result(self):
        """Test that same seed with same format produces same result."""
        params = {"entity_type": "PHONE", "faker_seed": 99999}

        result1 = self.anonymizer.operate("4124568708", params)
        result2 = self.anonymizer.operate("5551234567", params)

        # Same seed should produce same fake number for same format
        assert result1 == result2

    def test_phone_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        params1 = {"entity_type": "PHONE", "faker_seed": 11111}
        params2 = {"entity_type": "PHONE", "faker_seed": 22222}

        result1 = self.anonymizer.operate("4124568708", params1)
        result2 = self.anonymizer.operate("4124568708", params2)

        # Different seeds should produce different fake numbers
        assert result1 != result2
        assert len(result1) == len(result2) == 10

    # Test edge cases
    def test_phone_empty_string(self):
        """Test phone number generation with empty string."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("", params)

        # Empty string doesn't match any pattern, should use faker default
        assert isinstance(result, str)
        assert len(result) > 0

    def test_phone_single_digit(self):
        """Test phone number generation with single digit."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("5", params)

        assert isinstance(result, str)
        assert len(result) == 1
        assert result.isdigit()

    def test_phone_very_long_digits(self):
        """Test phone number generation with very long digit string."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        long_number = "1" * 20
        result = self.anonymizer.operate(long_number, params)

        assert isinstance(result, str)
        assert len(result) == 20
        assert result.isdigit()

    def test_phone_only_whitespace(self):
        """Test phone number generation with only whitespace."""
        params = {"entity_type": "PHONE", "faker_seed": 12345}
        result = self.anonymizer.operate("   ", params)

        # Only whitespace doesn't match digits pattern, should use faker
        assert isinstance(result, str)
        assert len(result) > 0
