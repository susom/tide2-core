"""
Tests for FormatPreservingEncryption class.

Tests the generalized template-based approach that preserves any input format
by extracting alphanumeric content, encrypting it, and reinserting non-alphanumeric
characters at their original positions.
"""

import warnings

import pytest

from tide2.cryptographic.fpe_strings import FormatPreservingEncryption


class TestFormatPreservingEncryptionInitialization:
    """Test FPE initialization and key handling."""

    def test_initialization_with_valid_keys(self):
        """Test initialization with valid 32-byte keys."""
        salt = b"0" * 32
        key = b"1" * 32
        fpe = FormatPreservingEncryption(salt, key)

        assert fpe.salt == salt
        assert fpe.key == key
        assert len(fpe.encryption_key) == 16  # FF3 requires 16-byte key
        assert fpe.encryption_key_hex == fpe.encryption_key.hex()

    def test_initialization_with_invalid_salt_length(self):
        """Test that initialization fails with invalid salt length."""
        salt = b"0" * 16  # Wrong length
        key = b"1" * 32

        with pytest.raises(ValueError, match="Salt must be exactly 32 bytes"):
            FormatPreservingEncryption(salt, key)

    def test_constants_defined(self):
        """Test that all required constants are defined."""
        assert FormatPreservingEncryption.FORMAT_NUMERIC == "numeric"
        assert FormatPreservingEncryption.FORMAT_ALPHANUMERIC == "alphanumeric"
        assert FormatPreservingEncryption.FORMAT_EMPTY == "empty"
        assert FormatPreservingEncryption.FORMAT_PASSTHROUGH == "passthrough"


class TestFormatDetection:
    """Test format detection logic."""

    @pytest.fixture
    def fpe(self):
        salt = b"0" * 32
        key = b"1" * 32
        return FormatPreservingEncryption(salt, key)

    def test_detect_numeric_with_dashes(self, fpe):
        """Test that numeric with dashes is detected as numeric (based on content)."""
        # New behavior: format detection is based on alphanumeric content only
        assert fpe.detect_format("123-45-6789") == "numeric"
        assert fpe.detect_format("123-456-789") == "numeric"
        assert fpe.detect_format("12-34-56") == "numeric"

    def test_detect_alphanumeric_with_dashes(self, fpe):
        """Test that alphanumeric with dashes is detected as alphanumeric."""
        assert fpe.detect_format("ABC-123-XYZ") == "alphanumeric"
        assert fpe.detect_format("test-123-data") == "alphanumeric"
        assert fpe.detect_format("A1B2-C3D4") == "alphanumeric"

    def test_detect_numeric(self, fpe):
        """Test pure numeric format detection."""
        assert fpe.detect_format("123456789") == "numeric"
        assert fpe.detect_format("1234567890") == "numeric"
        assert fpe.detect_format("000000") == "numeric"

    def test_detect_alphanumeric(self, fpe):
        """Test pure alphanumeric format detection."""
        assert fpe.detect_format("ABC123XYZ") == "alphanumeric"
        assert fpe.detect_format("Test123") == "alphanumeric"
        assert fpe.detect_format("abcdefghij") == "alphanumeric"

    def test_detect_empty(self, fpe):
        """Test empty format detection."""
        assert fpe.detect_format("") == "empty"
        assert fpe.detect_format("   ") == "empty"

    def test_detect_with_spaces(self, fpe):
        """Test that spaces are ignored in detection (content-based)."""
        assert fpe.detect_format("123 456 789") == "numeric"
        assert fpe.detect_format("ABC 123") == "alphanumeric"

    def test_detect_passthrough(self, fpe):
        """Test passthrough for non-alphanumeric only content."""
        assert fpe.detect_format("---") == "passthrough"
        assert fpe.detect_format("@#$%") == "passthrough"


class TestNumericEncryption:
    """Test pure numeric format encryption."""

    @pytest.fixture
    def fpe(self):
        salt = b"0" * 32
        key = b"1" * 32
        return FormatPreservingEncryption(salt, key)

    def test_numeric_roundtrip(self, fpe):
        """Test numeric encryption and decryption roundtrip."""
        plaintext = "1234567890"
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted)

        assert format_type == "numeric"
        assert encrypted != plaintext
        assert decrypted == plaintext
        assert len(encrypted) == len(plaintext)
        assert all(c.isdigit() for c in encrypted)

    def test_numeric_auto_detect_decrypt(self, fpe):
        """Test numeric decryption with auto-detection."""
        plaintext = "9876543210"
        encrypted, _ = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted)  # No format specified

        assert decrypted == plaintext

    def test_numeric_explicit_format_decrypt(self, fpe):
        """Test numeric decryption with explicit format."""
        plaintext = "1112223334"
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted, format_type)

        assert decrypted == plaintext


class TestAlphanumericEncryption:
    """Test pure alphanumeric format encryption."""

    @pytest.fixture
    def fpe(self):
        salt = b"0" * 32
        key = b"1" * 32
        return FormatPreservingEncryption(salt, key)

    def test_alphanumeric_roundtrip(self, fpe):
        """Test alphanumeric encryption and decryption roundtrip."""
        plaintext = "Abc123XyzGHI"
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted)

        assert format_type == "alphanumeric"
        assert encrypted != plaintext
        assert decrypted == plaintext
        assert len(encrypted) == len(plaintext)
        assert all(c.isalnum() for c in encrypted)

    def test_alphanumeric_case_preserved(self, fpe):
        """Test that case is preserved in alphanumeric encryption."""
        plaintext = "TestData123"
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted)

        assert format_type == "alphanumeric"
        assert decrypted == plaintext


class TestFormatPreservation:
    """Test that non-alphanumeric characters are preserved at original positions."""

    @pytest.fixture
    def fpe(self):
        salt = b"0" * 32
        key = b"1" * 32
        return FormatPreservingEncryption(salt, key)

    def test_ssn_format_preserved(self, fpe):
        """Test SSN format preservation."""
        ssn = "123-45-6789"
        encrypted, format_type = fpe.encrypt(ssn)
        decrypted = fpe.decrypt(encrypted)

        assert format_type == "numeric"
        assert encrypted != ssn
        assert decrypted == ssn
        assert len(encrypted) == len(ssn)
        # Dashes should be preserved at original positions
        assert encrypted[3] == "-"
        assert encrypted[6] == "-"
        assert encrypted[:3].isdigit()
        assert encrypted[4:6].isdigit()
        assert encrypted[7:].isdigit()

    def test_phone_format_preserved(self, fpe):
        """Test phone number format preservation."""
        phone = "(555) 123-4567"
        encrypted, format_type = fpe.encrypt(phone)
        decrypted = fpe.decrypt(encrypted)

        assert format_type == "numeric"
        assert decrypted == phone
        # Format should be preserved
        assert encrypted[0] == "("
        assert encrypted[4] == ")"
        assert encrypted[5] == " "
        assert encrypted[9] == "-"

    def test_alphanumeric_dash_format_preserved(self, fpe):
        """Test alphanumeric with dashes format preservation."""
        plaintext = "ABC-123-XYZ"
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted, format_type)

        assert format_type == "alphanumeric"
        assert encrypted != plaintext
        assert decrypted == plaintext
        # Dashes should be at original positions
        assert encrypted[3] == "-"
        assert encrypted[7] == "-"
        assert len(encrypted) == len(plaintext)

    def test_various_numeric_formats(self, fpe):
        """Test various numeric formats are preserved."""
        test_cases = [
            "12-34-56",
            "1-2-3-4-5-6-7",
            "123.456.789",
            "(123) 456",
            "+1-234-567-8901",
        ]

        for plaintext in test_cases:
            encrypted, format_type = fpe.encrypt(plaintext)
            decrypted = fpe.decrypt(encrypted, format_type)

            assert decrypted == plaintext, f"Failed for {plaintext}"
            assert len(encrypted) == len(plaintext), f"Length mismatch for {plaintext}"

    def test_same_content_same_encryption(self, fpe):
        """Test that same content encrypts to same digits regardless of format."""
        formats = [
            "(555) 123-4567",
            "555-123-4567",
            "555.123.4567",
            "5551234567",
        ]

        # Extract encrypted digits from each
        encrypted_digits = []
        for fmt in formats:
            encrypted, _ = fpe.encrypt(fmt)
            digits = "".join(c for c in encrypted if c.isdigit())
            encrypted_digits.append(digits)

        # All should have the same encrypted digits
        assert all(d == encrypted_digits[0] for d in encrypted_digits)


class TestSpecialCharacterHandling:
    """Test handling of special characters - now preserved at original positions."""

    @pytest.fixture
    def fpe(self):
        salt = b"0" * 32
        key = b"1" * 32
        return FormatPreservingEncryption(salt, key)

    def test_spaces_preserved_in_numeric(self, fpe):
        """Test that spaces are preserved at original positions."""
        plaintext = "123 456 789"
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted)

        assert format_type == "numeric"
        assert decrypted == plaintext
        # Spaces should be preserved at positions 3 and 7
        assert encrypted[3] == " "
        assert encrypted[7] == " "

    def test_spaces_preserved_in_alphanumeric(self, fpe):
        """Test that spaces are preserved in alphanumeric strings."""
        plaintext = "ABC 123 XYZ"
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted)

        assert format_type == "alphanumeric"
        assert decrypted == plaintext
        # Spaces should be preserved
        assert encrypted[3] == " "
        assert encrypted[7] == " "

    def test_special_chars_preserved_in_numeric(self, fpe):
        """Test that special characters are preserved at original positions."""
        plaintext = "123*456#789"
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted)

        assert format_type == "numeric"
        assert decrypted == plaintext
        # Special chars should be at original positions
        assert encrypted[3] == "*"
        assert encrypted[7] == "#"

    def test_special_chars_preserved_in_alphanumeric(self, fpe):
        """Test that special characters are preserved in alphanumeric."""
        plaintext = "ABC@123#XYZ"
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted)

        assert format_type == "alphanumeric"
        assert decrypted == plaintext
        # Special chars should be at original positions
        assert encrypted[3] == "@"
        assert encrypted[7] == "#"


class TestShortStrings:
    """Test behavior with short strings that require padding."""

    @pytest.fixture
    def fpe(self):
        # Reset class-level flag to ensure warnings are generated
        # (other tests may have set this to True via AnonymizeActor)
        FormatPreservingEncryption.suppress_short_input_warnings = False
        salt = b"0" * 32
        key = b"1" * 32
        return FormatPreservingEncryption(salt, key)

    def test_short_numeric_warning(self, fpe):
        """Test that short numeric strings generate warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plaintext = "12345"  # Too short for secure FPE (min 6 for numeric)
            encrypted, format_type = fpe.encrypt(plaintext)
            fpe.decrypt(encrypted)

            # Should generate a warning about padding
            assert len(w) > 0
            assert "too short" in str(w[0].message).lower()
            assert format_type == "numeric"

    def test_short_alphanumeric_warning(self, fpe):
        """Test that short alphanumeric strings generate warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plaintext = "ab1"  # Too short for secure FPE (min 4 for alphanumeric)
            encrypted, format_type = fpe.encrypt(plaintext)
            fpe.decrypt(encrypted)

            # Should generate a warning about padding
            assert len(w) > 0
            assert "too short" in str(w[0].message).lower()
            assert format_type == "alphanumeric"

    def test_minimum_length_numeric_no_warning(self, fpe):
        """Test that numeric strings of minimum length don't generate warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plaintext = "123456"  # Exactly min length for numeric
            encrypted, format_type = fpe.encrypt(plaintext)
            decrypted = fpe.decrypt(encrypted)

            # Should not generate padding warnings
            padding_warnings = [warning for warning in w if "too short" in str(warning.message).lower()]
            assert len(padding_warnings) == 0

            assert format_type == "numeric"
            assert encrypted != plaintext
            assert decrypted == plaintext

    def test_minimum_length_alphanumeric_no_warning(self, fpe):
        """Test that alphanumeric strings of minimum length don't generate warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plaintext = "abcd"  # Exactly min length for alphanumeric
            encrypted, format_type = fpe.encrypt(plaintext)
            decrypted = fpe.decrypt(encrypted)

            # Should not generate padding warnings
            padding_warnings = [warning for warning in w if "too short" in str(warning.message).lower()]
            assert len(padding_warnings) == 0

            assert format_type == "alphanumeric"
            assert encrypted != plaintext
            assert decrypted == plaintext


class TestDeterministicBehavior:
    """Test that encryption is deterministic."""

    @pytest.fixture
    def fpe(self):
        salt = b"0" * 32
        key = b"1" * 32
        return FormatPreservingEncryption(salt, key)

    def test_deterministic_encryption(self, fpe):
        """Test that same input produces same encrypted output."""
        plaintext = "1234567890"

        encrypted1, format1 = fpe.encrypt(plaintext)
        encrypted2, format2 = fpe.encrypt(plaintext)

        assert encrypted1 == encrypted2
        assert format1 == format2

    def test_deterministic_with_format(self, fpe):
        """Test that formatted input produces deterministic output."""
        ssn = "123-45-6789"

        encrypted1, _ = fpe.encrypt(ssn)
        encrypted2, _ = fpe.encrypt(ssn)

        assert encrypted1 == encrypted2

    def test_deterministic_with_dashes(self, fpe):
        """Test that dash formats are deterministic."""
        plaintext = "ABC-123-XYZ"

        encrypted1, _ = fpe.encrypt(plaintext)
        encrypted2, _ = fpe.encrypt(plaintext)

        assert encrypted1 == encrypted2

    def test_different_keys_different_output(self):
        """Test that different keys produce different encrypted outputs."""
        plaintext = "1234567890"

        fpe1 = FormatPreservingEncryption(b"0" * 32, b"1" * 32)
        fpe2 = FormatPreservingEncryption(b"1" * 32, b"0" * 32)

        encrypted1, _ = fpe1.encrypt(plaintext)
        encrypted2, _ = fpe2.encrypt(plaintext)

        assert encrypted1 != encrypted2

    def test_different_keys_different_output_same_salt(self):
        """Test that different keys produce different encrypted outputs."""
        plaintext = "1234567890"
        salt = b"0" * 32

        fpe1 = FormatPreservingEncryption(salt, b"1" * 32)
        fpe2 = FormatPreservingEncryption(salt, b"2" * 32)

        encrypted1, _ = fpe1.encrypt(plaintext)
        encrypted2, _ = fpe2.encrypt(plaintext)

        assert encrypted1 != encrypted2


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def fpe(self):
        salt = b"0" * 32
        key = b"1" * 32
        return FormatPreservingEncryption(salt, key)

    def test_empty_string(self, fpe):
        """Test empty string handling."""
        plaintext = ""
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted, format_type)

        assert format_type == "empty"
        assert encrypted == plaintext == ""
        assert decrypted == plaintext

    def test_whitespace_only(self, fpe):
        """Test whitespace-only string handling."""
        plaintext = "   "
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted, format_type)

        assert format_type == "empty"
        assert encrypted == plaintext
        assert decrypted == plaintext

    def test_cipher_caching(self, fpe):
        """Test that cipher instances are cached for performance."""
        plaintext1 = "1234567890"
        plaintext2 = "9876543210"

        # Encrypt both (same format, should use cached cipher)
        fpe.encrypt(plaintext1)
        initial_cache_size = len(fpe._cipher_cache)

        fpe.encrypt(plaintext2)
        # Cache size shouldn't grow for same format
        assert len(fpe._cipher_cache) == initial_cache_size

    def test_only_special_chars(self, fpe):
        """Test string with only special characters."""
        plaintext = "---"
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted)

        # No alphanumeric content, should be passthrough
        assert format_type == "passthrough"
        assert encrypted == plaintext
        assert decrypted == plaintext

    def test_mixed_case_alphanumeric(self, fpe):
        """Test mixed case alphanumeric preserves case."""
        plaintext = "AbC123XyZ"
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted)

        assert format_type == "alphanumeric"
        assert decrypted == plaintext
        # Verify case is preserved
        assert any(c.isupper() for c in decrypted)
        assert any(c.islower() for c in decrypted)

    def test_unicode_as_template(self, fpe):
        """Test that unicode characters are treated as template (format) characters."""
        plaintext = "测试123456"  # Unicode + 6 digits (min length for numeric)
        encrypted, format_type = fpe.encrypt(plaintext)
        decrypted = fpe.decrypt(encrypted)

        # Unicode chars are preserved as template, digits are encrypted
        assert format_type == "numeric"
        assert decrypted == plaintext
        # Unicode chars should be at original positions
        assert encrypted[0] == "测"
        assert encrypted[1] == "试"
        # Remaining should be digits
        assert encrypted[2:].isdigit()

    def test_pure_unicode_passthrough(self, fpe):
        """Test that pure unicode (no alphanumeric) is passthrough."""
        plaintext = "测试中文"
        encrypted, format_type = fpe.encrypt(plaintext)

        # No alphanumeric content, should be passthrough
        assert format_type == "passthrough"
        assert encrypted == plaintext

    def test_pure_special_chars_passthrough(self, fpe):
        """Test that pure special chars (no alphanumeric) is passthrough."""
        plaintext = "@#$%^&*()"
        encrypted, format_type = fpe.encrypt(plaintext)

        assert format_type == "passthrough"
        assert encrypted == plaintext


class TestTemplateExtraction:
    """Test the template extraction and application logic."""

    @pytest.fixture
    def fpe(self):
        salt = b"0" * 32
        key = b"1" * 32
        return FormatPreservingEncryption(salt, key)

    def test_extract_template_simple(self, fpe):
        """Test template extraction from simple formatted string."""
        content, template = fpe._extract_template("(555) 123-4567")

        assert content == "5551234567"
        assert template == [(0, "("), (4, ")"), (5, " "), (9, "-")]

    def test_extract_template_no_special_chars(self, fpe):
        """Test template extraction from pure alphanumeric."""
        content, template = fpe._extract_template("ABC123")

        assert content == "ABC123"
        assert template == []

    def test_apply_template_simple(self, fpe):
        """Test applying template to content."""
        template = [(0, "("), (4, ")"), (5, " "), (9, "-")]
        result = fpe._apply_template("8870246559", template)

        assert result == "(887) 024-6559"

    def test_apply_template_empty(self, fpe):
        """Test applying empty template."""
        result = fpe._apply_template("ABC123", [])

        assert result == "ABC123"
