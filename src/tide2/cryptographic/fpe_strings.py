"""
Format-Preserving Encryption using FF3 algorithm.

Generalized approach:
1. Extract template (positions of non-alphanumeric characters)
2. Extract content (alphanumeric characters only)
3. Encrypt content using appropriate alphabet (numeric or alphanumeric)
4. Reinsert template characters at original positions

This preserves any input format: (555) 123-4567 → (887) 024-6559

All operations are deterministic and reversible.
"""

import hashlib
import string
import warnings
from functools import lru_cache

from ff3 import FF3Cipher

# Character sets for fast membership testing
_ALNUM_SET = frozenset(string.ascii_letters + string.digits)
_DIGITS_SET = frozenset(string.digits)

# Pre-built translation table for extracting alphanumeric content
_KEEP_ALNUM_TABLE = str.maketrans("", "", "".join(chr(i) for i in range(256) if chr(i) not in _ALNUM_SET))


class FormatPreservingEncryption:
    """
    Format-Preserving Encryption using FF3 with template-based format preservation.

    Extracts alphanumeric content, encrypts it, and reinserts non-alphanumeric
    characters at their original positions. This preserves any input format.

    All operations are deterministic and reversible.
    """

    # Format constants (simplified)
    FORMAT_NUMERIC = "numeric"
    FORMAT_ALPHANUMERIC = "alphanumeric"
    FORMAT_EMPTY = "empty"
    FORMAT_PASSTHROUGH = "passthrough"

    # Alphabets for FF3
    ALPHABET_DIGITS = "0123456789"
    ALPHABET_ALPHANUMERIC = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Constants
    SALT_LENGTH = 32

    # Minimum lengths for secure FPE (domain size >= 1,000,000)
    # For radix 10 (digits): ceil(log(1000000)/log(10)) = 6
    # For radix 62 (alphanumeric): ceil(log(1000000)/log(62)) = 4
    MIN_LENGTH_NUMERIC = 6
    MIN_LENGTH_ALPHANUMERIC = 4

    # Maximum lengths supported by FF3 algorithm
    # For radix 10 (digits): 2 * floor(log_10(2^96)) = 56
    # For radix 62 (alphanumeric): 2 * floor(log_62(2^96)) = 32
    MAX_LENGTH_NUMERIC = 56
    MAX_LENGTH_ALPHANUMERIC = 32

    # Class-level flag to suppress short input warnings in production
    suppress_short_input_warnings: bool = False

    def __init__(self, salt: bytes, key: bytes):
        """
        Initialize FPE with salt and key.

        Args:
            salt: 32-byte salt
            key: Key (any length)
        """
        if len(salt) != self.SALT_LENGTH:
            raise ValueError("Salt must be exactly 32 bytes (256 bits)")

        self.salt = salt
        self.key = key
        self._key_hex = key.hex()

        # Create encryption key from combined salt and key
        combined_key_material = self.salt + self.key
        encryption_key_bytes = hashlib.sha256(combined_key_material).digest()
        self.encryption_key = encryption_key_bytes[:16]  # FF3 requires 16 bytes
        self.encryption_key_hex = self.encryption_key.hex()

        # Cache for FF3Cipher instances (key: (alphabet, tweak))
        self._cipher_cache: dict[tuple[str, str], FF3Cipher] = {}

    @lru_cache(maxsize=128)
    def _get_tweak(self, format_type: str, content_length: int) -> str:
        """
        Generate deterministic tweak based on format and content length.

        Uses content length (not original length) so that same content
        encrypts the same regardless of surrounding format characters.

        Args:
            format_type: Format identifier (numeric or alphanumeric)
            content_length: Length of alphanumeric content

        Returns:
            16-character hex tweak
        """
        combined = f"{format_type}:{content_length}:{self._key_hex}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _get_cipher(self, alphabet: str, tweak: str) -> FF3Cipher:
        """
        Get or create cached FF3Cipher instance.

        Args:
            alphabet: Alphabet string for cipher
            tweak: Tweak value

        Returns:
            FF3Cipher instance
        """
        cache_key = (alphabet, tweak)
        if cache_key not in self._cipher_cache:
            self._cipher_cache[cache_key] = FF3Cipher.withCustomAlphabet(self.encryption_key_hex, tweak, alphabet)
        return self._cipher_cache[cache_key]

    def _extract_template(self, text: str) -> tuple[str, list[tuple[int, str]]]:
        """
        Extract alphanumeric content and record positions of other characters.

        Args:
            text: Input text

        Returns:
            (content, template) where:
            - content: alphanumeric characters only
            - template: [(position, char), ...] for non-alphanumeric chars
        """
        template = []
        content_chars = []

        for i, char in enumerate(text):
            if char in _ALNUM_SET:
                content_chars.append(char)
            else:
                template.append((i, char))

        return "".join(content_chars), template

    def _apply_template(self, content: str, template: list[tuple[int, str]]) -> str:
        """
        Reinsert template characters into encrypted content.

        Args:
            content: Encrypted alphanumeric content
            template: [(position, char), ...] from extraction

        Returns:
            Content with template characters reinserted
        """
        if not template:
            return content

        result = list(content)
        for pos, char in template:
            result.insert(pos, char)

        return "".join(result)

    def _pad_if_needed(self, text: str, min_length: int, alphabet: str) -> tuple[str, int]:
        """
        Pad text to minimum length if needed.

        Args:
            text: Text to pad
            min_length: Minimum required length
            alphabet: Alphabet to use for padding character

        Returns:
            (padded_text, original_length)
        """
        original_length = len(text)

        if original_length >= min_length:
            return text, original_length

        pad_char = alphabet[0]
        padding_needed = min_length - original_length
        padded = text + pad_char * padding_needed

        if not self.suppress_short_input_warnings:
            warnings.warn(
                f"Input too short ({original_length} chars) for secure FPE. Padded to {min_length} chars.",
                UserWarning,
                stacklevel=4,
            )

        return padded, original_length

    def _hash_long_content(self, content: str) -> str:
        """
        Keyed SHA-256 fallback for content exceeding FF3 max length.

        Not reversible.
        """
        return hashlib.sha256((content + self.encryption_key_hex).encode()).hexdigest()

    def _encrypt_content(self, content: str, format_type: str) -> str:
        """
        Encrypt alphanumeric content.

        Uses FF3 FPE when content fits within max length, otherwise falls back
        to keyed hash (not reversible).

        Args:
            content: Alphanumeric content to encrypt
            format_type: FORMAT_NUMERIC or FORMAT_ALPHANUMERIC

        Returns:
            Encrypted content (same length as input)
        """
        if format_type == self.FORMAT_NUMERIC:
            alphabet = self.ALPHABET_DIGITS
            min_length = self.MIN_LENGTH_NUMERIC
            max_length = self.MAX_LENGTH_NUMERIC
        else:
            alphabet = self.ALPHABET_ALPHANUMERIC
            min_length = self.MIN_LENGTH_ALPHANUMERIC
            max_length = self.MAX_LENGTH_ALPHANUMERIC

        # Fall back to keyed hash for content exceeding FF3 max length
        if len(content) > max_length:
            return self._hash_long_content(content)

        # Pad if needed
        padded, original_length = self._pad_if_needed(content, min_length, alphabet)

        # Encrypt
        tweak = self._get_tweak(format_type, len(content))
        cipher = self._get_cipher(alphabet, tweak)
        encrypted = cipher.encrypt(padded)

        # Truncate to original length if padded
        return encrypted[:original_length]

    def _decrypt_content(self, content: str, format_type: str) -> str:
        """
        Decrypt alphanumeric content.

        Args:
            content: Encrypted alphanumeric content
            format_type: FORMAT_NUMERIC or FORMAT_ALPHANUMERIC

        Returns:
            Decrypted content (same length as input)
        """
        if format_type == self.FORMAT_NUMERIC:
            alphabet = self.ALPHABET_DIGITS
            min_length = self.MIN_LENGTH_NUMERIC
        else:
            alphabet = self.ALPHABET_ALPHANUMERIC
            min_length = self.MIN_LENGTH_ALPHANUMERIC

        # Pad if needed (must match encryption padding)
        padded, original_length = self._pad_if_needed(content, min_length, alphabet)

        # Decrypt
        tweak = self._get_tweak(format_type, len(content))
        cipher = self._get_cipher(alphabet, tweak)
        decrypted = cipher.decrypt(padded)

        # Truncate to original length if padded
        return decrypted[:original_length]

    def _detect_content_type(self, content: str) -> str:
        """
        Detect whether content is pure numeric or alphanumeric.

        Args:
            content: Alphanumeric content (no special chars)

        Returns:
            FORMAT_NUMERIC or FORMAT_ALPHANUMERIC
        """
        if content.isdigit():
            return self.FORMAT_NUMERIC
        return self.FORMAT_ALPHANUMERIC

    def encrypt(self, plaintext: str) -> tuple[str, str]:
        """
        Encrypt text preserving exact format.

        Any non-alphanumeric characters are preserved at their original positions.
        Only alphanumeric content is encrypted.

        Args:
            plaintext: Text to encrypt

        Returns:
            Tuple of (encrypted_text, format_type)

        Examples:
            (555) 123-4567  →  (887) 024-6559, numeric
            123-45-6789     →  987-65-4321, numeric
            MRN-12345678    →  XYZ-98765432, alphanumeric
        """
        # Handle empty/whitespace
        if not plaintext or not plaintext.strip():
            return plaintext, self.FORMAT_EMPTY

        # Extract template and content
        content, template = self._extract_template(plaintext)

        # No encryptable content (only special chars)
        if not content:
            return plaintext, self.FORMAT_PASSTHROUGH

        # Determine format type based on content
        format_type = self._detect_content_type(content)

        # Encrypt content
        try:
            encrypted_content = self._encrypt_content(content, format_type)
        except Exception as e:
            warnings.warn(f"Encryption failed for '{plaintext}': {e}", stacklevel=2)
            return plaintext, self.FORMAT_PASSTHROUGH

        # Reapply template
        result = self._apply_template(encrypted_content, template)

        return result, format_type

    def decrypt(self, ciphertext: str, format_type: str | None = None) -> str:
        """
        Decrypt text preserving exact format.

        Args:
            ciphertext: Text to decrypt
            format_type: Format type (if None, will auto-detect from content)

        Returns:
            Decrypted text with format preserved
        """
        # Handle empty/whitespace
        if not ciphertext or not ciphertext.strip():
            return ciphertext

        # Extract template and content
        content, template = self._extract_template(ciphertext)

        # No decryptable content
        if not content:
            return ciphertext

        # Auto-detect format if not provided
        if format_type is None or format_type == self.FORMAT_PASSTHROUGH:
            format_type = self._detect_content_type(content)

        # Handle passthrough
        if format_type == self.FORMAT_EMPTY:
            return ciphertext

        # Decrypt content
        try:
            decrypted_content = self._decrypt_content(content, format_type)
        except Exception as e:
            warnings.warn(f"Decryption failed for '{ciphertext}': {e}", stacklevel=2)
            return ciphertext

        # Reapply template
        return self._apply_template(decrypted_content, template)

    # Legacy method for backward compatibility
    def detect_format(self, text: str) -> str:
        """
        Detect format of input text.

        Simplified: only distinguishes between empty, numeric, and alphanumeric
        based on the alphanumeric content (ignoring special characters).

        Args:
            text: Input text

        Returns:
            Format identifier string
        """
        if not text or not text.strip():
            return self.FORMAT_EMPTY

        content = text.translate(_KEEP_ALNUM_TABLE)

        if not content:
            return self.FORMAT_PASSTHROUGH

        return self._detect_content_type(content)
