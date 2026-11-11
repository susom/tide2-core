"""
Collision-free format-preserving encryption on NIST FF1.

Generalized approach:
1. Extract template (positions of non-alphanumeric characters)
2. Extract content (alphanumeric characters only)
3. Permute the content inside its class (see below), keeping its length
4. Reinsert template characters at original positions

The pseudonym depends only on the key, the content, its class and the template.
It does not depend on the entity type or on the storage type, so the same ID
gets the same pseudonym as a STRING and as an INT64.

Classes (each (class, length) map is a permutation, so the mapping is one-to-one):
    N: digits without a leading zero. The output never starts with 0, so int() is safe.
       At 19 digits, values up to 2**63 - 1 and above 2**63 are separate classes.
    M: exactly "9223372036854775808" (2**63). It maps to itself, so "-9223372036854775808"
       (the minimum INT64 value) stays in range, and no other value maps onto it.
    Z: exactly "0". It maps to itself, so a negative value never becomes "-0".
    S: digit string of two or more characters that starts with 0.
    A: content that contains a letter. The output contains a letter.

Content shorter than FF1's minimum domain (10**6) is permuted with keyed tables.
Longer content uses FF1 with cycle-walking back into the input's class.

Relaxed mode (``mode="relaxed"``) is a faster path for INT64 values. A value that is an INT64
(a Python ``int`` or a digit string such as ``"123"`` or ``"-5"``, no leading zero) goes through a keyed
Feistel permutation of the INT64 range built on FarmHash. Its output is an INT64 with no collisions: a
non-negative input gives a non-negative output and a negative input a negative one, but zero is not a
fixed point (``0`` maps to another value, and another value maps to ``0``), so callers must not treat
``0`` as a sentinel in relaxed mode. The digit count is not kept. Everything else takes the FF1 path above.
Relaxed mode is keyed pseudonymization, not NIST-approved encryption: the permutation is exact, but
its secrecy rests on FarmHash acting as a keyed PRF, for which there is no proof. Use strict mode (FF1)
when an attacker may hold many known (input, output) pairs.

All operations are deterministic and reversible.
"""

import hashlib
import hmac
import itertools
import re
import string
import threading
import warnings

from farmhash import Fingerprint128  # ty: ignore[unresolved-import]
from fastfpe import ff1

# Character sets for fast membership testing
_ALNUM_SET = frozenset(string.ascii_letters + string.digits)

# Pre-built translation table for extracting alphanumeric content
_KEEP_ALNUM_TABLE = str.maketrans("", "", "".join(chr(i) for i in range(256) if chr(i) not in _ALNUM_SET))

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
_INT64_MAX_DIGITS = 19

_CLASS_NUMERIC = "N"
_CLASS_NUMERIC_ABOVE_INT64 = "H"
_CLASS_INT64_MIN_MAGNITUDE = "M"
_CLASS_ZERO = "Z"
_CLASS_ZERO_LED = "S"
_CLASS_ALPHANUMERIC = "A"

_KEY_LABEL = b"tide2-fpe-ff1-v1"
_SMALL_DOMAIN_LABEL = b"tide2-fpe-small-domain-v1"
_TWEAK_LABEL = "tide2-fpe-v2:"
_RELAXED_LABEL = b"tide2-fpe-relaxed-int64-v1"

MODE_STRICT = "strict"
MODE_RELAXED = "relaxed"
FPE_MODES = (MODE_STRICT, MODE_RELAXED)

_RELAXED_ROUNDS = 8
_INT64_MAGNITUDE_BITS = 63
# No leading zero and no "-0", so that str(int(text)) == text.
_INT64_TEXT = re.compile(r"0|-?[1-9][0-9]*")
# One past the longest INT64 string ("-9223372036854775808").
_INT64_TEXT_MAX_LENGTH = 20


def _feistel(x: int, prefixes: tuple[bytes, ...], width: int = _INT64_MAGNITUDE_BITS, decrypt: bool = False) -> int:
    """Apply the keyed Feistel permutation of ``[0, 2**width)`` to ``x``, or its inverse.

    The halves have ``width // 2`` and ``width - width // 2`` bits and swap sizes every round.
    The round function is FarmHash of a per-round key prefix and the other half.
    """
    low_bits = width // 2
    high_bits = width - low_bits
    masks = ((1 << low_bits) - 1, (1 << high_bits) - 1)
    a, b = x >> high_bits, x & masks[1]
    if decrypt:
        for i in range(len(prefixes) - 1, -1, -1):
            a, b = b ^ (Fingerprint128(prefixes[i] + a.to_bytes(4, "little")) & masks[i & 1]), a
    else:
        for i, prefix in enumerate(prefixes):
            a, b = b, a ^ (Fingerprint128(prefix + b.to_bytes(4, "little")) & masks[i & 1])
    return (a << high_bits) | b


class FormatPreservingEncryption:
    """Collision-free FF1 format-preserving encryption with template-based format preservation.

    Extracts alphanumeric content, encrypts it inside its class, and reinserts
    non-alphanumeric characters at their original positions. This preserves any
    input format. The mapping is one-to-one, deterministic and reversible, and it
    has no length limit.
    """

    # Format constants (simplified)
    FORMAT_NUMERIC = "numeric"
    FORMAT_ALPHANUMERIC = "alphanumeric"
    FORMAT_EMPTY = "empty"
    FORMAT_PASSTHROUGH = "passthrough"
    FORMAT_INT64 = "int64"

    # Alphabets for FF1
    ALPHABET_DIGITS = "0123456789"
    ALPHABET_ALPHANUMERIC = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Constants
    SALT_LENGTH = 32

    # Minimum lengths for FF1 (domain size >= 1,000,000); shorter content uses keyed tables
    # For radix 10 (digits): ceil(log(1000000)/log(10)) = 6
    # For radix 62 (alphanumeric): ceil(log(1000000)/log(62)) = 4
    MIN_LENGTH_NUMERIC = 6
    MIN_LENGTH_ALPHANUMERIC = 4

    # Cycle-walking cap; unreachable in practice
    MAX_WALK_STEPS = 10_000

    def __init__(self, salt: bytes, key: bytes, mode: str = MODE_STRICT):
        """Initialize FPE with salt and key.

        Args:
            salt: 32-byte salt.
            key: Key (any length).
            mode: ``"strict"`` encrypts every value with FF1. ``"relaxed"`` sends INT64 values
                through a faster permutation of the INT64 range and the rest through FF1.

        Raises:
            ValueError: If the salt is not exactly 32 bytes or the mode is unknown.
        """
        if len(salt) != self.SALT_LENGTH:
            raise ValueError("Salt must be exactly 32 bytes (256 bits)")
        if mode not in FPE_MODES:
            raise ValueError(f"Unknown FPE mode {mode!r}; valid modes: {list(FPE_MODES)}")

        self.salt = salt
        self.key = key
        self.mode = mode

        # AES-256 key; the label separates it from any other use of the same salt and key
        self.encryption_key = hashlib.sha256(_KEY_LABEL + salt + key).digest()
        self._ff1_key_hex = self.encryption_key.hex()
        self._small_domain_key = hmac.digest(self.encryption_key, _SMALL_DOMAIN_LABEL, "sha256")
        self._tweaks = {
            fmt: hashlib.sha256(f"{_TWEAK_LABEL}{fmt}".encode()).digest()[:8].hex()
            for fmt in (self.FORMAT_NUMERIC, self.FORMAT_ALPHANUMERIC)
        }

        self._tables: dict[tuple[str, int], tuple[dict[str, str], dict[str, str]]] = {}
        self._tables_lock = threading.Lock()

        # Index 0 is for values >= 0 and 1 for values < 0; the sign tag makes the two halves independent.
        self._relaxed_prefixes: tuple[tuple[bytes, ...], tuple[bytes, ...]] = ((), ())
        if mode == MODE_RELAXED:
            master = hashlib.sha256(_RELAXED_LABEL + salt + key).digest()
            round_keys = [hmac.digest(master, f"round:{i}".encode(), "sha256") for i in range(_RELAXED_ROUNDS)]
            self._relaxed_prefixes = (
                tuple(k + b"\x00" for k in round_keys),
                tuple(k + b"\x01" for k in round_keys),
            )

    def _relaxed_int64(self, value: int, decrypt: bool = False, width: int = _INT64_MAGNITUDE_BITS) -> int:
        """Permute ``value`` inside ``[-2**width, 2**width)``, keeping its sign.

        A negative ``v`` is mapped to ``-1 - v`` in ``[0, 2**width)``, so each sign half has its own
        permutation and the minimum INT64 value needs no special case.
        """
        if value >= 0:
            return _feistel(value, self._relaxed_prefixes[0], width, decrypt)
        return -1 - _feistel(-1 - value, self._relaxed_prefixes[1], width, decrypt)

    def _is_relaxed(self, text: str) -> bool:
        """Return True if ``text`` is an INT64 value that relaxed mode permutes directly."""
        return (
            self.mode == MODE_RELAXED
            and len(text) <= _INT64_TEXT_MAX_LENGTH
            and _INT64_TEXT.fullmatch(text) is not None
            and INT64_MIN <= int(text) <= INT64_MAX
        )

    def _extract_template(self, text: str) -> tuple[str, list[tuple[int, str]]]:
        """Extract alphanumeric content and record positions of other characters.

        Args:
            text: Input text.

        Returns:
            (content, template) where content is the alphanumeric characters only
            and template is [(position, char), ...] for the other characters.
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
        """Reinsert template characters into encrypted content.

        Args:
            content: Encrypted alphanumeric content.
            template: [(position, char), ...] from extraction.

        Returns:
            Content with template characters reinserted.
        """
        if not template:
            return content

        result = list(content)
        for pos, char in template:
            result.insert(pos, char)

        return "".join(result)

    @staticmethod
    def _classify(content: str) -> str:
        """Return the class of non-empty alphanumeric content."""
        if not content.isdigit():
            return _CLASS_ALPHANUMERIC
        if content == "0":
            return _CLASS_ZERO
        if content[0] == "0":
            return _CLASS_ZERO_LED
        if len(content) == _INT64_MAX_DIGITS and int(content) > INT64_MAX:
            return _CLASS_INT64_MIN_MAGNITUDE if int(content) == -INT64_MIN else _CLASS_NUMERIC_ABOVE_INT64
        return _CLASS_NUMERIC

    def _build_table(self, content_class: str, length: int) -> tuple[dict[str, str], dict[str, str]]:
        """Build the keyed permutation of one (class, length) set as forward and inverse dicts."""
        alphabet = self.ALPHABET_ALPHANUMERIC if content_class == _CLASS_ALPHANUMERIC else self.ALPHABET_DIGITS
        members = [
            s for s in map("".join, itertools.product(alphabet, repeat=length)) if self._classify(s) == content_class
        ]
        prefix = f"{content_class}:{length}:"
        shuffled = sorted(members, key=lambda s: hmac.digest(self._small_domain_key, f"{prefix}{s}".encode(), "sha256"))
        return dict(zip(members, shuffled, strict=True)), dict(zip(shuffled, members, strict=True))

    def _get_table(self, content_class: str, length: int) -> tuple[dict[str, str], dict[str, str]]:
        """Return the (forward, inverse) table for a class and length, building it on first use."""
        cache_key = (content_class, length)
        table = self._tables.get(cache_key)
        if table is None:
            with self._tables_lock:
                table = self._tables.get(cache_key)
                if table is None:
                    table = self._build_table(content_class, length)
                    self._tables[cache_key] = table
        return table

    def _walk(self, content: str, content_class: str, format_type: str, decrypt: bool) -> str:
        """Apply FF1 until the result is back in the input's class (cycle-walking)."""
        alphabet = self.ALPHABET_DIGITS if format_type == self.FORMAT_NUMERIC else self.ALPHABET_ALPHANUMERIC
        tweak = self._tweaks[format_type]
        cipher = ff1.decrypt if decrypt else ff1.encrypt
        value = content
        for _ in range(self.MAX_WALK_STEPS):
            value = cipher(self._ff1_key_hex, tweak, alphabet, value)
            if self._classify(value) == content_class:
                return value
        raise RuntimeError(f"FF1 cycle-walking did not terminate within {self.MAX_WALK_STEPS} steps")

    def _permute_content(self, content: str, format_type: str, decrypt: bool) -> str:
        """Encrypt or decrypt alphanumeric content inside its class, keeping its length."""
        content_class = self._classify(content)
        if content_class in (_CLASS_ZERO, _CLASS_INT64_MIN_MAGNITUDE):
            return content

        min_length = self.MIN_LENGTH_NUMERIC if format_type == self.FORMAT_NUMERIC else self.MIN_LENGTH_ALPHANUMERIC
        if len(content) < min_length:
            forward, inverse = self._get_table(content_class, len(content))
            return (inverse if decrypt else forward)[content]

        return self._walk(content, content_class, format_type, decrypt)

    def _encrypt_content(self, content: str, format_type: str) -> str:
        """Encrypt alphanumeric content (same class and length)."""
        return self._permute_content(content, format_type, decrypt=False)

    def _decrypt_content(self, content: str, format_type: str) -> str:
        """Decrypt alphanumeric content (same class and length)."""
        return self._permute_content(content, format_type, decrypt=True)

    def _detect_content_type(self, content: str) -> str:
        """Detect whether content is pure numeric or alphanumeric.

        Args:
            content: Alphanumeric content (no special chars).

        Returns:
            FORMAT_NUMERIC or FORMAT_ALPHANUMERIC.
        """
        if content.isdigit():
            return self.FORMAT_NUMERIC
        return self.FORMAT_ALPHANUMERIC

    def encrypt(self, plaintext: str) -> tuple[str, str]:
        """Encrypt text preserving exact format.

        Any non-alphanumeric characters are preserved at their original positions.
        Only alphanumeric content is encrypted, and it keeps its class and length.
        In relaxed mode an INT64 string is permuted inside the INT64 range instead; it keeps
        its sign but not its length.

        Args:
            plaintext: Text to encrypt.

        Returns:
            Tuple of (encrypted_text, format_type).

        Raises:
            RuntimeError: If cycle-walking exceeds MAX_WALK_STEPS (unreachable in practice).

        Examples:
            (555) 123-4567  ->  digits in the same positions, no leading zero, same format
            MRN-12345678    ->  letters and digits in the same positions, at least one letter
        """
        # Handle empty/whitespace
        if not plaintext or not plaintext.strip():
            return plaintext, self.FORMAT_EMPTY

        if self._is_relaxed(plaintext):
            return str(self._relaxed_int64(int(plaintext))), self.FORMAT_INT64

        # Extract template and content
        content, template = self._extract_template(plaintext)

        # No encryptable content (only special chars)
        if not content:
            return plaintext, self.FORMAT_PASSTHROUGH

        format_type = self._detect_content_type(content)

        # Errors propagate: returning the plaintext would leak it.
        encrypted_content = self._encrypt_content(content, format_type)

        return self._apply_template(encrypted_content, template), format_type

    def decrypt(self, ciphertext: str, format_type: str | None = None) -> str:
        """Decrypt text preserving exact format.

        Args:
            ciphertext: Text to decrypt.
            format_type: Format type returned by encrypt; only FORMAT_EMPTY changes the
                result, because the class is read from the ciphertext itself.

        Returns:
            Decrypted text with format preserved.
        """
        # Handle empty/whitespace
        if not ciphertext or not ciphertext.strip():
            return ciphertext

        # A relaxed output is an INT64 string, and no FF1 output is one.
        if format_type != self.FORMAT_EMPTY and self._is_relaxed(ciphertext):
            return str(self._relaxed_int64(int(ciphertext), decrypt=True))

        # Extract template and content
        content, template = self._extract_template(ciphertext)

        # No decryptable content
        if not content:
            return ciphertext

        if format_type == self.FORMAT_EMPTY:
            return ciphertext

        try:
            decrypted_content = self._decrypt_content(content, self._detect_content_type(content))
        except Exception as e:
            warnings.warn(f"Decryption failed for '{ciphertext}': {e}", stacklevel=2)
            return ciphertext

        return self._apply_template(decrypted_content, template)

    @staticmethod
    def _check_int64(value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"value must be an int, got {type(value).__name__}")
        if not INT64_MIN <= value <= INT64_MAX:
            raise ValueError("value is outside the INT64 range")

    def encrypt_int(self, value: int) -> int:
        """Encrypt an INT64 value; the result equals int(encrypt(str(value))) and stays in INT64.

        The sign is kept (non-negative stays non-negative, negative stays negative). In strict
        mode zero is a fixed point and the minimum INT64 value (-2**63) maps to itself, because
        the string mapping keeps the magnitude 2**63 fixed; in relaxed mode neither is a fixed
        point, and a non-zero value can map to zero.

        Args:
            value: Integer in the INT64 range.

        Returns:
            The pseudonymized integer.

        Raises:
            TypeError: If value is not an int.
            ValueError: If value is outside the INT64 range.
        """
        self._check_int64(value)
        if self.mode == MODE_RELAXED:
            return self._relaxed_int64(value)
        return int(self.encrypt(str(value))[0])

    def decrypt_int(self, value: int) -> int:
        """Invert encrypt_int.

        Args:
            value: Integer in the INT64 range.

        Returns:
            The original integer.

        Raises:
            TypeError: If value is not an int.
            ValueError: If value is outside the INT64 range.
        """
        self._check_int64(value)
        if self.mode == MODE_RELAXED:
            return self._relaxed_int64(value, decrypt=True)
        return int(self.decrypt(str(value)))

    # Legacy method for backward compatibility
    def detect_format(self, text: str) -> str:
        """Detect format of input text.

        Simplified: only distinguishes between empty, numeric, and alphanumeric
        based on the alphanumeric content (ignoring special characters).

        Args:
            text: Input text.

        Returns:
            Format identifier string.
        """
        if not text or not text.strip():
            return self.FORMAT_EMPTY

        content = text.translate(_KEEP_ALNUM_TABLE)

        if not content:
            return self.FORMAT_PASSTHROUGH

        return self._detect_content_type(content)
