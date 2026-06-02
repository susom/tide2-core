"""
Name tokenizer for splitting and classifying name components.

Handles structural classification of tokens (salutations, suffixes, prefixes, initials).
"""

from dataclasses import dataclass
from enum import Enum
from enum import auto

from ..utils.constants import NAME_CONSTANTS


class TokenType(Enum):
    """Structural classification of a name token."""

    SALUTATION = auto()  # Dr., Mr., Prof., etc.
    SUFFIX = auto()  # Jr., MD, PhD, III, etc.
    SURNAME_PREFIX = auto()  # van, de, von, della, etc. (particles)
    INITIAL = auto()  # A., J., single letters
    NAME = auto()  # Regular name token


@dataclass(slots=True)
class NameToken:
    """A tokenized name component with structural classification."""

    text: str
    token_type: TokenType
    trailing_sep: str = " "  # Separator after this token (space, comma-space, etc.)

    def __repr__(self) -> str:
        return f"NameToken({self.text!r}, {self.token_type.name})"


class NameTokenizer:
    """
    Tokenize name strings and classify structural components.

    Identifies:
    - Salutations (Dr., Mr., Prof.)
    - Suffixes (Jr., MD, PhD, III)
    - Surname prefixes/particles (van, de, von, della)
    - Initials (A., J.)
    - Regular name tokens
    """

    __slots__ = ("salutations", "suffixes", "surname_prefixes")

    def __init__(self):
        """Initialize with constants from NAME_CONSTANTS."""
        self.salutations = NAME_CONSTANTS.salutations
        self.suffixes = NAME_CONSTANTS.suffixes
        self.surname_prefixes = NAME_CONSTANTS.surname_prefixes

    def tokenize(self, text: str) -> tuple[list[NameToken], str, bool]:
        """
        Tokenize a name string into classified components.

        Args:
            text: The name string to tokenize

        Returns:
            Tuple of:
            - List of NameToken objects
            - Case format detected ("upper", "lower", "title")
            - Whether the name uses comma format (e.g., "Smith, John")
        """
        if not text or not text.strip():
            return [], "title", False

        text = text.strip()

        # Detect case format
        case_format = self._detect_case_format(text)

        # Detect comma format
        has_comma = "," in text

        # Tokenize
        tokens = self._tokenize_string(text)

        return tokens, case_format, has_comma

    def _detect_case_format(self, text: str) -> str:
        """Detect the case format of the input text."""
        # Only consider alphabetic characters
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return "title"

        if all(c.isupper() for c in alpha_chars):
            return "upper"
        if all(c.islower() for c in alpha_chars):
            return "lower"
        return "title"

    def _tokenize_string(self, text: str) -> list[NameToken]:
        """Split string into tokens, preserving comma positions."""
        tokens = []

        # Track comma positions for proper separator handling
        # Replace commas with " , " to make them separate tokens, then process
        parts = text.replace(",", " , ").split()

        i = 0
        while i < len(parts):
            part = parts[i]

            # Handle comma as separator marker
            if part == ",":
                # Attach comma to previous token's trailing separator
                if tokens:
                    tokens[-1].trailing_sep = ", "
                i += 1
                continue

            # Strip trailing possessive/apostrophe from the token text
            # e.g., "Harrison's" → "Harrison" with trailing_punct = "'s"
            #        "Harrison'" → "Harrison" with trailing_punct = "'"
            part, trailing_punct = self._strip_trailing_punct(part)

            # Classify the token
            token_type = self._classify_token(part, i, len(parts), tokens)

            # Determine trailing separator (default is space)
            # Prepend any stripped punctuation so it's preserved in output
            trailing_sep = trailing_punct + " "

            tokens.append(NameToken(part, token_type, trailing_sep))
            i += 1

        # Last token: remove trailing space but preserve any stripped punctuation
        if tokens:
            tokens[-1].trailing_sep = tokens[-1].trailing_sep.rstrip(" ")

        return tokens

    def _classify_token(self, token: str, position: int, total: int, preceding_tokens: list[NameToken]) -> TokenType:
        """
        Classify a token based on content and position.

        Args:
            token: The token text
            position: Position in the original token list
            total: Total number of tokens
            preceding_tokens: Tokens already classified (for context)
        """
        clean = token.lower().rstrip(".")

        # Check for salutation (typically at the start)
        if self._is_salutation_position(position, preceding_tokens):
            if clean in self.salutations:
                return TokenType.SALUTATION

        # Check for suffix
        # Suffixes can appear at the end, or after a comma
        if clean in self.suffixes:
            return TokenType.SUFFIX

        # Check for surname prefix (particle)
        # These are words like "van", "de", "von" that are part of surnames
        if clean in self.surname_prefixes:
            return TokenType.SURNAME_PREFIX

        # Check for initial (single letter with optional period)
        if self._is_initial(token):
            return TokenType.INITIAL

        return TokenType.NAME

    def _is_salutation_position(self, position: int, preceding_tokens: list[NameToken]) -> bool:
        """Check if this position could be a salutation."""
        if position == 0:
            return True
        # Salutation could also follow another salutation (rare but possible)
        if preceding_tokens and preceding_tokens[-1].token_type == TokenType.SALUTATION:
            return True
        return False

    def _strip_trailing_punct(self, token: str) -> tuple[str, str]:
        """
        Strip trailing possessive markers and apostrophes from a token.

        Handles: "Harrison's" → ("Harrison", "'s")
                 "Harrison'"  → ("Harrison", "'")
                 "Harrison"   → ("Harrison", "")

        Returns:
            Tuple of (cleaned_token, stripped_punctuation)
        """
        # Check for possessive 's or trailing apostrophe
        if len(token) > 2 and token.endswith("'s"):
            return token[:-2], "'s"
        if len(token) > 1 and token.endswith("'"):
            return token[:-1], "'"
        return token, ""

    def _is_initial(self, token: str) -> bool:
        """Check if token is an initial (single letter with optional period)."""
        cleaned = token.rstrip(".")
        return len(cleaned) == 1 and cleaned.isalpha()

    def tokens_to_string(self, tokens: list[NameToken]) -> str:
        """Reconstruct a string from tokens using their trailing separators."""
        if not tokens:
            return ""

        parts = []
        for token in tokens:
            parts.append(token.text)
            parts.append(token.trailing_sep)

        return "".join(parts).rstrip()
