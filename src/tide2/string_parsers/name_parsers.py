"""
Name parser for parsing and formatting human names.

Tokenizes names into structural components (salutations, suffixes, initials,
surname prefixes, name tokens) and formats them back. Classification into
first name vs surname is not performed — all name tokens are treated uniformly
to ensure consistent anonymization replacements.
"""

from dataclasses import dataclass
from enum import Enum

from .name_tokenizer import NameToken
from .name_tokenizer import NameTokenizer
from .name_tokenizer import TokenType


class NameType(Enum):
    """Classification of a name token."""

    FIRST_NAME = "first_name"
    SURNAME = "surname"
    AMBIGUOUS = "ambiguous"


@dataclass(slots=True)
class ParsedToken:
    """A token with both structural and semantic classification."""

    text: str
    token_type: TokenType  # Structural: SALUTATION, SUFFIX, SURNAME_PREFIX, INITIAL, NAME
    name_type: NameType | None  # Semantic: FIRST_NAME, SURNAME, AMBIGUOUS (only for name tokens)
    confidence: float
    trailing_sep: str = " "

    def __repr__(self) -> str:
        name_type_str = self.name_type.name if self.name_type else "None"
        return f"ParsedToken({self.text!r}, {self.token_type.name}, {name_type_str}, {self.confidence:.2f})"


class NameParser:
    """
    Parse names into token sequences for anonymization.

    Handles structural parsing (salutations, suffixes, prefixes, initials)
    without classifying tokens as first name vs surname. All name tokens
    are assigned NameType.AMBIGUOUS so the anonymizer uses a single
    unified replacement list for consistency.
    """

    __slots__ = ("tokenizer",)

    def __init__(self):
        """Initialize parser with tokenizer only."""
        self.tokenizer = NameTokenizer()

    def parse(self, name_string: str, entity_type: str | None = None) -> dict:
        """
        Parse a name string into tokens.

        Args:
            name_string: The name to parse
            entity_type: Optional hint (DOCTOR, PATIENT, PERSON, HCW) — unused,
                kept for API compatibility

        Returns:
            Dictionary with:
            - tokens: list[ParsedToken] - name tokens
            - salutation: str | None - extracted salutation
            - suffix: str | None - extracted suffix(es)
            - _case_format: str - detected case format
            - _is_comma: bool - whether comma format was used
        """
        if not name_string or not name_string.strip():
            return {}

        # Tokenize
        tokens, case_format, is_comma = self.tokenizer.tokenize(name_string)
        if not tokens:
            return {}

        result: dict = {
            "tokens": [],
            "salutation": None,
            "suffix": None,
            "_case_format": case_format,
            "_is_comma": is_comma,
        }

        # Extract salutation(s) from the beginning
        name_start_idx = 0
        salutation_parts = []
        for i, token in enumerate(tokens):
            if token.token_type == TokenType.SALUTATION:
                salutation_parts.append(token.text)
                name_start_idx = i + 1
            else:
                break

        if salutation_parts:
            result["salutation"] = " ".join(salutation_parts)

        # Extract suffix(es) from the end
        suffix_parts = []
        name_end_idx = len(tokens)
        for i in range(len(tokens) - 1, name_start_idx - 1, -1):
            if tokens[i].token_type == TokenType.SUFFIX:
                suffix_parts.insert(0, tokens[i].text)
                name_end_idx = i
            else:
                break

        if suffix_parts:
            result["suffix"] = " ".join(suffix_parts)
            # Clear comma from the last name token if suffix was extracted
            # This handles "John Smith, MD" where comma is before suffix, not name format
            if name_end_idx > name_start_idx:
                last_name_token = tokens[name_end_idx - 1]
                if ", " in last_name_token.trailing_sep:
                    last_name_token.trailing_sep = " "

        # Get remaining tokens (the actual name parts)
        name_tokens = tokens[name_start_idx:name_end_idx]

        # Check if there's actually a comma between name tokens
        # (not just before suffix which was already extracted)
        has_comma_in_name = any(", " in t.trailing_sep for t in name_tokens)

        # Convert to ParsedTokens
        parsed_tokens = self._to_parsed_tokens(name_tokens, has_comma_in_name)

        # Update _is_comma to reflect actual comma in name tokens
        result["_is_comma"] = has_comma_in_name

        result["tokens"] = parsed_tokens
        return result

    def _to_parsed_tokens(self, tokens: list[NameToken], is_comma: bool) -> list[ParsedToken]:
        """
        Convert NameTokens to ParsedTokens.

        All NAME tokens get NameType.AMBIGUOUS since we use a unified
        replacement list. Structural types (SURNAME_PREFIX, INITIAL)
        retain their semantic meaning for formatting purposes.
        """
        if not tokens:
            return []

        parsed_tokens = []

        for token in tokens:
            if token.token_type == TokenType.SURNAME_PREFIX:
                parsed = ParsedToken(
                    text=token.text,
                    token_type=token.token_type,
                    name_type=NameType.SURNAME,
                    confidence=1.0,
                    trailing_sep=token.trailing_sep,
                )
            elif token.token_type == TokenType.INITIAL:
                parsed = ParsedToken(
                    text=token.text,
                    token_type=token.token_type,
                    name_type=NameType.AMBIGUOUS,
                    confidence=0.70,
                    trailing_sep=token.trailing_sep,
                )
            else:
                # Regular NAME token — no classification needed
                parsed = ParsedToken(
                    text=token.text,
                    token_type=token.token_type,
                    name_type=NameType.AMBIGUOUS,
                    confidence=1.0,
                    trailing_sep=token.trailing_sep,
                )

            parsed_tokens.append(parsed)

        return parsed_tokens

    def format(self, parsed: dict) -> str:
        """
        Reconstruct a name string from parsed components.

        Args:
            parsed: Dictionary from parse() method

        Returns:
            Formatted name string with original casing restored
        """
        if not parsed:
            return ""

        parts = []

        # Add salutation
        if parsed.get("salutation"):
            parts.append(parsed["salutation"])
            parts.append(" ")

        # Add name tokens
        tokens: list[ParsedToken] = parsed.get("tokens", [])

        if parsed.get("_is_comma") and tokens:
            # Comma format: tokens before comma, then after
            surname_tokens = []
            firstname_tokens = []

            found_comma = False
            for token in tokens:
                if not found_comma:
                    surname_tokens.append(token)
                    if ", " in token.trailing_sep:
                        found_comma = True
                else:
                    firstname_tokens.append(token)

            # Build surname part
            surname_parts = []
            for token in surname_tokens:
                surname_parts.append(token.text)
            if surname_parts:
                parts.append(" ".join(surname_parts))

            # Add comma
            parts.append(", ")

            # Build first name part
            firstname_parts = []
            for token in firstname_tokens:
                firstname_parts.append(token.text)
            if firstname_parts:
                parts.append(" ".join(firstname_parts))
        else:
            # Standard format: just join all tokens
            for token in tokens:
                parts.append(token.text)
                if token.trailing_sep:
                    parts.append(token.trailing_sep)

        # Build name part and apply case format
        result = "".join(parts).rstrip()
        case_format = parsed.get("_case_format", "title")
        result = self._apply_case(result, case_format)

        # Add suffix AFTER case transformation to preserve original credential casing
        if parsed.get("suffix"):
            if not result.endswith(","):
                result += ", "
            else:
                result += " "
            result += parsed["suffix"]

        return result

    def _apply_case(self, text: str, case_format: str) -> str:
        """Apply the detected case format to the output text."""
        if case_format == "upper":
            return text.upper()
        if case_format == "lower":
            return text.lower()
        # Title case with intelligent handling
        return self._apply_title_case(text)

    def _apply_title_case(self, text: str) -> str:
        """
        Apply title case with special handling for name particles.

        Keeps particles like "van", "de", "von" lowercase.
        """
        if not text:
            return text

        words = text.split(" ")
        result = []

        for i, word in enumerate(words):
            if not word:
                result.append(word)
                continue

            # Check if it's a particle that should stay lowercase
            # (unless it's the first word)
            clean_word = word.lower().rstrip(".,")
            if i > 0 and clean_word in self.tokenizer.surname_prefixes:
                result.append(word.lower())
            # Capitalize first letter, keep rest of each hyphenated part
            elif "-" in word:
                # Handle hyphenated names: "Mary-Jane" -> "Mary-Jane"
                parts = word.split("-")
                capitalized = "-".join(p.capitalize() for p in parts)
                result.append(capitalized)
            else:
                result.append(word.capitalize())

        return " ".join(result)
