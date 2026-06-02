"""
HIPS-compliant name anonymizer.

Anonymizes names by replacing each token with a substitute from a single
unified name list, ensuring the same input always maps to the same
replacement regardless of context.
"""

import string
from pathlib import Path
from typing import ClassVar

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType

from tide2.cryptographic.string_selector import secure_string_selector
from tide2.string_parsers.name_parsers import NameParser
from tide2.string_parsers.name_parsers import ParsedToken
from tide2.string_parsers.name_tokenizer import TokenType
from tide2.utils.resource_utils import UNIFIED_NAMES_FILE
from tide2.utils.resource_utils import get_resource_path
from tide2.utils.resource_utils import load_stopwords


class HipsNamesAnonymizer(Operator):
    """
    Anonymizer that replaces names with fake names consistently.

    Uses a single unified name list for all token types so that the same
    input name always produces the same replacement, regardless of whether
    it appears as a first name or surname.
    """

    # Class-level cache to avoid reloading files
    _unified_names_list: ClassVar[list | None] = None
    _initials_list: ClassVar[list | None] = None
    _stopwords: ClassVar[frozenset[str] | None] = None
    _name_parser: ClassVar[NameParser | None] = None

    @classmethod
    def _load_names_data(cls):
        """Load names data from unified names file (only once per class)."""
        if cls._unified_names_list is None:
            unified_path = Path(get_resource_path(UNIFIED_NAMES_FILE))
            with unified_path.open(encoding="utf-8") as f:
                cls._unified_names_list = [line.strip().lower() for line in f if line.strip()]

        if cls._initials_list is None:
            cls._initials_list = list(string.ascii_uppercase)

        if cls._stopwords is None:
            cls._stopwords = load_stopwords()

        if cls._name_parser is None:
            cls._name_parser = NameParser()

    def __init__(self):
        """Initialize the HIPS names anonymizer with cached name lists."""
        super().__init__()
        # Load data once at class level
        self._load_names_data()

        # Use class-level cached data
        self.unified_names_list = self._unified_names_list
        self.initials_list = self._initials_list
        self.stopwords = self._stopwords
        self.name_parser = self._name_parser

        self.supported_entity_types = ["PERSON", "DOCTOR", "PATIENT", "HCW"]

    def is_spurious_value(self, text: str) -> bool:
        """
        Check if the text is spurious and should not be anonymized.

        A text is considered spurious if it is:
        - Empty or only whitespace
        - Single character (after stripping whitespace)
        - Single letter with punctuation (e.g., "A.", "B,")
        - Only punctuation (with or without whitespace)
        - A stopword (after stripping punctuation)
        - A combination of stopwords and punctuation
        """
        if not text or not text.strip():
            return True

        # Check if it's a single character after stripping whitespace
        if len(text.strip()) <= 1:
            return True

        # Check if it's only punctuation and/or whitespace
        stripped_text = text.strip()
        if all(char in string.punctuation + string.whitespace for char in stripped_text):
            return True

        # Strip punctuation and whitespace, then check if it's a stopword
        cleaned_text = stripped_text.strip(string.punctuation + string.whitespace).lower()

        # If nothing is left after stripping, it's spurious
        if not cleaned_text:
            return True

        # Check if it's a single letter with punctuation (e.g., "A.", "B,")
        if len(cleaned_text) == 1 and cleaned_text.isalpha():
            return True

        # Check if the cleaned text is a stopword
        return cleaned_text in self.stopwords

    def operate(self, text: str, params: dict) -> str:
        """Anonymize a person name using deterministic replacement from a unified name list.

        Args:
            text: The original name text.
            params: Operator parameters. Required/supported keys:
                - salt (str): Cryptographic salt for deterministic output.
                - key (str): Encryption key.
                - entity_type (str, optional): Entity type label used to guide
                  name parsing (e.g. first vs surname classification).

        Returns:
            The anonymized name string preserving token structure, or the
            original text if the value is spurious (stopword, single letter, etc.).
        """
        # Check if the text is spurious
        if self.is_spurious_value(text):
            return text

        salt = params["salt"]
        key = params["key"]
        entity_type = params.get("entity_type")

        # Parse the name into tokens
        parsed = self.name_parser.parse(text, entity_type=entity_type)
        if not parsed or not parsed.get("tokens"):
            return text

        # Anonymize each token using the unified list
        tokens: list[ParsedToken] = parsed["tokens"]
        for token in tokens:
            new_text = self._anonymize_token(token, salt, key)
            token.text = new_text

        # Format the result
        return self.name_parser.format(parsed)

    def _anonymize_token(self, token: ParsedToken, salt: str, key: str) -> str:
        """
        Anonymize a single token.

        All name tokens use the same unified list, so the same input
        always produces the same replacement.

        Args:
            token: The ParsedToken to anonymize
            salt: Salt for consistent selection
            key: Key for consistent selection

        Returns:
            Anonymized token text
        """
        # Surname prefixes (van, de, von) are kept unchanged
        if token.token_type == TokenType.SURNAME_PREFIX:
            return token.text

        # Initials get replaced with random initials
        if token.token_type == TokenType.INITIAL:
            return secure_string_selector(salt, key, self.initials_list, token.text.lower().rstrip("."))

        # All name tokens use the unified list
        return secure_string_selector(salt, key, self.unified_names_list, token.text.lower())

    def validate(self, params: dict) -> None:
        """Validate operator parameters."""
        entity_type = params.get("entity_type", "PERSON")
        if entity_type not in self.supported_entity_types:
            raise ValueError(f"Entity type '{entity_type}' is not supported for HipsNamesAnonymizer.")

        # Get the salt and key
        salt = params.get("salt")
        key = params.get("key")
        if not salt or not key:
            raise ValueError("Both 'salt' and 'key' must be provided for HipsNamesAnonymizer.")

    def operator_name(self) -> str:
        """Return the operator name."""
        return "hips_names"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize
