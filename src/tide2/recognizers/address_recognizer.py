"""
Simplified Address Recognizer for US addresses.

This recognizer uses direct regex matching (no windowing) with usaddress validation:
1. Comprehensive regex patterns capture full addresses in one match
2. usaddress library validates the matched text
3. Additional filters eliminate false positives

Design principles:
- No window expansion (avoids boundary issues)
- Regex defines exact span boundaries
- usaddress provides semantic validation
- High precision over high recall
"""

import re
from typing import ClassVar

from presidio_analyzer import EntityRecognizer
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.recognizer_result import RecognizerResult


class AddressRecognizer(EntityRecognizer):
    """
    High-precision US address recognizer using regex + usaddress validation.

    The recognizer returns LOCATION entity type for detected addresses.
    """

    SUPPORTED_ENTITY: ClassVar[str] = "LOCATION"

    # Street type suffixes
    STREET_TYPES: ClassVar[str] = (
        r"(?:St(?:reet)?|Ave(?:nue)?|Blvd|Boulevard|Dr(?:ive)?|Rd|Road|"
        r"Ln|Lane|Way|Ct|Court|Pl|Place|Cir(?:cle)?|Pkwy|Parkway|"
        r"Ter(?:race)?|Hwy|Highway|Trl|Trail|Loop|Pass|Run|Path)"
    )

    # US State patterns (abbreviations and full names)
    STATE_PATTERN: ClassVar[str] = (
        r"(?:CA|AZ|NV|OR|WA|TX|NY|FL|IL|OH|PA|GA|NC|MI|NJ|VA|MA|TN|IN|MO|"
        r"WI|MN|CO|AL|SC|LA|KY|OK|CT|UT|IA|AR|MS|KS|NE|NM|WV|ID|HI|ME|NH|"
        r"RI|MT|DE|SD|ND|AK|VT|WY|DC|California|Arizona|Nevada|Oregon|Texas)"
    )

    # ZIP code pattern
    ZIP_PATTERN: ClassVar[str] = r"\d{5}(?:-\d{4})?"

    # Direction prefixes
    DIRECTION: ClassVar[str] = r"(?:N\.?|S\.?|E\.?|W\.?|NE|NW|SE|SW|North|South|East|West)"

    # Medical/professional credentials that look like state abbreviations
    CREDENTIAL_ABBREVS: ClassVar[set[str]] = {
        "MD",
        "DO",
        "PA",
        "NP",
        "RN",
        "LPN",
        "MA",
        "MT",
        "DC",
        "ND",
        "PT",
        "OT",
        "DDS",
        "DMD",
        "DPM",
        "OD",
        "PhD",
        "PsyD",
        "MS",
        "BS",
        "BA",
    }

    # Medical measurement patterns to filter out
    MEDICAL_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"\b\d+\s*(?:mL|mg|mm|cm|mcg|units?|cc|mg/dL|mmol|mEq)\b", re.IGNORECASE
    )

    # Pattern 1: Full US address with ZIP
    # Matches: "456 Oak Ave, Oakland, CA 94612-1234"
    # Requires commas between street/city/state to avoid catastrophic
    # backtracking — the original [\,\s]+ separator overlapped with \s in
    # the street-name and city-name character classes, causing exponential
    # backtracking on dense clinical text.
    FULL_ADDRESS_PATTERN: ClassVar[re.Pattern] = re.compile(
        rf"""
        \b
        (\d{{1,5}})                              # Street number
        \s+
        (?:{DIRECTION}\s+)?                      # Optional direction
        ([A-Za-z][A-Za-z0-9\-\'\.]*(?:\s+[A-Za-z][A-Za-z0-9\-\'\.]*)*) # Street name (word-separated)
        \s+
        ({STREET_TYPES})                         # Street type
        \.?
        (?:\s+(?:Ste|Suite|Apt|Unit|Fl|Floor|\#)\s*[A-Za-z0-9\-]+)?  # Optional suite
        \s*,\s*                                  # Comma separator (unambiguous)
        ([A-Za-z][A-Za-z\-\']*(?:\s+[A-Za-z][A-Za-z\-\']*)*) # City name (word-separated)
        \s*,\s*                                  # Comma separator (unambiguous)
        ({STATE_PATTERN})                        # State
        \.?\s*
        ({ZIP_PATTERN})                          # ZIP code
        \b
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    # Pattern 2: Address with state and ZIP (city may be part of street line)
    # Matches: "725 Welch Rd. Palo Alto, CA 94304"
    # Requires comma before state to avoid backtracking — the original
    # separator [\,\s]+ overlapped with \s in the middle group causing
    # exponential backtracking on dense clinical text.
    ADDRESS_STATE_ZIP_PATTERN: ClassVar[re.Pattern] = re.compile(
        rf"""
        \b
        (\d{{1,5}})                              # Street number
        \s+
        (?:{DIRECTION}\s+)?                      # Optional direction
        ([A-Za-z][A-Za-z0-9\-\'\.]*(?:\s+[A-Za-z][A-Za-z0-9\-\'\.]*)*) # Street + city words
        \s*,\s*                                  # Comma separator (unambiguous)
        ({STATE_PATTERN})                        # State
        \.?\s*
        ({ZIP_PATTERN})                          # ZIP code
        \b
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    # Pattern 3: PO Box with location
    # Requires commas between city/state to avoid catastrophic backtracking
    # (same fix as FULL_ADDRESS_PATTERN and ADDRESS_STATE_ZIP_PATTERN).
    PO_BOX_PATTERN: ClassVar[re.Pattern] = re.compile(
        rf"""
        \b
        (?:P\.?\s*O\.?\s*Box|Post\s+Office\s+Box)
        \s+
        (\d+)                                    # Box number
        \s*,\s*                                  # Comma separator (unambiguous)
        ([A-Za-z][A-Za-z\-\']*(?:\s+[A-Za-z][A-Za-z\-\']*)*) # City (word-separated)
        \s*,\s*                                  # Comma separator (unambiguous)
        ({STATE_PATTERN})                        # State
        \.?\s*
        ({ZIP_PATTERN})?                         # Optional ZIP
        \b
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    # Minimum score threshold
    MIN_SCORE: ClassVar[float] = 0.85

    def __init__(
        self,
        supported_language: str = "en",
        supported_entity: str = "LOCATION",
    ):
        """Initialize the address recognizer."""
        super().__init__(
            supported_entities=[supported_entity],
            supported_language=supported_language,
            name="AddressRecognizer",
        )
        self._supported_entity = supported_entity

    def load(self) -> None:
        """No loading required - patterns are class-level constants."""
        pass

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        """
        Analyze text for US addresses.

        Args:
            text: Text to analyze
            entities: List of entities to detect
            nlp_artifacts: NLP artifacts (not used)

        Returns:
            List of RecognizerResult objects for detected addresses
        """
        if self._supported_entity not in entities:
            return []

        results: list[RecognizerResult] = []
        seen_spans: set[tuple[int, int]] = set()

        for pattern, score in [
            (self.FULL_ADDRESS_PATTERN, 0.95),
            (self.ADDRESS_STATE_ZIP_PATTERN, 0.90),
            (self.PO_BOX_PATTERN, 0.85),
        ]:
            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()

                # Skip overlapping spans
                if any(s <= start < e or s < end <= e for s, e in seen_spans):
                    continue

                candidate_text = text[start:end]

                # Validate with usaddress and filters
                if self._is_valid_address(candidate_text):
                    seen_spans.add((start, end))
                    results.append(
                        RecognizerResult(
                            entity_type=self._supported_entity,
                            start=start,
                            end=end,
                            score=score,
                            recognition_metadata={
                                "recognizer_name": self.name,
                            },
                        )
                    )

        return results

    def _is_valid_address(self, text: str) -> bool:
        """
        Validate address using usaddress and additional filters.

        Returns:
            True if text is a valid address, False otherwise.
        """
        # Filter 1: Reject text with medical measurements
        if self.MEDICAL_PATTERN.search(text):
            return False

        # Filter 2: Reject text with noise characters
        noise_chars = {"[", "]", "{", "}", "|", "\\", "<", ">"}
        if any(c in text for c in noise_chars):
            return False

        # Filter 3: Reject clinical workflow text
        clinical_terms = ["called pt", "left message", "vm", "voicemail", "study", "test location"]
        text_lower = text.lower()
        if any(term in text_lower for term in clinical_terms):
            return False

        # Filter 4: Reject if starts with year (1900-2099) followed by non-address content
        year_match = re.match(r"^(19|20)\d{2}\s+", text)
        if year_match:
            # Text starts with year - likely a date prefix, not street number
            return False

        # Filter 5: Reject duplicate addresses (same street appears twice)
        # Check if street type words appear more than once (with word boundaries)
        street_type_pattern = r"\b(?:Street|Avenue|Boulevard|Drive|Road|Lane|Court|Place|Circle|Parkway|Highway|Trail|Way|St|Ave|Blvd|Dr|Rd|Ln|Ct|Pl|Cir|Pkwy|Hwy|Trl)\b"
        street_type_matches = re.findall(street_type_pattern, text, re.IGNORECASE)
        if len(street_type_matches) > 1:
            # Multiple street types - likely duplicate or concatenated addresses
            return False

        # Filter 6: Validate with usaddress
        try:
            import usaddress

            parsed_components, _ = usaddress.tag(text)
        except Exception:
            return False

        # Must have street number and street type
        has_street_num = "AddressNumber" in parsed_components
        has_street_type = "StreetNamePostType" in parsed_components
        has_zip = "ZipCode" in parsed_components

        if not (has_street_num and has_street_type and has_zip):
            # PO Box is acceptable
            if "USPSBoxType" not in parsed_components:
                return False

        # Filter out credential false positives
        state_val = parsed_components.get("StateName", "")
        if state_val.upper() in self.CREDENTIAL_ABBREVS:
            if not has_zip:
                return False

        return True

    def get_supported_entities(self) -> list[str]:
        """Return supported entities."""
        return [self._supported_entity]
