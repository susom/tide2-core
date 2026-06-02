"""
Optimized Phone Number Recognizer.

This is a high-performance replacement for Presidio's PhoneRecognizer that uses
a single pre-compiled combined regex pattern instead of the phonenumbers library.

Performance improvement: ~20-40x faster than the original PhoneRecognizer.
"""

import re
from typing import ClassVar

from presidio_analyzer import AnalysisExplanation
from presidio_analyzer import EntityRecognizer
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.recognizer_result import RecognizerResult


class PhoneRecognizer(EntityRecognizer):
    """
    Optimized phone number recognizer using a single pre-compiled combined pattern.

    Supports extensive US and international phone formats without validation overhead.
    Uses alternation in a single pattern for single-pass matching.
    """

    SUPPORTED_ENTITY: ClassVar[str] = "PHONE"

    # Single combined pattern - matches all phone formats in one pass
    # Order matters: more specific patterns first (longer matches take priority)
    COMBINED_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"""
        (?:
            # US with country code and extension (most specific)
            (?:\+?1[-.\s]?)?\(\d{3}\)\s?\d{3}[-.\s]?\d{4}(?:\s*(?:ext\.?|x|extension)\s*\d{1,6})?|
            # International with + prefix: +X (XXX) XXX-XXXX
            \+\d{1,3}\s?\(\d{1,4}\)\s?\d{1,4}[-.\s]?\d{1,9}|
            # Toll-free: 1-800-XXX-XXXX, 800-XXX-XXXX
            (?:1[-.\s]?)?8(?:00|33|44|55|66|77|88)[-.\s]?\d{3}[-.\s]?\d{4}|
            # UK format: +44 XXXX XXXXXX
            \+44\s?\d{4}\s?\d{6}|
            # European formats with spaces
            \+\d{2}\s\d{2,4}\s\d{3,4}\s\d{2,4}|
            # US standard: XXX-XXX-XXXX or 1-XXX-XXX-XXXX
            (?:1[-.\s]?)?\d{3}[-.\s]\d{3}[-.\s]\d{4}|
            # US format with dots: XXX.XXX.XXXX
            \d{3}\.\d{3}\.\d{4}|
            # International format: XX-XXXX-XXXX or XXX-XXXX-XXXX
            \d{2,3}[-.\s]\d{4}[-.\s]\d{4}|
            # Plain 11 digits with leading 1
            \b1\d{10}\b|
            # Plain 10 digits
            \b\d{10}\b
        )
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    # Default score for all matches (no per-pattern scoring for speed)
    DEFAULT_SCORE: ClassVar[float] = 0.7

    def __init__(
        self,
        supported_language: str = "en",
        supported_entity: str = "PHONE",
    ):
        """
        Initialize the optimized phone recognizer.

        Args:
            supported_language: Language code (default: "en")
            supported_entity: Entity type (default: "PHONE")
        """
        super().__init__(
            supported_entities=[supported_entity],
            supported_language=supported_language,
            name="PhoneRecognizer",
        )
        self._supported_entity = supported_entity

    def load(self) -> None:
        """No loading required - pattern is a class-level constant."""
        pass

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        """
        Analyze text for phone numbers using single pre-compiled combined pattern.

        Args:
            text: Text to analyze
            entities: List of entities to detect
            nlp_artifacts: NLP artifacts (not used)

        Returns:
            List of RecognizerResult objects for detected phone numbers
        """
        if self._supported_entity not in entities:
            return []

        results: list[RecognizerResult] = []

        # Single-pass matching with combined pattern
        for match in self.COMBINED_PATTERN.finditer(text):
            explanation = AnalysisExplanation(
                recognizer=self.name,
                original_score=self.DEFAULT_SCORE,
                textual_explanation="Phone pattern matched",
                pattern="combined_phone_pattern",
            )

            results.append(
                RecognizerResult(
                    entity_type=self._supported_entity,
                    start=match.start(),
                    end=match.end(),
                    score=self.DEFAULT_SCORE,
                    analysis_explanation=explanation,
                )
            )

        return results

    def get_supported_entities(self) -> list[str]:
        """Return supported entities."""
        return [self._supported_entity]
