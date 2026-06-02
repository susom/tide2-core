"""
Optimized Base64 Image Recognizer.

This is a high-performance recognizer for detecting base64-encoded images in text.
Uses pre-compiled regex patterns for speed.

Detects:
- Generic base64 strings (500+ characters)
- Data URI scheme images (data:image/...;base64,...)
- PNG images (starts with iVBORw0KGgo signature)
"""

import re
from typing import ClassVar

from presidio_analyzer import AnalysisExplanation
from presidio_analyzer import EntityRecognizer
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.recognizer_result import RecognizerResult


class Base64ImageRecognizer(EntityRecognizer):
    """
    Optimized base64 image recognizer using pre-compiled patterns.

    Matches base64-encoded images in various formats including data URIs
    and raw base64 strings.
    """

    SUPPORTED_ENTITY: ClassVar[str] = "BASE64_IMAGE"

    # Pattern for data URI scheme images (highest confidence - explicit image declaration)
    DATA_URI_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}",
    )

    # Pattern for PNG images (high confidence - known PNG signature)
    PNG_SIGNATURE_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"iVBORw0KGgo[A-Za-z0-9+/]{100,}",
    )

    # Pattern for generic base64 strings (lower confidence - could be other data)
    GENERIC_BASE64_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"[A-Za-z0-9+/]{500,}[=]{0,2}",
    )

    # Confidence scores
    DATA_URI_SCORE: ClassVar[float] = 0.95
    PNG_SIGNATURE_SCORE: ClassVar[float] = 0.90
    GENERIC_BASE64_SCORE: ClassVar[float] = 0.70

    def __init__(
        self,
        supported_language: str = "en",
        supported_entity: str = "BASE64_IMAGE",
    ):
        """
        Initialize the optimized base64 image recognizer.

        Args:
            supported_language: Language code (default: "en")
            supported_entity: Entity type (default: "BASE64_IMAGE")
        """
        super().__init__(
            supported_entities=[supported_entity],
            supported_language=supported_language,
            name="Base64ImageRecognizer",
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
        Analyze text for base64-encoded images using pre-compiled patterns.

        Args:
            text: Text to analyze
            entities: List of entities to detect
            nlp_artifacts: NLP artifacts (not used)

        Returns:
            List of RecognizerResult objects for detected base64 images
        """
        if self._supported_entity not in entities:
            return []

        results: list[RecognizerResult] = []
        matched_spans: set[tuple[int, int]] = set()

        # Pattern 1: Data URI images (highest priority)
        for match in self.DATA_URI_PATTERN.finditer(text):
            span = (match.start(), match.end())
            if span not in matched_spans:
                matched_spans.add(span)
                results.append(self._create_result(span[0], span[1], self.DATA_URI_SCORE, "data_uri"))

        # Pattern 2: PNG signature
        for match in self.PNG_SIGNATURE_PATTERN.finditer(text):
            span = (match.start(), match.end())
            if not self._overlaps_existing(span, matched_spans):
                matched_spans.add(span)
                results.append(self._create_result(span[0], span[1], self.PNG_SIGNATURE_SCORE, "png_signature"))

        # Pattern 3: Generic base64 (lowest priority)
        for match in self.GENERIC_BASE64_PATTERN.finditer(text):
            span = (match.start(), match.end())
            if not self._overlaps_existing(span, matched_spans):
                matched_spans.add(span)
                results.append(self._create_result(span[0], span[1], self.GENERIC_BASE64_SCORE, "generic_base64"))

        return results

    def _overlaps_existing(self, new_span: tuple[int, int], existing_spans: set[tuple[int, int]]) -> bool:
        """Check if new span overlaps with any existing spans."""
        new_start, new_end = new_span
        return any(new_start < end and new_end > start for start, end in existing_spans)

    def _create_result(self, start: int, end: int, score: float, pattern_name: str) -> RecognizerResult:
        """Create a RecognizerResult with explanation."""
        explanation = AnalysisExplanation(
            recognizer=self.name,
            original_score=score,
            textual_explanation=f"Base64 image pattern matched ({pattern_name})",
            pattern=pattern_name,
        )
        return RecognizerResult(
            entity_type=self._supported_entity,
            start=start,
            end=end,
            score=score,
            analysis_explanation=explanation,
        )

    def get_supported_entities(self) -> list[str]:
        """Return supported entities."""
        return [self._supported_entity]
