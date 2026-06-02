"""
CachedResultsTransformerRecognizer for pre-computed NER results.

This is a lightweight recognizer that returns pre-computed NER results from
GPU batch processing. It simply converts the cached results into Presidio
RecognizerResult objects without any additional processing.

The GPU batch processing (batch_transformer.py) already:
- Runs transformer inference
- Aggregates BIO tokens into entity spans
- Reconstructs document-level spans from chunks
- Deduplicates overlapping entities
- Adds recognition_metadata in Presidio-compatible format

This recognizer just passes through those pre-computed results.

Usage:
    >>> from tide2.recognizers import create_cached_recognizer
    >>> recognizer = create_cached_recognizer(results_json)
    >>> results = analyzer.analyze(text, language="en", ad_hoc_recognizers=[recognizer])
"""

import json
import logging
from typing import Any
from typing import ClassVar

from presidio_analyzer import EntityRecognizer
from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

logger = logging.getLogger(__name__)


class CachedResultsTransformerRecognizer(EntityRecognizer):
    """
    Lightweight recognizer that returns pre-computed NER results.

    This recognizer simply passes through pre-computed results from GPU batch
    processing. No additional processing is done - the batch_transformer already:
    - Ran transformer inference on GPU
    - Aggregated BIO tokens into entity spans
    - Reconstructed document-level spans from chunks
    - Deduplicated overlapping entities
    - Added recognition_metadata in Presidio format

    Usage:
        >>> recognizer = create_cached_recognizer(results_json)
        >>> results = analyzer.analyze(text, language="en", ad_hoc_recognizers=[recognizer])
    """

    # Default supported entities - covers common NER entity types
    DEFAULT_SUPPORTED_ENTITIES: ClassVar[list[str]] = [
        "DATE",
        "DOCTOR",
        "PATIENT",
        "PERSON",
        "HOSPITAL",
        "ID",
        "PHONE",
        "PHONE_NUMBER",
        "LOCATION",
        "AGE",
        "WEB",
        "OTHER",
        "EMAIL_ADDRESS",
        "URL",
        "US_SSN",
        "DATE_TIME",
        "HCW",
        "VENDOR",  # Produced by Stanford deidentifier transformer model
    ]

    def __init__(
        self,
        results: str | list[dict[str, Any]] | None = None,
        supported_entities: list[str] | None = None,
        supported_language: str = "en",
        name: str = "CachedResultsTransformerRecognizer",
    ):
        """
        Initialize with pre-computed results from GPU batch processing.

        Args:
            results: Pre-computed NER results in Presidio format. Either JSON string
                    or list of dicts with entity_type, start, end, score, and optional
                    recognition_metadata. Can be None for notes with no entities.
            supported_entities: Entity types to support (default: common NER types)
            supported_language: Language code (default: "en")
            name: Recognizer name identifier
        """
        super().__init__(
            supported_entities=supported_entities or self.DEFAULT_SUPPORTED_ENTITIES,
            supported_language=supported_language,
            name=name,
        )

        # Parse and store results at construction time (immutable after init)
        self._cached_results: list[dict[str, Any]] = self._parse_results(results)

    def _parse_results(self, results: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """
        Parse results from JSON string or list format.

        Args:
            results: Either a JSON string or list of dicts with NER results.

        Returns:
            List of result dictionaries.
        """
        if results is None or results == "":
            return []

        if isinstance(results, str):
            try:
                return json.loads(results)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse cached results JSON: {e}")
                return []

        return list(results)  # Make a copy to ensure immutability

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        """
        Return pre-computed NER results as Presidio RecognizerResults.

        This is a lightweight pass-through - the GPU batch processing already
        computed all entity information including recognition_metadata.

        Args:
            text: The text being analyzed (unused - metadata already computed)
            entities: List of entity types to filter results by
            nlp_artifacts: NLP artifacts (unused)

        Returns:
            List of RecognizerResult objects from pre-computed results
        """
        results = []

        for cached_entity in self._cached_results:
            entity_type = cached_entity.get("entity_type")

            # Filter by requested entities
            if entity_type not in entities:
                continue

            # Create RecognizerResult directly from cached data
            # Copy recognition_metadata and remove recognizer_identifier so that
            # AnalyzerEngine.__add_recognizer_id_if_not_exists will set it correctly
            # to match THIS recognizer's ID (required for _enhance_using_context)
            recognition_metadata = dict(cached_entity.get("recognition_metadata") or {})
            recognition_metadata.pop("recognizer_identifier", None)

            result = RecognizerResult(
                entity_type=entity_type,
                start=cached_entity["start"],
                end=cached_entity["end"],
                score=float(cached_entity.get("score", 0.0)),
                analysis_explanation=None,  # Skip expensive explanation object
                recognition_metadata=recognition_metadata,
            )
            results.append(result)

        return results

    def load(self) -> None:
        """
        Load method required by the EntityRecognizer interface.

        This is a no-op for the cached results recognizer since there's
        nothing to load.
        """
        pass

    def get_supported_entities(self) -> list[str]:
        """
        Get the list of supported entities.

        Returns:
            List of entity types this recognizer claims to support
        """
        return self.supported_entities


def create_cached_recognizer(
    results: str | list[dict[str, Any]] | None,
    supported_entities: list[str] | None = None,
) -> CachedResultsTransformerRecognizer:
    """
    Create a lightweight recognizer from pre-computed GPU batch results.

    This factory creates a recognizer that passes through pre-computed NER
    results from GPU batch processing. Create a new instance for each note.

    Args:
        results: Pre-computed NER results from GPU batch in Presidio format.
                Either JSON string or list of dicts with entity_type, start,
                end, score, and recognition_metadata.
        supported_entities: Optional entity type filter.

    Returns:
        CachedResultsTransformerRecognizer for use as an ad-hoc recognizer.

    Example:
        >>> recognizer = create_cached_recognizer(record["recognizer_results_json"])
        >>> results = analyzer.analyze(text, language="en", ad_hoc_recognizers=[recognizer])
    """
    return CachedResultsTransformerRecognizer(
        results=results,
        supported_entities=supported_entities,
    )
