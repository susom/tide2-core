"""
Known Values Recognizer using Aho-Corasick algorithm.

This module provides a high-performance recognizer for known PII values
that uses the Aho-Corasick algorithm for multi-pattern string matching instead
of Presidio's regex-based pattern matching.

Performance Comparison:
- Presidio PatternRecognizer: Compiles regex on every analyze() call - O(patterns * pattern_length)
- KnownValuesRecognizer: Builds automaton once at init - O(text_length + num_matches) per analyze()

Expected speedup: 50-100x for typical clinical notes with patient identifiers.

Supported Entity Types:
- PHONE: Phone numbers in various formats
- PATIENT: Person names with automatic permutation generation
- LOCATION: Addresses and location information
- EMAIL_ADDRESS: Email addresses
- MRN: Medical record numbers
- HAR: Hospital account record numbers
- ACC_NUM: Accession numbers
- CSN_ID: Contact serial numbers/encounter IDs
- US_SSN: Social Security Numbers
- URL: Web URLs
- MEDICAL_LICENSE: Medical license numbers
- IP_ADDRESS: IP addresses

Usage:
    from tide2.recognizers import (
        KnownValuesRecognizer,
        create_recognizers_for_patient,
    )

    # Create recognizers for a patient
    patient_values = {
        'person': ['John Doe', 'J. Doe'],
        'phone_number': ['555-1234'],
        'mrn': ['MRN123456']
    }
    recognizers = create_recognizers_for_patient(patient_values)

    # Use with Presidio AnalyzerEngine as ad_hoc_recognizers
    results = analyzer.analyze(text, language="en", ad_hoc_recognizers=recognizers)
"""

import logging
import re
import threading
from typing import ClassVar

import ahocorasick
from presidio_analyzer import AnalysisExplanation
from presidio_analyzer import EntityRecognizer
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.recognizer_result import RecognizerResult

from tide2.utils.resource_utils import load_stopwords

logger = logging.getLogger(__name__)

# Maximum values per entity type to prevent memory issues with Aho-Corasick automaton
# These limits are based on analysis of 1M clinical notes distribution:
#
# High-volume encounter identifiers (accumulate across patient history):
#   - acc_num: max 51K, P50=87, P90=921, P99=5317
#   - csn_id: max 12K, P50=257, P90=1308, P99=3431
#   - har: max 6.7K, P50=104, P90=608, P99=1574
#
# Low-volume identifiers (stable per patient):
#   - person: max 13, P99=5 (individual words only, no permutations)
#   - mrn: max 1 per note
#   - phone_number: max 16, P99=6
#
MAX_VALUES_BY_TYPE: dict[str, int | None] = {
    # No limit needed - very low counts
    "person": None,  # Max 13 names x ~3 words = ~39 patterns
    "mrn": None,  # Max 1 per note
    "phone_number": None,  # Max 16 per note
    # High-volume encounter identifiers - use P90 as limit
    "acc_num": 1000,  # P90=921, captures ~90% of notes
    "csn_id": 1500,  # P90=1308, captures ~90% of notes
    "har": 700,  # P90=608, captures ~90% of notes
}
MAX_VALUES_DEFAULT = 200  # For email, ssn, url, etc.

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def generate_person_name_permutations(name_value: str) -> list[str]:
    """
    Extract individual name components for matching in clinical notes.

    This function extracts only the individual words from a name, avoiding
    any permutation explosion. This is the most memory-efficient approach
    and still captures any mention of the person's name components.

    Args:
        name_value: The original name string (e.g., "John Doe Smith", "Smith, John A.")

    Returns:
        List of individual name components (words) to match

    Examples:
        >>> sorted(generate_person_name_permutations("John Doe"))
        ['Doe', 'John']

        >>> sorted(generate_person_name_permutations("John Adam Smith"))
        ['Adam', 'John', 'Smith']
    """
    if not name_value or not name_value.strip():
        return []

    # Clean the name value - remove extra whitespace, commas, and normalize
    name_without_commas = re.sub(r",", " ", name_value)
    cleaned_name = " ".join(name_without_commas.strip().split())
    words = cleaned_name.split()

    # Return unique individual words only
    return list(set(words))


def merge_continuous_entities(
    results: list[RecognizerResult], text: str, max_gap_chars: int = 3
) -> list[RecognizerResult]:
    """
    Utility function to merge continuous/adjacent RecognizerResult objects of the same entity type.

    This function can be used independently of the KnownValuesRecognizer class to post-process
    any list of RecognizerResult objects from any recognizer.

    Args:
        results: List of RecognizerResult objects to potentially merge
        text: Original text for gap validation
        max_gap_chars: Maximum character gap to consider entities continuous (default: 3)

    Returns:
        List of RecognizerResult objects with continuous entities merged

    Example:
        >>> from presidio_analyzer.recognizer_result import RecognizerResult
        >>> result1 = RecognizerResult('PERSON', 0, 4, 0.9)  # "John"
        >>> result2 = RecognizerResult('PERSON', 5, 8, 0.9)  # "Doe"
        >>> text = "John Doe called"
        >>> merged = merge_continuous_entities([result1, result2], text)
        >>> len(merged)
        1
        >>> merged[0].start, merged[0].end
        (0, 8)
    """
    if len(results) <= 1:
        return results

    # Sort results by start position
    sorted_results = sorted(results, key=lambda r: r.start)
    merged_results = []

    i = 0
    while i < len(sorted_results):
        current = sorted_results[i]
        merged_entities = [current]
        j = i + 1

        # Look for adjacent entities of the same type
        while j < len(sorted_results):
            next_result = sorted_results[j]

            # Check if entities are continuous and of the same type
            if current.entity_type == next_result.entity_type and _are_entities_continuous(
                current, next_result, text, max_gap_chars
            ):
                merged_entities.append(next_result)
                current = next_result  # Update current for next iteration
                j += 1
            else:
                break

        # Create merged entity if we found continuous entities
        if len(merged_entities) > 1:
            merged_result = _create_merged_result(merged_entities)
            merged_results.append(merged_result)
        else:
            merged_results.append(merged_entities[0])

        i = j  # Move to next unprocessed entity

    return merged_results


def _are_entities_continuous(
    entity1: RecognizerResult, entity2: RecognizerResult, text: str, max_gap_chars: int = 3
) -> bool:
    """
    Check if two entities are continuous (adjacent or separated by minimal whitespace/punctuation).

    Args:
        entity1: First entity (should have lower start position)
        entity2: Second entity
        text: Original text
        max_gap_chars: Maximum character gap to consider continuous

    Returns:
        True if entities should be merged, False otherwise
    """
    # Entities must be of the same type to be continuous
    if entity1.entity_type != entity2.entity_type:
        return False

    # Ensure entity1 comes before entity2
    if entity1.start > entity2.start:
        entity1, entity2 = entity2, entity1

    gap_start = entity1.end
    gap_end = entity2.start

    # Check if entities overlap or are directly adjacent
    if gap_end <= gap_start:
        return True

    gap_length = gap_end - gap_start

    # If gap is too large, don't merge
    if gap_length > max_gap_chars:
        return False

    # Check what's in the gap - only allow whitespace and minimal punctuation
    gap_text = text[gap_start:gap_end]

    # Allow only whitespace, commas, periods, hyphens, and apostrophes
    allowed_gap_pattern = re.compile(r"^[\s,.\-\']*$")

    return bool(allowed_gap_pattern.match(gap_text))


def _create_merged_result(entities: list[RecognizerResult]) -> RecognizerResult:
    """
    Create a single merged RecognizerResult from a list of continuous entities.

    Args:
        entities: List of RecognizerResult objects to merge (should be sorted by position)

    Returns:
        Single RecognizerResult spanning all the entities
    """
    if not entities:
        raise ValueError("Cannot merge empty list of entities")

    # Use the first and last entities to define the span
    start_pos = min(entity.start for entity in entities)
    end_pos = max(entity.end for entity in entities)

    # Use the entity type from the first entity (they should all be the same)
    entity_type = entities[0].entity_type

    # Use the highest confidence score among the merged entities
    max_score = max(entity.score for entity in entities)

    # Create analysis explanation for the merged entity
    merged_explanation = AnalysisExplanation(
        recognizer="EntityMerger",
        original_score=max_score,
        textual_explanation=f"Merged {len(entities)} continuous {entity_type} entities",
        pattern="merged-continuous-entities",
    )

    return RecognizerResult(
        entity_type=entity_type, start=start_pos, end=end_pos, score=max_score, analysis_explanation=merged_explanation
    )


# ============================================================================
# MAIN RECOGNIZER CLASS
# ============================================================================


class KnownValuesRecognizer(EntityRecognizer):
    """
    High-performance recognizer for known PII values using Aho-Corasick algorithm.

    This recognizer uses the Aho-Corasick automaton for fast multi-pattern string
    matching instead of Presidio's regex-based pattern matching.

    Key Features:
    - O(text_length + num_matches) search complexity (vs O(patterns * text_length) for regex)
    - Automaton built once at initialization, not on every analyze() call
    - Case-insensitive matching with exact string matching
    - Automatic filtering of spurious values (single chars, stopwords)
    - PERSON name permutation generation for flexible matching
    - Selective entity merging based on configurable gap thresholds
    - Compatible with Presidio's AnalyzerEngine as ad_hoc_recognizer
    """

    # Class-level cache for stopwords
    _stopwords: frozenset[str] | None = None
    _stopwords_lock = threading.Lock()

    # Entity type mapping (lowercase key -> Presidio entity type)
    ENTITY_TYPE_MAPPING: ClassVar[dict[str, str]] = {
        "phone_number": "PHONE",
        "person": "PATIENT",
        "location": "LOCATION",
        "email_address": "EMAIL_ADDRESS",
        "mrn": "MRN",
        "har": "HAR",
        "acc_num": "ACC_NUM",
        "csn_id": "CSN_ID",
        "us_ssn": "US_SSN",
        "url": "URL",
        "medical_license": "MEDICAL_LICENSE",
        "ip_address": "IP_ADDRESS",
    }

    @classmethod
    def _load_stopwords(cls) -> frozenset[str]:
        """Thread-safe loading of stopwords with class-level caching."""
        if cls._stopwords is None:
            with cls._stopwords_lock:
                if cls._stopwords is None:
                    cls._stopwords = load_stopwords()
        return cls._stopwords

    def __init__(
        self,
        known_values: dict[str, list[str]],
        entity_type: str | None = None,
        supported_language: str = "en",
        name: str = "KnownValuesRecognizer",
        merge_person_max_gap_chars: int = 1,
        merge_location_max_gap_chars: int = 3,
        score: float = 0.95,
    ):
        """
        Initialize the recognizer with known values for a specific entity type.

        Args:
            known_values: Dictionary of known PII values with entity type keys
            entity_type: Specific entity type to recognize (e.g., 'person', 'phone_number')
            supported_language: Language code (default: "en")
            name: Name of the recognizer instance
            merge_person_max_gap_chars: Max gap for merging PERSON entities (-1 to disable)
            merge_location_max_gap_chars: Max gap for merging LOCATION entities (-1 to disable)
            score: Confidence score for matches (default: 0.95)

        Raises:
            ValueError: If entity_type is invalid or no values provided
        """
        # Validate entity type
        if entity_type is None:
            raise ValueError("entity_type cannot be None")

        self.primary_entity = self.ENTITY_TYPE_MAPPING.get(entity_type)
        if self.primary_entity is None:
            raise ValueError(f"Invalid entity_type '{entity_type}'")

        # Get values for this entity type
        entity_values = known_values.get(entity_type)
        if not entity_values:
            raise ValueError(f"No known values for entity type '{entity_type}'")

        # Initialize parent EntityRecognizer
        super().__init__(
            supported_entities=[self.primary_entity],
            supported_language=supported_language,
            name=name,
        )

        # Store configuration
        self.merge_person_max_gap_chars = merge_person_max_gap_chars
        self.merge_location_max_gap_chars = merge_location_max_gap_chars
        self.score = score
        self.stopwords = frozenset(self._load_stopwords())

        # Build the Aho-Corasick automaton from deny list
        deny_list = self._build_deny_list(entity_values)
        self.automaton = self._build_automaton(deny_list)

    def _is_spurious_value(self, value: str) -> bool:
        """Check if a value is spurious (single char, stopword, etc.)."""
        if not value or not value.strip():
            return True

        cleaned = value.strip()

        # Single character values are spurious
        if len(cleaned) <= 1:
            return True

        # Check if it's a stopword (case-insensitive)
        if cleaned.lower() in self.stopwords:
            return True

        # Check if it's just punctuation or whitespace
        return not any(c.isalnum() for c in cleaned)

    def _build_deny_list(self, entity_values: list[str]) -> list[str]:
        """Build the deny list from entity values, handling PERSON permutations."""
        deny_list = []

        for value in entity_values:
            if self._is_spurious_value(value):
                continue

            if self.primary_entity == "PATIENT":
                # Generate permutations for multi-word names
                cleaned_value = " ".join(value.strip().split())
                word_count = len(re.sub(r",", "", cleaned_value).split())

                if word_count > 1:
                    perms = generate_person_name_permutations(value)
                    for perm in perms:
                        if not self._is_spurious_value(perm):
                            deny_list.append(perm)
                else:
                    deny_list.append(value)
            else:
                deny_list.append(value)

        # Deduplicate
        return list(set(deny_list))

    def _build_automaton(self, deny_list: list[str]) -> ahocorasick.Automaton | None:
        """
        Build Aho-Corasick automaton from deny list patterns.

        The automaton is built once at initialization and reused for all analyze() calls.
        Patterns are stored in lowercase for case-insensitive matching.

        Returns None if deny_list is empty (all values filtered as spurious).
        """
        if not deny_list:
            return None

        automaton = ahocorasick.Automaton()

        for pattern in deny_list:
            if pattern:
                # Store lowercase pattern, but keep original for result
                lower_pattern = pattern.lower()
                # Value stored is (original_pattern, pattern_length)
                automaton.add_word(lower_pattern, (pattern, len(pattern)))

        automaton.make_automaton()
        return automaton

    def load(self) -> None:
        """Load the recognizer (no-op, automaton built at init)."""
        pass

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
        regex_flags: int | None = None,
    ) -> list[RecognizerResult]:
        """
        Analyze text to find known value matches using Aho-Corasick.

        Args:
            text: Text to analyze
            entities: List of entity types to look for
            nlp_artifacts: NLP artifacts (not used)
            regex_flags: Regex flags (not used, kept for interface compatibility)

        Returns:
            List of RecognizerResult objects for matches found
        """
        del nlp_artifacts  # Unused parameter
        if not text:
            return []

        # Check if our entity type is requested
        if self.primary_entity not in entities:
            return []

        # No patterns to match (all values were filtered as spurious)
        if self.automaton is None:
            return []

        results = []

        # Search using Aho-Corasick (case-insensitive)
        text_lower = text.lower()

        for end_idx, (original_pattern, pattern_len) in self.automaton.iter(text_lower):
            start_idx = end_idx - pattern_len + 1

            # Create result with explanation
            explanation = AnalysisExplanation(
                recognizer=self.name,
                original_score=self.score,
                textual_explanation=f"Matched known value: '{original_pattern}'",
                pattern=original_pattern,
            )

            result = RecognizerResult(
                entity_type=self.primary_entity,
                start=start_idx,
                end=end_idx + 1,  # Aho-Corasick returns inclusive end, we need exclusive
                score=self.score,
                analysis_explanation=explanation,
                recognition_metadata={
                    "recognizer_name": self.name,
                    "matched_pattern": original_pattern,
                },
            )
            results.append(result)

        # Apply entity merging if configured
        if len(results) > 1:
            results = self._merge_continuous_results(results, text)

        return results

    def _merge_continuous_results(self, results: list[RecognizerResult], text: str) -> list[RecognizerResult]:
        """Merge continuous entities based on entity type settings."""
        max_gap_chars = -1

        if self.primary_entity == "PATIENT":
            max_gap_chars = self.merge_person_max_gap_chars
        elif self.primary_entity == "LOCATION":
            max_gap_chars = self.merge_location_max_gap_chars

        if max_gap_chars == -1:
            return results

        return merge_continuous_entities(results, text, max_gap_chars)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_recognizers_for_patient(
    patient_known_values: dict[str, list[str]],
    merge_person_max_gap_chars: int = 1,
    merge_location_max_gap_chars: int = 3,
) -> list[KnownValuesRecognizer]:
    """
    Factory function to create multiple recognizer instances for all entity types for a patient.

    Creates a separate KnownValuesRecognizer for each entity type that has non-empty values
    in the patient_known_values dictionary. This allows a single analyzer to recognize
    multiple types of PII for the same patient.

    Values are truncated to prevent memory issues with large patient identifier sets:
    - PERSON: Limited to MAX_VALUES_PERSON (20) due to factorial permutation explosion
    - Other types: Limited to MAX_VALUES_DEFAULT (100)

    Args:
        patient_known_values: Dictionary of known PII values for a specific patient with
                             entity type keys (e.g., 'person', 'phone_number', 'mrn')
                             and lists of values as values
        merge_person_max_gap_chars: Maximum character gap for merging PERSON entities.
                                  If -1, no merging for PERSON entities (default: 1)
        merge_location_max_gap_chars: Maximum character gap for merging LOCATION entities.
                                    If -1, no merging for LOCATION entities (default: 3)

    Returns:
        List of configured KnownValuesRecognizer instances, one for each entity type
        that has non-empty values in patient_known_values

    Example:
        >>> patient_values = {
        ...     'person': ['John Doe', 'J. Doe'],
        ...     'phone_number': ['555-1234'],
        ...     'mrn': ['MRN123456']
        ... }
        >>> recognizers = create_recognizers_for_patient(patient_values)
        >>> len(recognizers)
        3
    """
    recognizers = []

    # Truncate values per entity type to prevent memory issues
    truncated_values = _truncate_patient_values(patient_known_values)

    for entity_type, values in truncated_values.items():
        if not values:
            continue

        # Skip unsupported entity types
        if entity_type not in KnownValuesRecognizer.ENTITY_TYPE_MAPPING:
            continue

        try:
            recognizer = KnownValuesRecognizer(
                known_values=truncated_values,
                entity_type=entity_type,
                merge_person_max_gap_chars=merge_person_max_gap_chars,
                merge_location_max_gap_chars=merge_location_max_gap_chars,
            )
            recognizers.append(recognizer)
        except ValueError:
            # Skip if no valid values for this entity type
            continue

    return recognizers


def _truncate_patient_values(patient_known_values: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Truncate patient values per entity type to prevent memory issues.

    Uses entity-specific limits from MAX_VALUES_BY_TYPE, falling back to MAX_VALUES_DEFAULT.
    Some entity types (like person) have no limit since individual word extraction is efficient.

    Args:
        patient_known_values: Original dictionary of known PII values

    Returns:
        Dictionary with truncated value lists
    """
    truncated = {}

    for entity_type, values in patient_known_values.items():
        if not values:
            truncated[entity_type] = values
            continue

        # Get entity-specific limit, or default. None means no limit.
        max_values = MAX_VALUES_BY_TYPE.get(entity_type, MAX_VALUES_DEFAULT)

        # If no limit (None), keep all values
        if max_values is None:
            truncated[entity_type] = values
            continue

        original_count = len(values)

        if original_count > max_values:
            # Truncate to max values
            truncated[entity_type] = values[:max_values]
            logger.warning(
                "Truncated %s values from %d to %d for entity type '%s' to prevent memory issues",
                entity_type,
                original_count,
                max_values,
                entity_type,
            )
        else:
            truncated[entity_type] = values

    return truncated
