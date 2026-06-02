"""Hospital Account Record (HAR) number recognizer.

Detects HAR identifiers in clinical text using regex pattern matching.
"""

from presidio_analyzer import Pattern
from presidio_analyzer import PatternRecognizer


class HarRecognizer(PatternRecognizer):
    """
    Recognizer for HAR (Hospital Account Record) numbers using Presidio framework.

    This recognizer identifies HAR numbers in text following the pattern:
    HAR: (optional space) followed by digits
    The recognized entity includes only the number, not the "HAR" prefix.
    """

    # Define the patterns for HAR detection
    PATTERNS = [
        Pattern(
            name="har_pattern",
            regex=r"(?<=\bHAR\s{0,10}:\s{0,10})\d+",
            score=0.95,  # High confidence score for explicit HAR pattern
        ),
        # Alternative pattern for edge cases where lookbehind might have issues
        Pattern(
            name="har_pattern_alt",
            regex=r"(?i)(?<=\bhar\s{0,10}:\s{0,10})\d+",
            score=0.95,  # Case-insensitive version
        ),
    ]

    def __init__(
        self,
        patterns: list[Pattern] | None = None,
        supported_language: str = "en",
        supported_entity: str = "HAR",
    ):
        """
        Initialize the HAR Recognizer.

        Args:
            patterns: List of patterns to use for detection (uses default if None)
            supported_language: Language supported by this recognizer
            supported_entity: The entity type this recognizer identifies
        """
        patterns = patterns or self.PATTERNS

        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            supported_language=supported_language,
            name="HarRecognizer",
        )
