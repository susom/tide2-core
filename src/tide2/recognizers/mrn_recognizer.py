"""Medical Record Number (MRN) recognizer.

Detects MRN identifiers in clinical text using regex patterns for common
formats (e.g., ``###-##-##``, ``###-##-####``).
"""

from presidio_analyzer import Pattern
from presidio_analyzer import PatternRecognizer


class MrnRecognizer(PatternRecognizer):
    """
    Recognizes Medical Record Numbers (MRN) in various formats using regex patterns.

    This recognizer identifies MRN patterns such as:
    - ###-##-##
    - ###-##-####
    - #######-#
    - ########
    - ##-##-##
    - ##########
    - MRN/MR/Medical Record followed by numbers
    - MED REC #: (####)######## or MED REC #: ########
    """

    # Define the patterns for MRN detection
    PATTERNS = [
        Pattern(name="mrn_ddd_dd_dd", regex=r"\b\d{3}-\d{2}-\d{2}(?:-\d)?\b", score=0.8),
        Pattern(name="mrn_ddd_dd_dddd", regex=r"\b\d{3}-\d{2}-\d{4}(?:-\d)?\b", score=0.8),
        Pattern(name="mrn_ddddddd_d", regex=r"\b\d{7}-\d\b", score=0.8),
        Pattern(
            name="mrn_8_digits",
            regex=r"\b\d{8}\b",
            score=0.7,  # Lower score as this could match other numbers
        ),
        Pattern(
            name="mrn_dd_dd_dd",
            regex=r"\b\d{2}-\d{2}-\d{2}\b",
            score=0.6,  # Lower score due to potential date conflicts
        ),
        Pattern(
            name="mrn_10_digits",
            regex=r"\b\d{10}\b",
            score=0.7,  # Lower score as this could match other numbers
        ),
        Pattern(name="mrn_with_mrn_colon_space", regex=r"(?i)(?<=MRN:\s)\d[\d -]*\d", score=0.95),
        Pattern(name="mrn_with_mrn_colon_no_space", regex=r"(?i)(?<=MRN:)\d[\d -]*\d", score=0.95),
        Pattern(name="mrn_with_mrn_colon_double_space", regex=r"(?i)(?<=MRN:\s\s)\d[\d -]*\d", score=0.95),
        Pattern(name="mrn_with_mr_hash", regex=r"(?i)(?<=MR#\s)[\d][\d -]*\d", score=0.95),
        Pattern(name="mrn_with_mrn_space", regex=r"(?i)(?<=MRN\s)[\d][\d -]*\d", score=0.95),
        Pattern(name="mrn_with_mr_space", regex=r"(?i)(?<=MR\s)[\d][\d -]*\d", score=0.90),
        Pattern(name="mrn_medical_record_number", regex=r"(?i)(?<=Medical Record Number:\s)[\d][\d -]*\d", score=0.95),
        Pattern(name="mrn_med_rec_hash_parentheses", regex=r"(?i)(?<=MED REC #:\s+)(?:\(\d+\))?\d+", score=0.95),
        Pattern(name="mrn_med_rec_hash_parentheses", regex=r"(?i)(?<=MED REC:\s+)(?:\(\d+\))?\d+", score=0.95),
    ]

    def __init__(
        self,
        patterns: list[Pattern] | None = None,
        supported_language: str = "en",
        supported_entity: str = "MRN",
    ):
        """
        Initialize the MRN recognizer.

        Args:
            patterns: Optional list of patterns to override defaults
            supported_language: Language code (default: "en")
            supported_entity: Entity type name (default: "MRN")
        """
        patterns = patterns or self.PATTERNS

        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            supported_language=supported_language,
            name="MrnRecognizer",
        )
