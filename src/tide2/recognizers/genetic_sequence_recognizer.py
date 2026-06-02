"""
High-performance genetic sequence recognizer for DNA/RNA sequences.

This recognizer identifies continuous sequences of nucleotide bases (A, T, G, C, U)
that are commonly found in genetic data, research reports, and clinical genetics notes.

Performance: Optimized regex-based recognition without external dependencies.
"""

import logging
import re

from presidio_analyzer import AnalysisExplanation
from presidio_analyzer import EntityRecognizer
from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

logger = logging.getLogger(__name__)


class GeneticSequenceRecognizer(EntityRecognizer):
    """
    Recognizer for genetic sequences (DNA/RNA).

    Detects continuous sequences of nucleotide bases:
    - DNA: A (Adenine), T (Thymine), G (Guanine), C (Cytosine)
    - RNA: A (Adenine), U (Uracil), G (Guanine), C (Cytosine)

    Sequences must be:
    - At least 20 characters long (configurable, default prevents false positives)
    - Continuous without spaces or line breaks
    - Contain only valid nucleotide letters (case-insensitive)

    Example sequences:
        GGATGTGTGTGACAGTTTCTGACCAATGTCTC
        AUGCAUGCAUGCAUGCAUGCAUG
        atcgatcgatcgatcgatcgatcg
    """

    GENETIC_SEQUENCE_ENTITY = "GENETIC_SEQUENCE"

    # Score thresholds for sequence length
    LONG_SEQUENCE_THRESHOLD = 50
    MEDIUM_SEQUENCE_THRESHOLD = 30
    # Maximum base frequency ratio for diversity bonus
    BASE_DIVERSITY_THRESHOLD = 0.7
    # Context window size for searching nearby keywords
    CONTEXT_WINDOW_SIZE = 100

    # Default context words (class-level tuple to avoid recreation)
    _DEFAULT_CONTEXT = (
        "sequence",
        "dna",
        "rna",
        "nucleotide",
        "genetic",
        "genome",
        "gene",
        "allele",
        "mutation",
        "variant",
        "base",
        "codon",
        "exon",
        "intron",
        "primer",
        "amplicon",
        "pcr",
        "sequencing",
    )

    def __init__(
        self,
        supported_language: str = "en",
        min_sequence_length: int = 20,
        context: list[str] | None = None,
    ):
        """
        Initialize the genetic sequence recognizer.

        Args:
            supported_language: Language code (default: "en")
            min_sequence_length: Minimum sequence length to detect (default: 20)
            context: Optional list of context words to boost confidence
        """
        super().__init__(
            supported_entities=[self.GENETIC_SEQUENCE_ENTITY],
            supported_language=supported_language,
            name="GeneticSequenceRecognizer",
        )

        self.min_sequence_length = min_sequence_length

        # Compile regex pattern for genetic sequences (case-insensitive)
        self._pattern = re.compile(rf"\b[ATGCUatgcu]{{{min_sequence_length},}}\b")

        # Pre-compile context detection regex
        all_keywords = list(context or []) + list(self._DEFAULT_CONTEXT)
        self._context_pattern = re.compile(
            r"\b(" + "|".join(re.escape(kw) for kw in all_keywords) + r")\b",
            re.IGNORECASE,
        )

    def load(self) -> None:
        """Load the recognizer. No external resources needed."""
        pass

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        """
        Analyze text to find genetic sequences.

        Args:
            text: Input text to analyze
            entities: List of entity types to detect (must include GENETIC_SEQUENCE)
            nlp_artifacts: Not used by this recognizer

        Returns:
            List of RecognizerResult objects for detected sequences
        """
        if self.GENETIC_SEQUENCE_ENTITY not in entities:
            return []

        results = []
        text_len = len(text)

        for match in self._pattern.finditer(text):
            start = match.start()
            end = match.end()
            seq_len = end - start

            # Get sequence and convert to uppercase once
            sequence_upper = match.group().upper()

            # Calculate score inline for performance
            # Base score from length
            if seq_len > self.LONG_SEQUENCE_THRESHOLD:
                score = 0.8
            elif seq_len > self.MEDIUM_SEQUENCE_THRESHOLD:
                score = 0.7
            else:
                score = 0.6

            # Check base distribution - count using string method
            # Combine T+U count for pyrimidine balance
            counts = (
                sequence_upper.count("A"),
                sequence_upper.count("T") + sequence_upper.count("U"),
                sequence_upper.count("G"),
                sequence_upper.count("C"),
            )
            if max(counts) / seq_len < self.BASE_DIVERSITY_THRESHOLD:
                score += 0.1

            # Context bonus - check chars before/after using regex pos/endpos
            context_start = start - self.CONTEXT_WINDOW_SIZE if start > self.CONTEXT_WINDOW_SIZE else 0
            context_end = end + self.CONTEXT_WINDOW_SIZE if end + self.CONTEXT_WINDOW_SIZE < text_len else text_len
            if self._context_pattern.search(text, context_start, context_end):
                score += 0.1

            # Cap at 0.95
            score = min(score, 0.95)

            results.append(
                RecognizerResult(
                    entity_type=self.GENETIC_SEQUENCE_ENTITY,
                    start=start,
                    end=end,
                    score=score,
                    analysis_explanation=AnalysisExplanation(
                        recognizer=self.__class__.__name__,
                        original_score=score,
                        textual_explanation=f"Genetic sequence ({seq_len} bases)",
                        pattern="Nucleotide sequence (A/T/G/C/U)",
                    ),
                )
            )

        return results
