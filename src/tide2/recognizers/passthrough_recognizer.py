"""No-op recognizer for ablation studies.

Implements the ``EntityRecognizer`` interface but never detects entities,
useful for isolating the effect of other recognizers in experiments.
"""

from presidio_analyzer import EntityRecognizer
from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts


class PassthroughRecognizer(EntityRecognizer):
    """
    A passthrough recognizer that implements the EntityRecognizer interface
    but never detects any entities. This is useful for ablation studies to
    ensure no unintended recognizers are active while maintaining the
    registry initialization structure.

    This recognizer can be safely added to the recognizer registry and will
    not interfere with the analysis pipeline while allowing the system to
    initialize properly.
    """

    ENTITY_TYPE = "PASSTHROUGH"

    def __init__(self, supported_entity: str = "PASSTHROUGH", supported_language: str = "en"):
        """
        Initialize the passthrough recognizer.

        Args:
            supported_entity: The entity type this recognizer claims to support
            supported_language: The language this recognizer supports
        """
        super().__init__(
            supported_entities=[supported_entity], supported_language=supported_language, name="PassthroughRecognizer"
        )

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        """
        Analyze text for entities but always return an empty list.

        This method implements the required analyze interface but deliberately
        returns no results, making it a true passthrough that detects nothing.

        Args:
            text: The text to analyze
            entities: List of entity types to look for
            nlp_artifacts: NLP artifacts from previous analysis stages

        Returns:
            Empty list - no entities are ever detected
        """
        # Always return empty list - this recognizer detects nothing
        return []

    def load(self) -> None:
        """
        Load method required by the EntityRecognizer interface.

        This is a no-op for the passthrough recognizer since there's
        nothing to load.
        """
        pass

    def get_supported_entities(self) -> list[str]:
        """
        Get the list of supported entities.

        Returns:
            List containing the single entity type this recognizer claims to support
        """
        return self.supported_entities
