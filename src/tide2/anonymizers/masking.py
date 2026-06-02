"""Masking anonymizer that replaces entities with their type label.

Replaces detected PII entities with [<entity_type>], e.g. "John Smith" -> "[PERSON]".
"""

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType


class MaskingAnonymizer(Operator):
    """Anonymizer that replaces entities with [<entity_type>] labels."""

    def __init__(self):
        """Initialize the masking anonymizer."""
        super().__init__()

    def operate(self, text: str, params: dict) -> str:
        """Replace the entity text with its type label.

        Args:
            text: The original text containing the entity.
            params: Operator parameters. Uses 'entity_type' to determine the label.

        Returns:
            String in the format [<entity_type>], e.g. [PERSON].
        """
        entity_type = params.get("entity_type", "UNKNOWN")
        return f"[{entity_type}]"

    def validate(self, params: dict) -> None:
        """Validate operator parameters. Accepts all entity types."""
        pass

    def operator_name(self) -> str:
        """Return the operator name."""
        return "masking"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize
