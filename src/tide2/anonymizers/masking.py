"""Masking anonymizer that replaces entities with their type label.

Replaces detected PII entities with [<entity_type>], e.g. "John Smith" -> "[PERSON]".
"""

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType

from tide2.anonymizers.guarded import mask
from tide2.anonymizers.guarded import record_fallback


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
                When 'fallback' is true the mask stands in for an operator that has
                no path for this entity, and is recorded as a fallback.

        Returns:
            String in the format [<entity_type>], e.g. [PERSON].
        """
        entity_type = params.get("entity_type", "UNKNOWN")
        if params.get("fallback"):
            record_fallback(entity_type)
        return mask(entity_type)

    def validate(self, params: dict) -> None:
        """Validate operator parameters. Accepts all entity types."""
        pass

    def operator_name(self) -> str:
        """Return the operator name."""
        return "masking"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize
