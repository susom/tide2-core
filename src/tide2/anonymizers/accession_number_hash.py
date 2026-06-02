"""
Anonymizer for accession numbers using a deterministic hashing algorithm.

This implements the identifier_hashing_algorithm compatible with the BigQuery function:

    CREATE TEMP FUNCTION identifier_hashing_algorithm(identifier STRING, salt STRING, study_id STRING, entity STRING)
    RETURNS STRING AS (
        LEFT(UPPER(TO_HEX(SHA256(CONCAT(
            COALESCE(UPPER(TRIM(salt)), '[S]'), '|',
            COALESCE(UPPER(TRIM(study_id)), '[U]'), '|',
            COALESCE(UPPER(TRIM(entity)), '[E]'), '|',
            UPPER(TRIM(identifier))
        )))),16)
    );
"""

from hashlib import sha256

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType


class AccessionNumberHashAnonymizer(Operator):
    """
    Anonymizer that replaces accession numbers with a deterministic hash.

    Uses SHA256 hashing with salt, study_id, and entity parameters to produce
    a 16-character uppercase hexadecimal identifier.

    This implementation is compatible with the BigQuery identifier_hashing_algorithm
    function, ensuring consistency between anonymization in Python and SQL.

    Parameters:
        salt (str, optional): Salt value for hashing. Defaults to '[S]' if None.
        study_id (str, optional): Study identifier. Defaults to '[U]' if None.
        entity_type (str): The entity type being anonymized (e.g., 'ACC_NUM').
            Defaults to '[E]' if None.
    """

    # Default values matching the SQL COALESCE behavior
    DEFAULT_SALT = "[S]"
    DEFAULT_STUDY_ID = "[U]"
    DEFAULT_ENTITY = "[E]"

    def __init__(self):
        """Initialize the accession number hash anonymizer."""
        super().__init__()

        self.entities_supported = {
            "DEFAULT",
            "ACC_NUM",
            "ACCESSION_NUMBER",
        }

    def _coalesce_param(self, value: str | None, default: str) -> str:
        """
        Mimic SQL COALESCE(UPPER(TRIM(value)), default) behavior.

        In SQL, COALESCE only replaces NULL values, not empty strings.
        So COALESCE(UPPER(TRIM('')), '[S]') returns '' (empty string), not '[S]'.

        Args:
            value: The input value (may be None)
            default: The default value to use if value is None

        Returns:
            Uppercase trimmed value, or default if value is None
        """
        if value is None:
            return default
        return value.strip().upper()

    def operate(self, text: str, params: dict) -> str:
        """
        Anonymize the accession number using deterministic hashing.

        The algorithm concatenates salt, study_id, entity, and identifier with '|'
        separator, applies SHA256, and returns the first 16 characters of the
        uppercase hex digest.

        Args:
            text: The accession number to anonymize
            params: Dictionary containing optional 'salt', 'study_id', and 'entity_type'

        Returns:
            16-character uppercase hexadecimal hash
        """
        salt = params.get("salt")
        study_id = params.get("study_id")
        entity = params.get("entity_type")

        # Apply COALESCE logic matching the SQL function
        salt_part = self._coalesce_param(salt, self.DEFAULT_SALT)
        study_id_part = self._coalesce_param(study_id, self.DEFAULT_STUDY_ID)
        entity_part = self._coalesce_param(entity, self.DEFAULT_ENTITY)
        identifier_part = text.strip().upper() if text else ""

        # Concatenate with '|' separator
        concat_string = f"{salt_part}|{study_id_part}|{entity_part}|{identifier_part}"

        # SHA256 hash -> hex -> uppercase -> first 16 characters
        hash_digest = sha256(concat_string.encode("utf-8")).hexdigest().upper()

        return hash_digest[:16]

    def validate(self, params: dict) -> None:
        """
        Validate operator parameters.

        Args:
            params: Dictionary that may contain 'entity_type', 'salt', 'study_id'

        Raises:
            ValueError: If entity_type is provided but not supported
        """
        entity_type = params.get("entity_type", "DEFAULT")
        if entity_type not in self.entities_supported:
            raise ValueError(
                f"Entity type '{entity_type}' is not supported for "
                f"AccessionNumberHashAnonymizer. Supported types: {self.entities_supported}"
            )

    def operator_name(self) -> str:
        """Return the operator name."""
        return "accession_number_hash"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize
