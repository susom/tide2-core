"""Format-preserving encryption operator for alphanumeric identifiers.

Uses FF3-based FPE to deterministically encrypt identifiers (MRNs, accession
numbers, etc.) while preserving their character format and length.
"""

from functools import lru_cache

from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType

from tide2.cryptographic.fpe_strings import FormatPreservingEncryption


@lru_cache(maxsize=16)
def _get_cached_fpe(salt: bytes, key: bytes) -> FormatPreservingEncryption:
    """
    Get a cached FormatPreservingEncryption instance.

    Cache is bounded to maxsize=16 to match typical core count.
    This is process-local, which is appropriate for Ray worker-local usage.

    Args:
        salt: 32-byte salt
        key: 32-byte key

    Returns:
        Cached FormatPreservingEncryption instance
    """
    return FormatPreservingEncryption(salt, key)


class HipsAlphaNumericAnonymizer(Operator):
    """
    Anonymizer that replaces alphanumeric identifiers with format-preserving encryption.

    Uses cached FPE instances for performance. The cache is process-local,
    which is appropriate for Ray worker-local usage patterns.
    """

    def __init__(self):
        """Initialize the HIPS alphanumeric anonymizer."""
        super().__init__()

        self.entities_supported = set(
            [
                "DEFAULT",
                "PHONE",
                "PHONE_NUMBER",
                "US_SSN",
                "MEDICAL_LICENSE",
                "MRN",
                "HAR",
                "ACC_NUM",
                "ID",
                "CSN_ID",
            ]
        )

    def operate(self, text: str, params: dict) -> str:
        """Anonymize the input text using format-preserving encryption.

        Args:
            text: The original alphanumeric text (e.g. MRN, accession number).
            params: Operator parameters. Required keys:
                - salt (str): Cryptographic salt for deterministic output.
                - key (str): Encryption key for FPE.

        Returns:
            The encrypted text preserving the original character format.
        """

        salt = params["salt"]
        key = params["key"]

        # Use cached FPE instance for performance
        fpe = _get_cached_fpe(salt, key)

        new_text, _type_recognized = fpe.encrypt(text)

        return new_text

    def validate(self, params: dict) -> None:
        """Validate operator parameters."""

        entity_type = params.get("entity_type", "DEFAULT")
        if entity_type not in self.entities_supported:
            raise ValueError(f"Entity type '{entity_type}' is not supported for HipsAlphaNumericAnonymizer.")

        # get the salt and key
        salt = params.get("salt")
        key = params.get("key")
        if not salt or not key:
            raise ValueError("Both 'salt' and 'key' must be provided for HipsAlphaNumericAnonymizer.")

    def operator_name(self) -> str:
        """Return the operator name."""
        return "hips_alphanumeric"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize


def clear_fpe_cache() -> None:
    """
    Clear the FPE instance cache.

    Call this when changing keys or when memory pressure is a concern.
    """
    _get_cached_fpe.cache_clear()


def get_fpe_cache_info() -> dict:
    """
    Get FPE cache statistics for monitoring.

    Returns:
        Dictionary with cache info
    """
    return _get_cached_fpe.cache_info()._asdict()
