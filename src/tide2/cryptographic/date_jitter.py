"""
Deterministic date jitter derivation for anonymization.

This module provides cryptographically secure, deterministic date jitter
derivation based on a patient identifier and encryption keys. The jitter
is consistent for the same patient across multiple runs, ensuring
reproducible anonymization.

Usage:
    from tide2.cryptographic.date_jitter import derive_date_jitter

    jitter = derive_date_jitter(
        patient_id="12345",
        salt=salt_bytes,
        key=key_bytes,
        max_jitter_days=180,
        min_jitter_days=3,
    )
    # Returns an integer between -180 and +180 (excluding -2 to +2)
"""

import functools
import hashlib
import hmac
import secrets
import struct
from typing import Any

from tide2.utils.nulls import is_null

# Default jitter range constants
DEFAULT_MAX_JITTER_DAYS = 180
DEFAULT_MIN_JITTER_DAYS = 3
ABSOLUTE_MIN_JITTER_DAYS = 3
_JITTER_CACHE_SIZE = 65536
_NULL_PATIENT_STRINGS = frozenset({"nan", "none", "null", ""})
# Dates after this birthday are moved to it (HIPAA Safe Harbor "90 or older").
DEFAULT_MAX_AGE = 90
_MIN_MAX_AGE = 1
_MAX_MAX_AGE = 150
# Logged by every caller that shifts dates without a birth date; formatted with max_age.
NO_BIRTH_DATE_WARNING = "dates are shifted but not capped at age %d; ages over 89 may be revealed"

# Key size requirements
REQUIRED_KEY_SIZE = 32


def derive_date_jitter(
    patient_id: str,
    salt: bytes,
    key: bytes,
    max_jitter_days: int = DEFAULT_MAX_JITTER_DAYS,
    min_jitter_days: int = DEFAULT_MIN_JITTER_DAYS,
) -> int:
    """
    Derive a deterministic date jitter value for a patient.

    Uses HMAC-SHA256 to derive a cryptographically secure, deterministic
    jitter value that is consistent for the same patient across runs.
    The jitter will be in the range [-max_jitter_days, -min_jitter_days] or
    [min_jitter_days, max_jitter_days].

    Args:
        patient_id: Unique patient identifier.
        salt: 32-byte salt for HMAC derivation.
        key: 32-byte key used as additional entropy.
        max_jitter_days: Maximum absolute jitter in days (default: 180).
        min_jitter_days: Minimum absolute jitter in days (default: 3).
            Must be at least 3 for HIPAA compliance.

    Returns:
        Integer jitter value in days. Will be either negative
        (in range [-max_jitter_days, -min_jitter_days]) or positive
        (in range [min_jitter_days, max_jitter_days]).

    Raises:
        ValueError: If keys are invalid or jitter parameters are out of range.
        TypeError: If patient_id is not a string.

    Example:
        >>> salt = bytes.fromhex("0" * 64)
        >>> key = bytes.fromhex("1" * 64)
        >>> jitter1 = derive_date_jitter("patient_123", salt, key)
        >>> jitter2 = derive_date_jitter("patient_123", salt, key)
        >>> jitter1 == jitter2  # Same patient = same jitter
        True
        >>> abs(jitter1) >= 3  # At least 3 days
        True
    """
    # Validate inputs
    if not isinstance(patient_id, str):
        raise TypeError(f"patient_id must be a string, got {type(patient_id).__name__}")

    if not patient_id:
        raise ValueError("patient_id cannot be empty")

    if not isinstance(salt, bytes) or len(salt) != REQUIRED_KEY_SIZE:
        raise ValueError(f"salt must be exactly {REQUIRED_KEY_SIZE} bytes")

    if not isinstance(key, bytes) or len(key) != REQUIRED_KEY_SIZE:
        raise ValueError(f"key must be exactly {REQUIRED_KEY_SIZE} bytes")

    if min_jitter_days < ABSOLUTE_MIN_JITTER_DAYS:
        raise ValueError(
            f"min_jitter_days must be at least {ABSOLUTE_MIN_JITTER_DAYS} for HIPAA compliance, got {min_jitter_days}"
        )

    if max_jitter_days < min_jitter_days:
        raise ValueError(f"max_jitter_days ({max_jitter_days}) must be >= min_jitter_days ({min_jitter_days})")

    # Create the HMAC key by combining salt and key
    # This ensures both contribute to the derivation
    combined_key = salt + key

    # Create the message to hash
    # Include context to prevent cross-purpose key reuse
    context = b"date_jitter_v1:"
    message = context + patient_id.encode("utf-8")

    # Compute HMAC-SHA256
    hmac_digest = hmac.new(
        key=combined_key,
        msg=message,
        digestmod=hashlib.sha256,
    ).digest()

    # Extract two values from the digest:
    # 1. First 4 bytes for magnitude (0 to max_jitter_days - min_jitter_days)
    # 2. Next byte for sign (positive or negative)
    magnitude_bytes = hmac_digest[:4]
    sign_byte = hmac_digest[4]

    # Convert magnitude bytes to unsigned integer
    magnitude_raw = struct.unpack(">I", magnitude_bytes)[0]

    # Calculate the range of valid magnitudes
    # Valid range: [min_jitter_days, max_jitter_days]
    jitter_range = max_jitter_days - min_jitter_days + 1

    # Map to the valid magnitude range
    magnitude = (magnitude_raw % jitter_range) + min_jitter_days

    # Determine sign (positive or negative)
    # Use the sign byte to decide direction
    is_positive = (sign_byte % 2) == 0

    # Return the final jitter value
    return magnitude if is_positive else -magnitude


def random_date_jitter(
    max_jitter_days: int = DEFAULT_MAX_JITTER_DAYS,
    min_jitter_days: int = DEFAULT_MIN_JITTER_DAYS,
) -> int:
    """
    Return a random date jitter with ``min_jitter_days <= abs(jitter) <= max_jitter_days``.

    Uses ``secrets``. The magnitude is uniform in the range and the sign is uniform, the same
    distribution ``derive_date_jitter`` produces, so no value falls below the minimum.

    Args:
        max_jitter_days: Maximum absolute jitter in days (default: 180).
        min_jitter_days: Minimum absolute jitter in days (default: 3, the lowest allowed).

    Returns:
        Integer jitter in days.

    Raises:
        ValueError: If the range is invalid.
    """
    is_valid, message = validate_jitter_parameters(max_jitter_days, min_jitter_days)
    if not is_valid:
        raise ValueError(message)
    magnitude = min_jitter_days + secrets.randbelow(max_jitter_days - min_jitter_days + 1)
    return magnitude if secrets.randbelow(2) == 0 else -magnitude


@functools.lru_cache(maxsize=_JITTER_CACHE_SIZE)
def _cached_derive(patient_id: str, salt: bytes, key: bytes, max_jitter_days: int, min_jitter_days: int) -> int:
    return derive_date_jitter(patient_id, salt, key, max_jitter_days, min_jitter_days)


def resolve_date_jitter(
    patient_id: Any,
    salt: bytes,
    key: bytes,
    max_jitter_days: int = DEFAULT_MAX_JITTER_DAYS,
    min_jitter_days: int = DEFAULT_MIN_JITTER_DAYS,
) -> int:
    """
    Return a patient's date jitter: derived when the patient is known, random when it is null.

    A null patient is ``None``, NaN, or a string that is blank, ``nan``, ``none`` or ``null``
    (case-insensitive, after stripping). Both paths use the same range.

    Args:
        patient_id: Patient identifier, or a null value.
        salt: 32-byte salt.
        key: 32-byte key.
        max_jitter_days: Maximum absolute jitter in days (default: 180).
        min_jitter_days: Minimum absolute jitter in days (default: 3).

    Returns:
        Integer jitter in days.

    Raises:
        ValueError: If the range or the keys are invalid.
    """
    if not is_null(patient_id):
        patient = str(patient_id).strip()
        if patient.lower() not in _NULL_PATIENT_STRINGS:
            return _cached_derive(patient, salt, key, max_jitter_days, min_jitter_days)
    return random_date_jitter(max_jitter_days, min_jitter_days)


def derive_date_jitter_batch(
    patient_ids: list[str],
    salt: bytes,
    key: bytes,
    max_jitter_days: int = DEFAULT_MAX_JITTER_DAYS,
    min_jitter_days: int = DEFAULT_MIN_JITTER_DAYS,
) -> list[int]:
    """
    Derive date jitter values for multiple patients.

    Convenience function for batch processing. Each patient's jitter
    is derived independently using derive_date_jitter().

    Args:
        patient_ids: List of unique patient identifiers.
        salt: 32-byte salt for HMAC derivation.
        key: 32-byte key used as additional entropy.
        max_jitter_days: Maximum absolute jitter in days (default: 180).
        min_jitter_days: Minimum absolute jitter in days (default: 3).

    Returns:
        List of integer jitter values, one per patient.

    Example:
        >>> jitters = derive_date_jitter_batch(
        ...     ["patient_1", "patient_2", "patient_3"],
        ...     salt,
        ...     key,
        ... )
        >>> len(jitters)
        3
    """
    return [
        derive_date_jitter(
            patient_id=pid,
            salt=salt,
            key=key,
            max_jitter_days=max_jitter_days,
            min_jitter_days=min_jitter_days,
        )
        for pid in patient_ids
    ]


def validate_jitter_parameters(
    max_jitter_days: int,
    min_jitter_days: int,
) -> tuple[bool, str | None]:
    """
    Validate jitter parameters without raising exceptions.

    Useful for configuration validation before processing.

    Args:
        max_jitter_days: Maximum absolute jitter in days.
        min_jitter_days: Minimum absolute jitter in days.

    Returns:
        Tuple of (is_valid, error_message).
        If valid, returns (True, None).
        If invalid, returns (False, error_description).

    Example:
        >>> validate_jitter_parameters(180, 3)
        (True, None)
        >>> validate_jitter_parameters(180, 1)
        (False, 'min_jitter_days must be at least 3 for HIPAA compliance, got 1')
    """
    if min_jitter_days < ABSOLUTE_MIN_JITTER_DAYS:
        return (
            False,
            f"min_jitter_days must be at least {ABSOLUTE_MIN_JITTER_DAYS} for HIPAA compliance, got {min_jitter_days}",
        )

    if max_jitter_days < min_jitter_days:
        return (
            False,
            f"max_jitter_days ({max_jitter_days}) must be >= min_jitter_days ({min_jitter_days})",
        )

    return (True, None)


def validate_max_age(value: Any) -> int:
    """Return ``value`` as an int age between 1 and 150.

    Raises:
        ValueError: If ``value`` is not an integer in range.
    """
    max_age = int(value)
    if not _MIN_MAX_AGE <= max_age <= _MAX_MAX_AGE:
        raise ValueError(f"max_age must be between {_MIN_MAX_AGE} and {_MAX_MAX_AGE}, got {max_age}.")
    return max_age
