"""Cell-level anonymization of one structured value.

``anonymize_cell`` gives the result ``AnonymizerWorker.process_note`` gives for the value as a one-span
note of the given entity type, without a Presidio engine. Backends that anonymize table columns (Ray,
BigQuery) call it once per cell.

This module is shipped on its own in the BigQuery UDF bundle, where ``tide2`` and ``tide2.anonymizers``
are namespace packages (their ``__init__.py`` files are not bundled). It must import sibling modules by
full module path, for example ``from tide2.anonymizers.guarded import guarded``, never from a package
root, and must not import heavy modules (``faker``, ``usaddress``, ``tide2.actors``).
``tests/test_cell_isolation.py`` enforces this.
"""

from __future__ import annotations

import functools
import secrets
from typing import Any

from presidio_anonymizer.operators import Operator

from tide2.anonymizers.date_jitter import DateJitterAnonymizer
from tide2.anonymizers.guarded import guarded
from tide2.anonymizers.guarded import mask
from tide2.anonymizers.guarded import safe_entity_type
from tide2.anonymizers.hips_alphanumeric import HipsAlphaNumericAnonymizer
from tide2.cryptographic.date_jitter import derive_date_jitter
from tide2.cryptographic.fpe_strings import FormatPreservingEncryption
from tide2.utils.nulls import is_null

__docformat__ = "google"

DEFAULT_MAX_JITTER_DAYS = 180
DEFAULT_MIN_JITTER_DAYS = 3
RANDOM_JITTER_SPAN = 357
RANDOM_JITTER_OFFSET = 178
_JITTER_CACHE_SIZE = 65536

FPE_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "MRN",
        "HAR",
        "ID",
        "CSN_ID",
        "US_SSN",
        "PHONE_NUMBER",
        "PHONE",
        "MEDICAL_LICENSE",
    }
)
DATE_ENTITY_TYPES: frozenset[str] = frozenset({"DATE", "DATE_TIME"})
SUPPORTED_ENTITY_TYPES: frozenset[str] = FPE_ENTITY_TYPES | DATE_ENTITY_TYPES


@functools.cache
def _operators() -> tuple[Operator, Operator]:
    """Return the guarded ``(fpe, date_jitter)`` operator instances, built once per process.

    Keys travel in the per-call params, so the instances do not depend on them.
    """
    FormatPreservingEncryption.suppress_short_input_warnings = True
    return guarded(HipsAlphaNumericAnonymizer)(), guarded(DateJitterAnonymizer)()


@functools.lru_cache(maxsize=_JITTER_CACHE_SIZE)
def _derived_jitter(patient_id: str, salt: bytes, key: bytes) -> int:
    """Return the deterministic jitter for a patient; cached because it is a pure function."""
    return derive_date_jitter(
        patient_id=patient_id,
        salt=salt,
        key=key,
        max_jitter_days=DEFAULT_MAX_JITTER_DAYS,
        min_jitter_days=DEFAULT_MIN_JITTER_DAYS,
    )


def compute_jitter_for_patient(patient_id: Any, salt: bytes, key: bytes) -> int:
    """Return a patient's date jitter in days, or a random one when the patient is null.

    Mirrors ``AnonymizerWorker._compute_jitter_for_patient``.

    Args:
        patient_id: Patient identifier, or a null value.
        salt: 32-byte salt.
        key: 32-byte key.

    Returns:
        Integer jitter in days.
    """
    if is_null(patient_id):
        return secrets.randbelow(RANDOM_JITTER_SPAN) - RANDOM_JITTER_OFFSET
    patient = str(patient_id).strip()
    if patient.lower() in ("nan", "none", "null", ""):
        return secrets.randbelow(RANDOM_JITTER_SPAN) - RANDOM_JITTER_OFFSET
    return _derived_jitter(patient, salt, key)


def anonymize_cell(
    value: Any,
    entity_type: str | None,
    salt: bytes,
    key: bytes,
    patient_id: Any = None,
    jitter: int | None = None,
) -> str | None:
    """Anonymize one cell as a one-span note of ``entity_type``.

    Args:
        value: Cell value, or a null value.
        entity_type: One of ``SUPPORTED_ENTITY_TYPES``.
        salt: 32-byte salt.
        key: 32-byte key.
        patient_id: Patient identifier used to derive the date jitter.
        jitter: Explicit date jitter in days; wins over ``patient_id``.

    Returns:
        The anonymized string, ``None`` for a null value, or the entity mask (for example ``[DATE]``)
        when the operator cannot handle the value.

    Raises:
        ValueError: If ``entity_type`` is missing or not supported.
    """
    if not isinstance(entity_type, str) or entity_type not in SUPPORTED_ENTITY_TYPES:
        raise ValueError(f"Unsupported entity_type: {entity_type!r}")

    if is_null(value):
        return None

    text = str(value)
    if text == "":
        return ""

    # process_note relabels a type equal to its own span text as ENTITY, which falls to the DEFAULT mask.
    if safe_entity_type(entity_type, text) != entity_type:
        return mask("ENTITY")

    fpe, date_jitter = _operators()
    if entity_type in DATE_ENTITY_TYPES:
        resolved = compute_jitter_for_patient(patient_id, salt, key) if is_null(jitter) else int(jitter)  # type: ignore[arg-type]
        operator, params = date_jitter, {"jitter": resolved}
    else:
        operator, params = fpe, {"salt": salt, "key": key}
    params["entity_type"] = entity_type

    # The guarded operator validates inside operate() and masks on error.
    return operator.operate(text, params)
