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
import logging
from typing import Any

from presidio_anonymizer.operators import Operator

from tide2.anonymizers.date_jitter import DateJitterAnonymizer
from tide2.anonymizers.guarded import guarded
from tide2.anonymizers.guarded import mask
from tide2.anonymizers.guarded import safe_entity_type
from tide2.anonymizers.hips_alphanumeric import HipsAlphaNumericAnonymizer
from tide2.cryptographic.date_jitter import DEFAULT_MAX_AGE
from tide2.cryptographic.date_jitter import DEFAULT_MAX_JITTER_DAYS
from tide2.cryptographic.date_jitter import DEFAULT_MIN_JITTER_DAYS
from tide2.cryptographic.date_jitter import UNCAPPED_DATES_WARNING
from tide2.cryptographic.date_jitter import resolve_date_jitter
from tide2.cryptographic.fpe_strings import FormatPreservingEncryption
from tide2.utils.nulls import is_null
from tide2.utils.nulls import is_null_or_blank

__docformat__ = "google"

logger = logging.getLogger(__name__)

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
DATE_PART_ENTITY_TYPES: frozenset[str] = frozenset({"DATE_DAY", "DATE_MONTH", "DATE_YEAR"})
SUPPORTED_ENTITY_TYPES: frozenset[str] = FPE_ENTITY_TYPES | DATE_ENTITY_TYPES | DATE_PART_ENTITY_TYPES


@functools.cache
def _operators() -> tuple[Operator, Operator]:
    """Return the guarded ``(fpe, date_jitter)`` operator instances, built once per process.

    Keys travel in the per-call params, so the instances do not depend on them.
    """
    FormatPreservingEncryption.suppress_short_input_warnings = True
    return guarded(HipsAlphaNumericAnonymizer)(), guarded(DateJitterAnonymizer)()


@functools.cache
def _warn_no_birth_date() -> None:
    """Log the missing-birth-date warning once per process."""
    logger.warning("anonymize_cell called without birth_date: %s", UNCAPPED_DATES_WARNING)


def compute_jitter_for_patient(
    patient_id: Any,
    salt: bytes,
    key: bytes,
    max_jitter_days: int = DEFAULT_MAX_JITTER_DAYS,
    min_jitter_days: int = DEFAULT_MIN_JITTER_DAYS,
) -> int:
    """Return a patient's date jitter in days, or a random one when the patient is null.

    Mirrors ``AnonymizerWorker._compute_jitter_for_patient``.

    Args:
        patient_id: Patient identifier, or a null value.
        salt: 32-byte salt.
        key: 32-byte key.
        max_jitter_days: Maximum absolute jitter in days.
        min_jitter_days: Minimum absolute jitter in days.

    Returns:
        Integer jitter in days.
    """
    return resolve_date_jitter(patient_id, salt, key, max_jitter_days, min_jitter_days)


def anonymize_cell(
    value: Any,
    entity_type: str | None,
    salt: bytes,
    key: bytes,
    patient_id: Any = None,
    jitter: int | None = None,
    max_jitter_days: int = DEFAULT_MAX_JITTER_DAYS,
    min_jitter_days: int = DEFAULT_MIN_JITTER_DAYS,
    birth_date: Any = None,
    max_age: int = DEFAULT_MAX_AGE,
    full_date: Any = None,
) -> str | None:
    """Anonymize one cell as a one-span note of ``entity_type``.

    Args:
        value: Cell value, or a null value.
        entity_type: One of ``SUPPORTED_ENTITY_TYPES``.
        salt: 32-byte salt.
        key: 32-byte key.
        patient_id: Patient identifier used to derive the date jitter.
        jitter: Explicit date jitter in days; wins over ``patient_id``.
        max_jitter_days: Maximum absolute derived or random jitter in days.
        min_jitter_days: Minimum absolute derived or random jitter in days.
        birth_date: Real birth date of the patient. Dates after the ``max_age``-th
            birthday are moved to it; without it dates are not capped (a warning is
            logged once per process).
        max_age: Age at which dates are capped.
        full_date: Real full date of the row. Required for the part types
            (``DATE_DAY``, ``DATE_MONTH``, ``DATE_YEAR``), where ``value`` is the day,
            month or year held on its own and the result is that part of the shifted
            ``full_date``; a null ``full_date`` gives ``[DATE]``. Ignored otherwise. To keep
            the parts of a row consistent with each other and with the full-date column,
            pass the same non-null ``patient_id`` (and settings) or the same ``jitter``
            to every call; without either, each call draws a different random shift.

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

    if entity_type in DATE_PART_ENTITY_TYPES and isinstance(value, float) and value.is_integer():
        # A nullable integer column read as float holds 3.0, not 3.
        value = int(value)
    text = str(value)
    if text == "":
        return ""

    # process_note relabels a type equal to its own span text as ENTITY, which falls to the DEFAULT mask.
    if safe_entity_type(entity_type, text) != entity_type:
        return mask("ENTITY")

    fpe, date_jitter = _operators()
    if entity_type in DATE_ENTITY_TYPES | DATE_PART_ENTITY_TYPES:
        if is_null(jitter):
            resolved = resolve_date_jitter(patient_id, salt, key, max_jitter_days, min_jitter_days)
        else:
            resolved = int(jitter)  # type: ignore[arg-type]
        operator, params = date_jitter, {"jitter": resolved}
        if is_null_or_blank(birth_date):
            _warn_no_birth_date()
        else:
            params["birth_date"] = birth_date
            params["max_age"] = max_age
        if entity_type in DATE_PART_ENTITY_TYPES:
            params["full_date"] = full_date
    else:
        operator, params = fpe, {"salt": salt, "key": key}
    params["entity_type"] = entity_type

    # The guarded operator validates inside operate() and masks on error.
    return operator.operate(text, params)
