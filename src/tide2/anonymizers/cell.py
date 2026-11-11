"""Cell-level anonymization of one structured value.

``anonymize_cell`` gives the result ``AnonymizerWorker.process_note`` gives for the value as a one-span
note of the given entity type, without a Presidio engine. This holds for any policy that is given to both.
Backends that anonymize table columns (Ray, BigQuery) call it once per cell.

This module is shipped on its own in the BigQuery UDF bundle, where ``tide2`` and ``tide2.anonymizers``
are namespace packages (their ``__init__.py`` files are not bundled). It must import sibling modules by
full module path, for example ``from tide2.anonymizers.guarded import guarded``, never from a package
root, and must not import heavy modules (``faker``, ``usaddress``, ``tide2.actors``) at load time.
The ``FREE_TEXT`` type imports ``tide2.anonymizers.free_text`` (and through it the actors and the
operators) only when such a cell is anonymized.
``tests/test_cell_isolation.py`` enforces this.
"""

from __future__ import annotations

import functools
import logging
import numbers
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from typing import Any

from presidio_anonymizer.operators import Operator

from tide2.anonymizers.accession_number_hash import AccessionNumberHashAnonymizer
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
from tide2.cryptographic.date_jitter import validate_jitter_parameters
from tide2.cryptographic.date_jitter import validate_max_age
from tide2.cryptographic.fpe_strings import FPE_MODES
from tide2.cryptographic.fpe_strings import MODE_STRICT
from tide2.utils.nulls import is_null
from tide2.utils.nulls import is_null_or_blank

__docformat__ = "google"

logger = logging.getLogger(__name__)

FPE_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "MRN",
        "HAR",
        "ACC_NUM",
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
FREE_TEXT_ENTITY_TYPES: frozenset[str] = frozenset({"FREE_TEXT"})
ZIP_ENTITY_TYPES: frozenset[str] = frozenset({"ZIP_CODE"})
SUPPORTED_ENTITY_TYPES: frozenset[str] = (
    FPE_ENTITY_TYPES | DATE_ENTITY_TYPES | DATE_PART_ENTITY_TYPES | FREE_TEXT_ENTITY_TYPES | ZIP_ENTITY_TYPES
)
# Types a run may leave as they are: every type of the free-text operator map except DEFAULT, plus the dates.
KEEPABLE_ENTITY_TYPES: frozenset[str] = (
    FPE_ENTITY_TYPES
    | DATE_ENTITY_TYPES
    | DATE_PART_ENTITY_TYPES
    | frozenset(
        {
            "PATIENT",
            "DOCTOR",
            "HCW",
            "PERSON",
            "HOSPITAL",
            "LOCATION",
            "AGE",
            "WEB",
            "OTHER",
            "VENDOR",
            "EMAIL_ADDRESS",
            "URL",
            "BASE64_IMAGE",
            "GENETIC_SEQUENCE",
            "ORGANIZATION",
            "ZIP_CODE",
            "ZIP",
            "ZIPCODE",
        }
    )
)


def parse_keep_entities(value: str | Iterable[str] | None) -> frozenset[str]:
    """Return the entity types to leave as they are, from a comma-separated string or a list.

    Names are stripped and upper-cased. ``DATE`` also covers ``DATE_DAY``, ``DATE_MONTH`` and
    ``DATE_YEAR``; ``DATE_TIME`` is listed on its own.

    Args:
        value: ``None``, a comma-separated string, or an iterable of names.

    Returns:
        The set of entity types to keep; empty for ``None`` or an empty value.

    Raises:
        ValueError: If a name is not in ``KEEPABLE_ENTITY_TYPES``.
    """
    if value is None:
        return frozenset()
    names = value.split(",") if isinstance(value, str) else value
    kept = {str(name).strip().upper() for name in names if str(name).strip()}
    unknown = sorted(kept - KEEPABLE_ENTITY_TYPES)
    if unknown:
        raise ValueError(f"Unknown keep_entities {unknown}; valid types: {sorted(KEEPABLE_ENTITY_TYPES)}")
    if "DATE" in kept:
        kept |= DATE_PART_ENTITY_TYPES
    return frozenset(kept)


MAX_ZIP5 = 99999
_warned_integer_zip = False


@functools.cache
def _zip_operator() -> Operator:
    """Return the guarded ZIP operator, built once per process.

    ``zip_code`` is imported here, not at module level, so a bundle that has no ZIP module or population file
    still runs every other entity type except ``FREE_TEXT``.
    """
    from tide2.anonymizers.zip_code import ZipCodeAnonymizer

    return guarded(ZipCodeAnonymizer)()


def _zip_text(value: Any) -> str:
    """Return the text of a ZIP cell, restoring the leading zeros an integer column has lost.

    A ZIP stored as an integer (``2139`` for ``02139``) is left-padded to 5 digits and a warning is logged once
    per process. Any other value is returned as text; a value that is not a ZIP is masked by the operator.
    """
    global _warned_integer_zip  # noqa: PLW0603 # one warning per process
    if isinstance(value, numbers.Integral) and not isinstance(value, bool) and 0 <= value <= MAX_ZIP5:
        if not _warned_integer_zip:
            _warned_integer_zip = True
            logger.warning(
                "ZIP code given as an integer: padding to 5 digits. The column lost its leading zeros upstream; "
                "store ZIP codes as strings."
            )
        return f"{int(value):05d}"
    return str(value)


@dataclass(frozen=True)
class AnonymizationPolicy:
    """Anonymization options of a run, checked one time when the policy is made.

    Secrets (``salt``, ``key``, ``acc_num_salt``) and row context (``patient_id``, ``jitter``, ...) are not
    part of it. A frozen policy is hashable, so it can be a cache key.

    Raises:
        ValueError: If a value is out of range, a name is unknown, or one entity type gets two options.
    """

    min_jitter_days: int = DEFAULT_MIN_JITTER_DAYS
    max_jitter_days: int = DEFAULT_MAX_JITTER_DAYS
    max_age: int = DEFAULT_MAX_AGE
    keep_entities: frozenset[str] = frozenset()
    acc_num_hash: bool = False
    acc_num_study_id: str | None = None
    jitter_required: bool = False
    fpe_mode: str = MODE_STRICT
    zip_mode: str | None = None

    def __post_init__(self) -> None:
        """Check the options and store ``keep_entities`` in parsed form."""
        is_valid, message = validate_jitter_parameters(self.max_jitter_days, self.min_jitter_days)
        if not is_valid:
            raise ValueError(message)
        validate_max_age(self.max_age)
        object.__setattr__(self, "keep_entities", parse_keep_entities(self.keep_entities))
        if self.acc_num_hash and "ACC_NUM" in self.keep_entities:
            raise ValueError("acc_num_hash and keep_entities both set an option for ACC_NUM; use one")
        if self.fpe_mode not in FPE_MODES:
            raise ValueError(f"Unknown fpe_mode {self.fpe_mode!r}; valid modes: {list(FPE_MODES)}")
        if self.zip_mode is not None:
            from tide2.anonymizers.zip_code import validate_zip_mode  # see _zip_operator

            validate_zip_mode(self.zip_mode)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AnonymizationPolicy:
        """Make a policy from the policy keys of a mapping, for example CLI args or runner kwargs.

        Other keys and ``None`` values are ignored, so the field defaults apply. ``keep_entities`` may be a
        comma-separated string or a list.
        """
        names = {field.name for field in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in names and value is not None})


DEFAULT_POLICY = AnonymizationPolicy()


@functools.cache
def _operators() -> tuple[Operator, Operator]:
    """Return the guarded ``(fpe, date_jitter)`` operator instances, built once per process.

    Keys travel in the per-call params, so the instances do not depend on them.
    """
    return guarded(HipsAlphaNumericAnonymizer)(), guarded(DateJitterAnonymizer)()


@functools.cache
def _acc_num_hash() -> Operator:
    """Return the guarded ``accession_number_hash`` operator, built once per process."""
    return guarded(AccessionNumberHashAnonymizer)()


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


def anonymize_cell(  # noqa: PLR0911
    value: Any,
    entity_type: str | None,
    salt: bytes,
    key: bytes,
    patient_id: Any = None,
    jitter: int | None = None,
    birth_date: Any = None,
    full_date: Any = None,
    recognizer_results: Any = None,
    acc_num_salt: str | None = None,
    policy: AnonymizationPolicy = DEFAULT_POLICY,
) -> str | None:
    """Anonymize one cell as a one-span note of ``entity_type``.

    Args:
        value: Cell value, or a null value.
        entity_type: One of ``SUPPORTED_ENTITY_TYPES``.
        salt: 32-byte salt.
        key: 32-byte key.
        patient_id: Patient identifier used to derive the date jitter.
        jitter: Explicit date jitter in days; wins over ``patient_id``.
        birth_date: Real birth date of the patient. Dates after the ``policy.max_age``-th
            birthday are moved to it; without it dates are not capped (a warning is
            logged once per process).
        full_date: Real full date of the row. Required for the part types
            (``DATE_DAY``, ``DATE_MONTH``, ``DATE_YEAR``), where ``value`` is the day,
            month or year held on its own and the result is that part of the shifted
            ``full_date``; a null ``full_date`` gives ``[DATE]``. Ignored otherwise. To keep
            the parts of a row consistent with each other and with the full-date column,
            pass the same non-null ``patient_id`` (and settings) or the same ``jitter``
            to every call; without either, each call draws a different random shift.
        recognizer_results: Spans found in the text by the recognition stages, as a JSON string or
            a parsed list. Required for ``FREE_TEXT``, where they index into ``str(value)``;
            ignored otherwise.
        acc_num_salt: Salt for accession number hashing (a secret, not in the policy).
        policy: Run options (see ``AnonymizationPolicy``). It is not checked here. ``keep_entities`` types are
            returned as their text. ``fpe_mode`` selects FF1 (``"strict"``, matches ``process_note``) or, for a
            value that is an INT64 (an ``int`` or a digit string such as ``"123"`` or ``"-5"``, no leading zero),
            a faster keyed permutation of the INT64 range (``"relaxed"``): no collisions, the sign is kept, but
            not the digit count, and zero is not a fixed point (do not use ``0`` as a sentinel); other values
            use FF1. With ``acc_num_hash``, an ``ACC_NUM`` cell uses ``accession_number_hash`` with
            ``acc_num_salt``, ``acc_num_study_id`` and ``patient_id``. With ``jitter_required``, a date cell with
            no ``jitter`` gives the date mask. ``zip_mode`` (``"strict"`` or ``"relaxed"``; ``None`` gives the
            default of ``AnonymizerWorker``) applies to ``ZIP_CODE`` and the ZIP codes in ``FREE_TEXT``; an integer ZIP
            is padded to 5 digits (see ``ZipCodeAnonymizer``). A ``FREE_TEXT`` cell gives the policy to the note
            worker.

    Returns:
        The anonymized string, ``None`` for a null value, or the entity mask (for example ``[DATE]``)
        when the operator cannot handle the value. For ``FREE_TEXT``, ``None`` also when the spans are
        unusable (null, malformed or outside the text), and spans of the Faker types (email, URL,
        organization) give random, non-deterministic values.

    Raises:
        ValueError: If ``entity_type`` is missing or not supported.
    """
    if not isinstance(entity_type, str) or entity_type not in SUPPORTED_ENTITY_TYPES:
        raise ValueError(f"Unsupported entity_type: {entity_type!r}")
    zip_mode = policy.zip_mode
    if entity_type in ZIP_ENTITY_TYPES and zip_mode is None:
        from tide2.anonymizers.zip_code import DEFAULT_ZIP_MODE  # see _zip_operator

        zip_mode = DEFAULT_ZIP_MODE

    if is_null(value):
        return None

    if (
        entity_type in DATE_PART_ENTITY_TYPES
        and isinstance(value, numbers.Real)
        and not isinstance(value, numbers.Integral)
        and float(value).is_integer()
    ):
        # A nullable integer column read as float holds 3.0, not 3. numbers.Real also matches NumPy floats.
        value = int(value)
    text = _zip_text(value) if entity_type in ZIP_ENTITY_TYPES else str(value)
    if text == "":
        return ""

    if entity_type in FREE_TEXT_ENTITY_TYPES:
        from tide2.anonymizers.free_text import anonymize_free_text  # heavy imports, free text only

        if is_null_or_blank(birth_date):
            _warn_no_birth_date()
        return anonymize_free_text(
            text, recognizer_results, salt, key, patient_id, jitter, birth_date, acc_num_salt, policy
        )

    # process_note relabels a type equal to its own span text as ENTITY, which falls to the DEFAULT mask.
    if safe_entity_type(entity_type, text) != entity_type:
        return mask("ENTITY")

    if entity_type in policy.keep_entities:
        return text

    if entity_type in ZIP_ENTITY_TYPES:
        return _zip_operator().operate(text, {"mode": zip_mode, "entity_type": entity_type})

    fpe, date_jitter = _operators()
    if entity_type in DATE_ENTITY_TYPES | DATE_PART_ENTITY_TYPES:
        if is_null(jitter):
            if policy.jitter_required:
                return mask(entity_type if entity_type in DATE_ENTITY_TYPES else "DATE")
            resolved = resolve_date_jitter(patient_id, salt, key, policy.max_jitter_days, policy.min_jitter_days)
        else:
            resolved = int(jitter)  # type: ignore[arg-type]
        operator, params = date_jitter, {"jitter": resolved}
        if is_null_or_blank(birth_date):
            _warn_no_birth_date()
        else:
            params["birth_date"] = birth_date
            params["max_age"] = policy.max_age
        if entity_type in DATE_PART_ENTITY_TYPES:
            params["full_date"] = full_date
    elif entity_type == "ACC_NUM" and policy.acc_num_hash:
        patient = None if is_null(patient_id) else str(patient_id).strip()
        if patient is not None and patient.lower() in ("nan", "none", "null", ""):
            patient = None
        operator = _acc_num_hash()
        params = {"salt": acc_num_salt, "study_id": policy.acc_num_study_id, "patient_id": patient}
    else:
        operator, params = fpe, {"salt": salt, "key": key, "fpe_mode": policy.fpe_mode}
    params["entity_type"] = entity_type

    # The guarded operator validates inside operate() and masks on error.
    return operator.operate(text, params)
