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
# Types a free-text run may leave as they are: every type of the free-text operator map except DEFAULT, plus the dates.
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


TEXT_METHODS: tuple[str, ...] = ("keep", "acc_num_hash")
# Types each free-text method accepts. A date part exists only in a cell, never as a span of a note.
_TEXT_METHOD_TYPES: dict[str, frozenset[str]] = {
    "keep": KEEPABLE_ENTITY_TYPES - DATE_PART_ENTITY_TYPES,
    "acc_num_hash": frozenset({"ACC_NUM"}),
}


def _as_int(name: str, value: Any) -> int:
    """Return ``value`` as an int: an int, a whole-number float or a numeric string; anything else is an error."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ValueError(f"{name} must be an integer, got {value!r}")


def parse_text_operators(
    value: str | Mapping[str, str] | Iterable[str | tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    """Return the free-text method of each entity type, from a string, a mapping or a list.

    Types are stripped and upper-cased, methods stripped and lower-cased.

    Args:
        value: ``None``; a string such as ``"DATE=keep,ACC_NUM=acc_num_hash"``; a mapping of type to method;
            or an iterable of ``"TYPE=method"`` strings or ``(type, method)`` pairs.

    Returns:
        ``(entity_type, method)`` pairs sorted by type, so a policy that holds them stays hashable. Empty for
        ``None`` or an empty value.

    Raises:
        ValueError: If an item is malformed, a type is given twice, a method is unknown, or a method does not
            accept the type (see ``TEXT_METHODS``).
    """
    if value is None:
        return ()
    if isinstance(value, Mapping):
        items: list[Any] = list(value.items())
    elif isinstance(value, str):
        items = [part for part in value.split(",") if part.strip()]
    else:
        try:
            items = list(value)
        except TypeError:
            raise ValueError(
                f"Invalid text_operators {value!r}; use a string, a mapping or a list of TYPE=method"
            ) from None
    pairs: dict[str, str] = {}
    for item in items:
        if isinstance(item, str):
            entity_type, separator, method = item.partition("=")
            if not separator:
                raise ValueError(f"Invalid text_operators item {item!r}; use TYPE=method")
        else:
            try:
                entity_type, method = item
            except (TypeError, ValueError):
                raise ValueError(f"Invalid text_operators item {item!r}; use TYPE=method") from None
        entity_type, method = str(entity_type).strip().upper(), str(method).strip().lower()
        if entity_type in pairs:
            raise ValueError(f"Entity type {entity_type} is given twice in text_operators")
        if method not in TEXT_METHODS:
            raise ValueError(f"Unknown text_operators method {method!r}; valid methods: {list(TEXT_METHODS)}")
        if entity_type not in _TEXT_METHOD_TYPES[method]:
            raise ValueError(
                f"text_operators method {method!r} does not accept {entity_type!r}; "
                f"valid types: {sorted(_TEXT_METHOD_TYPES[method])}"
            )
        pairs[entity_type] = method
    return tuple(sorted(pairs.items()))


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

    Secrets (``salt``, ``key``, ``acc_num_salt``, ``acc_num_study_id``) and row context (``patient_id``,
    ``jitter``, ...) are not part of it. A frozen policy is hashable, so it can be a cache key.

    The method of a structured cell is not a policy field: the caller picks it per column with the ``method``
    argument of ``anonymize_cell``. Free text (notes and ``FREE_TEXT`` cells) mixes many entity types in one value,
    so ``text_operators`` picks a method for each entity type there. A structured cell is always anonymized: to
    keep a column, do not send it to anonymization.

    Raises:
        ValueError: If a value is out of range or ``text_operators`` is invalid.
    """

    min_jitter_days: int = DEFAULT_MIN_JITTER_DAYS
    max_jitter_days: int = DEFAULT_MAX_JITTER_DAYS
    max_age: int = DEFAULT_MAX_AGE
    text_operators: tuple[tuple[str, str], ...] = ()
    jitter_required: bool = False
    fpe_mode: str = MODE_STRICT
    zip_mode: str | None = None

    def __post_init__(self) -> None:
        """Check the options and store ``text_operators`` in parsed form."""
        for name in ("min_jitter_days", "max_jitter_days"):
            object.__setattr__(self, name, _as_int(name, getattr(self, name)))
        is_valid, message = validate_jitter_parameters(self.max_jitter_days, self.min_jitter_days)
        if not is_valid:
            raise ValueError(message)
        object.__setattr__(self, "max_age", validate_max_age(self.max_age))
        object.__setattr__(self, "text_operators", parse_text_operators(self.text_operators))
        if self.fpe_mode not in FPE_MODES:
            raise ValueError(f"Unknown fpe_mode {self.fpe_mode!r}; valid modes: {list(FPE_MODES)}")
        if self.zip_mode is not None:
            from tide2.anonymizers.zip_code import validate_zip_mode  # see _zip_operator

            validate_zip_mode(self.zip_mode)

    def text_operator(self, entity_type: str) -> str | None:
        """Return the free-text method of ``entity_type``, or ``None`` when the default applies."""
        return dict(self.text_operators).get(entity_type)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AnonymizationPolicy:
        """Make a policy from the policy keys of a mapping, for example CLI args or runner kwargs.

        Other keys and ``None`` values are ignored, so the field defaults apply. ``text_operators`` may be a
        string such as ``"DATE=keep"``, a mapping of type to method, or a list.
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


def check_acc_num_secrets(
    policy: AnonymizationPolicy, acc_num_salt: str | None, acc_num_study_id: str | None, method: str | None = None
) -> None:
    """Raise ``ValueError`` if ``acc_num_hash`` is used (``method`` or ``policy.text_operators``) without both secrets."""
    uses_acc_num_hash = method == "acc_num_hash" or "acc_num_hash" in dict(policy.text_operators).values()
    if uses_acc_num_hash and (acc_num_salt is None or not acc_num_study_id):
        raise ValueError("acc_num_hash needs acc_num_salt and acc_num_study_id")


def _check_method(
    method: str | None,
    entity_type: str,
    acc_num_salt: str | None,
    acc_num_study_id: str | None,
    policy: AnonymizationPolicy,
) -> None:
    """Raise ``ValueError`` if ``method`` cannot apply to a cell of ``entity_type``."""
    if method is None:
        return
    if method == "keep":
        raise ValueError("Method 'keep' is for free text only (text_operators); leave the column out")
    if method not in TEXT_METHODS:
        raise ValueError(f"Unknown method {method!r}; valid methods: {[m for m in TEXT_METHODS if m != 'keep']}")
    if entity_type in FREE_TEXT_ENTITY_TYPES:
        raise ValueError("A FREE_TEXT cell takes its methods from policy.text_operators, not from method")
    if entity_type not in _TEXT_METHOD_TYPES[method]:
        raise ValueError(f"Method {method!r} does not accept entity_type {entity_type!r}")
    check_acc_num_secrets(policy, acc_num_salt, acc_num_study_id, method)


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
    acc_num_study_id: str | None = None,
    method: str | None = None,
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
        acc_num_study_id: Study ID for accession number hashing (a secret, not in the policy).
        method: Method for this cell, chosen by the caller for the column. ``None`` gives the default of the entity
            type. ``"acc_num_hash"`` (``ACC_NUM`` only) uses ``accession_number_hash`` with ``acc_num_salt``,
            ``acc_num_study_id`` and ``patient_id``. ``"keep"`` is not allowed: leave the column out.
            ``FREE_TEXT`` takes no method; it takes its methods from ``policy.text_operators``.
        policy: Run options (see ``AnonymizationPolicy``). It is not checked here. ``fpe_mode`` selects FF1 (``"strict"``, matches
            ``process_note``) or, for a value that is an INT64 (an ``int`` or a digit string such as ``"123"`` or
            ``"-5"``, no leading zero), a faster keyed permutation of the INT64 range (``"relaxed"``): no collisions,
            the sign is kept, but not the digit count, and zero is not a fixed point (do not use ``0`` as a sentinel);
            other values use FF1. With ``jitter_required``, a date cell with no
            ``jitter`` gives the date mask. ``zip_mode`` (``"strict"`` or ``"relaxed"``; ``None`` gives the default of
            ``AnonymizerWorker``) applies to ``ZIP_CODE`` and the ZIP codes in ``FREE_TEXT``; an integer ZIP is padded
            to 5 digits (see ``ZipCodeAnonymizer``). A ``FREE_TEXT`` cell gives the policy to the note worker.

    Returns:
        The anonymized string, ``None`` for a null value, or the entity mask (for example ``[DATE]``)
        when the operator cannot handle the value. For ``FREE_TEXT``, ``None`` also when the spans are
        unusable (null, malformed or outside the text), and spans of the Faker types (email, URL,
        organization) give random, non-deterministic values.

    Raises:
        ValueError: If ``entity_type`` is missing or not supported, or ``method`` is unknown, not allowed for the
            entity type, or ``"acc_num_hash"`` without ``acc_num_salt`` or ``acc_num_study_id``.
    """
    if not isinstance(entity_type, str) or entity_type not in SUPPORTED_ENTITY_TYPES:
        raise ValueError(f"Unsupported entity_type: {entity_type!r}")
    _check_method(method, entity_type, acc_num_salt, acc_num_study_id, policy)
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
            text, recognizer_results, salt, key, patient_id, jitter, birth_date, acc_num_salt, acc_num_study_id, policy
        )

    # process_note relabels a type equal to its own span text as ENTITY, which falls to the DEFAULT mask.
    if safe_entity_type(entity_type, text) != entity_type:
        return mask("ENTITY")

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
    elif method == "acc_num_hash":
        patient = None if is_null(patient_id) else str(patient_id).strip()
        if patient is not None and patient.lower() in ("nan", "none", "null", ""):
            patient = None
        operator = _acc_num_hash()
        params = {"salt": acc_num_salt, "study_id": acc_num_study_id, "patient_id": patient}
    else:
        operator, params = fpe, {"salt": salt, "key": key, "fpe_mode": policy.fpe_mode}
    params["entity_type"] = entity_type

    # The guarded operator validates inside operate() and masks on error.
    return operator.operate(text, params)
