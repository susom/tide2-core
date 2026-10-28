"""HIPAA Safe Harbor ZIP code anonymizer.

45 CFR 164.514(b)(2)(i)(B) removes ZIP codes, except the initial three digits when, according to the current
publicly available Census data, the area formed by all ZIP codes with the same three initial digits contains more
than 20,000 people. The initial three digits of every area with 20,000 or fewer people must be changed to ``000``.

Two modes:

- ``safe-harbor-strict``: the regulation as written. A ZIP becomes its 3-digit prefix, or ``000`` when the prefix
  area has 20,000 people or fewer.
- ``safe-harbor-relaxed`` (the default): **not Safe Harbor compliant.** A ZIP in a prefix area with more than 20,000
  people is kept as its 5 digits; otherwise it becomes its real 3-digit prefix. Use it only under an expert
  determination or a data use agreement that allows 5-digit ZIPs.

The population comes from ``resources/zip3_population_2020.tsv`` (2020 Census total population by ZCTA, summed by
the first three digits; see ``scripts/zip/build_zip3_population.py``). ZCTAs approximate USPS ZIP areas. A prefix
missing from the table counts as population 0, so it is restricted (fail closed).

ZIP codes are strings end to end: they may start with ``0`` (New England, NJ, Puerto Rico, military) and the
restricted prefix is the string ``"000"``. A 3- or 4-digit value is never zero-padded here; it is not a valid ZIP
and the operator rejects it (the guarded operator masks it).

This module imports only the standard library, presidio and sibling modules by full path, so it can be bundled
like ``cell.py``.
"""

import csv
import functools
import re
from pathlib import Path

from presidio_anonymizer.entities import InvalidParamError
from presidio_anonymizer.operators import Operator
from presidio_anonymizer.operators import OperatorType

from tide2.utils.resource_utils import ZIP3_POPULATION_FILE
from tide2.utils.resource_utils import get_resource_path

__docformat__ = "google"

MODE_STRICT = "safe-harbor-strict"
MODE_RELAXED = "safe-harbor-relaxed"
ZIP_MODES: tuple[str, ...] = (MODE_STRICT, MODE_RELAXED)
# The mode used when none is given. It is not Safe Harbor compliant: pass MODE_STRICT for that.
DEFAULT_ZIP_MODE = MODE_RELAXED
# Labels other than ZIP_CODE that transformer models emit for a ZIP code.
ZIP_ENTITY_ALIASES: frozenset[str] = frozenset({"ZIP", "ZIPCODE"})

SAFE_HARBOR_POPULATION_THRESHOLD = 20000
RESTRICTED_ZIP3 = "000"

# A 5-digit ZIP with an optional ZIP+4 suffix. ASCII digits only: Unicode digits are not ZIPs.
_ZIP_PATTERN = re.compile(r"([0-9]{5})(?:[- ]?[0-9]{4})?")


@functools.cache
def load_zip3_population() -> dict[str, int]:
    """Return the 2020 population of each 3-digit ZIP prefix, keyed by the prefix string (``"021"``).

    Returns:
        A dict shared by all callers; do not modify it.
    """
    population: dict[str, int] = {}
    with Path(get_resource_path(ZIP3_POPULATION_FILE)).open(encoding="utf-8", newline="") as f:
        rows = (row for row in csv.reader(f, delimiter="\t") if row and not row[0].startswith("#"))
        next(rows, None)  # header
        for zip3, count in rows:
            population[zip3] = int(count)
    return population


def validate_zip_mode(mode: object) -> None:
    """Raise ``ValueError`` unless ``mode`` is one of ``ZIP_MODES``."""
    if mode not in ZIP_MODES:
        raise ValueError(f"zip mode must be one of {ZIP_MODES}, got {mode!r}")


def anonymize_zip(zip_text: str, mode: str, population: dict[str, int] | None = None) -> str:
    """Anonymize one ZIP code according to the Safe Harbor rule.

    Args:
        zip_text: A 5-digit ZIP, optionally followed by a ZIP+4 suffix (``-1234``, `` 1234`` or ``1234``).
            Surrounding whitespace is ignored.
        mode: ``"safe-harbor-strict"`` or ``"safe-harbor-relaxed"``.
        population: Population by 3-digit prefix; defaults to the bundled 2020 Census table.

    Returns:
        A string: the 3-digit prefix, ``"000"`` or the 5-digit ZIP (relaxed mode only). The ZIP+4 suffix is
        always dropped.

    Raises:
        ValueError: If ``mode`` is invalid or ``zip_text`` is not a 5-digit ZIP.
    """
    validate_zip_mode(mode)
    match = _ZIP_PATTERN.fullmatch(zip_text.strip())
    if match is None:
        raise ValueError("not a 5-digit ZIP code")
    zip5 = match.group(1)
    zip3 = zip5[:3]
    table = load_zip3_population() if population is None else population
    if table.get(zip3, 0) > SAFE_HARBOR_POPULATION_THRESHOLD:
        return zip3 if mode == MODE_STRICT else zip5
    return RESTRICTED_ZIP3 if mode == MODE_STRICT else zip3


class ZipCodeAnonymizer(Operator):
    """Presidio operator that reduces ZIP codes to the HIPAA Safe Harbor form.

    Operator name: ``zip_code_safe_harbor``. Params: ``mode`` (required), one of ``ZIP_MODES``. See the module
    docstring for the rules. Wrap it with ``guarded`` to mask a value that is not a ZIP.
    """

    def operate(self, text: str, params: dict | None = None) -> str:
        """Return the Safe Harbor form of ``text``; see ``anonymize_zip``."""
        return anonymize_zip(text, (params or {}).get("mode", ""))

    def validate(self, params: dict | None = None) -> None:
        """Check that ``mode`` is a valid ZIP mode."""
        mode = (params or {}).get("mode")
        if mode not in ZIP_MODES:
            raise InvalidParamError(f"Parameter 'mode' must be one of {ZIP_MODES}, got {mode!r}.")

    def operator_name(self) -> str:
        """Return the operator name."""
        return "zip_code_safe_harbor"

    def operator_type(self) -> OperatorType:
        """Return the operator type."""
        return OperatorType.Anonymize
