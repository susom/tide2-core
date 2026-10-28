"""Tests for the day, month and year entity types of tide2.anonymizers.cell."""

import logging
from datetime import date

import pytest

from tide2.anonymizers import cell

SALT = b"\x01" * 32
KEY = b"\x02" * 32
FULL = date(2023, 3, 30)
PARTS = [("DATE_DAY", 30), ("DATE_MONTH", 3), ("DATE_YEAR", 2023)]


def _parts_of(text: str) -> list[str]:
    year, month, day = text.split("-")
    return [str(int(day)), str(int(month)), year]


@pytest.mark.parametrize("birth_date", [None, "1930-06-15"])
@pytest.mark.parametrize("patient_id", ["FAKE-P0001", "FAKE-P0002"])
def test_parts_reassemble_to_the_full_date(patient_id: str, birth_date: str | None) -> None:
    kwargs = {"patient_id": patient_id, "birth_date": birth_date}
    full = cell.anonymize_cell(FULL.isoformat(), "DATE", SALT, KEY, **kwargs)
    got = [cell.anonymize_cell(value, kind, SALT, KEY, full_date=FULL, **kwargs) for kind, value in PARTS]
    assert [got[0], got[1], got[2]] == _parts_of(full)


def test_explicit_jitter_and_numeric_inputs() -> None:
    assert cell.anonymize_cell(30, "DATE_DAY", SALT, KEY, jitter=5, full_date="2023-03-30") == "4"
    assert cell.anonymize_cell(30.0, "DATE_DAY", SALT, KEY, jitter=5, full_date="2023-03-30") == "4"
    assert cell.anonymize_cell("2023", "DATE_YEAR", SALT, KEY, jitter=300, full_date="2023-03-30") == "2024"


def test_null_value_stays_null() -> None:
    assert cell.anonymize_cell(None, "DATE_DAY", SALT, KEY, jitter=5, full_date=FULL) is None
    assert cell.anonymize_cell(float("nan"), "DATE_MONTH", SALT, KEY, jitter=5, full_date=FULL) is None


def test_null_or_missing_full_date_is_masked() -> None:
    assert cell.anonymize_cell(30, "DATE_DAY", SALT, KEY, jitter=5) == "[DATE]"
    assert cell.anonymize_cell(30, "DATE_DAY", SALT, KEY, jitter=5, full_date=float("nan")) == "[DATE]"


def test_mismatch_is_masked() -> None:
    assert cell.anonymize_cell(29, "DATE_DAY", SALT, KEY, jitter=5, full_date=FULL) == "[DATE]"


def test_invalid_full_date_is_masked_by_entity_type() -> None:
    assert cell.anonymize_cell(30, "DATE_DAY", SALT, KEY, jitter=5, full_date="nope") == "[DATE_DAY]"


def test_missing_birth_date_warns(caplog: pytest.LogCaptureFixture) -> None:
    cell._warn_no_birth_date.cache_clear()
    with caplog.at_level(logging.WARNING, logger=cell.logger.name):
        cell.anonymize_cell(30, "DATE_DAY", SALT, KEY, jitter=5, full_date=FULL)
        cell.anonymize_cell(3, "DATE_MONTH", SALT, KEY, jitter=5, full_date=FULL)
    assert len(caplog.records) == 1


def test_fpe_ignores_full_date() -> None:
    assert cell.anonymize_cell("48291037", "MRN", SALT, KEY, full_date=FULL) == cell.anonymize_cell(
        "48291037", "MRN", SALT, KEY
    )
