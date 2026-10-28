"""Tests for tide2.anonymizers.cell: parity with AnonymizerWorker.process_note on a one-span note."""

import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.anonymizers import cell

SALT = b"\x01" * 32
KEY = b"\x02" * 32


def _case(
    case_id: str, value: str | None, entity_type: str, patient_id: str | None = "FAKE-P0001", jitter: int | None = None
) -> dict:
    return {"id": case_id, "value": value, "entity_type": entity_type, "patient_id": patient_id, "jitter": jitter}


# Synthetic values only.
CASES = [
    _case("fpe_mrn_8digits", "48291037", "MRN"),
    _case("fpe_mrn_leading_zeros", "00482910", "MRN"),
    _case("fpe_short_1digit", "7", "ID"),
    _case("fpe_short_2digits", "42", "ID"),
    _case("fpe_short_3digits", "901", "ID"),
    _case("fpe_short_4digits", "1842", "ID"),
    _case("fpe_short_5digits", "59201", "ID"),
    _case("fpe_6digits", "604821", "HAR", "FAKE-P0002"),
    _case("fpe_7digits", "7381920", "HAR", "FAKE-P0002"),
    _case("fpe_csn_10digits", "1312849201", "CSN_ID"),
    _case("fpe_19digits", "9182736455463728191", "ID", "FAKE-P0002"),
    _case("fpe_alnum_with_separators", "AB-9021-X7", "MEDICAL_LICENSE"),
    _case("fpe_phone_formatted", "(555) 014-8821", "PHONE_NUMBER"),
    _case("fpe_phone_dashes", "555-019-3344", "PHONE", "FAKE-P0002"),
    _case("fpe_ssn_formatted", "900-12-3456", "US_SSN"),
    _case("fpe_symbols_only_masked", "---", "MRN"),
    _case("date_iso_derived_p1", "2024-03-15", "DATE"),
    _case("date_iso_derived_p2", "2024-03-15", "DATE", "FAKE-P0002"),
    _case("date_iso_explicit_jitter", "2024-03-15", "DATE", jitter=14),
    _case("date_us_slash", "07/22/2023", "DATE"),
    _case("date_month_name", "January 5st, 2022", "DATE", "FAKE-P0002", 10),
    _case("date_month_day_year_slash", "March/12/2021", "DATE", jitter=-7),
    _case("datetime_iso_with_time", "2024-06-10 14:30:00", "DATE_TIME"),
    _case("date_unparseable_masked", "not-a-valid-date", "DATE"),
    _case("date_null_patient_random_jitter", "2024-05-20", "DATE", None),
    _case("null_value_mrn", None, "MRN"),
    _case("null_value_date", None, "DATE"),
]


@pytest.fixture(scope="module")
def worker() -> AnonymizerWorker:
    return AnonymizerWorker(salt=SALT, key=KEY)


def _is_random_jitter_case(case: dict) -> bool:
    return case["entity_type"] in cell.DATE_ENTITY_TYPES and case["patient_id"] is None and case["jitter"] is None


@pytest.mark.parametrize(
    "case", [c for c in CASES if c["value"] is not None and not _is_random_jitter_case(c)], ids=lambda c: c["id"]
)
def test_matches_process_note(worker: AnonymizerWorker, case: dict) -> None:
    value = case["value"]
    expected = worker.process_note(
        note_text=value,
        original_text_hash=worker.compute_text_hash(value),
        recognizer_results_json=[{"entity_type": case["entity_type"], "start": 0, "end": len(value), "score": 1.0}],
        patient_id=case["patient_id"],
        jitter=case["jitter"],
    )["anonymized_note_text"]
    got = cell.anonymize_cell(
        value, case["entity_type"], SALT, KEY, patient_id=case["patient_id"], jitter=case["jitter"]
    )
    assert got == expected


def test_null_patient_gets_a_random_shift() -> None:
    out = cell.anonymize_cell("2024-05-20", "DATE", SALT, KEY)
    assert out != "2024-05-20"
    assert out.count("-") == 2


def test_null_value_stays_null() -> None:
    assert cell.anonymize_cell(None, "MRN", SALT, KEY) is None
    assert cell.anonymize_cell(float("nan"), "DATE", SALT, KEY) is None


def test_numeric_value_is_stringified() -> None:
    assert cell.anonymize_cell(48291037, "MRN", SALT, KEY) == cell.anonymize_cell("48291037", "MRN", SALT, KEY)


@pytest.mark.parametrize("value", [1, 7, 604821, 48291037, 9182736455463728191])
def test_int64_style_value_round_trips_through_int_without_losing_a_digit(value: int) -> None:
    out = cell.anonymize_cell(str(value), "ID", SALT, KEY)
    assert not out.startswith("0")
    assert len(out) == len(str(value))
    assert len(str(int(out))) == len(str(value))


@pytest.mark.parametrize("entity_type", ["PERSON", "ACC_NUM", "", None])
def test_unsupported_entity_type_raises(entity_type: str | None) -> None:
    with pytest.raises(ValueError, match="Unsupported entity_type"):
        cell.anonymize_cell("x", entity_type, SALT, KEY)


def test_same_patient_gets_the_same_jitter() -> None:
    first = cell.compute_jitter_for_patient("FAKE-P0001", SALT, KEY)
    assert first == cell.compute_jitter_for_patient(" FAKE-P0001 ", SALT, KEY)
    assert 3 <= abs(first) <= 180


def test_different_keys_give_different_output() -> None:
    other = b"\x03" * 32
    assert cell.anonymize_cell("48291037", "MRN", SALT, KEY) != cell.anonymize_cell("48291037", "MRN", SALT, other)
