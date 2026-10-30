"""Tests for the FREE_TEXT type of tide2.anonymizers.cell (synthetic text only)."""

import json
import logging
from datetime import date

import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.anonymizers import cell
from tide2.anonymizers import free_text
from tide2.anonymizers.hips_names import HipsNamesAnonymizer

SALT = b"\x01" * 32
KEY = b"\x02" * 32
ACC_SALT = "acc-salt"
ACC_STUDY = "study-1"

TEXT = "Patient Rhea Orleans, MRN 48291037, seen on 03/14/2025 at 555-019-3344 in Springfield."


def _span(text: str, value: str, entity_type: str) -> dict:
    start = text.index(value)
    return {"entity_type": entity_type, "start": start, "end": start + len(value), "score": 1.0}


SPANS = [
    _span(TEXT, "Rhea Orleans", "PERSON"),
    _span(TEXT, "48291037", "MRN"),
    _span(TEXT, "03/14/2025", "DATE"),
    _span(TEXT, "555-019-3344", "PHONE_NUMBER"),
    _span(TEXT, "Springfield", "LOCATION"),
]


@pytest.fixture(autouse=True)
def _fresh_workers():
    free_text._worker.cache_clear()
    yield
    free_text._worker.cache_clear()


def _free_text(text: str = TEXT, spans=SPANS, **kwargs):
    params = {"patient_id": "FAKE-P0001", "acc_num_salt": ACC_SALT, "acc_num_study_id": ACC_STUDY, **kwargs}
    return cell.anonymize_cell(text, "FREE_TEXT", SALT, KEY, recognizer_results=json.dumps(spans), **params)


def _note(text: str, spans, **kwargs) -> str:
    worker = AnonymizerWorker(SALT, KEY, acc_num_salt=ACC_SALT, acc_num_study_id=ACC_STUDY)
    return worker.process_note(text, worker.compute_text_hash(text), json.dumps(spans), **kwargs)[
        "anonymized_note_text"
    ]


def test_matches_process_note() -> None:
    expected = _note(TEXT, SPANS, patient_id="FAKE-P0001", birth_date="1980-05-01")
    assert _free_text(birth_date="1980-05-01") == expected
    assert "48291037" not in expected
    assert "Rhea Orleans" not in expected


def test_date_matches_date_cell_of_same_row() -> None:
    shifted = cell.anonymize_cell("03/14/2025", "DATE", SALT, KEY, patient_id="FAKE-P0001")
    assert shifted != "03/14/2025"
    assert shifted in _free_text()


def test_date_uses_explicit_jitter() -> None:
    shifted = cell.anonymize_cell("03/14/2025", "DATE", SALT, KEY, patient_id="FAKE-P0001", jitter=14)
    assert shifted in _free_text(jitter=14)


def test_date_is_capped_at_max_age() -> None:
    text = "Seen on 2025-03-14."
    spans = [_span(text, "2025-03-14", "DATE")]
    birth = date(1900, 1, 1)
    expected = cell.anonymize_cell("2025-03-14", "DATE", SALT, KEY, patient_id="P1", birth_date=birth)
    assert expected in _free_text(text, spans, patient_id="P1", birth_date=birth)


def test_acc_num_uses_accession_settings() -> None:
    text = "Accession ACC123456 reviewed."
    spans = [_span(text, "ACC123456", "ACC_NUM")]
    worker = AnonymizerWorker(SALT, KEY, acc_num_salt=ACC_SALT, acc_num_study_id=ACC_STUDY, acc_num_hash=True)
    expected = worker.process_note(text, worker.compute_text_hash(text), json.dumps(spans), "FAKE-P0001")[
        "anonymized_note_text"
    ]
    assert _free_text(text, spans, acc_num_hash=True) == expected
    assert _free_text(text, spans) == _note(text, spans, patient_id="FAKE-P0001")
    assert _free_text(text, spans) != expected
    assert "ACC123456" not in expected


def test_empty_spans_leave_text_unchanged() -> None:
    assert _free_text(spans=[]) == TEXT
    assert cell.anonymize_cell(TEXT, "FREE_TEXT", SALT, KEY, recognizer_results=[]) == TEXT


def test_json_string_and_list_give_same_output() -> None:
    as_list = cell.anonymize_cell(TEXT, "FREE_TEXT", SALT, KEY, "FAKE-P0001", recognizer_results=SPANS)
    assert as_list == _free_text()


@pytest.mark.parametrize(
    "results",
    [None, "", "{not json", '{"a": 1}', json.dumps([{"entity_type": "PERSON", "start": 0, "end": 9999}])],
    ids=["none", "empty", "malformed", "not_a_list", "span_outside_text"],
)
def test_unusable_spans_give_none_and_log_hash_not_text(results, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        out = cell.anonymize_cell(TEXT, "FREE_TEXT", SALT, KEY, "FAKE-P0001", recognizer_results=results)
    assert out is None
    assert AnonymizerWorker.compute_text_hash(TEXT)[:16] in caplog.text
    assert "Rhea" not in caplog.text
    assert "48291037" not in caplog.text


def test_failing_operator_masks_only_that_entity(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(self, text, params):
        raise RuntimeError("operator failure")

    monkeypatch.setattr(HipsNamesAnonymizer, "operate", boom)
    with caplog.at_level(logging.ERROR):
        out = _free_text()
    assert "[PERSON]" in out
    assert "48291037" not in out
    assert "masked an entity of a cell" in caplog.text


def test_entity_type_without_operator_uses_placeholder() -> None:
    text = "Seen by Dr. Orleans."
    spans = [_span(text, "Orleans", "SOMETHING_NEW")]
    out = _free_text(text, spans)
    assert "Orleans" not in out
    assert "[SOMETHING_NEW]" in out or "[ENTITY]" in out or "[DEFAULT]" in out


def test_null_and_empty_values() -> None:
    assert cell.anonymize_cell(None, "FREE_TEXT", SALT, KEY, recognizer_results=[]) is None
    assert cell.anonymize_cell("", "FREE_TEXT", SALT, KEY, recognizer_results=None) == ""


def test_worker_is_built_once_per_settings() -> None:
    _free_text()
    _free_text()
    assert free_text._worker.cache_info().misses == 1
    cell.anonymize_cell(TEXT, "FREE_TEXT", SALT, b"\x03" * 32, "FAKE-P0001", recognizer_results=SPANS)
    assert free_text._worker.cache_info().misses == 2


def test_invalid_key_raises_value_error() -> None:
    with pytest.raises(ValueError, match="key must be 32 bytes"):
        cell.anonymize_cell(TEXT, "FREE_TEXT", SALT, b"short", recognizer_results=SPANS)


def test_free_text_is_supported() -> None:
    assert "FREE_TEXT" in cell.SUPPORTED_ENTITY_TYPES
