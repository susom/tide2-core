"""FREE_TEXT cells end to end on the synthetic notebook sample notes (regex recognizers, no model)."""

import json
import re
from pathlib import Path

import orjson
import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.recognizer import RecognizerWorker
from tide2.anonymizers import cell
from tide2.anonymizers import free_text
from tide2.anonymizers import presidio_patches

SAMPLE = Path(__file__).resolve().parents[1] / "notebooks" / "sample_data"
SALT = b"\x00" * 32
KEY = b"\x11" * 32
ACC = {"acc_num_salt": "s", "acc_num_study_id": "study"}
DOB = re.compile(r"\b(?:DOB|Date of Birth)\s*:\s*(\d{1,2})/(\d{1,2})/(\d{4})", re.I)
SERVICE = re.compile(r"\b(?:Date of Service|Date of Visit|Admission Date|Date)\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", re.I)


@pytest.fixture(scope="module")
def rows():
    free_text._worker.cache_clear()
    was_patched = presidio_patches.is_remove_duplicates_patched()
    recognizer = RecognizerWorker()
    out = []
    for i in range(1, 14):
        text = (SAMPLE / "text_files" / f"note_{i}.txt").read_text()
        phi_path = SAMPLE / "patient_phi" / f"note_{i}.json"
        phi = json.loads(phi_path.read_text()) if phi_path.exists() else {}
        spans = recognizer.process_note(text, AnonymizerWorker.compute_text_hash(text), None, phi or None)
        dob = DOB.search(text)
        service = SERVICE.search(text)
        out.append(
            {
                "patient_id": f"P{i:03d}",
                "birth_date": f"{dob.group(3)}-{int(dob.group(1)):02d}-{int(dob.group(2)):02d}" if dob else None,
                "service_date": service.group(1) if service else None,
                "text": text,
                "spans": spans["recognizer_results_json"],
                "phi": phi,
            }
        )
    # RecognizerWorker patches Presidio globally; later recognizer tests expect the original.
    if not was_patched:
        presidio_patches.unpatch_remove_duplicates()
    return out


def _cell(row: dict) -> str | None:
    return cell.anonymize_cell(
        row["text"],
        "FREE_TEXT",
        SALT,
        KEY,
        row["patient_id"],
        birth_date=row["birth_date"],
        recognizer_results=row["spans"],
        **ACC,
    )


def _has_faker_spans(row: dict) -> bool:
    return any(
        s["entity_type"] in {"EMAIL_ADDRESS", "WEB", "URL", "ORGANIZATION", "GENETIC_SEQUENCE"}
        for s in orjson.loads(row["spans"])
    )


def test_no_cell_fails(rows: list[dict]) -> None:
    assert all(_cell(r) is not None for r in rows)


def test_matches_process_batch_without_faker_spans(rows: list[dict]) -> None:
    batch = AnonymizerWorker(SALT, KEY, **ACC).process_batch(
        {
            "note_text": [r["text"] for r in rows],
            "recognizer_results_json": [r["spans"] for r in rows],
            "patient_id": [r["patient_id"] for r in rows],
            "birth_date": [r["birth_date"] for r in rows],
        }
    )
    expected = list(batch["anonymized_note_text"])
    compared = [(r, e) for r, e in zip(rows, expected, strict=True) if not _has_faker_spans(r)]
    assert compared
    for row, text in compared:
        assert _cell(row) == text


def test_service_date_column_matches_date_in_text(rows: list[dict]) -> None:
    checked = 0
    for row in rows:
        if not row["service_date"]:
            continue
        shifted = cell.anonymize_cell(
            row["service_date"], "DATE", SALT, KEY, patient_id=row["patient_id"], birth_date=row["birth_date"]
        )
        assert shifted in _cell(row)
        checked += 1
    assert checked > 0


def test_known_phi_is_not_left(rows: list[dict]) -> None:
    for row in rows:
        out = _cell(row)
        for values in row["phi"].values():
            for value in values:
                assert not value or value not in out
