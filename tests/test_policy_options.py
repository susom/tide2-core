"""Tests that notes and cells take the same AnonymizationPolicy options."""

import json

import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.anonymizers import cell
from tide2.anonymizers import free_text
from tide2.anonymizers.accession_number_hash import AccessionNumberHashAnonymizer
from tide2.anonymizers.cell import DEFAULT_POLICY
from tide2.anonymizers.cell import AnonymizationPolicy
from tide2.cryptographic.fpe_strings import FormatPreservingEncryption

SALT = b"\x01" * 32
KEY = b"\x02" * 32
ACC_SALT = "acc-salt"
ACC_STUDY = "study-1"

# Synthetic text only.
TEXT = "Patient Rhea Orleans, MRN 48291037, seen on 03/14/2025."
SPANS = [
    {"entity_type": "PERSON", "start": 8, "end": 20, "score": 1.0},
    {"entity_type": "MRN", "start": 26, "end": 34, "score": 1.0},
    {"entity_type": "DATE", "start": 44, "end": 54, "score": 1.0},
]


@pytest.fixture(autouse=True)
def _fresh_workers():
    free_text._worker.cache_clear()
    yield
    free_text._worker.cache_clear()


def _note(policy: AnonymizationPolicy, spans=SPANS, jitter: int | None = None) -> str:
    worker = AnonymizerWorker(SALT, KEY, policy=policy)
    result = worker.process_note(TEXT, worker.compute_text_hash(TEXT), json.dumps(spans), "FAKE-P0001", jitter)
    return result["anonymized_note_text"]


def test_worker_default_policy() -> None:
    worker = AnonymizerWorker(SALT, KEY)
    assert worker.policy == DEFAULT_POLICY
    assert worker.policy.text_operators == ()


def test_worker_reads_fpe_mode() -> None:
    spans = [SPANS[1]]
    strict = FormatPreservingEncryption(SALT, KEY, "strict").encrypt("48291037")[0]
    relaxed = FormatPreservingEncryption(SALT, KEY, "relaxed").encrypt("48291037")[0]
    assert strict != relaxed
    assert f"MRN {strict}," in _note(AnonymizationPolicy(), spans)
    assert f"MRN {relaxed}," in _note(AnonymizationPolicy(fpe_mode="relaxed"), spans)


def test_cell_default_strict() -> None:
    strict = FormatPreservingEncryption(SALT, KEY, "strict").encrypt("48291037")[0]
    assert cell.anonymize_cell("48291037", "MRN", SALT, KEY, policy=DEFAULT_POLICY) == strict


@pytest.mark.parametrize("patient_id", ["FAKE-P0001", None])
def test_cell_acc_num_hash(patient_id: str | None) -> None:
    policy = AnonymizationPolicy(acc_num_study_id=ACC_STUDY)
    expected = AccessionNumberHashAnonymizer().operate(
        "ACC123456", {"salt": ACC_SALT, "study_id": ACC_STUDY, "patient_id": patient_id}
    )
    out = cell.anonymize_cell(
        "ACC123456",
        "ACC_NUM",
        SALT,
        KEY,
        patient_id=patient_id,
        acc_num_salt=ACC_SALT,
        method="acc_num_hash",
        policy=policy,
    )
    assert out == expected
    note_policy = AnonymizationPolicy(acc_num_study_id=ACC_STUDY, text_operators={"ACC_NUM": "acc_num_hash"})
    worker = AnonymizerWorker(SALT, KEY, policy=note_policy, acc_num_salt=ACC_SALT)
    span = [{"entity_type": "ACC_NUM", "start": 0, "end": 9, "score": 1.0}]
    note = worker.process_note("ACC123456", worker.compute_text_hash("ACC123456"), json.dumps(span), patient_id, 0)
    assert note["anonymized_note_text"] == expected
    plain = cell.anonymize_cell("ACC123456", "ACC_NUM", SALT, KEY, patient_id=patient_id, acc_num_salt=ACC_SALT)
    assert plain != expected


def test_cell_jitter_required() -> None:
    policy = AnonymizationPolicy(jitter_required=True)
    assert cell.anonymize_cell("2024-05-20", "DATE", SALT, KEY, policy=policy) == "[DATE]"
    day = cell.anonymize_cell(15, "DATE_DAY", SALT, KEY, full_date="2024-05-15", policy=policy)
    assert day == "[DATE]"
    shifted = cell.anonymize_cell("2024-05-20", "DATE", SALT, KEY, jitter=5, policy=policy)
    assert shifted == "2024-05-25"


@pytest.mark.parametrize(
    ("policy", "jitter"),
    [
        (AnonymizationPolicy(fpe_mode="relaxed"), 7),
        (AnonymizationPolicy(jitter_required=True), None),
        (AnonymizationPolicy(text_operators="PERSON=keep"), 7),
    ],
)
def test_free_text_cell_equals_note(policy: AnonymizationPolicy, jitter: int | None) -> None:
    out = cell.anonymize_cell(
        TEXT,
        "FREE_TEXT",
        SALT,
        KEY,
        patient_id="FAKE-P0001",
        jitter=jitter,
        recognizer_results=json.dumps(SPANS),
        policy=policy,
    )
    assert out == _note(policy, jitter=jitter)
