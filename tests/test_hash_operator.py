"""Tests for the hash method: ID entity types replaced by a keyed hash in cells, notes and the CLI."""

import json
from unittest.mock import MagicMock

import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.anonymizers import cell
from tide2.anonymizers import free_text
from tide2.anonymizers.cell import AnonymizationPolicy
from tide2.anonymizers.keyed_hash import KeyedHashAnonymizer
from tide2.runner import cli
from tide2.runner import local_runner

SALT = b"\x01" * 32
KEY = b"\x02" * 32
HASH_MRN = AnonymizationPolicy(text_operators="MRN=hash")

# Synthetic text only.
TEXT = "Patient Rhea Orleans, MRN 48291037, seen on 03/14/2025."
SPANS = [
    {"entity_type": "PERSON", "start": 8, "end": 20, "score": 1.0},
    {"entity_type": "MRN", "start": 26, "end": 34, "score": 1.0},
    {"entity_type": "DATE", "start": 44, "end": 54, "score": 1.0},
]


def _hash(value: str) -> str:
    return KeyedHashAnonymizer().operate(value, {"salt": SALT, "key": KEY})


@pytest.fixture(autouse=True)
def _fresh_workers():
    free_text._worker.cache_clear()
    yield
    free_text._worker.cache_clear()


def test_policy_accepts_hash_for_id_types_only() -> None:
    policy = AnonymizationPolicy.from_mapping({"text_operators": " mrn=hash, csn_id=HASH "})
    assert policy.text_operators == (("CSN_ID", "hash"), ("MRN", "hash"))
    for bad in ("DATE=hash", "NOPE=hash", "PERSON=hash"):
        with pytest.raises(ValueError, match="text_operators"):
            AnonymizationPolicy.from_mapping({"text_operators": f"MRN=hash,{bad}"})


def test_cell_hashes_with_the_method_argument() -> None:
    assert cell.anonymize_cell("123456", "MRN", SALT, KEY, method="hash") == _hash("123456")
    plain = cell.anonymize_cell("123456", "HAR", SALT, KEY)
    assert plain != _hash("123456")
    assert plain is not None
    assert len(plain) == 6


def test_cell_text_operators_do_not_hash_a_structured_cell() -> None:
    assert cell.anonymize_cell("123456", "MRN", SALT, KEY, policy=HASH_MRN) != _hash("123456")


def test_cell_hash_rejects_a_non_id_type() -> None:
    with pytest.raises(ValueError, match="does not accept"):
        cell.anonymize_cell("2024-03-15", "DATE", SALT, KEY, method="hash")


def test_cell_int_and_str_agree() -> None:
    assert cell.anonymize_cell(123456, "MRN", SALT, KEY, method="hash") == cell.anonymize_cell(
        "123456", "MRN", SALT, KEY, method="hash"
    )


def test_cell_ignores_fpe_mode() -> None:
    strict = AnonymizationPolicy(fpe_mode="strict")
    relaxed = AnonymizationPolicy(fpe_mode="relaxed")
    assert cell.anonymize_cell("123456", "MRN", SALT, KEY, method="hash", policy=strict) == cell.anonymize_cell(
        "123456", "MRN", SALT, KEY, method="hash", policy=relaxed
    )


def test_note_and_cell_give_the_same_hash() -> None:
    worker = AnonymizerWorker(SALT, KEY, policy=HASH_MRN)
    result = worker.process_note(TEXT, worker.compute_text_hash(TEXT), json.dumps(SPANS), "FAKE-P0001", 7)
    assert f"MRN {_hash('48291037')}," in result["anonymized_note_text"]
    assert cell.anonymize_cell("48291037", "MRN", SALT, KEY, method="hash") == _hash("48291037")
    items = {item["entity_type"]: item for item in json.loads(result["anonymizer_results_json"])}
    assert items["MRN"]["operator"] == "keyed_hash"


def test_free_text_cell_hashes() -> None:
    worker = AnonymizerWorker(SALT, KEY, policy=HASH_MRN)
    note = worker.process_note(TEXT, worker.compute_text_hash(TEXT), json.dumps(SPANS), "FAKE-P0001", 7)
    out = cell.anonymize_cell(
        TEXT,
        "FREE_TEXT",
        SALT,
        KEY,
        patient_id="FAKE-P0001",
        jitter=7,
        recognizer_results=json.dumps(SPANS),
        policy=HASH_MRN,
    )
    assert out == note["anonymized_note_text"]
    assert out is not None
    assert _hash("48291037") in out


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    runner = MagicMock()
    runner.run_anonymization.return_value = {}
    runner.run_pipeline.return_value = {}
    monkeypatch.setattr(local_runner, "LocalJobRunner", lambda **_kw: runner)
    return runner


def test_cli_forwards_hash(fake_runner: MagicMock) -> None:
    base = ["run", "anonymizer", "--input", "in", "--output", "out", "--salt", "s", "--key", "k"]
    cli.main([*base, "--text-operators", "MRN=hash"])
    assert fake_runner.run_anonymization.call_args.kwargs["text_operators"] == "MRN=hash"


def test_cli_rejects_hash_on_a_non_id_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "cmd_run", lambda _args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "anonymizer", "--input", "in", "--output", "out", "--text-operators", "DATE=hash"])
    assert exc.value.code == 2
