"""Tests for the single jitter generator in tide2.cryptographic.date_jitter and its range plumbing."""

import re
import sys
from pathlib import Path

import orjson
import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.anonymizer import create_anonymizer_actor
from tide2.cryptographic.date_jitter import derive_date_jitter
from tide2.cryptographic.date_jitter import random_date_jitter
from tide2.cryptographic.date_jitter import resolve_date_jitter
from tide2.runner import cli

SALT = b"\x01" * 32
KEY = b"\x02" * 32

# Outputs of derive_date_jitter before the refactor: (patient, default range, range 10..60).
DERIVE_VECTORS = [
    ("patient_1", 83, 24),
    ("12345", -52, -36),
    ("abc-def", 47, 48),
    ("\u00dcn\u00ef", -143, -10),
    ("0", 40, 19),
]


@pytest.mark.parametrize(("patient", "default", "narrow"), DERIVE_VECTORS)
def test_derive_outputs_are_unchanged(patient: str, default: int, narrow: int) -> None:
    assert derive_date_jitter(patient, SALT, KEY) == default
    assert derive_date_jitter(patient, SALT, KEY, 60, 10) == narrow


@pytest.mark.parametrize(("max_days", "min_days"), [(180, 3), (60, 10), (5, 3), (30, 30), (400, 3)])
def test_random_jitter_stays_in_range(max_days: int, min_days: int) -> None:
    values = [random_date_jitter(max_days, min_days) for _ in range(20000)]
    assert all(min_days <= abs(v) <= max_days for v in values)
    assert any(v > 0 for v in values)
    assert any(v < 0 for v in values)


def test_random_jitter_hits_both_ends_of_a_small_range() -> None:
    assert {abs(random_date_jitter(5, 3)) for _ in range(2000)} == {3, 4, 5}


@pytest.mark.parametrize(("max_days", "min_days"), [(180, 2), (180, 0), (180, -3), (10, 20)])
def test_invalid_range_raises(max_days: int, min_days: int) -> None:
    with pytest.raises(ValueError, match="jitter_days"):
        random_date_jitter(max_days, min_days)
    with pytest.raises(ValueError, match="jitter_days"):
        resolve_date_jitter("P1", SALT, KEY, max_days, min_days)
    with pytest.raises(ValueError, match="jitter_days"):
        resolve_date_jitter(None, SALT, KEY, max_days, min_days)


@pytest.mark.parametrize("patient", [None, float("nan"), "", "  ", "nan", "None", " NULL "])
def test_null_like_patient_gets_a_random_jitter_in_range(patient: object) -> None:
    values = {resolve_date_jitter(patient, SALT, KEY, 20, 10) for _ in range(300)}
    assert all(10 <= abs(v) <= 20 for v in values)
    assert len(values) > 1


def test_known_patient_is_derived_and_whitespace_is_stripped() -> None:
    assert resolve_date_jitter(" patient_1 ", SALT, KEY) == 83
    assert resolve_date_jitter("patient_1", SALT, KEY, 60, 10) == 24
    assert resolve_date_jitter(12345, SALT, KEY) == -52


def _worker(min_jitter_days: int = 3, max_jitter_days: int = 180) -> AnonymizerWorker:
    return AnonymizerWorker(salt=SALT, key=KEY, min_jitter_days=min_jitter_days, max_jitter_days=max_jitter_days)


def test_worker_defaults() -> None:
    worker = _worker()
    assert (worker.min_jitter_days, worker.max_jitter_days) == (3, 180)
    assert worker._compute_jitter_for_patient("patient_1") == 83


def test_worker_null_patient_jitter_follows_the_range() -> None:
    worker = _worker(min_jitter_days=10, max_jitter_days=20)
    values = [worker._compute_jitter_for_patient(None) for _ in range(1000)]
    assert all(10 <= abs(v) <= 20 for v in values)


def test_worker_null_patient_notes_report_jitter_in_range() -> None:
    worker = _worker(min_jitter_days=10, max_jitter_days=20)
    for i in range(1000):
        text = "2024-05-20"
        result = worker.process_note(
            note_text=text,
            original_text_hash=f"h{i}",
            recognizer_results_json=[{"entity_type": "DATE", "start": 0, "end": len(text), "score": 1.0}],
            patient_id=None,
        )
        assert 10 <= abs(result["jitter"]) <= 20
        assert orjson.loads(result["anonymizer_results_json"])[0]["text"] != "[DATE]"


@pytest.mark.parametrize(("max_days", "min_days"), [(180, 2), (10, 20)])
def test_worker_rejects_invalid_range(max_days: int, min_days: int) -> None:
    with pytest.raises(ValueError, match="jitter_days"):
        _worker(min_jitter_days=min_days, max_jitter_days=max_days)


def test_actor_factory_passes_the_range() -> None:
    actor_cls = create_anonymizer_actor(SALT, KEY, min_jitter_days=10, max_jitter_days=20)
    worker_cls = getattr(getattr(actor_cls, "__ray_metadata__", None), "modified_class", actor_cls)
    worker = worker_cls()
    assert (worker.min_jitter_days, worker.max_jitter_days) == (10, 20)


def test_cli_defaults_and_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(cli, "cmd_run", lambda args: captured.update(args=args))
    cli.main(["run", "anonymizer", "--input", "in", "--output", "out"])
    assert (captured["args"].min_jitter_days, captured["args"].max_jitter_days) == (3, 180)
    cli.main(
        ["run", "anonymizer", "--input", "in", "--output", "out", "--min-jitter-days", "10", "--max-jitter-days", "20"]
    )
    assert (captured["args"].min_jitter_days, captured["args"].max_jitter_days) == (10, 20)


@pytest.mark.parametrize(
    "extra",
    [["--min-jitter-days", "2"], ["--min-jitter-days", "30", "--max-jitter-days", "20"], ["--max-jitter-days", "2"]],
)
def test_cli_rejects_invalid_range(monkeypatch: pytest.MonkeyPatch, extra: list[str]) -> None:
    monkeypatch.setattr(cli, "cmd_run", lambda _args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "anonymizer", "--input", "in", "--output", "out", *extra])
    assert exc.value.code == 2


def test_randomness_is_only_in_the_jitter_util() -> None:
    src = Path(sys.modules["tide2"].__path__[0])
    users = {
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if re.search(r"randbelow|random\.(randint|choice)", p.read_text(encoding="utf-8"))
        and "jitter" in p.read_text(encoding="utf-8")
    }
    assert users <= {str(Path("cryptographic/date_jitter.py"))}
