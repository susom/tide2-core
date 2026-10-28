"""Tests for the birth_date / max_age age cap wiring: cells, actors, stage columns, runner and CLI."""

import logging
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.anonymizer import create_anonymizer_actor
from tide2.actors.recognizer import RecognizerWorker
from tide2.anonymizers import cell
from tide2.runner import cli
from tide2.runner import local_runner
from tide2.runner.local_runner import ANONYMIZER_STAGE_COLUMNS
from tide2.runner.local_runner import FINAL_OUTPUT_COLUMNS
from tide2.runner.local_runner import LLM_RECOGNIZER_STAGE_COLUMNS
from tide2.runner.local_runner import RECOGNIZER_STAGE_COLUMNS
from tide2.runner.local_runner import TRANSFORMER_STAGE_COLUMNS
from tide2.runner.local_runner import _resolve_merged_batch
from tide2.runner.local_runner import validate_stage_columns
from tide2.utils.batch_columns import copy_passthrough
from tide2.utils.batch_columns import type_columns

SALT = b"\x01" * 32
KEY = b"\x02" * 32
# Synthetic: born 1930-06-15, so the 90th birthday is 2020-06-15.
DOB = "1930-06-15"
NOTE = "Seen 2023-03-15."
SPANS = [{"entity_type": "DATE", "start": 5, "end": 15, "score": 1.0}]


def _worker(**kwargs) -> AnonymizerWorker:
    return AnonymizerWorker(salt=SALT, key=KEY, **kwargs)


def _batch(birth_dates: list | None, jitters: list | None = None) -> dict:
    n = len(birth_dates) if birth_dates is not None else 1
    batch = {
        "note_text": [NOTE] * n,
        "recognizer_results_json": [SPANS] * n,
        "jitter": jitters if jitters is not None else [5] * n,
    }
    if birth_dates is not None:
        batch["birth_date"] = birth_dates
    return batch


# --- cells ---------------------------------------------------------------


def test_cell_caps_with_birth_date() -> None:
    assert cell.anonymize_cell("2023-03-15", "DATE", SALT, KEY, jitter=5, birth_date=DOB) == "2020-06-20"
    assert cell.anonymize_cell("2023-03-15", "DATE", SALT, KEY, jitter=5, birth_date=DOB, max_age=80) == "2010-06-20"


def test_cell_matches_process_note_with_birth_date() -> None:
    worker = _worker()
    for value, entity_type in [("2023-03-15", "DATE"), ("2024-06-10 14:30:00", "DATE_TIME"), ("2019-01-01", "DATE")]:
        expected = worker.process_note(
            note_text=value,
            original_text_hash="h",
            recognizer_results_json=[{"entity_type": entity_type, "start": 0, "end": len(value), "score": 1.0}],
            patient_id="FAKE-P0001",
            birth_date=DOB,
        )["anonymized_note_text"]
        assert cell.anonymize_cell(value, entity_type, SALT, KEY, patient_id="FAKE-P0001", birth_date=DOB) == expected


def test_cell_fpe_ignores_birth_date() -> None:
    assert cell.anonymize_cell("48291037", "MRN", SALT, KEY, birth_date=DOB) == cell.anonymize_cell(
        "48291037", "MRN", SALT, KEY
    )


def test_cell_warns_once_without_birth_date(caplog: pytest.LogCaptureFixture) -> None:
    cell._warn_no_birth_date.cache_clear()
    with caplog.at_level(logging.WARNING, logger=cell.__name__):
        for _ in range(3):
            assert cell.anonymize_cell("2023-03-15", "DATE", SALT, KEY, jitter=5) == "2023-03-20"
        cell.anonymize_cell("48291037", "MRN", SALT, KEY)
    assert [r.getMessage() for r in caplog.records] == [
        "anonymize_cell called without birth_date: dates are shifted but not capped at age 90; "
        "ages over 89 may be revealed"
    ]


# --- anonymizer actor ----------------------------------------------------


def test_worker_caps_with_birth_date_and_drops_the_column() -> None:
    out = _worker().process_batch(_batch([DOB, None]))
    assert out["anonymized_note_text"] == ["Seen 2020-06-20.", "Seen 2023-03-20."]
    assert "birth_date" not in out


def test_worker_without_column_is_unchanged() -> None:
    assert _worker().process_batch(_batch(None))["anonymized_note_text"] == ["Seen 2023-03-20."]


def test_worker_max_age_and_factory() -> None:
    assert _worker(max_age=80).process_batch(_batch([DOB]))["anonymized_note_text"] == ["Seen 2010-06-20."]
    actor_cls = create_anonymizer_actor(SALT, KEY, max_age=80)
    worker_cls = getattr(getattr(actor_cls, "__ray_metadata__", None), "modified_class", actor_cls)
    assert worker_cls().max_age == 80


@pytest.mark.parametrize("max_age", [0, 151])
def test_worker_rejects_invalid_max_age(max_age: int) -> None:
    with pytest.raises(ValueError, match="max_age"):
        _worker(max_age=max_age)


def test_worker_warns_once_then_debug_for_null_birth_dates(caplog: pytest.LogCaptureFixture) -> None:
    worker = _worker()
    logger_name = "tide2.actors.anonymizer"
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        worker.process_batch(_batch([None, None, DOB]))
        worker.process_batch(_batch([None]))
        worker.process_batch(_batch([DOB]))
    records = [r for r in caplog.records if "without birth_date" in r.getMessage()]
    assert [r.levelno for r in records] == [logging.WARNING, logging.DEBUG]
    assert records[0].getMessage().startswith("2 note(s) without birth_date")


def test_worker_no_birth_date_warning_for_masked_dates(caplog: pytest.LogCaptureFixture) -> None:
    worker = _worker(jitter_required=True)
    with caplog.at_level(logging.DEBUG, logger="tide2.actors.anonymizer"):
        out = worker.process_batch(_batch([None], jitters=[None]))
    assert out["anonymized_note_text"] == ["Seen [DATE]."]
    assert not [r for r in caplog.records if "without birth_date" in r.getMessage()]


# --- route through the stages --------------------------------------------


def test_recognizer_forwards_birth_date_like_jitter() -> None:
    worker_cls = getattr(RecognizerWorker, "__ray_actor_class__", RecognizerWorker)
    worker = worker_cls.__new__(worker_cls)
    worker.process_note = lambda **_kw: {"recognizer_results_json": "[]", "entity_count": 0}
    batch = {"text_hash": ["h1", "h2"], "note_text": ["a", "b"], "patient_id": ["p1", "p2"]}

    assert "birth_date" not in worker.process_batch(batch)
    out = worker.process_batch({**batch, "jitter": [1, 2], "birth_date": [DOB, None]})
    assert out["birth_date"] == [DOB, None]
    assert out["jitter"] == [1, 2]


def test_copy_passthrough_forwards_birth_date() -> None:
    res: dict = {}
    copy_passthrough({"BIRTH_DATE": [DOB, None], "jitter": [1, 2]}, res)
    assert res["birth_date"] == [DOB, None]
    res = {}
    copy_passthrough({"jitter": [1]}, res)
    assert "birth_date" not in res


def test_type_columns_writes_birth_date_as_iso_string() -> None:
    from datetime import date

    assert type_columns({"birth_date": [date(1930, 6, 15), None]})["birth_date"] == [DOB, None]
    assert type_columns({"birth_date": [None]})["birth_date"].type == pa.string()


def test_merge_coalesces_birth_date() -> None:
    batch = pa.table(
        {
            "row_id": ["r1", "r2"],
            "text_hash_regex": ["h1", "h2"],
            "results_regex": ["[]", "[]"],
            "results_llm": ["[]", "[]"],
            "birth_date_regex": [DOB, None],
            "birth_date_llm": [None, "1940-01-01"],
        }
    )
    assert _resolve_merged_batch(batch).to_pydict()["birth_date"] == [DOB, "1940-01-01"]


def test_stage_contracts_carry_birth_date_but_the_sink_does_not() -> None:
    for contract in (
        TRANSFORMER_STAGE_COLUMNS,
        RECOGNIZER_STAGE_COLUMNS,
        LLM_RECOGNIZER_STAGE_COLUMNS,
        ANONYMIZER_STAGE_COLUMNS,
    ):
        assert "birth_date" in contract.optional
        assert "birth_date" not in contract.produces
    assert "birth_date" not in FINAL_OUTPUT_COLUMNS

    source = ["note_text", "text_hash", "row_id", "birth_date"]
    before_anonymizer = validate_stage_columns(
        source, [("transformer", TRANSFORMER_STAGE_COLUMNS), ("recognizer", RECOGNIZER_STAGE_COLUMNS)]
    )
    assert "birth_date" in before_anonymizer
    final = validate_stage_columns(
        source, [("llm_recognizer", LLM_RECOGNIZER_STAGE_COLUMNS), ("anonymizer", ANONYMIZER_STAGE_COLUMNS)]
    )
    assert "birth_date" not in FINAL_OUTPUT_COLUMNS & final


# --- runner and CLI -------------------------------------------------------


def test_runner_warns_when_the_column_is_missing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=local_runner.__name__):
        local_runner._warn_if_no_birth_date(["note_text", "BIRTH_DATE"], "in/", 90)
        assert not caplog.records
        local_runner._warn_if_no_birth_date(["note_text", "jitter"], "in/", 90)
    assert [r.getMessage() for r in caplog.records] == [
        "No birth_date column in in/: dates are shifted but not capped at age 90; ages over 89 may be revealed"
    ]


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    runner = MagicMock()
    runner.run_anonymization.return_value = {}
    runner.run_pipeline.return_value = {}
    monkeypatch.setattr(local_runner, "LocalJobRunner", lambda **_kw: runner)
    return runner


def test_cli_forwards_max_age_to_anonymizer(fake_runner: MagicMock) -> None:
    base = ["run", "anonymizer", "--input", "in", "--output", "out", "--salt", "s", "--key", "k"]
    cli.main(base)
    assert fake_runner.run_anonymization.call_args.kwargs["max_age"] == 90
    cli.main([*base, "--max-age", "95"])
    assert fake_runner.run_anonymization.call_args.kwargs["max_age"] == 95


def test_cli_forwards_max_age_to_pipeline(fake_runner: MagicMock) -> None:
    cli.main(["run", "pipeline", "--input", "in", "--output", "out", "--model", "m", "--max-age", "95"])
    assert fake_runner.run_pipeline.call_args.kwargs["anonymizer_kwargs"]["max_age"] == 95


@pytest.mark.parametrize("max_age", ["0", "151"])
def test_cli_rejects_invalid_max_age(monkeypatch: pytest.MonkeyPatch, max_age: str) -> None:
    monkeypatch.setattr(cli, "cmd_run", lambda _args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "anonymizer", "--input", "in", "--output", "out", "--max-age", max_age])
    assert exc.value.code == 2
