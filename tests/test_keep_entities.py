"""Tests for keep_entities: entity types left as they are in cells, free text and the runner."""

import json
from unittest.mock import MagicMock

import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.anonymizer import create_anonymizer_actor
from tide2.anonymizers import cell
from tide2.anonymizers import free_text
from tide2.runner import cli
from tide2.runner import local_runner

SALT = b"\x01" * 32
KEY = b"\x02" * 32

# Synthetic text only.
TEXT = "Patient Rhea Orleans, MRN 48291037, seen on 03/14/2025 at 555-019-3344 in Springfield."


def _span(value: str, entity_type: str) -> dict:
    start = TEXT.index(value)
    return {"entity_type": entity_type, "start": start, "end": start + len(value), "score": 1.0}


SPANS = [
    _span("Rhea Orleans", "PERSON"),
    _span("48291037", "MRN"),
    _span("03/14/2025", "DATE"),
    _span("555-019-3344", "PHONE_NUMBER"),
    _span("Springfield", "LOCATION"),
]


@pytest.fixture(autouse=True)
def _fresh_workers():
    free_text._worker.cache_clear()
    yield
    free_text._worker.cache_clear()


def _note(**worker_kwargs) -> dict:
    worker = AnonymizerWorker(SALT, KEY, **worker_kwargs)
    return worker.process_note(TEXT, worker.compute_text_hash(TEXT), json.dumps(SPANS), "FAKE-P0001", 7)


# --- parse_keep_entities ----------------------------------------------------


def test_parse_accepts_none_string_and_list() -> None:
    assert cell.parse_keep_entities(None) == frozenset()
    assert cell.parse_keep_entities("") == frozenset()
    assert cell.parse_keep_entities(" age, mrn ") == {"AGE", "MRN"}
    assert cell.parse_keep_entities(["person", "Doctor"]) == {"PERSON", "DOCTOR"}


def test_parse_date_covers_the_part_types_but_not_date_time() -> None:
    assert cell.parse_keep_entities("DATE") == {"DATE", "DATE_DAY", "DATE_MONTH", "DATE_YEAR"}
    assert cell.parse_keep_entities("DATE_MONTH") == {"DATE_MONTH"}
    assert "DATE_TIME" not in cell.parse_keep_entities("DATE")


@pytest.mark.parametrize("name", ["NOPE", "DEFAULT", "FREE_TEXT"])
def test_parse_rejects_unknown_default_and_free_text(name: str) -> None:
    with pytest.raises(ValueError, match="keep_entities"):
        cell.parse_keep_entities(f"DATE,{name}")


def test_keepable_types_cover_the_operator_map() -> None:
    worker = AnonymizerWorker(SALT, KEY)
    mapped = set(worker._create_operators_for_note(0)) - {"DEFAULT"}
    assert mapped <= cell.KEEPABLE_ENTITY_TYPES
    assert {"DATE", "DATE_TIME"} <= cell.KEEPABLE_ENTITY_TYPES


# --- cells --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "entity_type"),
    [("48291037", "MRN"), ("555-019-3344", "PHONE_NUMBER"), ("2024-03-15", "DATE"), ("2024-03-15 10:20", "DATE_TIME")],
)
def test_cell_of_a_kept_type_is_returned_as_is(value: str, entity_type: str) -> None:
    kept = cell.parse_keep_entities(entity_type)
    assert cell.anonymize_cell(value, entity_type, SALT, KEY, patient_id="FAKE-P0001", keep_entities=kept) == value
    assert cell.anonymize_cell(value, entity_type, SALT, KEY, patient_id="FAKE-P0001") != value


def test_cell_of_a_type_not_listed_is_unchanged() -> None:
    kept = cell.parse_keep_entities("DATE")
    plain = cell.anonymize_cell("48291037", "MRN", SALT, KEY)
    assert cell.anonymize_cell("48291037", "MRN", SALT, KEY, keep_entities=kept) == plain != "48291037"


def test_cell_date_keeps_the_part_types_and_date_month_alone_keeps_only_the_month() -> None:
    date_kept = cell.parse_keep_entities("DATE")
    for entity_type, value in [("DATE_DAY", 3.0), ("DATE_MONTH", 4), ("DATE_YEAR", "2024")]:
        assert cell.anonymize_cell(value, entity_type, SALT, KEY, keep_entities=date_kept) == str(int(value))

    month_kept = cell.parse_keep_entities("DATE_MONTH")
    kwargs = {"patient_id": "FAKE-P0001", "full_date": "2024-03-15", "keep_entities": month_kept}
    assert cell.anonymize_cell(3, "DATE_MONTH", SALT, KEY, **kwargs) == "3"
    assert cell.anonymize_cell(15, "DATE_DAY", SALT, KEY, **kwargs) != "15"


def test_cell_null_and_empty_stay_null_and_empty() -> None:
    kept = cell.parse_keep_entities("MRN")
    assert cell.anonymize_cell(None, "MRN", SALT, KEY, keep_entities=kept) is None
    assert cell.anonymize_cell("", "MRN", SALT, KEY, keep_entities=kept) == ""


# --- free text -----------------------------------------------------------------


def test_free_text_cell_keeps_listed_spans_only() -> None:
    kept = cell.parse_keep_entities("DATE")
    result = cell.anonymize_cell(
        TEXT,
        "FREE_TEXT",
        SALT,
        KEY,
        patient_id="FAKE-P0001",
        jitter=7,
        recognizer_results=json.dumps(SPANS),
        keep_entities=kept,
    )
    assert "03/14/2025" in result
    for value in ("Rhea Orleans", "48291037", "555-019-3344", "Springfield"):
        assert value not in result


def test_free_text_cell_matches_the_worker() -> None:
    kept = cell.parse_keep_entities("DATE,PERSON")
    expected = _note(keep_entities=kept)["anonymized_note_text"]
    result = cell.anonymize_cell(
        TEXT,
        "FREE_TEXT",
        SALT,
        KEY,
        patient_id="FAKE-P0001",
        jitter=7,
        recognizer_results=json.dumps(SPANS),
        keep_entities=kept,
    )
    assert result == expected


def test_worker_records_the_keep_operator() -> None:
    result = _note(keep_entities=frozenset({"DATE", "LOCATION"}))
    items = {item["entity_type"]: item for item in json.loads(result["anonymizer_results_json"])}
    assert items["DATE"]["operator"] == "keep"
    assert items["LOCATION"]["operator"] == "keep"
    assert items["PERSON"]["operator"] != "keep"
    assert "Springfield" in result["anonymized_note_text"]
    assert "Rhea Orleans" not in result["anonymized_note_text"]


def test_an_empty_set_changes_nothing() -> None:
    assert _note()["anonymized_note_text"] == _note(keep_entities=frozenset())["anonymized_note_text"]
    assert "keep" not in _note()["anonymizer_results_json"]


def test_kept_dates_are_not_failed_when_jitter_is_required() -> None:
    worker = AnonymizerWorker(SALT, KEY, jitter_required=True, keep_entities=frozenset({"DATE", "DATE_TIME"}))
    result = worker.process_note(TEXT, worker.compute_text_hash(TEXT), json.dumps(SPANS), "FAKE-P0001", None)
    assert result["stage_status"] == "success"
    assert "03/14/2025" in result["anonymized_note_text"]

    only_date = AnonymizerWorker(SALT, KEY, jitter_required=True, keep_entities=frozenset({"DATE"}))
    result = only_date.process_note(TEXT, only_date.compute_text_hash(TEXT), json.dumps(SPANS), "FAKE-P0001", None)
    assert result["stage_status"] == "failed"


def test_factory_passes_keep_entities() -> None:
    actor = create_anonymizer_actor(SALT, KEY, keep_entities=frozenset({"DATE"}))()
    assert actor.keep_entities == {"DATE"}


# --- runner and CLI -------------------------------------------------------------


def test_runner_rejects_unknown_names_before_ray_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = local_runner.LocalJobRunner()

    def fail(*_args, **_kwargs):
        raise AssertionError("the job started")

    monkeypatch.setattr(runner, "_init_ray", fail)
    with pytest.raises(ValueError, match="keep_entities"):
        runner.run_anonymization("in", "out", salt_hex="00" * 32, key_hex="11" * 32, keep_entities="NOPE")
    with pytest.raises(ValueError, match="keep_entities"):
        runner.run_pipeline("in", "out", "model", anonymizer_kwargs={"keep_entities": "NOPE"})


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    runner = MagicMock()
    runner.run_anonymization.return_value = {}
    runner.run_pipeline.return_value = {}
    monkeypatch.setattr(local_runner, "LocalJobRunner", lambda **_kw: runner)
    return runner


def test_cli_forwards_keep_entities_to_anonymizer_and_pipeline(fake_runner: MagicMock) -> None:
    base = ["run", "anonymizer", "--input", "in", "--output", "out", "--salt", "s", "--key", "k"]
    cli.main(base)
    assert fake_runner.run_anonymization.call_args.kwargs["keep_entities"] is None
    cli.main([*base, "--keep-entities", "DATE,age"])
    assert fake_runner.run_anonymization.call_args.kwargs["keep_entities"] == "DATE,age"

    cli.main(["run", "pipeline", "--input", "in", "--output", "out", "--model", "m", "--keep-entities", "DATE"])
    assert fake_runner.run_pipeline.call_args.kwargs["anonymizer_kwargs"]["keep_entities"] == "DATE"


def test_cli_reads_keep_entities_from_the_config_file(fake_runner: MagicMock, tmp_path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text("keep_entities: [DATE, AGE]\n")
    cli.main(
        ["run", "anonymizer", "--input", "in", "--output", "out", "--salt", "s", "--key", "k", "--config", str(config)]
    )
    assert fake_runner.run_anonymization.call_args.kwargs["keep_entities"] == ["DATE", "AGE"]


def test_cli_rejects_unknown_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "cmd_run", lambda _args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "anonymizer", "--input", "in", "--output", "out", "--keep-entities", "DATE,NOPE"])
    assert exc.value.code == 2
