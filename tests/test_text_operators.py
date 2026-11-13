"""Tests for text_operators: a method per entity type in free text, and the method argument of cells."""

import json
from unittest.mock import MagicMock

import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.anonymizer import create_anonymizer_actor
from tide2.anonymizers import cell
from tide2.anonymizers import free_text
from tide2.anonymizers.cell import AnonymizationPolicy
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


def _note(**policy_kwargs) -> dict:
    worker = AnonymizerWorker(SALT, KEY, policy=AnonymizationPolicy(**policy_kwargs))
    return worker.process_note(TEXT, worker.compute_text_hash(TEXT), json.dumps(SPANS), "FAKE-P0001", 7)


# --- parse_text_operators -----------------------------------------------------


def test_parse_accepts_none_string_mapping_and_list() -> None:
    expected = (("AGE", "keep"), ("MRN", "keep"))
    assert cell.parse_text_operators(None) == ()
    assert cell.parse_text_operators("") == ()
    assert cell.parse_text_operators(" age = KEEP, mrn=keep ") == expected
    assert cell.parse_text_operators({"mrn": "keep", "Age": "keep"}) == expected
    assert cell.parse_text_operators(["mrn=keep", "age=keep"]) == expected
    assert cell.parse_text_operators(iter(expected)) == expected


def test_parse_date_is_not_expanded_to_the_part_types() -> None:
    assert cell.parse_text_operators("DATE=keep") == (("DATE", "keep"),)


@pytest.mark.parametrize(
    "value",
    [
        "NOPE=keep",
        "DEFAULT=keep",
        "FREE_TEXT=keep",
        "DATE_DAY=keep",
        "DATE=nope",
        "DATE",
        "DATE=keep,date=keep",
        "MRN=acc_num_hash",
    ],
)
def test_parse_rejects_bad_items(value: str) -> None:
    with pytest.raises(ValueError, match=r"text_operators|Entity type"):
        cell.parse_text_operators(value)


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
def test_text_operators_do_not_change_a_structured_cell(value: str, entity_type: str) -> None:
    plain = cell.anonymize_cell(value, entity_type, SALT, KEY, patient_id="FAKE-P0001")
    policy = AnonymizationPolicy(text_operators={entity_type: "keep"})
    assert cell.anonymize_cell(value, entity_type, SALT, KEY, patient_id="FAKE-P0001", policy=policy) == plain
    assert plain != value


def test_cell_date_parts_are_still_jittered_when_date_is_kept() -> None:
    kwargs = {"patient_id": "FAKE-P0001", "full_date": "2024-03-15"}
    policy = AnonymizationPolicy(text_operators="DATE=keep")
    for entity_type, value in [("DATE_DAY", 15), ("DATE_MONTH", 3), ("DATE_YEAR", 2024)]:
        plain = cell.anonymize_cell(value, entity_type, SALT, KEY, **kwargs)
        assert cell.anonymize_cell(value, entity_type, SALT, KEY, policy=policy, **kwargs) == plain


def test_cell_null_and_empty_stay_null_and_empty() -> None:
    policy = AnonymizationPolicy(text_operators="MRN=keep")
    assert cell.anonymize_cell(None, "MRN", SALT, KEY, policy=policy) is None
    assert cell.anonymize_cell("", "MRN", SALT, KEY, policy=policy) == ""


def test_cell_keep_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="free text only"):
        cell.anonymize_cell("48291037", "MRN", SALT, KEY, method="keep")


@pytest.mark.parametrize(
    ("entity_type", "method", "kwargs"),
    [
        ("MRN", "acc_num_hash", {"acc_num_salt": "s", "acc_num_study_id": "u"}),
        ("FREE_TEXT", "acc_num_hash", {"acc_num_salt": "s", "acc_num_study_id": "u"}),
        ("ACC_NUM", "nope", {}),
        ("ACC_NUM", "acc_num_hash", {"acc_num_study_id": "u"}),
        ("ACC_NUM", "acc_num_hash", {"acc_num_salt": "s"}),
        ("ACC_NUM", "acc_num_hash", {"acc_num_salt": "s", "acc_num_study_id": ""}),
    ],
)
def test_cell_method_fails_closed(entity_type: str, method: str, kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"method|FREE_TEXT|acc_num"):
        cell.anonymize_cell("ACC123456", entity_type, SALT, KEY, method=method, **kwargs)


def test_cell_acc_num_hash_matches_the_note_and_the_default_is_fpe() -> None:
    policy = AnonymizationPolicy(text_operators={"ACC_NUM": "acc_num_hash"})
    secrets = {"acc_num_salt": "fake-acc-salt", "acc_num_study_id": "FAKE-STUDY"}
    kwargs = {"patient_id": "FAKE-P0001", "policy": policy, **secrets}
    hashed = cell.anonymize_cell("ACC123456", "ACC_NUM", SALT, KEY, method="acc_num_hash", **kwargs)
    worker = AnonymizerWorker(SALT, KEY, policy=policy, **secrets)
    span = [{"entity_type": "ACC_NUM", "start": 0, "end": 9, "score": 1.0}]
    note = worker.process_note("ACC123456", worker.compute_text_hash("ACC123456"), json.dumps(span), "FAKE-P0001", 0)
    assert hashed == note["anonymized_note_text"]
    assert cell.anonymize_cell("ACC123456", "ACC_NUM", SALT, KEY, **kwargs) != hashed


# --- free text -----------------------------------------------------------------


def test_free_text_cell_keeps_listed_spans_only() -> None:
    result = cell.anonymize_cell(
        TEXT,
        "FREE_TEXT",
        SALT,
        KEY,
        patient_id="FAKE-P0001",
        jitter=7,
        recognizer_results=json.dumps(SPANS),
        policy=AnonymizationPolicy(text_operators="DATE=keep"),
    )
    assert "03/14/2025" in result
    for value in ("Rhea Orleans", "48291037", "555-019-3344", "Springfield"):
        assert value not in result


def test_free_text_cell_matches_the_worker() -> None:
    expected = _note(text_operators="DATE=keep,PERSON=keep")["anonymized_note_text"]
    result = cell.anonymize_cell(
        TEXT,
        "FREE_TEXT",
        SALT,
        KEY,
        patient_id="FAKE-P0001",
        jitter=7,
        recognizer_results=json.dumps(SPANS),
        policy=AnonymizationPolicy(text_operators="DATE=keep,PERSON=keep"),
    )
    assert result == expected


def test_worker_records_the_keep_operator() -> None:
    result = _note(text_operators="DATE=keep,LOCATION=keep")
    items = {item["entity_type"]: item for item in json.loads(result["anonymizer_results_json"])}
    assert items["DATE"]["operator"] == "keep"
    assert items["LOCATION"]["operator"] == "keep"
    assert items["PERSON"]["operator"] != "keep"
    assert "Springfield" in result["anonymized_note_text"]
    assert "Rhea Orleans" not in result["anonymized_note_text"]


def test_an_empty_map_changes_nothing() -> None:
    assert _note()["anonymized_note_text"] == _note(text_operators={})["anonymized_note_text"]
    assert "keep" not in _note()["anonymizer_results_json"]


def test_kept_dates_are_not_failed_when_jitter_is_required() -> None:
    worker = AnonymizerWorker(
        SALT, KEY, policy=AnonymizationPolicy(jitter_required=True, text_operators="DATE=keep,DATE_TIME=keep")
    )
    result = worker.process_note(TEXT, worker.compute_text_hash(TEXT), json.dumps(SPANS), "FAKE-P0001", None)
    assert result["stage_status"] == "success"
    assert "03/14/2025" in result["anonymized_note_text"]

    only_date = AnonymizerWorker(
        SALT, KEY, policy=AnonymizationPolicy(jitter_required=True, text_operators="DATE=keep")
    )
    result = only_date.process_note(TEXT, only_date.compute_text_hash(TEXT), json.dumps(SPANS), "FAKE-P0001", None)
    assert result["stage_status"] == "failed"


def test_factory_passes_text_operators() -> None:
    actor = create_anonymizer_actor(SALT, KEY, policy=AnonymizationPolicy(text_operators="DATE=keep"))()
    assert actor.policy.text_operators == (("DATE", "keep"),)


# --- runner and CLI -------------------------------------------------------------


def test_runner_rejects_unknown_names_before_ray_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = local_runner.LocalJobRunner()

    def fail(*_args, **_kwargs):
        raise AssertionError("the job started")

    monkeypatch.setattr(runner, "_init_ray", fail)
    with pytest.raises(ValueError, match="text_operators"):
        runner.run_anonymization("in", "out", salt_hex="00" * 32, key_hex="11" * 32, text_operators="NOPE=keep")
    with pytest.raises(ValueError, match="text_operators"):
        runner.run_pipeline("in", "out", "model", anonymizer_kwargs={"text_operators": "NOPE=keep"})


@pytest.mark.parametrize("execution_mode", ["discrete", "streamed"])
def test_pipeline_forwards_text_operators_given_as_a_one_shot_iterable(
    monkeypatch: pytest.MonkeyPatch, tmp_path, execution_mode: str
) -> None:
    runner = local_runner.LocalJobRunner()
    forwarded: dict = {}

    def capture(*_args, **kwargs):
        forwarded.update(kwargs.get("anonymizer_kwargs") or _args[-1])
        raise RuntimeError("stop")

    monkeypatch.setattr(local_runner, "resolve_input_files", lambda _path: ["in.parquet"])
    monkeypatch.setattr(local_runner, "_inspect_pipeline_input", lambda _files: MagicMock())
    monkeypatch.setattr(runner, "_streamed_fallback_reason", lambda **_kw: None)
    monkeypatch.setattr(runner, "_run_pipeline_streamed", capture)
    monkeypatch.setattr(runner, "_apply_pipeline_recommendations", capture)

    with pytest.raises(RuntimeError, match="stop"):
        runner.run_pipeline(
            "in",
            str(tmp_path),
            "model",
            salt_hex="00" * 32,
            key_hex="11" * 32,
            anonymizer_kwargs={"text_operators": iter(["DATE=keep"])},
            execution_mode=execution_mode,
        )

    assert cell.parse_text_operators(forwarded["text_operators"]) == (("DATE", "keep"),)


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    runner = MagicMock()
    runner.run_anonymization.return_value = {}
    runner.run_pipeline.return_value = {}
    monkeypatch.setattr(local_runner, "LocalJobRunner", lambda **_kw: runner)
    return runner


def test_cli_forwards_text_operators_to_anonymizer_and_pipeline(fake_runner: MagicMock) -> None:
    base = ["run", "anonymizer", "--input", "in", "--output", "out", "--salt", "s", "--key", "k"]
    cli.main(base)
    assert fake_runner.run_anonymization.call_args.kwargs["text_operators"] is None
    cli.main([*base, "--text-operators", "DATE=keep,age=keep"])
    assert fake_runner.run_anonymization.call_args.kwargs["text_operators"] == "DATE=keep,age=keep"

    cli.main(["run", "pipeline", "--input", "in", "--output", "out", "--model", "m", "--text-operators", "DATE=keep"])
    assert fake_runner.run_pipeline.call_args.kwargs["anonymizer_kwargs"]["text_operators"] == "DATE=keep"


def test_cli_reads_text_operators_from_the_config_file(fake_runner: MagicMock, tmp_path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text("text_operators: {DATE: keep, AGE: keep}\n")
    cli.main(
        ["run", "anonymizer", "--input", "in", "--output", "out", "--salt", "s", "--key", "k", "--config", str(config)]
    )
    assert fake_runner.run_anonymization.call_args.kwargs["text_operators"] == {"DATE": "keep", "AGE": "keep"}


def test_cli_rejects_unknown_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "cmd_run", lambda _args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "anonymizer", "--input", "in", "--output", "out", "--text-operators", "DATE=keep,NOPE=keep"])
    assert exc.value.code == 2
