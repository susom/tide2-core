"""Tests for the HIPAA Safe Harbor ZIP code anonymizer, its wiring and its 2020 Census data."""

import argparse
import io
import logging
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from presidio_anonymizer.entities import InvalidParamError

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.anonymizer import create_anonymizer_actor
from tide2.anonymizers import ZipCodeAnonymizer
from tide2.anonymizers import cell
from tide2.anonymizers.guarded import guarded
from tide2.anonymizers.zip_code import MODE_RELAXED
from tide2.anonymizers.zip_code import MODE_STRICT
from tide2.anonymizers.zip_code import SAFE_HARBOR_POPULATION_THRESHOLD
from tide2.anonymizers.zip_code import ZIP_MODE_HIPS
from tide2.anonymizers.zip_code import ZIP_MODES
from tide2.anonymizers.zip_code import anonymize_zip
from tide2.anonymizers.zip_code import load_zip3_population
from tide2.runner import cli
from tide2.runner import local_runner

SALT = b"\x01" * 32
KEY = b"\x02" * 32

# Prefixes at or below 20,000 people in the shipped 2020 table. A data refresh that changes this set must be reviewed.
RESTRICTED_2020 = {
    "036": 13153,
    "059": 3352,
    "102": 15082,
    "202": 0,
    "203": 772,
    "204": 60,
    "205": 44,
    "369": 17596,
    "556": 16415,
    "692": 8626,
    "753": 0,
    "772": 4280,
    "821": 392,
    "823": 14702,
    "878": 17364,
    "879": 16082,
    "884": 16740,
    "893": 10701,
}

SMALL_TABLE = {"021": 20001, "943": 5_000_000, "036": 20000, "100": 0, "600": 12}


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("zip_text", "strict", "relaxed"),
    [
        ("94305", "943", "94305"),
        ("02139", "021", "02139"),  # leading zero, 20,001 people: kept
        ("94305-1234", "943", "94305"),
        ("94305 1234", "943", "94305"),
        ("943051234", "943", "94305"),
        ("  94305  ", "943", "94305"),
        ("03601", "000", "036"),  # exactly 20,000 people: restricted
        ("10001", "000", "100"),  # 0 people
        ("60012", "000", "600"),
        ("77777", "000", "777"),  # prefix missing from the table counts as 0: fail closed
    ],
)
def test_rules_with_injected_population(zip_text: str, strict: str, relaxed: str) -> None:
    assert anonymize_zip(zip_text, MODE_STRICT, SMALL_TABLE) == strict
    assert anonymize_zip(zip_text, MODE_RELAXED, SMALL_TABLE) == relaxed


def test_threshold_boundary() -> None:
    assert SAFE_HARBOR_POPULATION_THRESHOLD == 20000
    table = {"111": 20000, "222": 20001}
    assert anonymize_zip("11100", MODE_STRICT, table) == "000"
    assert anonymize_zip("22200", MODE_STRICT, table) == "222"


@pytest.mark.parametrize("bad", ["", "9430", "2139", "943", "ABCDE", "943051", "94305-12", "9430512345", "１２３４５"])
def test_non_zip_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="5-digit ZIP"):
        anonymize_zip(bad, MODE_STRICT, SMALL_TABLE)


@pytest.mark.parametrize("mode", ["hips", "", None, "strict"])
def test_invalid_mode_is_rejected(mode: Any) -> None:
    with pytest.raises(ValueError, match="zip mode"):
        anonymize_zip("94305", mode, SMALL_TABLE)


# ---------------------------------------------------------------------------
# Leading zeros
# ---------------------------------------------------------------------------


def test_every_leading_zero_prefix_keeps_its_zeros() -> None:
    zero_prefixes = [p for p in load_zip3_population() if p.startswith("0")]
    assert len(zero_prefixes) == 82
    for prefix in zero_prefixes:
        zip5 = prefix + "01"
        strict = anonymize_zip(zip5, MODE_STRICT)
        relaxed = anonymize_zip(zip5, MODE_RELAXED)
        assert isinstance(strict, str)
        assert isinstance(relaxed, str)
        if prefix in RESTRICTED_2020:
            assert (strict, relaxed) == ("000", prefix)
        else:
            assert (strict, relaxed) == (prefix, zip5)
        assert len(strict) == 3
        assert len(relaxed) == (3 if prefix in RESTRICTED_2020 else 5)


def test_restricted_value_is_the_string_000() -> None:
    out = anonymize_zip("03601", MODE_STRICT)
    assert out == "000"
    assert isinstance(out, str)


def test_three_and_four_digit_values_are_masked_not_padded() -> None:
    op = guarded(ZipCodeAnonymizer)()
    params = {"mode": MODE_STRICT, "entity_type": "ZIP_CODE"}
    assert op.operate("2139", params) == "[ZIP_CODE]"
    assert op.operate("021", params) == "[ZIP_CODE]"


def test_output_survives_a_parquet_round_trip() -> None:
    out = [anonymize_zip(z, MODE_STRICT) for z in ("03601", "02139", "94305")]
    buf = io.BytesIO()
    pq.write_table(pa.table({"zip": out}), buf)
    buf.seek(0)
    assert pq.read_table(buf).column("zip").to_pylist() == ["000", "021", "943"]


# ---------------------------------------------------------------------------
# Shipped 2020 data
# ---------------------------------------------------------------------------


def test_data_file_shape_and_total() -> None:
    table = load_zip3_population()
    assert len(table) == 894
    assert all(len(k) == 3 and k.isascii() and k.isdigit() for k in table)
    assert all(isinstance(v, int) and v >= 0 for v in table.values())
    assert sum(table.values()) == 334_726_586


def test_restricted_prefixes_are_pinned() -> None:
    table = load_zip3_population()
    assert {k: v for k, v in table.items() if v <= SAFE_HARBOR_POPULATION_THRESHOLD} == RESTRICTED_2020


def test_near_threshold_prefix_is_kept() -> None:
    assert load_zip3_population()["576"] == 20310
    assert anonymize_zip("57601", MODE_STRICT) == "576"


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


def test_operator_metadata_and_validate() -> None:
    op = ZipCodeAnonymizer()
    assert op.operator_name() == "zip_code_safe_harbor"
    for mode in ZIP_MODES:
        op.validate({"mode": mode})
    for params in ({}, {"mode": "hips"}, {"mode": None}, None):
        with pytest.raises(InvalidParamError):
            op.validate(params)


def test_bare_operator_raises_and_guarded_operator_masks() -> None:
    params = {"mode": MODE_STRICT, "entity_type": "ZIP_CODE"}
    with pytest.raises(ValueError, match="5-digit ZIP"):
        ZipCodeAnonymizer().operate("nope", params)
    assert guarded(ZipCodeAnonymizer)().operate("nope", params) == "[ZIP_CODE]"
    assert guarded(ZipCodeAnonymizer)().operate("02139", params) == "021"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _anonymize_note(worker: AnonymizerWorker, note: str, spans: list[tuple[int, int, str]]) -> str:
    results = [{"entity_type": t, "start": s, "end": e, "score": 1.0} for s, e, t in spans]
    return worker.process_note(
        note_text=note,
        original_text_hash=worker.compute_text_hash(note),
        recognizer_results_json=results,
        patient_id="FAKE-P0001",
        jitter=5,
    )["anonymized_note_text"]


def test_default_mode_keeps_the_hips_operator() -> None:
    worker = AnonymizerWorker(salt=SALT, key=KEY)
    assert worker.zip_mode == ZIP_MODE_HIPS
    assert worker._base_operators["ZIP_CODE"].operator_name == "hips_location"
    assert "ZIP" not in worker._base_operators
    explicit = AnonymizerWorker(salt=SALT, key=KEY, zip_mode="hips")
    note = "Lives at 02139."
    spans = [(9, 14, "ZIP_CODE")]
    assert _anonymize_note(worker, note, spans) == _anonymize_note(explicit, note, spans)


@pytest.mark.parametrize("label", ["ZIP_CODE", "ZIP", "ZIPCODE"])
@pytest.mark.parametrize(("mode", "expected"), [(MODE_STRICT, "021 and 000"), (MODE_RELAXED, "02139 and 036")])
def test_safe_harbor_modes_in_a_note(label: str, mode: str, expected: str) -> None:
    worker = AnonymizerWorker(salt=SALT, key=KEY, zip_mode=mode)
    note = "02139 and 03601"
    out = _anonymize_note(worker, note, [(0, 5, label), (10, 15, label)])
    assert out == expected


def test_zip_with_plus_four_in_a_note() -> None:
    worker = AnonymizerWorker(salt=SALT, key=KEY, zip_mode=MODE_STRICT)
    assert _anonymize_note(worker, "ZIP 94305-1234.", [(4, 14, "ZIP_CODE")]) == "ZIP 943."


def test_non_zip_span_is_masked_in_a_note() -> None:
    worker = AnonymizerWorker(salt=SALT, key=KEY, zip_mode=MODE_STRICT)
    assert _anonymize_note(worker, "ZIP 2139.", [(4, 8, "ZIP_CODE")]) == "ZIP [ZIP_CODE]."


def test_invalid_worker_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="zip mode"):
        AnonymizerWorker(salt=SALT, key=KEY, zip_mode="strict")


def test_actor_factory_forwards_zip_mode() -> None:
    actor = create_anonymizer_actor(SALT, KEY, zip_mode=MODE_RELAXED)()  # ty: ignore[missing-argument]
    assert actor.zip_mode == MODE_RELAXED
    assert actor._base_operators["ZIP_CODE"].operator_name == "zip_code_safe_harbor"


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ZIP_MODES)
@pytest.mark.parametrize("value", ["02139", "03601", "94305-1234", "00601", "2139", "ABCDE"])
def test_cell_matches_process_note(mode: str, value: str) -> None:
    worker = AnonymizerWorker(salt=SALT, key=KEY, zip_mode=mode)
    expected = _anonymize_note(worker, value, [(0, len(value), "ZIP_CODE")])
    assert cell.anonymize_cell(value, "ZIP_CODE", SALT, KEY, zip_mode=mode) == expected


def test_cell_requires_a_valid_zip_mode() -> None:
    for mode in (None, "hips", "strict"):
        with pytest.raises(ValueError, match="zip mode"):
            cell.anonymize_cell("02139", "ZIP_CODE", SALT, KEY, zip_mode=mode)
    # Other entity types ignore zip_mode.
    assert cell.anonymize_cell("48291037", "MRN", SALT, KEY) is not None


def test_cell_nulls_and_empty() -> None:
    assert cell.anonymize_cell(None, "ZIP_CODE", SALT, KEY, zip_mode=MODE_STRICT) is None
    assert cell.anonymize_cell(float("nan"), "ZIP_CODE", SALT, KEY, zip_mode=MODE_STRICT) is None
    assert cell.anonymize_cell("", "ZIP_CODE", SALT, KEY, zip_mode=MODE_STRICT) == ""


def test_cell_pads_integer_zips_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(cell, "_warned_integer_zip", False)
    with caplog.at_level(logging.WARNING, logger=cell.logger.name):
        assert cell.anonymize_cell(2139, "ZIP_CODE", SALT, KEY, zip_mode=MODE_RELAXED) == "02139"
        assert cell.anonymize_cell(601, "ZIP_CODE", SALT, KEY, zip_mode=MODE_STRICT) == "006"
        assert cell.anonymize_cell(0, "ZIP_CODE", SALT, KEY, zip_mode=MODE_STRICT) == "000"
    assert len([r for r in caplog.records if "leading zeros" in r.message]) == 1


@pytest.mark.parametrize("value", [-1, 100000, True])
def test_cell_masks_integers_that_are_not_zips(value: object) -> None:
    assert cell.anonymize_cell(value, "ZIP_CODE", SALT, KEY, zip_mode=MODE_STRICT) == "[ZIP_CODE]"


# ---------------------------------------------------------------------------
# Runner and CLI
# ---------------------------------------------------------------------------


def test_cli_choices_match_the_operator_constants() -> None:
    assert (ZIP_MODE_HIPS, *ZIP_MODES) == cli.ZIP_MODE_CHOICES


def _parse_run_args(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> argparse.Namespace:
    captured: dict[str, argparse.Namespace] = {}
    monkeypatch.setattr(cli, "cmd_run", lambda args: captured.setdefault("args", args))
    monkeypatch.setattr(sys, "argv", ["tide2-runner", *argv])
    cli.main()
    return captured["args"]


BASE_ARGS = ["run", "anonymizer", "-i", "in", "-o", "out", "--salt", "s", "--key", "k"]


def test_cli_zip_mode_default_and_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _parse_run_args(monkeypatch, BASE_ARGS).zip_mode == "hips"
    args = _parse_run_args(monkeypatch, [*BASE_ARGS, "--zip-mode", "safe-harbor-strict"])
    assert args.zip_mode == "safe-harbor-strict"


def test_cli_rejects_an_unknown_zip_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["tide2-runner", *BASE_ARGS, "--zip-mode", "strict"])
    with pytest.raises(SystemExit):
        cli.main()


def test_yaml_zip_mode_backfills_and_cli_wins(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = tmp_path / "runner.yaml"
    config.write_text("zip_mode: safe-harbor-relaxed\n")
    args = _parse_run_args(monkeypatch, [*BASE_ARGS, "--config", str(config)])
    assert args.zip_mode == "safe-harbor-relaxed"
    args = _parse_run_args(monkeypatch, [*BASE_ARGS, "--config", str(config), "--zip-mode", "safe-harbor-strict"])
    assert args.zip_mode == "safe-harbor-strict"


def test_yaml_invalid_zip_mode_exits(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = tmp_path / "runner.yaml"
    config.write_text("zip_mode: strict\n")
    monkeypatch.setattr(sys, "argv", ["tide2-runner", *BASE_ARGS, "--config", str(config)])
    with pytest.raises(SystemExit):
        cli.main()


class _FakeRunner:
    calls: dict[str, dict]

    def __init__(self, **_: object) -> None:
        pass

    def run_anonymization(self, **kwargs: object) -> dict:
        _FakeRunner.calls["anonymizer"] = kwargs
        return {}

    def run_pipeline(self, **kwargs: object) -> dict:
        _FakeRunner.calls["pipeline"] = kwargs
        return {}

    def shutdown(self) -> None:
        pass


@pytest.mark.parametrize("job", ["anonymizer", "pipeline"])
def test_cli_forwards_zip_mode(monkeypatch: pytest.MonkeyPatch, job: str) -> None:
    _FakeRunner.calls = {}
    monkeypatch.setattr(local_runner, "LocalJobRunner", _FakeRunner)
    argv = ["run", job, "-i", "in", "-o", "out", "--zip-mode", "safe-harbor-strict"]
    argv += ["--salt", "s", "--key", "k"] if job == "anonymizer" else ["--model", "m"]
    monkeypatch.setattr(sys, "argv", ["tide2-runner", *argv])
    cli.main()
    sent = _FakeRunner.calls[job]
    assert (sent["zip_mode"] if job == "anonymizer" else sent["anonymizer_kwargs"]["zip_mode"]) == "safe-harbor-strict"


def test_cli_default_does_not_send_zip_mode_to_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeRunner.calls = {}
    monkeypatch.setattr(local_runner, "LocalJobRunner", _FakeRunner)
    monkeypatch.setattr(sys, "argv", ["tide2-runner", "run", "pipeline", "-i", "in", "-o", "out", "--model", "m"])
    cli.main()
    assert "zip_mode" not in _FakeRunner.calls["pipeline"]["anonymizer_kwargs"]


def test_runner_validates_zip_mode_before_starting_ray() -> None:
    bad_modes: list[Any] = ["strict", "", None]
    for mode in bad_modes:
        with pytest.raises(ValueError, match="Invalid zip_mode"):
            local_runner._validate_zip_mode(mode)
    for mode in ("hips", *ZIP_MODES):
        local_runner._validate_zip_mode(mode)
