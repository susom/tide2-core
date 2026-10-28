"""Tests for the HIPAA Safe Harbor ZIP code anonymizer, its wiring and its 2020 Census data."""

import argparse
import inspect
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
from tide2.anonymizers.zip_code import DEFAULT_ZIP_MODE
from tide2.anonymizers.zip_code import MODE_RELAXED
from tide2.anonymizers.zip_code import MODE_STRICT
from tide2.anonymizers.zip_code import SAFE_HARBOR_POPULATION_THRESHOLD
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


@pytest.mark.parametrize("mode", ["hips", "default", "", None, "strict", "relaxed"])
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


def test_default_mode_is_relaxed_and_there_are_only_two_modes() -> None:
    assert DEFAULT_ZIP_MODE == MODE_RELAXED
    assert ZIP_MODES == (MODE_STRICT, MODE_RELAXED)
    worker = AnonymizerWorker(salt=SALT, key=KEY)
    assert worker.zip_mode == MODE_RELAXED
    for entity in ("ZIP_CODE", "ZIP", "ZIPCODE"):
        assert worker._base_operators[entity].operator_name == "zip_code_safe_harbor"
        assert worker._base_operators[entity].params == {"mode": MODE_RELAXED}
    explicit = AnonymizerWorker(salt=SALT, key=KEY, zip_mode=MODE_RELAXED)
    note = "Lives at 02139 and 03601."
    spans = [(9, 14, "ZIP_CODE"), (19, 24, "ZIP_CODE")]
    assert _anonymize_note(worker, note, spans) == _anonymize_note(explicit, note, spans) == "Lives at 02139 and 036."


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
    for mode in ("strict", "hips"):
        with pytest.raises(ValueError, match="zip mode"):
            AnonymizerWorker(salt=SALT, key=KEY, zip_mode=mode)


@pytest.mark.parametrize("entity", ["LOCATION", "HOSPITAL", "VENDOR"])
def test_location_operators_carry_the_zip_mode(entity: str) -> None:
    worker = AnonymizerWorker(salt=SALT, key=KEY, zip_mode=MODE_STRICT)
    assert worker._base_operators[entity].operator_name == "hips_location"
    assert worker._base_operators[entity].params["zip_mode"] == MODE_STRICT


# ---------------------------------------------------------------------------
# ZIP inside an address
# ---------------------------------------------------------------------------

ADDRESS = "123 Main Street, Cambridge, MA 02139"


@pytest.mark.parametrize(
    ("address", "mode", "zip_part"),
    [
        (ADDRESS, MODE_STRICT, "MA 021"),
        (ADDRESS, MODE_RELAXED, "MA 02139"),
        ("123 Main Street, Cambridge, MA 02139-4307", MODE_STRICT, "MA 021"),
        ("123 Main Street, Cambridge, MA 02139-4307", MODE_RELAXED, "MA 02139"),
        ("456 Oak Avenue, Newark, NJ 03601", MODE_STRICT, "NJ 000"),
        ("456 Oak Avenue, Newark, NJ 03601", MODE_RELAXED, "NJ 036"),
    ],
)
def test_zip_inside_an_address_follows_the_mode(address: str, mode: str, zip_part: str) -> None:
    worker = AnonymizerWorker(salt=SALT, key=KEY, zip_mode=mode)
    out = _anonymize_note(worker, address, [(0, len(address), "LOCATION")])
    assert out.endswith(zip_part)
    assert "-4307" not in out


def test_the_address_zip_is_not_a_random_real_zip_any_more() -> None:
    worker = AnonymizerWorker(salt=SALT, key=KEY)
    out = _anonymize_note(worker, ADDRESS, [(0, len(ADDRESS), "LOCATION")])
    assert out.endswith("MA 02139")


def test_address_with_a_malformed_zip_drops_it_and_is_degraded() -> None:
    from tide2.anonymizers.guarded import end_note
    from tide2.anonymizers.guarded import start_note
    from tide2.anonymizers.guarded import summarize
    from tide2.anonymizers.hips_locations import HipsLocationAnonymizer

    op = guarded(HipsLocationAnonymizer)()
    events = start_note()
    try:
        out = op.operate(
            "789 Pine Road, Springfield, IL 6280",
            {"salt": SALT, "key": KEY, "entity_type": "LOCATION", "zip_mode": MODE_STRICT},
        )
        status = summarize(events)
    finally:
        end_note()
    assert out.endswith(", IL")
    assert "6280" not in out
    assert status == ("degraded", "LOCATION")


def test_hips_location_without_zip_mode_keeps_the_random_zip() -> None:
    from tide2.anonymizers.hips_locations import HipsLocationAnonymizer

    out = HipsLocationAnonymizer().operate(ADDRESS, {"salt": SALT, "key": KEY, "entity_type": "LOCATION"})
    assert not out.endswith("MA 02139")
    assert out.rsplit(" ", 1)[1].isdigit()


def test_hips_location_validates_zip_mode() -> None:
    from tide2.anonymizers.hips_locations import HipsLocationAnonymizer

    op = HipsLocationAnonymizer()
    base = {"salt": SALT, "key": KEY, "entity_type": "LOCATION"}
    op.validate(base)
    op.validate({**base, "zip_mode": MODE_STRICT})
    with pytest.raises(ValueError, match="zip_mode"):
        op.validate({**base, "zip_mode": "hips"})


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


def test_cell_defaults_to_relaxed_and_rejects_invalid_modes() -> None:
    assert cell.anonymize_cell("02139", "ZIP_CODE", SALT, KEY) == "02139"
    assert cell.anonymize_cell("03601", "ZIP_CODE", SALT, KEY) == "036"
    for mode in ("hips", "strict"):
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


def test_cli_and_runner_defaults_match_the_operator_constants() -> None:
    assert ZIP_MODES == cli.ZIP_MODE_CHOICES
    assert cli.DEFAULT_ZIP_MODE == DEFAULT_ZIP_MODE
    run_anonymization = inspect.signature(local_runner.LocalJobRunner.run_anonymization)
    assert run_anonymization.parameters["zip_mode"].default == DEFAULT_ZIP_MODE
    assert inspect.signature(create_anonymizer_actor).parameters["zip_mode"].default == DEFAULT_ZIP_MODE


def _parse_run_args(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> argparse.Namespace:
    captured: dict[str, argparse.Namespace] = {}
    monkeypatch.setattr(cli, "cmd_run", lambda args: captured.setdefault("args", args))
    monkeypatch.setattr(sys, "argv", ["tide2-runner", *argv])
    cli.main()
    return captured["args"]


BASE_ARGS = ["run", "anonymizer", "-i", "in", "-o", "out", "--salt", "s", "--key", "k"]


def test_cli_zip_mode_default_and_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _parse_run_args(monkeypatch, BASE_ARGS).zip_mode == "safe-harbor-relaxed"
    args = _parse_run_args(monkeypatch, [*BASE_ARGS, "--zip-mode", "safe-harbor-strict"])
    assert args.zip_mode == "safe-harbor-strict"


@pytest.mark.parametrize("mode", ["strict", "relaxed", "hips", "default"])
def test_cli_rejects_an_unknown_zip_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(sys, "argv", ["tide2-runner", *BASE_ARGS, "--zip-mode", mode])
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
@pytest.mark.parametrize(
    ("flag", "expected"), [(["--zip-mode", "safe-harbor-strict"], "safe-harbor-strict"), ([], "safe-harbor-relaxed")]
)
def test_cli_forwards_zip_mode(monkeypatch: pytest.MonkeyPatch, job: str, flag: list[str], expected: str) -> None:
    _FakeRunner.calls = {}
    monkeypatch.setattr(local_runner, "LocalJobRunner", _FakeRunner)
    argv = ["run", job, "-i", "in", "-o", "out", *flag]
    argv += ["--salt", "s", "--key", "k"] if job == "anonymizer" else ["--model", "m"]
    monkeypatch.setattr(sys, "argv", ["tide2-runner", *argv])
    cli.main()
    sent = _FakeRunner.calls[job]
    assert (sent["zip_mode"] if job == "anonymizer" else sent["anonymizer_kwargs"]["zip_mode"]) == expected


def test_runner_validates_zip_mode_before_starting_ray() -> None:
    bad_modes: list[Any] = ["strict", "hips", "", None]
    for mode in bad_modes:
        with pytest.raises(ValueError, match="Invalid zip_mode"):
            local_runner._validate_zip_mode(mode)
    for mode in ZIP_MODES:
        local_runner._validate_zip_mode(mode)
