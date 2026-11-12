"""Tests for AnonymizationPolicy and its use in the CLI and the runner."""

import dataclasses
from unittest.mock import MagicMock

import pytest

from tide2.anonymizers import cell
from tide2.anonymizers.cell import AnonymizationPolicy
from tide2.anonymizers.cell import parse_text_operators
from tide2.cryptographic.date_jitter import DEFAULT_MAX_AGE
from tide2.cryptographic.date_jitter import DEFAULT_MAX_JITTER_DAYS
from tide2.cryptographic.date_jitter import DEFAULT_MIN_JITTER_DAYS
from tide2.cryptographic.date_jitter import validate_jitter_parameters
from tide2.runner import cli
from tide2.runner import local_runner

RUN = ["run", "anonymizer", "--input", "in", "--output", "out", "--salt", "s", "--key", "k"]


def test_defaults() -> None:
    policy = AnonymizationPolicy()
    assert policy.min_jitter_days == DEFAULT_MIN_JITTER_DAYS
    assert policy.max_jitter_days == DEFAULT_MAX_JITTER_DAYS
    assert policy.max_age == DEFAULT_MAX_AGE
    assert policy.text_operators == ()
    assert policy.acc_num_study_id is None
    assert policy.jitter_required is False


def test_frozen_and_hashable() -> None:
    policy = AnonymizationPolicy(text_operators=(("AGE", "keep"),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.max_age = 90  # ty: ignore[invalid-assignment]
    assert policy == AnonymizationPolicy.from_mapping({"text_operators": ["age=keep"]})
    assert hash(policy) == hash(AnonymizationPolicy.from_mapping({"text_operators": {"AGE": "keep"}}))


def test_text_operators_parsed() -> None:
    policy = AnonymizationPolicy.from_mapping({"text_operators": "date=keep, age=KEEP"})
    assert policy.text_operators == parse_text_operators("date=keep, age=KEEP") == (("AGE", "keep"), ("DATE", "keep"))
    assert policy.text_operator("DATE") == "keep"
    assert policy.text_operator("MRN") is None


def test_bad_jitter() -> None:
    _, message = validate_jitter_parameters(DEFAULT_MAX_JITTER_DAYS, 1)
    with pytest.raises(ValueError, match=message):
        AnonymizationPolicy(min_jitter_days=1)


@pytest.mark.parametrize("max_age", [0, 151])
def test_bad_max_age(max_age: int) -> None:
    with pytest.raises(ValueError, match="max_age"):
        AnonymizationPolicy(max_age=max_age)


def test_whole_number_options_become_ints() -> None:
    policy = AnonymizationPolicy(min_jitter_days=3.0, max_jitter_days="180", max_age=90.0)
    assert (policy.min_jitter_days, policy.max_jitter_days, policy.max_age) == (3, 180, 90)
    assert all(type(v) is int for v in (policy.min_jitter_days, policy.max_jitter_days, policy.max_age))
    assert cell.anonymize_cell("2024-03-15", "DATE", b"s" * 32, b"k" * 32, patient_id=None, policy=policy)


@pytest.mark.parametrize("bad", [3.5, True, "x", None])
def test_fractional_or_non_numeric_jitter_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="min_jitter_days must be an integer"):
        AnonymizationPolicy(min_jitter_days=bad)


@pytest.mark.parametrize("bad", [[None], [["DATE"]], 5])
def test_malformed_text_operators_raise_value_error(bad: object) -> None:
    with pytest.raises(ValueError, match="text_operators"):
        AnonymizationPolicy(text_operators=bad)


def test_unknown_text_operator() -> None:
    with pytest.raises(ValueError, match="text_operators"):
        AnonymizationPolicy.from_mapping({"text_operators": "NOPE=keep"})


def test_acc_num_hash_needs_study_id() -> None:
    with pytest.raises(ValueError, match="acc_num_study_id"):
        AnonymizationPolicy(text_operators={"ACC_NUM": "acc_num_hash"})
    policy = AnonymizationPolicy(text_operators={"ACC_NUM": "acc_num_hash"}, acc_num_study_id="S1")
    assert policy.text_operator("ACC_NUM") == "acc_num_hash"


def test_from_mapping_ignores_other_keys() -> None:
    assert AnonymizationPolicy.from_mapping({"num_actors": 4, "max_age": None}) == AnonymizationPolicy()


def test_no_secrets() -> None:
    names = [field.name for field in dataclasses.fields(AnonymizationPolicy)]
    assert not [name for name in names if "salt" in name or "key" in name]


@pytest.mark.parametrize(
    "extra",
    [["--text-operators", "ACC_NUM=acc_num_hash"], ["--min-jitter-days", "1"], ["--max-age", "0"]],
)
def test_cli_rejects(monkeypatch: pytest.MonkeyPatch, extra: list[str]) -> None:
    monkeypatch.setattr(cli, "cmd_run", lambda _args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main([*RUN, *extra])
    assert exc.value.code == 2


def test_runners_reject_before_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = local_runner.LocalJobRunner()
    init_ray = MagicMock()
    monkeypatch.setattr(runner, "_init_ray", init_ray)
    with pytest.raises(ValueError, match="acc_num_study_id"):
        runner.run_anonymization(
            "in", "out", salt_hex="00" * 32, key_hex="11" * 32, text_operators="ACC_NUM=acc_num_hash"
        )
    with pytest.raises(ValueError, match="acc_num_study_id"):
        runner.run_pipeline("in", "out", "model", anonymizer_kwargs={"text_operators": "ACC_NUM=acc_num_hash"})
    init_ray.assert_not_called()


def test_fpe_mode() -> None:
    assert AnonymizationPolicy().fpe_mode == "strict"
    with pytest.raises(ValueError, match="fpe_mode"):
        AnonymizationPolicy(fpe_mode="x")


def test_cli_fpe_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = MagicMock()
    runner.run_anonymization.return_value = {}
    runner.run_pipeline.return_value = {}
    monkeypatch.setattr(local_runner, "LocalJobRunner", lambda **_kw: runner)
    cli.main([*RUN, "--fpe-mode", "relaxed"])
    assert runner.run_anonymization.call_args.kwargs["fpe_mode"] == "relaxed"
    cli.main(["run", "pipeline", "--input", "in", "--output", "out", "--model", "m", "--fpe-mode", "relaxed"])
    assert runner.run_pipeline.call_args.kwargs["anonymizer_kwargs"]["fpe_mode"] == "relaxed"
    monkeypatch.setattr(cli, "cmd_run", lambda _args: None)
    with pytest.raises(SystemExit) as exc:
        cli.main([*RUN, "--fpe-mode", "x"])
    assert exc.value.code == 2
