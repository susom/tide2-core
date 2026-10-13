"""Tests for the machine detection and the hardware rules in resources/hardware_rules.yaml.

The tests need no Ray cluster and no GPU: machines are built by hand.
"""

import argparse
import inspect
import subprocess
import sys

import pytest

from tide2.runner import cli
from tide2.runner import hardware
from tide2.runner.hardware import Machine
from tide2.runner.hardware import apply_settings
from tide2.runner.hardware import detect_machine
from tide2.runner.hardware import extract_gpu_family
from tide2.runner.hardware import load_rules
from tide2.runner.hardware import recommend
from tide2.runner.hardware import render_settings
from tide2.runner.local_runner import LocalJobRunner
from tide2.transformers.config import CANONICAL_MODEL_NAME
from tide2.transformers.config import DEPRECATED_MODEL_NAME
from tide2.transformers.config import load_model_config

MODEL = CANONICAL_MODEL_NAME

# Machines used below. The reference box is the one every measurement was taken on.
REFERENCE = Machine(cpus=16, ram_gb=64.0, gpus=1, gpu_name="NVIDIA L4", gpu_ram_gb=22.5)
CPU_ONLY = Machine(cpus=16, ram_gb=64.0)
TINY_CPU = Machine(cpus=2, ram_gb=8.0)
TINY_GPU = Machine(cpus=2, ram_gb=8.0, gpus=1, gpu_name="Tesla T4", gpu_ram_gb=15.0)
LARGE_CPU = Machine(cpus=128, ram_gb=512.0)
GPU_SERVER = Machine(cpus=64, ram_gb=256.0, gpus=1, gpu_name="NVIDIA L4", gpu_ram_gb=22.5)
LOW_RAM = Machine(cpus=16, ram_gb=16.0)
T4_BOX = Machine(cpus=16, ram_gb=64.0, gpus=1, gpu_name="Tesla T4", gpu_ram_gb=15.0)
TWO_GPUS = Machine(cpus=32, ram_gb=128.0, gpus=2, gpu_name="NVIDIA L4", gpu_ram_gb=22.5)

# ---------------------------------------------------------------------------
# One row per kind of hardware. To add a rule to the YAML, add a row here.
#   machine, rules that match, then the settings that must come out.
# ---------------------------------------------------------------------------

HARDWARE_TABLE = [
    pytest.param(
        REFERENCE,
        ("cpu-stages", "gpu"),
        {
            "transformer": {
                "num_transformer_actors": 3,
                "num_gpus": 0.33,
                "transformer_cpus": 4.0,
                "num_agg_actors": 0,
                "override_num_blocks": 16,
                "gpu_batch_size": 64,
                "batch_size": 512,
            },
            "recognizer": {"num_actors": 14, "worker_num_cpus": 1.0, "override_num_blocks": 32},
            "anonymizer": {"num_actors": 14, "worker_num_cpus": 1.0, "override_num_blocks": 32},
            "runner": {"no_progress_timeout_s": 600},
        },
        id="reference-L4-16cpu",
    ),
    pytest.param(
        CPU_ONLY,
        ("cpu-stages", "cpu-only", "cpu-only-slow-start"),
        {
            "transformer": {
                "num_transformer_actors": 4,
                "num_gpus": 0.0,
                "num_agg_actors": 4,
                "override_num_blocks": 16,
            },
            "recognizer": {"num_actors": 14},
            "runner": {"no_progress_timeout_s": 1200},
        },
        id="cpu-only-16cpu",
    ),
    pytest.param(
        LARGE_CPU,
        ("cpu-stages", "many-cores", "cpu-only"),
        {
            "transformer": {"num_transformer_actors": 32, "num_agg_actors": 38},
            "recognizer": {"num_actors": 126},
            "runner": {"no_progress_timeout_s": 600},
        },
        id="large-cpu-128cpu",
    ),
    pytest.param(
        GPU_SERVER,
        ("cpu-stages", "many-cores", "gpu"),
        {"transformer": {"num_transformer_actors": 3}, "recognizer": {"num_actors": 62}},
        id="gpu-server-64cpu",
    ),
    pytest.param(
        T4_BOX,
        ("cpu-stages", "gpu"),
        {"transformer": {"num_transformer_actors": 2, "num_gpus": 0.5}},
        id="T4-16cpu-fewer-actors-for-15GB",
    ),
    pytest.param(
        TWO_GPUS,
        ("cpu-stages", "gpu"),
        {"transformer": {"num_transformer_actors": 6, "num_gpus": 0.33}},
        id="two-L4",
    ),
    pytest.param(
        LOW_RAM,
        ("cpu-stages", "cpu-only", "cpu-only-slow-start", "low-ram"),
        {"recognizer": {"num_actors": 8}, "anonymizer": {"num_actors": 8}},
        id="16cpu-16GB-ram-fewer-actors",
    ),
    pytest.param(
        TINY_CPU,
        ("low-ram", "tiny"),
        {
            "transformer": {
                "num_transformer_actors": 1,
                "num_gpus": 0,
                "transformer_cpus": 0.25,
                "read_cpus": 0.25,
                "write_cpus": 0.25,
                "agg_num_cpus": 0.5,
                "num_agg_actors": 1,
                "enable_checkpoint": False,
            },
            "recognizer": {
                "num_actors": 1,
                "worker_num_cpus": 0.5,
                "read_cpus": 0.25,
                "write_cpus": 0.25,
                "enable_checkpoint": False,
            },
            "anonymizer": {"num_actors": 1, "worker_num_cpus": 0.5, "enable_checkpoint": False},
            "runner": {"no_progress_timeout_s": 1200},
        },
        id="tiny-2cpu",
    ),
    pytest.param(
        TINY_GPU,
        ("low-ram", "tiny"),
        {"transformer": {"num_gpus": 1, "num_transformer_actors": 1}, "runner": {"no_progress_timeout_s": 1200}},
        id="tiny-2cpu-T4",
    ),
]


@pytest.mark.parametrize(("machine", "rule_names", "expected"), HARDWARE_TABLE)
def test_hardware_table(machine, rule_names, expected):
    settings = recommend(machine, MODEL)

    assert settings.rules == rule_names
    for stage, knobs in expected.items():
        for knob, value in knobs.items():
            assert getattr(settings, stage)[knob] == value, f"{stage}.{knob}"


@pytest.mark.parametrize(
    ("cpus", "gpus", "rule_names"),
    [
        (4, 0, ("tiny",)),
        (5, 0, ("cpu-stages", "cpu-only", "cpu-only-slow-start")),
        (63, 0, ("cpu-stages", "cpu-only", "cpu-only-slow-start")),
        (64, 0, ("cpu-stages", "many-cores", "cpu-only")),
        (4, 1, ("tiny",)),
        (5, 1, ("cpu-stages", "gpu")),
    ],
)
def test_rule_boundaries(cpus, gpus, rule_names):
    machine = Machine(cpus=cpus, ram_gb=64.0, gpus=gpus, gpu_name="NVIDIA L4" if gpus else None, gpu_ram_gb=22.5 * gpus)
    assert recommend(machine).rules == rule_names


def test_without_rules_a_machine_gets_the_safe_defaults():
    settings = recommend(REFERENCE, use_rules=False)

    assert settings.rules == ()
    assert settings.recognizer["num_actors"] == 8
    assert settings.transformer["num_transformer_actors"] == 1
    assert settings.transformer["num_gpus"] == 1
    assert settings.transformer["num_agg_actors"] == 0
    assert settings.runner == {"no_progress_timeout_s": 600}
    assert set(settings.sources.values()) == {"default"}


def test_defaults_without_a_gpu_run_on_cpu():
    settings = recommend(CPU_ONLY, use_rules=False)

    assert settings.transformer["num_gpus"] == 0
    assert settings.transformer["num_agg_actors"] == 1


# ---------------------------------------------------------------------------
# A user can add hardware with a file of their own
# ---------------------------------------------------------------------------

CUSTOM_RULES = """
gpu_families: [L4]
defaults:
  transformer: {num_transformer_actors: 1, num_gpus: gpus, num_agg_actors: 0}
  cpu_stages: {num_actors: 2}
rules:
  - name: my-big-gpu
    when: {min_gpu_ram_gb: 40}
    let: {per_gpu: "gpu_ram_gb // 10"}
    transformer: {num_transformer_actors: "gpus * per_gpu"}
"""


def test_a_rules_file_of_your_own_adds_hardware(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(CUSTOM_RULES)
    big = Machine(cpus=32, ram_gb=256.0, gpus=1, gpu_name="NVIDIA H100", gpu_ram_gb=80.0)

    custom = recommend(big, rules_path=str(path))
    packaged = recommend(big)

    assert custom.rules == ("my-big-gpu",)
    assert custom.transformer["num_transformer_actors"] == 8
    assert packaged.transformer["num_transformer_actors"] == 3


def test_the_environment_variable_selects_the_rules_file(tmp_path, monkeypatch):
    path = tmp_path / "rules.yaml"
    path.write_text(CUSTOM_RULES)
    monkeypatch.setenv(hardware.RULES_ENV_VAR, str(path))

    assert recommend(REFERENCE).rules == ()
    assert recommend(REFERENCE).recognizer["num_actors"] == 2


VALID_DEFAULTS = "defaults: {transformer: {num_gpus: 0, num_transformer_actors: 1, num_agg_actors: 1}, cpu_stages: {num_actors: 1}}\n"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("rulez: []\n", "unknown key"),
        (VALID_DEFAULTS + "rules: [{name: a, whenn: {}}]\n", "unknown key"),
        (VALID_DEFAULTS + "rules: [{name: a, when: {min_cpu: 1}}]\n", "unknown key"),
        ("defaults: {transformer: {num_gpus: 1}}\n", "must set"),
    ],
)
def test_a_misspelled_or_incomplete_rules_file_fails_loudly(tmp_path, text, message):
    path = tmp_path / "bad.yaml"
    path.write_text(text)

    with pytest.raises(ValueError, match=message):
        load_rules(str(path))


@pytest.mark.parametrize(
    "formula",
    ["__import__('os').system('true')", "cpus.__class__", "open('x')", "[1, 2]", "unknown_name + 1", "cpus if"],
)
def test_formulas_cannot_run_code(formula):
    with pytest.raises((ValueError, SyntaxError)):
        hardware._evaluate(formula, {"cpus": 4})


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("max(1, cpus // 2)", 2),
        ("round(cpus * 0.875)", 4),
        ("0 if cpus > 8 else 1", 1),
        ("-cpus + 10", 6),
        ("1 < cpus <= 4", True),
    ],
)
def test_formulas_evaluate(formula, expected):
    assert hardware._evaluate(formula, {"cpus": 4}) == expected


# ---------------------------------------------------------------------------
# The packaged rules only name settings that the stage jobs accept
# ---------------------------------------------------------------------------


def test_every_knob_in_the_packaged_rules_is_a_parameter_of_its_stage():
    """A misspelled knob would otherwise fail only on the hardware that matches its rule."""
    accepted = {
        "transformer": set(inspect.signature(LocalJobRunner.run_transformer).parameters),
        "recognizer": set(inspect.signature(LocalJobRunner.run_recognition).parameters),
        "anonymizer": set(inspect.signature(LocalJobRunner.run_anonymization).parameters),
    }
    rules = load_rules()
    layers = [rules["defaults"], *rules["rules"], *({"transformer": r["transformer"]} for r in rules["model_rules"])]
    for layer in layers:
        for stage in ("transformer", "recognizer", "anonymizer"):
            knobs = set(layer.get(stage) or {})
            if stage != "transformer":
                knobs |= set(layer.get("cpu_stages") or {})
            assert knobs <= accepted[stage], (
                f"{layer.get('name', 'defaults')}: {stage} {sorted(knobs - accepted[stage])}"
            )


def test_every_machine_in_the_table_resolves_every_formula():
    for machine in (REFERENCE, CPU_ONLY, TINY_CPU, TINY_GPU, LARGE_CPU, GPU_SERVER, LOW_RAM, T4_BOX, TWO_GPUS):
        assert recommend(machine, MODEL).recognizer["num_actors"] >= 1


# ---------------------------------------------------------------------------
# Precedence: a value the caller passed always wins
# ---------------------------------------------------------------------------

ALL_KNOBS = [
    ("transformer", "num_transformer_actors", 99),
    ("transformer", "num_gpus", 0.75),
    ("transformer", "transformer_cpus", 8.0),
    ("transformer", "gpu_batch_size", 128),
    ("transformer", "batch_size", 1024),
    ("transformer", "num_agg_actors", 4),
    ("transformer", "override_num_blocks", 64),
    ("transformer", "enable_checkpoint", True),
    ("recognizer", "num_actors", 25),
    ("recognizer", "worker_num_cpus", 2.0),
    ("recognizer", "override_num_blocks", 64),
    ("recognizer", "enable_checkpoint", True),
    ("recognizer", "read_cpus", 0.5),
    ("anonymizer", "num_actors", 25),
    ("anonymizer", "worker_num_cpus", 2.0),
    ("anonymizer", "write_cpus", 2.0),
    ("runner", "no_progress_timeout_s", 999),
]


@pytest.mark.parametrize(("stage", "knob", "user_value"), ALL_KNOBS)
@pytest.mark.parametrize("machine", [REFERENCE, TINY_CPU, CPU_ONLY, LARGE_CPU], ids=["gpu", "tiny", "cpu", "large"])
def test_a_value_the_caller_passed_wins(stage, knob, user_value, machine):
    kwargs = {"transformer": {}, "recognizer": {}, "anonymizer": {}, "runner": {}}
    kwargs[stage][knob] = user_value

    entries = apply_settings(recommend(machine, MODEL), **kwargs)

    assert kwargs[stage][knob] == user_value
    entry = next(e for e in entries if e.stage == stage and e.knob == knob)
    assert (entry.value, entry.source) == (user_value, "USER")


def test_none_counts_as_unset():
    kwargs = {"transformer": {"num_gpus": None}, "recognizer": {"num_actors": None}, "anonymizer": {}, "runner": {}}

    apply_settings(recommend(REFERENCE), **kwargs)

    assert kwargs["transformer"]["num_gpus"] == 0.33
    assert kwargs["recognizer"]["num_actors"] == 14


def test_every_entry_names_where_it_came_from():
    kwargs = {"transformer": {"gpu_batch_size": 32}, "recognizer": {}, "anonymizer": {}, "runner": {}}
    settings = recommend(REFERENCE, MODEL)

    entries = {(e.stage, e.knob): e.source for e in apply_settings(settings, **kwargs)}

    assert entries[("transformer", "gpu_batch_size")] == "USER"
    assert entries[("transformer", "num_transformer_actors")] == "gpu"
    assert entries[("recognizer", "num_actors")] == "cpu-stages"
    assert entries[("recognizer", "worker_num_cpus")] == "default"
    assert entries[("transformer", "batch_size")] == "measured L4"


def test_render_settings():
    kwargs = {"transformer": {"gpu_batch_size": 64}, "recognizer": {}, "anonymizer": {}, "runner": {}}
    settings = recommend(REFERENCE, MODEL)

    table = render_settings(settings, apply_settings(settings, **kwargs))

    assert "Detected: 16 CPU | 64 GB RAM | 1× NVIDIA L4 (22.5 GB)" in table
    assert f"Rules:    cpu-stages, gpu   Model: {MODEL} (measured, L4)" in table
    assert "num_transformer_actors" in table
    assert "USER" in table


# ---------------------------------------------------------------------------
# Measured models
# ---------------------------------------------------------------------------


def test_an_unmeasured_model_gets_no_batch_size():
    settings = recommend(REFERENCE, "StanfordAIMI/stanford-deidentifier-v2")

    assert "gpu_batch_size" not in settings.transformer
    assert "batch_size" not in settings.transformer
    assert settings.model_status == "(unmeasured)"


def test_a_measured_model_without_enough_gpu_memory_gets_no_batch_size():
    settings = recommend(Machine(cpus=16, ram_gb=64.0, gpus=1, gpu_name="NVIDIA L4", gpu_ram_gb=6.0), MODEL)

    assert "gpu_batch_size" not in settings.transformer
    assert settings.model_status == "(insufficient VRAM)"


def test_a_measured_model_on_another_gpu_family_gets_no_batch_size():
    settings = recommend(T4_BOX, MODEL)

    assert "gpu_batch_size" not in settings.transformer
    assert settings.model_status == "(unmeasured for T4)"


def test_a_measured_model_on_the_measured_gpu_gets_its_batch_sizes():
    settings = recommend(REFERENCE, MODEL)

    assert (settings.transformer["gpu_batch_size"], settings.transformer["batch_size"]) == (64, 512)
    assert settings.model_status == "(measured, L4)"


def test_the_deprecated_model_name_gets_the_same_batch_sizes():
    with pytest.deprecated_call(match="20260211_debertav3_finetuned.*deprecated"):
        deprecated = recommend(REFERENCE, DEPRECATED_MODEL_NAME)

    assert deprecated.transformer == recommend(REFERENCE, MODEL).transformer


def test_resolving_the_deprecated_model_key_keeps_its_own_explanation():
    with pytest.deprecated_call(match="20260211_debertav3_finetuned.*deprecated"):
        deprecated = load_model_config(DEPRECATED_MODEL_NAME)
    canonical = load_model_config(MODEL)

    assert {k for k in canonical if deprecated[k] != canonical[k]} == {"DEFAULT_EXPLANATION"}


@pytest.mark.parametrize(
    ("gpu_name", "family"),
    [
        ("NVIDIA L4", "L4"),
        ("Tesla T4", "T4"),
        ("NVIDIA A100-SXM4-80GB", "A100"),
        ("NVIDIA A10G", "A10G"),
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA L40", None),
        ("NVIDIA L40S", None),
        ("RTX 4090", None),
        (None, None),
    ],
)
def test_gpu_family_matches_whole_words(gpu_name, family):
    assert extract_gpu_family(gpu_name, load_rules()["gpu_families"]) == family


# ---------------------------------------------------------------------------
# Detection reads only this machine
# ---------------------------------------------------------------------------


def test_detect_machine_does_not_start_ray():
    import ray

    if ray.is_initialized():
        ray.shutdown()
    detect_machine.cache_clear()

    machine = detect_machine()

    assert not ray.is_initialized()
    assert machine.cpus >= 1
    assert machine.gpus >= 0


def _fake_nvidia_smi(monkeypatch, output):
    def run(*_args, **_kwargs):
        if output is None:
            raise FileNotFoundError("nvidia-smi")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=output)

    monkeypatch.setattr(hardware.subprocess, "run", run)


def test_gpus_are_read_from_nvidia_smi(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _fake_nvidia_smi(monkeypatch, "NVIDIA L4, 23034\nNVIDIA L4, 23034\n")

    assert hardware._detect_gpus() == (2, "NVIDIA L4", 22.5)


def test_cuda_visible_devices_limits_the_gpus(monkeypatch):
    _fake_nvidia_smi(monkeypatch, "NVIDIA L4, 23034\nNVIDIA L4, 23034\n")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert hardware._detect_gpus()[0] == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert hardware._detect_gpus() == (0, None, 0.0)


def test_no_nvidia_smi_means_no_gpu(monkeypatch):
    _fake_nvidia_smi(monkeypatch, None)

    assert hardware._detect_gpus() == (0, None, 0.0)


# ---------------------------------------------------------------------------
# The runner uses the detected machine, with its own overrides on top
# ---------------------------------------------------------------------------


def test_runner_overrides_win_over_detection(monkeypatch):
    import tide2.runner.local_runner as lr

    monkeypatch.setattr(lr, "detect_machine", lambda: REFERENCE)

    assert LocalJobRunner()._machine() == REFERENCE
    assert LocalJobRunner(num_cpus=4)._machine().cpus == 4
    assert LocalJobRunner(num_gpus=0.33)._machine().gpus == 1
    assert LocalJobRunner(num_gpus=0)._machine().gpus == 0


def test_runner_without_autotune_uses_only_the_safe_defaults(monkeypatch):
    import tide2.runner.local_runner as lr

    monkeypatch.setattr(lr, "detect_machine", lambda: REFERENCE)

    assert LocalJobRunner(hardware_autotune=True)._settings().rules == ("cpu-stages", "gpu")
    assert LocalJobRunner(hardware_autotune=False)._settings().rules == ()
    assert LocalJobRunner(hardware_autotune=False)._auto_num_actors() == 8


@pytest.mark.parametrize("hardware_autotune", [True, False])
def test_init_ray_leaves_object_store_to_ray_by_default(monkeypatch, hardware_autotune):
    """Without object_store_gb the runner passes no object_store_memory, so Ray applies its own caps."""
    import tide2.runner.local_runner as lr

    captured: dict = {}
    monkeypatch.setattr(lr.ray, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(lr, "log_ray_cluster_info", lambda: None)
    monkeypatch.setattr(lr, "configure_data_context", lambda **_kwargs: None)

    LocalJobRunner(hardware_autotune=hardware_autotune)._init_ray()

    assert "object_store_memory" not in captured


def test_init_ray_applies_an_explicit_object_store_gb(monkeypatch):
    import tide2.runner.local_runner as lr

    captured: dict = {}
    monkeypatch.setattr(lr.ray, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(lr, "log_ray_cluster_info", lambda: None)
    monkeypatch.setattr(lr, "configure_data_context", lambda **_kwargs: None)

    LocalJobRunner(object_store_gb=2)._init_ray()

    assert captured["object_store_memory"] == 2 * 1024**3


def test_no_progress_timeout_is_sticky_across_stages():
    """Stages re-apply the runner's timeout instead of resetting it to the default."""
    runner = LocalJobRunner(no_progress_timeout_s=1200)
    assert runner._data_context_kwargs(verbose_progress=True) == {
        "verbose_progress": True,
        "no_progress_timeout_s": 1200,
    }


def test_no_progress_timeout_unset_leaves_library_default():
    runner = LocalJobRunner()
    assert runner._data_context_kwargs(verbose_progress=True) == {"verbose_progress": True}


# ---------------------------------------------------------------------------
# CLI / YAML resolution of the opt-out flag
# ---------------------------------------------------------------------------


def _parse_run_args(monkeypatch, argv: list[str]) -> argparse.Namespace:
    """Run the CLI parser (and YAML merge) without executing the job."""
    captured: dict[str, argparse.Namespace] = {}

    def fake_cmd_run(args: argparse.Namespace) -> None:
        captured["args"] = args

    monkeypatch.setattr(cli, "cmd_run", fake_cmd_run)
    monkeypatch.setattr(sys, "argv", ["tide2-runner", *argv])
    cli.main()
    return captured["args"]


@pytest.mark.parametrize(("argv", "expected"), [([], None), (["--no-hardware-autotune"], False)])
def test_cli_hardware_autotune_flag(monkeypatch, argv, expected):
    """Unset stays None so YAML can still speak; the flag turns autotune off."""
    args = _parse_run_args(monkeypatch, ["run", "pipeline", "-i", "in.parquet", "-o", "out", *argv])
    assert args.hardware_autotune is expected


@pytest.mark.parametrize("yaml_value", [True, False])
def test_yaml_hardware_autotune_is_honoured(monkeypatch, tmp_path, yaml_value):
    """A YAML `hardware_autotune: false` must reach cmd_run."""
    config = tmp_path / "runner.yaml"
    config.write_text(f"hardware_autotune: {str(yaml_value).lower()}\n")

    args = _parse_run_args(monkeypatch, ["run", "pipeline", "-i", "in.parquet", "-o", "out", "--config", str(config)])
    assert args.hardware_autotune is yaml_value


def test_cli_flag_beats_yaml(monkeypatch, tmp_path):
    config = tmp_path / "runner.yaml"
    config.write_text("hardware_autotune: true\n")

    args = _parse_run_args(
        monkeypatch,
        ["run", "pipeline", "-i", "in.parquet", "-o", "out", "--config", str(config), "--no-hardware-autotune"],
    )
    assert args.hardware_autotune is False


def test_the_test_table_covers_every_packaged_rule():
    """Adding a rule to the YAML without a row in HARDWARE_TABLE leaves it untested."""
    covered = {name for param in HARDWARE_TABLE for name in param.values[1]}
    assert covered >= {rule["name"] for rule in load_rules()["rules"]}
