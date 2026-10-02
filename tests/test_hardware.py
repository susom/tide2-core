"""Tests for hardware detection and per-stage settings recommendation.

Validates the pure-function contracts of hardware autotuning without requiring
a live Ray cluster:
1. Profile matching over synthetic NodeShapes at boundaries.
2. Cluster vs node distinction and heterogeneous cluster handling.
3. Precedence: user-supplied knobs always win with source="USER".
4. Model gating: measured-only batch sizes, VRAM/GPU-family checks, deprecation alias.
5. Deprecation is non-destructive: old explanation preserved.
6. Reference-box parity: exact equality with reference literals.
7. Opt-out: hardware_autotune=False preserves legacy defaults.
8. Small-box pairing: fractional CPUs + enable_checkpoint=False emitted together.
"""

import argparse
import inspect
import sys

import pytest

from tide2.runner import cli
from tide2.runner.hardware import CANONICAL_MODEL
from tide2.runner.hardware import DEPRECATED_MODEL
from tide2.runner.hardware import HardwareFacts
from tide2.runner.hardware import NodeShape
from tide2.runner.hardware import apply_recommendations
from tide2.runner.hardware import classify_profile
from tide2.runner.hardware import detect_hardware
from tide2.runner.hardware import extract_gpu_family
from tide2.runner.hardware import recommend_object_store_gb
from tide2.runner.hardware import recommend_settings
from tide2.runner.hardware import render_settings_table
from tide2.runner.local_runner import LocalJobRunner
from tide2.transformers.config import load_model_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_facts(
    cpu_count: float,
    gpu_count: float,
    gpu_name: str | None = None,
    vram_gb: float | None = None,
    ram_gb: float = 64.0,
    cluster_nodes: int = 1,
    homogeneous: bool = True,
    profile_override: str | None = None,
) -> HardwareFacts:
    """Construct synthetic HardwareFacts for testing."""
    node = NodeShape(
        cpu_count=cpu_count,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
    )
    nodes = tuple(node for _ in range(cluster_nodes))
    profile = profile_override or classify_profile(node, homogeneous)
    return HardwareFacts(
        cluster_cpu=cpu_count * cluster_nodes,
        cluster_gpu=gpu_count * cluster_nodes,
        nodes=nodes,
        homogeneous=homogeneous,
        node=node if homogeneous else None,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# 1. Profile matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cpu_count", "gpu_count", "expected_profile"),
    [
        # Small-box boundary, at most four CPUs
        (4.0, 0.0, "small-box-cpu"),
        (4.0, 1.0, "small-box-gpu"),
        # Mid-size boundary, above four and below sixty-four CPUs
        (5.0, 0.0, "cpu-only"),
        (5.0, 1.0, "gpu-workstation"),
        (16.0, 0.0, "cpu-only"),
        (16.0, 1.0, "gpu-workstation"),
        (63.0, 0.0, "cpu-only"),
        (63.0, 1.0, "gpu-workstation"),
        # Large boundary, sixty-four CPUs or more
        (64.0, 0.0, "large-cpu"),
        (64.0, 1.0, "gpu-server"),
        (128.0, 0.0, "large-cpu"),
        (128.0, 2.0, "gpu-server"),
    ],
)
def test_profile_matching_boundaries(cpu_count, gpu_count, expected_profile):
    """Classify profiles along both (gpu_present, cpu_class) axes across all boundaries."""
    node = NodeShape(
        cpu_count=cpu_count,
        gpu_count=gpu_count,
        gpu_name="NVIDIA L4" if gpu_count > 0 else None,
        vram_gb=24.0 if gpu_count > 0 else None,
        ram_gb=64.0,
    )
    assert classify_profile(node, homogeneous=True) == expected_profile


def test_profile_matching_v2_ordering_fixes():
    """Verify the two critical cases that 1-D ordered lists misclassified."""
    # 64 CPU + GPU must match gpu-server, NOT large-cpu
    node_64_gpu = NodeShape(cpu_count=64.0, gpu_count=1.0, gpu_name="NVIDIA L4", vram_gb=24.0, ram_gb=128.0)
    assert classify_profile(node_64_gpu, homogeneous=True) == "gpu-server"

    # 4 CPU + GPU must match small-box-gpu, NOT cpu-only or small-box-cpu
    node_4_gpu = NodeShape(cpu_count=4.0, gpu_count=1.0, gpu_name="NVIDIA L4", vram_gb=24.0, ram_gb=16.0)
    assert classify_profile(node_4_gpu, homogeneous=True) == "small-box-gpu"


def test_profile_matching_unknown():
    """Heterogeneous clusters or missing nodes yield unknown profile."""
    assert classify_profile(None, homogeneous=True) == "unknown"
    node = NodeShape(cpu_count=16.0, gpu_count=1.0, gpu_name="NVIDIA L4", vram_gb=24.0, ram_gb=64.0)
    assert classify_profile(node, homogeneous=False) == "unknown"


# ---------------------------------------------------------------------------
# 2. Cluster vs node
# ---------------------------------------------------------------------------


def test_cluster_vs_node_scaling():
    """Multi-node cluster matches on node shape; counts scale with cluster CPUs."""
    # 14 nodes of 16 CPUs with 1 L4 GPU each (production-style topology, 224 CPUs total)
    single_node = NodeShape(
        cpu_count=16.0,
        gpu_count=1.0,
        gpu_name="NVIDIA L4",
        vram_gb=24.0,
        ram_gb=64.0,
    )
    nodes = tuple(single_node for _ in range(14))
    hw = HardwareFacts(
        cluster_cpu=224.0,
        cluster_gpu=14.0,
        nodes=nodes,
        homogeneous=True,
        node=single_node,
        profile=classify_profile(single_node, homogeneous=True),
    )

    # Must match gpu-workstation based on the 16-CPU node, NOT large-cpu based on 224 cluster CPUs
    assert hw.profile == "gpu-workstation"

    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)
    # Autotuning is restricted to single-node, single-GPU; multi-node recommendations are withheld
    assert rec.transformer == {}
    assert rec.recognizer == {}
    assert rec.anonymizer == {}
    assert rec.runner == {}


def test_multi_gpu_single_node_withholds_recommendations():
    """Multi-GPU single-node setup withholds recommendations."""
    hw = make_facts(cpu_count=128.0, gpu_count=2.0, gpu_name="NVIDIA L4", vram_gb=24.0)
    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)
    assert rec.transformer == {}
    assert rec.recognizer == {}
    assert rec.anonymizer == {}
    assert rec.runner == {}


def test_ray_uninitialized_guard():
    """detect_hardware does not auto-initialize Ray when Ray is stopped."""
    import ray

    if ray.is_initialized():
        ray.shutdown()

    hw = detect_hardware()
    assert not ray.is_initialized()
    assert hw.node is not None


def test_multi_node_ray_shapes_withhold_gpu_model():
    """_shapes_from_ray_nodes withholds gpu_name and vram_gb on multi-node clusters."""
    from tide2.runner.hardware import _shapes_from_ray_nodes

    fake_nodes = [
        {"Alive": True, "Resources": {"CPU": 16.0, "GPU": 1.0, "memory": 64 * 1024**3}},
        {"Alive": True, "Resources": {"CPU": 16.0, "GPU": 1.0, "memory": 64 * 1024**3}},
    ]
    shapes = _shapes_from_ray_nodes(fake_nodes)
    assert len(shapes) == 2
    for s in shapes:
        assert s.gpu_name is None
        assert s.vram_gb is None


def test_heterogeneous_cluster_yields_empty_recommendations():
    """Heterogeneous cluster produces profile='unknown' and empty recommendations."""
    node1 = NodeShape(cpu_count=16.0, gpu_count=1.0, gpu_name="NVIDIA L4", vram_gb=24.0, ram_gb=64.0)
    node2 = NodeShape(cpu_count=8.0, gpu_count=0.0, gpu_name=None, vram_gb=None, ram_gb=32.0)
    hw = HardwareFacts(
        cluster_cpu=24.0,
        cluster_gpu=1.0,
        nodes=(node1, node2),
        homogeneous=False,
        node=None,
        profile="unknown",
    )

    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)
    assert rec.profile == "unknown"
    assert rec.transformer == {}
    assert rec.recognizer == {}
    assert rec.anonymizer == {}
    assert rec.runner == {}


# ---------------------------------------------------------------------------
# 3. Precedence: user-supplied knobs always win
# ---------------------------------------------------------------------------


ALL_KNOBS = [
    ("transformer", "num_transformer_actors", 99),
    ("transformer", "num_gpus", 0.75),
    ("transformer", "transformer_cpus", 8.0),
    ("transformer", "gpu_batch_size", 128),
    ("transformer", "batch_size", 1024),
    ("transformer", "num_agg_actors", 4),
    ("transformer", "override_num_blocks", 64),
    ("transformer", "enable_checkpoint", False),
    ("recognizer", "num_actors", 25),
    ("recognizer", "worker_num_cpus", 2.0),
    ("recognizer", "override_num_blocks", 64),
    ("recognizer", "enable_checkpoint", False),
    ("recognizer", "read_cpus", 0.5),
    ("recognizer", "write_cpus", 2.0),
    ("anonymizer", "num_actors", 25),
    ("anonymizer", "worker_num_cpus", 2.0),
    ("anonymizer", "override_num_blocks", 64),
    ("anonymizer", "enable_checkpoint", False),
    ("anonymizer", "read_cpus", 0.5),
    ("anonymizer", "write_cpus", 2.0),
    ("runner", "no_progress_timeout_s", 999),
    ("runner", "object_store_gb", 50.0),
]


@pytest.mark.parametrize(("stage", "knob", "user_value"), ALL_KNOBS)
@pytest.mark.parametrize("profile", ["gpu-workstation", "small-box-cpu", "cpu-only", "large-cpu"])
def test_precedence_user_value_wins_exhaustively(stage, knob, user_value, profile):
    """Any knob passed by user must be preserved with source='USER' across all profiles."""
    hw = make_facts(cpu_count=16.0, gpu_count=1.0, gpu_name="NVIDIA L4", vram_gb=24.0, profile_override=profile)
    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)

    stage_kwargs = {
        "transformer": {},
        "recognizer": {},
        "anonymizer": {},
        "runner": {},
    }
    stage_kwargs[stage][knob] = user_value

    applied = apply_recommendations(
        rec,
        transformer=stage_kwargs["transformer"],
        recognizer=stage_kwargs["recognizer"],
        anonymizer=stage_kwargs["anonymizer"],
        runner=stage_kwargs["runner"],
    )

    target_dict = getattr(applied, stage)
    assert target_dict[knob] == user_value

    entry = next(e for e in applied.entries if e.stage == stage and e.knob == knob)
    assert entry.value == user_value
    assert entry.source == "USER"


# ---------------------------------------------------------------------------
# 4. Model gating
# ---------------------------------------------------------------------------


def test_model_gating_unmeasured_model():
    """Unmeasured model yields no gpu_batch_size and no transformer batch_size."""
    hw = make_facts(cpu_count=16.0, gpu_count=1.0, gpu_name="NVIDIA L4", vram_gb=24.0)
    rec = recommend_settings(hw, model_name="StanfordAIMI/stanford-deidentifier-v2")

    assert "gpu_batch_size" not in rec.transformer
    assert "batch_size" not in rec.transformer
    assert rec.model_status == "(unmeasured)"


def test_model_gating_insufficient_vram():
    """Measured model on insufficient VRAM yields no batch size recommendations."""
    # Measured peak VRAM is 6.77 GB; 6.0 GB is below min_vram_gb (7.77 GB)
    hw = make_facts(cpu_count=16.0, gpu_count=1.0, gpu_name="NVIDIA L4", vram_gb=6.0)
    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)

    assert "gpu_batch_size" not in rec.transformer
    assert "batch_size" not in rec.transformer
    assert rec.model_status == "(insufficient VRAM)"


def test_model_gating_non_matching_gpu_family():
    """Measured model on a non-matching GPU family yields no batch size recommendations."""
    hw = make_facts(cpu_count=16.0, gpu_count=1.0, gpu_name="NVIDIA T4", vram_gb=16.0)
    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)

    assert "gpu_batch_size" not in rec.transformer
    assert "batch_size" not in rec.transformer
    assert "unmeasured" in rec.model_status


def test_model_gating_matching_l4():
    """Measured model on matching L4 with sufficient VRAM yields recommended batch sizes."""
    hw = make_facts(cpu_count=16.0, gpu_count=1.0, gpu_name="NVIDIA L4", vram_gb=24.0)
    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)

    assert rec.transformer["gpu_batch_size"] == 64
    assert rec.transformer["batch_size"] == 512
    assert rec.model_status == "(measured, L4)"


@pytest.mark.parametrize(
    ("gpu_name", "expected_family"),
    [
        ("NVIDIA L4", "L4"),
        ("Tesla T4", "T4"),
        ("NVIDIA A100-SXM4-80GB", "A100"),
        ("NVIDIA A100 80GB PCIe", "A100"),
        ("NVIDIA A10G", "A10G"),
        ("Tesla V100-PCIE-32GB", "V100"),
        ("NVIDIA H100 80GB HBM3", "H100"),
        ("NVIDIA L40", None),
        ("NVIDIA L40S", None),
        ("L40S", None),
        ("RTX 4090", None),
        (None, None),
    ],
)
def test_extract_gpu_family_exact_tokens(gpu_name, expected_family):
    """Ensure extract_gpu_family matches whole tokens so L40/L40S are not misclassified as L4."""
    if extract_gpu_family(gpu_name) != expected_family:
        raise ValueError(f"extract_gpu_family({gpu_name!r}) != {expected_family!r}")


def test_model_gating_deprecated_model_alias():
    """Deprecated key yields identical recommendation as canonical key with DeprecationWarning."""
    hw = make_facts(cpu_count=16.0, gpu_count=1.0, gpu_name="NVIDIA L4", vram_gb=24.0)

    with pytest.deprecated_call(match="20260211_debertav3_finetuned.*deprecated"):
        rec_deprecated = recommend_settings(hw, model_name=DEPRECATED_MODEL)

    rec_canonical = recommend_settings(hw, model_name=CANONICAL_MODEL)

    assert rec_deprecated.transformer["gpu_batch_size"] == rec_canonical.transformer["gpu_batch_size"]
    assert rec_deprecated.transformer["batch_size"] == rec_canonical.transformer["batch_size"]
    assert rec_deprecated.transformer == rec_canonical.transformer


# ---------------------------------------------------------------------------
# 5. Deprecation is non-destructive
# ---------------------------------------------------------------------------


def test_deprecation_is_non_destructive():
    """Resolving the deprecated model key preserves its own DEFAULT_EXPLANATION."""
    with pytest.deprecated_call(match="20260211_debertav3_finetuned.*deprecated"):
        deprecated_cfg = load_model_config(DEPRECATED_MODEL)

    canonical_cfg = load_model_config(CANONICAL_MODEL)

    # Explanation text MUST differ between the two keys
    assert deprecated_cfg["DEFAULT_EXPLANATION"] != canonical_cfg["DEFAULT_EXPLANATION"]
    assert deprecated_cfg["DEFAULT_EXPLANATION"] == "Identified as {} by finetuned microsoft/deberta-v3-base model"
    assert (
        canonical_cfg["DEFAULT_EXPLANATION"]
        == "Identified as {} by the stanford-med-hdr/tide2-sentry-clinical-ner NER model"
    )

    # DEFAULT_EXPLANATION is the *only* difference: the alias in §6 of the plan is a
    # confirmed same-checkpoint rename, not a name-similarity guess.
    assert set(deprecated_cfg) == set(canonical_cfg)
    differing = {k for k in canonical_cfg if deprecated_cfg[k] != canonical_cfg[k]}
    assert differing == {"DEFAULT_EXPLANATION"}


# ---------------------------------------------------------------------------
# 6. Reference-box parity (the gate)
# ---------------------------------------------------------------------------


def test_reference_box_parity():
    """Resolved settings on reference hardware match hard-coded literals exactly."""
    # 16 vCPU, 1x NVIDIA L4 (24GB), 64GB RAM
    node = NodeShape(
        cpu_count=16.0,
        gpu_count=1.0,
        gpu_name="NVIDIA L4",
        vram_gb=24.0,
        ram_gb=64.0,
    )
    hw = HardwareFacts(
        cluster_cpu=16.0,
        cluster_gpu=1.0,
        nodes=(node,),
        homogeneous=True,
        node=node,
        profile="gpu-workstation",
    )
    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)
    applied = apply_recommendations(
        rec,
        transformer={},
        recognizer={},
        anonymizer={},
        runner={},
    )

    # Exact parity with local_runner.py reference literals
    assert applied.transformer["num_transformer_actors"] == 3
    assert applied.transformer["num_gpus"] == 0.33
    assert applied.transformer["transformer_cpus"] == 4.0
    assert applied.transformer["batch_size"] == 512
    assert applied.transformer["gpu_batch_size"] == 64
    assert applied.transformer["override_num_blocks"] == 16
    assert applied.transformer["num_agg_actors"] == 0

    assert applied.recognizer["num_actors"] == 14
    assert applied.recognizer["num_cpus"] == 0
    assert applied.recognizer["worker_num_cpus"] == 1.0
    assert applied.recognizer["override_num_blocks"] == 32

    assert applied.anonymizer["num_actors"] == 14
    assert applied.anonymizer["num_cpus"] == 0
    assert applied.anonymizer["worker_num_cpus"] == 1.0
    assert applied.anonymizer["override_num_blocks"] == 32

    assert applied.runner["no_progress_timeout_s"] == 600
    assert applied.runner["object_store_gb"] == 19.2


# ---------------------------------------------------------------------------
# 7. Opt-out
# ---------------------------------------------------------------------------


def test_hardware_autotune_opt_out():
    """hardware_autotune=False leaves user kwargs unchanged by recommendations."""
    hw = make_facts(cpu_count=16.0, gpu_count=1.0, gpu_name="NVIDIA L4", vram_gb=24.0)
    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)

    t_kw: dict = {}
    r_kw: dict = {}
    a_kw: dict = {}
    run_kw: dict = {}

    applied = apply_recommendations(
        rec,
        transformer=t_kw,
        recognizer=r_kw,
        anonymizer=a_kw,
        runner=run_kw,
        hardware_autotune=False,
    )

    # All applied settings must have source="default", none with source="auto"
    for entry in applied.entries:
        assert entry.source in ("default", "USER")

    # Table rendering still works cleanly
    table = render_settings_table(applied)
    assert "default" in table
    assert "auto" not in table


# ---------------------------------------------------------------------------
# 8. Small-box pairing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gpu_count", [0.0, 1.0])
def test_small_box_fractional_cpus_and_checkpoint_disabled_pairing(gpu_count):
    """small-box profiles emit fractional CPUs AND enable_checkpoint=False together."""
    hw = make_facts(
        cpu_count=2.0,
        gpu_count=gpu_count,
        gpu_name="NVIDIA T4" if gpu_count > 0 else None,
        vram_gb=16.0 if gpu_count > 0 else None,
        ram_gb=8.0,
    )
    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)

    assert rec.profile in ("small-box-cpu", "small-box-gpu")

    # Assert enable_checkpoint is False across stages
    assert rec.transformer["enable_checkpoint"] is False
    assert rec.recognizer["enable_checkpoint"] is False
    assert rec.anonymizer["enable_checkpoint"] is False
    assert rec.runner["enable_checkpoint"] is False

    # Assert fractional CPUs are emitted across stages
    assert rec.transformer["read_cpus"] < 1.0
    assert rec.transformer["write_cpus"] < 1.0
    assert rec.recognizer["worker_num_cpus"] < 1.0
    assert rec.recognizer["read_cpus"] < 1.0
    assert rec.anonymizer["worker_num_cpus"] < 1.0
    assert rec.anonymizer["read_cpus"] < 1.0

    # Extended hang guard timeout
    assert rec.runner["no_progress_timeout_s"] == 1200


def test_small_box_recommended_keys_are_accepted_by_stage_methods():
    """Every recommended knob must be a real parameter of the stage it targets.

    small-box-* emits the widest key set (fractional CPUs across every operator),
    so an unknown key here would surface as a TypeError only on a 2-CPU host.
    """
    hw = make_facts(cpu_count=2.0, gpu_count=1.0, gpu_name="NVIDIA T4", vram_gb=16.0, ram_gb=8.0)
    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)
    applied = apply_recommendations(rec, transformer={}, recognizer={}, anonymizer={})

    stage_methods = {
        "transformer": LocalJobRunner.run_transformer,
        "recognizer": LocalJobRunner.run_recognition,
        "anonymizer": LocalJobRunner.run_anonymization,
    }
    for stage, method in stage_methods.items():
        accepted = set(inspect.signature(method).parameters)
        emitted = set(getattr(applied, stage))
        assert emitted <= accepted, f"{stage} recommends unknown kwargs: {sorted(emitted - accepted)}"


# ---------------------------------------------------------------------------
# Runner plumbing: the resolved timeout must survive per-stage reconfiguration
# ---------------------------------------------------------------------------


def test_no_progress_timeout_is_sticky_across_stages():
    """Stages re-apply the runner's timeout instead of resetting it to the default."""
    runner = LocalJobRunner(no_progress_timeout_s=1200)
    assert runner._data_context_kwargs(verbose_progress=True) == {
        "verbose_progress": True,
        "no_progress_timeout_s": 1200,
    }


def test_no_progress_timeout_unset_leaves_library_default():
    """With no timeout resolved, stage kwargs are untouched."""
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


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], None),
        (["--no-hardware-autotune"], False),
    ],
)
def test_cli_hardware_autotune_flag(monkeypatch, argv, expected):
    """Unset stays None so YAML can still speak; the flag turns autotune off."""
    args = _parse_run_args(monkeypatch, ["run", "pipeline", "-i", "in.parquet", "-o", "out", *argv])
    assert args.hardware_autotune is expected
    # cmd_run's resolution: only an explicit False disables autotuning.
    assert (args.hardware_autotune is not False) is (expected is not False)


@pytest.mark.parametrize("yaml_value", [True, False])
def test_yaml_hardware_autotune_is_honoured(monkeypatch, tmp_path, yaml_value):
    """A YAML `hardware_autotune: false` must reach cmd_run.

    Regression guard: with argparse defaulting the flag to True, `_apply_config`
    would refuse to backfill and the YAML key would be silently ignored.
    """
    config = tmp_path / "runner.yaml"
    config.write_text(f"hardware_autotune: {str(yaml_value).lower()}\n")

    args = _parse_run_args(
        monkeypatch,
        ["run", "pipeline", "-i", "in.parquet", "-o", "out", "--config", str(config)],
    )
    assert args.hardware_autotune is yaml_value


def test_cli_flag_beats_yaml(monkeypatch, tmp_path):
    """An explicit CLI flag is never overwritten by the config file."""
    config = tmp_path / "runner.yaml"
    config.write_text("hardware_autotune: true\n")

    args = _parse_run_args(
        monkeypatch,
        ["run", "pipeline", "-i", "in.parquet", "-o", "out", "--config", str(config), "--no-hardware-autotune"],
    )
    assert args.hardware_autotune is False


# ---------------------------------------------------------------------------
# Table rendering and helpers
# ---------------------------------------------------------------------------


def test_render_settings_table_output():
    """render_settings_table formats the expected human-readable output."""
    hw = make_facts(cpu_count=16.0, gpu_count=1.0, gpu_name="NVIDIA L4", vram_gb=22.5, ram_gb=62.7)
    rec = recommend_settings(hw, model_name=CANONICAL_MODEL)
    applied = apply_recommendations(
        rec,
        transformer={"gpu_batch_size": 64},
        recognizer={},
        anonymizer={},
        runner={},
    )
    table = render_settings_table(applied)

    assert "Detected: 16 CPU | 1× NVIDIA L4 (22.5 GB) | 62.7 GB RAM | 1 node (homogeneous)" in table
    assert "Profile:  gpu-workstation   Model: stanford-med-hdr/tide2-sentry-clinical-ner (measured, L4)" in table
    assert "num_transformer_actors" in table
    assert "gpu_batch_size" in table
    assert "USER" in table
    assert "auto" in table


def test_detect_hardware_local_fallback():
    """detect_hardware returns valid HardwareFacts even without an active Ray cluster."""
    hw = detect_hardware()
    assert isinstance(hw, HardwareFacts)
    assert hw.cluster_cpu > 0
    assert len(hw.nodes) >= 1
    assert hw.profile in (
        "gpu-workstation",
        "small-box-cpu",
        "small-box-gpu",
        "cpu-only",
        "large-cpu",
        "gpu-server",
        "unknown",
    )


def test_recommend_object_store_gb():
    """recommend_object_store_gb correctly computes ~30% of RAM."""
    hw = make_facts(cpu_count=16.0, gpu_count=0.0, ram_gb=64.0)
    assert recommend_object_store_gb(hw) == 19.2

    hw_no_node = HardwareFacts(
        cluster_cpu=16.0,
        cluster_gpu=0.0,
        nodes=(),
        homogeneous=False,
        node=None,
        profile="unknown",
    )
    assert recommend_object_store_gb(hw_no_node) is None
