"""Hardware detection and per-stage settings recommendation for TIDE 2.0.

This module consolidates hardware detection into pure functions over frozen
dataclasses, allowing settings recommendations to be computed and unit-tested
without requiring an active Ray cluster. ``detect_hardware`` is the only place in
the codebase that reads ``ray.cluster_resources()``, ``ray.nodes()``, or ``psutil``
for tuning purposes; everything downstream is a pure function of its output.

The application rule is the whole contract:

    A recommendation is used **only where the caller passed ``None``**. Any value
    the caller supplied — Python kwarg, CLI flag, or YAML key — wins
    unconditionally and is never overridden, clamped, or "corrected."

Cluster vs node: per-node facts drive per-actor sizing (``worker_num_cpus``,
``num_gpus``, ``gpu_batch_size``, object-store bytes), while cluster totals drive
counts (``num_actors``, ``num_transformer_actors``), because a Ray Data actor pool
is cluster-wide. Profiles are matched on the **node** shape, never the cluster
total: fourteen 16-CPU GPU nodes are 224 cluster CPUs but each is a
``gpu-workstation``, not a ``large-cpu`` box.

Profiles:
- ``gpu-workstation``: 4 < CPUs < 64 with GPU present (reference box: 16 vCPU, 1x L4 24 GB).
- ``small-box-cpu``: CPUs <= 4 with no GPU (emits fractional CPUs + checkpointing disabled).
- ``small-box-gpu``: CPUs <= 4 with GPU present (emits fractional CPUs + checkpointing disabled).
- ``cpu-only``: 4 < CPUs < 64 with no GPU (subsumes the legacy 0.25-of-CPUs transformer rule).
- ``large-cpu``: CPUs >= 64 with no GPU (extrapolated from reference box ratios).
- ``gpu-server``: CPUs >= 64 with GPU present (extrapolated from reference box ratios).
- ``unknown``: Heterogeneous cluster or missing node facts (emits no recommendations).

Measured models rule:
Transformer batch sizes (``gpu_batch_size`` and ``batch_size``) are recommended only
for models and GPU families that have empirical sweep measurements. For unmeasured
models, insufficient VRAM, or mismatched GPU families, batch size recommendations
are withheld and existing defaults stand.

Procedure for adding a model to the measured table:
1. Run the batch-size sweep using ``scripts/benchmark_stage_throughput.py`` on the target GPU.
2. Record wall time, throughput, and peak VRAM per batch size.
3. Select the batch size that achieves maximum throughput within headroom (leaving ample VRAM).
4. Add an entry to ``MEASURED_MODELS`` with ``model_name``, ``gpu_family``,
   ``recommended_gpu_batch_size``, ``recommended_batch_size``, ``peak_vram_gb``, and ``min_vram_gb``.

Standard for aliasing a model name:
Only alias on **confirmed checkpoint identity**, never on name similarity. The one
alias today (``20260211_debertav3_finetuned`` -> the canonical name) qualifies because
the two registry entries in ``resources/bert_transformer_configuration.json`` were
diffed and are identical in every key except ``DEFAULT_EXPLANATION``. Aliasing on a
similar-looking name would silently apply one model's VRAM envelope to another and
can OOM a GPU mid-run. The alias applies to the batch-size lookup only: both registry
entries are kept intact, so the deprecated key still returns its own explanation text
and output is byte-identical for callers who do not switch names.
"""

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from tide2.transformers.config import CANONICAL_MODEL_NAME
from tide2.transformers.config import DEPRECATED_MODEL_NAME
from tide2.transformers.config import warn_if_deprecated_model

logger = logging.getLogger(__name__)

# The model registry owns these names and the deprecation warning text; this
# module only keys ``MEASURED_MODELS`` off them.
CANONICAL_MODEL = CANONICAL_MODEL_NAME
DEPRECATED_MODEL = DEPRECATED_MODEL_NAME
VRAM_MARGIN_GB = 1.0

# The reference box: 16 vCPU / 1x L4 24 GB / 64 GB RAM. Every measured number in this
# module was taken on it. ``REFERENCE_NODE_CPU_ACTORS`` was formerly
# ``local_runner.TARGET_NODE_CPU_ACTORS``; the ratio between the two is what the
# extrapolated profiles scale by.
REFERENCE_NODE_CPUS = 16
REFERENCE_NODE_CPU_ACTORS = 14
REFERENCE_CPU_ACTOR_RATIO = REFERENCE_NODE_CPU_ACTORS / REFERENCE_NODE_CPUS

# CPU-class boundaries of the profile matrix, matched on the *node* shape.
SMALL_BOX_MAX_CPUS = 4
LARGE_BOX_MIN_CPUS = 64

# Reference-box per-actor sizing, reproduced exactly by the gpu-workstation profile.
TRANSFORMER_ACTORS_PER_GPU = 3
REFERENCE_GPU_FRACTION = 0.33
REFERENCE_TRANSFORMER_CPUS = 4.0
TRANSFORMER_BLOCKS = 16
CPU_STAGE_BLOCKS = 32

# Legacy CPU-only heuristics, preserved so a GPU-less host keeps its actor counts.
CPU_TRANSFORMER_ACTOR_FRACTION = 0.25
CPU_AGG_ACTOR_FRACTION = 0.3

# Hang-guard timeouts: a small box spends a much larger fraction of the run on
# cold model load, so it needs a longer no-progress window.
DEFAULT_NO_PROGRESS_TIMEOUT_S = 600
SLOW_START_NO_PROGRESS_TIMEOUT_S = 1200


@dataclass(frozen=True)
class NodeShape:
    """Hardware specification of a single cluster node."""

    cpu_count: float
    gpu_count: float
    gpu_name: str | None
    vram_gb: float | None
    ram_gb: float


@dataclass(frozen=True)
class HardwareFacts:
    """Cluster-wide hardware facts and matched profile."""

    cluster_cpu: float
    cluster_gpu: float
    nodes: tuple[NodeShape, ...]
    homogeneous: bool
    node: NodeShape | None
    profile: str


@dataclass(frozen=True)
class MeasuredModelEntry:
    """Empirically measured batch size recommendation for a model and GPU family."""

    model_name: str
    gpu_family: str
    recommended_gpu_batch_size: int
    recommended_batch_size: int
    peak_vram_gb: float
    min_vram_gb: float


MEASURED_MODELS: dict[tuple[str, str], MeasuredModelEntry] = {
    (CANONICAL_MODEL, "L4"): MeasuredModelEntry(
        model_name=CANONICAL_MODEL,
        gpu_family="L4",
        recommended_gpu_batch_size=64,
        recommended_batch_size=512,
        peak_vram_gb=6.77,
        min_vram_gb=7.77,
    ),
}


@dataclass(frozen=True)
class Recommendations:
    """Per-stage setting recommendations produced for a hardware fact set."""

    hw: HardwareFacts
    profile: str
    model_name: str | None
    model_status: str
    transformer: dict[str, Any]
    recognizer: dict[str, Any]
    anonymizer: dict[str, Any]
    runner: dict[str, Any]


@dataclass(frozen=True)
class SettingEntry:
    """A single resolved setting with provenance."""

    stage: str
    knob: str
    value: Any
    source: str


@dataclass
class Applied:
    """Container of resolved settings and configuration dictionaries."""

    hw: HardwareFacts
    model_name: str | None
    model_status: str
    entries: list[SettingEntry]
    transformer: dict[str, Any]
    recognizer: dict[str, Any]
    anonymizer: dict[str, Any]
    runner: dict[str, Any]


def extract_gpu_family(gpu_name: str | None) -> str | None:
    """Extract standard GPU family identifier from device name."""
    if not gpu_name:
        return None
    name_upper = gpu_name.upper()
    for family in ("L4", "T4", "A100", "H100", "V100", "A10G"):
        if re.search(rf"\b{re.escape(family)}\b", name_upper):
            return family
    return None


def classify_profile(node: NodeShape | None, homogeneous: bool = True) -> str:
    """Classify hardware into one of the canonical profiles or 'unknown'.

    Matching is performed on (gpu_present, cpu_class) of the node shape:
    - cpu <= 4: 'small-box-gpu' (if GPU) or 'small-box-cpu' (if no GPU)
    - 4 < cpu < 64: 'gpu-workstation' (if GPU) or 'cpu-only' (if no GPU)
    - cpu >= 64: 'gpu-server' (if GPU) or 'large-cpu' (if no GPU)

    Heterogeneous clusters (or missing node shape) return 'unknown'.
    """
    if not homogeneous or node is None:
        return "unknown"
    gpu_present = node.gpu_count > 0
    cpus = node.cpu_count
    if cpus <= SMALL_BOX_MAX_CPUS:
        return "small-box-gpu" if gpu_present else "small-box-cpu"
    if cpus < LARGE_BOX_MIN_CPUS:
        return "gpu-workstation" if gpu_present else "cpu-only"
    return "gpu-server" if gpu_present else "large-cpu"


@lru_cache(maxsize=1)
def _detect_system_ram_gb() -> float:
    """Detect system RAM in GB using psutil if available."""
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:  # pragma: no cover - psutil is optional
        logger.debug("psutil RAM probe failed; RAM-derived settings withheld", exc_info=True)
        return 0.0


@lru_cache(maxsize=1)
def _detect_local_gpu_info() -> tuple[float, str | None, float | None]:
    """Detect local GPU count, name, and VRAM in GB using PyTorch.

    Cached: the driver's devices are fixed for the process, and the probe
    initializes a CUDA context, so it must happen at most once.
    """
    try:
        import torch

        if torch.cuda.is_available():
            count = float(torch.cuda.device_count())
            if count > 0:
                name = torch.cuda.get_device_name(0)
                vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
                return count, name, vram
    except Exception:  # pragma: no cover - depends on local CUDA availability
        logger.debug("GPU probe failed; assuming no local GPU", exc_info=True)
    return 0.0, None, None


def _ray_cluster_totals() -> tuple[float, float] | None:
    """Return ``(CPU, GPU)`` totals from a reachable Ray cluster, else ``None``."""
    try:
        import ray

        if not ray.is_initialized():
            return None
        res = ray.cluster_resources()
    except Exception:
        logger.debug("Ray cluster resources unavailable; falling back to a local probe", exc_info=True)
        return None
    return float(res.get("CPU", 0.0)), float(res.get("GPU", 0.0))


def _ray_alive_nodes() -> list[dict[str, Any]]:
    """Return the alive entries of ``ray.nodes()``, or an empty list if unavailable."""
    try:
        import ray

        if not ray.is_initialized():
            return []
        return [n for n in ray.nodes() if n.get("Alive", False)]
    except Exception:
        logger.debug("ray.nodes() unavailable; treating the cluster as one node", exc_info=True)
        return []


def alive_node_cpus() -> list[float]:
    """CPU capacity of each alive Ray node.

    Exposed so callers that only need per-node CPU capacity don't have to read
    ``ray.nodes()`` themselves — this module owns that probe.

    Returns:
        One CPU count per alive node; empty when Ray is unavailable.
    """
    return [float(n.get("Resources", {}).get("CPU", 0.0)) for n in _ray_alive_nodes()]


def _shapes_from_ray_nodes(nodes_data: list[dict[str, Any]]) -> list[NodeShape]:
    """Build a NodeShape per alive Ray node.

    GPU name and VRAM come from the driver's local device: Ray reports GPU *counts*,
    not models. On a single-node cluster psutil's view of the driver is the node's
    RAM; on a multi-node cluster only Ray's per-node ``memory`` resource applies,
    since psutil describes the driver's machine alone.
    """
    _, gpu_name, vram = _detect_local_gpu_info()
    driver_ram = _detect_system_ram_gb()
    single_node = len(nodes_data) == 1

    shapes: list[NodeShape] = []
    for node in nodes_data:
        resources = node.get("Resources", {})
        gpu_count = float(resources.get("GPU", 0.0))
        ram_gb = (
            driver_ram if single_node and driver_ram > 0 else round(float(resources.get("memory", 0.0)) / (1024**3), 1)
        )
        shapes.append(
            NodeShape(
                cpu_count=float(resources.get("CPU", 0.0)),
                gpu_count=gpu_count,
                gpu_name=gpu_name if (single_node and gpu_count > 0) else None,
                vram_gb=vram if (single_node and gpu_count > 0) else None,
                ram_gb=ram_gb,
            )
        )
    return shapes


def _facts_from_shapes(shapes: list[NodeShape], cluster_cpu: float, cluster_gpu: float) -> HardwareFacts:
    """Assemble HardwareFacts, matching the profile on the common node shape."""
    first = shapes[0]
    homogeneous = all(
        s.cpu_count == first.cpu_count and s.gpu_count == first.gpu_count and abs(s.ram_gb - first.ram_gb) < 1.0
        for s in shapes
    )
    common_node = first if homogeneous else None
    return HardwareFacts(
        cluster_cpu=cluster_cpu if cluster_cpu > 0 else sum(s.cpu_count for s in shapes),
        cluster_gpu=cluster_gpu if cluster_gpu > 0 else sum(s.gpu_count for s in shapes),
        nodes=tuple(shapes),
        homogeneous=homogeneous,
        node=common_node,
        profile=classify_profile(common_node, homogeneous),
    )


def detect_hardware() -> HardwareFacts:
    """Detect cluster (or, with no Ray cluster, local) hardware configuration.

    This is the only function in the codebase that reads ``ray.cluster_resources()``,
    ``ray.nodes()``, or ``psutil`` for tuning purposes; everything downstream is a
    pure function of the ``HardwareFacts`` it returns.
    """
    totals = _ray_cluster_totals()
    cluster_cpu, cluster_gpu = totals if totals else (0.0, 0.0)

    if totals:
        node_data = _ray_alive_nodes()
        if node_data:
            return _facts_from_shapes(_shapes_from_ray_nodes(node_data), cluster_cpu, cluster_gpu)

        if cluster_cpu > 0:
            # Cluster reachable but per-node detail is not: treat the totals as one node.
            _, gpu_name, vram = _detect_local_gpu_info()
            shape = NodeShape(
                cpu_count=cluster_cpu,
                gpu_count=cluster_gpu,
                gpu_name=gpu_name if cluster_gpu > 0 else None,
                vram_gb=vram if cluster_gpu > 0 else None,
                ram_gb=_detect_system_ram_gb(),
            )
            return _facts_from_shapes([shape], cluster_cpu, cluster_gpu)

    gpu_count, gpu_name, vram = _detect_local_gpu_info()
    shape = NodeShape(
        cpu_count=float(os.cpu_count() or SMALL_BOX_MAX_CPUS),
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        vram_gb=vram,
        ram_gb=_detect_system_ram_gb(),
    )
    return _facts_from_shapes([shape], shape.cpu_count, gpu_count)


def recommend_object_store_gb(hw: HardwareFacts) -> float | None:
    """Recommend Ray object store memory in GB (~30% of system RAM)."""
    if hw.node is not None and hw.node.ram_gb > 0:
        return round(hw.node.ram_gb * 0.3, 1)
    return None


def _resolve_measured_batch_sizes(
    hw: HardwareFacts,
    model_name: str | None,
) -> tuple[str, int | None, int | None]:
    """Resolve the measured batch sizes for ``model_name`` on ``hw``.

    Returns ``(model_status, batch_size, gpu_batch_size)``. Both sizes are ``None``
    unless a sweep exists for this exact (model, GPU family) pair and the node has
    the VRAM it was measured at — §6's measured-models-only rule. Withholding beats
    scaling a guess that could OOM a GPU mid-run.
    """
    if model_name is None:
        return "none", None, None

    lookup_model = model_name
    if model_name == DEPRECATED_MODEL:
        warn_if_deprecated_model(model_name, stacklevel=4)
        # Alias only for this lookup: the two registry entries stay distinct so the
        # deprecated key keeps returning its own DEFAULT_EXPLANATION.
        lookup_model = CANONICAL_MODEL

    if hw.cluster_gpu <= 0 or hw.node is None:
        return "(unmeasured)", None, None

    gpu_family = extract_gpu_family(hw.node.gpu_name)
    entry = MEASURED_MODELS.get((lookup_model, gpu_family or ""))
    if entry is None:
        if any(m == lookup_model for (m, _family) in MEASURED_MODELS):
            return f"(unmeasured for {gpu_family or 'unknown GPU'})", None, None
        return "(unmeasured)", None, None

    node_vram = hw.node.vram_gb
    if node_vram is not None and node_vram < entry.min_vram_gb:
        return "(insufficient VRAM)", None, None

    return f"(measured, {gpu_family})", entry.recommended_batch_size, entry.recommended_gpu_batch_size


def _cpu_stage_recs(num_actors: int) -> dict[str, Any]:
    """Recognizer/anonymizer settings at the reference box's one-CPU-per-slot ratio."""
    return {
        "num_actors": num_actors,
        # num_cpus=0 is kept for additive slot resolution (0 + 1.0 = 1.0).
        "num_cpus": 0,
        "worker_num_cpus": 1.0,
        "override_num_blocks": CPU_STAGE_BLOCKS,
    }


def _small_box_cpu_stage_recs() -> dict[str, Any]:
    """Recognizer/anonymizer settings for a <=4-CPU box.

    Fractional CPUs *and* ``enable_checkpoint=False`` are emitted together because
    each alone is known not to clear the 0/1 deadlock (README: "Why small boxes
    deadlock").
    """
    return {
        "num_actors": 1,
        "worker_num_cpus": 0.5,
        "read_cpus": 0.25,
        "write_cpus": 0.25,
        "enable_checkpoint": False,
    }


def _gpu_transformer_recs(
    hw: HardwareFacts,
    batch_size: int | None,
    gpu_batch_size: int | None,
) -> dict[str, Any]:
    """Transformer settings for a GPU node, at the reference box's per-GPU sizing."""
    recs: dict[str, Any] = {
        "num_transformer_actors": max(1, round(hw.cluster_gpu * TRANSFORMER_ACTORS_PER_GPU)),
        "num_gpus": REFERENCE_GPU_FRACTION,
        "transformer_cpus": REFERENCE_TRANSFORMER_CPUS,
        "num_agg_actors": 0,
        "override_num_blocks": TRANSFORMER_BLOCKS,
    }
    if batch_size is not None:
        recs["batch_size"] = batch_size
    if gpu_batch_size is not None:
        recs["gpu_batch_size"] = gpu_batch_size
    return recs


def _cpu_transformer_recs(hw: HardwareFacts) -> dict[str, Any]:
    """Transformer settings for a GPU-less node (the legacy site-3 heuristics)."""
    return {
        "num_transformer_actors": max(1, int(hw.cluster_cpu * CPU_TRANSFORMER_ACTOR_FRACTION)),
        "num_gpus": 0.0,
        "num_agg_actors": max(1, int(hw.cluster_cpu * CPU_AGG_ACTOR_FRACTION)),
        "override_num_blocks": TRANSFORMER_BLOCKS,
    }


def _small_box_transformer_recs(
    hw: HardwareFacts,
    batch_size: int | None,
    gpu_batch_size: int | None,
) -> dict[str, Any]:
    """Transformer settings for a <=4-CPU box: every operator budgeted fractionally."""
    gpu_present = hw.cluster_gpu > 0
    node_gpus = float(hw.node.gpu_count) if hw.node is not None else 1.0
    recs: dict[str, Any] = {
        "num_transformer_actors": 1,
        "num_gpus": node_gpus if gpu_present else 0.0,
        "transformer_cpus": 0.25,
        "read_cpus": 0.25,
        "write_cpus": 0.25,
        "agg_num_cpus": 0.5,
        "num_agg_actors": 1,
        "enable_checkpoint": False,
    }
    if gpu_present:
        if batch_size is not None:
            recs["batch_size"] = batch_size
        if gpu_batch_size is not None:
            recs["gpu_batch_size"] = gpu_batch_size
    return recs


def _runner_recs(hw: HardwareFacts, no_progress_timeout_s: int, *, disable_checkpoint: bool = False) -> dict[str, Any]:
    """Cluster-level settings: hang guard, object store, and the small-box opt-out."""
    recs: dict[str, Any] = {"no_progress_timeout_s": no_progress_timeout_s}
    if disable_checkpoint:
        recs["enable_checkpoint"] = False
    obj_gb = recommend_object_store_gb(hw)
    if obj_gb is not None:
        recs["object_store_gb"] = obj_gb
    return recs


def recommend_settings(
    hw: HardwareFacts,
    model_name: str | None = None,
) -> Recommendations:
    """Recommend per-stage settings from detected hardware facts.

    Per-node facts drive per-actor sizing (``worker_num_cpus``, ``num_gpus``,
    ``gpu_batch_size``); cluster totals drive counts (``num_actors``,
    ``num_transformer_actors``), since a Ray Data actor pool is cluster-wide.
    An ``unknown`` profile (heterogeneous cluster, or no node facts) recommends
    nothing at all rather than averaging two machine shapes.
    """
    model_status, batch_size_rec, gpu_batch_size_rec = _resolve_measured_batch_sizes(hw, model_name)

    empty = Recommendations(
        hw=hw,
        profile=hw.profile,
        model_name=model_name,
        model_status=model_status,
        transformer={},
        recognizer={},
        anonymizer={},
        runner={},
    )
    if (
        hw.profile == "unknown"
        or len(hw.nodes) > 1
        or hw.cluster_gpu > 1.0
        or (hw.node is not None and hw.node.gpu_count > 1.0)
    ):
        if len(hw.nodes) > 1 or hw.cluster_gpu > 1.0 or (hw.node is not None and hw.node.gpu_count > 1.0):
            logger.info(
                "Hardware autotuning is restricted to single-node, single-GPU environments. "
                "Multi-node or multi-GPU setup detected (nodes=%d, cluster_gpu=%.1f); withholding recommendations.",
                len(hw.nodes),
                hw.cluster_gpu,
            )
        return empty

    if hw.profile in ("small-box-cpu", "small-box-gpu"):
        t_rec = _small_box_transformer_recs(hw, batch_size_rec, gpu_batch_size_rec)
        cpu_stage = _small_box_cpu_stage_recs()
        run_rec = _runner_recs(hw, SLOW_START_NO_PROGRESS_TIMEOUT_S, disable_checkpoint=True)
    else:
        if hw.profile in ("gpu-workstation", "gpu-server"):
            t_rec = _gpu_transformer_recs(hw, batch_size_rec, gpu_batch_size_rec)
            timeout = DEFAULT_NO_PROGRESS_TIMEOUT_S
        else:
            t_rec = _cpu_transformer_recs(hw)
            timeout = DEFAULT_NO_PROGRESS_TIMEOUT_S if hw.profile == "large-cpu" else SLOW_START_NO_PROGRESS_TIMEOUT_S

        if hw.profile in ("large-cpu", "gpu-server"):
            # Extrapolated from the reference box, not measured: actors ~= CPUs - 2
            # per node at 1.0 CPU each.
            node_cpus = hw.node.cpu_count if hw.node else hw.cluster_cpu
            num_nodes = len(hw.nodes) or 1
            num_actors = max(1, int(node_cpus - 2)) * num_nodes
        else:
            num_actors = max(1, round(hw.cluster_cpu * REFERENCE_CPU_ACTOR_RATIO))

        cpu_stage = _cpu_stage_recs(num_actors)
        run_rec = _runner_recs(hw, timeout)

    return Recommendations(
        hw=hw,
        profile=hw.profile,
        model_name=model_name,
        model_status=model_status,
        transformer=t_rec,
        recognizer=dict(cpu_stage),
        anonymizer=dict(cpu_stage),
        runner=run_rec,
    )


def apply_recommendations(
    rec: Recommendations,
    *,
    transformer: dict[str, Any],
    recognizer: dict[str, Any],
    anonymizer: dict[str, Any],
    runner: dict[str, Any] | None = None,
    hardware_autotune: bool = True,
) -> Applied:
    """Apply recommendations to user kwargs dictionaries in-place.

    Precedence rule: user-supplied values (non-None) are never overridden.
    Recommendations only fill values where the key was unset or None.
    If hardware_autotune is False or profile is 'unknown', recommendations are
    not applied and legacy hard-coded defaults stand.
    """
    if runner is None:
        runner = {}

    cpus = rec.hw.cluster_cpu or REFERENCE_NODE_CPUS
    legacy_rec_actors = REFERENCE_NODE_CPU_ACTORS if cpus >= REFERENCE_NODE_CPUS else max(1, int(cpus - 2))

    if rec.hw.cluster_gpu > 0:
        legacy_transformer = {
            "num_transformer_actors": 3,
            "num_gpus": 0.33,
            "transformer_cpus": 4.0,
            "batch_size": 512,
            "gpu_batch_size": 64,
            "override_num_blocks": 16,
            "num_agg_actors": 0,
        }
    else:
        legacy_transformer = {
            "override_num_blocks": 16,
        }

    # Both CPU stages shipped with the same pre-recommender defaults.
    legacy_cpu_stage = {
        "num_actors": legacy_rec_actors,
        "num_cpus": 0,
        "worker_num_cpus": 1.0,
        "override_num_blocks": 32,
    }
    legacy_recognizer = dict(legacy_cpu_stage)
    legacy_anonymizer = dict(legacy_cpu_stage)

    legacy_runner = {
        "no_progress_timeout_s": 600,
        "object_store_gb": recommend_object_store_gb(rec.hw),
    }

    core_knobs = {
        "transformer": ["num_transformer_actors", "num_gpus", "transformer_cpus", "gpu_batch_size"],
        "recognizer": ["num_actors", "worker_num_cpus"],
        "anonymizer": ["num_actors", "worker_num_cpus"],
        "runner": ["no_progress_timeout_s", "object_store_gb"],
    }

    stages_config = [
        ("transformer", transformer, rec.transformer, legacy_transformer),
        ("recognizer", recognizer, rec.recognizer, legacy_recognizer),
        ("anonymizer", anonymizer, rec.anonymizer, legacy_anonymizer),
        ("runner", runner, rec.runner, legacy_runner),
    ]

    entries: list[SettingEntry] = []

    for stage_name, user_dict, rec_dict, leg_dict in stages_config:
        stage_core = core_knobs[stage_name]
        candidate_keys = list(stage_core)
        for k in list(user_dict.keys()) + list(rec_dict.keys()) + list(leg_dict.keys()):
            if k not in candidate_keys:
                candidate_keys.append(k)

        for key in candidate_keys:
            if key in user_dict and user_dict[key] is not None:
                source = "USER"
                value = user_dict[key]
            elif hardware_autotune and rec.profile != "unknown" and key in rec_dict and rec_dict[key] is not None:
                source = "auto"
                value = rec_dict[key]
                user_dict[key] = value
            elif key in leg_dict and leg_dict[key] is not None:
                source = "default"
                value = leg_dict[key]
                user_dict[key] = value
            else:
                continue

            if key == "num_cpus" and source != "USER":
                continue

            if key in stage_core or source in ("USER", "auto"):
                entries.append(
                    SettingEntry(
                        stage=stage_name,
                        knob=key,
                        value=value,
                        source=source,
                    )
                )

    return Applied(
        hw=rec.hw,
        model_name=rec.model_name,
        model_status=rec.model_status,
        entries=entries,
        transformer=transformer,
        recognizer=recognizer,
        anonymizer=anonymizer,
        runner=runner,
    )


def render_settings_table(applied: Applied) -> str:
    """Render the human-readable summary and resolved settings table."""
    hw = applied.hw
    cpu_val = int(hw.cluster_cpu) if hw.cluster_cpu.is_integer() else hw.cluster_cpu

    if hw.cluster_gpu > 0:
        gpu_count_val = int(hw.cluster_gpu) if hw.cluster_gpu.is_integer() else hw.cluster_gpu
        gpu_name_str = hw.node.gpu_name if (hw.node and hw.node.gpu_name) else "GPU"
        vram_str = f" ({hw.node.vram_gb:.1f} GB)" if (hw.node and hw.node.vram_gb is not None) else ""
        gpu_desc = f"{gpu_count_val}× {gpu_name_str}{vram_str}"
    else:
        gpu_desc = "0 GPU"

    ram_desc = f"{hw.node.ram_gb:.1f} GB RAM" if (hw.node and hw.node.ram_gb > 0) else "unknown RAM"
    node_count = len(hw.nodes)
    node_str = f"{node_count} node{'s' if node_count != 1 else ''}"
    homo_str = "homogeneous" if hw.homogeneous else "heterogeneous"
    detected_line = f"Detected: {cpu_val} CPU | {gpu_desc} | {ram_desc} | {node_str} ({homo_str})"

    profile_disp = hw.profile
    if hw.profile in ("large-cpu", "gpu-server"):
        profile_disp = f"{hw.profile} (extrapolated, unmeasured)"

    model_disp = f"{applied.model_name} {applied.model_status}".strip() if applied.model_name else "none"

    profile_line = f"Profile:  {profile_disp:<17} Model: {model_disp}"

    lines = [
        detected_line,
        profile_line,
        "",
        f" {'stage':<12} {'knob':<24} {'value':>6}   {'source'}",
    ]

    for entry in applied.entries:
        val_str = str(entry.value)
        lines.append(f" {entry.stage:<12} {entry.knob:<24} {val_str:>6}   {entry.source}")

    return "\n".join(lines)
