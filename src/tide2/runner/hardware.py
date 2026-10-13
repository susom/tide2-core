"""What this machine has, and the settings that suit it.

``detect_machine`` reads the machine: CPUs, RAM, GPUs and GPU memory. Container limits count,
and Ray is not asked. ``recommend`` turns those facts into per-stage settings with the rules in
``resources/hardware_rules.yaml``. That file is the only place to add or change a rule.

A recommendation is used only where the caller passed ``None``. A value the caller supplied (Python
kwarg, CLI flag or YAML key) always wins and is never overridden, clamped or corrected.
"""

import ast
import logging
import operator
import os
import re
import subprocess  # nosec B404 # only used to run nvidia-smi
from dataclasses import dataclass
from dataclasses import field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from tide2.transformers.config import CANONICAL_MODEL_NAME
from tide2.transformers.config import DEPRECATED_MODEL_NAME
from tide2.transformers.config import warn_if_deprecated_model
from tide2.utils.resource_utils import get_resource_path

logger = logging.getLogger(__name__)

RULES_FILE = "hardware_rules.yaml"
RULES_ENV_VAR = "TIDE2_HARDWARE_RULES"

STAGES = ("transformer", "recognizer", "anonymizer", "runner")
FACTS = ("cpus", "ram_gb", "gpus", "gpu_ram_gb")
CPU_STAGES = ("recognizer", "anonymizer")

_RULE_KEYS = {"name", "when", "let", "cpu_stages", *STAGES}
_MODEL_RULE_KEYS = {"model", "gpu_family", "min_gpu_ram_gb", "transformer"}


@dataclass(frozen=True)
class Machine:
    """What this machine has. ``gpu_ram_gb`` is the memory of one GPU."""

    cpus: int
    ram_gb: float
    gpus: int = 0
    gpu_name: str | None = None
    gpu_ram_gb: float = 0.0


@dataclass(frozen=True)
class SettingEntry:
    """One resolved setting. ``source`` is USER, ``default``, or the name of the rule that set it."""

    stage: str
    knob: str
    value: Any
    source: str


@dataclass(frozen=True)
class Settings:
    """Recommended settings per stage, and where each one came from."""

    machine: Machine
    rules: tuple[str, ...]
    model_status: str
    transformer: dict[str, Any]
    recognizer: dict[str, Any]
    anonymizer: dict[str, Any]
    runner: dict[str, Any]
    model_name: str | None = None
    sources: dict[tuple[str, str], str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _cgroup_value(name: str) -> str | None:
    try:
        return Path("/sys/fs/cgroup", name).read_text().split()[0]
    except (OSError, IndexError):
        return None


def _usable_cpus() -> int:
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, round(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, count)


def _usable_ram_gb() -> float:
    try:
        import psutil

        total = psutil.virtual_memory().total
    except Exception:
        logger.debug("RAM probe failed", exc_info=True)
        return 0.0
    limit = _cgroup_value("memory.max")
    if limit is not None and limit.isdigit():
        total = min(total, int(limit))
    return round(total / 1024**3, 1)


def _detect_gpus() -> tuple[int, str | None, float]:
    """Return ``(count, name, memory of one GPU in GB)`` from ``nvidia-smi``; no GPU when it is absent."""
    try:
        out = subprocess.run(  # nosec B603 B607 # fixed argv, no shell, no user input
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0, None, 0.0
    rows = [line.rsplit(",", 1) for line in out.strip().splitlines() if "," in line]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        rows = rows[: len([t for t in visible.split(",") if t.strip() not in ("", "-1")])]
    if not rows:
        return 0, None, 0.0
    return len(rows), rows[0][0].strip(), round(float(rows[0][1]) / 1024, 1)


@lru_cache(maxsize=1)
def detect_machine() -> Machine:
    """Read this machine's CPUs, RAM, GPUs and GPU memory. Cached: the machine does not change."""
    gpus, gpu_name, gpu_ram_gb = _detect_gpus()
    return Machine(cpus=_usable_cpus(), ram_gb=_usable_ram_gb(), gpus=gpus, gpu_name=gpu_name, gpu_ram_gb=gpu_ram_gb)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARE = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_FUNCTIONS = {"max": max, "min": min, "round": round, "int": int, "abs": abs}


def _evaluate(expression: str, names: dict[str, Any]) -> Any:
    """Evaluate a rule formula. Only numbers, the given names, arithmetic and a few functions are allowed."""

    def visit(node: ast.AST) -> Any:  # noqa: PLR0911
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -visit(node.operand) if isinstance(node.op, ast.USub) else visit(node.operand)
        if isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
            left = visit(node.left)
            for op, right_node in zip(node.ops, node.comparators, strict=True):
                right = visit(right_node)
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return visit(node.body) if visit(node.test) else visit(node.orelse)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](*[visit(arg) for arg in node.args])
        raise ValueError(f"not allowed in a hardware rule formula: {ast.dump(node)[:60]}")

    return visit(ast.parse(expression, mode="eval").body)


def _check_rules(rules: dict[str, Any], source: str) -> None:
    """Fail on a misspelled key, because a rule that silently does nothing is worse than an error."""

    def check(section: dict[str, Any], allowed: set[str], where: str) -> None:
        unknown = set(section) - allowed
        if unknown:
            raise ValueError(f"{source}: unknown key(s) {sorted(unknown)} in {where}; allowed: {sorted(allowed)}")

    check(rules, {"gpu_families", "defaults", "rules", "model_rules"}, "the top level")
    defaults = rules.get("defaults") or {}
    check(defaults, {"cpu_stages", *STAGES}, "defaults")
    needed = {
        "transformer": {"num_gpus", "num_transformer_actors", "num_agg_actors"},
        "cpu_stages": {"num_actors"},
    }
    for section, knobs in needed.items():
        missing = knobs - set(defaults.get(section) or {})
        if missing:
            raise ValueError(f"{source}: defaults.{section} must set {sorted(missing)}; the runner relies on them")
    allowed_when = {f"{bound}_{fact}" for bound in ("min", "max") for fact in FACTS}
    for rule in rules.get("rules") or []:
        name = rule.get("name", "<unnamed>")
        check(rule, _RULE_KEYS, f"rule {name!r}")
        check(rule.get("when") or {}, allowed_when, f"the 'when' of rule {name!r}")
    for rule in rules.get("model_rules") or []:
        check(rule, _MODEL_RULE_KEYS, f"model rule {rule.get('model', '<unnamed>')!r}")


@lru_cache(maxsize=4)
def _load_rules(source: str) -> dict[str, Any]:
    with Path(source).open(encoding="utf-8") as handle:
        rules = yaml.safe_load(handle) or {}
    _check_rules(rules, source)
    return rules


def load_rules(path: str | None = None) -> dict[str, Any]:
    """Load the rules from ``path``, else from ``$TIDE2_HARDWARE_RULES``, else from the packaged file."""
    return _load_rules(path or os.environ.get(RULES_ENV_VAR) or get_resource_path(RULES_FILE))


def _matches(when: dict[str, Any] | None, machine: Machine) -> bool:
    for key, bound in (when or {}).items():
        kind, fact = key.split("_", 1)
        value = getattr(machine, fact)
        if (kind == "min" and value < bound) or (kind == "max" and value > bound):
            return False
    return True


def extract_gpu_family(gpu_name: str | None, families: list[str] | tuple[str, ...]) -> str | None:
    """Return the GPU family named in ``gpu_name`` (a whole word, so L40 is not L4), else None."""
    if not gpu_name:
        return None
    for family in families:
        if re.search(rf"\b{re.escape(family)}\b", gpu_name.upper()):
            return family
    return None


def _model_rule(machine: Machine, model_name: str | None, rules: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Return ``(status, rule)``: the measured rule for this model on this GPU, or why there is none."""
    if model_name is None:
        return "none", None
    if model_name == DEPRECATED_MODEL_NAME:
        warn_if_deprecated_model(model_name, stacklevel=4)
        model_name = CANONICAL_MODEL_NAME  # same checkpoint; only the batch-size lookup is aliased
    candidates = [r for r in rules.get("model_rules") or [] if r["model"] == model_name]
    if machine.gpus == 0 or not candidates:
        return "(unmeasured)", None
    family = extract_gpu_family(machine.gpu_name, rules.get("gpu_families") or [])
    rule = next((r for r in candidates if r.get("gpu_family") == family), None)
    if rule is None:
        return f"(unmeasured for {family or 'unknown GPU'})", None
    if machine.gpu_ram_gb < rule.get("min_gpu_ram_gb", 0):
        return "(insufficient VRAM)", None
    return f"(measured, {family})", rule


def recommend(
    machine: Machine,
    model_name: str | None = None,
    *,
    use_rules: bool = True,
    rules_path: str | None = None,
) -> Settings:
    """Return the settings for ``machine``: the defaults, then every matching rule, then the model rule.

    Args:
        machine: The facts to match, from ``detect_machine`` or built by hand.
        model_name: Transformer model, for the measured batch sizes.
        use_rules: False returns the safe defaults only.
        rules_path: A rules file to use instead of the packaged one.
    """
    rules = load_rules(rules_path)
    names: dict[str, Any] = {fact: getattr(machine, fact) for fact in FACTS}
    values: dict[str, dict[str, Any]] = {stage: {} for stage in STAGES}
    sources: dict[tuple[str, str], str] = {}

    def apply_layer(layer: dict[str, Any], source: str) -> None:
        local = dict(names)
        for var, expression in (layer.get("let") or {}).items():
            local[var] = _evaluate(expression, local)
        sections = {stage: dict(layer.get(stage) or {}) for stage in STAGES}
        for stage in CPU_STAGES:
            sections[stage] = {**(layer.get("cpu_stages") or {}), **sections[stage]}
        for stage, knobs in sections.items():
            for knob, value in knobs.items():
                values[stage][knob] = _evaluate(value, local) if isinstance(value, str) else value
                sources[(stage, knob)] = source

    apply_layer(rules.get("defaults") or {}, "default")
    matched: list[str] = []
    model_status = "none" if model_name is None else "(rules off)"
    if use_rules:
        for rule in rules.get("rules") or []:
            if _matches(rule.get("when"), machine):
                apply_layer(rule, rule["name"])
                matched.append(rule["name"])
        model_status, model_rule = _model_rule(machine, model_name, rules)
        if model_rule is not None:
            apply_layer({"transformer": model_rule["transformer"]}, f"measured {model_rule['gpu_family']}")

    return Settings(
        machine=machine,
        rules=tuple(matched),
        model_status=model_status,
        model_name=model_name,
        sources=sources,
        **values,
    )


# ---------------------------------------------------------------------------
# Applying and showing
# ---------------------------------------------------------------------------


def apply_settings(
    settings: Settings,
    *,
    transformer: dict[str, Any],
    recognizer: dict[str, Any],
    anonymizer: dict[str, Any],
    runner: dict[str, Any],
) -> list[SettingEntry]:
    """Fill each stage's kwargs in place from ``settings``, only where the caller left a knob unset (None).

    Returns:
        One entry per setting, with the source USER, ``default``, or the rule name.
    """
    entries: list[SettingEntry] = []
    for stage, user in (
        ("transformer", transformer),
        ("recognizer", recognizer),
        ("anonymizer", anonymizer),
        ("runner", runner),
    ):
        recommended = getattr(settings, stage)
        for knob in [*recommended, *(k for k in user if k not in recommended)]:
            if user.get(knob) is not None:
                entries.append(SettingEntry(stage, knob, user[knob], "USER"))
            elif knob in recommended:
                user[knob] = recommended[knob]
                entries.append(SettingEntry(stage, knob, recommended[knob], settings.sources[(stage, knob)]))
    return entries


def render_settings(settings: Settings, entries: list[SettingEntry]) -> str:
    """Return the machine, the matched rules, and the table of resolved settings."""
    machine = settings.machine
    gpu = f"{machine.gpus}× {machine.gpu_name or 'GPU'} ({machine.gpu_ram_gb:g} GB)" if machine.gpus else "0 GPU"
    model = f"{settings.model_name} {settings.model_status}".strip() if settings.model_name else "none"
    lines = [
        f"Detected: {machine.cpus} CPU | {machine.ram_gb:g} GB RAM | {gpu}",
        f"Rules:    {', '.join(settings.rules) or 'none (defaults only)'}   Model: {model}",
        "",
        f" {'stage':<12} {'knob':<24} {'value':>8}   source",
    ]
    for entry in entries:
        lines.append(f" {entry.stage:<12} {entry.knob:<24} {entry.value!s:>8}   {entry.source}")
    return "\n".join(lines)
