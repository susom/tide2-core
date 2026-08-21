"""GPU regression tests for transformer CUDA-OOM recovery (opportunistic).

Wraps the D2 verification (``scripts/verify_oom_fix.py``) as pytest cases,
parametrized over both code paths (``direct`` and ``ray``). Marked
``integration`` and skipped when no CUDA device is present, so it auto-skips in
PR CI (no GPU) and runs in a GPU nightly if one exists.

These assert real GPU-memory behavior (no monotonic growth, OOM recovery, peak
headroom, full block coverage) — the CPU control-flow guarantees live in
``tests/test_transformer_oom_recovery.py``.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_oom_fix.py"
_SHIELD = "/data/neurips2026_data/shield_dataset/shield_pii_dataset.parquet"


def _load_verify_module():
    spec = importlib.util.spec_from_file_location("verify_oom_fix", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_oom_fix"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not Path(_SHIELD).exists(), reason="SHIELD dataset not available")
@pytest.mark.parametrize("mode", ["direct", "ray"])
def test_oom_fix_holds(mode, tmp_path):
    verify = _load_verify_module()
    args = SimpleNamespace(
        mode=mode,
        parquet=_SHIELD,
        column="text",
        model="StanfordAIMI/stanford-deidentifier-v2",
        limit=256 if mode == "direct" else 400,
        batch=128,
        passes=10,
        oom_count=4096,
        workdir=str(tmp_path),
    )
    runner = verify.run_direct if mode == "direct" else verify.run_ray
    assert runner(args) is True
