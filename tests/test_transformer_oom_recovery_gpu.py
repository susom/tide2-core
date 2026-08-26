"""GPU regression tests for transformer CUDA-OOM recovery (opportunistic).

Wraps the verification helpers in ``tests/oom_verification.py`` as pytest cases,
parametrized over both code paths (``direct`` and ``ray``) plus a compile-aware
``reserved-bounded`` scenario. Marked ``integration`` and skipped when no CUDA
device is present, so it auto-skips in PR CI (no GPU) and runs on a GPU box.

These assert real GPU-memory behavior — no monotonic growth in either
``allocated`` **or** ``reserved``, OOM recovery, peak headroom, full block
coverage, and bounded ``reserved`` across shape churn. Crucially they also assert
the handled-OOM path *actually fired* (the actor's handled-OOM counter is > 0) so
the scenario cannot pass vacuously, and that the whole-device footprint returns to
baseline after recovery. The CPU control-flow guarantees live in
``tests/test_transformer_oom_recovery.py``.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

# Importable helper module (no test_ prefix, not collected). tests/ is on
# sys.path under pytest's prepend import mode, so a plain import works — no
# more importlib-loading a scripts/ CLI (review Finding #4).
import oom_verification  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]

_SHIELD = oom_verification.DEFAULT_PARQUET


@pytest.mark.skipif(not Path(_SHIELD).exists(), reason="SHIELD dataset not available")
@pytest.mark.parametrize("mode", ["direct", "ray"])
def test_oom_fix_holds(mode, tmp_path):
    args = SimpleNamespace(
        mode=mode,
        parquet=_SHIELD,
        column="text",
        model="StanfordAIMI/stanford-deidentifier-v2",
        limit=256 if mode == "direct" else 400,
        batch=128,
        passes=10,
        # Rows fed to the forced-OOM forward. Direct tiles the longest real texts;
        # ray synthesizes this many near-max-length notes. The model runs in fp16,
        # so ~4096 full-length half-precision sequences (~45GB) are needed to
        # overflow a 24GB L4 and force the handled-OOM path.
        oom_count=4096,
        compile_churn=False,
        workdir=str(tmp_path),
    )
    runner = oom_verification.run_direct if mode == "direct" else oom_verification.run_ray
    assert runner(args) is True


@pytest.mark.skipif(not Path(_SHIELD).exists(), reason="SHIELD dataset not available")
def test_reserved_bounded_eager():
    """The default (eager) path keeps reserved bounded across shape churn.

    This is the honest, compile-aware check: it asserts on ``reserved`` (the
    CUDA-graph pool metric that the old allocated-only checks were blind to) and
    would fail if the leak-prone reduce-overhead compile were re-enabled by
    default. Reproduce the blocker manually with
    ``python tests/oom_verification.py --mode reserved-bounded --compile-churn``.
    """
    args = SimpleNamespace(
        parquet=_SHIELD,
        column="text",
        model="StanfordAIMI/stanford-deidentifier-v2",
        limit=64,
        compile_churn=False,
    )
    assert oom_verification.run_reserved_bounded(args) is True
