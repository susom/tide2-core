import sys

import pandas as pd
import pyarrow  # noqa: F401, ICN001
import pytest


@pytest.fixture(autouse=True)
def _shutdown_ray_after_test(request):
    """Stop any Ray runtime a test left running so later tests do not attach to it.

    Workers of a reused cluster never see environment variables set after it started.
    A test module that sets ``KEEP_RAY_BETWEEN_TESTS = True`` owns its cluster in a module-scoped fixture.
    """
    yield
    if getattr(request.module, "KEEP_RAY_BETWEEN_TESTS", False):
        return
    ray = sys.modules.get("ray")
    if ray is not None and ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def notes():
    """6 notes across 2 patients, so per-patient jitter is exercised."""
    rows = []
    for i in range(6):
        rows.append(
            {
                # "John Doe" must match STUB_ENTITY in tests/test_pipeline_parity.py.
                "note_text": (
                    f"Patient John Doe (MRN 1234{i:02d}) was seen on 03/1{i % 9}/2021 "
                    f"at 555-010{i % 9}-2020. Follow-up scheduled."
                ),
                "patient_id": f"patient-{i % 2}",
            }
        )
    return pd.DataFrame(rows)
