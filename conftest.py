import sys

import pyarrow  # noqa: F401, ICN001
import pytest


@pytest.fixture(autouse=True)
def _shutdown_ray_after_test():
    """Stop any Ray runtime a test left running so later tests do not attach to it.

    Workers of a reused cluster never see environment variables set after it started.
    """
    yield
    ray = sys.modules.get("ray")
    if ray is not None and ray.is_initialized():
        ray.shutdown()
