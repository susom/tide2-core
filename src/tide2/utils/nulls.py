"""Shared null/NaN detection for scalar batch values.

Ray Data batches arrive as pandas/numpy containers, so a "missing" value can be
``None``, a float ``nan``, a numpy floating ``nan``, or ``pd.NA``. Every stage
needs the same answer for all of them, so the predicate lives here rather than
being reimplemented per actor.
"""

import contextlib
import math
from typing import Any

import numpy as np
import pandas as pd


def is_null(value: Any) -> bool:
    """Check if a scalar value is null/NaN.

    Handles ``None``, Python and numpy ``nan``, and pandas ``NA``. Non-scalar
    values (lists, arrays, dicts) are never considered null, so a column of
    list-valued cells can be passed through safely.

    Args:
        value: The scalar value to test.

    Returns:
        True if the value represents a missing scalar, False otherwise.
    """
    if value is None:
        return True
    with contextlib.suppress(Exception):
        res = pd.isna(value)
        if isinstance(res, (bool, np.bool_)):
            return bool(res)
    with contextlib.suppress(TypeError, ValueError):
        if isinstance(value, float) and math.isnan(value):
            return True
        if isinstance(value, (np.floating, np.integer)) and np.isnan(value):
            return True
    return False
