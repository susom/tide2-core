"""Shared null/NaN detection for scalar batch values.

Ray Data batches arrive as pandas/numpy containers, so a "missing" value can be
``None``, a float ``nan``, a numpy floating ``nan``, or ``pd.NA``. Every stage
needs the same answer for all of them, so the predicate lives here rather than
being reimplemented per actor.
"""

import contextlib
import math
import sys
from typing import Any

import numpy as np


def is_null(value: Any) -> bool:
    """Check if a scalar value is null/NaN.

    Handles ``None``, Python and numpy ``nan``, pandas ``NA``, and literal
    ``"<NA>"`` strings. Non-scalar values (lists, arrays, dicts) are never
    considered null, so a column of list-valued cells can be passed through safely.

    Args:
        value: The scalar value to test.

    Returns:
        True if the value represents a missing scalar, False otherwise.
    """
    if value is None:
        return True
    if isinstance(value, (list, dict, set, tuple, np.ndarray)):
        return False
    if str(value).strip() == "<NA>":
        return True
    with contextlib.suppress(Exception):
        pd = sys.modules.get("pandas")
        if pd is not None:
            res = pd.isna(value)
            if isinstance(res, (bool, np.bool_)):
                return bool(res)
    with contextlib.suppress(TypeError, ValueError):
        if (isinstance(value, float) and math.isnan(value)) or (
            isinstance(value, (np.floating, np.integer)) and np.isnan(value)
        ):
            return True
    return False


def row_id_patient_part(patient_id: Any) -> str:
    """Return the patient part of a derived ``row_id``: ``"None"`` when missing or blank."""
    if is_null(patient_id) or str(patient_id).strip() == "":
        return "None"
    return str(patient_id)
