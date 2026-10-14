"""Tests for null/NaN detection utilities."""

import numpy as np

from tide2.utils.nulls import is_null
from tide2.utils.nulls import row_id_patient_part


def test_is_null_with_none():
    assert is_null(None) is True


def test_is_null_with_nan():
    assert is_null(float("nan")) is True
    assert is_null(np.nan) is True


def test_is_null_with_string_na():
    assert is_null("<NA>") is True
    assert is_null("  <NA>  ") is True
    assert is_null("None") is False
    assert is_null("") is False
    assert is_null("valid text") is False


def test_is_null_with_pandas_na():
    try:
        import pandas as pd

        assert is_null(pd.NA) is True
    except ImportError:
        pass


def test_is_null_with_non_scalars():
    assert is_null([1, 2, 3]) is False
    assert is_null({"key": "val"}) is False


def test_row_id_patient_part():
    assert row_id_patient_part("p123") == "p123"
    assert row_id_patient_part("") == "None"
    assert row_id_patient_part(None) == "None"
    assert row_id_patient_part(float("nan")) == "None"


def test_import_core_without_pandas():
    """Verify importing core utilities without pandas succeeds in a clean subprocess."""
    import subprocess
    import sys

    script = (
        "import sys\n"
        "sys.modules['pandas'] = None\n"
        "from tide2.utils.span_metrics import resolve_recognizer_results, resolve_overlapping_spans\n"
        "from tide2.utils.nulls import is_null\n"
        "assert is_null(None) is True\n"
        "assert is_null('<NA>') is True\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603
