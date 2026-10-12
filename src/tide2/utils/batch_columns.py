"""Case-insensitive column accessor for Ray Data batch dicts."""

import warnings
from typing import Any

import numpy as np
import pyarrow as pa

from tide2.utils.nulls import is_null


def _check_deprecated_patient_uid(container: Any, location: str = "") -> None:
    """Emit a DeprecationWarning and raise ValueError if deprecated patient_uid is present."""
    if container is None:
        return
    has_uid = False
    if hasattr(container, "columns"):
        has_uid = any(isinstance(c, str) and c.lower() == "patient_uid" for c in container.columns)
    elif isinstance(container, (str, bytes)):
        has_uid = (
            container.lower() == "patient_uid" if isinstance(container, str) else container.lower() == b"patient_uid"
        )
    else:
        try:
            if "patient_uid" in container:
                has_uid = True
            else:
                for item in container:
                    if isinstance(item, str) and item.lower() == "patient_uid":
                        has_uid = True
                        break
        except Exception:
            has_uid = False

    if has_uid:
        loc_str = f" in {location}" if location else ""
        msg = f"`patient_uid`{loc_str} is deprecated and no longer supported. Please use `patient_id` instead."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        raise ValueError(msg)


class BatchColumns:
    """Case-insensitive column accessor for Ray Data batch dicts.

    Parquet files may have columns in any case (e.g. "JITTER" vs "jitter").
    ``detect_columns`` returns the actual file column names, so the batch dict
    keys match the file.  This helper maps requested (lowercase) column names
    to whatever key actually exists in the batch, with zero data copying.

    Usage::

        cols = BatchColumns(batch)
        jitters = cols.get("jitter", [None] * n)
        texts   = cols["note_text"]
    """

    __slots__ = ("_batch", "_lower_map")

    def __init__(self, batch: dict[str, Any]) -> None:
        """Build a case-insensitive index over *batch* keys."""
        self._batch = batch
        self._lower_map: dict[str, str] = {k.lower(): k for k in batch}

    def get(self, name: str, default: Any = None) -> Any:
        """Look up *name* case-insensitively, returning *default* if absent."""
        actual = self._lower_map.get(name.lower())
        if actual is None:
            return default
        return self._batch[actual]

    def __getitem__(self, name: str) -> Any:
        actual = self._lower_map.get(name.lower())
        if actual is None:
            raise KeyError(name)
        return self._batch[actual]

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._lower_map


PASSTHROUGH_COLS: tuple[str, ...] = ("patient_identifiers", "patient_id", "jitter", "row_id")


def copy_passthrough(
    batch: dict[str, Any] | BatchColumns,
    res: dict[str, list[Any]],
    *,
    indices: list[int] | None = None,
    empty: bool = False,
) -> None:
    """Copy the optional passthrough columns from *batch* into *res* in place.

    Args:
        batch: Incoming Ray Data batch or BatchColumns accessor.
        res: Output batch being built; mutated in place.
        indices: Optional list of row indices to slice from *batch*.
        empty: When True, emit empty lists instead of copying values.
    """
    _check_deprecated_patient_uid(batch)
    cols = batch if isinstance(batch, BatchColumns) else BatchColumns(batch)
    for col in PASSTHROUGH_COLS:
        if col in cols:
            if empty:
                res[col] = []
            elif indices is not None:
                src = cols[col]
                res[col] = [src[i] for i in indices]
            else:
                res[col] = list(cols[col])


_INT_COLUMNS = ("entity_count", "jitter")
_STRING_COLUMNS = (
    "recognizer_results_json",
    "predictions_raw_json",
    "anonymized_note_text",
    "anonymizer_results_json",
    "stage_status_json",
    "patient_identifiers",
    "patient_id",
    "row_id",
)


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if is_null(value):
        return None
    # Ray turns a nullable integer column into float + NaN; 5.0 must read back as "5"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _int_value(value: Any) -> int | None:
    return None if is_null(value) else int(value)


def type_columns(res: dict[str, Any]) -> dict[str, Any]:
    """Give the fixed result and passthrough columns of *res* one Arrow type each, in place.

    ``jitter`` and ``entity_count`` become int64 and every other listed column becomes
    string, including ``patient_id`` whatever its input type. Values are normalized
    first (NaN to null, ``5.0`` to ``5`` or ``"5"``), so a batch of numbers read as float
    with NaN, a batch of strings and a batch of nulls write the same Parquet type, and a
    directory reads as one dataset. A column with any value stays a list, which Ray infers
    from the normalized values; an all-null column becomes a typed Arrow array, because
    Ray would write it as an Arrow ``null`` column.
    """
    for names, convert, arrow_type in (
        (_INT_COLUMNS, _int_value, pa.int64()),
        (_STRING_COLUMNS, _string_value, pa.string()),
    ):
        for name in names:
            values = res.get(name)
            if not isinstance(values, list):
                continue
            converted = [convert(value) for value in values]
            all_null = bool(converted) and all(value is None for value in converted)
            res[name] = pa.array(converted, type=arrow_type) if all_null else converted
    return res
