"""Case-insensitive column accessor for Ray Data batch dicts."""

from typing import Any


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
