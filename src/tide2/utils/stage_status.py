"""Per-stage status shared by every actor.

Each stage that handles a row adds one entry to the row's ``stage_status_json``
(``{"recognizer": {"status": "failed", "reason": "KeyError:start"}}``) and the
row's ``processing_status`` is the worst status across stages. A ``reason`` holds
an exception class and a fixed code, or entity types; never an exception message,
which can echo note text.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from tide2.utils.nulls import is_null

SUCCESS = "success"
DEGRADED = "degraded"
FAILED = "failed"

_RANK = {SUCCESS: 0, DEGRADED: 1, FAILED: 2}


class NoteError(Exception):
    """A per-note input problem; ``code`` is a fixed identifier that is safe to record."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def load_stage_status(value: Any) -> dict[str, dict[str, str]]:
    """Parse a ``stage_status_json`` value; anything null or malformed is an empty dict."""
    if is_null(value) or not isinstance(value, (str, bytes)):
        return {}
    try:
        parsed = json.loads(value)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def rollup_status(stages: dict[str, dict[str, str]]) -> str:
    """Return the worst status across stage entries (``success`` when there are none)."""
    worst = SUCCESS
    for entry in stages.values():
        status = entry.get("status", SUCCESS) if isinstance(entry, dict) else SUCCESS
        if _RANK.get(status, 0) > _RANK[worst]:
            worst = status
    return worst


def append_status(previous: Any, stage: str, status: str, reason: str | None = None) -> tuple[str, str]:
    """Add this stage's entry to the row's status.

    Args:
        previous: The row's incoming ``stage_status_json`` (may be null or malformed).
        stage: The stage key, such as ``"recognizer"``.
        status: ``success``, ``degraded`` or ``failed``.
        reason: Optional short reason; see the module docstring for what it may hold.

    Returns:
        ``(stage_status_json, processing_status)``.
    """
    stages = load_stage_status(previous)
    entry = {"status": status}
    if reason:
        entry["reason"] = reason
    stages[stage] = entry
    return json.dumps(stages), rollup_status(stages)


def _entry_rank(entry: Any) -> int:
    return _RANK.get(entry.get("status"), 0) if isinstance(entry, dict) else 0


def merge_stage_status(first: Any, second: Any, first_status: Any = None, second_status: Any = None) -> tuple[str, str]:
    """Merge two branches' status into ``(stage_status_json, processing_status)``.

    The result is never better than the worst of the two ``*_status`` flags and the
    stage entries. A stage present in both branches keeps its worse entry.
    """
    stages = load_stage_status(first)
    for key, entry in load_stage_status(second).items():
        if key not in stages or _entry_rank(entry) > _entry_rank(stages[key]):
            stages[key] = entry
    status = rollup_status(stages)
    for flag in (first_status, second_status):
        if _RANK.get(flag, 0) > _RANK[status]:
            status = flag
    return json.dumps(stages), status


def is_failed(processing_status: Any) -> bool:
    """Return True when an incoming ``processing_status`` value marks the row as failed."""
    return processing_status == FAILED


def failure_reason(exc: BaseException) -> str:
    """Return ``ExceptionClass`` or ``ExceptionClass:code``; never the exception message."""
    code = getattr(exc, "code", None)
    name = type(exc).__name__
    return f"{name}:{code}" if isinstance(code, str) else name


def log_note_failure(logger: logging.Logger, stage: str, text_hash: Any, exc: BaseException) -> None:
    """Log a failed note without its text; the traceback, which can echo input, is debug-only."""
    logger.error("%s failed note %s: %s", stage, str(text_hash)[:16], failure_reason(exc))
    logger.debug("%s failure detail", stage, exc_info=True)


def handle_row(
    logger: logging.Logger,
    stage: str,
    text_hash: Any,
    upstream_status: Any,
    upstream_stage: Any,
    fn: Callable[[], Any],
    status_of: Callable[[Any], tuple[str, str | None]] | None = None,
) -> tuple[Any, Any, str]:
    """Run one row's work under the failure contract.

    A row that failed upstream is not processed and keeps its stage JSON. An exception
    from ``fn`` fails the row. Otherwise the stage is recorded as ``success``, or as the
    ``(status, reason)`` that ``status_of(result)`` returns.

    Returns:
        ``(result, stage_status_json, processing_status)``; ``result`` is None for a failed row.
    """
    if is_failed(upstream_status):
        return None, upstream_stage, FAILED
    try:
        result = fn()
    except Exception as exc:
        log_note_failure(logger, stage, text_hash, exc)
        return (None, *append_status(upstream_stage, stage, FAILED, failure_reason(exc)))
    status, reason = status_of(result) if status_of else (SUCCESS, None)
    return (result, *append_status(upstream_stage, stage, status, reason))
