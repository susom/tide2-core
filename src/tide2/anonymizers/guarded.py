"""Per-entity failure isolation for the anonymizer operators.

``guarded`` wraps an operator so that an exception while anonymizing one entity
replaces only that entity with ``[ENTITY_TYPE]``; the other entities in the note
are anonymized normally. Operators call ``record_fallback`` when none of their
paths applies. The outcomes are collected per note and summarized as a status.
"""

import contextvars

from presidio_anonymizer.operators import Operator

from tide2.utils.stage_status import DEGRADED
from tide2.utils.stage_status import FAILED
from tide2.utils.stage_status import SUCCESS
from tide2.utils.stage_status import failure_reason

_events: contextvars.ContextVar[list[tuple[str, str]] | None] = contextvars.ContextVar(
    "tide2_anonymizer_events", default=None
)


def mask(entity_type: str) -> str:
    """Return the placeholder for an entity type."""
    return f"[{entity_type}]"


def start_note() -> list[tuple[str, str]]:
    """Start collecting outcomes for one note and return the list they are appended to."""
    events: list[tuple[str, str]] = []
    _events.set(events)
    return events


def end_note() -> None:
    """Stop collecting outcomes."""
    _events.set(None)


def record_fallback(entity_type: str) -> None:
    """Record that no path of an operator applied; a no-op outside ``start_note``."""
    events = _events.get()
    if events is not None:
        events.append(("fallback", entity_type))


def record_error(entity_type: str, reason: str) -> None:
    """Record an entity-level error as ``ENTITY_TYPE:reason``; a no-op outside ``start_note``."""
    events = _events.get()
    if events is not None:
        events.append(("error", f"{entity_type}:{reason}"))


def summarize(events: list[tuple[str, str]]) -> tuple[str, str | None]:
    """Return ``(status, reason)`` for a note: the first error, else the fallback entity types."""
    for kind, label in events:
        if kind == "error":
            return FAILED, label
    fallbacks = sorted({label for kind, label in events if kind == "fallback"})
    if fallbacks:
        return DEGRADED, ",".join(fallbacks)
    return SUCCESS, None


def guarded(cls: type[Operator]) -> type[Operator]:
    """Return a subclass of ``cls`` that masks an entity instead of raising.

    Presidio calls ``validate`` and ``operate`` per entity; both run inside the
    guard. An empty result for non-empty text is also masked, so no text is
    removed without a placeholder.
    """

    def validate(self: Operator, params: dict) -> None:
        """Do nothing; the real validation runs inside ``operate``."""

    def operate(self: Operator, text: str, params: dict) -> str:
        entity_type = params.get("entity_type", "DEFAULT")
        try:
            cls.validate(self, params)
            result = cls.operate(self, text, params)
        except Exception as exc:
            record_error(entity_type, failure_reason(exc))
            return mask(entity_type)
        if text.strip() and not result.strip():
            record_fallback(entity_type)
            return mask(entity_type)
        return result

    return type(cls.__name__, (cls,), {"validate": validate, "operate": operate, "__doc__": cls.__doc__})
