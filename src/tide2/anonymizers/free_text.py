"""Anonymization of one free-text cell from precomputed recognizer spans.

``anonymize_cell`` dispatches here for the ``FREE_TEXT`` entity type. The spans come from the
recognition stages and index into the exact cell text; no recognizer runs here.

This module imports ``tide2.actors.anonymizer`` and the operators it needs (``faker``, ``usaddress``),
so ``tide2.anonymizers.cell`` imports it only when a ``FREE_TEXT`` cell is anonymized. It must also
load without ``presidio_analyzer``, spaCy and torch; ``tests/test_cell_isolation.py`` enforces this.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.anonymizers.cell import DEFAULT_POLICY
from tide2.anonymizers.cell import AnonymizationPolicy
from tide2.utils.stage_status import DEGRADED
from tide2.utils.stage_status import FAILED
from tide2.utils.stage_status import log_note_failure

__docformat__ = "google"

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def _worker(salt: bytes, key: bytes, acc_num_salt: str | None, policy: AnonymizationPolicy) -> AnonymizerWorker:
    """Return the worker for these settings, built once per process (the engine setup is costly)."""
    return AnonymizerWorker(salt, key, policy=policy, acc_num_salt=acc_num_salt)


def anonymize_free_text(
    text: str,
    recognizer_results: Any,
    salt: bytes,
    key: bytes,
    patient_id: Any = None,
    jitter: int | None = None,
    birth_date: Any = None,
    acc_num_salt: str | None = None,
    policy: AnonymizationPolicy = DEFAULT_POLICY,
) -> str | None:
    """Anonymize a free-text cell with the spans found by the recognition stages.

    Gives the ``anonymized_note_text`` that ``AnonymizerWorker.process_note`` gives for the same
    text, spans, patient, jitter and birth date. Spans of the Faker types (email, web, URL,
    organization, genetic sequence) are replaced by random values, so the result is not
    deterministic for text that holds them.

    Args:
        text: Cell text; the spans index into it.
        recognizer_results: Spans as a JSON string or an already parsed list of dicts.
        salt: 32-byte salt.
        key: 32-byte key.
        patient_id: Patient identifier used to derive the date jitter and the accession hash.
        jitter: Explicit date jitter in days; wins over ``patient_id``.
        birth_date: Real birth date of the patient. Dates after the ``policy.max_age``-th birthday are
            moved to it; without it dates are not capped.
        acc_num_salt: Salt for accession number hashing.
        policy: Run options, given to the note worker.

    Returns:
        The anonymized text, or ``None`` when the spans are unusable (null, malformed or outside
        the text); the failure is logged by text hash, never the text. An entity whose operator
        fails is masked (for example ``[PERSON]``) and the rest of the text is kept.

    Raises:
        ValueError: If the keys or the policy are invalid.
    """
    worker = _worker(salt, key, acc_num_salt, policy)
    text_hash = AnonymizerWorker.compute_text_hash(text)
    try:
        result = worker.process_note(text, text_hash, recognizer_results, patient_id, jitter, birth_date)
    except Exception as exc:
        log_note_failure(logger, "anonymizer", text_hash, exc)
        return None

    if result["stage_status"] == FAILED:
        logger.error("anonymizer masked an entity of a cell: %s", result["stage_reason"])
    elif result["stage_status"] == DEGRADED:
        logger.debug("anonymizer degraded an entity of a cell: %s", result["stage_reason"])
    return result["anonymized_note_text"]
