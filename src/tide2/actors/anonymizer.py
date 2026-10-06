"""
Ray Actor for batch anonymization processing using Presidio AnonymizerEngine.

This module provides a Ray Actor for batch processing clinical notes using
Microsoft Presidio's AnonymizerEngine with custom anonymizers.

Architecture:
    AnonymizerSupervisor (used by map_batches)
        └── AnonymizerWorker (does actual processing, can be killed on timeout)

    The supervisor pattern enables true note-level timeouts. When a note hangs,
    ray.kill() terminates the worker process and a new worker is spawned.

Output columns:
    - text_hash: SHA256 hash of original note_text
    - patient_id: Patient identifier (passed through from input)
    - anonymized_note_text: The anonymized text
    - anonymizer_results_json: JSON with anonymization details
    - entity_count: Number of entities anonymized
    - processing_timestamp: ISO timestamp of processing
    - processing_status: ``success``, ``degraded`` or ``failed`` (worst across stages)
    - stage_status_json: per-stage status and reason (see ``tide2.utils.stage_status``)

A note that cannot be anonymized is emitted as a failed row with null text and
results; a failure in one entity masks that entity as ``[ENTITY_TYPE]``.
"""

import hashlib
import logging
import os
import secrets
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson
import ray
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.entities import RecognizerResult

from tide2.anonymizers import AccessionNumberHashAnonymizer
from tide2.anonymizers import AgeGroupAnonymizer
from tide2.anonymizers import DateJitterAnonymizer
from tide2.anonymizers import FakerAnonymizer
from tide2.anonymizers import HipsAlphaNumericAnonymizer
from tide2.anonymizers import HipsLocationAnonymizer
from tide2.anonymizers import HipsNamesAnonymizer
from tide2.anonymizers import MaskingAnonymizer
from tide2.anonymizers import presidio_patches
from tide2.anonymizers.guarded import end_note
from tide2.anonymizers.guarded import guarded
from tide2.anonymizers.guarded import record_error
from tide2.anonymizers.guarded import start_note
from tide2.anonymizers.guarded import summarize
from tide2.cryptographic.date_jitter import derive_date_jitter
from tide2.cryptographic.fpe_strings import FormatPreservingEncryption
from tide2.utils.batch_columns import BatchColumns
from tide2.utils.batch_columns import _check_deprecated_patient_uid
from tide2.utils.batch_columns import type_all_null_columns
from tide2.utils.nulls import is_null
from tide2.utils.span_metrics import resolve_recognizer_results
from tide2.utils.stage_status import FAILED
from tide2.utils.stage_status import NoteError
from tide2.utils.stage_status import append_status
from tide2.utils.stage_status import failure_reason
from tide2.utils.stage_status import is_failed
from tide2.utils.stage_status import log_note_failure

logger = logging.getLogger(__name__)

# Key size requirements
REQUIRED_KEY_SIZE = 32

# Chunk size for anonymization: notes longer than this are split into chunks
# to avoid O(n*m) string concatenation in Presidio's TextReplaceBuilder.
# Must match or exceed recognizer chunk size so entities don't cross boundaries.
MAX_ANON_CHUNK_SIZE = 100_000


class AnonymizerWorker:
    """
    Worker class that executes anonymization processing directly under Ray Data.

    This worker holds the AnonymizerEngine state and processes batches of notes.

    Attributes:
        anonymizer_engine: The Presidio AnonymizerEngine instance.
    """

    def __init__(
        self,
        salt: bytes,
        key: bytes,
        acc_num_salt: str | None = None,
        acc_num_study_id: str | None = None,
        jitter_required: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the worker with an AnonymizerEngine.

        Args:
            salt: 32-byte salt for HIPS anonymizers.
            key: 32-byte key for HIPS anonymizers.
            acc_num_salt: Salt for accession number hashing (fixed per run).
            acc_num_study_id: Study ID for accession number hashing (fixed per run).
            jitter_required: If True, notes without a jitter value fail instead
                of computing one automatically.
            **kwargs: Deprecated parameters. Passing any deprecated argument
                will raise a ValueError with a deprecation warning.
        """
        from tide2.actors import check_deprecated_actor_kwargs

        check_deprecated_actor_kwargs(kwargs, "AnonymizerWorker")

        if salt is None or key is None:
            raise ValueError("Both salt and key must be provided")

        if len(salt) != REQUIRED_KEY_SIZE:
            raise ValueError(f"salt must be {REQUIRED_KEY_SIZE} bytes, got {len(salt)}")

        if len(key) != REQUIRED_KEY_SIZE:
            raise ValueError(f"key must be {REQUIRED_KEY_SIZE} bytes, got {len(key)}")

        # Apply Presidio patches - must be done in __init__ (not module level)
        # for Ray workers since they are separate processes.
        presidio_patches.disable_whitespace_merging()
        presidio_patches.patch_conflict_resolution()

        self.salt = salt
        self.key = key
        self.acc_num_salt = acc_num_salt
        self.acc_num_study_id = acc_num_study_id
        self.jitter_required = jitter_required

        # Suppress short input warnings in batch processing (reduces log noise)
        FormatPreservingEncryption.suppress_short_input_warnings = True

        # Initialize Presidio AnonymizerEngine
        self.anonymizer_engine = AnonymizerEngine()
        # Each operator is guarded so an error masks one entity instead of failing the note.
        for operator_cls in (
            AccessionNumberHashAnonymizer,
            FakerAnonymizer,
            DateJitterAnonymizer,
            HipsNamesAnonymizer,
            HipsAlphaNumericAnonymizer,
            HipsLocationAnonymizer,
            AgeGroupAnonymizer,
        ):
            self.anonymizer_engine.add_anonymizer(guarded(operator_cls))
        self.anonymizer_engine.add_anonymizer(MaskingAnonymizer)

        # Pre-create base operators with the provided keys
        self._base_operators = self._create_base_operators()

        logger.info("AnonymizerWorker initialized with Presidio AnonymizerEngine")

    @staticmethod
    def compute_text_hash(text: str | None) -> str:
        """Compute SHA256 hash of text."""
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    def _create_base_operators(self) -> dict[str, OperatorConfig]:
        """Create base operator configuration with the provided keys."""
        return {
            # Unknown entity types get a visible placeholder (recorded as a fallback), never a deletion.
            "DEFAULT": OperatorConfig("masking", {"fallback": True}),
            "OTHER": OperatorConfig("masking"),
            "BASE64_IMAGE": OperatorConfig("masking"),
            "GENETIC_SEQUENCE": OperatorConfig("faker_anonymizer"),
            "AGE": OperatorConfig("age_grouping", {"upper_limit": 89}),
            "EMAIL_ADDRESS": OperatorConfig("faker_anonymizer"),
            "WEB": OperatorConfig("faker_anonymizer"),
            "URL": OperatorConfig("faker_anonymizer"),
            "PHONE_NUMBER": OperatorConfig(
                "hips_alphanumeric",
                {"salt": self.salt, "key": self.key},
            ),
            "PHONE": OperatorConfig(
                "hips_alphanumeric",
                {"salt": self.salt, "key": self.key},
            ),
            "ORGANIZATION": OperatorConfig("faker_anonymizer"),
            "VENDOR": OperatorConfig(
                "hips_location",
                {"salt": self.salt, "key": self.key},
            ),
            "US_SSN": OperatorConfig(
                "hips_alphanumeric",
                {"salt": self.salt, "key": self.key},
            ),
            "ZIP_CODE": OperatorConfig(
                "hips_location",
                {"salt": self.salt, "key": self.key},
            ),
            "LOCATION": OperatorConfig(
                "hips_location",
                {"salt": self.salt, "key": self.key},
            ),
            "HOSPITAL": OperatorConfig(
                "hips_location",
                {"salt": self.salt, "key": self.key},
            ),
            "MEDICAL_LICENSE": OperatorConfig(
                "hips_alphanumeric",
                {"salt": self.salt, "key": self.key},
            ),
            "PERSON": OperatorConfig(
                "hips_names",
                {"salt": self.salt, "key": self.key},
            ),
            "DOCTOR": OperatorConfig(
                "hips_names",
                {"salt": self.salt, "key": self.key},
            ),
            "PATIENT": OperatorConfig(
                "hips_names",
                {"salt": self.salt, "key": self.key},
            ),
            "HCW": OperatorConfig(
                "hips_names",
                {"salt": self.salt, "key": self.key},
            ),
            "MRN": OperatorConfig(
                "hips_alphanumeric",
                {"salt": self.salt, "key": self.key},
            ),
            "HAR": OperatorConfig(
                "hips_alphanumeric",
                {"salt": self.salt, "key": self.key},
            ),
            # ACC_NUM is handled separately with per-note patient_id
            # See _create_operators_for_note()
            "ID": OperatorConfig(
                "hips_alphanumeric",
                {"salt": self.salt, "key": self.key},
            ),
            "CSN_ID": OperatorConfig(
                "hips_alphanumeric",
                {"salt": self.salt, "key": self.key},
            ),
        }

    def _create_operators_for_note(
        self,
        date_jitter: int | None = None,
        patient_id: Any = None,
        mask_dates: bool = False,
        **kwargs: Any,
    ) -> dict[str, OperatorConfig]:
        """
        Create operators including per-note parameters.

        Args:
            date_jitter: Jitter value for date anonymization.
            patient_id: Patient ID used as entity param for ACC_NUM hashing.
            mask_dates: Replace dates with the entity mask instead of shifting them.

        Returns:
            Dictionary of operator configurations for this note.
        """
        _check_deprecated_patient_uid(kwargs, location="AnonymizerWorker._create_operators_for_note")
        operators = self._base_operators.copy()

        # Random jitter between 4-60 days if not provided
        if date_jitter is None:
            date_jitter = secrets.randbelow(57) + 4

        if mask_dates:
            operators.update({"DATE_TIME": OperatorConfig("masking"), "DATE": OperatorConfig("masking")})
        else:
            operators.update(
                {
                    "DATE_TIME": OperatorConfig("date_jitter", {"jitter": date_jitter}),
                    "DATE": OperatorConfig("date_jitter", {"jitter": date_jitter}),
                }
            )

        # Convert numeric patient_id to string, map null/NaN/nan/none to None
        clean_patient_id: str | None = None
        if not is_null(patient_id):
            val_str = str(patient_id).strip()
            if val_str.lower() not in ("nan", "none", "null", ""):
                clean_patient_id = val_str

        # ACC_NUM uses accession_number_hash with per-note patient_id as SQL entity
        operators["ACC_NUM"] = OperatorConfig(
            "accession_number_hash",
            {
                "salt": self.acc_num_salt,
                "study_id": self.acc_num_study_id,
                "patient_id": clean_patient_id,
            },
        )

        return operators

    def _parse_recognizer_results(self, results_json: str | list | None, text_length: int) -> list[RecognizerResult]:
        """Parse and validate recognizer results.

        Args:
            results_json: A JSON list (string or already parsed) of results.
            text_length: Length of the note; every span must lie inside it.

        Raises:
            NoteError: If the results are null, malformed, or any span is invalid. A span we
                cannot trust cannot be masked, so the note fails instead of passing through.
        """
        if is_null(results_json) or results_json == "":
            raise NoteError("recognizer_results")

        if isinstance(results_json, (str, bytes)):
            try:
                # orjson parses 3-10x faster than the stdlib json
                results_list = orjson.loads(results_json)
            except orjson.JSONDecodeError as e:
                raise NoteError("recognizer_results") from e
        else:
            results_list = results_json

        if not isinstance(results_list, list):
            raise NoteError("recognizer_results")

        parsed = []
        for item in results_list:
            if isinstance(item, RecognizerResult):
                entity_type, start, end, score = item.entity_type, item.start, item.end, item.score
            elif isinstance(item, dict):
                entity_type = item.get("entity_type", "UNKNOWN")
                start, end, score = item.get("start"), item.get("end"), item.get("score", 1.0)
            else:
                raise NoteError("invalid_span")

            if (
                not isinstance(entity_type, str)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not (0 <= start <= end <= text_length)
            ):
                raise NoteError("invalid_span")
            parsed.append(
                item if isinstance(item, RecognizerResult) else RecognizerResult(entity_type, start, end, score)
            )
        return parsed

    def _compute_jitter_for_patient(self, patient_id: Any = None, **kwargs: Any) -> int:
        """
        Compute deterministic jitter for a patient when not provided.

        Uses the cryptographic date jitter derivation function to ensure
        consistent jitter for the same patient across runs.

        Args:
            patient_id: Patient identifier. If None, NaN, or empty,
                generates a random jitter.

        Returns:
            Integer jitter value in days.
        """
        _check_deprecated_patient_uid(kwargs, location="AnonymizerWorker._compute_jitter_for_patient")
        if is_null(patient_id):
            return secrets.randbelow(357) - 178  # Random between -178 and +178
        val_str = str(patient_id).strip()
        if val_str.lower() in ("nan", "none", "null", ""):
            return secrets.randbelow(357) - 178  # Random between -178 and +178

        return derive_date_jitter(
            patient_id=val_str,
            salt=self.salt,
            key=self.key,
            max_jitter_days=180,
            min_jitter_days=3,
        )

    def _anonymize_chunked(
        self,
        note_text: str,
        recognizer_results: list[RecognizerResult],
        operators: dict[str, OperatorConfig],
    ) -> tuple[str, list[dict]]:
        """
        Anonymize a long note by splitting into chunks.

        Presidio's TextReplaceBuilder does O(n) string concatenation per entity,
        making total cost O(entities * text_length). Chunking reduces this to
        O(entities * chunk_size).

        Entities that cross chunk boundaries are assigned to the chunk where they
        start and the chunk boundary is extended to include them.

        Args:
            note_text: Full note text.
            recognizer_results: Resolved recognizer results (sorted not required).
            operators: Operator configurations for anonymization.

        Returns:
            Tuple of (anonymized_text, result_items_list).
        """
        text_len = len(note_text)
        chunk_size = MAX_ANON_CHUNK_SIZE

        # Sort results by start position for efficient chunking
        recognizer_results.sort(key=lambda r: r.start)

        # Build chunk boundaries, adjusting for entities that cross boundaries
        chunk_boundaries = []  # list of (chunk_start, chunk_end)
        pos = 0
        result_idx = 0
        while pos < text_len:
            chunk_end = min(pos + chunk_size, text_len)

            # Extend chunk to include any entity that starts before chunk_end
            # but extends beyond it
            while result_idx < len(recognizer_results):
                r = recognizer_results[result_idx]
                if r.start >= chunk_end:
                    break
                chunk_end = max(chunk_end, r.end)
                result_idx += 1

            chunk_boundaries.append((pos, chunk_end))
            pos = chunk_end

        # Assign results to chunks and anonymize each chunk
        anonymized_chunks = []
        all_result_items = []
        result_idx = 0
        cumulative_offset = 0  # Track offset shift due to anonymization changing text length

        for chunk_start, chunk_end in chunk_boundaries:
            chunk_text = note_text[chunk_start:chunk_end]

            # Collect results for this chunk
            chunk_results = []
            while result_idx < len(recognizer_results):
                r = recognizer_results[result_idx]
                if r.start >= chunk_end:
                    break
                # Adjust offsets relative to chunk start
                chunk_results.append(
                    RecognizerResult(
                        entity_type=r.entity_type,
                        start=r.start - chunk_start,
                        end=r.end - chunk_start,
                        score=r.score,
                    )
                )
                result_idx += 1

            # Anonymize this chunk
            chunk_result = self.anonymizer_engine.anonymize(
                text=chunk_text,
                analyzer_results=chunk_results,
                operators=operators,
                merge_entities_with_spaces=False,  # rename-proof public equiv. of disable_whitespace_merging()
            )

            anonymized_chunks.append(chunk_result.text)

            # Adjust result item positions back to document-level coordinates
            for item in chunk_result.items:
                all_result_items.append(
                    {
                        "start": item.start + cumulative_offset,
                        "end": item.end + cumulative_offset,
                        "entity_type": item.entity_type,
                        "text": item.text,
                        "operator": item.operator,
                    }
                )

            cumulative_offset += len(chunk_result.text)

        anonymized_text = "".join(anonymized_chunks)
        return anonymized_text, all_result_items

    def process_note(
        self,
        note_text: str,
        original_text_hash: str,
        recognizer_results_json: str | list | None,
        patient_id: str | None = None,
        jitter: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process a single note and return results.

        This method is called by AnonymizerSupervisor for each note.
        If this method hangs, the supervisor will kill this worker via ray.kill().

        Args:
            note_text: The note text to anonymize.
            original_text_hash: SHA256 hash of the note.
            recognizer_results_json: Pre-computed recognizer results (JSON string).
            patient_id: Patient identifier.
            jitter: Per-note jitter value (computed if None/NaN).

        Returns:
            Dictionary with the anonymized note and ``stage_status`` (``success``,
            ``degraded`` when an entity fell back to its mask, or ``failed`` when an
            entity-level error was masked) with a ``stage_reason``.

        Raises:
            NoteError: If the recognizer results are unusable. Any other exception
                comes from outside the operators; the caller fails the note.
        """
        _check_deprecated_patient_uid(kwargs, location="AnonymizerWorker.process_note")
        recognizer_results = self._parse_recognizer_results(recognizer_results_json, len(note_text))

        events = start_note()
        try:
            # With jitter_required and no jitter, dates are masked instead of shifted
            mask_dates = False
            if is_null(jitter):
                if self.jitter_required:
                    mask_dates = True
                    record_error("DATE", "jitter_required")
                else:
                    jitter = self._compute_jitter_for_patient(patient_id)

            # Resolve conflicts and merge adjacent date spans in one pass
            recognizer_results = resolve_recognizer_results(
                recognizer_results,
                strategy="longest_wins",
                merge_adjacent_types={
                    "HCW",
                    "DOCTOR",
                    "HOSPITAL",
                    "VENDOR",
                    "DATE",
                    "DATE_TIME",
                    "PATIENT",
                    "PERSON",
                    "PHONE",
                    "ORGANIZATION",
                    "LOCATION",
                },
                text=note_text,
            )

            # Create operators with jitter and per-note patient_id
            operators = self._create_operators_for_note(jitter, patient_id, mask_dates=mask_dates)

            # Use chunked anonymization for long notes to avoid O(n*m) string copies
            if len(note_text) > MAX_ANON_CHUNK_SIZE:
                anonymized_text, result_items = self._anonymize_chunked(note_text, recognizer_results, operators)
                entity_count = len(result_items)
                anonymizer_json = orjson.dumps(result_items).decode("utf-8")
            else:
                # Short notes: use standard Presidio path
                anonymized_result = self.anonymizer_engine.anonymize(
                    text=note_text,
                    analyzer_results=recognizer_results,
                    operators=operators,
                    merge_entities_with_spaces=False,  # rename-proof public equiv. of disable_whitespace_merging()
                )

                anonymized_text = anonymized_result.text
                entity_count = len(anonymized_result.items)

                result_items = [
                    {
                        "start": item.start,
                        "end": item.end,
                        "entity_type": item.entity_type,
                        "text": item.text,
                        "operator": item.operator,
                    }
                    for item in anonymized_result.items
                ]
                anonymizer_json = orjson.dumps(result_items).decode("utf-8")
        finally:
            end_note()

        stage_status, stage_reason = summarize(events)
        return {
            "text_hash": original_text_hash,
            "patient_id": patient_id,
            "anonymized_note_text": anonymized_text,
            "anonymizer_results_json": anonymizer_json,
            "entity_count": entity_count,
            "stage_status": stage_status,
            "stage_reason": stage_reason,
        }

    def process_batch(self, batch: dict[str, Any]) -> dict[str, list[Any]]:  # noqa: PLR0915
        """
        Process a batch of notes in a single call. No IPC per note.

        Called by AnonymizerSupervisor to avoid per-note ray.get() overhead.

        Input columns are read into ``input_*`` locals (e.g. ``input_patient_ids``)
        and kept distinct from the output accumulators they feed (e.g.
        ``patient_ids``). This separation is deliberate: collapsing an input column
        and its output accumulator onto one name appends results back onto the input
        list, producing a ragged result dict that Ray silently drops at block-build
        time (0-row output).

        Args:
            batch: Dictionary with columnar data (note_text, recognizer_results_json, etc.).

        Returns:
            Dictionary with columnar results for all notes in the batch. Every list
            has one entry per input note; a failed note has null results.
        """
        original_text_hashes = []
        patient_ids = []
        anonymized_texts = []
        anonymizer_results_json_list = []
        entity_counts = []
        processing_statuses = []
        stage_statuses = []
        row_ids = []

        cols = BatchColumns(batch)
        _check_deprecated_patient_uid(cols, location="AnonymizerWorker.process_batch")
        batch_size = len(cols["note_text"])
        jitters = cols.get("jitter", [None] * batch_size)
        input_patient_ids = cols.get("patient_id", [None] * batch_size)
        input_row_ids = cols.get("row_id", [None] * batch_size)
        recognizer_results_list = cols.get("recognizer_results_json", [None] * batch_size)
        upstream_status_col = cols.get("processing_status", [None] * batch_size)
        upstream_stage_col = cols.get("stage_status_json", [None] * batch_size)

        note_texts = cols["note_text"]
        for i in range(batch_size):
            raw_note = note_texts[i]
            note_text = "" if is_null(raw_note) else str(raw_note)
            recognizer_results_json = recognizer_results_list[i] if i < len(recognizer_results_list) else None
            patient_id = input_patient_ids[i] if i < len(input_patient_ids) else None
            jitter = jitters[i] if i < len(jitters) else None

            original_text_hash = self.compute_text_hash(note_text)

            row_id = input_row_ids[i] if i < len(input_row_ids) else None
            upstream_stage = upstream_stage_col[i]

            result = None
            if is_failed(upstream_status_col[i]):
                stage_json, status = upstream_stage, FAILED
            else:
                try:
                    result = self.process_note(
                        note_text=note_text,
                        original_text_hash=original_text_hash,
                        recognizer_results_json=recognizer_results_json,
                        patient_id=patient_id,
                        jitter=jitter,
                    )
                    stage_json, status = append_status(
                        upstream_stage, "anonymizer", result["stage_status"], result["stage_reason"]
                    )
                    if result["stage_status"] == FAILED:
                        logger.error(
                            "anonymizer masked an entity of note %s: %s",
                            original_text_hash[:16],
                            result["stage_reason"],
                        )
                except Exception as exc:
                    result = None
                    log_note_failure(logger, "anonymizer", original_text_hash, exc)
                    stage_json, status = append_status(upstream_stage, "anonymizer", FAILED, failure_reason(exc))

            original_text_hashes.append(original_text_hash)
            patient_ids.append(patient_id)
            anonymized_texts.append(None if result is None else result["anonymized_note_text"])
            anonymizer_results_json_list.append(None if result is None else result["anonymizer_results_json"])
            entity_counts.append(None if result is None else result["entity_count"])
            processing_statuses.append(status)
            stage_statuses.append(stage_json)
            row_ids.append(row_id)

        batch_timestamp = datetime.now(UTC).isoformat()
        out = {
            "text_hash": original_text_hashes,
            "patient_id": patient_ids,
            "anonymized_note_text": anonymized_texts,
            "anonymizer_results_json": anonymizer_results_json_list,
            "entity_count": entity_counts,
            "processing_timestamp": [batch_timestamp] * len(original_text_hashes),
            "processing_status": processing_statuses,
            "stage_status_json": stage_statuses,
        }
        # Preserve row_id for checkpointing when input batch has the column
        if "row_id" in cols:
            out["row_id"] = row_ids
        return type_all_null_columns(out)

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """Process a batch of notes directly under Ray Data map_batches."""
        return self.process_batch(batch)


class AnonymizerSupervisor:
    """
    Deprecated supervisor shim for backwards compatibility.

    Delegates directly to AnonymizerWorker in-process. Ray Data now drives
    AnonymizerWorker directly with hang protection provided by Ray Data's
    execution-level no-progress timeout.
    """

    def __init__(
        self,
        salt: bytes,
        key: bytes,
        acc_num_salt: str | None = None,
        acc_num_study_id: str | None = None,
        jitter_required: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize supervisor shim (deprecated).

        Args:
            salt: 32-byte salt for HIPS anonymizers.
            key: 32-byte key for HIPS anonymizers.
            acc_num_salt: Salt for accession number hashing.
            acc_num_study_id: Study ID for accession number hashing.
            jitter_required: If True, notes without a jitter value fail instead
                of computing one automatically.
            **kwargs: Deprecated parameters. Passing any deprecated argument
                will raise a ValueError with a deprecation warning.
        """
        import warnings

        from tide2.actors import check_deprecated_actor_kwargs

        warnings.warn(
            "AnonymizerSupervisor is deprecated and will be removed in a future release. "
            "Pass AnonymizerWorker (or AnonymizerActor) directly to map_batches.",
            DeprecationWarning,
            stacklevel=2,
        )
        check_deprecated_actor_kwargs(kwargs, "AnonymizerSupervisor")
        self.worker = AnonymizerWorker(
            salt=salt,
            key=key,
            acc_num_salt=acc_num_salt,
            acc_num_study_id=acc_num_study_id,
            jitter_required=jitter_required,
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """Delegate batch processing directly to in-process worker."""
        return self.worker.process_batch(batch)


# Backwards compatibility aliases
AnonymizerActor = AnonymizerWorker
AnonymizerWorkerActor = ray.remote(AnonymizerWorker)


def _load_key_material(key_material: bytes | str | os.PathLike) -> bytes:
    """Load key material from bytes or file path."""
    if isinstance(key_material, bytes):
        return key_material
    # It's a path - read the file
    with Path(key_material).open("rb") as f:
        return f.read()


def create_anonymizer_actor(
    salt: bytes | str | os.PathLike,
    key: bytes | str | os.PathLike,
    acc_num_salt: str | None = None,
    acc_num_study_id: str | None = None,
    jitter_required: bool = False,
    **kwargs: Any,
) -> type[AnonymizerWorker]:
    """
    Factory function to create an AnonymizerWorker subclass with specific keys.

    This unified factory accepts keys as either raw bytes or file paths,
    making it work for both local/batch processing and cluster modes.

    Args:
        salt: 32-byte salt (bytes) or path to salt file
        key: 32-byte key (bytes) or path to key file
        acc_num_salt: Salt for accession number hashing (fixed per run)
        acc_num_study_id: Study ID for accession number hashing (fixed per run)
        jitter_required: If True, notes without a jitter value fail instead
            of computing one automatically
        **kwargs: Deprecated parameters. Passing any deprecated argument
            will raise a ValueError with a deprecation warning.

    Returns:
        A class that can be used with Ray Data's map_batches()

    Examples:
        # With raw bytes
        Actor = create_anonymizer_actor(salt_bytes, key_bytes)

        # With file paths
        Actor = create_anonymizer_actor("/path/to/salt.key", "/path/to/key.key")

        # Mixed
        Actor = create_anonymizer_actor(Path("/keys/salt.key"), key_bytes)
    """
    from tide2.actors import check_deprecated_actor_kwargs

    check_deprecated_actor_kwargs(kwargs, "create_anonymizer_actor")
    # Load key material (handles both bytes and file paths)
    salt_bytes = _load_key_material(salt)
    key_bytes = _load_key_material(key)

    class ConfiguredAnonymizerActor(AnonymizerWorker):
        """Pre-configured AnonymizerWorker with captured key material."""

        def __init__(self):
            super().__init__(
                salt=salt_bytes,
                key=key_bytes,
                acc_num_salt=acc_num_salt,
                acc_num_study_id=acc_num_study_id,
                jitter_required=jitter_required,
            )

    return ConfiguredAnonymizerActor


# Backwards compatibility alias
create_anonymizer_actor_class = create_anonymizer_actor
