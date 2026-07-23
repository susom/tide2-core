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
    - patient_uid: Patient identifier (passed through from input)
    - anonymized_note_text: The anonymized text
    - anonymizer_results_json: JSON with anonymization details
    - entity_count: Number of entities anonymized
    - processing_timestamp: ISO timestamp of processing
"""

import hashlib
import logging
import math
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
from tide2.anonymizers import presidio_patches
from tide2.cryptographic.date_jitter import derive_date_jitter
from tide2.cryptographic.fpe_strings import FormatPreservingEncryption
from tide2.utils.batch_columns import BatchColumns
from tide2.utils.span_metrics import resolve_recognizer_results

logger = logging.getLogger(__name__)

# Key size requirements
REQUIRED_KEY_SIZE = 32

# Per-note timeout - worker is killed if exceeded
# 60s is sufficient based on benchmarks; anything longer indicates a hang
NOTE_PROCESSING_TIMEOUT_SECONDS = 60

# Chunk size for anonymization: notes longer than this are split into chunks
# to avoid O(n*m) string concatenation in Presidio's TextReplaceBuilder.
# Must match or exceed recognizer chunk size so entities don't cross boundaries.
MAX_ANON_CHUNK_SIZE = 100_000


class NoteProcessingTimeoutError(Exception):
    """Raised when note processing exceeds the timeout limit."""

    pass


@ray.remote
class AnonymizerWorker:
    """
    Ray Actor that does the actual anonymization processing.

    This worker holds the AnonymizerEngine state and processes individual notes.
    It is managed by AnonymizerSupervisor which handles timeouts by killing
    and respawning this worker if a note hangs.

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
        """
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
        self.anonymizer_engine.add_anonymizer(AccessionNumberHashAnonymizer)
        self.anonymizer_engine.add_anonymizer(FakerAnonymizer)
        self.anonymizer_engine.add_anonymizer(DateJitterAnonymizer)
        self.anonymizer_engine.add_anonymizer(HipsNamesAnonymizer)
        self.anonymizer_engine.add_anonymizer(HipsAlphaNumericAnonymizer)
        self.anonymizer_engine.add_anonymizer(HipsLocationAnonymizer)
        self.anonymizer_engine.add_anonymizer(AgeGroupAnonymizer)

        # Pre-create base operators with the provided keys
        self._base_operators = self._create_base_operators()

        logger.info("AnonymizerWorker initialized with Presidio AnonymizerEngine")

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute SHA256 hash of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _create_base_operators(self) -> dict[str, OperatorConfig]:
        """Create base operator configuration with the provided keys."""
        return {
            # Use redact for unknown entity types (built-in Presidio operator)
            "DEFAULT": OperatorConfig("redact"),
            "OTHER": OperatorConfig("redact"),
            "BASE64_IMAGE": OperatorConfig("redact"),
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
            # ACC_NUM is handled separately with per-note patient_uid
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
        patient_uid: str | None = None,
    ) -> dict[str, OperatorConfig]:
        """
        Create operators including per-note parameters.

        Args:
            date_jitter: Jitter value for date anonymization.
            patient_uid: Patient UID used as entity param for ACC_NUM hashing.

        Returns:
            Dictionary of operator configurations for this note.
        """
        operators = self._base_operators.copy()

        # Random jitter between 4-60 days if not provided
        if date_jitter is None:
            date_jitter = secrets.randbelow(57) + 4

        operators.update(
            {
                "DATE_TIME": OperatorConfig("date_jitter", {"jitter": date_jitter}),
                "DATE": OperatorConfig("date_jitter", {"jitter": date_jitter}),
            }
        )

        # ACC_NUM uses the accession_number_hash anonymizer with per-note patient_uid
        # The entity parameter is the patient_uid which varies per note
        operators["ACC_NUM"] = OperatorConfig(
            "accession_number_hash",
            {
                "salt": self.acc_num_salt,
                "study_id": self.acc_num_study_id,
                "entity_type": patient_uid,  # Per-note: patient_uid as entity
            },
        )

        return operators

    def _parse_recognizer_results(self, results_json: str | list | None) -> list[RecognizerResult]:
        """Parse recognizer results from JSON string or list using orjson for speed."""
        if not results_json:
            return []

        try:
            # Use orjson for faster parsing (3-10x faster than stdlib json)
            results_list = orjson.loads(results_json) if isinstance(results_json, (str, bytes)) else results_json

            if not results_list:
                return []

            return [
                result
                if isinstance(result, RecognizerResult)
                else RecognizerResult(
                    entity_type=result.get("entity_type", "UNKNOWN"),
                    start=result.get("start", 0),
                    end=result.get("end", 0),
                    score=result.get("score", 1.0),
                )
                for result in results_list
            ]

        except (orjson.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to parse recognizer results: {e}")
            return []

    def _compute_jitter_for_patient(self, patient_uid: str | None) -> int:
        """
        Compute deterministic jitter for a patient when not provided.

        Uses the cryptographic date jitter derivation function to ensure
        consistent jitter for the same patient across runs.

        Args:
            patient_uid: Patient identifier. If None or empty,
                generates a random jitter.

        Returns:
            Integer jitter value in days.
        """
        if not patient_uid:
            # Fallback to random jitter if no patient ID
            return secrets.randbelow(357) - 178  # Random between -178 and +178

        return derive_date_jitter(
            patient_id=patient_uid,
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
        patient_uid: str | None,
        jitter: int | None,
    ) -> dict[str, Any]:
        """
        Process a single note and return results.

        This method is called by AnonymizerSupervisor for each note.
        If this method hangs, the supervisor will kill this worker via ray.kill().

        Args:
            note_text: The note text to anonymize.
            original_text_hash: SHA256 hash of the note.
            recognizer_results_json: Pre-computed recognizer results (JSON string).
            patient_uid: Patient identifier.
            jitter: Per-note jitter value (computed if None/NaN).

        Returns:
            Dictionary with processing results for this note.
        """
        # Compute jitter from patient ID if not provided or if NaN
        jitter_missing = jitter is None or (isinstance(jitter, float) and math.isnan(jitter))
        if jitter_missing:
            if self.jitter_required:
                raise ValueError(f"Jitter value is required but missing for note {original_text_hash[:16]}")
            jitter = self._compute_jitter_for_patient(patient_uid)

        # Parse recognizer results
        recognizer_results = self._parse_recognizer_results(recognizer_results_json)

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

        # Create operators with jitter and per-note patient_uid
        operators = self._create_operators_for_note(jitter, patient_uid)

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

        return {
            "text_hash": original_text_hash,
            "patient_uid": patient_uid,
            "anonymized_note_text": anonymized_text,
            "anonymizer_results_json": anonymizer_json,
            "entity_count": entity_count,
            "processing_status": "success",
            "error_message": None,
        }

    def process_batch(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """
        Process a batch of notes in a single call. No IPC per note.

        Called by AnonymizerSupervisor to avoid per-note ray.get() overhead.

        Input columns are read into ``input_*`` locals (e.g. ``input_patient_uids``)
        and kept distinct from the output accumulators they feed (e.g.
        ``patient_uids``). This separation is deliberate: collapsing an input column
        and its output accumulator onto one name appends results back onto the input
        list, producing a ragged result dict that Ray silently drops at block-build
        time (0-row output).

        Args:
            batch: Dictionary with columnar data (note_text, recognizer_results_json, etc.).

        Returns:
            Dictionary with columnar results for all notes in the batch. Every list
            has the same length (one entry per successfully processed note).
        """
        original_text_hashes = []
        patient_uids = []
        anonymized_texts = []
        anonymizer_results_json_list = []
        entity_counts = []
        processing_statuses = []
        error_messages = []
        row_ids = []

        cols = BatchColumns(batch)
        batch_size = len(cols["note_text"])
        jitters = cols.get("jitter", [None] * batch_size)
        input_patient_uids = cols.get("patient_uid", [None] * batch_size)
        input_row_ids = cols.get("row_id", [None] * batch_size)
        recognizer_results_list = cols.get("recognizer_results_json", [None] * batch_size)

        note_texts = cols["note_text"]
        for i in range(batch_size):
            note_text = note_texts[i]
            recognizer_results_json = recognizer_results_list[i] if i < len(recognizer_results_list) else None
            patient_uid = input_patient_uids[i] if i < len(input_patient_uids) else None
            jitter = jitters[i] if i < len(jitters) else None

            original_text_hash = self.compute_text_hash(note_text)

            row_id = input_row_ids[i] if i < len(input_row_ids) else None

            try:
                result = self.process_note(
                    note_text=note_text,
                    original_text_hash=original_text_hash,
                    recognizer_results_json=recognizer_results_json,
                    patient_uid=patient_uid,
                    jitter=jitter,
                )
                original_text_hashes.append(result["text_hash"])
                patient_uids.append(result["patient_uid"])
                anonymized_texts.append(result["anonymized_note_text"])
                anonymizer_results_json_list.append(result["anonymizer_results_json"])
                entity_counts.append(result["entity_count"])
                processing_statuses.append(result["processing_status"])
                error_messages.append(result["error_message"])
                row_ids.append(row_id)
            except Exception:
                logger.exception(
                    "Error anonymizing note %s in batch, skipping (will retry on next run)", original_text_hash[:8]
                )
                continue

        result = {
            "text_hash": original_text_hashes,
            "patient_uid": patient_uids,
            "anonymized_note_text": anonymized_texts,
            "anonymizer_results_json": anonymizer_results_json_list,
            "entity_count": entity_counts,
            "processing_status": processing_statuses,
            "error_message": error_messages,
        }
        # Preserve row_id for checkpointing when input batch has the column
        if "row_id" in batch:
            result["row_id"] = row_ids
        return result


class AnonymizerSupervisor:
    """
    Supervisor that wraps AnonymizerWorker with batch-level timeout.

    This class is used by Ray Data's map_batches(). It sends the entire batch
    to the worker in a single remote call to avoid per-note IPC overhead.
    If the batch times out, the worker is killed and respawned.
    """

    # Batch timeout: generous enough for large/slow batches, short enough to detect hangs.
    BATCH_TIMEOUT_SECONDS = 120

    def __init__(
        self,
        salt: bytes,
        key: bytes,
        acc_num_salt: str | None = None,
        acc_num_study_id: str | None = None,
        timeout: int = NOTE_PROCESSING_TIMEOUT_SECONDS,
        jitter_required: bool = False,
        worker_num_cpus: int | float | None = None,
    ) -> None:
        """
        Initialize supervisor with a worker actor.

        Args:
            salt: 32-byte salt for HIPS anonymizers.
            key: 32-byte key for HIPS anonymizers.
            acc_num_salt: Salt for accession number hashing.
            acc_num_study_id: Study ID for accession number hashing.
            timeout: Legacy per-note timeout (kept for backwards compatibility).
            jitter_required: If True, notes without a jitter value fail instead
                of computing one automatically.
            worker_num_cpus: CPUs to reserve for each worker actor. None = Ray
                default (1). Set lower to fit small boxes.
        """
        self.salt = salt
        self.key = key
        self.acc_num_salt = acc_num_salt
        self.acc_num_study_id = acc_num_study_id
        self.timeout = timeout
        self.jitter_required = jitter_required
        self._worker_num_cpus = worker_num_cpus
        self.worker = self._spawn_worker()
        self.worker_kills = 0
        logger.info(
            f"AnonymizerSupervisor initialized with {self.BATCH_TIMEOUT_SECONDS}s batch timeout, "
            f"worker_num_cpus={self._worker_num_cpus}"
        )

    def _spawn_worker(self) -> ray.actor.ActorHandle:
        """Create a new AnonymizerWorker actor, applying the CPU override when set."""
        cls = AnonymizerWorker
        if self._worker_num_cpus is not None:
            cls = cls.options(num_cpus=self._worker_num_cpus)
        return cls.remote(
            salt=self.salt,
            key=self.key,
            acc_num_salt=self.acc_num_salt,
            acc_num_study_id=self.acc_num_study_id,
            jitter_required=self.jitter_required,
        )

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute SHA256 hash of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """
        Process a batch of notes via a single remote call to worker.

        Sends the entire batch to worker.process_batch() in one ray.get().
        If timeout occurs, kills worker, respawns, and marks all notes as failed.

        Args:
            batch: Dictionary with columnar data from Ray Data.

        Returns:
            Dictionary with processed results in columnar format.
        """
        cols = BatchColumns(batch)
        batch_size = len(cols["note_text"])
        batch_timestamp = datetime.now(UTC).isoformat()

        try:
            ref = self.worker.process_batch.remote(batch)
            result = ray.get(ref, timeout=self.BATCH_TIMEOUT_SECONDS)

            # Add timestamp column (use actual result size since failed notes are skipped)
            result_size = len(result["text_hash"])
            result["processing_timestamp"] = [batch_timestamp] * result_size

        except ray.exceptions.GetTimeoutError:
            logger.warning(f"Batch timeout after {self.BATCH_TIMEOUT_SECONDS}s ({batch_size} notes), killing worker")
            ray.kill(self.worker)
            self.worker = self._spawn_worker()
            self.worker_kills += 1

            return self._failed_batch(
                batch,
                f"BatchTimeout: exceeded {self.BATCH_TIMEOUT_SECONDS}s for {batch_size} notes",
            )

        except ray.exceptions.ActorDiedError as e:
            logger.warning("Worker died anonymizing batch of %d notes, respawning", batch_size)
            self.worker = self._spawn_worker()
            self.worker_kills += 1

            return self._failed_batch(
                batch,
                f"ActorDiedError: {str(e)[:400]}",
            )

        except Exception as e:
            logger.exception("Error anonymizing batch of %d notes", batch_size)

            return self._failed_batch(
                batch,
                f"{type(e).__name__}: {str(e)[:400]}",
            )

        else:
            return result

    def _failed_batch(self, batch: dict[str, Any], error: str) -> dict[str, list[Any]]:
        """Log failure and return empty result so failed notes are not checkpointed."""
        cols = BatchColumns(batch)
        for note_text in cols["note_text"]:
            th = self.compute_text_hash(note_text)
            logger.error("Note %s failed: %s (will retry on next run)", th[:16], error)
        result = {
            "text_hash": [],
            "patient_uid": [],
            "anonymized_note_text": [],
            "anonymizer_results_json": [],
            "entity_count": [],
            "processing_timestamp": [],
            "processing_status": [],
            "error_message": [],
        }
        # Preserve row_id column in output schema when input has it
        if "row_id" in batch:
            result["row_id"] = []
        return result


# Backwards compatibility alias
AnonymizerActor = AnonymizerSupervisor


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
    worker_num_cpus: int | float | None = None,
) -> type[AnonymizerSupervisor]:
    """
    Factory function to create an AnonymizerSupervisor class with specific keys.

    This unified factory accepts keys as either raw bytes or file paths,
    making it work for both local/batch processing and cluster modes.

    Args:
        salt: 32-byte salt (bytes) or path to salt file
        key: 32-byte key (bytes) or path to key file
        acc_num_salt: Salt for accession number hashing (fixed per run)
        acc_num_study_id: Study ID for accession number hashing (fixed per run)
        jitter_required: If True, notes without a jitter value fail instead
            of computing one automatically
        worker_num_cpus: CPUs to reserve for each worker actor. None = Ray
            default (1). Set lower to fit small boxes.

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
    # Load key material (handles both bytes and file paths)
    salt_bytes = _load_key_material(salt)
    key_bytes = _load_key_material(key)

    class ConfiguredAnonymizerActor(AnonymizerSupervisor):
        """Pre-configured AnonymizerSupervisor with captured key material."""

        def __init__(self):
            super().__init__(
                salt=salt_bytes,
                key=key_bytes,
                acc_num_salt=acc_num_salt,
                acc_num_study_id=acc_num_study_id,
                jitter_required=jitter_required,
                worker_num_cpus=worker_num_cpus,
            )

    return ConfiguredAnonymizerActor


# Backwards compatibility alias
create_anonymizer_actor_class = create_anonymizer_actor
