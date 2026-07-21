"""
Ray Actor for batch recognition processing.

This module provides a Ray Actor that maintains AnalyzerEngine state for
processing batches of clinical notes. Each actor instance holds a single
AnalyzerEngine with regex-based recognizers, and uses pre-computed DL results
passed via ad-hoc recognizers.

Architecture:
    RecognizerSupervisor (used by map_batches)
        └── RecognizerWorker (does actual processing, can be killed on timeout)

    The supervisor pattern enables true note-level timeouts. When a note hangs,
    ray.kill() terminates the worker process and a new worker is spawned.

Thread/Process Safety:
    Each Actor maintains its own AnalyzerEngine instance. A blank spaCy tokenizer
    (no NER/POS/DEP pipeline, no ``en_core_web_*`` download) is built once per actor
    at initialization. Pre-computed DL results and known values are passed as ad-hoc
    recognizers per-note.
"""

import json
import logging
import math
import time as _time
from datetime import UTC
from datetime import datetime
from typing import Any

import numpy as np
import ray
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer import EntityRecognizer
from presidio_analyzer import RecognizerRegistry
from presidio_analyzer import RecognizerResult
from presidio_analyzer.context_aware_enhancers import ContextAwareEnhancer
from presidio_analyzer.nlp_engine import NerModelConfiguration
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import DateRecognizer

from tide2.recognizers import AccessionRecognizer
from tide2.recognizers import AddressRecognizer
from tide2.recognizers import Base64ImageRecognizer
from tide2.recognizers import CachedResultsTransformerRecognizer
from tide2.recognizers import EmailRecognizer
from tide2.recognizers import GeneticSequenceRecognizer
from tide2.recognizers import HarRecognizer
from tide2.recognizers import MrnRecognizer
from tide2.recognizers import PhoneRecognizer
from tide2.recognizers import SsnRecognizer
from tide2.recognizers import UrlRecognizer
from tide2.recognizers import create_cached_recognizer
from tide2.recognizers import create_recognizers_for_patient
from tide2.utils.batch_columns import BatchColumns
from tide2.utils.span_metrics import resolve_recognizer_results


class _BlankSpacyNlpEngine(SpacyNlpEngine):
    """SpacyNlpEngine backed by a pre-loaded ``spacy.blank`` model (tokenization only).

    Sets the required attributes directly instead of calling ``super().__init__()``.
    Newer presidio versions eagerly call ``spacy.load("en_core_web_lg")`` inside
    ``SpacyNlpEngine.__init__``; that model is neither needed (we only tokenize) nor
    shipped in our images, so calling the parent constructor raises
    ``OSError: [E050] Can't find model 'en_core_web_lg'`` and kills the actor. Bypassing
    it keeps this engine correct on both lazy- and eager-loading presidio builds.
    """

    def __init__(self, loaded_spacy_model):
        self.nlp = {"en": loaded_spacy_model}
        self.models = [{"lang_code": "en", "model_name": "blank"}]
        self.ner_model_configuration = NerModelConfiguration()


# Threshold constants for logging
LONG_NOTE_CHAR_THRESHOLD = 100_000
SLOW_PROCESSING_SECONDS = 10

# Per-note timeout - worker is killed if exceeded
# 60s is sufficient based on benchmarks; anything longer indicates a hang
NOTE_PROCESSING_TIMEOUT_SECONDS = 60


class NoteProcessingTimeoutError(Exception):
    """Raised when note processing exceeds the timeout limit."""

    pass


# Long note chunking parameters
# Notes longer than this will be processed in chunks to avoid memory issues
MAX_NOTE_CHUNK_SIZE = 100_000  # 100KB per chunk
CHUNK_OVERLAP = 2000  # Character overlap to avoid missing entities at boundaries

# All entity types to detect - includes both regex recognizer types and transformer types
# This ensures transformer entities (PATIENT, DOCTOR, etc.) are not filtered out
ALL_SUPPORTED_ENTITIES = [
    # Transformer entity types (from StanfordAIMI model and others)
    "PATIENT",
    "DOCTOR",
    "HCW",  # Healthcare Worker
    "PERSON",
    "HOSPITAL",
    "LOCATION",
    "DATE",
    "AGE",
    "ID",
    "PHONE",
    "WEB",
    "OTHER",
    "VENDOR",
    # Regex recognizer entity types
    "DATE_TIME",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "URL",
    "MRN",
    "HAR",
    "ACC_NUM",
    "CSN_ID",
    "BASE64_IMAGE",
    "GENETIC_SEQUENCE",
    # Known values recognizer types
    "MEDICAL_LICENSE",
]

logger = logging.getLogger(__name__)


def _is_null(value: Any) -> bool:
    """Check if a value is null/NaN (handles numpy NaN, None, and pandas NA)."""
    if value is None:
        return True
    try:
        # Handle numpy/pandas null values
        if isinstance(value, float) and math.isnan(value):
            return True
        if isinstance(value, (np.floating, np.integer)) and np.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


class NoOpContextEnhancer(ContextAwareEnhancer):
    """No-op context enhancer that returns results unchanged for maximum batch throughput."""

    def __init__(self) -> None:
        """Initialize with dummy parameters since we won't use them."""
        super().__init__(
            context_similarity_factor=0.0,
            min_score_with_context_similarity=0.0,
            context_prefix_count=0,
            context_suffix_count=0,
        )

    def enhance_using_context(
        self,
        text: str,
        raw_results: list[RecognizerResult],
        nlp_artifacts: NlpArtifacts,
        recognizers: list[EntityRecognizer],
        context: list[str] | None = None,
    ) -> list[RecognizerResult]:
        """Return results unchanged without any context enhancement.

        This is a no-op override for maximum batch throughput — it skips
        the expensive context similarity computations of the parent class.

        Args:
            text: The analyzed text.
            raw_results: Recognition results to (not) enhance.
            nlp_artifacts: NLP artifacts from the analyzer (unused).
            recognizers: List of recognizers that produced the results (unused).
            context: Optional context words (unused).

        Returns:
            The raw_results list, unmodified.
        """
        return raw_results


@ray.remote
class RecognizerWorker:
    """
    Ray Actor that does the actual recognition processing.

    This worker holds the AnalyzerEngine state and processes individual notes.
    It is managed by RecognizerSupervisor which handles timeouts by killing
    and respawning this worker if a note hangs.

    Attributes:
        analyzer: The Presidio AnalyzerEngine instance with regex recognizers.
    """

    def __init__(self) -> None:
        """
        Initialize the worker with an AnalyzerEngine.

        Creates a minimal NLP engine with spaCy (tokenization only), registers
        regex-based recognizers, and disables context enhancement for maximum
        throughput in batch processing scenarios.
        """
        # Patch Presidio's O(n²) remove_duplicates with a no-op passthrough.
        # Deduplication is handled downstream on the anonymizer side.
        from tide2.anonymizers.presidio_patches import patch_remove_duplicates

        patch_remove_duplicates()

        # Use spacy.blank("en") instead of en_core_web_sm for tokenization.
        # The blank model has the same tokenizer, stop words, and punctuation
        # detection but skips NER/POS/DEP pipelines we don't use (~16x faster).
        # See: https://microsoft.github.io/presidio/analyzer/nlp_engines/spacy_stanza/
        # NOTE: _BlankSpacyNlpEngine must NOT call super().__init__() — newer presidio
        # builds eagerly load en_core_web_lg there; see its docstring for the rationale.
        import spacy

        blank_nlp = spacy.blank("en")
        blank_nlp.max_length = 2_000_000
        nlp_engine = _BlankSpacyNlpEngine(loaded_spacy_model=blank_nlp)

        # Create recognizer registry with regex-based recognizers only
        registry = RecognizerRegistry()

        # Add standard Presidio recognizers
        registry.add_recognizer(DateRecognizer())

        # Add high-performance custom recognizers
        registry.add_recognizer(EmailRecognizer())
        registry.add_recognizer(PhoneRecognizer())
        registry.add_recognizer(SsnRecognizer())
        registry.add_recognizer(UrlRecognizer())

        # Add medical-specific recognizers
        registry.add_recognizer(MrnRecognizer())
        registry.add_recognizer(HarRecognizer())
        registry.add_recognizer(AccessionRecognizer())

        # Add specialized content recognizers
        registry.add_recognizer(Base64ImageRecognizer())
        registry.add_recognizer(GeneticSequenceRecognizer())

        # Add address recognizer (high-precision US address detection)
        registry.add_recognizer(AddressRecognizer())

        # Remove SpaCy recognizer to avoid conflicts with cached DL results
        registry.remove_recognizer("SpacyRecognizer")

        self.analyzer = AnalyzerEngine(
            registry=registry,
            nlp_engine=nlp_engine,
            supported_languages=["en"],
            context_aware_enhancer=NoOpContextEnhancer(),  # No-op enhancer for batch performance
        )

        logger.info("RecognizerWorker initialized with optimized AnalyzerEngine")

    def process_note(
        self,
        note_text: str,
        text_hash: str,
        cached_results: str | None,
        patient_identifiers: str | dict | None,
    ) -> dict[str, Any]:
        """
        Process a single note and return results.

        This method is called by RecognizerSupervisor for each note.
        If this method hangs, the supervisor will kill this worker via ray.kill().

        Args:
            note_text: The note text to process.
            text_hash: SHA256 hash of the note (used as ID).
            cached_results: Optional pre-computed DL results (JSON string).
            patient_identifiers: Optional patient PHI dict (JSON string or dict).

        Returns:
            Dictionary with processing results for this note.
        """
        start_time = _time.time()

        # Handle empty/null notes
        if not note_text or _is_null(note_text):
            return {
                "text_hash": text_hash,
                "recognizer_results_json": "[]",
                "entity_count": 0,
                "processing_status": "success",
                "error_message": None,
            }

        note_len = len(note_text)

        # Build ad-hoc recognizers list
        ad_hoc_recognizers = self._build_ad_hoc_recognizers(
            cached_results=cached_results,
            patient_identifiers=patient_identifiers,
            text_hash=text_hash,
        )

        # Process note (with chunking for very long notes)
        if note_len > MAX_NOTE_CHUNK_SIZE:
            logger.info(f"Chunking long note ({note_len:,} chars): {text_hash[:16]}...")
            analyzer_results = self._process_long_note(
                note_text=note_text,
                ad_hoc_recognizers=ad_hoc_recognizers,
            )
        else:
            analyzer_results = self._analyze_text(
                text=note_text,
                ad_hoc_recognizers=ad_hoc_recognizers,
            )

        elapsed = _time.time() - start_time

        # Log slow processing
        if elapsed > SLOW_PROCESSING_SECONDS or note_len > LONG_NOTE_CHAR_THRESHOLD:
            logger.info(
                f"Processed note {text_hash[:16]} ({note_len:,} chars) "
                f"in {elapsed:.2f}s, found {len(analyzer_results)} entities"
            )

        # Serialize results to JSON
        results_json = json.dumps([r.to_dict() for r in analyzer_results])

        return {
            "text_hash": text_hash,
            "recognizer_results_json": results_json,
            "entity_count": len(analyzer_results),
            "processing_status": "success",
            "error_message": None,
        }

    def _build_ad_hoc_recognizers(
        self,
        cached_results: str | None,
        patient_identifiers: str | dict | None,
        text_hash: str,
    ) -> list:
        """Build list of ad-hoc recognizers for a note."""
        ad_hoc_recognizers = []

        # Add cached DL results recognizer if available
        if cached_results and not _is_null(cached_results):
            cached_recognizer = create_cached_recognizer(results=cached_results)
            ad_hoc_recognizers.append(cached_recognizer)

        # Add known values recognizers if patient PHI is available
        if patient_identifiers and not _is_null(patient_identifiers):
            try:
                phi_dict = (
                    json.loads(patient_identifiers) if isinstance(patient_identifiers, str) else patient_identifiers
                )
                if phi_dict and isinstance(phi_dict, dict):
                    known_value_recognizers = create_recognizers_for_patient(phi_dict)
                    ad_hoc_recognizers.extend(known_value_recognizers)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse patient_identifiers for note {text_hash}: {e}")

        return ad_hoc_recognizers

    def _analyze_text(
        self,
        text: str,
        ad_hoc_recognizers: list | None = None,
    ) -> list[RecognizerResult]:
        """Run analyzer on text and return results."""
        return self.analyzer.analyze(
            text=text,
            language="en",
            entities=ALL_SUPPORTED_ENTITIES,
            ad_hoc_recognizers=ad_hoc_recognizers if ad_hoc_recognizers else None,
        )

    def _process_long_note(
        self,
        note_text: str,
        ad_hoc_recognizers: list | None = None,
    ) -> list[RecognizerResult]:
        """
        Process very long notes by chunking with overlap.

        This handles notes that are too large to process in a single pass
        without risking memory issues. The overlap ensures entities that
        span chunk boundaries are still captured.

        Cached results recognizers (CachedResultsTransformerRecognizer) are
        excluded from per-chunk processing because their entities use
        document-level offsets.  They are added once after all chunks.

        Args:
            note_text: The full note text.
            ad_hoc_recognizers: Optional ad-hoc recognizers to use.

        Returns:
            Deduplicated list of RecognizerResults with document-level offsets.
        """
        all_results = []
        note_len = len(note_text)

        # Separate cached-results recognizers from per-chunk recognizers.
        # Cached recognizers carry document-level offsets and must not be
        # re-run on every chunk (which would duplicate all entities N times).
        chunk_recognizers = []
        cached_recognizers = []
        for rec in ad_hoc_recognizers or []:
            if isinstance(rec, CachedResultsTransformerRecognizer):
                cached_recognizers.append(rec)
            else:
                chunk_recognizers.append(rec)

        # Process in chunks with overlap
        chunk_start = 0
        chunk_num = 0

        while chunk_start < note_len:
            chunk_end = min(chunk_start + MAX_NOTE_CHUNK_SIZE, note_len)
            chunk_text = note_text[chunk_start:chunk_end]

            # Analyze this chunk (without cached recognizers)
            chunk_results = self._analyze_text(
                text=chunk_text,
                ad_hoc_recognizers=chunk_recognizers if chunk_recognizers else None,
            )

            # Adjust offsets to document-level
            for result in chunk_results:
                # Create new result with adjusted offsets
                adjusted_result = RecognizerResult(
                    entity_type=result.entity_type,
                    start=result.start + chunk_start,
                    end=result.end + chunk_start,
                    score=result.score,
                    analysis_explanation=result.analysis_explanation,
                    recognition_metadata=result.recognition_metadata,
                )
                all_results.append(adjusted_result)

            chunk_num += 1
            # Move to next chunk, accounting for overlap
            chunk_start = chunk_end - CHUNK_OVERLAP if chunk_end < note_len else note_len

        # Add cached results once (already at document-level offsets)
        for rec in cached_recognizers:
            cached_results = rec.analyze(note_text, ALL_SUPPORTED_ENTITIES)
            all_results.extend(cached_results)

        # Deduplicate overlapping results from chunk boundaries
        # Using the O(n log n) resolution algorithm
        if len(all_results) > 0:
            all_results = resolve_recognizer_results(all_results, strategy="longest_wins")

        logger.debug(
            f"Processed {note_len:,} chars in {chunk_num} chunks, found {len(all_results)} deduplicated entities"
        )

        return all_results

    def process_batch(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """
        Process a batch of notes in a single call. No IPC per note.

        Called by RecognizerSupervisor to avoid per-note ray.get() overhead.
        Each note is processed via process_note() internally.

        Args:
            batch: Dictionary with columnar data (note_text, text_hash, etc.).

        Returns:
            Dictionary with columnar results for all notes in the batch.
        """
        out_text_hashes = []
        results_json_list = []
        entity_counts = []
        processing_statuses = []
        error_messages = []

        cols = BatchColumns(batch)
        batch_size = len(cols["note_text"])
        note_texts = cols["note_text"]
        input_text_hashes = cols["text_hash"]
        cached_results_col = cols.get("recognizer_results_json", [None] * batch_size)
        patient_identifiers_col = cols.get("patient_identifiers", [None] * batch_size)

        for i in range(batch_size):
            note_text = note_texts[i]
            text_hash = input_text_hashes[i]
            cached_results = cached_results_col[i]
            patient_identifiers = patient_identifiers_col[i]

            try:
                result = self.process_note(
                    note_text=note_text,
                    text_hash=text_hash,
                    cached_results=cached_results,
                    patient_identifiers=patient_identifiers,
                )
                out_text_hashes.append(result["text_hash"])
                results_json_list.append(result["recognizer_results_json"])
                entity_counts.append(result["entity_count"])
                processing_statuses.append(result["processing_status"])
                error_messages.append(result["error_message"])
            except Exception:
                logger.exception("Error processing note %s in batch, skipping (will retry on next run)", text_hash)
                continue

        return {
            "text_hash": out_text_hashes,
            "recognizer_results_json": results_json_list,
            "entity_count": entity_counts,
            "processing_status": processing_statuses,
            "error_message": error_messages,
        }


class RecognizerSupervisor:
    """
    Supervisor actor that wraps RecognizerWorker with batch-level timeout.

    This actor is used by Ray Data's map_batches(). It sends the entire batch
    to the worker in a single remote call to avoid per-note IPC overhead.
    If the batch times out, the worker is killed and respawned.
    """

    # Batch timeout: generous enough for large/slow batches, short enough to detect hangs.
    # At ~50ms/note, 100 notes = 5s expected. 120s allows for outlier notes.
    BATCH_TIMEOUT_SECONDS = 120

    def __init__(
        self,
        batch_timeout: int = BATCH_TIMEOUT_SECONDS,
        worker_num_cpus: int | float | None = None,
    ) -> None:
        """
        Initialize supervisor with a worker actor.

        Args:
            batch_timeout: Seconds before a batch is killed and marked failed.
            worker_num_cpus: CPUs to reserve for each worker actor. None = Ray
                default (1). Set lower to fit small boxes.
        """
        self.batch_timeout = batch_timeout
        self._worker_num_cpus = worker_num_cpus
        self.worker = self._spawn_worker()
        self.worker_kills = 0
        logger.info(
            f"RecognizerSupervisor initialized with {self.batch_timeout}s batch timeout, "
            f"worker_num_cpus={self._worker_num_cpus}"
        )

    def _spawn_worker(self) -> ray.actor.ActorHandle:
        """Create a new RecognizerWorker, applying the CPU override when set."""
        cls = RecognizerWorker
        if self._worker_num_cpus is not None:
            cls = cls.options(num_cpus=self._worker_num_cpus)
        return cls.remote()

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
            result = ray.get(ref, timeout=self.batch_timeout)

            # Add timestamp column (use actual result size since failed notes are skipped)
            result_size = len(result["text_hash"])
            result["processing_timestamp"] = [batch_timestamp] * result_size

        except ray.exceptions.GetTimeoutError:
            logger.warning(f"Batch timeout after {self.batch_timeout}s ({batch_size} notes), killing worker")
            ray.kill(self.worker)
            self.worker = self._spawn_worker()
            self.worker_kills += 1

            return self._failed_batch(
                batch,
                f"BatchTimeout: exceeded {self.batch_timeout}s for {batch_size} notes",
            )

        except ray.exceptions.ActorDiedError as e:
            logger.warning("Worker died processing batch of %d notes, respawning", batch_size)
            self.worker = self._spawn_worker()
            self.worker_kills += 1

            return self._failed_batch(
                batch,
                f"ActorDiedError: {str(e)[:400]}",
            )

        except Exception as e:
            logger.exception("Error processing batch of %d notes", batch_size)

            return self._failed_batch(
                batch,
                f"{type(e).__name__}: {str(e)[:400]}",
            )

        else:
            return result

    def _failed_batch(self, batch: dict[str, Any], error: str) -> dict[str, list[Any]]:
        """Log failure and return empty result so failed notes are not checkpointed."""
        cols = BatchColumns(batch)
        text_hashes = list(cols["text_hash"])
        for th in text_hashes:
            logger.error("Note %s failed: %s (will retry on next run)", th, error)
        return {
            "text_hash": [],
            "recognizer_results_json": [],
            "entity_count": [],
            "processing_timestamp": [],
            "processing_status": [],
            "error_message": [],
        }


# Backwards compatibility alias
RecognizerActor = RecognizerSupervisor
