"""
Ray Actor for LLM-based PHI recognition processing.

This module provides a Ray Actor that uses LlmJsonRecognizer for LLM-based
entity detection in clinical text. Each actor instance holds a single
LlmJsonRecognizer and processes batches of notes serially.

Architecture:
    LlmRecognizerSupervisor (used by map_batches)
        └── LlmRecognizerWorker (does actual processing, can be killed on timeout)

    The supervisor pattern enables batch-level timeouts. When a batch hangs
    (e.g., an HTTP call blocks indefinitely), ray.kill() terminates the worker
    process and a new worker is spawned. The batch is returned empty so notes
    retry on the next run.

Concurrency:
    Each actor processes notes serially within a batch. Throughput comes from
    Ray's ActorPoolStrategy(size=N) — many actors process different batches
    in parallel, naturally matching the LLM API's rate limit.
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
from presidio_analyzer import RecognizerResult

from tide2.recognizers.llm_json_recognizer import LlmJsonRecognizer
from tide2.utils.batch_columns import BatchColumns
from tide2.utils.span_metrics import resolve_recognizer_results

# Chunking parameters for long notes
CHARS_PER_TOKEN = 4  # Approximation: 1 token ≈ 4 characters for English text
DEFAULT_CONTEXT_LENGTH = 128_000  # Default context window in tokens if not specified
LLM_CHUNK_OVERLAP = 2_000  # Character overlap to avoid missing entities at boundaries

logger = logging.getLogger(__name__)


def _is_null(value: Any) -> bool:
    """Check if a value is null/NaN (handles numpy NaN, None, and pandas NA)."""
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
        if isinstance(value, (np.floating, np.integer)) and np.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


@ray.remote
class LlmRecognizerWorker:
    """
    Ray Actor that does the actual LLM-based recognition processing.

    This worker holds the LlmJsonRecognizer state and processes batches of notes.
    It is managed by LlmRecognizerSupervisor which handles timeouts by killing
    and respawning this worker if a batch hangs.
    """

    def __init__(
        self,
        project_id: str | int,
        provider_type: str = "google",
        model_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        region: str = "us-central1",
        endpoint_id: int | None = None,
        max_retries: int = 3,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        prompt_name: str = "phi_detection",
    ) -> None:
        """
        Initialize the worker with an LlmJsonRecognizer.

        Args:
            project_id: Google Cloud project ID or project number.
            provider_type: LLM provider type (e.g., 'google', 'openai', 'anthropic').
            model_name: Name of the model to use.
            temperature: Model temperature for response generation.
            max_tokens: Maximum tokens for LLM output.
            region: Cloud region for the API.
            endpoint_id: Optional Vertex AI endpoint ID.
            max_retries: Maximum retry attempts for failed LLM requests.
            context_length: Model context window in tokens. Used to derive the
                maximum chunk size for long notes (context_length * 4 chars/token).
            prompt_name: Name of the prompt config in resources/llm_prompts/ (default: "phi_detection").
        """
        self.recognizer = LlmJsonRecognizer(
            project_id=project_id,
            provider_type=provider_type,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            region=region,
            endpoint_id=endpoint_id,
            max_retries=max_retries,
            prompt_name=prompt_name,
        )
        self.max_chunk_size = context_length * CHARS_PER_TOKEN
        logger.info(
            "LlmRecognizerWorker initialized with %s model=%s, chunk_size=%d chars (%d tokens * %d chars/token)",
            provider_type,
            model_name,
            self.max_chunk_size,
            context_length,
            CHARS_PER_TOKEN,
        )

    def _process_note(self, note_text: str) -> list[RecognizerResult]:
        """
        Process a single note, chunking if necessary.

        Args:
            note_text: The clinical text to analyze.

        Returns:
            List of RecognizerResult objects with document-level offsets.
        """
        if len(note_text) <= self.max_chunk_size:
            return self.recognizer.analyze(
                text=note_text,
                entities=self.recognizer.get_supported_entities(),
            )

        # Chunk long notes with overlap
        logger.info("Chunking long note (%d chars) for LLM processing", len(note_text))
        all_results: list[RecognizerResult] = []
        note_len = len(note_text)
        chunk_start = 0
        chunk_num = 0

        while chunk_start < note_len:
            chunk_end = min(chunk_start + self.max_chunk_size, note_len)
            chunk_text = note_text[chunk_start:chunk_end]

            chunk_results = self.recognizer.analyze(
                text=chunk_text,
                entities=self.recognizer.get_supported_entities(),
            )

            # Adjust offsets to document-level
            for result in chunk_results:
                adjusted = RecognizerResult(
                    entity_type=result.entity_type,
                    start=result.start + chunk_start,
                    end=result.end + chunk_start,
                    score=result.score,
                    analysis_explanation=result.analysis_explanation,
                    recognition_metadata=result.recognition_metadata,
                )
                all_results.append(adjusted)

            chunk_num += 1
            chunk_start = chunk_end - LLM_CHUNK_OVERLAP if chunk_end < note_len else note_len

        # Deduplicate overlapping results from chunk boundaries
        if all_results:
            all_results = resolve_recognizer_results(all_results, strategy="longest_wins")

        logger.debug(
            "Processed %d chars in %d chunks, found %d deduplicated entities",
            note_len,
            chunk_num,
            len(all_results),
        )
        return all_results

    def process_batch(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """
        Process a batch of notes via the LLM recognizer.

        Each note is processed serially within the batch. Per-note exceptions are
        caught and logged; the note is skipped and will retry on the next run.

        Args:
            batch: Dictionary with columnar data (note_text, text_hash).

        Returns:
            Dictionary with columnar results for successfully processed notes.
        """
        out_text_hashes: list[str] = []
        results_json_list: list[str] = []
        entity_counts: list[int] = []
        processing_statuses: list[str] = []
        error_messages: list[str | None] = []

        cols = BatchColumns(batch)
        batch_size = len(cols["note_text"])
        note_texts = cols["note_text"]
        input_text_hashes = cols["text_hash"]

        for i in range(batch_size):
            note_text = note_texts[i]
            text_hash = input_text_hashes[i]

            try:
                # Handle empty/null notes
                if not note_text or _is_null(note_text):
                    out_text_hashes.append(text_hash)
                    results_json_list.append("[]")
                    entity_counts.append(0)
                    processing_statuses.append("success")
                    error_messages.append(None)
                    continue

                start_time = _time.time()
                results = self._process_note(note_text)
                elapsed = _time.time() - start_time

                # Serialize only the fields the downstream anonymizer needs.
                # RecognizerResult.to_dict() includes AnalysisExplanation objects
                # that are not JSON-serializable; we skip them here.
                results_json = json.dumps(
                    [
                        {
                            "entity_type": r.entity_type,
                            "start": r.start,
                            "end": r.end,
                            "score": r.score,
                        }
                        for r in results
                    ]
                )

                logger.info(
                    "Processed note %s (%d chars) in %.2fs, found %d entities",
                    text_hash[:16],
                    len(note_text),
                    elapsed,
                    len(results),
                )

                out_text_hashes.append(text_hash)
                results_json_list.append(results_json)
                entity_counts.append(len(results))
                processing_statuses.append("success")
                error_messages.append(None)

            except Exception:
                logger.exception(
                    "Error processing note %s in batch, skipping (will retry on next run)",
                    text_hash,
                )
                continue

        return {
            "text_hash": out_text_hashes,
            "recognizer_results_json": results_json_list,
            "entity_count": entity_counts,
            "processing_status": processing_statuses,
            "error_message": error_messages,
        }


class LlmRecognizerSupervisor:
    """
    Supervisor actor that wraps LlmRecognizerWorker with batch-level timeout.

    This actor is used by Ray Data's map_batches(). It sends the entire batch
    to the worker in a single remote call to avoid per-note IPC overhead.
    If the batch times out, the worker is killed and respawned.

    The default batch_timeout is higher than RecognizerSupervisor (300s vs 120s)
    because LLM API calls are slower than regex — a single note can take 1-5
    seconds per LLM call, and with retries/chunking it could be longer.
    """

    BATCH_TIMEOUT_SECONDS = 300

    def __init__(
        self,
        project_id: str | int,
        provider_type: str = "google",
        model_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        region: str = "us-central1",
        endpoint_id: int | None = None,
        max_retries: int = 3,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        batch_timeout: int = BATCH_TIMEOUT_SECONDS,
        prompt_name: str = "phi_detection",
        worker_num_cpus: int | float | None = None,
    ) -> None:
        """
        Initialize supervisor with a worker actor.

        Args:
            project_id: Google Cloud project ID or project number.
            provider_type: LLM provider type (e.g., 'google', 'openai', 'anthropic').
            model_name: Name of the model to use.
            temperature: Model temperature for response generation.
            max_tokens: Maximum tokens for LLM output.
            region: Cloud region for the API.
            endpoint_id: Optional Vertex AI endpoint ID.
            max_retries: Maximum retry attempts for failed LLM requests.
            context_length: Model context window in tokens. Used to derive the
                maximum chunk size for long notes (context_length * 4 chars/token).
            batch_timeout: Seconds before a batch is killed and marked failed.
            prompt_name: Name of the prompt config in resources/llm_prompts/ (default: "phi_detection").
            worker_num_cpus: CPUs to reserve for each worker actor. None = Ray default (1).
                Set to 0 for I/O-bound oversubscription (logical scheduling only).
        """
        self.batch_timeout = batch_timeout
        self._worker_num_cpus = worker_num_cpus
        self._worker_kwargs = {
            "project_id": project_id,
            "provider_type": provider_type,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "region": region,
            "endpoint_id": endpoint_id,
            "max_retries": max_retries,
            "context_length": context_length,
            "prompt_name": prompt_name,
        }
        self.worker = self._spawn_worker()
        self.worker_kills = 0
        logger.info(
            "LlmRecognizerSupervisor initialized with %ds batch timeout, model=%s, worker_num_cpus=%s",
            self.batch_timeout,
            model_name,
            self._worker_num_cpus,
        )

    def _spawn_worker(self):
        cls = LlmRecognizerWorker
        if self._worker_num_cpus is not None:
            cls = cls.options(num_cpus=self._worker_num_cpus)
        return cls.remote(**self._worker_kwargs)

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """
        Process a batch of notes via a single remote call to worker.

        Sends the entire batch to worker.process_batch() in one ray.get().
        If timeout occurs, kills worker, respawns, and returns empty batch.

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

        except ray.exceptions.GetTimeoutError:
            logger.warning(
                "Batch timeout after %ds (%d notes), killing worker",
                self.batch_timeout,
                batch_size,
            )
            ray.kill(self.worker)
            self.worker = self._spawn_worker()
            self.worker_kills += 1

            return self._failed_batch(
                batch,
                f"BatchTimeout: exceeded {self.batch_timeout}s for {batch_size} notes",
            )

        except ray.exceptions.ActorDiedError as e:
            logger.warning(
                "Worker died processing batch of %d notes, respawning",
                batch_size,
            )
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
            # Add timestamp column (use actual result size since failed notes are skipped)
            result_size = len(result["text_hash"])
            result["processing_timestamp"] = [batch_timestamp] * result_size
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
LlmRecognizerActor = LlmRecognizerSupervisor
