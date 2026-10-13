"""
Ray Actor for LLM-based PHI recognition processing.

This module provides a Ray Actor that uses LlmJsonRecognizer for LLM-based
entity detection in clinical text. Each actor instance holds a single
LlmJsonRecognizer and processes batches of notes serially. Ray Data drives
``LlmRecognizerWorker`` directly; a hang (for example an HTTP call that never
returns) is caught by Ray Data's execution-level no-progress timeout.

Concurrency:
    Each actor processes notes serially within a batch. Throughput comes from
    Ray's ActorPoolStrategy(size=N) — many actors process different batches
    in parallel, naturally matching the LLM API's rate limit.
"""

import json
import logging
import time as _time
from datetime import UTC
from datetime import datetime
from functools import partial
from typing import Any

from presidio_analyzer import RecognizerResult

from tide2.recognizers.llm_json_recognizer import LlmJsonRecognizer
from tide2.utils.batch_columns import BatchColumns
from tide2.utils.batch_columns import copy_passthrough
from tide2.utils.batch_columns import type_columns
from tide2.utils.nulls import is_null
from tide2.utils.span_metrics import resolve_recognizer_results
from tide2.utils.stage_status import handle_row

# Chunking parameters for long notes
CHARS_PER_TOKEN = 4  # Approximation: 1 token ≈ 4 characters for English text
DEFAULT_CONTEXT_LENGTH = 128_000  # Default context window in tokens if not specified
LLM_CHUNK_OVERLAP = 2_000  # Character overlap to avoid missing entities at boundaries

logger = logging.getLogger(__name__)


class LlmRecognizerWorker:
    """
    Worker class that executes LLM-based recognition directly under Ray Data.

    This worker holds the LlmJsonRecognizer state and processes batches of notes.
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

    def _recognize_note(self, note_text: Any, text_hash: str) -> tuple[str, int]:
        """Return ``(recognizer_results_json, entity_count)`` for one note."""
        if is_null(note_text) or not note_text:
            return "[]", 0
        start_time = _time.time()
        results = self._process_note(note_text)
        elapsed = _time.time() - start_time

        # RecognizerResult.to_dict() includes AnalysisExplanation objects that are not
        # JSON-serializable, so only the fields the anonymizer needs are written.
        results_json = json.dumps(
            [{"entity_type": r.entity_type, "start": r.start, "end": r.end, "score": r.score} for r in results]
        )
        logger.info(
            "Processed note %s (%d chars) in %.2fs, found %d entities",
            text_hash[:16],
            len(note_text),
            elapsed,
            len(results),
        )
        return results_json, len(results)

    def process_batch(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """
        Process a batch of notes via the LLM recognizer.

        Each note is processed serially within the batch. A per-note exception is
        logged and the note is emitted as a failed row, so the batch keeps one
        row per input row.

        Args:
            batch: Dictionary with columnar data (note_text, text_hash).

        Returns:
            Dictionary with columnar results, one entry per input note.
        """
        out_text_hashes: list[str] = []
        out_note_texts: list[str] = []
        results_json_list: list[str | None] = []
        entity_counts: list[int | None] = []
        processing_statuses: list[str] = []
        stage_statuses: list[str | None] = []

        cols = BatchColumns(batch)
        batch_size = len(cols["note_text"])
        note_texts = cols["note_text"]
        input_text_hashes = cols["text_hash"]
        upstream_status_col = cols.get("processing_status", [None] * batch_size)
        upstream_stage_col = cols.get("stage_status_json", [None] * batch_size)

        if batch_size == 0:
            res: dict[str, list[Any]] = {
                "text_hash": [],
                "note_text": [],
                "recognizer_results_json": [],
                "entity_count": [],
                "processing_timestamp": [],
                "processing_status": [],
                "stage_status_json": [],
            }
            copy_passthrough(batch, res, empty=True)
            return res

        for i in range(batch_size):
            note_text = note_texts[i]
            text_hash = input_text_hashes[i]
            result, stage_json, status = handle_row(
                logger,
                "llm_recognizer",
                text_hash,
                upstream_status_col[i],
                upstream_stage_col[i],
                partial(self._recognize_note, note_text, text_hash),
            )
            results_json, entity_count = result if result is not None else (None, None)

            out_text_hashes.append(text_hash)
            out_note_texts.append("" if is_null(note_text) else str(note_text))
            results_json_list.append(results_json)
            entity_counts.append(entity_count)
            processing_statuses.append(status)
            stage_statuses.append(stage_json)

        batch_timestamp = datetime.now(UTC).isoformat()
        res = {
            "text_hash": out_text_hashes,
            "note_text": out_note_texts,
            "recognizer_results_json": results_json_list,
            "entity_count": entity_counts,
            "processing_timestamp": [batch_timestamp] * len(out_text_hashes),
            "processing_status": processing_statuses,
            "stage_status_json": stage_statuses,
        }
        copy_passthrough(batch, res)
        return type_columns(res)

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """Process a batch of notes directly under Ray Data map_batches."""
        return self.process_batch(batch)


LlmRecognizerActor = LlmRecognizerWorker
