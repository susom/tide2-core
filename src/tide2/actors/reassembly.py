"""
Ray Actor for chunk-to-document reassembly.

Bridges transformer chunk-level predictions and the recognizer by
reassembling pre-grouped chunks into document-level results.

Input contract (one row per document, chunks pre-grouped in BigQuery):
    text_hash, patient_id, note_text, chunks_json
    Optional: patient_identifiers (passed through if present)

    chunks_json is a JSON array of objects:
        [{"chunk_id": 0, "char_offset_start": 0, "predictions_json": "..."}, ...]

Output (one row per document):
    text_hash, patient_id, note_text, recognizer_results_json, entity_count,
    processing_timestamp
    Optional: patient_identifiers (if present in input)

Usage:
    ds = ray.data.read_parquet(reassembly_input_path)
    ds = ds.map_batches(
        ReassemblyActor,
        batch_size=500,
        concurrency=num_actors,
        fn_constructor_kwargs={"model_name": "StanfordAIMI/stanford-deidentifier-v2"},
    )
    ds.write_parquet(output_path, compression="zstd")
"""

import json
import logging
from datetime import UTC
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class ReassemblyActor:
    """
    Ray Actor that reassembles chunk-level transformer predictions into
    document-level recognizer results.

    Expects each input row to contain a chunks_json column with all chunks
    for that document pre-grouped (e.g. via BigQuery ARRAY_AGG).
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the reassembly actor.

        Args:
            model_name: Transformer model name for entity-type mapping.
        """
        from tide2.runner.transformer import reassemble_chunks_for_document

        self._model_name = model_name
        self._reassemble = reassemble_chunks_for_document

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """Reassemble chunked predictions into document-level results."""
        text_hashes = list(batch["text_hash"])
        note_texts = list(batch["note_text"])
        patient_ids = list(batch.get("patient_id", [""] * len(text_hashes)))
        chunks_jsons = list(batch["chunks_json"])
        has_patient_identifiers = "patient_identifiers" in batch
        patient_identifiers_col = list(batch["patient_identifiers"]) if has_patient_identifiers else None

        out: dict[str, list] = {
            "text_hash": [],
            "patient_id": [],
            "note_text": [],
            "recognizer_results_json": [],
            "entity_count": [],
            "processing_timestamp": [],
        }
        if has_patient_identifiers:
            out["patient_identifiers"] = []

        timestamp = datetime.now(tz=UTC).isoformat()

        for i in range(len(text_hashes)):
            try:
                chunk_rows = json.loads(chunks_jsons[i]) if chunks_jsons[i] else []
            except (json.JSONDecodeError, TypeError):
                chunk_rows = []

            results_json, count = self._reassemble(chunk_rows, note_texts[i] or "", self._model_name)

            out["text_hash"].append(text_hashes[i])
            out["patient_id"].append(patient_ids[i])
            out["note_text"].append(note_texts[i])
            out["recognizer_results_json"].append(results_json)
            out["entity_count"].append(count)
            out["processing_timestamp"].append(timestamp)
            if has_patient_identifiers:
                out["patient_identifiers"].append(patient_identifiers_col[i])

        return out
