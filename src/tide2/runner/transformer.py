"""
Transformer utilities for document chunking and prediction aggregation.

This module provides utilities for processing documents with transformer models:
- Document chunking with overlap
- Chunk prediction aggregation back to document-level entities
- Format conversion to Presidio RecognizerResult format
"""

import json
import logging
from datetime import UTC
from datetime import datetime
from typing import Any

import pandas as pd

from tide2.transformers.config import format_transformer_recognizer_name
from tide2.utils.text_processing import compute_text_hash
from tide2.utils.text_processing import deduplicate_overlapping_entities
from tide2.utils.text_processing import reconstruct_document_spans
from tide2.utils.text_processing import split_text_to_word_chunks

logger = logging.getLogger(__name__)

# Default processing parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 40
DEFAULT_BATCH_SIZE = 512

# Below this document count, reassembly runs serially instead of in a process pool
# (pool setup overhead is not worth it for small inputs).
MIN_DOCS_FOR_PROCESS_POOL = 100


def chunk_document_row(row: dict[str, Any], chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    """
    Expand a single document row into multiple chunk rows.

    This function is used with Ray Data's flat_map to convert each document
    into multiple chunks, maintaining metadata for later reconstruction.

    note_text is NOT included in chunk rows — only chunk_text is needed for
    GPU inference. This avoids duplicating full documents across chunks.

    Args:
        row: Dictionary with document data (text_hash, note_text, patient_id).
        chunk_size: Maximum chunk size in tokens.
        chunk_overlap: Overlap between chunks in tokens.

    Returns:
        List of dictionaries, one per chunk, with:
            - text_hash: Document hash
            - patient_id: Patient ID
            - chunk_id: Sequential chunk identifier
            - chunk_text: The actual chunk text
            - char_offset_start: Start position of chunk in original document
    """
    note_text = row.get("note_text") or ""
    text_hash = row.get("text_hash")
    patient_id = row.get("patient_id", "")

    # Compute text_hash if not provided
    if not text_hash:
        text_hash = compute_text_hash(note_text)

    # Split into chunks
    chunk_metadata_list = split_text_to_word_chunks(len(note_text), chunk_size, chunk_overlap, return_metadata=True)

    result = []
    for chunk_meta in chunk_metadata_list:
        chunk_text = note_text[chunk_meta["start"] : chunk_meta["end"]]
        chunk_id = chunk_meta["chunk_id"]
        result.append(
            {
                "text_hash": text_hash,
                "patient_id": patient_id or "",
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "char_offset_start": chunk_meta["start"],
                "chunk_uid": f"{text_hash}_{chunk_id}",
            }
        )

    return result


def reassemble_chunks_for_document(
    chunk_rows: list[dict[str, Any]],
    note_text: str,
    model_name: str,
) -> tuple[str, int]:
    """
    Reassemble chunk predictions into document-level recognizer results for a single document.

    This is the core reassembly logic used by both the small-scale pandas path
    (reassemble_document_predictions) and the streaming ReassemblyActor.

    Args:
        chunk_rows: List of chunk dicts, each with:
            - chunk_id: Sequential chunk identifier
            - char_offset_start: Start position of chunk in original document
            - predictions_json: JSON-serialized chunk predictions
        note_text: Original document text.
        model_name: Transformer model name for recognition_metadata.

    Returns:
        Tuple of (recognizer_results_json, entity_count).
    """
    recognizer_name = format_transformer_recognizer_name(model_name)

    # Check for failed chunks — skip reassembly if any chunk failed
    failed_chunks = [r for r in chunk_rows if r.get("chunk_status") == "failed"]
    if failed_chunks:
        failed_ids = [r.get("chunk_id") for r in failed_chunks]
        logger.warning(
            "Document has %d failed chunk(s) (chunk_ids=%s), skipping reassembly",
            len(failed_chunks),
            failed_ids,
        )
        return "[]", 0

    # Build chunk predictions list for reconstruction
    chunk_predictions = []
    for row in chunk_rows:
        try:
            predictions = json.loads(row["predictions_json"]) if row.get("predictions_json") else []
        except (json.JSONDecodeError, TypeError):
            predictions = []

        chunk_predictions.append(
            {
                "chunk_id": row["chunk_id"],
                "char_offset_start": row["char_offset_start"],
                "predictions": predictions,
            }
        )

    # Reconstruct document spans
    entities = reconstruct_document_spans(chunk_predictions, note_text)

    # Deduplicate overlapping entities
    entities = deduplicate_overlapping_entities(entities, iou_threshold=0.5)

    # Convert to Presidio RecognizerResult format
    ner_results = []
    for e in entities:
        start = e["start"]
        end = e["end"]
        matched_text = note_text[start:end] if note_text and start < len(note_text) else ""

        ner_results.append(
            {
                "entity_type": e["entity"],
                "start": start,
                "end": end,
                "score": e["score"],
                "analysis_explanation": None,
                "recognition_metadata": {
                    "recognizer_name": recognizer_name,
                    "matched_pattern": matched_text,
                    "recognizer_identifier": f"{recognizer_name}_{id(e)}",
                },
            }
        )

    return json.dumps(ner_results, ensure_ascii=False), len(ner_results)


def _reassemble_one_document(args: tuple[str, str, list[dict[str, Any]], str, str]) -> dict[str, Any]:
    """Reassemble a single document's chunk predictions. Used by ProcessPoolExecutor."""
    text_hash, patient_id, chunk_rows, note_text, model_name = args
    recognizer_results_json, entity_count = reassemble_chunks_for_document(chunk_rows, note_text, model_name)
    return {
        "text_hash": text_hash,
        "patient_id": patient_id,
        "note_text": note_text,
        "recognizer_results_json": recognizer_results_json,
        "entity_count": entity_count,
    }


def reassemble_document_predictions(
    df_chunks: pd.DataFrame,
    df_notes: pd.DataFrame,
    model_name: str,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """
    Reassemble chunk-level predictions into document-level recognizer results.

    This function operates on bounded pandas DataFrames (e.g. from reading
    parquet files) rather than within a Ray streaming pipeline, avoiding
    the need for groupby().map_groups() which breaks streaming.

    Uses multiprocessing to parallelize across documents when max_workers > 1.

    Args:
        df_chunks: DataFrame with chunk-level predictions:
            - text_hash, chunk_id, char_offset_start, predictions_json, patient_id
        df_notes: DataFrame with original documents:
            - text_hash, note_text, patient_id
        model_name: Transformer model name for recognition_metadata.
        max_workers: Number of parallel workers. Defaults to os.cpu_count().

    Returns:
        DataFrame with document-level results:
            - text_hash, patient_id, note_text, recognizer_results_json,
              entity_count, processing_timestamp
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    processing_timestamp = datetime.now(tz=UTC).isoformat()

    # Build note_text lookup
    note_lookup = df_notes.set_index("text_hash")["note_text"].to_dict()

    # Pre-group chunks into per-document work items
    work_items = []
    for text_hash, group in df_chunks.groupby("text_hash"):
        note_text = note_lookup.get(text_hash, "")
        patient_id = group["patient_id"].iloc[0] if "patient_id" in group.columns else ""
        chunk_rows = group.to_dict("records")
        work_items.append((text_hash, patient_id, chunk_rows, note_text, model_name))

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    if max_workers <= 1 or len(work_items) < MIN_DOCS_FOR_PROCESS_POOL:
        rows = [_reassemble_one_document(item) for item in work_items]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            rows = list(pool.map(_reassemble_one_document, work_items, chunksize=64))

    for row in rows:
        row["processing_timestamp"] = processing_timestamp

    return pd.DataFrame(rows)


def prepare_reassembly_input(
    df_chunks: pd.DataFrame,
    df_notes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join chunks with notes and aggregate chunks into a JSON array per document.

    This is the local equivalent of the BigQuery query:
        SELECT n.text_hash, n.patient_id, n.note_text,
               TO_JSON_STRING(ARRAY_AGG(
                   STRUCT(c.chunk_id, c.char_offset_start, c.predictions_json)
                   ORDER BY c.chunk_id
               )) AS chunks_json
        FROM notes n JOIN chunks c USING (text_hash)
        GROUP BY n.text_hash, n.patient_id, n.note_text

    Args:
        df_chunks: DataFrame with chunk-level predictions:
            text_hash, chunk_id, char_offset_start, predictions_json
        df_notes: DataFrame with original documents:
            text_hash, note_text, patient_id

    Returns:
        DataFrame with one row per document:
            text_hash, patient_id, note_text, chunks_json
    """
    note_lookup = df_notes.set_index("text_hash")[["note_text", "patient_id"]].to_dict("index")

    rows = []
    for text_hash, group in df_chunks.groupby("text_hash"):
        note_info = note_lookup.get(text_hash, {})

        chunks = group[["chunk_id", "char_offset_start", "predictions_json"]].sort_values("chunk_id").to_dict("records")

        rows.append(
            {
                "text_hash": text_hash,
                "patient_id": note_info.get("patient_id", ""),
                "note_text": note_info.get("note_text", ""),
                "chunks_json": json.dumps(chunks, ensure_ascii=False),
            }
        )

    return pd.DataFrame(rows)
