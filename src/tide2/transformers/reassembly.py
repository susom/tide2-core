"""
Chunk-to-document reassembly for transformer NER predictions.

Long documents are split into overlapping chunks before transformer inference
(see ``chunk_document_row``); this module reconstructs document-level entity
spans from the per-chunk predictions (see ``reassemble_chunks_for_document``).
"""

import json
import logging
from typing import Any

from tide2.transformers.config import format_transformer_recognizer_name
from tide2.utils.text_processing import compute_text_hash
from tide2.utils.text_processing import deduplicate_overlapping_entities
from tide2.utils.text_processing import reconstruct_document_spans
from tide2.utils.text_processing import split_text_to_word_chunks

logger = logging.getLogger(__name__)


def chunk_document_row(row: dict[str, Any], chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    """
    Expand a single document row into multiple chunk rows.

    note_text is NOT included in chunk rows — only chunk_text is needed for
    inference. This avoids duplicating full documents across chunks.

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
