#!/usr/bin/env python3
"""
TIDE 2.0 - PHI Inspection Visualizer

An interactive interface for:
- Inspecting clinical notes before and after de-identification side by side
- Visualizing detected PHI entities and anonymization replacements
- Filtering by entity type and recognizer engine
- Tabular breakdown of detected PHI metadata
- Auto-detecting and loading pipeline Parquet outputs or JSON directories

Usage:
    tide2-visualizer [path]
    # or
    streamlit run unified_interface.py [-- path]
"""

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import streamlit as st
from spacy import displacy

from tide2.runner.local_runner import add_row_id
from tide2.utils.text_processing import compute_text_hash

# Color mapping for entity types
COLORS = {
    "PERSON": "#FFAB91",
    "DOCTOR": "#6BCF7F",
    "MEDICAL_LICENSE": "#E8F5E8",
    "LOCATION": "#4FC3F7",
    "DATE_TIME": "#FFD93D",
    "EMAIL_ADDRESS": "#B39DDB",
    "PHONE_NUMBER": "#80DEEA",
    "PHONE": "#E6F3FF",
    "URL": "#F48FB1",
    "WEB": "#FFE4E1",
    "US_SSN": "#FFB74D",
    "US_DRIVER_LICENSE": "#81C784",
    "US_PASSPORT": "#F0F8E8",
    "US_ITIN": "#FFF2CC",
    "US_BANK_NUMBER": "#FFE082",
    "IBAN_CODE": "#F5F5DC",
    "CREDIT_CARD": "#FDF6E3",
    "CRYPTO": "#E0E0E0",
    "IP_ADDRESS": "#F0F4F8",
    "ID": "#F8F9FA",
    "AGE": "#D1C4E9",
    "HOSPITAL": "#C8E6C9",
    "ZIP": "#B2EBF2",
    "CITY": "#FFE0B2",
    "STATE": "#FFCCBC",
    "COUNTRY": "#D7CCC8",
}

RECOGNIZER_COLUMNS = {
    "row_id",
    "text_hash",
    "patient_id",
    "note_text",
    "original_note_text",
    "phi_note_text",
    "text",
    "value",
    "recognizer_results_json",
    "recognizer_results",
    "entities",
}

ANONYMIZER_COLUMNS = {
    "row_id",
    "text_hash",
    "patient_id",
    "anonymized_note_text",
    "deid_note_text",
    "anonymized_text",
    "note_text",
    "modified_text",
    "text",
    "anonymizer_results_json",
    "items",
    "anonymizer_results",
    "operators",
}

TRANSFORMER_COLUMNS = {
    "row_id",
    "text_hash",
    "patient_id",
    "note_text",
    "text",
    "value",
}


def parse_entities(raw: Any) -> list[dict[str, Any]]:
    """Parse entities from raw JSON string, list, numpy array, or dict format.

    Args:
        raw: Raw entity data, such as a JSON string, a list of dicts, or None.

    Returns:
        List of entity dictionaries.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []

    data: Any = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            return []

    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def get_recognizer_name(entity: dict[str, Any]) -> str:
    """Extract recognizer name from an entity dictionary.

    Args:
        entity: Entity dictionary, optionally containing ``recognition_metadata``.

    Returns:
        Recognizer name string, or ``"-"`` if not specified.
    """
    if "recognition_metadata" in entity and isinstance(entity["recognition_metadata"], dict):
        rec_name = entity["recognition_metadata"].get("recognizer_name")
        if rec_name:
            return str(rec_name)
    if entity.get("recognizer_name"):
        return str(entity["recognizer_name"])
    return "-"


def get_recognizers_from_entities(entities: list[dict[str, Any]]) -> list[str]:
    """Extract unique sorted recognizer names from a list of entity dicts.

    Args:
        entities: List of entity dictionaries.

    Returns:
        Sorted list of unique non-empty recognizer name strings.
    """
    recognizers = set()
    for entity in entities:
        name = get_recognizer_name(entity)
        if name != "-":
            recognizers.add(name)
    return sorted(recognizers)


def filter_entities(
    entities: list[dict[str, Any]],
    selected_recognizers: list[str] | None = None,
    selected_entity_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter entity dicts by recognizers and entity types.

    Args:
        entities: List of entity dictionaries.
        selected_recognizers: Recognizer names to keep. If None, empty, or
            containing ``"All"``, recognizer filtering is skipped.
        selected_entity_types: Entity types to keep. If None, empty, or
            containing ``"All"``, entity type filtering is skipped.

    Returns:
        Filtered list of entity dictionaries matching both criteria.
    """
    if not entities:
        return []

    filter_rec = bool(selected_recognizers and "All" not in selected_recognizers)
    filter_type = bool(selected_entity_types and "All" not in selected_entity_types)

    if not filter_rec and not filter_type:
        return entities

    filtered = []
    for ent in entities:
        if filter_rec and get_recognizer_name(ent) not in selected_recognizers:
            continue
        if filter_type and ent.get("entity_type") not in selected_entity_types:
            continue
        filtered.append(ent)
    return filtered


def filter_entities_by_recognizers(
    entities: list[dict[str, Any]],
    selected_recognizers: list[str],
) -> list[dict[str, Any]]:
    """Filter entities to include only those from the specified recognizers.

    Args:
        entities: List of entity dictionaries.
        selected_recognizers: List of recognizer names to keep.

    Returns:
        Filtered list of entity dictionaries.
    """
    return filter_entities(entities, selected_recognizers=selected_recognizers)


def convert_to_displacy_format(text: str, entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert text and entity dicts into spaCy displacy manual rendering format.

    Validates entity offsets to ensure they fall within the text bounds.

    Args:
        text: The source text containing the entities.
        entities: List of entity dicts, each with ``start``, ``end``, and
            ``entity_type`` keys.

    Returns:
        Dictionary with ``text`` and ``ents`` keys suitable for
        ``displacy.render(..., manual=True)``.
    """
    valid_ents = []
    text_len = len(text)
    for ent in entities:
        try:
            start = int(ent["start"])
            end = int(ent["end"])
            label = str(ent.get("entity_type", "ENTITY"))
            if 0 <= start < end <= text_len:
                valid_ents.append({"start": start, "end": end, "label": label})
        except (KeyError, ValueError, TypeError):
            continue

    return {
        "text": text,
        "ents": valid_ents,
    }


def collect_entity_types(
    original_phi: list[dict[str, Any]],
    anonymized_phi: list[dict[str, Any]],
) -> set[str]:
    """Collect unique entity types from original and anonymized entity lists.

    Args:
        original_phi: Entities from the recognizer stage.
        anonymized_phi: Entities or items from the anonymizer stage.

    Returns:
        Set of unique entity type strings.
    """
    entity_types: set[str] = set()
    for ent in original_phi:
        if ent.get("entity_type"):
            entity_types.add(str(ent["entity_type"]))
    for ent in anonymized_phi:
        if ent.get("entity_type"):
            entity_types.add(str(ent["entity_type"]))
    return entity_types


def build_legend_html(entity_types: set[str]) -> str:
    """Build HTML legend for entity type color chips.

    Args:
        entity_types: Set of entity type strings to include in the legend.

    Returns:
        HTML string with colored chips and labels, or empty string if no types.
    """
    if not entity_types:
        return ""
    legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px;'>"
    for entity_type in sorted(entity_types):
        color = COLORS.get(entity_type, "#CCCCCC")
        legend_html += f"""
            <div style='display: flex; align-items: center; margin: 4px;'>
                <div style='width: 14px; height: 14px; background-color: {color};
                           border: 1px solid #bbb; margin-right: 6px; border-radius: 3px;'></div>
                <span style='font-size: 13px; font-weight: 500;'>{entity_type}</span>
            </div>
            """
    legend_html += "</div>"
    return legend_html


def _json_row_id(data: dict[str, Any], fallback_id: str) -> str:
    """Return a JSON record's identity: its non-blank ``row_id``, else the key/filename identity.

    Anonymizer JSON carries only ``text`` and ``items``, and its filename is the note's identity,
    so an id is never derived from the (replacement) text.
    """
    supplied = data.get("row_id")
    if supplied is not None and str(supplied).strip():
        return str(supplied)
    return fallback_id


def _parse_json_record(data: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    """Parse one recognizer and/or anonymizer JSON document into a unified record.

    ``text`` is the anonymized text when the document has anonymizer fields (``items``,
    ``modified_text``, ``anonymized_note_text``) and the original text otherwise.
    """
    has_rec = "recognizer_results" in data or "value" in data or "original_text" in data
    has_anon = "items" in data or "modified_text" in data or "anonymized_note_text" in data

    orig_text = ""
    orig_phi: list[dict[str, Any]] = []
    if has_rec:
        orig_text = str(
            data.get("value") or data.get("original_text") or (data.get("text") if not has_anon else "") or ""
        )
        orig_phi = parse_entities(data.get("recognizer_results") or data.get("original_phi") or [])

    mod_text = ""
    mod_phi: list[dict[str, Any]] = []
    if has_anon:
        mod_text = str(data.get("modified_text") or data.get("anonymized_note_text") or data.get("text") or "")
        mod_phi = parse_entities(data.get("items") or data.get("modified_phi") or [])
    elif not has_rec and "text" in data:
        orig_text = str(data.get("text", ""))

    row_id = _json_row_id(data, str(data.get("key") or data.get("text_hash") or fallback_id))
    return {
        "sample_id": row_id,
        "row_id": row_id,
        "text_hash": str(data["text_hash"]) if data.get("text_hash") else None,
        "original_text": orig_text,
        "original_phi": orig_phi,
        "modified_text": mod_text,
        "modified_phi": mod_phi,
    }


def load_analyzer_results(
    analyzer_dir: str | os.PathLike[str], max_notes: int | None = None
) -> dict[str, dict[str, Any]]:
    """Load analyzer results from a directory of JSON files.

    Each JSON file is expected to have ``key`` (sample ID), ``value`` (original
    text), and ``recognizer_results`` (list of entity dicts) fields.

    Args:
        analyzer_dir: Path to directory containing analyzer result JSON files.
        max_notes: Optional maximum number of notes to load.

    Returns:
        Dictionary mapping sample IDs to dicts with ``original_text`` and
        ``original_phi`` keys.
    """
    dir_path = Path(analyzer_dir)
    if not dir_path.is_dir():
        return {}

    results = {}
    for file_path in sorted(dir_path.glob("*.json")):
        if max_notes is not None and len(results) >= max_notes:
            break
        with contextlib.suppress(Exception):
            with file_path.open(encoding="utf-8") as file:
                data = json.load(file)
            sample_id = str(data.get("key") or file_path.stem)
            orig_text = str(data.get("value") or data.get("text") or data.get("original_text") or "")
            text_hash = str(data["text_hash"]) if data.get("text_hash") else None
            row_id = _json_row_id(data, sample_id)
            orig_phi = parse_entities(data.get("recognizer_results") or data.get("original_phi") or [])
            results[sample_id] = {
                "sample_id": sample_id,
                "row_id": row_id,
                "text_hash": text_hash,
                "original_text": orig_text,
                "original_phi": orig_phi,
            }

    return results


def load_anonymized_results(
    anonymizer_dir: str | os.PathLike[str],
    max_notes: int | None = None,
    allowed_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load anonymized results from a directory of JSON files.

    Each JSON file is expected to have ``text`` (anonymized text) and ``items``
    (list of anonymization item dicts) fields. The sample ID is derived from
    the filename stem or a ``key``/``row_id`` field.

    Args:
        anonymizer_dir: Path to directory containing anonymized result JSON files.
        max_notes: Optional maximum number of notes to load.
        allowed_ids: Optional set of sample IDs to restrict loading to.

    Returns:
        Dictionary mapping sample IDs to dicts with ``modified_text`` and
        ``modified_phi`` keys.
    """
    dir_path = Path(anonymizer_dir)
    if not dir_path.is_dir():
        return {}

    results = {}
    for file_path in sorted(dir_path.glob("*.json")):
        if max_notes is not None and len(results) >= max_notes:
            break
        if allowed_ids is not None and file_path.stem not in allowed_ids:
            continue
        with contextlib.suppress(Exception):
            with file_path.open(encoding="utf-8") as file:
                data = json.load(file)
            sample_id = str(data.get("key") or file_path.stem)
            if allowed_ids is not None and sample_id not in allowed_ids:
                continue
            mod_text = str(data.get("text") or data.get("modified_text") or data.get("anonymized_note_text") or "")
            text_hash = str(data["text_hash"]) if data.get("text_hash") else None
            row_id = _json_row_id(data, sample_id)
            mod_phi = parse_entities(
                data.get("items") or data.get("modified_phi") or data.get("anonymizer_results") or []
            )
            results[sample_id] = {
                "sample_id": sample_id,
                "row_id": row_id,
                "text_hash": text_hash,
                "modified_text": mod_text,
                "modified_phi": mod_phi,
            }

    return results


def _resolve_text_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None]:
    """Determine recognizer/anonymizer text and entity columns without ambiguous assignment.

    Returns:
        Tuple of (orig_text_col, rec_phi_col, anon_text_col, anon_phi_col).
    """
    rec_phi_col = next(
        (c for c in ("recognizer_results_json", "recognizer_results", "entities") if c in df.columns),
        None,
    )
    anon_phi_col = next(
        (c for c in ("anonymizer_results_json", "items", "anonymizer_results", "operators") if c in df.columns),
        None,
    )

    has_rec = rec_phi_col is not None
    has_anon = anon_phi_col is not None

    orig_text_candidates = ("original_note_text", "phi_note_text", "value")
    anon_text_candidates = ("anonymized_note_text", "deid_note_text", "anonymized_text", "modified_text")

    explicit_orig = next((c for c in orig_text_candidates if c in df.columns), None)
    explicit_anon = next((c for c in anon_text_candidates if c in df.columns), None)

    ambiguous_text = next((c for c in ("note_text", "text") if c in df.columns), None)

    orig_text_col = explicit_orig
    anon_text_col = explicit_anon

    if not has_rec and has_anon and not explicit_anon and ambiguous_text:
        anon_text_col = ambiguous_text
    elif explicit_orig is None and ambiguous_text:
        orig_text_col = ambiguous_text

    return orig_text_col, rec_phi_col, anon_text_col, anon_phi_col


_ID_TEXT_COLUMNS = (
    "note_text",
    "original_note_text",
    "phi_note_text",
    "text",
    "value",
    "anonymized_note_text",
    "deid_note_text",
    "anonymized_text",
    "modified_text",
)


def _ensure_table_row_id(table: pa.Table) -> pa.Table:
    """Return the table with a complete string ``row_id``, derived with ``add_row_id`` where it is missing.

    A missing ``text_hash`` is computed from the note text, as ``normalize_source_batch`` does, so the
    derived ``row_id`` is the one the pipeline would have written for the same note and patient.
    """
    if "row_id" not in table.column_names or table["row_id"].null_count > 0:
        has_hash = "text_hash" in table.column_names
        if not has_hash or table["text_hash"].null_count > 0:
            text_col = next((c for c in _ID_TEXT_COLUMNS if c in table.column_names), None)
            texts = table[text_col].to_pylist() if text_col else [None] * len(table)
            supplied = table["text_hash"].to_pylist() if has_hash else [None] * len(table)
            hashes = pa.array(
                [
                    h if h is not None else compute_text_hash(t if t is not None else "")
                    for h, t in zip(supplied, texts, strict=True)
                ],
                type=pa.string(),
            )
            if has_hash:
                table = table.set_column(table.column_names.index("text_hash"), "text_hash", hashes)
            else:
                table = table.append_column("text_hash", hashes)
        table = add_row_id(table)
    return table.set_column(table.column_names.index("row_id"), "row_id", table["row_id"].cast(pa.string()))


def _stage_pushdown_filter(schema_names: set[str], target_ids: set[str], target_hashes: set[str]) -> Any:
    """Build a scan predicate that keeps every row that could resolve to one of ``target_ids``.

    Rows with a null ``row_id`` are kept because their id is derived after the scan, and a stage with
    no ``row_id`` column is narrowed by ``text_hash`` (part of the derivation). The exact match on the
    normalized ``row_id`` is applied after the scan.
    """
    if "row_id" in schema_names:
        row_id = pads.field("row_id")
        return row_id.cast(pa.string()).isin(sorted(target_ids)) | row_id.is_null()
    if "text_hash" in schema_names and target_hashes:
        text_hash = pads.field("text_hash")
        return text_hash.isin(sorted(target_hashes)) | text_hash.is_null()
    return None


def _load_results_from_dataframe(df: pd.DataFrame, max_notes: int | None = None) -> dict[str, dict[str, Any]]:
    """Parse unified sample records from a pandas DataFrame.

    Args:
        df: DataFrame containing recognizer and/or anonymizer columns.
        max_notes: Optional maximum number of rows to load.

    Returns:
        Dictionary mapping sample IDs to sample data dictionaries.
    """
    if df.empty:
        return {}

    if max_notes and len(df) > max_notes:
        df = df.iloc[:max_notes]

    orig_text_col, rec_phi_col, anon_text_col, anon_phi_col = _resolve_text_columns(df)

    results: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        row_id = str(row["row_id"])
        th_val = str(row["text_hash"]) if "text_hash" in df.columns and pd.notna(row["text_hash"]) else None
        orig_text = str(row[orig_text_col]) if orig_text_col and pd.notna(row[orig_text_col]) else ""
        orig_phi = parse_entities(row[rec_phi_col]) if rec_phi_col else []
        anon_text = str(row[anon_text_col]) if anon_text_col and pd.notna(row[anon_text_col]) else ""
        anon_phi = parse_entities(row[anon_phi_col]) if anon_phi_col else []

        results[row_id] = {
            "sample_id": row_id,
            "row_id": row_id,
            "text_hash": th_val,
            "original_text": orig_text,
            "original_phi": orig_phi,
            "modified_text": anon_text,
            "modified_phi": anon_phi,
        }

    return results


def _scan_dataset_to_pandas(
    path: Path,
    target_cols: set[str],
    max_notes: int | None = None,
    target_ids: set[str] | None = None,
    target_hashes: set[str] | None = None,
) -> pd.DataFrame:
    """Scan a parquet file or dataset directory, one batch at a time, into a frame with a ``row_id`` column.

    Each batch gets its ``row_id`` completed (see ``_ensure_table_row_id``) before anything is counted, so
    ``max_notes`` always limits normalized rows. With ``target_ids`` only rows that resolve to one of those
    ids are kept (a stage matched to the rows chosen from another stage), and the scan ends once every
    target has been found; ``target_hashes`` lets a stage without a ``row_id`` column push the match down.
    """
    ds = pads.dataset(path)
    schema_names = set(ds.schema.names)
    cols = [c for c in schema_names if c in target_cols]
    if not cols:
        return pd.DataFrame()

    filter_expr = _stage_pushdown_filter(schema_names, target_ids, target_hashes or set()) if target_ids else None

    tables: list[pa.Table] = []
    collected = 0
    found: set[str] = set()
    for batch in ds.scanner(columns=cols, filter=filter_expr).to_batches():
        table = _ensure_table_row_id(pa.Table.from_batches([batch]))
        if target_ids:
            table = table.filter(pa.array([row_id in target_ids for row_id in table["row_id"].to_pylist()], pa.bool_()))
        if max_notes is not None:
            table = table.slice(0, max_notes - collected)
        if len(table) == 0:
            continue
        tables.append(table)
        collected += len(table)
        if max_notes is not None and collected >= max_notes:
            break
        if target_ids:
            found.update(table["row_id"].to_pylist())
            if found >= target_ids:
                break

    return pa.concat_tables(tables).to_pandas() if tables else pd.DataFrame()


def load_from_parquet_file(
    file_or_dir: str | os.PathLike[str], max_notes: int | None = None
) -> dict[str, dict[str, Any]]:
    """Load sample records from a Parquet file or directory of Parquet files.

    Args:
        file_or_dir: Path to a ``.parquet`` file or a directory containing Parquet files.
        max_notes: Optional maximum number of rows to load.

    Returns:
        Dictionary mapping sample IDs to sample data dictionaries.

    Raises:
        FileNotFoundError: If the path does not exist or directory contains no Parquet files.
    """
    path = Path(file_or_dir)
    if path.is_file():
        df = _scan_dataset_to_pandas(path, RECOGNIZER_COLUMNS | ANONYMIZER_COLUMNS, max_notes=max_notes)
    elif path.is_dir():
        parquet_files = sorted(path.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
        df = _scan_dataset_to_pandas(path, RECOGNIZER_COLUMNS | ANONYMIZER_COLUMNS, max_notes=max_notes)
    else:
        raise FileNotFoundError(f"Parquet path not found: {path}")

    return _load_results_from_dataframe(df, max_notes=max_notes)


def _find_pipeline_stage_dirs(root: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Locate recognizer, anonymizer, and transformer stage directories."""
    rec_dir = None
    for stage in ("merged_recognizer_output", "llm_recognizer_output", "recognizer_output"):
        cand = root / stage
        if cand.is_dir() and any(cand.glob("**/*.parquet")):
            rec_dir = cand
            break

    anon_dir = None
    cand_anon = root / "anonymizer_output"
    if cand_anon.is_dir() and any(cand_anon.glob("**/*.parquet")):
        anon_dir = cand_anon

    trans_dir = None
    cand_trans = root / "transformer_output"
    if cand_trans.is_dir() and any(cand_trans.glob("**/*.parquet")):
        trans_dir = cand_trans

    return rec_dir, anon_dir, trans_dir


def _load_pipeline_recognizer_stage(
    recognizer_dir: Path,
    transformer_dir: Path | None,
    max_notes: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Load recognizer parquet output and optional transformer backfill."""
    rec_df = _scan_dataset_to_pandas(recognizer_dir, RECOGNIZER_COLUMNS, max_notes=max_notes)

    text_col = next((c for c in ("note_text", "original_note_text", "text", "value") if c in rec_df.columns), None)
    rec_phi_col = next(
        (c for c in ("recognizer_results_json", "recognizer_results", "entities") if c in rec_df.columns), None
    )

    rec_results: dict[str, dict[str, Any]] = {}

    for _, row in rec_df.iterrows():
        row_id = str(row["row_id"])
        text_val = ""
        if text_col and pd.notna(row.get(text_col)) and str(row.get(text_col)).strip():
            text_val = str(row[text_col])

        phi_val = parse_entities(row.get(rec_phi_col)) if rec_phi_col else []
        th_val = str(row["text_hash"]) if "text_hash" in rec_df.columns and pd.notna(row["text_hash"]) else None

        rec_results[row_id] = {
            "sample_id": row_id,
            "row_id": row_id,
            "text_hash": th_val,
            "original_text": text_val,
            "original_phi": phi_val,
        }

    # Recover missing note_text from the transformer stage, matched to the selected recognizer rows
    missing_ids = {rid for rid, item in rec_results.items() if not item["original_text"]}
    if transformer_dir and missing_ids:
        missing_hashes = {rec_results[rid]["text_hash"] for rid in missing_ids if rec_results[rid]["text_hash"]}
        trans_df = _scan_dataset_to_pandas(
            transformer_dir, TRANSFORMER_COLUMNS, target_ids=missing_ids, target_hashes=missing_hashes
        )
        t_col = next((c for c in ("note_text", "text", "value") if c in trans_df.columns), None)
        if t_col:
            text_map = {str(r["row_id"]): str(r[t_col]) for _, r in trans_df[["row_id", t_col]].dropna().iterrows()}
            for rid in missing_ids:
                if rid in text_map:
                    rec_results[rid]["original_text"] = text_map[rid]

    return rec_results


def _load_pipeline_anonymizer_stage(
    anonymizer_dir: Path,
    rec_results: dict[str, dict[str, Any]],
    max_notes: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Load anonymizer parquet output, matched to the recognizer rows when there are any."""
    target_ids = set(rec_results) or None
    target_hashes = {item["text_hash"] for item in rec_results.values() if item["text_hash"]}
    anon_df = _scan_dataset_to_pandas(
        anonymizer_dir,
        ANONYMIZER_COLUMNS,
        max_notes=None if target_ids else max_notes,
        target_ids=target_ids,
        target_hashes=target_hashes,
    )

    text_col = next(
        (
            c
            for c in ("anonymized_note_text", "deid_note_text", "anonymized_text", "note_text", "text")
            if c in anon_df.columns
        ),
        None,
    )
    phi_col = next(
        (c for c in ("anonymizer_results_json", "items", "anonymizer_results", "operators") if c in anon_df.columns),
        None,
    )

    anon_results: dict[str, dict[str, Any]] = {}
    for _, row in anon_df.iterrows():
        row_id = str(row["row_id"])
        th_val = str(row["text_hash"]) if "text_hash" in anon_df.columns and pd.notna(row["text_hash"]) else None
        mod_text = str(row[text_col]) if text_col and pd.notna(row.get(text_col)) else ""
        mod_phi = parse_entities(row.get(phi_col)) if phi_col else []

        anon_results[row_id] = {
            "sample_id": row_id,
            "row_id": row_id,
            "text_hash": th_val,
            "modified_text": mod_text,
            "modified_phi": mod_phi,
        }

    return anon_results


def load_pipeline_output_dir(
    output_dir: str | os.PathLike[str], max_notes: int | None = None
) -> dict[str, dict[str, Any]]:
    """Load and join recognizer and anonymizer Parquet stages from a pipeline run.

    Discovers:
    - Recognizer stage: ``merged_recognizer_output/``, else ``llm_recognizer_output/``,
      else ``recognizer_output/``.
    - Anonymizer stage: ``anonymizer_output/``.
    - Transformer stage: ``transformer_output/`` (used to backfill ``note_text``
      if omitted in the recognizer output).

    Rows are aligned strictly by ``row_id``.

    Args:
        output_dir: Directory containing pipeline output stage folders.
        max_notes: Optional maximum number of rows to load.

    Returns:
        Dictionary mapping sample IDs to merged sample records.

    Raises:
        ValueError: If neither recognizer nor anonymizer stage output is found.
    """
    root = Path(output_dir)
    rec_dir, anon_dir, trans_dir = _find_pipeline_stage_dirs(root)

    if not rec_dir and not anon_dir:
        raise ValueError(f"No pipeline stage parquet files found under {root}")

    rec_results: dict[str, dict[str, Any]] = {}
    if rec_dir:
        rec_results = _load_pipeline_recognizer_stage(rec_dir, trans_dir, max_notes=max_notes)

    anon_results: dict[str, dict[str, Any]] = {}
    if anon_dir:
        anon_results = _load_pipeline_anonymizer_stage(anon_dir, rec_results, max_notes=max_notes)

    results: dict[str, dict[str, Any]] = {}
    all_keys = list(rec_results.keys())
    for k in anon_results:
        if k not in rec_results:
            all_keys.append(k)

    for k in all_keys:
        rec = rec_results.get(k, {})
        anon = anon_results.get(k, {})
        th = rec.get("text_hash") or anon.get("text_hash")
        results[k] = {
            "sample_id": k,
            "row_id": k,
            "text_hash": th,
            "original_text": rec.get("original_text", ""),
            "original_phi": rec.get("original_phi", []),
            "modified_text": anon.get("modified_text", ""),
            "modified_phi": anon.get("modified_phi", []),
        }

    return results


def _load_data_file(p: Path, max_notes: int | None = None) -> tuple[dict[str, dict[str, Any]], str]:
    """Load data from a single Parquet or JSON file."""
    if p.suffix.lower() == ".parquet":
        return load_from_parquet_file(p, max_notes=max_notes), "Single Parquet File"
    if p.suffix.lower() == ".json":
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Could not load JSON file {p}: {e}") from e

        record = _parse_json_record(data, p.stem)
        return {record["row_id"]: record}, "Single JSON File"

    raise ValueError(f"Unsupported file format '{p.suffix}'. Expected .parquet or .json")


def _load_data_dir(p: Path, max_notes: int | None = None) -> tuple[dict[str, dict[str, Any]], str]:
    """Load data from a directory by auto-detecting layout."""
    for stage in ("merged_recognizer_output", "llm_recognizer_output", "recognizer_output", "anonymizer_output"):
        if (p / stage).is_dir() and any((p / stage).glob("**/*.parquet")):
            return load_pipeline_output_dir(p, max_notes=max_notes), "Pipeline Output Directory (Parquet)"

    rec_json_dir = p / "cli_recognizer_json"
    anon_json_dir = p / "cli_anonymizer_json"
    if rec_json_dir.is_dir() or anon_json_dir.is_dir():
        rec_res = load_analyzer_results(rec_json_dir, max_notes=max_notes) if rec_json_dir.is_dir() else {}
        allowed_ids = set(rec_res.keys()) if rec_res else None
        anon_res = (
            load_anonymized_results(anon_json_dir, max_notes=max_notes, allowed_ids=allowed_ids)
            if anon_json_dir.is_dir()
            else {}
        )
        all_ids = list(rec_res.keys())
        for k in anon_res:
            if k not in rec_res:
                all_ids.append(k)
        if max_notes and len(all_ids) > max_notes:
            all_ids = all_ids[:max_notes]

        combined: dict[str, dict[str, Any]] = {}
        for sid in all_ids:
            r = rec_res.get(sid, {})
            a = anon_res.get(sid, {})
            row_id = r.get("row_id") or a.get("row_id") or sid
            combined[row_id] = {
                "sample_id": row_id,
                "row_id": row_id,
                "text_hash": r.get("text_hash") or a.get("text_hash"),
                "original_text": r.get("original_text", ""),
                "original_phi": r.get("original_phi", []),
                "modified_text": a.get("modified_text", ""),
                "modified_phi": a.get("modified_phi", []),
            }
        return combined, "JSON Directory (cli_recognizer_json / cli_anonymizer_json)"

    if any(p.glob("*.parquet")):
        return load_from_parquet_file(p, max_notes=max_notes), "Parquet Directory"

    json_files = list(p.glob("*.json"))
    if json_files:
        if max_notes and len(json_files) > max_notes:
            json_files = json_files[:max_notes]
        combined = {}
        for jf in json_files:
            with contextlib.suppress(Exception):
                with jf.open(encoding="utf-8") as f:
                    data = json.load(f)
                record = _parse_json_record(data, jf.stem)
                existing = combined.get(record["row_id"])
                if existing is None:
                    combined[record["row_id"]] = record
                    continue
                # A recognizer file and an anonymizer file for one note share an identity: merge their fields.
                for field, value in record.items():
                    if value:
                        existing[field] = value
        if combined:
            return combined, "JSON Files Directory"

    raise ValueError(f"No recognizable Parquet stages or JSON files found in {p}")


def _resolve_data_path(user_input: str) -> Path:
    """Normalize and resolve user-supplied path to prevent path traversal issues.

    Args:
        user_input: Raw string path from user input or CLI.

    Returns:
        Resolved Path object.
    """
    clean_path = os.path.normpath(user_input.strip())
    return Path(clean_path).resolve()


def load_data(path: str | os.PathLike[str], max_notes: int | None = None) -> tuple[dict[str, dict[str, Any]], str]:
    """Unified data loader with automatic format detection.

    Auto-detects:
    1. Pipeline Parquet output directory (containing stage subdirectories).
    2. JSON results directory (containing ``cli_recognizer_json/`` and/or ``cli_anonymizer_json/``).
    3. Directory containing ``*.parquet`` files directly.
    4. Directory containing ``*.json`` files directly.
    5. Single ``.parquet`` file.
    6. Single ``.json`` file.

    Args:
        path: Path to file or folder to load.
        max_notes: Optional maximum number of notes to load to conserve memory.

    Returns:
        Tuple of (results_dict, format_description_str).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If format is unsupported or contains no recognizable data.
    """
    if not path:
        return {}, "No path provided"

    p = _resolve_data_path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_file():
        return _load_data_file(p, max_notes=max_notes)

    return _load_data_dir(p, max_notes=max_notes)


def parse_cli_args(args: list[str] | None = None) -> str:
    """Extract data path from CLI arguments if provided.

    When launched via ``tide2-visualizer [path]`` (which runs
    ``streamlit run unified_interface.py -- [path]``), the CLI arguments
    appear in ``sys.argv``.

    Args:
        args: List of command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        The first positional path argument, or an empty string if none provided.
    """
    raw_args = sys.argv[1:] if args is None else args
    for arg in raw_args:
        if not arg.startswith("-"):
            return arg
    return ""


def _render_sidebar_data_location() -> None:
    """Render the sidebar path input and loader controls."""
    with st.sidebar:
        st.header("Data Location")
        data_path_input = st.text_input(
            "Path to Output Folder or File:",
            value=st.session_state.get("data_path", ""),
            help="Path to pipeline output folder, JSON results folder, or Parquet file",
            key="data_path_widget",
        )
        row_limit = st.number_input(
            "Max notes to load:",
            min_value=10,
            max_value=100000,
            value=st.session_state.get("row_limit", 2000),
            step=500,
            help="Cap the number of notes loaded to conserve memory on large pipeline runs",
            key="row_limit_widget",
        )
        load_clicked = st.button("Load Data", use_container_width=True)

        if load_clicked or st.session_state.get("needs_initial_load", False):
            st.session_state["needs_initial_load"] = False
            target_path = data_path_input.strip()
            st.session_state["row_limit"] = int(row_limit)
            if target_path:
                st.session_state["data_path"] = target_path
                try:
                    with st.spinner("Loading data..."):
                        results, format_desc = load_data(target_path, max_notes=int(row_limit))
                    st.session_state["results"] = results
                    st.session_state["format_desc"] = format_desc
                    st.session_state["load_error"] = None
                    st.session_state["current_index"] = 0
                    st.session_state["jump_input"] = 1
                except Exception as e:
                    st.session_state["load_error"] = str(e)
                    st.session_state["results"] = {}
                    st.session_state["format_desc"] = ""

        if st.session_state.get("load_error"):
            st.error(st.session_state["load_error"])
        elif st.session_state.get("format_desc"):
            st.success(f"Format: {st.session_state['format_desc']}")

        st.divider()


def _render_sidebar_navigation(results: dict[str, dict[str, Any]]) -> tuple[str, int]:
    """Render sidebar navigation controls and return (selected_sample_id, total_navigable_samples)."""
    all_sample_ids = list(results.keys())

    with st.sidebar:
        st.header("Sample Navigation")

        def on_phi_filter_change():
            st.session_state.current_index = 0
            st.session_state.jump_input = 1

        only_with_phi = st.checkbox(
            "Only notes with PHI",
            value=False,
            key="only_with_phi_checkbox",
            on_change=on_phi_filter_change,
        )

        if only_with_phi:
            active_samples = [
                sid
                for sid in all_sample_ids
                if len(results[sid].get("original_phi", [])) > 0 or len(results[sid].get("modified_phi", [])) > 0
            ]
        else:
            active_samples = all_sample_ids

        total_samples = len(active_samples)
        if total_samples == 0:
            st.warning("No notes match the 'Only notes with PHI' filter.")
            st.stop()

        st.session_state.current_index = max(0, min(st.session_state.current_index, total_samples - 1))

        def step_prev(total: int):
            st.session_state.current_index = (st.session_state.current_index - 1) % total
            st.session_state.jump_input = st.session_state.current_index + 1

        def step_next(total: int):
            st.session_state.current_index = (st.session_state.current_index + 1) % total
            st.session_state.jump_input = st.session_state.current_index + 1

        def on_jump():
            st.session_state.current_index = max(0, min(int(st.session_state.jump_input) - 1, total_samples - 1))

        def on_search(samples: list[str]):
            query = st.session_state.search_input.strip().lower()
            if query:
                for idx, sid in enumerate(samples):
                    rec = results[sid]
                    rid = str(rec.get("row_id") or "").lower()
                    th = str(rec.get("text_hash") or "").lower()
                    if query in sid.lower() or (rid and query in rid) or (th and query in th):
                        st.session_state.current_index = idx
                        st.session_state.jump_input = idx + 1
                        return

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.button("< Prev", on_click=step_prev, args=(total_samples,), use_container_width=True)
        with col2:
            st.markdown(
                f"<div style='text-align: center; padding-top: 6px;'><b>Note {st.session_state.current_index + 1} / {total_samples}</b></div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.button("Next >", on_click=step_next, args=(total_samples,), use_container_width=True)

        st.session_state["jump_input"] = st.session_state.current_index + 1
        st.number_input(
            "Jump to note #:",
            min_value=1,
            max_value=total_samples,
            key="jump_input",
            on_change=on_jump,
        )
        st.text_input(
            "Find by Row ID / Hash:",
            key="search_input",
            on_change=on_search,
            args=(active_samples,),
        )

        selected_sample = active_samples[st.session_state.current_index]
        display_id = results[selected_sample].get("row_id") or selected_sample
        st.caption(f"**Current Row ID:** `{display_id}`")
        st.divider()

    return selected_sample, total_samples


def _render_sidebar_filters(
    results: dict[str, dict[str, Any]],
    selected_sample: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render sidebar filters and return filtered original and modified entities."""
    with st.sidebar:
        st.header("PHI Filters")
        available_entity_types = sorted(
            {
                ent["entity_type"]
                for sample in results.values()
                for ent in sample.get("original_phi", []) + sample.get("modified_phi", [])
                if ent.get("entity_type")
            }
        )
        available_recognizers = sorted(
            {
                get_recognizer_name(ent)
                for sample in results.values()
                for ent in sample.get("original_phi", [])
                if get_recognizer_name(ent) != "-"
            }
        )

        selected_entity_types = ["All"]
        if available_entity_types:
            selected_entity_types = st.multiselect(
                "Entity Types:",
                options=["All", *available_entity_types],
                default=["All"],
                key="filter_entity_types",
            )
            if not selected_entity_types:
                selected_entity_types = ["All"]

        selected_recognizers = ["All"]
        if available_recognizers:
            selected_recognizers = st.multiselect(
                "Recognizers:",
                options=["All", *available_recognizers],
                default=["All"],
                key="filter_recognizers",
            )
            if not selected_recognizers:
                selected_recognizers = ["All"]

    sample_data = results[selected_sample]
    original_phi = sample_data.get("original_phi", [])
    modified_phi = sample_data.get("modified_phi", [])

    filtered_original_phi = filter_entities(original_phi, selected_recognizers, selected_entity_types)
    filtered_modified_phi = filter_entities(modified_phi, None, selected_entity_types)

    with st.sidebar:
        st.info(f"Showing {len(filtered_original_phi)} of {len(original_phi)} detected entities in active note")

    return filtered_original_phi, filtered_modified_phi


def _render_comparison_view(
    original_text: str,
    modified_text: str,
    filtered_original_phi: list[dict[str, Any]],
    filtered_modified_phi: list[dict[str, Any]],
) -> None:
    """Render side-by-side or single stage displacy view."""
    if original_text and modified_text:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Note (Detected PHI)")
            doc = convert_to_displacy_format(original_text, filtered_original_phi)
            st.html(displacy.render(doc, style="ent", manual=True, options={"colors": COLORS}))
        with col2:
            st.subheader("De-identified / Anonymized Note")
            doc = convert_to_displacy_format(modified_text, filtered_modified_phi)
            st.html(displacy.render(doc, style="ent", manual=True, options={"colors": COLORS}))
    elif original_text:
        st.subheader("Original Note (Detected PHI)")
        doc = convert_to_displacy_format(original_text, filtered_original_phi)
        st.html(displacy.render(doc, style="ent", manual=True, options={"colors": COLORS}))
    elif modified_text:
        st.subheader("De-identified / Anonymized Note")
        doc = convert_to_displacy_format(modified_text, filtered_modified_phi)
        st.html(displacy.render(doc, style="ent", manual=True, options={"colors": COLORS}))
    else:
        st.warning("No text content available for this note.")


def _render_phi_table(
    original_text: str,
    original_phi: list[dict[str, Any]],
    filtered_original_phi: list[dict[str, Any]],
) -> None:
    """Render tabular breakdown of detected PHI entities."""
    st.divider()
    st.subheader("Detected PHI in this Note")
    if filtered_original_phi:
        table_rows = []
        for ent in filtered_original_phi:
            start = ent.get("start", 0)
            end = ent.get("end", 0)
            excerpt = (
                original_text[start:end]
                if (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(original_text))
                else ent.get("text", "")
            )
            rec_name = get_recognizer_name(ent)
            score = ent.get("score")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
            table_rows.append(
                {
                    "Entity Type": ent.get("entity_type", ""),
                    "Found Text": excerpt,
                    "Range": f"{start}-{end}",
                    "Recognizer": rec_name,
                    "Score": score_str,
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    elif not original_phi:
        st.info("No PHI detected in this note.")
    else:
        st.info("No PHI matching the selected filters.")


def main() -> None:
    """Streamlit main entrypoint for the PHI visualizer."""
    st.set_page_config(page_title="TIDE 2.0 - PHI Visualizer", layout="wide")

    st.title("TIDE 2.0 - PHI Visualizer")
    st.markdown("**Inspect clinical notes before and after de-identification side by side**")
    st.divider()

    if "current_index" not in st.session_state:
        st.session_state["current_index"] = 0
    if "jump_input" not in st.session_state:
        st.session_state["jump_input"] = 1

    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True
        cli_path = parse_cli_args()
        initial_path = cli_path or "./data/output"
        st.session_state["data_path"] = initial_path
        if initial_path and Path(initial_path).exists():
            st.session_state["needs_initial_load"] = True

    _render_sidebar_data_location()

    results: dict[str, dict[str, Any]] = st.session_state.get("results", {})
    if not results:
        st.info("Please enter a valid path to pipeline output or JSON results in the sidebar and click **Load Data**.")
        return

    selected_sample, total_samples = _render_sidebar_navigation(results)
    filtered_orig_phi, filtered_mod_phi = _render_sidebar_filters(results, selected_sample)

    sample_data = results[selected_sample]
    orig_text = sample_data.get("original_text", "")
    orig_phi = sample_data.get("original_phi", [])
    mod_text = sample_data.get("modified_text", "")
    display_id = sample_data.get("row_id") or selected_sample

    st.caption(f"Viewing note **{st.session_state.current_index + 1} of {total_samples}** | Row ID: `{display_id}`")

    _render_comparison_view(orig_text, mod_text, filtered_orig_phi, filtered_mod_phi)
    if orig_text or orig_phi:
        _render_phi_table(orig_text, orig_phi, filtered_orig_phi)

    st.divider()
    st.subheader("Entity Types Legend")
    all_entity_types = collect_entity_types(filtered_orig_phi, filtered_mod_phi)
    legend = build_legend_html(all_entity_types)
    if legend:
        st.html(legend)
    else:
        st.caption("No entities present.")


if __name__ == "__main__":
    main()
