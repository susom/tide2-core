#!/usr/bin/env python3
"""
TIDE 2.0 - PHI Inspection Visualizer

An interactive interface for:
- Inspecting clinical notes before and after de-identification side by side
- Visualizing detected PHI entities and anonymization replacements
- Filtering by entity type and recognizer engine
- Tabular breakdown of detected PHI metadata
- Auto-detecting and loading pipeline Parquet outputs

Usage:
    tide2-visualizer [path]
    # or
    streamlit run unified_interface.py [-- path]
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import streamlit as st
from spacy import displacy

from tide2.runner.local_runner import add_row_id
from tide2.utils.nulls import is_null
from tide2.utils.stage_status import FAILED
from tide2.utils.stage_status import load_stage_status
from tide2.utils.stage_status import merge_stage_status
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
    "recognizer_results_json",
    "processing_status",
    "stage_status_json",
}

ANONYMIZER_COLUMNS = {
    "row_id",
    "text_hash",
    "patient_id",
    "anonymized_note_text",
    "anonymizer_results_json",
    "processing_status",
    "stage_status_json",
}

TRANSFORMER_COLUMNS = {
    "row_id",
    "text_hash",
    "patient_id",
    "note_text",
}


def parse_entities(raw: Any) -> list[dict[str, Any]]:
    """Parse entities from raw JSON string, list, numpy array, or dict format.

    Args:
        raw: Raw entity data, such as a JSON string, a list of dicts, or None.

    Returns:
        List of entity dictionaries.
    """
    if is_null(raw):
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


def _ensure_table_row_id(table: pa.Table) -> pa.Table:
    """Return the table with a complete string ``row_id``, derived with ``add_row_id`` where it is missing."""
    if "row_id" not in table.column_names or table["row_id"].null_count > 0:
        has_hash = "text_hash" in table.column_names
        if not has_hash or table["text_hash"].null_count > 0:
            text_col = next((c for c in ("note_text", "anonymized_note_text") if c in table.column_names), None)
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
    """Build a scan predicate that keeps every row that could resolve to one of ``target_ids``."""
    if "row_id" in schema_names:
        row_id = pads.field("row_id")
        return row_id.cast(pa.string()).isin(sorted(target_ids)) | row_id.is_null()
    if "text_hash" in schema_names and target_hashes:
        text_hash = pads.field("text_hash")
        return text_hash.isin(sorted(target_hashes)) | text_hash.is_null()
    return None


def _scan_dataset_to_table(
    path: Path,
    target_cols: set[str],
    max_notes: int | None = None,
    target_ids: set[str] | None = None,
    target_hashes: set[str] | None = None,
) -> pa.Table | None:
    """Scan a Parquet file or dataset directory into a PyArrow Table with a guaranteed ``row_id`` column."""
    ds = pads.dataset(path)
    schema_names = set(ds.schema.names)
    cols = [c for c in schema_names if c in target_cols]
    if not cols:
        return None

    filter_expr = _stage_pushdown_filter(schema_names, target_ids, target_hashes or set()) if target_ids else None
    batch_size = min(max_notes, 512) if max_notes is not None and not target_ids else 512

    tables: list[pa.Table] = []
    collected = 0
    found: set[str] = set()
    scanner = ds.scanner(
        columns=cols,
        filter=filter_expr,
        batch_size=batch_size,
        batch_readahead=1,
        fragment_readahead=1,
    )
    for batch in scanner.to_batches():
        cur_batch = batch.slice(0, max_notes - collected) if max_notes is not None and not target_ids else batch
        table = _ensure_table_row_id(pa.Table.from_batches([cur_batch]))
        if target_ids:
            table = table.filter(pa.array([row_id in target_ids for row_id in table["row_id"].to_pylist()], pa.bool_()))
        if max_notes is not None:
            table = table.slice(0, max_notes - collected)
        if len(table) == 0:
            continue
        tables.append(pa.Table.from_pydict(table.to_pydict(), schema=table.schema))
        collected += len(table)
        if max_notes is not None and collected >= max_notes:
            break
        if target_ids:
            found.update(table["row_id"].to_pylist())
            if found >= target_ids:
                break

    return pa.concat_tables(tables, promote_options="default") if tables else None


def _load_results_from_table(table: pa.Table | None, max_notes: int | None = None) -> dict[str, dict[str, Any]]:
    """Parse sample records from a PyArrow Table keyed by ``row_id``."""
    if table is None or len(table) == 0:
        return {}

    if max_notes and len(table) > max_notes:
        table = table.slice(0, max_notes)

    results: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        row_id = str(row["row_id"])
        text_hash = row.get("text_hash")
        note_text = row.get("note_text")
        anon_text = row.get("anonymized_note_text")
        proc_status = row.get("processing_status")
        stage_status = row.get("stage_status_json")
        results[row_id] = {
            "sample_id": row_id,
            "row_id": row_id,
            "text_hash": str(text_hash) if not is_null(text_hash) else None,
            "original_text": str(note_text) if not is_null(note_text) else "",
            "original_phi": parse_entities(row.get("recognizer_results_json")),
            "modified_text": str(anon_text) if not is_null(anon_text) else "",
            "modified_phi": parse_entities(row.get("anonymizer_results_json")),
            "processing_status": str(proc_status) if not is_null(proc_status) else None,
            "stage_status_json": str(stage_status) if not is_null(stage_status) else None,
        }

    return results


def load_from_parquet_file(
    file_or_dir: str | os.PathLike[str], max_notes: int | None = None
) -> dict[str, dict[str, Any]]:
    """Load sample records from a Parquet file or directory of Parquet files."""
    path = Path(file_or_dir)
    if path.is_file():
        table = _scan_dataset_to_table(path, RECOGNIZER_COLUMNS | ANONYMIZER_COLUMNS, max_notes=max_notes)
    elif path.is_dir():
        if not any(path.glob("*.parquet")):
            raise FileNotFoundError(f"No parquet files found in {path}")
        table = _scan_dataset_to_table(path, RECOGNIZER_COLUMNS | ANONYMIZER_COLUMNS, max_notes=max_notes)
    else:
        raise FileNotFoundError(f"Parquet path not found: {path}")

    return _load_results_from_table(table, max_notes=max_notes)


def _find_pipeline_stage_dirs(root: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Locate recognizer, anonymizer, and transformer stage directories."""
    rec_dir = next(
        (
            cand
            for stage in ("merged_recognizer_output", "llm_recognizer_output", "recognizer_output")
            if (cand := root / stage).is_dir() and any(cand.glob("*.parquet"))
        ),
        None,
    )
    anon_cand = root / "anonymizer_output"
    anon_dir = anon_cand if anon_cand.is_dir() and any(anon_cand.glob("*.parquet")) else None
    trans_cand = root / "transformer_output"
    trans_dir = trans_cand if trans_cand.is_dir() and any(trans_cand.glob("*.parquet")) else None
    return rec_dir, anon_dir, trans_dir


def _load_pipeline_recognizer_stage(
    recognizer_dir: Path,
    transformer_dir: Path | None,
    max_notes: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Load recognizer Parquet output and optional transformer backfill keyed by ``row_id``."""
    rec_table = _scan_dataset_to_table(recognizer_dir, RECOGNIZER_COLUMNS, max_notes=max_notes)
    rec_results = _load_results_from_table(rec_table, max_notes=max_notes)

    missing_ids = {rid for rid, item in rec_results.items() if not item["original_text"].strip()}
    if transformer_dir and missing_ids:
        missing_hashes = {rec_results[rid]["text_hash"] for rid in missing_ids if rec_results[rid]["text_hash"]}
        trans_table = _scan_dataset_to_table(
            transformer_dir, TRANSFORMER_COLUMNS, target_ids=missing_ids, target_hashes=missing_hashes
        )
        if trans_table is not None and "note_text" in trans_table.column_names:
            for row in trans_table.to_pylist():
                rid = str(row["row_id"])
                note_text = row.get("note_text")
                if rid in missing_ids and not is_null(note_text):
                    rec_results[rid]["original_text"] = str(note_text)

    return rec_results


def _load_pipeline_anonymizer_stage(
    anonymizer_dir: Path,
    rec_results: dict[str, dict[str, Any]],
    max_notes: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Load anonymizer Parquet output, matched by ``row_id`` to ``rec_results`` when present."""
    target_ids = set(rec_results) or None
    target_hashes = {item["text_hash"] for item in rec_results.values() if item["text_hash"]}
    anon_table = _scan_dataset_to_table(
        anonymizer_dir,
        ANONYMIZER_COLUMNS,
        max_notes=None if target_ids else max_notes,
        target_ids=target_ids,
        target_hashes=target_hashes,
    )
    return _load_results_from_table(anon_table)


def load_pipeline_output_dir(
    output_dir: str | os.PathLike[str], max_notes: int | None = None
) -> dict[str, dict[str, Any]]:
    """Load and join recognizer and anonymizer Parquet stages from a pipeline run by ``row_id``."""
    root = Path(output_dir)
    rec_dir, anon_dir, trans_dir = _find_pipeline_stage_dirs(root)

    if not rec_dir and not anon_dir:
        raise ValueError(f"No pipeline stage parquet files found under {root}")

    rec_results = _load_pipeline_recognizer_stage(rec_dir, trans_dir, max_notes=max_notes) if rec_dir else {}
    anon_results = _load_pipeline_anonymizer_stage(anon_dir, rec_results, max_notes=max_notes) if anon_dir else {}

    all_ids = list(rec_results) + [rid for rid in anon_results if rid not in rec_results]
    results: dict[str, dict[str, Any]] = {}
    for rid in all_ids:
        rec = rec_results.get(rid, {})
        anon = anon_results.get(rid, {})
        rec_status = rec.get("processing_status")
        anon_status = anon.get("processing_status")
        rec_stage_json = rec.get("stage_status_json")
        anon_stage_json = anon.get("stage_status_json")
        if rec_status or anon_status or rec_stage_json or anon_stage_json:
            merged_stage_json, merged_status = merge_stage_status(
                rec_stage_json, anon_stage_json, rec_status, anon_status
            )
        else:
            merged_stage_json, merged_status = None, None
        results[rid] = {
            "sample_id": rid,
            "row_id": rid,
            "text_hash": rec.get("text_hash") or anon.get("text_hash"),
            "original_text": rec.get("original_text", ""),
            "original_phi": rec.get("original_phi", []),
            "modified_text": anon.get("modified_text", ""),
            "modified_phi": anon.get("modified_phi", []),
            "processing_status": merged_status,
            "stage_status_json": merged_stage_json,
        }

    return results


def _resolve_data_path(user_input: str) -> Path:
    """Normalize and resolve user-supplied path."""
    return Path(os.path.normpath(user_input.strip())).resolve()


def load_data(path: str | os.PathLike[str], max_notes: int | None = None) -> tuple[dict[str, dict[str, Any]], str]:
    """Load Parquet pipeline stages, a Parquet directory, or a single Parquet file."""
    if not path:
        return {}, "No path provided"

    p = _resolve_data_path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_file():
        if p.suffix.lower() != ".parquet":
            raise ValueError(f"Unsupported file format '{p.suffix}'. Expected .parquet")
        return load_from_parquet_file(p, max_notes=max_notes), "Single Parquet File"

    rec_dir, anon_dir, _ = _find_pipeline_stage_dirs(p)
    if rec_dir or anon_dir:
        return load_pipeline_output_dir(p, max_notes=max_notes), "Pipeline Output Directory (Parquet)"

    if any(p.glob("*.parquet")):
        return load_from_parquet_file(p, max_notes=max_notes), "Parquet Directory"

    raise ValueError(f"No recognizable Parquet stages or Parquet files found in {p}")


def parse_cli_args(args: list[str] | None = None) -> str:
    """Extract data path from CLI arguments if provided."""
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
            help="Path to pipeline output folder or Parquet file",
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


def _is_recognition_unavailable(
    original_phi: list[dict[str, Any]],
    processing_status: str | None,
    stage_status_json: str | None,
) -> bool:
    """Return True when recognition failed and produced no detections."""
    if original_phi or processing_status != FAILED:
        return False
    stages = load_stage_status(stage_status_json)
    if not stages:
        return True
    return any(
        isinstance(stages.get(stage), dict) and stages[stage].get("status") == FAILED
        for stage in ("transformer", "bio_aggregation", "recognizer", "llm_recognizer")
    )


def _render_phi_table(
    original_text: str,
    original_phi: list[dict[str, Any]],
    filtered_original_phi: list[dict[str, Any]],
    processing_status: str | None = None,
    stage_status_json: str | None = None,
) -> None:
    """Render tabular breakdown of detected PHI entities."""
    st.divider()
    st.subheader("Detected PHI in this Note")
    if processing_status == FAILED:
        detail = f" ({stage_status_json})" if stage_status_json else ""
        st.warning(f"Processing failed for this note{detail}.")

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
        st.dataframe(pa.Table.from_pylist(table_rows), use_container_width=True)
    elif _is_recognition_unavailable(original_phi, processing_status, stage_status_json):
        st.warning("PHI detection results are unavailable for this note.")
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
        st.info("Please enter a valid path to pipeline output or Parquet file in the sidebar and click **Load Data**.")
        return

    selected_sample, total_samples = _render_sidebar_navigation(results)
    filtered_orig_phi, filtered_mod_phi = _render_sidebar_filters(results, selected_sample)

    sample_data = results[selected_sample]
    orig_text = sample_data.get("original_text", "")
    orig_phi = sample_data.get("original_phi", [])
    mod_text = sample_data.get("modified_text", "")
    proc_status = sample_data.get("processing_status")
    stage_status_json = sample_data.get("stage_status_json")
    display_id = sample_data.get("row_id") or selected_sample

    st.caption(f"Viewing note **{st.session_state.current_index + 1} of {total_samples}** | Row ID: `{display_id}`")

    _render_comparison_view(orig_text, mod_text, filtered_orig_phi, filtered_mod_phi)
    if orig_text or orig_phi or proc_status == FAILED:
        _render_phi_table(orig_text, orig_phi, filtered_orig_phi, proc_status, stage_status_json)

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
