#!/usr/bin/env python3
"""
TIDE 2.0 - Simplified PHI Visualizer

A minimal interface for:
- Selecting samples from analyzer/recognizer results
- Filtering by specific recognizers
- Side-by-side comparison of original vs anonymized text with highlighted entities
- Loading data from JSON files, DataFrames, or BigQuery tables

Usage:
    streamlit run unified_interface.py
"""

import glob
import json
import os
from enum import StrEnum

import pandas as pd
import streamlit as st
from spacy import displacy


class VisualizationMode(StrEnum):
    """Visualization mode options."""

    BOTH = "Both (Recognizer + Anonymizer)"
    RECOGNIZER_ONLY = "Recognizer Only"
    ANONYMIZER_ONLY = "Anonymizer Only"


class DataSource(StrEnum):
    """Data source options."""

    JSON_FILES = "JSON Files"
    BIGQUERY = "BigQuery Table"
    DATAFRAME = "DataFrame (Parquet/CSV)"


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
}

TEXT_HASH_COLUMN = "text_hash"


def load_analyzer_results(analyzer_dir: str) -> dict:
    """Load analyzer results from a directory of JSON files.

    Each JSON file is expected to have ``key`` (sample ID), ``value`` (original
    text), and ``recognizer_results`` (list of entity dicts) fields.

    Args:
        analyzer_dir: Path to directory containing analyzer result JSON files.

    Returns:
        Dictionary mapping sample IDs to dicts with ``original_text`` and
        ``original_phi`` keys.
    """
    if not os.path.isdir(analyzer_dir):
        return {}

    file_names = glob.glob(os.path.join(analyzer_dir, "*.json"))
    results = {}

    for file_name in file_names:
        try:
            with open(file_name) as file:
                data = json.load(file)
                results[data["key"]] = {
                    "original_text": data["value"],
                    "original_phi": data["recognizer_results"],
                }
        except Exception:
            continue

    return results


def load_anonymized_results(anonymizer_dir: str) -> dict:
    """Load anonymized results from a directory of JSON files.

    Each JSON file is expected to have ``text`` (anonymized text) and ``items``
    (list of anonymization item dicts) fields. The sample ID is derived from
    the filename (stem before the first dot).

    Args:
        anonymizer_dir: Path to directory containing anonymized result JSON files.

    Returns:
        Dictionary mapping sample IDs to dicts with ``modified_text`` and
        ``modified_phi`` keys.
    """
    if not os.path.isdir(anonymizer_dir):
        return {}

    file_names = glob.glob(os.path.join(anonymizer_dir, "*.json"))
    results = {}

    for file_name in file_names:
        try:
            id = os.path.basename(file_name).split(".")[0]
            with open(file_name) as file:
                data = json.load(file)
                results[id] = {
                    "modified_text": data["text"],
                    "modified_phi": data["items"],
                }
        except Exception:
            continue

    return results


def load_recognizer_results_from_dataframe(df: pd.DataFrame) -> dict:
    """Load recognizer results from a DataFrame.

    Expected columns:
    - text_hash: unique identifier (SHA-256 hex digest of the original text)
    - text: original text
    - recognizer_results: JSON string or dict with entity results
    """
    results = {}

    if TEXT_HASH_COLUMN not in df.columns:
        raise ValueError(f"DataFrame must have a '{TEXT_HASH_COLUMN}' column")

    # Determine text column
    text_col = None
    for col in ["text", "value", "note_text", "phi_note_text", "original_note_text"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            "DataFrame must have a text column (text, value, note_text, phi_note_text, or original_note_text)"
        )

    # Determine recognizer results column
    recognizer_col = None
    for col in ["recognizer_results", "recognizer_results_json", "entities"]:
        if col in df.columns:
            recognizer_col = col
            break

    if recognizer_col is None:
        raise ValueError("DataFrame must have a recognizer_results column")

    for _, row in df.iterrows():
        try:
            sample_id = str(row[TEXT_HASH_COLUMN])
            text = row[text_col] if pd.notna(row[text_col]) else ""

            # Parse recognizer results
            recognizer_data = row[recognizer_col]
            if pd.isna(recognizer_data):
                entities = []
            elif isinstance(recognizer_data, str):
                entities = json.loads(recognizer_data)
            elif isinstance(recognizer_data, (list, dict)):
                entities = recognizer_data if isinstance(recognizer_data, list) else [recognizer_data]
            else:
                entities = []

            results[sample_id] = {
                "original_text": text,
                "original_phi": entities,
            }
        except Exception:
            continue

    return results


def load_anonymizer_results_from_dataframe(df: pd.DataFrame) -> dict:
    """Load anonymizer results from a DataFrame.

    Expected columns:
    - text_hash: unique identifier (SHA-256 hex digest of the original text)
    - deid_note_text or anonymized_text: anonymized text
    - items or anonymizer_results: JSON string or dict with anonymization results
    """
    results = {}

    if TEXT_HASH_COLUMN not in df.columns:
        raise ValueError(f"DataFrame must have a '{TEXT_HASH_COLUMN}' column")

    # Determine text column
    text_col = None
    for col in ["anonymized_note_text", "deid_note_text", "anonymized_text", "text", "modified_text"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            "DataFrame must have an anonymized text column "
            "(deid_note_text, anonymized_text, anonymized_note_text, text, or modified_text)"
        )

    # Determine items column (optional for anonymizer-only mode)
    items_col = None
    for col in ["items", "anonymizer_results", "anonymizer_results_json", "operators"]:
        if col in df.columns:
            items_col = col
            break

    for _, row in df.iterrows():
        try:
            sample_id = str(row[TEXT_HASH_COLUMN])
            text = row[text_col] if pd.notna(row[text_col]) else ""

            # Parse items if available
            items = []
            if items_col and pd.notna(row.get(items_col)):
                items_data = row[items_col]
                if isinstance(items_data, str):
                    items = json.loads(items_data)
                elif isinstance(items_data, list):
                    items = items_data

            results[sample_id] = {
                "modified_text": text,
                "modified_phi": items,
            }
        except Exception:
            continue

    return results


def get_table_row_count(table_id: str, only_with_entities: bool = False, entities_column: str = "entity_count") -> int:
    """Get the total row count for a BigQuery table.

    Args:
        table_id: Full table ID in format 'project.dataset.table'
        only_with_entities: If True, count only records with at least one entity
        entities_column: Column name to check for entity count

    Returns:
        Total number of rows in the table
    """
    try:
        from google.cloud import bigquery

        client = bigquery.Client()

        # Validate table ID
        import re

        parts = table_id.strip().split(".")
        if len(parts) != 3:
            raise ValueError("Invalid table ID format")
        if not all(re.match(r"^[A-Za-z0-9_-]+$", part) for part in parts):
            raise ValueError("Invalid characters in table ID")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", entities_column):
            raise ValueError("Invalid entities column name")

        if only_with_entities:
            query = f"SELECT COUNT(*) as cnt FROM `{table_id}` WHERE {entities_column} > 0"  # noqa: S608
        else:
            query = f"SELECT COUNT(*) as cnt FROM `{table_id}`"  # noqa: S608

        result = client.query(query).result()
        for row in result:
            return row.cnt
        return 0
    except Exception:
        return 0


def load_from_bigquery(
    table_id: str,
    limit: int = 1000,
    only_with_entities: bool = False,
    entities_column: str = "entity_count",
    total_rows: int | None = None,
    filter_column: str | None = None,
    filter_values: list[str] | None = None,
) -> pd.DataFrame:
    """Load data from a BigQuery table using random sampling or ID filtering.

    When *filter_column* and *filter_values* are provided the query fetches
    only the rows whose *filter_column* value is in *filter_values* (no
    random sampling).  Otherwise the query uses ``TABLESAMPLE SYSTEM`` for
    random sampling as before.

    Args:
        table_id: Full table ID in format 'project.dataset.table'
        limit: Maximum number of rows to load
        only_with_entities: If True, only load records with at least one entity
        entities_column: Column name to check for entity count (default: entity_count)
        total_rows: Total rows in table (used to calculate sample percentage)
        filter_column: Column name to filter on (e.g. an ID column shared with
            another table).
        filter_values: Values to keep for *filter_column*.

    Returns:
        DataFrame with the query results
    """
    try:
        from google.cloud import bigquery

        client = bigquery.Client()
        # Validate table ID format to prevent SQL injection
        parts = table_id.strip().split(".")
        if len(parts) != 3:
            raise ValueError("Invalid table ID format. Expected 'project.dataset.table'")

        # Use parameterized query is not possible for table names, so validate strictly
        import re

        if not all(re.match(r"^[A-Za-z0-9_-]+$", part) for part in parts):
            raise ValueError("Invalid characters in table ID")

        # Validate entities_column name
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", entities_column):
            raise ValueError("Invalid entities column name")

        where_clauses: list[str] = []
        job_config = None

        if filter_column and filter_values:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", filter_column):
                raise ValueError("Invalid filter column name")
            where_clauses.append(f"CAST({filter_column} AS STRING) IN UNNEST(@filter_values)")
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("filter_values", "STRING", filter_values),
                ]
            )
            sample_clause = ""
        else:
            sample_clause = ""
            if total_rows and total_rows > 0:
                target_percentage = (limit * 2 / total_rows) * 100
                sample_percentage = max(0.1, min(100.0, target_percentage))
                if sample_percentage < 100.0:
                    sample_clause = f" TABLESAMPLE SYSTEM ({sample_percentage:.2f} PERCENT)"

        if only_with_entities:
            where_clauses.append(f"{entities_column} > 0")

        where_clause = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # When filtering by a column, deduplicate so that LIMIT counts unique
        # values rather than being consumed by duplicates.
        qualify_clause = ""
        if filter_column and filter_values:
            qualify_clause = f" QUALIFY ROW_NUMBER() OVER (PARTITION BY {filter_column}) = 1"

        query = f"SELECT * FROM `{table_id}`{sample_clause}{where_clause}{qualify_clause} LIMIT {int(limit)}"  # noqa: S608

        df = client.query(query, job_config=job_config).to_dataframe()
        return df
    except ImportError:
        raise ImportError("google-cloud-bigquery is required. Install with: pip install google-cloud-bigquery")


def load_dataframe_from_file(file_path: str) -> pd.DataFrame:
    """Load a pandas DataFrame from a Parquet or CSV file.

    Args:
        file_path: Path to the file. Must end with ``.parquet`` or ``.csv``.

    Returns:
        DataFrame loaded from the file.

    Raises:
        ValueError: If the file extension is not ``.parquet`` or ``.csv``.
    """
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    raise ValueError("Unsupported file format. Use .parquet or .csv")


def get_recognizers_from_entities(entities: list[dict]) -> list[str]:
    """Extract unique recognizer names from a list of entity dicts.

    Looks for the ``recognizer_name`` field inside each entity's
    ``recognition_metadata`` dict.

    Args:
        entities: List of entity dicts, each optionally containing a
            ``recognition_metadata`` dict with a ``recognizer_name`` key.

    Returns:
        Sorted list of unique recognizer name strings.
    """
    recognizers = set()
    for entity in entities:
        if "recognition_metadata" in entity and "recognizer_name" in entity["recognition_metadata"]:
            recognizers.add(entity["recognition_metadata"]["recognizer_name"])
    return sorted(list(recognizers))


def filter_entities_by_recognizers(entities: list[dict], selected_recognizers: list[str]) -> list[dict]:
    """Filter entities to include only those from the specified recognizers.

    If ``selected_recognizers`` is empty or contains ``"All"``, all entities
    are returned unfiltered.

    Args:
        entities: List of entity dicts, each containing ``recognition_metadata``
            with a ``recognizer_name`` key.
        selected_recognizers: List of recognizer names to keep. Pass ``["All"]``
            or an empty list to skip filtering.

    Returns:
        Filtered list of entity dicts whose recognizer name is in
        ``selected_recognizers``.
    """
    if not selected_recognizers or "All" in selected_recognizers:
        return entities

    filtered = []
    for entity in entities:
        if (
            "recognition_metadata" in entity
            and "recognizer_name" in entity["recognition_metadata"]
            and entity["recognition_metadata"]["recognizer_name"] in selected_recognizers
        ):
            filtered.append(entity)

    return filtered


def convert_to_displacy_format(text: str, entities: list[dict]) -> dict:
    """Convert text and entity dicts into spaCy displacy manual rendering format.

    Args:
        text: The source text containing the entities.
        entities: List of entity dicts, each with ``start``, ``end``, and
            ``entity_type`` keys.

    Returns:
        Dictionary with ``text`` and ``ents`` keys suitable for
        ``displacy.render(..., manual=True)``.
    """
    return {
        "text": text,
        "ents": [{"start": ent["start"], "end": ent["end"], "label": ent["entity_type"]} for ent in entities],
    }


def validate_json_paths(
    viz_mode: VisualizationMode,
    analyzer_dir: str | None,
    anonymizer_dir: str | None,
) -> tuple[dict, dict, str | None]:
    """Validate JSON directory paths and load data if valid.

    Args:
        viz_mode: Current visualization mode.
        analyzer_dir: Path to analyzer results directory (may be None or empty).
        anonymizer_dir: Path to anonymizer results directory (may be None or empty).

    Returns:
        Tuple of (analyzer_results, anonymized_results, error_message).
        If error_message is not None, results should be ignored and the error
        displayed to the user. Error messages starting with "info:" are
        informational; others are errors.
    """
    if viz_mode == VisualizationMode.RECOGNIZER_ONLY:
        if not analyzer_dir:
            return {}, {}, "info:Please specify the recognizer results folder in the sidebar"
        if not os.path.exists(analyzer_dir):
            return {}, {}, "Recognizer folder does not exist"
        return load_analyzer_results(analyzer_dir), {}, None

    if viz_mode == VisualizationMode.ANONYMIZER_ONLY:
        if not anonymizer_dir:
            return {}, {}, "info:Please specify the anonymizer results folder in the sidebar"
        if not os.path.exists(anonymizer_dir):
            return {}, {}, "Anonymizer folder does not exist"
        return {}, load_anonymized_results(anonymizer_dir), None

    # BOTH
    if not analyzer_dir or not anonymizer_dir:
        return {}, {}, "info:Please specify both analyzer and anonymizer folders in the sidebar"
    if not os.path.exists(analyzer_dir) or not os.path.exists(anonymizer_dir):
        return {}, {}, "One or both folders do not exist"
    return load_analyzer_results(analyzer_dir), load_anonymized_results(anonymizer_dir), None


def validate_dataframe_paths(
    viz_mode: VisualizationMode,
    recognizer_file: str | None,
    anonymizer_file: str | None,
) -> tuple[dict, dict, str | None]:
    """Validate DataFrame file paths and load data if valid.

    Args:
        viz_mode: Current visualization mode.
        recognizer_file: Path to recognizer results file (parquet/csv).
        anonymizer_file: Path to anonymizer results file (parquet/csv).

    Returns:
        Tuple of (analyzer_results, anonymized_results, error_message).
        If error_message is not None, results should be ignored.
    """
    analyzer_results = {}
    anonymized_results = {}

    if viz_mode in [VisualizationMode.BOTH, VisualizationMode.RECOGNIZER_ONLY] and recognizer_file:
        if os.path.exists(recognizer_file):
            df = load_dataframe_from_file(recognizer_file)
            analyzer_results = load_recognizer_results_from_dataframe(df)
        else:
            return {}, {}, f"File not found: {recognizer_file}"

    if viz_mode in [VisualizationMode.BOTH, VisualizationMode.ANONYMIZER_ONLY] and anonymizer_file:
        if os.path.exists(anonymizer_file):
            df = load_dataframe_from_file(anonymizer_file)
            anonymized_results = load_anonymizer_results_from_dataframe(df)
        else:
            return {}, {}, f"File not found: {anonymizer_file}"

    return analyzer_results, anonymized_results, None


def clamp_sample_index(index: int, num_samples: int) -> int:
    """Clamp a sample index to valid bounds.

    Args:
        index: Current sample index (may be out of bounds).
        num_samples: Total number of samples (must be > 0).

    Returns:
        Clamped index in range [0, num_samples - 1].
    """
    if index >= num_samples:
        return 0
    if index < 0:
        return num_samples - 1
    return index


def get_sample_data(
    viz_mode: VisualizationMode,
    analyzer_results: dict,
    anonymized_results: dict,
    sample_id: str,
) -> tuple[str, list, str, list]:
    """Extract sample data for the given sample ID based on visualization mode.

    Args:
        viz_mode: Current visualization mode.
        analyzer_results: Dict mapping sample IDs to analyzer result dicts.
        anonymized_results: Dict mapping sample IDs to anonymized result dicts.
        sample_id: The sample ID to look up.

    Returns:
        Tuple of (original_text, original_phi, anonymized_text, anonymized_phi).
    """
    original_text = ""
    original_phi = []
    anonymized_text = ""
    anonymized_phi = []

    if viz_mode in [VisualizationMode.BOTH, VisualizationMode.RECOGNIZER_ONLY]:
        current_sample = analyzer_results.get(sample_id, {})
        original_text = current_sample.get("original_text", "")
        original_phi = current_sample.get("original_phi", [])

    if viz_mode in [VisualizationMode.BOTH, VisualizationMode.ANONYMIZER_ONLY]:
        anonymized_data = anonymized_results.get(sample_id, {})
        anonymized_text = anonymized_data.get("modified_text", "No anonymized text available")
        anonymized_phi = anonymized_data.get("modified_phi", [])

    return original_text, original_phi, anonymized_text, anonymized_phi


def collect_entity_types(
    viz_mode: VisualizationMode,
    filtered_original_phi: list[dict],
    anonymized_phi: list[dict],
) -> set[str]:
    """Collect unique entity types from displayed entities based on visualization mode.

    Args:
        viz_mode: Current visualization mode.
        filtered_original_phi: Filtered original entities (recognizer side).
        anonymized_phi: Anonymized entities.

    Returns:
        Set of entity type strings.
    """
    entity_types: set[str] = set()
    if viz_mode in [VisualizationMode.BOTH, VisualizationMode.RECOGNIZER_ONLY]:
        for ent in filtered_original_phi:
            entity_types.add(ent["entity_type"])
    if viz_mode in [VisualizationMode.BOTH, VisualizationMode.ANONYMIZER_ONLY]:
        for ent in anonymized_phi:
            entity_types.add(ent["entity_type"])
    return entity_types


def build_legend_html(entity_types: set[str]) -> str:
    """Build HTML legend for entity type color boxes.

    Args:
        entity_types: Set of entity type strings to include in the legend.

    Returns:
        HTML string with colored boxes and labels, or empty string if no types.
    """
    if not entity_types:
        return ""
    legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
    for entity_type in sorted(entity_types):
        color = COLORS.get(entity_type, "#CCCCCC")
        legend_html += f"""
            <div style='display: flex; align-items: center; margin: 5px;'>
                <div style='width: 15px; height: 15px; background-color: {color};
                           border: 1px solid #ccc; margin-right: 5px;'></div>
                <span style='font-size: 12px;'>{entity_type}</span>
            </div>
            """
    legend_html += "</div>"
    return legend_html


def main():
    """Streamlit main entrypoint for the PHI visualizer."""
    st.set_page_config(page_title="TIDE 2.0 - Simple PHI Visualizer", layout="wide")

    st.title("TIDE 2.0 - PHI Visualizer")
    st.markdown("**Compare original and anonymized text with recognizer filtering**")
    st.divider()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Visualization mode selection
        viz_mode = st.radio(
            "Visualization Mode:",
            options=[mode.value for mode in VisualizationMode],
            index=0,
            help="Choose what to visualize",
        )
        viz_mode = VisualizationMode(viz_mode)

        st.divider()

        # Data source selection
        data_source = st.radio(
            "Data Source:",
            options=[source.value for source in DataSource],
            index=0,
            help="Choose how to load data",
        )
        data_source = DataSource(data_source)

        st.divider()

    # Initialize results
    analyzer_results = {}
    anonymized_results = {}

    # Load data based on source type
    if data_source == DataSource.JSON_FILES:
        with st.sidebar:
            st.subheader("JSON Configuration")

            if viz_mode in [VisualizationMode.BOTH, VisualizationMode.RECOGNIZER_ONLY]:
                analyzer_dir = st.text_input(
                    "Recognizer Results Folder:", help="Path to folder with recognizer JSON files"
                )
            else:
                analyzer_dir = None

            if viz_mode in [VisualizationMode.BOTH, VisualizationMode.ANONYMIZER_ONLY]:
                anonymizer_dir = st.text_input(
                    "Anonymizer Results Folder:", help="Path to folder with anonymized JSON files"
                )
            else:
                anonymizer_dir = None

            # Quick folder shortcuts
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Current Dir", key="current"):
                    st.rerun()
            with col2:
                if st.button("Home Dir", key="home"):
                    st.rerun()

        analyzer_results, anonymized_results, error = validate_json_paths(viz_mode, analyzer_dir, anonymizer_dir)
        if error is not None:
            if error.startswith("info:"):
                st.info(error[5:])
            else:
                st.error(error)
            return

    elif data_source == DataSource.BIGQUERY:
        with st.sidebar:
            st.subheader("BigQuery Configuration")

            if viz_mode in [VisualizationMode.BOTH, VisualizationMode.RECOGNIZER_ONLY]:
                recognizer_table = st.text_input(
                    "Recognizer Table:",
                    value="",
                    help="BigQuery table ID (project.dataset.table)",
                )
            else:
                recognizer_table = None

            if viz_mode in [VisualizationMode.BOTH, VisualizationMode.ANONYMIZER_ONLY]:
                anonymizer_table = st.text_input(
                    "Anonymizer Table:",
                    help="BigQuery table ID (project.dataset.table)",
                )
            else:
                anonymizer_table = None

            row_limit = st.number_input("Row Limit:", min_value=10, max_value=10000, value=100, step=10)

            only_with_entities = st.checkbox(
                "Only records with entities",
                value=True,
                help="Filter to only load records that have at least one recognized entity",
            )

            load_button = st.button("Load from BigQuery")

        if load_button:
            try:
                recognizer_df = None

                if viz_mode in [VisualizationMode.BOTH, VisualizationMode.RECOGNIZER_ONLY] and recognizer_table:
                    with st.spinner("Getting table info..."):
                        total_rows = get_table_row_count(
                            recognizer_table,
                            only_with_entities=only_with_entities,
                        )
                        st.session_state["recognizer_total_rows"] = total_rows

                    if total_rows > 0 and row_limit > total_rows:
                        st.warning(
                            f"Requested {row_limit} rows but table only has {total_rows} matching rows. "
                            f"Will return all {total_rows} rows."
                        )

                    with st.spinner("Loading recognizer results from BigQuery (random sample)..."):
                        recognizer_df = load_from_bigquery(
                            recognizer_table,
                            limit=row_limit,
                            only_with_entities=only_with_entities,
                            total_rows=total_rows,
                        )
                        analyzer_results = load_recognizer_results_from_dataframe(recognizer_df)
                        st.session_state["analyzer_results"] = analyzer_results
                        st.session_state["sample_index"] = 0

                if viz_mode in [VisualizationMode.BOTH, VisualizationMode.ANONYMIZER_ONLY] and anonymizer_table:
                    if viz_mode == VisualizationMode.BOTH and recognizer_df is not None:
                        # Filter the anonymizer table to the same text_hash
                        # values sampled from the recognizer table.
                        text_hashes = [str(v) for v in recognizer_df[TEXT_HASH_COLUMN].dropna().unique()]
                        with st.spinner("Loading anonymizer results from BigQuery (matching recognizer sample)..."):
                            df = load_from_bigquery(
                                anonymizer_table,
                                limit=row_limit,
                                only_with_entities=only_with_entities,
                                filter_column=TEXT_HASH_COLUMN,
                                filter_values=text_hashes,
                            )
                    else:
                        with st.spinner("Getting table info..."):
                            total_rows = get_table_row_count(
                                anonymizer_table,
                                only_with_entities=only_with_entities,
                            )
                            st.session_state["anonymizer_total_rows"] = total_rows

                        if total_rows > 0 and row_limit > total_rows:
                            st.warning(
                                f"Requested {row_limit} rows but table only has {total_rows} matching rows. "
                                f"Will return all {total_rows} rows."
                            )

                        with st.spinner("Loading anonymizer results from BigQuery (random sample)..."):
                            df = load_from_bigquery(
                                anonymizer_table,
                                limit=row_limit,
                                only_with_entities=only_with_entities,
                                total_rows=total_rows,
                            )

                    anonymized_results = load_anonymizer_results_from_dataframe(df)
                    st.session_state["anonymized_results"] = anonymized_results
                    st.session_state["sample_index"] = 0

            except Exception as e:
                st.error(f"Error loading from BigQuery: {e}")
                return

        # Use cached results if available
        if "analyzer_results" in st.session_state:
            analyzer_results = st.session_state["analyzer_results"]
        if "anonymized_results" in st.session_state:
            anonymized_results = st.session_state["anonymized_results"]

    elif data_source == DataSource.DATAFRAME:
        with st.sidebar:
            st.subheader("DataFrame Configuration")

            if viz_mode in [VisualizationMode.BOTH, VisualizationMode.RECOGNIZER_ONLY]:
                recognizer_file = st.text_input(
                    "Recognizer File:",
                    key="recognizer_file_path",
                    help="Path to parquet or CSV file with recognizer results",
                )
            else:
                recognizer_file = None

            if viz_mode in [VisualizationMode.BOTH, VisualizationMode.ANONYMIZER_ONLY]:
                anonymizer_file = st.text_input(
                    "Anonymizer File:",
                    key="anonymizer_file_path",
                    help="Path to parquet or CSV file with anonymizer results",
                )
            else:
                anonymizer_file = None

            load_button = st.button("Load DataFrame", key="load_dataframe_btn")

        if load_button:
            try:
                analyzer_results, anonymized_results, error = validate_dataframe_paths(
                    viz_mode, recognizer_file, anonymizer_file
                )
                if error is not None:
                    st.error(error)
                    return
                if analyzer_results:
                    st.session_state["analyzer_results"] = analyzer_results
                if anonymized_results:
                    st.session_state["anonymized_results"] = anonymized_results
            except Exception as e:
                st.error(f"Error loading DataFrame: {e}")
                return

        # Use cached results if available
        if "analyzer_results" in st.session_state:
            analyzer_results = st.session_state["analyzer_results"]
        if "anonymized_results" in st.session_state:
            anonymized_results = st.session_state["anonymized_results"]

    # Validate we have data to display
    if viz_mode == VisualizationMode.RECOGNIZER_ONLY and not analyzer_results:
        st.info("No recognizer results loaded. Please load data using the sidebar options.")
        return
    if viz_mode == VisualizationMode.ANONYMIZER_ONLY and not anonymized_results:
        st.info("No anonymizer results loaded. Please load data using the sidebar options.")
        return
    if viz_mode == VisualizationMode.BOTH and not analyzer_results:
        st.info("No recognizer results loaded. Please load data using the sidebar options.")
        return

    # Determine sample IDs based on visualization mode
    if viz_mode == VisualizationMode.ANONYMIZER_ONLY:
        sample_ids = list(anonymized_results.keys())
        sample_count = len(anonymized_results)
    else:
        sample_ids = list(analyzer_results.keys())
        sample_count = len(analyzer_results)

    if not sample_ids:
        st.error("No samples found in the loaded data")
        return

    # Initialize sample index in session state
    if "sample_index" not in st.session_state:
        st.session_state["sample_index"] = 0

    # Ensure index is within bounds
    st.session_state["sample_index"] = clamp_sample_index(st.session_state["sample_index"], len(sample_ids))

    def on_sample_change():
        """Callback when sample selector changes."""
        selected = st.session_state.sample_selector
        if selected in sample_ids:
            st.session_state["sample_index"] = sample_ids.index(selected)

    with st.sidebar:
        st.header("Sample Selection")
        st.info(f"Found {sample_count} samples")

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("< Prev", key="nav_prev", use_container_width=True):
                new_index = (st.session_state["sample_index"] - 1) % len(sample_ids)
                st.session_state["sample_index"] = new_index
                st.rerun()
        with col2:
            st.markdown(
                f"<div style='text-align: center; padding-top: 5px;'>{st.session_state['sample_index'] + 1}/{len(sample_ids)}</div>",
                unsafe_allow_html=True,
            )
        with col3:
            if st.button("Next >", key="nav_next", use_container_width=True):
                new_index = (st.session_state["sample_index"] + 1) % len(sample_ids)
                st.session_state["sample_index"] = new_index
                st.rerun()

        # Get current sample based on index
        current_index = st.session_state["sample_index"]
        selected_sample = sample_ids[current_index]

        # Dropdown to select specific sample - use on_change callback
        st.selectbox(
            "Choose a sample:",
            sample_ids,
            index=current_index,
            format_func=lambda x: f"{x}",
            key="sample_selector",
            on_change=on_sample_change,
        )

    # Get current sample data based on mode
    original_text, original_phi, anonymized_text, anonymized_phi = get_sample_data(
        viz_mode, analyzer_results, anonymized_results, selected_sample
    )

    # Get all recognizers from current sample (only for recognizer modes)
    all_recognizers = []
    if viz_mode in [VisualizationMode.BOTH, VisualizationMode.RECOGNIZER_ONLY] and original_phi:
        all_recognizers = get_recognizers_from_entities(original_phi)

    # Recognizer selection (only show for recognizer modes)
    selected_recognizers = ["All"]
    if viz_mode in [VisualizationMode.BOTH, VisualizationMode.RECOGNIZER_ONLY]:
        with st.sidebar:
            st.header("Recognizer Filter")

            if all_recognizers:
                st.info(f"Available: {', '.join(all_recognizers)}")

                recognizer_options = ["All"] + all_recognizers
                selected_recognizers = st.multiselect("Select recognizers:", recognizer_options, default=["All"])

                if not selected_recognizers:
                    selected_recognizers = ["All"]
            else:
                st.warning("No recognizers found in this sample")

    # Filter entities by selected recognizers (only for original text)
    filtered_original_phi = filter_entities_by_recognizers(original_phi, selected_recognizers)

    # Display statistics
    if viz_mode in [VisualizationMode.BOTH, VisualizationMode.RECOGNIZER_ONLY]:
        with st.sidebar:
            st.divider()
            total_entities = len(original_phi)
            filtered_entities = len(filtered_original_phi)

            if "All" not in selected_recognizers:
                st.success(f"Showing {filtered_entities} of {total_entities} entities")
            else:
                st.success(f"Showing all {total_entities} entities")

    # Main display based on visualization mode
    if viz_mode == VisualizationMode.BOTH:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Text (Filtered by Recognizer)")
            original_doc = convert_to_displacy_format(original_text, filtered_original_phi)
            original_html = displacy.render(original_doc, style="ent", manual=True, options={"colors": COLORS})
            st.html(original_html)

        with col2:
            st.subheader("Anonymized Text (All Entities)")
            anonymized_doc = convert_to_displacy_format(anonymized_text, anonymized_phi)
            anonymized_html = displacy.render(anonymized_doc, style="ent", manual=True, options={"colors": COLORS})
            st.html(anonymized_html)

    elif viz_mode == VisualizationMode.RECOGNIZER_ONLY:
        st.subheader("Original Text with Recognized Entities")
        original_doc = convert_to_displacy_format(original_text, filtered_original_phi)
        original_html = displacy.render(original_doc, style="ent", manual=True, options={"colors": COLORS})
        st.html(original_html)

    elif viz_mode == VisualizationMode.ANONYMIZER_ONLY:
        st.subheader("Anonymized Text")
        anonymized_doc = convert_to_displacy_format(anonymized_text, anonymized_phi)
        anonymized_html = displacy.render(anonymized_doc, style="ent", manual=True, options={"colors": COLORS})
        st.html(anonymized_html)

    # Simple legend
    st.divider()
    st.subheader("Entity Types Legend")

    all_entity_types = collect_entity_types(viz_mode, filtered_original_phi, anonymized_phi)
    legend = build_legend_html(all_entity_types)
    if legend:
        st.html(legend)
    else:
        st.info("No entities to display")


if __name__ == "__main__":
    main()
