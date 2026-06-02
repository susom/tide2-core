#!/usr/bin/env python3
"""Create and read parquet files for tide2-runner run pipeline.

Usage
-----
Create a parquet file from a JSON file of notes::

    uv run scripts/parquet_utils.py create notes.json output.parquet

Read and print a parquet file::

    uv run scripts/parquet_utils.py read output.parquet

JSON input format (list of objects, only note_text is required)::

    [
      {
        "note_text": "Patient Jane Doe was seen on 01/15/2024.",
        "patient_id": "P001",
        "patient_identifiers": {
          "person": ["Jane Doe"],
          "mrn": ["MRN001"]
        }
      }
    ]

Pipeline input schema
---------------------
Required columns:
  note_text             Clinical note text.

Auto-populated when absent:
  text_hash             SHA256 of note_text.
  patient_id            Defaults to text_hash.
  patient_identifiers   JSON string of known PHI values per entity type.

Optional pass-through columns:
  patient_uid
  jitter
  recognizer_results_json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_text_hash(text: str) -> str:
    """Return the SHA256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_parquet_files(path: str | Path) -> list[Path]:
    """Return sorted parquet files from a file, directory, or glob pattern."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.parquet"))
    matches = sorted(glob.glob(str(path), recursive=True))
    return [Path(m) for m in matches if Path(m).is_file()]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def create_pipeline_parquet(
    notes: list[dict],
    output_path: str | Path,
    compression: str = "snappy",
) -> Path:
    """Write *notes* to a parquet file ready for ``tide2-runner run pipeline``.

    Each dict in *notes* must contain ``note_text``.  All other fields are
    optional and populated with defaults when absent.

    Args:
        notes: List of note dicts.
        output_path: Destination ``.parquet`` file path.
        compression: Parquet compression codec.

    Returns:
        Resolved path to the written file.

    Raises:
        ValueError: If *notes* is empty or any entry is missing ``note_text``.
    """
    if not notes:
        raise ValueError("notes list cannot be empty")

    rows: list[dict] = []
    for i, note in enumerate(notes):
        if "note_text" not in note or note["note_text"] is None:
            raise ValueError(f"notes[{i}] is missing required key 'note_text'")

        note_text: str = str(note["note_text"])
        text_hash: str = note.get("text_hash") or _compute_text_hash(note_text)
        patient_id: str = note.get("patient_id") or text_hash

        raw_pi = note.get("patient_identifiers", "{}")
        if isinstance(raw_pi, dict):
            patient_identifiers = json.dumps(raw_pi)
        else:
            patient_identifiers = str(raw_pi) if raw_pi else "{}"

        row: dict = {
            "note_text": note_text,
            "text_hash": text_hash,
            "patient_id": patient_id,
            "patient_identifiers": patient_identifiers,
        }

        for optional_col in (
            "patient_uid",
            "jitter",
            "recognizer_results_json",
        ):
            if optional_col in note:
                row[optional_col] = note[optional_col]

        rows.append(row)

    df = pd.DataFrame(rows)

    ordered_cols = ["note_text", "text_hash", "patient_id", "patient_identifiers"]
    for col in ("patient_uid", "jitter", "recognizer_results_json"):
        if col in df.columns:
            ordered_cols.append(col)
    df = df[ordered_cols]

    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False, compression=compression)
    return dest


def read_pipeline_parquet(
    path: str | Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read pipeline parquet file(s) into a DataFrame.

    Column names are normalised to lowercase.  Accepts a single file,
    directory, or glob pattern.

    Args:
        path: Parquet file, directory, or glob pattern.
        columns: Subset of columns to load (case-insensitive).  ``None``
            loads all columns.

    Returns:
        DataFrame with lowercase column names.

    Raises:
        FileNotFoundError: If no parquet files are found at *path*.
        ValueError: If any requested *columns* are absent from the schema.
    """
    parquet_files = _resolve_parquet_files(path)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at: {path}")

    tables: list[pa.Table] = []
    for file_path in parquet_files:
        pf = pq.ParquetFile(file_path)
        schema_names = pf.schema_arrow.names
        lower_to_actual = {c.lower(): c for c in schema_names}

        if columns is not None:
            missing = [c for c in columns if c.lower() not in lower_to_actual]
            if missing:
                raise ValueError(
                    f"Requested columns {missing} not found in {file_path}. "
                    f"Available: {schema_names}"
                )
            actual_cols = [lower_to_actual[c.lower()] for c in columns]
            table = pf.read(columns=actual_cols)
        else:
            table = pf.read()

        tables.append(table)

    combined = pa.concat_tables(tables, promote_options="default")
    df = combined.to_pandas()
    df.columns = df.columns.str.lower()
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_create(args: argparse.Namespace) -> None:
    """Handle the ``create`` sub-command."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with input_path.open() as fh:
        notes = json.load(fh)

    if not isinstance(notes, list):
        print("ERROR: JSON input must be a list of note objects", file=sys.stderr)
        sys.exit(1)

    dest = create_pipeline_parquet(notes, args.output, compression=args.compression)
    print(f"Written {len(notes)} note(s) to {dest}")
    df = read_pipeline_parquet(dest)
    print(df.to_string())


def cmd_read(args: argparse.Namespace) -> None:
    """Handle the ``read`` sub-command."""
    columns = args.columns.split(",") if args.columns else None
    df = read_pipeline_parquet(args.path, columns=columns)
    output = {
        "shape": df.shape,
        "columns": list(df.columns),
        "data": df.to_dict(orient="records")
    }
    print(json.dumps(output, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Create and read parquet files for tide2-runner run pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create sub-command
    p_create = subparsers.add_parser(
        "create",
        help="Create a pipeline parquet file from a JSON file of notes.",
    )
    p_create.add_argument("input", help="Path to JSON input file (list of note dicts).")
    p_create.add_argument("output", help="Destination parquet file path.")
    p_create.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "gzip", "zstd", "none"],
        help="Parquet compression codec (default: snappy).",
    )
    p_create.set_defaults(func=cmd_create)

    # read sub-command
    p_read = subparsers.add_parser(
        "read",
        help="Read and print a pipeline parquet file.",
    )
    p_read.add_argument(
        "path", help="Path to parquet file, directory, or glob pattern."
    )
    p_read.add_argument(
        "--columns",
        default=None,
        help="Comma-separated list of columns to load (default: all).",
    )
    p_read.set_defaults(func=cmd_read)

    return parser


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
