"""
Minimal integration test for configurable LLM prompts + MaskingAnonymizer.

Part 1: Run LLM recognizer on sample data via JobRunner (llm_recognizer_mode="only")
Part 2: Read recognizer output and apply MaskingAnonymizer directly

Usage:
    # Set your GCP project ID before running:
    export GCP_PROJECT_ID="your-project-id"

    python notebooks/tide2_llm_only.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
if not PROJECT_ID:
    print("Error: set GCP_PROJECT_ID environment variable")
    sys.exit(1)
MODEL_NAME = "gemini-2.5-flash"
PROMPT_NAME = "phi_detection"

# Resolve sample_data relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = SCRIPT_DIR / "sample_data"
if not SAMPLE_DATA_DIR.exists():
    print(f"Error: sample_data/ not found at {SAMPLE_DATA_DIR}")
    sys.exit(1)

OUTPUT_DIR = str(SAMPLE_DATA_DIR / "llm_test_output")

# ---------------------------------------------------------------------------
# Part 1: Run LLM recognizer via JobRunner
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART 1: LLM Recognition")
print("=" * 60)

# Load sample text files (same pattern as tide2_pipeline.ipynb)
text_files_dir = SAMPLE_DATA_DIR / "text_files"
records = []
for txt_file in sorted(text_files_dir.glob("*.txt")):
    note_id = txt_file.stem
    note_text = txt_file.read_text(encoding="utf-8")
    records.append(
        {
            "note_text": note_text,
            "patient_id": note_id,
            "patient_identifiers": "{}",
        }
    )

df = pd.DataFrame(records)
print(f"Loaded {len(df)} notes from {text_files_dir}")

from tide2.runner import LocalJobRunner

runner = LocalJobRunner(num_cpus=4, object_store_gb=2)
try:
    result = runner.run_pipeline(
        input_data=df,
        output_dir=OUTPUT_DIR,
        model_name="unused-in-llm-only-mode",
        run_transformer=False,
        run_recognizer=False,
        run_anonymizer=False,
        produce_visualizer_json=True,
        salt_hex="00" * 32,
        key_hex="11" * 32,
        llm_recognizer_mode="only",
        llm_recognizer_kwargs={
            "project_id": PROJECT_ID,
            "model_name": MODEL_NAME,
            "num_actors": 2,
            "batch_size": 5,
            "prompt_name": PROMPT_NAME,
        },
    )
    print("\nLLM Recognition result:")
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
finally:
    runner.shutdown()

# ---------------------------------------------------------------------------
# Part 2: Apply MaskingAnonymizer to recognizer output
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2: MaskingAnonymizer")
print("=" * 60)

import pyarrow.parquet as pq
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.entities import RecognizerResult

from tide2.anonymizers import presidio_patches
from tide2.anonymizers.masking import MaskingAnonymizer

# Apply Presidio patches
presidio_patches.disable_whitespace_merging()
presidio_patches.patch_conflict_resolution()

# Initialize engine with MaskingAnonymizer
engine = AnonymizerEngine()
engine.add_anonymizer(MaskingAnonymizer)

# Read recognizer output + original note text
output_path = Path(OUTPUT_DIR)
recognizer_output_dir = output_path / "04_recognizer_output"
rec_files = list(recognizer_output_dir.glob("**/*.parquet"))
if not rec_files:
    print("No recognizer output found. Part 1 may have failed.")
    sys.exit(1)

input_parquet = output_path / "01_transformer_input.parquet"
df_input = pd.read_parquet(input_parquet)[["text_hash", "note_text"]]

# Create output directory for anonymizer visualizer JSONs
anon_json_dir = output_path / "cli_anonymizer_json"
anon_json_dir.mkdir(parents=True, exist_ok=True)

for pf in rec_files:
    df_rec = pq.read_table(pf).to_pandas()
    df_merged = df_rec.merge(df_input, on="text_hash", how="left")

    for _, row in df_merged.iterrows():
        text_hash = row["text_hash"]
        note_text = row["note_text"]
        results_json = row.get("recognizer_results_json", "[]")
        if not results_json or results_json == "[]":
            print(f"\n  [{text_hash[:16]}...] No entities found")
            continue

        # Deserialize recognizer results
        results_list = json.loads(results_json)
        recognizer_results = [
            RecognizerResult(
                entity_type=r["entity_type"],
                start=r["start"],
                end=r["end"],
                score=r.get("score", 1.0),
            )
            for r in results_list
        ]

        # Apply masking for all entity types
        operators = {"DEFAULT": OperatorConfig("masking")}
        anonymized = engine.anonymize(
            text=note_text,
            analyzer_results=recognizer_results,
            operators=operators,
        )

        # Save anonymizer JSON for visualizer
        anon_json = {
            "text": anonymized.text,
            "items": [
                {
                    "start": item.start,
                    "end": item.end,
                    "entity_type": item.entity_type,
                    "operator": item.operator,
                }
                for item in anonymized.items
            ],
        }
        anon_json_path = anon_json_dir / f"{text_hash}.json"
        with anon_json_path.open("w", encoding="utf-8") as f:
            json.dump(anon_json, f, indent=2)

        print(f"\n--- Note {text_hash[:16]}... ---")
        print(f"  Entities found: {len(recognizer_results)}")
        print("  Anonymized text (first 500 chars):")
        print(f"    {anonymized.text[:500]}")

# Print visualizer paths
cli_rec = output_path / "cli_recognizer_json"
rec_count = len(list(cli_rec.glob("*.json"))) if cli_rec.exists() else 0
anon_count = len(list(anon_json_dir.glob("*.json")))

print("\n" + "=" * 60)
print("VISUALIZER DATA DIRECTORIES")
print("=" * 60)
print(f"\nRecognizer JSON ({rec_count} files):")
print(f"  {cli_rec}")
print(f"\nAnonymizer JSON ({anon_count} files):")
print(f"  {anon_json_dir}")
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
