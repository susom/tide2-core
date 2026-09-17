"""
Minimal integration test for configurable LLM prompts + MaskingAnonymizer.

Part 1: Run LlmJsonRecognizer directly over sample notes.
Part 2: Apply MaskingAnonymizer to the recognizer output.

Usage:
    # Set your GCP project ID before running:
    export GCP_PROJECT_ID="your-project-id"

    python notebooks/tide2_llm_only.py
"""

import json
import os
import sys
from pathlib import Path

from presidio_analyzer import RecognizerResult as AnalyzerRecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.entities import RecognizerResult

from tide2.anonymizers import presidio_patches
from tide2.anonymizers.masking import MaskingAnonymizer
from tide2.recognizers.llm_json_recognizer import LlmJsonRecognizer

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

# ---------------------------------------------------------------------------
# Part 1: Run LlmJsonRecognizer directly over each note
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART 1: LLM Recognition")
print("=" * 60)

text_files_dir = SAMPLE_DATA_DIR / "text_files"
notes = {txt_file.stem: txt_file.read_text(encoding="utf-8") for txt_file in sorted(text_files_dir.glob("*.txt"))}
print(f"Loaded {len(notes)} notes from {text_files_dir}")

recognizer = LlmJsonRecognizer(
    project_id=PROJECT_ID,
    provider_type="google",
    model_name=MODEL_NAME,
    prompt_name=PROMPT_NAME,
)

recognizer_results_by_note: dict[str, list[AnalyzerRecognizerResult]] = {}
for note_id, note_text in notes.items():
    results = recognizer.analyze(text=note_text, entities=recognizer.supported_entities_list, nlp_artifacts=None)
    recognizer_results_by_note[note_id] = results
    print(f"  [{note_id}] {len(results)} entities found")

# ---------------------------------------------------------------------------
# Part 2: Apply MaskingAnonymizer to recognizer output
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2: MaskingAnonymizer")
print("=" * 60)

# Apply Presidio patches
presidio_patches.disable_whitespace_merging()
presidio_patches.patch_conflict_resolution()

# Initialize engine with MaskingAnonymizer
engine = AnonymizerEngine()
engine.add_anonymizer(MaskingAnonymizer)

output_dir = SAMPLE_DATA_DIR / "llm_test_output"
output_dir.mkdir(parents=True, exist_ok=True)

for note_id, note_text in notes.items():
    recognizer_results = recognizer_results_by_note[note_id]
    if not recognizer_results:
        print(f"\n  [{note_id}] No entities found")
        continue

    operators = {"DEFAULT": OperatorConfig("masking")}
    # presidio-anonymizer defines its own RecognizerResult distinct from presidio-analyzer's;
    # convert explicitly rather than relying on structural compatibility.
    anonymizer_results = [
        RecognizerResult(entity_type=r.entity_type, start=r.start, end=r.end, score=r.score) for r in recognizer_results
    ]
    anonymized = engine.anonymize(
        text=note_text,
        analyzer_results=anonymizer_results,
        operators=operators,
    )

    result_json = {
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
    (output_dir / f"{note_id}.json").write_text(json.dumps(result_json, indent=2), encoding="utf-8")

    print(f"\n--- Note {note_id} ---")
    print(f"  Entities found: {len(recognizer_results)}")
    print("  Anonymized text (first 500 chars):")
    print(f"    {anonymized.text[:500]}")

print(f"\nWrote {len(notes)} anonymized note(s) to {output_dir}")
