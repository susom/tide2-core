"""Unit tests for the TIDE 2.0 visualizer data loading, filtering, and CLI logic."""

import json
from pathlib import Path

import pandas as pd
import pytest

from tide2.cli.main_visualizer import main as visualizer_cli_main
from tide2.cli.unified_interface import build_legend_html
from tide2.cli.unified_interface import collect_entity_types
from tide2.cli.unified_interface import convert_to_displacy_format
from tide2.cli.unified_interface import filter_entities
from tide2.cli.unified_interface import filter_entities_by_recognizers
from tide2.cli.unified_interface import get_recognizer_name
from tide2.cli.unified_interface import get_recognizers_from_entities
from tide2.cli.unified_interface import load_analyzer_results
from tide2.cli.unified_interface import load_anonymized_results
from tide2.cli.unified_interface import load_data
from tide2.cli.unified_interface import load_pipeline_output_dir
from tide2.cli.unified_interface import parse_cli_args
from tide2.cli.unified_interface import parse_entities


def test_parse_entities():
    """Verify parsing of entities from JSON strings, lists, dicts, and empty values."""
    assert parse_entities(None) == []
    assert parse_entities(float("nan")) == []
    assert parse_entities("not valid json") == []
    assert parse_entities("123") == []

    # Valid JSON string
    raw_list = [{"entity_type": "PERSON", "start": 0, "end": 4}]
    assert parse_entities(json.dumps(raw_list)) == raw_list

    raw_dict = {"entity_type": "DATE_TIME", "start": 5, "end": 10}
    assert parse_entities(json.dumps(raw_dict)) == [raw_dict]

    # Direct list and dict
    assert parse_entities(raw_list) == raw_list
    assert parse_entities(raw_dict) == [raw_dict]


def test_get_recognizer_name():
    """Verify recognizer name extraction from entity dictionaries."""
    assert get_recognizer_name({}) == "-"
    assert get_recognizer_name({"recognizer_name": "TestRecognizer"}) == "TestRecognizer"
    assert get_recognizer_name({"recognition_metadata": {"recognizer_name": "MetaRecognizer"}}) == "MetaRecognizer"


def test_get_recognizers_from_entities():
    """Verify extracting unique sorted recognizers."""
    entities = [
        {"recognition_metadata": {"recognizer_name": "RecB"}},
        {"recognizer_name": "RecA"},
        {"recognition_metadata": {"recognizer_name": "RecB"}},
        {},
    ]
    assert get_recognizers_from_entities(entities) == ["RecA", "RecB"]


def test_filter_entities():
    """Verify entity filtering by recognizer and entity type."""
    entities = [
        {
            "entity_type": "PERSON",
            "start": 0,
            "end": 8,
            "recognition_metadata": {"recognizer_name": "BERT"},
        },
        {
            "entity_type": "DATE_TIME",
            "start": 10,
            "end": 20,
            "recognition_metadata": {"recognizer_name": "DateRegex"},
        },
        {
            "entity_type": "LOCATION",
            "start": 22,
            "end": 30,
            "recognizer_name": "BERT",
        },
    ]

    # No filter / "All"
    assert len(filter_entities(entities, selected_recognizers=["All"], selected_entity_types=["All"])) == 3
    assert len(filter_entities(entities, selected_recognizers=None, selected_entity_types=None)) == 3

    # Filter by recognizer
    bert_only = filter_entities(entities, selected_recognizers=["BERT"])
    assert len(bert_only) == 2
    assert all(get_recognizer_name(e) == "BERT" for e in bert_only)

    # Filter by entity type
    person_only = filter_entities(entities, selected_entity_types=["PERSON"])
    assert len(person_only) == 1
    assert person_only[0]["entity_type"] == "PERSON"

    # Filter by both
    combo = filter_entities(entities, selected_recognizers=["BERT"], selected_entity_types=["PERSON"])
    assert len(combo) == 1
    assert combo[0]["entity_type"] == "PERSON"

    # Backward compatibility helper
    assert len(filter_entities_by_recognizers(entities, ["BERT"])) == 2


def test_convert_to_displacy_format():
    """Verify conversion to displacy format with boundary validation."""
    text = "Hello John Doe!"
    entities = [
        {"start": 6, "end": 14, "entity_type": "PERSON"},
        {"start": -1, "end": 4, "entity_type": "INVALID"},  # negative start
        {"start": 10, "end": 5, "entity_type": "INVALID"},  # end <= start
        {"start": 6, "end": 100, "entity_type": "INVALID"},  # beyond text length
    ]
    doc = convert_to_displacy_format(text, entities)
    assert doc["text"] == text
    assert len(doc["ents"]) == 1
    assert doc["ents"][0] == {"start": 6, "end": 14, "label": "PERSON"}


def test_collect_entity_types_and_legend():
    """Verify entity type collection and HTML legend generation."""
    orig = [{"entity_type": "PERSON"}, {"entity_type": "DATE_TIME"}]
    anon = [{"entity_type": "DATE_TIME"}, {"entity_type": "LOCATION"}]

    types = collect_entity_types(orig, anon)
    assert types == {"PERSON", "DATE_TIME", "LOCATION"}

    html = build_legend_html(types)
    assert "PERSON" in html
    assert "DATE_TIME" in html
    assert "LOCATION" in html

    assert build_legend_html(set()) == ""


def test_load_data_single_parquet_file(tmp_path: Path):
    """Verify loading a single Parquet file containing both stages."""
    df = pd.DataFrame(
        [
            {
                "row_id": "row_1",
                "text_hash": "hash_1",
                "note_text": "Patient John Doe visited on 2024-01-01.",
                "recognizer_results_json": json.dumps([{"entity_type": "PERSON", "start": 8, "end": 16}]),
                "anonymized_note_text": "Patient [PERSON] visited on [DATE_TIME].",
                "anonymizer_results_json": json.dumps([{"entity_type": "PERSON", "start": 8, "end": 16}]),
            }
        ]
    )
    file_path = tmp_path / "sample.parquet"
    df.to_parquet(file_path)

    results, desc = load_data(file_path)
    assert desc == "Single Parquet File"
    assert "row_1" in results
    assert results["row_1"]["original_text"] == "Patient John Doe visited on 2024-01-01."
    assert results["row_1"]["modified_text"] == "Patient [PERSON] visited on [DATE_TIME]."
    assert len(results["row_1"]["original_phi"]) == 1
    assert len(results["row_1"]["modified_phi"]) == 1


def test_load_data_parquet_directory(tmp_path: Path):
    """Verify loading a directory directly containing Parquet files."""
    df1 = pd.DataFrame([{"row_id": "note_1", "note_text": "First note"}])
    df2 = pd.DataFrame([{"row_id": "note_2", "note_text": "Second note"}])
    df1.to_parquet(tmp_path / "part-0.parquet")
    df2.to_parquet(tmp_path / "part-1.parquet")

    results, desc = load_data(tmp_path)
    assert desc == "Parquet Directory"
    assert "note_1" in results
    assert "note_2" in results
    assert results["note_1"]["original_text"] == "First note"


def test_load_data_pipeline_output_dir(tmp_path: Path):
    """Verify pipeline output stage discovery and row alignment."""
    out_dir = tmp_path / "pipeline_run"
    out_dir.mkdir()

    rec_dir = out_dir / "recognizer_output"
    rec_dir.mkdir()
    rec_df = pd.DataFrame(
        [
            {
                "row_id": "p1",
                "text_hash": "th1",
                "note_text": "Original note 1",
                "recognizer_results_json": json.dumps([{"entity_type": "PERSON", "start": 0, "end": 8}]),
            }
        ]
    )
    rec_df.to_parquet(rec_dir / "part-0.parquet")

    anon_dir = out_dir / "anonymizer_output"
    anon_dir.mkdir()
    anon_df = pd.DataFrame(
        [
            {
                "row_id": "p1",
                "text_hash": "th1",
                "anonymized_note_text": "Deidentified note 1",
                "anonymizer_results_json": json.dumps([{"entity_type": "PERSON", "start": 0, "end": 8}]),
            }
        ]
    )
    anon_df.to_parquet(anon_dir / "part-0.parquet")

    results, desc = load_data(out_dir)
    assert desc == "Pipeline Output Directory (Parquet)"
    assert "p1" in results
    assert results["p1"]["original_text"] == "Original note 1"
    assert results["p1"]["modified_text"] == "Deidentified note 1"
    assert len(results["p1"]["original_phi"]) == 1
    assert len(results["p1"]["modified_phi"]) == 1


def test_load_pipeline_output_recovers_text_from_transformer_partial_null(tmp_path: Path):
    """Verify that note_text is recovered per-row from transformer when recognizer has mixed populated/null text."""
    out_dir = tmp_path / "pipeline_run_partial"
    out_dir.mkdir()

    trans_dir = out_dir / "transformer_output"
    trans_dir.mkdir()
    pd.DataFrame(
        [
            {"row_id": "r1", "text_hash": "th1", "note_text": "Transformer Text 1"},
            {"row_id": "r2", "text_hash": "th2", "note_text": "Transformer Text 2"},
        ]
    ).to_parquet(trans_dir / "part-0.parquet")

    rec_dir = out_dir / "recognizer_output"
    rec_dir.mkdir()
    pd.DataFrame(
        [
            {"row_id": "r1", "text_hash": "th1", "note_text": "Recognizer Text 1", "recognizer_results_json": "[]"},
            {"row_id": "r2", "text_hash": "th2", "note_text": None, "recognizer_results_json": "[]"},
        ]
    ).to_parquet(rec_dir / "part-0.parquet")

    results = load_pipeline_output_dir(out_dir)
    assert "r1" in results and "r2" in results
    assert results["r1"]["original_text"] == "Recognizer Text 1"
    assert results["r2"]["original_text"] == "Transformer Text 2"


def test_load_pipeline_output_recovers_text_from_transformer(tmp_path: Path):
    """Verify that note_text is recovered from transformer_output if missing in recognizer."""
    out_dir = tmp_path / "pipeline_run_recovering"
    out_dir.mkdir()

    # Transformer stage has note_text and text_hash
    trans_dir = out_dir / "transformer_output"
    trans_dir.mkdir()
    pd.DataFrame([{"text_hash": "th100", "note_text": "Recovered text"}]).to_parquet(trans_dir / "part-0.parquet")

    # Recognizer stage lacks note_text
    rec_dir = out_dir / "recognizer_output"
    rec_dir.mkdir()
    pd.DataFrame([{"row_id": "r100", "text_hash": "th100", "recognizer_results_json": "[]"}]).to_parquet(
        rec_dir / "part-0.parquet"
    )

    results = load_pipeline_output_dir(out_dir)
    assert "r100" in results
    assert results["r100"]["original_text"] == "Recovered text"


def test_load_data_json_directory(tmp_path: Path):
    """Verify loading from cli_recognizer_json and cli_anonymizer_json subdirectories."""
    out_dir = tmp_path / "json_run"
    out_dir.mkdir()

    rec_json_dir = out_dir / "cli_recognizer_json"
    rec_json_dir.mkdir()
    rec_data = {
        "key": "sample_alpha",
        "value": "Note with John",
        "recognizer_results": [{"entity_type": "PERSON", "start": 10, "end": 14}],
    }
    (rec_json_dir / "sample_alpha.json").write_text(json.dumps(rec_data), encoding="utf-8")

    anon_json_dir = out_dir / "cli_anonymizer_json"
    anon_json_dir.mkdir()
    anon_data = {
        "text": "Note with [NAME]",
        "items": [{"entity_type": "PERSON", "start": 10, "end": 16}],
    }
    (anon_json_dir / "sample_alpha.json").write_text(json.dumps(anon_data), encoding="utf-8")

    results, desc = load_data(out_dir)
    assert desc == "JSON Directory (cli_recognizer_json / cli_anonymizer_json)"
    assert "sample_alpha" in results
    assert results["sample_alpha"]["original_text"] == "Note with John"
    assert results["sample_alpha"]["modified_text"] == "Note with [NAME]"

    # Test backward-compatible helper loaders
    assert len(load_analyzer_results(rec_json_dir)) == 1
    assert len(load_anonymized_results(anon_json_dir)) == 1


def test_load_data_json_directory_max_notes(tmp_path: Path):
    """Verify max_notes bounds the loading of JSON directory files."""
    out_dir = tmp_path / "json_run_bounded"
    out_dir.mkdir()
    rec_json_dir = out_dir / "cli_recognizer_json"
    rec_json_dir.mkdir()
    anon_json_dir = out_dir / "cli_anonymizer_json"
    anon_json_dir.mkdir()

    for i in range(10):
        sid = f"sample_{i}"
        (rec_json_dir / f"{sid}.json").write_text(json.dumps({"key": sid, "value": f"text {i}"}), encoding="utf-8")
        (anon_json_dir / f"{sid}.json").write_text(json.dumps({"text": f"anon {i}", "items": []}), encoding="utf-8")

    results, desc = load_data(out_dir, max_notes=3)
    assert desc == "JSON Directory (cli_recognizer_json / cli_anonymizer_json)"
    assert len(results) == 3


def test_load_data_loose_json_files(tmp_path: Path):
    """Verify loading directory directly containing JSON files."""
    data = {
        "key": "sample_loose",
        "value": "Original",
        "recognizer_results": [],
        "text": "Anonymized",
        "items": [],
    }
    (tmp_path / "loose.json").write_text(json.dumps(data), encoding="utf-8")

    results, desc = load_data(tmp_path)
    assert desc == "JSON Files Directory"
    assert "sample_loose" in results
    assert results["sample_loose"]["original_text"] == "Original"
    assert results["sample_loose"]["modified_text"] == "Anonymized"


def test_load_data_single_json_file(tmp_path: Path):
    """Verify loading an individual JSON file."""
    data = {
        "key": "single_sample",
        "value": "Original note",
        "recognizer_results": [{"entity_type": "DATE_TIME", "start": 0, "end": 4}],
    }
    file_path = tmp_path / "single.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    results, desc = load_data(file_path)
    assert desc == "Single JSON File"
    assert "single_sample" in results
    assert results["single_sample"]["original_text"] == "Original note"


def test_load_data_single_json_anonymizer_disambiguation(tmp_path: Path):
    """Verify loading a single anonymizer JSON populates only modified_text and modified_phi."""
    data = {
        "text": "Patient [PERSON] visited on [DATE].",
        "items": [{"entity_type": "PERSON", "start": 8, "end": 16}],
    }
    file_path = tmp_path / "anon_single.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    results, desc = load_data(file_path)
    assert desc == "Single JSON File"
    assert "anon_single" in results
    sample = results["anon_single"]
    assert sample["original_text"] == ""
    assert sample["original_phi"] == []
    assert sample["modified_text"] == "Patient [PERSON] visited on [DATE]."
    assert len(sample["modified_phi"]) == 1


def test_load_data_preserves_text_hash(tmp_path: Path):
    """Verify text_hash is preserved in pipeline and single parquet loads."""
    df = pd.DataFrame(
        [
            {
                "row_id": "row_123",
                "text_hash": "hash_abc",
                "note_text": "Sample text",
                "recognizer_results_json": "[]",
            }
        ]
    )
    p_file = tmp_path / "test.parquet"
    df.to_parquet(p_file)

    results, _ = load_data(p_file)
    assert "row_123" in results
    assert results["row_123"]["text_hash"] == "hash_abc"


def test_load_anonymizer_only_parquet_disambiguation(tmp_path: Path):
    """Verify an anonymizer-only Parquet file with text/items assigns text to modified_text."""
    df = pd.DataFrame(
        [
            {
                "row_id": "anon_1",
                "text": "Anonymized text content",
                "items": json.dumps([{"entity_type": "PERSON", "start": 0, "end": 10}]),
            }
        ]
    )
    p_file = tmp_path / "anon_only.parquet"
    df.to_parquet(p_file)

    results, _ = load_data(p_file)
    assert "anon_1" in results
    sample = results["anon_1"]
    assert sample["original_text"] == ""
    assert sample["modified_text"] == "Anonymized text content"
    assert len(sample["modified_phi"]) == 1


def test_load_data_max_notes_cap(tmp_path: Path):
    """Verify max_notes bounds the number of loaded records."""
    df = pd.DataFrame([{"row_id": f"r_{i}", "note_text": f"Note {i}"} for i in range(10)])
    p_file = tmp_path / "multi.parquet"
    df.to_parquet(p_file)

    results, _ = load_data(p_file, max_notes=3)
    assert len(results) == 3


def test_load_data_errors(tmp_path: Path):
    """Verify error handling on invalid or empty paths."""
    # Empty path
    res, desc = load_data("")
    assert res == {}
    assert desc == "No path provided"

    # Non-existent path
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nonexistent")

    # Unsupported file extension
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b,c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data(csv_file)

    # Empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No recognizable Parquet stages or JSON files"):
        load_data(empty_dir)


def test_parse_cli_args():
    """Verify CLI argument extraction for visualizer."""
    assert parse_cli_args([]) == ""
    assert parse_cli_args(["--server.port=8501"]) == ""
    assert parse_cli_args(["./data/output"]) == "./data/output"
    assert parse_cli_args(["--theme=dark", "./data/output"]) == "./data/output"


def test_main_visualizer_cli_delegation(monkeypatch):
    """Verify main_visualizer correctly passes arguments to subprocess."""
    captured_cmd = []

    def fake_run(cmd, check=True):
        captured_cmd.extend(cmd)

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    visualizer_cli_main(["./custom/path"])

    assert "streamlit" in captured_cmd
    assert "run" in captured_cmd
    assert "--" in captured_cmd
    assert "./custom/path" in captured_cmd
