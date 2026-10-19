"""Unit tests for the TIDE 2.0 visualizer data loading, filtering, and CLI logic."""

import inspect
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import tide2.cli.unified_interface as ui
from tide2.cli.main_visualizer import main as visualizer_cli_main
from tide2.cli.unified_interface import build_legend_html
from tide2.cli.unified_interface import collect_entity_types
from tide2.cli.unified_interface import convert_to_displacy_format
from tide2.cli.unified_interface import filter_entities
from tide2.cli.unified_interface import filter_entities_by_recognizers
from tide2.cli.unified_interface import get_recognizer_name
from tide2.cli.unified_interface import get_recognizers_from_entities
from tide2.cli.unified_interface import load_data
from tide2.cli.unified_interface import load_pipeline_output_dir
from tide2.cli.unified_interface import parse_cli_args
from tide2.cli.unified_interface import parse_entities
from tide2.runner.local_runner import add_row_id


def test_guard_no_pandas_in_visualizer():
    """Ensure pandas is not imported or used in unified_interface."""
    source = inspect.getsource(ui)
    assert "import pandas" not in source
    assert "from pandas" not in source
    assert "pd." not in source


def test_parse_entities():
    """Verify parsing of entities from JSON strings, lists, dicts, numpy arrays, and empty values."""
    assert parse_entities(None) == []
    assert parse_entities(float("nan")) == []
    assert parse_entities("not valid json") == []
    assert parse_entities("123") == []

    raw_list = [{"entity_type": "PERSON", "start": 0, "end": 4}]
    assert parse_entities(json.dumps(raw_list)) == raw_list

    raw_dict = {"entity_type": "DATE_TIME", "start": 5, "end": 10}
    assert parse_entities(json.dumps(raw_dict)) == [raw_dict]

    assert parse_entities(raw_list) == raw_list
    assert parse_entities(raw_dict) == [raw_dict]
    assert parse_entities(np.array(raw_list, dtype=object)) == raw_list


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

    assert len(filter_entities(entities, selected_recognizers=["All"], selected_entity_types=["All"])) == 3
    assert len(filter_entities(entities, selected_recognizers=None, selected_entity_types=None)) == 3

    bert_only = filter_entities(entities, selected_recognizers=["BERT"])
    assert len(bert_only) == 2
    assert all(get_recognizer_name(e) == "BERT" for e in bert_only)

    person_only = filter_entities(entities, selected_entity_types=["PERSON"])
    assert len(person_only) == 1
    assert person_only[0]["entity_type"] == "PERSON"

    combo = filter_entities(entities, selected_recognizers=["BERT"], selected_entity_types=["PERSON"])
    assert len(combo) == 1
    assert combo[0]["entity_type"] == "PERSON"

    assert len(filter_entities_by_recognizers(entities, ["BERT"])) == 2


def test_convert_to_displacy_format():
    """Verify conversion to displacy format with boundary validation."""
    text = "Hello John Doe!"
    entities = [
        {"start": 6, "end": 14, "entity_type": "PERSON"},
        {"start": -1, "end": 4, "entity_type": "INVALID"},
        {"start": 10, "end": 5, "entity_type": "INVALID"},
        {"start": 6, "end": 100, "entity_type": "INVALID"},
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
    file_path = tmp_path / "sample.parquet"
    pq.write_table(
        pa.Table.from_pylist(
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
        ),
        file_path,
    )

    results, desc = load_data(file_path)
    assert desc == "Single Parquet File"
    assert "row_1" in results
    assert results["row_1"]["original_text"] == "Patient John Doe visited on 2024-01-01."
    assert results["row_1"]["modified_text"] == "Patient [PERSON] visited on [DATE_TIME]."
    assert len(results["row_1"]["original_phi"]) == 1
    assert len(results["row_1"]["modified_phi"]) == 1


def test_load_data_parquet_directory(tmp_path: Path):
    """Verify loading a directory directly containing Parquet files."""
    pq.write_table(pa.Table.from_pylist([{"row_id": "note_1", "note_text": "First note"}]), tmp_path / "part-0.parquet")
    pq.write_table(
        pa.Table.from_pylist([{"row_id": "note_2", "note_text": "Second note"}]), tmp_path / "part-1.parquet"
    )

    results, desc = load_data(tmp_path)
    assert desc == "Parquet Directory"
    assert "note_1" in results
    assert "note_2" in results
    assert results["note_1"]["original_text"] == "First note"


def test_load_data_pipeline_output_dir(tmp_path: Path):
    """Verify pipeline output stage discovery and row alignment by row_id."""
    out_dir = tmp_path / "pipeline_run"
    out_dir.mkdir()

    rec_dir = out_dir / "recognizer_output"
    rec_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_id": "p1",
                    "text_hash": "th1",
                    "note_text": "Original note 1",
                    "recognizer_results_json": json.dumps([{"entity_type": "PERSON", "start": 0, "end": 8}]),
                }
            ]
        ),
        rec_dir / "part-0.parquet",
    )

    anon_dir = out_dir / "anonymizer_output"
    anon_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_id": "p1",
                    "text_hash": "th1",
                    "anonymized_note_text": "Deidentified note 1",
                    "anonymizer_results_json": json.dumps([{"entity_type": "PERSON", "start": 0, "end": 8}]),
                }
            ]
        ),
        anon_dir / "part-0.parquet",
    )

    results, desc = load_data(out_dir)
    assert desc == "Pipeline Output Directory (Parquet)"
    assert "p1" in results
    assert results["p1"]["original_text"] == "Original note 1"
    assert results["p1"]["modified_text"] == "Deidentified note 1"
    assert len(results["p1"]["original_phi"]) == 1
    assert len(results["p1"]["modified_phi"]) == 1


def test_load_pipeline_anonymizer_hash_only_fallback(tmp_path: Path):
    """Verify that when row_id is absent from both stages, row_id is derived using add_row_id and rows align."""
    out_dir = tmp_path / "pipeline_run_hash_only"
    out_dir.mkdir()

    rec_dir = out_dir / "recognizer_output"
    rec_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "text_hash": "th_fallback",
                    "note_text": "Sample text",
                    "recognizer_results_json": "[]",
                }
            ]
        ),
        rec_dir / "part-0.parquet",
    )

    anon_dir = out_dir / "anonymizer_output"
    anon_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "text_hash": "th_fallback",
                    "anonymized_note_text": "Anonymized sample",
                    "anonymizer_results_json": "[]",
                }
            ]
        ),
        anon_dir / "part-0.parquet",
    )

    results = load_pipeline_output_dir(out_dir)
    assert len(results) == 1
    sample = next(iter(results.values()))
    assert sample["original_text"] == "Sample text"
    assert sample["modified_text"] == "Anonymized sample"
    assert sample["row_id"] is not None


def test_load_pipeline_output_recovers_text_from_transformer_partial_null(tmp_path: Path):
    """Verify that note_text is recovered per-row from transformer when recognizer has mixed populated/null text."""
    out_dir = tmp_path / "pipeline_run_partial"
    out_dir.mkdir()

    trans_dir = out_dir / "transformer_output"
    trans_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"row_id": "r1", "text_hash": "th1", "note_text": "Transformer Text 1"},
                {"row_id": "r2", "text_hash": "th2", "note_text": "Transformer Text 2"},
            ]
        ),
        trans_dir / "part-0.parquet",
    )

    rec_dir = out_dir / "recognizer_output"
    rec_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"row_id": "r1", "text_hash": "th1", "note_text": "Recognizer Text 1", "recognizer_results_json": "[]"},
                {"row_id": "r2", "text_hash": "th2", "note_text": None, "recognizer_results_json": "[]"},
            ]
        ),
        rec_dir / "part-0.parquet",
    )

    results = load_pipeline_output_dir(out_dir)
    assert "r1" in results and "r2" in results
    assert results["r1"]["original_text"] == "Recognizer Text 1"
    assert results["r2"]["original_text"] == "Transformer Text 2"


def test_load_pipeline_output_recovers_text_from_transformer(tmp_path: Path):
    """Verify that note_text is recovered from transformer_output if missing in recognizer."""
    out_dir = tmp_path / "pipeline_run_recovering"
    out_dir.mkdir()

    trans_dir = out_dir / "transformer_output"
    trans_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{"row_id": "r100", "text_hash": "th100", "note_text": "Recovered text"}]),
        trans_dir / "part-0.parquet",
    )

    rec_dir = out_dir / "recognizer_output"
    rec_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{"row_id": "r100", "text_hash": "th100", "recognizer_results_json": "[]"}]),
        rec_dir / "part-0.parquet",
    )

    results = load_pipeline_output_dir(out_dir)
    assert "r100" in results
    assert results["r100"]["original_text"] == "Recovered text"


def test_load_pipeline_output_duplicate_hashes_differing_order(tmp_path: Path):
    """Verify rows with identical text_hash but distinct row_ids align correctly despite differing stage order."""
    out_dir = tmp_path / "pipeline_run_dup_hash"
    out_dir.mkdir()

    rec_dir = out_dir / "recognizer_output"
    rec_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_id": "patient_A_note",
                    "text_hash": "shared_hash",
                    "note_text": "Common note text",
                    "recognizer_results_json": "[]",
                },
                {
                    "row_id": "patient_B_note",
                    "text_hash": "shared_hash",
                    "note_text": "Common note text",
                    "recognizer_results_json": "[]",
                },
            ]
        ),
        rec_dir / "part-0.parquet",
    )

    anon_dir = out_dir / "anonymizer_output"
    anon_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_id": "patient_B_note",
                    "text_hash": "shared_hash",
                    "anonymized_note_text": "Anon Patient B",
                    "anonymizer_results_json": "[]",
                },
                {
                    "row_id": "patient_A_note",
                    "text_hash": "shared_hash",
                    "anonymized_note_text": "Anon Patient A",
                    "anonymizer_results_json": "[]",
                },
            ]
        ),
        anon_dir / "part-0.parquet",
    )

    results = load_pipeline_output_dir(out_dir)
    assert results["patient_A_note"]["modified_text"] == "Anon Patient A"
    assert results["patient_B_note"]["modified_text"] == "Anon Patient B"


def test_load_pipeline_output_max_notes_pairs_stages_in_differing_order(tmp_path: Path):
    """Verify max_notes=1 yields one paired comparison when stage row order differs."""
    out_dir = tmp_path / "pipeline_run_order"
    out_dir.mkdir()
    (out_dir / "recognizer_output").mkdir()
    (out_dir / "anonymizer_output").mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"row_id": "A", "text_hash": "ha", "note_text": "Note A", "recognizer_results_json": "[]"},
                {"row_id": "B", "text_hash": "hb", "note_text": "Note B", "recognizer_results_json": "[]"},
            ]
        ),
        out_dir / "recognizer_output" / "part-0.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"row_id": "B", "text_hash": "hb", "anonymized_note_text": "Anon B", "anonymizer_results_json": "[]"},
                {"row_id": "A", "text_hash": "ha", "anonymized_note_text": "Anon A", "anonymizer_results_json": "[]"},
            ]
        ),
        out_dir / "anonymizer_output" / "part-0.parquet",
    )

    results = load_pipeline_output_dir(out_dir, max_notes=1)
    assert list(results) == ["A"]
    assert results["A"]["original_text"] == "Note A"
    assert results["A"]["modified_text"] == "Anon A"


def test_load_pipeline_output_derives_row_id_before_matching(tmp_path: Path):
    """Verify stages without row_id, or with null row_id values, pair on the derived row_id under max_notes."""
    out_dir = tmp_path / "pipeline_run_derived"
    out_dir.mkdir()
    (out_dir / "recognizer_output").mkdir()
    (out_dir / "anonymizer_output").mkdir()
    (out_dir / "transformer_output").mkdir()
    keys = [("ha", "p1"), ("hb", "p2")]
    derived = {h: add_row_id(pa.table({"text_hash": [h], "patient_id": [p]}))["row_id"].to_pylist()[0] for h, p in keys}

    pq.write_table(
        pa.Table.from_pylist(
            [
                {"text_hash": "ha", "patient_id": "p1", "recognizer_results_json": "[]"},
                {"text_hash": "hb", "patient_id": "p2", "recognizer_results_json": "[]"},
            ]
        ),
        out_dir / "recognizer_output" / "part-0.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"text_hash": "hb", "patient_id": "p2", "note_text": "Note B"},
                {"text_hash": "ha", "patient_id": "p1", "note_text": "Note A"},
            ]
        ),
        out_dir / "transformer_output" / "part-0.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"row_id": None, "text_hash": "hb", "patient_id": "p2", "anonymized_note_text": "Anon B"},
                {"row_id": None, "text_hash": "ha", "patient_id": "p1", "anonymized_note_text": "Anon A"},
            ]
        ),
        out_dir / "anonymizer_output" / "part-0.parquet",
    )

    results = load_pipeline_output_dir(out_dir, max_notes=1)
    assert list(results) == [derived["ha"]]
    assert results[derived["ha"]]["original_text"] == "Note A"
    assert results[derived["ha"]]["modified_text"] == "Anon A"


def test_load_pipeline_output_does_not_merge_patients_sharing_text_hash(tmp_path: Path):
    """Verify rows with one text_hash but different patients stay separate when row_id must be derived."""
    out_dir = tmp_path / "pipeline_run_patients"
    out_dir.mkdir()
    (out_dir / "recognizer_output").mkdir()
    (out_dir / "anonymizer_output").mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"text_hash": "same", "patient_id": "p1", "note_text": "Same", "recognizer_results_json": "[]"},
                {"text_hash": "same", "patient_id": "p2", "note_text": "Same", "recognizer_results_json": "[]"},
            ]
        ),
        out_dir / "recognizer_output" / "part-0.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"text_hash": "same", "patient_id": "p2", "anonymized_note_text": "Anon p2"},
                {"text_hash": "same", "patient_id": "p1", "anonymized_note_text": "Anon p1"},
            ]
        ),
        out_dir / "anonymizer_output" / "part-0.parquet",
    )

    results = load_pipeline_output_dir(out_dir)
    assert sorted(r["modified_text"] for r in results.values()) == ["Anon p1", "Anon p2"]
    assert len(results) == 2


def test_load_data_preserves_text_hash(tmp_path: Path):
    """Verify text_hash is preserved in pipeline and single parquet loads."""
    p_file = tmp_path / "test.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_id": "row_123",
                    "text_hash": "hash_abc",
                    "note_text": "Sample text",
                    "recognizer_results_json": "[]",
                }
            ]
        ),
        p_file,
    )

    results, _ = load_data(p_file)
    assert "row_123" in results
    assert results["row_123"]["text_hash"] == "hash_abc"


def test_load_anonymizer_only_parquet(tmp_path: Path):
    """Verify an anonymizer-only Parquet file populates modified_text and modified_phi."""
    p_file = tmp_path / "anon_only.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_id": "anon_1",
                    "anonymized_note_text": "Anonymized text content",
                    "anonymizer_results_json": json.dumps([{"entity_type": "PERSON", "start": 0, "end": 10}]),
                }
            ]
        ),
        p_file,
    )

    results, _ = load_data(p_file)
    assert "anon_1" in results
    sample = results["anon_1"]
    assert sample["original_text"] == ""
    assert sample["modified_text"] == "Anonymized text content"
    assert len(sample["modified_phi"]) == 1


def test_load_data_max_notes_cap(tmp_path: Path):
    """Verify max_notes bounds the number of loaded records."""
    p_file = tmp_path / "multi.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"row_id": f"r_{i}", "note_text": f"Note {i}"} for i in range(10)]),
        p_file,
    )

    results, _ = load_data(p_file, max_notes=3)
    assert len(results) == 3


def test_load_pipeline_preserves_failed_status(tmp_path: Path, monkeypatch):
    """Verify processing_status and stage_status_json are preserved across stages and shown as a warning."""
    out_dir = tmp_path / "pipeline_failed"
    out_dir.mkdir()
    (out_dir / "recognizer_output").mkdir()
    (out_dir / "anonymizer_output").mkdir()

    stage_json = json.dumps({"recognizer": {"status": "failed", "reason": "ValueError"}})
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_id": "r_fail",
                    "text_hash": "h_fail",
                    "note_text": "Note that failed recognition",
                    "recognizer_results_json": None,
                    "processing_status": "failed",
                    "stage_status_json": stage_json,
                }
            ]
        ),
        out_dir / "recognizer_output" / "part-0.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_id": "r_fail",
                    "text_hash": "h_fail",
                    "anonymized_note_text": None,
                    "anonymizer_results_json": None,
                    "processing_status": "failed",
                    "stage_status_json": stage_json,
                }
            ]
        ),
        out_dir / "anonymizer_output" / "part-0.parquet",
    )

    results = load_pipeline_output_dir(out_dir)
    rec = results["r_fail"]
    assert rec["processing_status"] == "failed"
    assert json.loads(rec["stage_status_json"]) == {"recognizer": {"status": "failed", "reason": "ValueError"}}

    warnings: list[str] = []
    infos: list[str] = []
    monkeypatch.setattr(ui.st, "divider", lambda: None)
    monkeypatch.setattr(ui.st, "subheader", lambda *_: None)
    monkeypatch.setattr(ui.st, "warning", warnings.append)
    monkeypatch.setattr(ui.st, "info", infos.append)

    ui._render_phi_table(
        rec["original_text"], rec["original_phi"], [], rec["processing_status"], rec["stage_status_json"]
    )
    assert len(warnings) == 2
    assert "Processing failed" in warnings[0]
    assert "unavailable" in warnings[1]
    assert infos == []


def test_render_phi_table_shows_detections_when_only_anonymizer_failed(tmp_path: Path, monkeypatch):
    """Verify successful recognition followed by anonymizer failure renders both warning and detections/filter info."""
    out_dir = tmp_path / "pipeline_anon_failed"
    out_dir.mkdir()
    (out_dir / "recognizer_output").mkdir()
    (out_dir / "anonymizer_output").mkdir()

    rec_stage_json = json.dumps({"recognizer": {"status": "success"}})
    anon_stage_json = json.dumps(
        {
            "recognizer": {"status": "success"},
            "anonymizer": {"status": "failed", "reason": "MRN:RuntimeError"},
        }
    )
    phi_ent = {"entity_type": "PERSON", "start": 0, "end": 4, "score": 0.99}

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_id": "r_anon_fail",
                    "text_hash": "h1",
                    "note_text": "John seen today",
                    "recognizer_results_json": json.dumps([phi_ent]),
                    "processing_status": "success",
                    "stage_status_json": rec_stage_json,
                }
            ]
        ),
        out_dir / "recognizer_output" / "part-0.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "row_id": "r_anon_fail",
                    "text_hash": "h1",
                    "anonymized_note_text": "Anon seen today",
                    "anonymizer_results_json": json.dumps([phi_ent]),
                    "processing_status": "failed",
                    "stage_status_json": anon_stage_json,
                }
            ]
        ),
        out_dir / "anonymizer_output" / "part-0.parquet",
    )

    results = load_pipeline_output_dir(out_dir)
    rec = results["r_anon_fail"]
    assert rec["processing_status"] == "failed"
    assert len(rec["original_phi"]) == 1

    warnings: list[str] = []
    infos: list[str] = []
    tables: list[pa.Table] = []
    monkeypatch.setattr(ui.st, "divider", lambda: None)
    monkeypatch.setattr(ui.st, "subheader", lambda *_: None)
    monkeypatch.setattr(ui.st, "warning", warnings.append)
    monkeypatch.setattr(ui.st, "info", infos.append)
    monkeypatch.setattr(ui.st, "dataframe", lambda tbl, **_: tables.append(tbl))

    # Case 1: detections pass filters -> warning + dataframe rendered
    ui._render_phi_table(
        rec["original_text"],
        rec["original_phi"],
        rec["original_phi"],
        rec["processing_status"],
        rec["stage_status_json"],
    )
    assert len(warnings) == 1
    assert "Processing failed" in warnings[0]
    assert len(tables) == 1
    assert infos == []

    # Case 2: filters hide all detections -> warning + "No PHI matching the selected filters."
    warnings.clear()
    tables.clear()
    ui._render_phi_table(
        rec["original_text"],
        rec["original_phi"],
        [],
        rec["processing_status"],
        rec["stage_status_json"],
    )
    assert len(warnings) == 1
    assert tables == []
    assert infos == ["No PHI matching the selected filters."]


def test_load_data_errors(tmp_path: Path):
    """Verify error handling on invalid or empty paths."""
    res, desc = load_data("")
    assert res == {}
    assert desc == "No path provided"

    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nonexistent")

    json_file = tmp_path / "test.json"
    json_file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data(json_file)

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No recognizable Parquet stages or Parquet files"):
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
