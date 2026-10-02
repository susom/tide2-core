"""Unit tests for Parquet-only pipeline input and pandas removal from local_runner."""

import hashlib
import inspect
import json

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest
import ray

import tide2.runner.local_runner as lr
from tide2.runner.local_runner import DEFAULT_ROW_ID_PATIENT_ID
from tide2.runner.local_runner import LocalJobRunner
from tide2.runner.local_runner import PipelineInputInfo
from tide2.runner.local_runner import _inspect_pipeline_input
from tide2.runner.local_runner import _resolve_merged_batch
from tide2.runner.local_runner import add_row_id
from tide2.runner.local_runner import normalize_source_batch
from tide2.utils.text_processing import compute_text_hash

# ---------------------------------------------------------------------------
# 1 & 2. System boundary validation: DataFrame rejection and path resolution
# ---------------------------------------------------------------------------


def test_run_pipeline_raises_type_error_on_dataframe(tmp_path):
    """Passing a pandas.DataFrame to run_pipeline must raise TypeError directing to to_parquet."""
    runner = LocalJobRunner()
    df = pd.DataFrame({"note_text": ["test note"]})
    with pytest.raises(TypeError, match=r"DataFrame input to run_pipeline was removed.*df\.to_parquet"):
        runner.run_pipeline(
            input_path=df,
            output_dir=str(tmp_path / "out"),
            model_name="test-model",
        )


def test_run_pipeline_raises_value_error_on_unresolved_or_non_parquet(tmp_path):
    """Non-existent paths or non-parquet files must raise ValueError."""
    runner = LocalJobRunner()
    # Non-existent path
    with pytest.raises(ValueError, match=r"No input files found matching"):
        runner.run_pipeline(
            input_path=str(tmp_path / "non_existent.parquet"),
            output_dir=str(tmp_path / "out"),
            model_name="test-model",
        )

    # Non-parquet file
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hello")
    with pytest.raises(ValueError, match=r"Resolved file does not end with \.parquet"):
        runner.run_pipeline(
            input_path=str(txt_file),
            output_dir=str(tmp_path / "out"),
            model_name="test-model",
        )


# ---------------------------------------------------------------------------
# 3. _inspect_pipeline_input (metadata only, schema validation)
# ---------------------------------------------------------------------------


def test_inspect_pipeline_input_metadata_only(tmp_path, monkeypatch):
    """_inspect_pipeline_input reads only schema and row count, never data pages."""
    f = tmp_path / "data.parquet"
    tbl = pa.table(
        {
            "note_text": ["note 1", "note 2", "note 3"],
            "patient_id": ["pat_1", "pat_2", "pat_3"],
            "row_id": ["r1", "r2", "r3"],
        }
    )
    pq.write_table(tbl, f)

    orig_dataset = pads.dataset

    class DatasetDataGuard:
        def __init__(self, ds):
            self._ds = ds

        def to_table(self, *args, **kwargs):
            raise AssertionError("to_table() reads data pages and must not be called by _inspect_pipeline_input")

        def __getattr__(self, name):
            return getattr(self._ds, name)

    def guarded_dataset(*args, **kwargs):
        ds = orig_dataset(*args, **kwargs)
        return DatasetDataGuard(ds)

    monkeypatch.setattr(pads, "dataset", guarded_dataset)

    info = _inspect_pipeline_input([str(f)])
    assert isinstance(info, PipelineInputInfo)
    assert info.num_rows == 3
    assert info.has_row_id is True
    assert info.has_patient_id is True
    assert info.has_text_hash is False
    assert info.is_patient_id_numeric is False
    assert "note_text" in info.columns


def test_inspect_pipeline_input_lowercase_collision(tmp_path):
    """Duplicate column names differing only in casing raise ValueError."""
    f = tmp_path / "collision.parquet"
    # Create Arrow schema with case collision
    f1 = pa.field("Note_Text", pa.string())
    f2 = pa.field("note_text", pa.string())
    schema = pa.schema([f1, f2])
    tbl = pa.Table.from_arrays([pa.array(["a"]), pa.array(["b"])], schema=schema)
    pq.write_table(tbl, f)

    with pytest.raises(ValueError, match=r"Duplicate column after lowercasing"):
        _inspect_pipeline_input([str(f)])


def test_inspect_pipeline_input_missing_note_text(tmp_path):
    """Input without note_text raises ValueError."""
    f = tmp_path / "no_note.parquet"
    tbl = pa.table({"patient_id": ["p1"]})
    pq.write_table(tbl, f)

    with pytest.raises(ValueError, match=r"Input data must contain a 'note_text' column"):
        _inspect_pipeline_input([str(f)])


def test_inspect_pipeline_input_patient_id_types(tmp_path):
    """Integer and float patient_id are accepted (is_patient_id_numeric=True), while other types raise TypeError."""
    # Integer patient_id
    f_int = tmp_path / "int_pat.parquet"
    pq.write_table(pa.table({"note_text": ["n1"], "patient_id": [12345]}), f_int)
    info_int = _inspect_pipeline_input([str(f_int)])
    assert info_int.has_patient_id is True
    assert info_int.is_patient_id_numeric is True

    # Float patient_id
    f_flt = tmp_path / "flt_pat.parquet"
    pq.write_table(pa.table({"note_text": ["n1"], "patient_id": [123.45]}), f_flt)
    info_flt = _inspect_pipeline_input([str(f_flt)])
    assert info_flt.has_patient_id is True
    assert info_flt.is_patient_id_numeric is True

    # Unsupported type (date or binary)
    f_bad = tmp_path / "bad_pat.parquet"
    pq.write_table(pa.table({"note_text": ["n1"], "patient_id": [b"raw_bytes"]}), f_bad)
    with pytest.raises(TypeError, match=r"Column 'patient_id' has unsupported type"):
        _inspect_pipeline_input([str(f_bad)])


# ---------------------------------------------------------------------------
# 3b. normalize_source_batch
# ---------------------------------------------------------------------------


def test_normalize_source_batch():
    """normalize_source_batch lowercases columns, derives text_hash/row_id if absent, and never writes patient_id."""
    tbl = pa.table(
        {
            "NOTE_TEXT": ["Clinical report"],
            "JITTER": [5],
        }
    )
    normalized = normalize_source_batch(tbl)
    assert set(normalized.column_names) == {"note_text", "jitter", "text_hash", "row_id"}
    assert "patient_id" not in normalized.column_names

    expected_hash = compute_text_hash("Clinical report")
    assert normalized["text_hash"].to_pylist() == [expected_hash]
    expected_row_id = hashlib.sha256(f"{expected_hash}:{DEFAULT_ROW_ID_PATIENT_ID}".encode()).hexdigest()
    assert normalized["row_id"].to_pylist() == [expected_row_id]

    # Pre-existing row_id and text_hash must be left untouched
    tbl_custom = pa.table(
        {
            "note_text": ["Clinical report"],
            "text_hash": ["custom_hash"],
            "row_id": ["custom_row_id"],
        }
    )
    res_custom = normalize_source_batch(tbl_custom)
    assert res_custom["text_hash"].to_pylist() == ["custom_hash"]
    assert res_custom["row_id"].to_pylist() == ["custom_row_id"]


# ---------------------------------------------------------------------------
# 4. add_row_id golden-value parity & properties
# ---------------------------------------------------------------------------


def test_add_row_id_golden_parity():
    """add_row_id matches the historical pandas formula for string, int, null, and missing patient_id."""
    text_hashes = ["hash_abc", "hash_def", "hash_ghi", "hash_jkl"]
    # 1. String patient_id
    # 2. Integer patient_id
    # 3. Null patient_id
    # 4. Alphanumeric patient_id
    patient_ids_series = pd.Series(["pat_123", 456, None, "MRN-987X"])
    old_formula_pandas = (
        (pd.Series(text_hashes) + ":" + patient_ids_series.fillna("None").astype(str))
        .apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
        .tolist()
    )

    # Arrow table with mixed string representations / nulls
    tbl_str = pa.table(
        {
            "text_hash": text_hashes,
            "patient_id": ["pat_123", "456", None, "MRN-987X"],
        }
    )
    res = add_row_id(tbl_str)
    assert res["row_id"].to_pylist() == old_formula_pandas

    # Table with integer patient_id
    tbl_int = pa.table(
        {
            "text_hash": ["hash_1"],
            "patient_id": [456],
        }
    )
    res_int = add_row_id(tbl_int)
    # Crucial property: patient_id column type is preserved as int!
    assert pa.types.is_integer(res_int["patient_id"].type)
    expected_int_row_id = hashlib.sha256(b"hash_1:456").hexdigest()
    assert res_int["row_id"].to_pylist() == [expected_int_row_id]

    # Table without patient_id column
    tbl_no_pid = pa.table({"text_hash": ["hash_none"]})
    res_no_pid = add_row_id(tbl_no_pid)
    expected_none_row_id = hashlib.sha256(f"hash_none:{DEFAULT_ROW_ID_PATIENT_ID}".encode()).hexdigest()
    assert res_no_pid["row_id"].to_pylist() == [expected_none_row_id]
    assert "patient_id" not in res_no_pid.column_names

    # Null patient_id stays null (default "None" is NOT written into patient_id column)
    tbl_null_pid = pa.table({"text_hash": ["h"], "patient_id": [None]})
    res_null = add_row_id(tbl_null_pid)
    assert res_null["patient_id"].to_pylist() == [None]


# ---------------------------------------------------------------------------
# 5. Input forms resolution
# ---------------------------------------------------------------------------


def test_input_forms_resolution(tmp_path):
    """Single file, directory, glob, and list of files all resolve properly."""
    sub = tmp_path / "notes_dir"
    sub.mkdir()
    f1 = sub / "part-1.parquet"
    f2 = sub / "part-2.parquet"
    tbl = pa.table({"note_text": ["text"]})
    pq.write_table(tbl, f1)
    pq.write_table(tbl, f2)

    # 1. Single file
    assert lr.resolve_input_files(str(f1)) == [str(f1)]
    # 2. Directory
    assert sorted(lr.resolve_input_files(str(sub))) == sorted([str(f1), str(f2)])
    # 3. Glob
    assert sorted(lr.resolve_input_files(f"{sub}/*.parquet")) == sorted([str(f1), str(f2)])
    # 4. List of files
    assert lr.resolve_input_files([str(f1), str(f2)]) == [str(f1), str(f2)]


# ---------------------------------------------------------------------------
# 6. Merge mode (_resolve_merged_batch & 1:1 join without fan-out)
# ---------------------------------------------------------------------------


def test_resolve_merged_batch_unit():
    """_resolve_merged_batch correctly merges regex and LLM spans using longest_wins."""
    r1 = json.dumps([{"entity_type": "PERSON", "start": 0, "end": 4, "score": 0.8}])
    l1 = json.dumps([{"entity_type": "PERSON", "start": 0, "end": 9, "score": 0.95}])

    tbl = pa.table(
        {
            "row_id": ["r1", "r2", "r3"],
            "text_hash_regex": ["h1", "h2", None],
            "text_hash_llm": ["h1", None, "h3"],
            "results_regex": [r1, r1, None],
            "results_llm": [l1, None, l1],
            "note_text": ["text1", "text2", "text3"],
            "patient_id": ["p1", "p2", "p3"],
        }
    )

    resolved = _resolve_merged_batch(tbl)
    assert "results_regex" not in resolved.column_names
    assert "results_llm" not in resolved.column_names
    assert "text_hash" in resolved.column_names
    assert resolved["text_hash"].to_pylist() == ["h1", "h2", "h3"]

    # For row 1, longest_wins keeps (0..9)
    res_1 = json.loads(resolved["recognizer_results_json"].to_pylist()[0])
    assert len(res_1) == 1
    assert res_1[0]["start"] == 0 and res_1[0]["end"] == 9
    assert resolved["entity_count"].to_pylist()[0] == 1


# ---------------------------------------------------------------------------
# 7. Visualizer JSON helpers with pyarrow
# ---------------------------------------------------------------------------


def test_visualizer_json_helpers(tmp_path):
    """_write_recognizer_json_files and _write_anonymizer_json_files write valid JSON."""
    runner = LocalJobRunner()
    rec_dir = tmp_path / "rec_out"
    rec_dir.mkdir()
    trans_dir = tmp_path / "trans_out"
    trans_dir.mkdir()
    anon_dir = tmp_path / "anon_out"
    anon_dir.mkdir()

    # Recognizer output with note_text
    tbl_rec = pa.table(
        {
            "text_hash": ["h1"],
            "note_text": ["Sample note"],
            "recognizer_results_json": [json.dumps([{"entity_type": "NAME", "start": 0, "end": 6, "score": 0.9}])],
        }
    )
    pq.write_table(tbl_rec, rec_dir / "part-0.parquet")

    # Anonymizer output
    tbl_anon = pa.table(
        {
            "text_hash": ["h1"],
            "anonymized_note_text": ["<NAME> note"],
            "anonymizer_results_json": [json.dumps([{"entity_type": "NAME", "start": 0, "end": 6}])],
        }
    )
    pq.write_table(tbl_anon, anon_dir / "part-0.parquet")

    runner._write_visualizer_json(
        output_path=tmp_path,
        transformer_output_path=trans_dir,
        recognizer_output_path=rec_dir,
        anonymizer_output_path=anon_dir,
    )

    rec_json = tmp_path / "cli_recognizer_json" / "h1.json"
    assert rec_json.exists()
    rec_data = json.loads(rec_json.read_text())
    assert rec_data["key"] == "h1"
    assert rec_data["value"] == "Sample note"
    assert len(rec_data["recognizer_results"]) == 1

    anon_json = tmp_path / "cli_anonymizer_json" / "h1.json"
    assert anon_json.exists()
    anon_data = json.loads(anon_json.read_text())
    assert anon_data["text"] == "<NAME> note"
    assert len(anon_data["items"]) == 1


# ---------------------------------------------------------------------------
# 3c & 3d. Intermediates removal and Section 2c materialization pass
# ---------------------------------------------------------------------------


def test_no_intermediate_parquet_files_written_in_discrete(tmp_path, monkeypatch):
    """01_transformer_input.parquet and 03_llm_recognizer_input.parquet are never written."""
    f = tmp_path / "in.parquet"
    pq.write_table(pa.table({"note_text": ["text1"], "row_id": ["r1"]}), f)

    out_dir = tmp_path / "pipeline_out"
    runner = LocalJobRunner()

    monkeypatch.setattr(runner, "_init_ray", lambda: None)
    monkeypatch.setattr(runner, "_apply_pipeline_recommendations", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "run_transformer", lambda **_k: {"transformer": True})
    monkeypatch.setattr(runner, "run_recognition", lambda **_k: {"recognizer": True})
    monkeypatch.setattr(runner, "run_anonymization", lambda **_k: {"anonymizer": True})

    runner.run_pipeline(
        input_path=str(f),
        output_dir=str(out_dir),
        model_name="test-model",
    )

    assert not (out_dir / "01_transformer_input.parquet").exists()
    assert not (out_dir / "03_llm_recognizer_input.parquet").exists()


def test_materialization_pass_section_2c(tmp_path, monkeypatch):
    """Section 2c pass materializes 01_normalized_input when checkpointing=True and row_id is absent."""
    f = tmp_path / "in.parquet"
    pq.write_table(pa.table({"note_text": ["note without row id"]}), f)

    out_dir = tmp_path / "pipeline_out"
    runner = LocalJobRunner()

    stages_called = {}
    monkeypatch.setattr(runner, "_init_ray", lambda: None)
    monkeypatch.setattr(runner, "_apply_pipeline_recommendations", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner,
        "run_transformer",
        lambda **k: (stages_called.setdefault("transformer", k), {"transformer": True})[1],
    )
    monkeypatch.setattr(runner, "run_recognition", lambda **_k: {"recognizer": True})
    monkeypatch.setattr(runner, "run_anonymization", lambda **_k: {"anonymizer": True})

    # Case 1: enable_checkpoint=True, row_id absent -> materialization runs
    runner.run_pipeline(
        input_path=str(f),
        output_dir=str(out_dir),
        model_name="test-model",
        transformer_kwargs={"enable_checkpoint": True},
    )

    norm_dir = out_dir / "01_normalized_input"
    assert norm_dir.exists()
    assert (norm_dir / "_SUCCESS").exists()
    assert stages_called["transformer"]["input_path"] == str(norm_dir)
    assert stages_called["transformer"]["_normalize"] is False
    assert stages_called["transformer"]["_id_column"] == "row_id"

    # Case 2: Resume reuses existing normalized dir with _SUCCESS
    stages_called.clear()
    runner.run_pipeline(
        input_path=str(f),
        output_dir=str(out_dir),
        model_name="test-model",
        transformer_kwargs={"enable_checkpoint": True},
    )
    assert stages_called["transformer"]["input_path"] == str(norm_dir)

    # Case 3: enable_checkpoint=False -> no materialization pass
    out_dir_no_chk = tmp_path / "out_no_chk"
    stages_called.clear()
    runner.run_pipeline(
        input_path=str(f),
        output_dir=str(out_dir_no_chk),
        model_name="test-model",
        transformer_kwargs={"enable_checkpoint": False},
        recognizer_kwargs={"enable_checkpoint": False},
        anonymizer_kwargs={"enable_checkpoint": False},
    )
    assert not (out_dir_no_chk / "01_normalized_input").exists()
    assert stages_called["transformer"]["_normalize"] is True


def test_standalone_run_anonymization_row_id(tmp_path, monkeypatch):
    """Standalone run_anonymization materializes 00_normalized_anonymizer_input only when checkpointing on and row_id absent."""
    runner = LocalJobRunner()
    salt_file = tmp_path / "salt.bin"
    key_file = tmp_path / "key.bin"
    salt_file.write_text("00" * 32)
    key_file.write_text("11" * 32)

    # Case 1: input already has row_id
    f_with_row_id = tmp_path / "with_row_id.parquet"
    pq.write_table(
        pa.table(
            {
                "text_hash": ["h1"],
                "note_text": ["text"],
                "recognizer_results_json": ["[]"],
                "row_id": ["r1"],
            }
        ),
        f_with_row_id,
    )
    out_1 = tmp_path / "out_1"
    res_1 = runner.run_anonymization(
        input_path=str(f_with_row_id),
        output_path=str(out_1),
        salt_path=str(salt_file),
        key_path=str(key_file),
        dry_run=True,
    )
    assert "row_id" in res_1["columns_detected"]
    assert not (out_1 / "00_normalized_anonymizer_input").exists()

    # Case 2: input lacks row_id and checkpointing is on -> materializes with _SUCCESS
    f_no_row_id = tmp_path / "no_row_id.parquet"
    pq.write_table(
        pa.table(
            {
                "text_hash": ["h2"],
                "note_text": ["text 2"],
                "recognizer_results_json": ["[]"],
            }
        ),
        f_no_row_id,
    )
    out_2 = tmp_path / "out_2"
    res_2 = runner.run_anonymization(
        input_path=str(f_no_row_id),
        output_path=str(out_2),
        salt_path=str(salt_file),
        key_path=str(key_file),
        enable_checkpoint=True,
        dry_run=True,
    )
    norm_anon_dir = out_2 / "00_normalized_anonymizer_input"
    assert norm_anon_dir.exists()
    assert (norm_anon_dir / "_SUCCESS").exists()
    assert "row_id" in res_2["columns_detected"]


def test_merge_mode_ray_join_no_fanout(tmp_path):
    """Merge mode joins on row_id: duplicate text_hashes for distinct patients do not fan out."""
    rec_out = tmp_path / "04a_regex"
    rec_out.mkdir()
    llm_out = tmp_path / "03b_llm"
    llm_out.mkdir()

    # Two patients with same note text (same text_hash, different row_ids)
    r1 = json.dumps([{"entity_type": "PERSON", "start": 0, "end": 4, "score": 0.8}])
    l1 = json.dumps([{"entity_type": "PERSON", "start": 0, "end": 9, "score": 0.95}])

    tbl_regex = pa.table(
        {
            "row_id": ["row_pat1", "row_pat2"],
            "text_hash": ["same_hash", "same_hash"],
            "patient_id": ["pat_1", "pat_2"],
            "note_text": ["John Doe exam", "John Doe exam"],
            "recognizer_results_json": [r1, r1],
        }
    )
    pq.write_table(tbl_regex, rec_out / "part-0.parquet")

    tbl_llm = pa.table(
        {
            "row_id": ["row_pat1", "row_pat2"],
            "text_hash": ["same_hash", "same_hash"],
            "recognizer_results_json": [l1, l1],
        }
    )
    pq.write_table(tbl_llm, llm_out / "part-0.parquet")

    merged_out = tmp_path / "04_recognizer_output"

    regex_ds = ray.data.read_parquet(str(rec_out)).select_columns(
        ["row_id", "text_hash", "recognizer_results_json", "patient_id", "note_text"]
    )
    regex_ds = regex_ds.rename_columns(
        {
            "recognizer_results_json": "results_regex",
            "text_hash": "text_hash_regex",
        }
    )

    llm_ds = ray.data.read_parquet(str(llm_out)).select_columns(["row_id", "recognizer_results_json", "text_hash"])
    llm_ds = llm_ds.rename_columns(
        {
            "recognizer_results_json": "results_llm",
            "text_hash": "text_hash_llm",
        }
    )

    joined_ds = regex_ds.join(
        llm_ds,
        join_type="full_outer",
        num_partitions=2,
        on=("row_id",),
        aggregator_ray_remote_args={"num_cpus": 0.25},
    )
    merged_ds = joined_ds.map_batches(_resolve_merged_batch, batch_format="pyarrow")
    merged_ds.write_parquet(str(merged_out), compression="zstd")

    out_tbl = pads.dataset(list(merged_out.glob("*.parquet"))).to_table()
    # Parity check: exactly 2 output rows (NO fan-out to 4 rows!)
    assert len(out_tbl) == 2
    row_ids = sorted(out_tbl["row_id"].to_pylist())
    assert row_ids == ["row_pat1", "row_pat2"]
    patient_ids = sorted(out_tbl["patient_id"].to_pylist())
    assert patient_ids == ["pat_1", "pat_2"]


# ---------------------------------------------------------------------------
# 8. Guard test: No pandas in local_runner.py
# ---------------------------------------------------------------------------


def test_guard_no_pandas_in_local_runner():
    """Ensure import pandas or pd does not exist in local_runner source."""
    source = inspect.getsource(lr)
    assert "import pandas" not in source
    assert "from pandas" not in source
    assert "pd." not in source
