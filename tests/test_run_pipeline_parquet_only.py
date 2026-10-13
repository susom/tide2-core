"""Unit tests for Parquet-only pipeline input and pandas removal from local_runner."""

import hashlib
import inspect
import json
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest
import ray

import tide2.runner.local_runner as lr
from tide2.runner.cli import main as cli_main
from tide2.runner.local_runner import DEFAULT_ROW_ID_PATIENT_ID
from tide2.runner.local_runner import LocalJobRunner
from tide2.runner.local_runner import PipelineInputInfo
from tide2.runner.local_runner import _inspect_pipeline_input
from tide2.runner.local_runner import _resolve_merged_batch
from tide2.runner.local_runner import add_row_id
from tide2.runner.local_runner import normalize_source_batch
from tide2.runner.utils import write_visualizer_json
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
    """Integer and float patient_id are accepted, while other types raise TypeError."""
    # Integer patient_id
    f_int = tmp_path / "int_pat.parquet"
    pq.write_table(pa.table({"note_text": ["n1"], "patient_id": [12345]}), f_int)
    info_int = _inspect_pipeline_input([str(f_int)])
    assert info_int.has_patient_id is True

    # Float patient_id
    f_flt = tmp_path / "flt_pat.parquet"
    pq.write_table(pa.table({"note_text": ["n1"], "patient_id": [123.45]}), f_flt)
    info_flt = _inspect_pipeline_input([str(f_flt)])
    assert info_flt.has_patient_id is True

    # Unsupported type (date or binary)
    f_bad = tmp_path / "bad_pat.parquet"
    pq.write_table(pa.table({"note_text": ["n1"], "patient_id": [b"raw_bytes"]}), f_bad)
    with pytest.raises(TypeError, match=r"Column 'patient_id' has unsupported type"):
        _inspect_pipeline_input([str(f_bad)])


@pytest.mark.parametrize("with_text_hash", [False, True])
@pytest.mark.parametrize(
    "bad_values",
    [[b"raw bytes"], [12345], [1.5]],
    ids=["binary", "integer", "float"],
)
def test_inspect_pipeline_input_rejects_non_string_note_text(tmp_path, bad_values, with_text_hash):
    """A non-string note_text fails the same way with or without a supplied text_hash."""
    columns = {"note_text": bad_values}
    if with_text_hash:
        columns["text_hash"] = ["h1"]
    f = tmp_path / "bad_note.parquet"
    pq.write_table(pa.table(columns), f)

    with pytest.raises(TypeError, match=r"Column 'note_text' has unsupported type"):
        _inspect_pipeline_input([str(f)])


@pytest.mark.parametrize("note_type", [pa.string(), pa.large_string(), pa.null()])
def test_inspect_pipeline_input_accepts_string_note_text(tmp_path, note_type):
    """String, large_string and an all-null note_text column are accepted."""
    f = tmp_path / "ok_note.parquet"
    pq.write_table(pa.table({"note_text": pa.array([None], type=note_type)}), f)

    assert "note_text" in _inspect_pipeline_input([str(f)]).columns


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


@pytest.mark.parametrize(
    ("regex_pid", "llm_pid", "expected"),
    [
        ([None, None], ["7", "8"], ["7", "8"]),
        (["7", "8"], [None, None], ["7", "8"]),
        (["a", None], [None, "8"], ["a", "8"]),
    ],
)
def test_resolve_merged_batch_coalesces_patient_id(regex_pid, llm_pid, expected):
    """The regex and LLM copies of patient_id (both string) merge into one string column."""
    tbl = pa.table(
        {
            "row_id": ["r1", "r2"],
            "text_hash_regex": ["h1", "h2"],
            "text_hash_llm": ["h1", "h2"],
            "results_regex": ["[]", "[]"],
            "results_llm": ["[]", "[]"],
            "patient_id_regex": pa.array(regex_pid, type=pa.string()),
            "patient_id_llm": pa.array(llm_pid, type=pa.string()),
        }
    )

    resolved = _resolve_merged_batch(tbl)

    assert resolved["patient_id"].type == pa.string()
    assert resolved["patient_id"].to_pylist() == expected


# ---------------------------------------------------------------------------
# 7. Visualizer JSON helpers with pyarrow
# ---------------------------------------------------------------------------


def _write_visualizer_stage_folders(root, *, recognizer_folders=("recognizer",), with_anonymizer=True):
    for stage in ("transformer", *recognizer_folders):
        folder = root / lr.STAGE_DIRS[stage]
        folder.mkdir(parents=True)
        results = [{"entity_type": "NAME", "start": 0, "end": 6, "score": 0.9}] if stage != "transformer" else []
        columns = {"text_hash": ["h1"], "note_text": ["Sample note"]}
        if stage != "transformer":
            columns["recognizer_results_json"] = [json.dumps(results)]
            if stage == "merged_recognizer":
                columns["recognizer_results_json"] = [json.dumps(results * 2)]
        pq.write_table(pa.table(columns), folder / "part-0.parquet")
    if with_anonymizer:
        folder = root / lr.STAGE_DIRS["anonymizer"]
        folder.mkdir(parents=True)
        tbl_anon = pa.table(
            {
                "text_hash": ["h1"],
                "anonymized_note_text": ["<NAME> note"],
                "anonymizer_results_json": [json.dumps([{"entity_type": "NAME", "start": 0, "end": 6}])],
            }
        )
        pq.write_table(tbl_anon, folder / "part-0.parquet")


def test_write_visualizer_json_writes_both_folders(tmp_path):
    _write_visualizer_stage_folders(tmp_path)

    counts = write_visualizer_json(tmp_path)

    assert counts == {"recognizer_json": 1, "anonymizer_json": 1}
    rec_data = json.loads((tmp_path / "cli_recognizer_json" / "h1.json").read_text())
    assert rec_data["key"] == "h1"
    assert rec_data["value"] == "Sample note"
    assert len(rec_data["recognizer_results"]) == 1
    anon_data = json.loads((tmp_path / "cli_anonymizer_json" / "h1.json").read_text())
    assert anon_data["text"] == "<NAME> note"
    assert len(anon_data["items"]) == 1


def test_write_visualizer_json_prefers_the_merged_recognizer_folder(tmp_path):
    _write_visualizer_stage_folders(tmp_path, recognizer_folders=("recognizer", "llm_recognizer", "merged_recognizer"))

    write_visualizer_json(tmp_path)

    rec_data = json.loads((tmp_path / "cli_recognizer_json" / "h1.json").read_text())
    assert len(rec_data["recognizer_results"]) == 2


def test_write_visualizer_json_after_a_streamed_run_writes_only_the_anonymizer_json(tmp_path):
    _write_visualizer_stage_folders(tmp_path, recognizer_folders=())
    shutil.rmtree(tmp_path / lr.STAGE_DIRS["transformer"])

    counts = write_visualizer_json(tmp_path, json_dir=tmp_path / "viz")

    assert counts == {"recognizer_json": 0, "anonymizer_json": 1}
    assert (tmp_path / "viz" / "cli_anonymizer_json" / "h1.json").exists()
    assert not (tmp_path / "viz" / "cli_recognizer_json").exists()


def test_write_visualizer_json_keeps_file_names_inside_the_json_folders(tmp_path):
    run = tmp_path / "run"
    for stage in ("recognizer", "anonymizer"):
        folder = run / lr.STAGE_DIRS[stage]
        folder.mkdir(parents=True)
        columns = {"text_hash": ["../../escape", "/tmp/absolute"], "note_text": ["a", "b"]}
        if stage == "recognizer":
            columns["recognizer_results_json"] = ["[]", "[]"]
        else:
            columns["anonymized_note_text"] = ["a", "b"]
            columns["anonymizer_results_json"] = ["[]", "[]"]
        pq.write_table(pa.table(columns), folder / "part-0.parquet")

    counts = write_visualizer_json(run)

    assert counts == {"recognizer_json": 2, "anonymizer_json": 2}
    written = sorted(p.relative_to(run).parent.as_posix() for p in run.rglob("*.json"))
    assert written == ["cli_anonymizer_json"] * 2 + ["cli_recognizer_json"] * 2
    assert not (tmp_path / "escape.json").exists()
    recognizer = [json.loads(p.read_text()) for p in (run / "cli_recognizer_json").glob("*.json")]
    assert {r["text_hash"] for r in recognizer} == {"../../escape", "/tmp/absolute"}
    anonymizer_ids = {p.stem for p in (run / "cli_anonymizer_json").glob("*.json")}
    assert {r["key"] for r in recognizer} == anonymizer_ids


def test_write_visualizer_json_keeps_rows_that_share_a_text_hash(tmp_path):
    run = tmp_path / "run"
    for stage in ("recognizer", "anonymizer"):
        folder = run / lr.STAGE_DIRS[stage]
        folder.mkdir(parents=True)
        columns = {"text_hash": ["same", "same"], "row_id": ["r1", "r2"], "note_text": ["a", "a"]}
        if stage == "recognizer":
            columns["recognizer_results_json"] = ["[]", "[]"]
        else:
            columns["anonymized_note_text"] = ["x", "y"]
            columns["anonymizer_results_json"] = ["[]", "[]"]
        pq.write_table(pa.table(columns), folder / "part-0.parquet")

    counts = write_visualizer_json(run)

    assert counts == {"recognizer_json": 2, "anonymizer_json": 2}
    files = sorted(p.name for p in (run / "cli_anonymizer_json").glob("*.json"))
    assert files == ["r1.json", "r2.json"]
    assert json.loads((run / "cli_anonymizer_json" / "r2.json").read_text())["text"] == "y"
    assert json.loads((run / "cli_recognizer_json" / "r2.json").read_text())["key"] == "r2"


def test_write_visualizer_json_needs_a_local_folder(tmp_path):
    with pytest.raises(ValueError, match="local output_dir"):
        write_visualizer_json("gs://bucket/run")
    with pytest.raises(ValueError, match="not a directory"):
        write_visualizer_json(tmp_path / "missing")


def test_visualizer_json_command_writes_the_files(tmp_path):
    _write_visualizer_stage_folders(tmp_path)

    cli_main(["visualizer-json", "-o", str(tmp_path)])

    assert (tmp_path / "cli_recognizer_json" / "h1.json").exists()
    assert (tmp_path / "cli_anonymizer_json" / "h1.json").exists()


# ---------------------------------------------------------------------------
# 3c & 3d. Intermediates removal and Section 2c materialization pass
# ---------------------------------------------------------------------------


def test_no_intermediate_parquet_files_written_in_discrete(tmp_path, monkeypatch):
    """The runner writes no copy of the input; only stage folders appear under the output directory."""
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

    assert not list(out_dir.glob("*.parquet"))
    assert not list(out_dir.glob("*input*"))


def test_pipeline_does_not_materialize_normalized_input(tmp_path, monkeypatch):
    """With checkpointing on and no row_id, the first stage gets the original input and normalizes it in the read."""
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

    for enable_checkpoint in (True, False):
        stages_called.clear()
        out = out_dir / str(enable_checkpoint)
        runner.run_pipeline(
            input_path=str(f),
            output_dir=str(out),
            model_name="test-model",
            transformer_kwargs={"enable_checkpoint": enable_checkpoint},
        )
        assert not list(out.glob("*input*"))
        assert stages_called["transformer"]["input_path"] == [str(f)]
        assert "_normalize" not in stages_called["transformer"]
        assert stages_called["transformer"]["_id_column"] == "row_id"


def test_standalone_run_anonymization_row_id(tmp_path):
    """Standalone run_anonymization never writes a normalized copy, with or without row_id."""
    runner = LocalJobRunner()
    salt_file = tmp_path / "salt.bin"
    key_file = tmp_path / "key.bin"
    salt_file.write_text("00" * 32)
    key_file.write_text("11" * 32)

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
    f_no_row_id = tmp_path / "no_row_id.parquet"
    pq.write_table(
        pa.table({"text_hash": ["h2"], "note_text": ["text 2"], "recognizer_results_json": ["[]"]}),
        f_no_row_id,
    )

    for name, source, expect_row_id in (
        ("out_1", f_with_row_id, True),
        ("out_2", f_no_row_id, False),
    ):
        out = tmp_path / name
        res = runner.run_anonymization(
            input_path=str(source),
            output_path=str(out),
            salt_path=str(salt_file),
            key_path=str(key_file),
            enable_checkpoint=True,
            dry_run=True,
        )
        assert ("row_id" in res["columns_detected"]) is expect_row_id
        assert not list(out.glob("*input*"))


def test_run_anonymization_takes_the_keys_as_files_or_as_hex(tmp_path):
    runner = LocalJobRunner()
    source = tmp_path / "in.parquet"
    pq.write_table(pa.table({"note_text": ["text"], "recognizer_results_json": ["[]"]}), source)
    salt_file = tmp_path / "salt.bin"
    key_file = tmp_path / "key.bin"
    salt_file.write_text("00" * 32)
    key_file.write_text("11" * 32)

    by_hex = runner.run_anonymization(
        str(source), str(tmp_path / "a"), salt_hex="00" * 32, key_hex="11" * 32, dry_run=True
    )
    by_file = runner.run_anonymization(
        str(source), str(tmp_path / "b"), salt_path=str(salt_file), key_path=str(key_file), dry_run=True
    )
    assert by_hex["keys_loaded"] and by_file["keys_loaded"]

    with pytest.raises(ValueError, match="salt_path and key_path, or salt_hex and key_hex"):
        runner.run_anonymization(str(source), str(tmp_path / "c"), salt_hex="00" * 32, dry_run=True)
    with pytest.raises(ValueError, match="32 bytes"):
        runner.run_anonymization(str(source), str(tmp_path / "d"), salt_hex="00", key_hex="11" * 32, dry_run=True)

    mixed = [
        {"salt_path": str(salt_file), "key_path": str(key_file), "salt_hex": "00" * 32},
        {"salt_path": str(salt_file), "key_path": str(key_file), "key_hex": "11" * 32},
        {"salt_path": str(salt_file), "key_hex": "11" * 32},
        {},
    ]
    for keys in mixed:
        with pytest.raises(ValueError, match="nothing from the other pair"):
            runner.run_anonymization(str(source), str(tmp_path / "e"), dry_run=True, **keys)


def test_stage_jobs_accept_a_raw_file_without_text_hash(tmp_path):
    """Every stage job derives text_hash in the read, so only note_text (and the recognizer results) are required."""
    runner = LocalJobRunner()
    salt_file = tmp_path / "salt.bin"
    key_file = tmp_path / "key.bin"
    salt_file.write_text("00" * 32)
    key_file.write_text("11" * 32)
    raw = tmp_path / "raw.parquet"
    pq.write_table(pa.table({"NOTE_TEXT": ["text"], "recognizer_results_json": ["[]"]}), raw)

    recognition = runner.run_recognition(str(raw), str(tmp_path / "r"), dry_run=True)
    anonymization = runner.run_anonymization(
        str(raw), str(tmp_path / "a"), salt_path=str(salt_file), key_path=str(key_file), dry_run=True
    )

    assert "text_hash" not in recognition["columns_detected"]
    assert "text_hash" not in anonymization["columns_detected"]


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

    merged_out = tmp_path / "merged_recognizer_output"

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
