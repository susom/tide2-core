"""Tests verifying default and configurable checkpoint id_column behavior (issue #98).

Verifies:
1. All stage runners (recognition, transformer, anonymization, llm_recognition, pipeline)
   default _id_column to 'row_id'.
2. Custom _id_column ('text_hash') can be explicitly passed for backwards compatibility.
3. Unsupported _id_column values are rejected with ValueError across all Python APIs
   and rejected in the CLI.
4. Streamed pipeline mode resolves single sink precedence and strips _id_column from all
   stage kwargs, preventing unexpected argument failures in LlmRecognizerWorker.
5. tide2-runner CLI exposes --id-column with default 'row_id' (choices: row_id, text_hash)
   and forwards it correctly.
6. YAML configs with 'id_column' or '_id_column' are respected unless overridden by CLI.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
import ray

import tide2.runner.local_runner as lr
from tide2.runner import cli
from tide2.runner.local_runner import LocalJobRunner
from tide2.runner.local_runner import PipelineInputInfo
from tide2.runner.local_runner import _configure_checkpoint
from tide2.runner.local_runner import run_anonymization_simple
from tide2.runner.local_runner import run_pipeline_simple
from tide2.runner.local_runner import run_recognition_simple
from tide2.runner.local_runner import run_transformer_simple


@pytest.fixture
def mock_stage_environment(monkeypatch):
    """Mocks Ray and dataset execution for fast unit testing of runner checkpoint config."""
    captured_checkpoint = {}
    fake_ctx = SimpleNamespace(checkpoint_config=None)
    monkeypatch.setattr(ray.data.DataContext, "get_current", staticmethod(lambda: fake_ctx))
    monkeypatch.setattr(lr, "configure_data_context", lambda **_kwargs: None)
    monkeypatch.setattr(lr, "resolve_input_files", lambda _p: ["/fake/in.parquet"])
    monkeypatch.setattr(
        lr,
        "detect_columns",
        lambda *_a, **_k: ["note_text", "text_hash", "patient_id", "row_id", "recognizer_results_json"],
    )

    orig_configure_checkpoint = lr._configure_checkpoint

    def fake_configure_checkpoint(ctx, enable, output_dir, id_column):
        captured_checkpoint["enable"] = enable
        captured_checkpoint["output_dir"] = output_dir
        captured_checkpoint["id_column"] = id_column
        return orig_configure_checkpoint(ctx, enable=enable, output_dir=output_dir, id_column=id_column)

    monkeypatch.setattr(lr, "_configure_checkpoint", fake_configure_checkpoint)

    fake_ds = MagicMock(name="dataset")
    fake_ds.map_batches.return_value = fake_ds
    fake_ds.count.return_value = 1
    fake_ds.write_parquet.return_value = None

    monkeypatch.setattr(ray.data, "read_parquet", lambda *_a, **_k: fake_ds)
    monkeypatch.setattr(lr, "_read_stage_source", lambda *_a, **_k: fake_ds)

    monkeypatch.setattr(LocalJobRunner, "build_recognizer_stage", lambda *_args, **_kwargs: fake_ds)
    monkeypatch.setattr(LocalJobRunner, "build_anonymizer_stage", lambda *_args, **_kwargs: fake_ds)
    monkeypatch.setattr(LocalJobRunner, "build_llm_recognizer_stage", lambda *_args, **_kwargs: fake_ds)

    return captured_checkpoint


def _make_runner(monkeypatch):
    runner = LocalJobRunner()
    monkeypatch.setattr(runner, "_init_ray", lambda: None)
    return runner


class TestDefaultIdColumnAcrossStages:
    """All stage runners default _id_column to 'row_id'."""

    def test_run_recognition_defaults_to_row_id(self, monkeypatch, tmp_path, mock_stage_environment):
        runner = _make_runner(monkeypatch)
        runner.run_recognition(
            input_path="in",
            output_path=str(tmp_path / "rec_out"),
            num_actors=1,
            enable_checkpoint=True,
        )
        assert mock_stage_environment["id_column"] == "row_id"

    def test_run_anonymization_defaults_to_row_id(self, monkeypatch, tmp_path, mock_stage_environment):
        runner = _make_runner(monkeypatch)
        salt_file = tmp_path / "salt.bin"
        key_file = tmp_path / "key.bin"
        salt_file.write_text("00" * 32)
        key_file.write_text("11" * 32)

        runner.run_anonymization(
            input_path="in",
            output_path=str(tmp_path / "anon_out"),
            salt_path=str(salt_file),
            key_path=str(key_file),
            num_actors=1,
            enable_checkpoint=True,
        )
        assert mock_stage_environment["id_column"] == "row_id"

    def test_run_transformer_defaults_to_row_id(self, monkeypatch, tmp_path, mock_stage_environment):
        runner = _make_runner(monkeypatch)
        monkeypatch.setattr(runner, "_resolve_transformer_resources", lambda *_a, **_k: (0, True, 1, 1))

        from tide2.transformers import config as tconfig

        monkeypatch.setattr(
            tconfig,
            "load_model_config",
            lambda _m: {"MODEL_MAX_LENGTH": 512, "CHUNK_OVERLAP_SIZE": 40},
        )

        runner.run_transformer(
            input_path="in",
            output_path=str(tmp_path / "trans_out"),
            model_name="test-model",
            model_path="/fake/model/path",
            enable_checkpoint=True,
        )
        assert mock_stage_environment["id_column"] == "row_id"

    def test_run_llm_recognition_defaults_to_row_id(self, monkeypatch, tmp_path, mock_stage_environment):
        runner = _make_runner(monkeypatch)
        runner.run_llm_recognition(
            input_path="in",
            output_path=str(tmp_path / "llm_out"),
            project_id="test-proj",
            num_actors=1,
            enable_checkpoint=True,
        )
        assert mock_stage_environment["id_column"] == "row_id"

    def test_run_pipeline_defaults_to_row_id(self, monkeypatch, tmp_path):
        runner = _make_runner(monkeypatch)
        monkeypatch.setattr(runner, "_apply_pipeline_recommendations", lambda *_a, **_k: None)

        stages_called: dict = {}
        monkeypatch.setattr(
            runner,
            "run_transformer",
            lambda **k: (stages_called.setdefault("transformer", k), {"transformer": True})[1],
        )
        monkeypatch.setattr(
            runner,
            "run_recognition",
            lambda **k: (stages_called.setdefault("recognizer", k), {"recognizer": True})[1],
        )
        monkeypatch.setattr(
            runner,
            "run_anonymization",
            lambda **k: (stages_called.setdefault("anonymizer", k), {"anonymizer": True})[1],
        )

        f = tmp_path / "in.parquet"
        import pyarrow.parquet as pq

        pq.write_table(pa.table({"note_text": ["note"], "row_id": ["r1"]}), f)

        runner.run_pipeline(
            input_path=str(f),
            output_dir=str(tmp_path / "out"),
            model_name="test-model",
        )

        assert stages_called["transformer"]["_id_column"] == "row_id"
        assert stages_called["recognizer"]["_id_column"] == "row_id"
        assert stages_called["anonymizer"]["_id_column"] == "row_id"


class TestCustomIdColumnSupport:
    """Explicit _id_column ('text_hash') is honored across all runners for backwards compatibility."""

    def test_run_recognition_honors_custom_id_column(self, monkeypatch, tmp_path, mock_stage_environment):
        runner = _make_runner(monkeypatch)
        runner.run_recognition(
            input_path="in",
            output_path=str(tmp_path / "rec_out"),
            num_actors=1,
            enable_checkpoint=True,
            _id_column="text_hash",
        )
        assert mock_stage_environment["id_column"] == "text_hash"

    def test_run_transformer_honors_custom_id_column(self, monkeypatch, tmp_path, mock_stage_environment):
        runner = _make_runner(monkeypatch)
        monkeypatch.setattr(runner, "_resolve_transformer_resources", lambda *_a, **_k: (0, True, 1, 1))

        from tide2.transformers import config as tconfig

        monkeypatch.setattr(
            tconfig,
            "load_model_config",
            lambda _m: {"MODEL_MAX_LENGTH": 512, "CHUNK_OVERLAP_SIZE": 40},
        )

        runner.run_transformer(
            input_path="in",
            output_path=str(tmp_path / "trans_out"),
            model_name="test-model",
            model_path="/fake/model/path",
            enable_checkpoint=True,
            _id_column="text_hash",
        )
        assert mock_stage_environment["id_column"] == "text_hash"

    def test_run_llm_recognition_honors_custom_id_column(self, monkeypatch, tmp_path, mock_stage_environment):
        runner = _make_runner(monkeypatch)
        runner.run_llm_recognition(
            input_path="in",
            output_path=str(tmp_path / "llm_out"),
            project_id="test-proj",
            num_actors=1,
            enable_checkpoint=True,
            _id_column="text_hash",
        )
        assert mock_stage_environment["id_column"] == "text_hash"

    def test_run_anonymization_honors_custom_id_column(self, monkeypatch, tmp_path, mock_stage_environment):
        runner = _make_runner(monkeypatch)
        salt_file = tmp_path / "salt.bin"
        key_file = tmp_path / "key.bin"
        salt_file.write_text("00" * 32)
        key_file.write_text("11" * 32)

        runner.run_anonymization(
            input_path="in",
            output_path=str(tmp_path / "anon_out"),
            salt_path=str(salt_file),
            key_path=str(key_file),
            num_actors=1,
            enable_checkpoint=True,
            _id_column="text_hash",
        )
        assert mock_stage_environment["id_column"] == "text_hash"

    def test_run_pipeline_forwards_custom_id_column(self, monkeypatch, tmp_path):
        runner = _make_runner(monkeypatch)
        monkeypatch.setattr(runner, "_apply_pipeline_recommendations", lambda *_a, **_k: None)

        stages_called: dict = {}
        monkeypatch.setattr(
            runner,
            "run_transformer",
            lambda **k: (stages_called.setdefault("transformer", k), {"transformer": True})[1],
        )
        monkeypatch.setattr(
            runner,
            "run_recognition",
            lambda **k: (stages_called.setdefault("recognizer", k), {"recognizer": True})[1],
        )
        monkeypatch.setattr(
            runner,
            "run_anonymization",
            lambda **k: (stages_called.setdefault("anonymizer", k), {"anonymizer": True})[1],
        )

        f = tmp_path / "in.parquet"
        import pyarrow.parquet as pq

        pq.write_table(pa.table({"note_text": ["note"], "row_id": ["r1"]}), f)

        runner.run_pipeline(
            input_path=str(f),
            output_dir=str(tmp_path / "out"),
            model_name="test-model",
            _id_column="text_hash",
        )

        assert stages_called["transformer"]["_id_column"] == "text_hash"
        assert stages_called["recognizer"]["_id_column"] == "text_hash"
        assert stages_called["anonymizer"]["_id_column"] == "text_hash"

    def test_run_pipeline_allows_per_stage_override(self, monkeypatch, tmp_path):
        runner = _make_runner(monkeypatch)
        monkeypatch.setattr(runner, "_apply_pipeline_recommendations", lambda *_a, **_k: None)

        stages_called: dict = {}
        monkeypatch.setattr(
            runner,
            "run_transformer",
            lambda **k: (stages_called.setdefault("transformer", k), {"transformer": True})[1],
        )
        monkeypatch.setattr(
            runner,
            "run_recognition",
            lambda **k: (stages_called.setdefault("recognizer", k), {"recognizer": True})[1],
        )
        monkeypatch.setattr(
            runner,
            "run_anonymization",
            lambda **k: (stages_called.setdefault("anonymizer", k), {"anonymizer": True})[1],
        )

        f = tmp_path / "in.parquet"
        import pyarrow.parquet as pq

        pq.write_table(pa.table({"note_text": ["note"], "row_id": ["r1"]}), f)

        runner.run_pipeline(
            input_path=str(f),
            output_dir=str(tmp_path / "out"),
            model_name="test-model",
            _id_column="row_id",
            transformer_kwargs={"_id_column": "text_hash"},
        )

        assert stages_called["transformer"]["_id_column"] == "text_hash"
        assert stages_called["recognizer"]["_id_column"] == "row_id"
        assert stages_called["anonymizer"]["_id_column"] == "row_id"


class TestInvalidIdColumnRejection:
    """Unsupported _id_column names raise ValueError in Python and fail in CLI."""

    def test_configure_checkpoint_rejects_invalid_id_column(self, tmp_path):
        ctx = SimpleNamespace(checkpoint_config=None)
        with pytest.raises(ValueError, match=r"Invalid _id_column: 'unsupported'"):
            _configure_checkpoint(ctx, enable=True, output_dir=tmp_path / "out", id_column="unsupported")

    def test_stage_runners_reject_invalid_id_column(self, monkeypatch, tmp_path):
        runner = _make_runner(monkeypatch)
        salt_file = tmp_path / "salt.bin"
        key_file = tmp_path / "key.bin"
        salt_file.write_text("00" * 32)
        key_file.write_text("11" * 32)

        with pytest.raises(ValueError, match=r"Invalid _id_column: 'bad_id'"):
            runner.run_recognition(input_path="in", output_path="out", _id_column="bad_id")

        with pytest.raises(ValueError, match=r"Invalid _id_column: 'bad_id'"):
            runner.run_anonymization(
                input_path="in",
                output_path="out",
                salt_path=str(salt_file),
                key_path=str(key_file),
                _id_column="bad_id",
            )

        with pytest.raises(ValueError, match=r"Invalid _id_column: 'bad_id'"):
            runner.run_transformer(input_path="in", output_path="out", model_name="m", _id_column="bad_id")

        with pytest.raises(ValueError, match=r"Invalid _id_column: 'bad_id'"):
            runner.run_llm_recognition(input_path="in", output_path="out", project_id="p", _id_column="bad_id")

        with pytest.raises(ValueError, match=r"Invalid _id_column: 'bad_id'"):
            runner.run_pipeline(input_path="in", output_dir="out", model_name="m", _id_column="bad_id")

    def test_run_pipeline_rejects_invalid_stage_override_upfront(self, monkeypatch):
        runner = _make_runner(monkeypatch)
        stage_called = False

        def mock_transformer(**_kwargs):
            nonlocal stage_called
            stage_called = True
            return {}

        monkeypatch.setattr(runner, "run_transformer", mock_transformer)

        with pytest.raises(ValueError, match=r"Invalid _id_column: 'bad_stage'"):
            runner.run_pipeline(
                input_path="in.parquet",
                output_dir="out_dir",
                model_name="m",
                anonymizer_kwargs={"_id_column": "bad_stage"},
            )
        assert not stage_called

        with pytest.raises(ValueError, match=r"Invalid _id_column: None"):
            runner.run_pipeline(
                input_path="in.parquet",
                output_dir="out_dir",
                model_name="m",
                transformer_kwargs={"_id_column": None},
            )
        assert not stage_called

    def test_cli_rejects_unsupported_id_column_choice(self, monkeypatch):
        monkeypatch.setattr(LocalJobRunner, "_init_ray", lambda *_a, **_k: None)
        with pytest.raises(SystemExit) as exc:
            cli.main(["run", "recognizer", "-i", "in.parquet", "-o", "out_dir", "--id-column", "unsupported"])
        assert exc.value.code == 2

    @pytest.mark.parametrize(
        "yaml_content",
        [
            "id_column: invalid_name\n",
            "id_column: null\n",
            "id_column: ''\n",
            "_id_column: null\n",
            "_id_column: ''\n",
        ],
    )
    def test_cli_rejects_invalid_id_column_from_config(self, monkeypatch, tmp_path, yaml_content):
        monkeypatch.setattr(LocalJobRunner, "_init_ray", lambda *_a, **_k: None)
        cfg_file = tmp_path / "bad_config.yaml"
        cfg_file.write_text(yaml_content)

        with pytest.raises(SystemExit) as exc:
            cli.main(["run", "recognizer", "-i", "in.parquet", "-o", "out_dir", "--config", str(cfg_file)])
        assert exc.value.code == 1


class TestStreamedPipelineIdColumnHandling:
    """Streamed pipeline pops _id_column from all stage kwargs and resolves precedence cleanly."""

    def test_streamed_pipeline_strips_id_column_from_llm_and_stage_kwargs(self, monkeypatch, tmp_path):
        runner = _make_runner(monkeypatch)
        monkeypatch.setattr(lr, "configure_data_context", lambda **_kwargs: None)
        monkeypatch.setattr(lr, "check_streamed_admission", lambda _mins, cpus: cpus)
        monkeypatch.setattr(lr, "_streamed_source_columns", lambda _contracts: ["note_text", "row_id", "text_hash"])

        captured_ctor_kwargs: dict = {}
        fake_ds = MagicMock(name="dataset")
        fake_ds.map_batches.return_value = fake_ds
        monkeypatch.setattr(lr, "_read_stage_source", lambda *_a, **_k: fake_ds)

        def fake_build_llm_stage(_self, ds, **kwargs):
            captured_ctor_kwargs.update(kwargs.get("fn_constructor_kwargs", {}))
            return fake_ds

        monkeypatch.setattr(LocalJobRunner, "build_llm_recognizer_stage", fake_build_llm_stage)
        monkeypatch.setattr(LocalJobRunner, "build_anonymizer_stage", lambda *_a, **_k: fake_ds)
        monkeypatch.setattr(lr, "_check_streamed_row_parity", lambda *_a, **_k: None)
        monkeypatch.setattr(lr, "list_parquet_files", lambda *_a, **_k: [])

        captured_checkpoint: dict = {}

        def fake_configure_checkpoint(ctx, enable, output_dir, id_column):
            captured_checkpoint["id_column"] = id_column

        monkeypatch.setattr(lr, "_configure_checkpoint", fake_configure_checkpoint)

        info = PipelineInputInfo(
            columns={"note_text", "row_id", "text_hash"},
            actual_to_lower={"note_text": "note_text", "row_id": "row_id", "text_hash": "text_hash"},
            lower_to_actual={"note_text": "note_text", "row_id": "row_id", "text_hash": "text_hash"},
            num_rows=1,
            has_row_id=True,
            has_text_hash=True,
            has_patient_id=False,
        )

        anonymizer_kw = {"num_actors": 1, "worker_num_cpus": 1.0, "_id_column": "text_hash"}
        llm_kw = {"num_actors": 1, "project_id": "proj", "_id_column": "text_hash"}

        out_dir = tmp_path / "streamed_out"
        out_dir.mkdir()

        runner._run_pipeline_streamed(
            input_files=["in.parquet"],
            input_info=info,
            output_path=Path(out_dir),
            model_name="test-model",
            salt_hex="00" * 32,
            key_hex="11" * 32,
            transformer_kwargs={},
            recognizer_kwargs={},
            anonymizer_kwargs=anonymizer_kw,
            llm_recognizer_mode="only",
            llm_recognizer_kwargs=llm_kw,
            start_time=0.0,
            _id_column="row_id",
        )

        # _id_column must NOT be in constructor kwargs for LlmRecognizerWorker
        assert "_id_column" not in captured_ctor_kwargs
        assert "_id_column" not in llm_kw
        assert "_id_column" not in anonymizer_kw
        # Anonymizer override takes precedence for the sink
        assert captured_checkpoint["id_column"] == "text_hash"

    def test_streamed_pipeline_upstream_override_does_not_affect_sink(self, monkeypatch, tmp_path):
        runner = _make_runner(monkeypatch)
        monkeypatch.setattr(lr, "configure_data_context", lambda **_kwargs: None)
        monkeypatch.setattr(lr, "check_streamed_admission", lambda _mins, cpus: cpus)
        monkeypatch.setattr(lr, "_streamed_source_columns", lambda _contracts: ["note_text", "row_id", "text_hash"])

        fake_ds = MagicMock(name="dataset")
        fake_ds.map_batches.return_value = fake_ds
        monkeypatch.setattr(lr, "_read_stage_source", lambda *_a, **_k: fake_ds)
        monkeypatch.setattr(LocalJobRunner, "build_llm_recognizer_stage", lambda *_a, **_k: fake_ds)
        monkeypatch.setattr(LocalJobRunner, "build_anonymizer_stage", lambda *_a, **_k: fake_ds)
        monkeypatch.setattr(lr, "_check_streamed_row_parity", lambda *_a, **_k: None)
        monkeypatch.setattr(lr, "list_parquet_files", lambda *_a, **_k: [])

        captured_checkpoint: dict = {}

        def fake_configure_checkpoint(ctx, enable, output_dir, id_column):
            captured_checkpoint["id_column"] = id_column

        monkeypatch.setattr(lr, "_configure_checkpoint", fake_configure_checkpoint)

        info = PipelineInputInfo(
            columns={"note_text", "row_id", "text_hash"},
            actual_to_lower={"note_text": "note_text", "row_id": "row_id", "text_hash": "text_hash"},
            lower_to_actual={"note_text": "note_text", "row_id": "row_id", "text_hash": "text_hash"},
            num_rows=1,
            has_row_id=True,
            has_text_hash=True,
            has_patient_id=False,
        )

        recognizer_kw = {"num_actors": 1, "worker_num_cpus": 1.0, "_id_column": "text_hash"}
        anonymizer_kw = {"num_actors": 1, "worker_num_cpus": 1.0}

        out_dir = tmp_path / "streamed_out2"
        out_dir.mkdir()

        runner._run_pipeline_streamed(
            input_files=["in.parquet"],
            input_info=info,
            output_path=Path(out_dir),
            model_name="test-model",
            salt_hex="00" * 32,
            key_hex="11" * 32,
            transformer_kwargs={},
            recognizer_kwargs=recognizer_kw,
            anonymizer_kwargs=anonymizer_kw,
            llm_recognizer_mode="only",
            llm_recognizer_kwargs={},
            start_time=0.0,
            _id_column="row_id",
        )

        assert "_id_column" not in recognizer_kw
        # Upstream recognizer override does not override anonymizer sink; row_id from pipeline stands
        assert captured_checkpoint["id_column"] == "row_id"

    def test_streamed_pipeline_validates_all_stage_overrides(self, monkeypatch, tmp_path):
        runner = _make_runner(monkeypatch)
        monkeypatch.setattr(lr, "configure_data_context", lambda **_kwargs: None)
        monkeypatch.setattr(lr, "check_streamed_admission", lambda _mins, cpus: cpus)
        monkeypatch.setattr(lr, "_streamed_source_columns", lambda _contracts: ["note_text", "row_id", "text_hash"])
        fake_ds = MagicMock(name="dataset")
        monkeypatch.setattr(lr, "_read_stage_source", lambda *_a, **_k: fake_ds)
        monkeypatch.setattr(LocalJobRunner, "build_llm_recognizer_stage", lambda *_a, **_k: fake_ds)
        monkeypatch.setattr(LocalJobRunner, "build_anonymizer_stage", lambda *_a, **_k: fake_ds)

        info = PipelineInputInfo(
            columns={"note_text", "row_id", "text_hash"},
            actual_to_lower={"note_text": "note_text", "row_id": "row_id", "text_hash": "text_hash"},
            lower_to_actual={"note_text": "note_text", "row_id": "row_id", "text_hash": "text_hash"},
            num_rows=1,
            has_row_id=True,
            has_text_hash=True,
            has_patient_id=False,
        )

        out_dir = tmp_path / "streamed_out3"
        out_dir.mkdir()

        with pytest.raises(ValueError, match=r"Invalid _id_column: 'bad_stage_id'"):
            runner._run_pipeline_streamed(
                input_files=["in.parquet"],
                input_info=info,
                output_path=Path(out_dir),
                model_name="test-model",
                salt_hex="00" * 32,
                key_hex="11" * 32,
                transformer_kwargs={},
                recognizer_kwargs={"_id_column": "bad_stage_id"},
                anonymizer_kwargs={"_id_column": "row_id"},
                llm_recognizer_mode="only",
                llm_recognizer_kwargs={},
                start_time=0.0,
                _id_column="row_id",
            )


class TestSimpleRunnersIdColumn:
    """Simple wrapper functions default to 'row_id' and accept _id_column."""

    def test_run_recognition_simple_defaults_and_forwards(self, monkeypatch):
        captured = {}

        def mock_run_recognition(_self, **kwargs):
            captured.update(kwargs)
            return {"status": "ok"}

        monkeypatch.setattr(LocalJobRunner, "_init_ray", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(LocalJobRunner, "shutdown", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(LocalJobRunner, "run_recognition", mock_run_recognition)

        run_recognition_simple("in.parquet", "out_dir")
        assert captured["_id_column"] == "row_id"

        run_recognition_simple("in.parquet", "out_dir", _id_column="text_hash")
        assert captured["_id_column"] == "text_hash"

    def test_run_anonymization_simple_defaults_and_forwards(self, monkeypatch):
        captured = {}

        def mock_run_anonymization(_self, **kwargs):
            captured.update(kwargs)
            return {"status": "ok"}

        monkeypatch.setattr(LocalJobRunner, "_init_ray", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(LocalJobRunner, "shutdown", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(LocalJobRunner, "run_anonymization", mock_run_anonymization)

        run_anonymization_simple("in.parquet", "out_dir", "salt.bin", "key.bin")
        assert captured["_id_column"] == "row_id"

        run_anonymization_simple("in.parquet", "out_dir", "salt.bin", "key.bin", _id_column="text_hash")
        assert captured["_id_column"] == "text_hash"

    def test_run_transformer_simple_defaults_and_forwards(self, monkeypatch):
        captured = {}

        def mock_run_transformer(_self, **kwargs):
            captured.update(kwargs)
            return {"status": "ok"}

        monkeypatch.setattr(LocalJobRunner, "_init_ray", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(LocalJobRunner, "shutdown", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(LocalJobRunner, "run_transformer", mock_run_transformer)

        run_transformer_simple("in.parquet", "out_dir", "model")
        assert captured["_id_column"] == "row_id"

        run_transformer_simple("in.parquet", "out_dir", "model", _id_column="text_hash")
        assert captured["_id_column"] == "text_hash"

    def test_run_pipeline_simple_defaults_and_forwards(self, monkeypatch):
        captured = {}

        def mock_run_pipeline(_self, **kwargs):
            captured.update(kwargs)
            return {"status": "ok"}

        monkeypatch.setattr(LocalJobRunner, "_init_ray", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(LocalJobRunner, "shutdown", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(LocalJobRunner, "run_pipeline", mock_run_pipeline)

        run_pipeline_simple("in.parquet", "out_dir", "model")
        assert captured["_id_column"] == "row_id"

        run_pipeline_simple("in.parquet", "out_dir", "model", _id_column="text_hash")
        assert captured["_id_column"] == "text_hash"


class TestCliIdColumn:
    """CLI tide2-runner run <stage> exposes --id-column with default 'row_id'."""

    @pytest.fixture
    def mock_runner_methods(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(LocalJobRunner, "_init_ray", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(LocalJobRunner, "shutdown", lambda *_args, **_kwargs: None)

        monkeypatch.setattr(
            LocalJobRunner,
            "run_recognition",
            lambda _self, **kw: (captured.setdefault("recognizer", kw), {"status": "ok"})[1],
        )
        monkeypatch.setattr(
            LocalJobRunner,
            "run_anonymization",
            lambda _self, **kw: (captured.setdefault("anonymizer", kw), {"status": "ok"})[1],
        )
        monkeypatch.setattr(
            LocalJobRunner,
            "run_transformer",
            lambda _self, **kw: (captured.setdefault("transformer", kw), {"status": "ok"})[1],
        )
        monkeypatch.setattr(
            LocalJobRunner,
            "run_llm_recognition",
            lambda _self, **kw: (captured.setdefault("llm_recognizer", kw), {"status": "ok"})[1],
        )
        monkeypatch.setattr(
            LocalJobRunner,
            "run_pipeline",
            lambda _self, **kw: (captured.setdefault("pipeline", kw), {"status": "ok"})[1],
        )
        return captured

    def test_cli_recognizer_default_and_custom(self, mock_runner_methods):
        cli.main(["run", "recognizer", "-i", "in.parquet", "-o", "out_dir"])
        assert mock_runner_methods["recognizer"]["_id_column"] == "row_id"

        mock_runner_methods.clear()
        cli.main(["run", "recognizer", "-i", "in.parquet", "-o", "out_dir", "--id-column", "text_hash"])
        assert mock_runner_methods["recognizer"]["_id_column"] == "text_hash"

    def test_cli_anonymizer_default_and_custom(self, mock_runner_methods):
        cli.main(["run", "anonymizer", "-i", "in.parquet", "-o", "out_dir", "--salt", "s", "--key", "k"])
        assert mock_runner_methods["anonymizer"]["_id_column"] == "row_id"

        mock_runner_methods.clear()
        cli.main(
            [
                "run",
                "anonymizer",
                "-i",
                "in.parquet",
                "-o",
                "out_dir",
                "--salt",
                "s",
                "--key",
                "k",
                "--id-column",
                "text_hash",
            ]
        )
        assert mock_runner_methods["anonymizer"]["_id_column"] == "text_hash"

    def test_cli_transformer_default_and_custom(self, mock_runner_methods):
        cli.main(["run", "transformer", "-i", "in.parquet", "-o", "out_dir", "--model", "test-model"])
        assert mock_runner_methods["transformer"]["_id_column"] == "row_id"

        mock_runner_methods.clear()
        cli.main(
            [
                "run",
                "transformer",
                "-i",
                "in.parquet",
                "-o",
                "out_dir",
                "--model",
                "test-model",
                "--id-column",
                "text_hash",
            ]
        )
        assert mock_runner_methods["transformer"]["_id_column"] == "text_hash"

    def test_cli_llm_recognizer_default_and_custom(self, mock_runner_methods):
        cli.main(["run", "llm-recognizer", "-i", "in.parquet", "-o", "out_dir", "--project-id", "proj"])
        assert mock_runner_methods["llm_recognizer"]["_id_column"] == "row_id"

        mock_runner_methods.clear()
        cli.main(
            [
                "run",
                "llm-recognizer",
                "-i",
                "in.parquet",
                "-o",
                "out_dir",
                "--project-id",
                "proj",
                "--id-column",
                "text_hash",
            ]
        )
        assert mock_runner_methods["llm_recognizer"]["_id_column"] == "text_hash"

    def test_cli_pipeline_default_and_custom(self, mock_runner_methods):
        cli.main(["run", "pipeline", "-i", "in.parquet", "-o", "out_dir", "--model", "test-model"])
        assert mock_runner_methods["pipeline"]["_id_column"] == "row_id"

        mock_runner_methods.clear()
        cli.main(
            [
                "run",
                "pipeline",
                "-i",
                "in.parquet",
                "-o",
                "out_dir",
                "--model",
                "test-model",
                "--id-column",
                "text_hash",
            ]
        )
        assert mock_runner_methods["pipeline"]["_id_column"] == "text_hash"

    def test_cli_config_yaml_support(self, mock_runner_methods, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("id_column: text_hash\n")

        cli.main(["run", "recognizer", "-i", "in.parquet", "-o", "out_dir", "--config", str(cfg_file)])
        assert mock_runner_methods["recognizer"]["_id_column"] == "text_hash"

        # CLI flag overrides config
        mock_runner_methods.clear()
        cli.main(
            [
                "run",
                "recognizer",
                "-i",
                "in.parquet",
                "-o",
                "out_dir",
                "--config",
                str(cfg_file),
                "--id-column",
                "row_id",
            ]
        )
        assert mock_runner_methods["recognizer"]["_id_column"] == "row_id"

        # Underscore _id_column in YAML is also supported
        cfg_file_underscore = tmp_path / "config_underscore.yaml"
        cfg_file_underscore.write_text("_id_column: text_hash\n")
        mock_runner_methods.clear()
        cli.main(["run", "recognizer", "-i", "in.parquet", "-o", "out_dir", "--config", str(cfg_file_underscore)])
        assert mock_runner_methods["recognizer"]["_id_column"] == "text_hash"
