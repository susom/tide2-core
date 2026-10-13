"""Tests verifying default and configurable checkpoint id_column behavior (issue #98).

Verifies:
1. All stage runners (recognition, transformer, anonymization, llm_recognition, pipeline)
   default _id_column to 'row_id'.
2. Custom _id_column (e.g. 'text_hash') can be explicitly passed for backwards compatibility.
3. tide2-runner CLI exposes --id-column with default 'row_id' and forwards it correctly.
4. YAML configs with 'id_column' or '_id_column' are respected unless overridden by CLI.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
import ray

import tide2.runner.local_runner as lr
from tide2.runner import cli
from tide2.runner.local_runner import LocalJobRunner
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
    """Explicit _id_column (e.g. 'text_hash') is honored across all runners for backwards compatibility."""

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
            transformer_kwargs={"_id_column": "custom_trans_id"},
        )

        assert stages_called["transformer"]["_id_column"] == "custom_trans_id"
        assert stages_called["recognizer"]["_id_column"] == "row_id"
        assert stages_called["anonymizer"]["_id_column"] == "row_id"


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
                "custom_id",
            ]
        )
        assert mock_runner_methods["recognizer"]["_id_column"] == "custom_id"

        # Underscore _id_column in YAML is also supported
        cfg_file_underscore = tmp_path / "config_underscore.yaml"
        cfg_file_underscore.write_text("_id_column: custom_underscore_id\n")
        mock_runner_methods.clear()
        cli.main(["run", "recognizer", "-i", "in.parquet", "-o", "out_dir", "--config", str(cfg_file_underscore)])
        assert mock_runner_methods["recognizer"]["_id_column"] == "custom_underscore_id"
