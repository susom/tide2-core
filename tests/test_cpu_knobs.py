"""Tests for the small-hardware CPU knobs and checkpoint gating.

Covers the behaviour introduced for ≲4-CPU boxes (e.g. Google Colab):

1. Supervisor ``worker_num_cpus`` override is applied to the worker actor only
   when set (``.options(num_cpus=...)``), never otherwise.
2. ``_configure_checkpoint`` gates Ray Data checkpointing on ``enable`` and uses
   the expected ``id_column``.
3. Each stage's read/write (and the transformer's flat_map) carry the expected
   ``ray_remote_args["num_cpus"]`` per-operator reservation.

No live Ray cluster is needed: the worker classes and the ``ray.data`` calls are
mocked, so these run fast and deterministically.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ray

import tide2.runner.local_runner as lr
from tide2.runner.local_runner import _configure_checkpoint

# ---------------------------------------------------------------------------
# 1. Supervisor worker_num_cpus override gating
# ---------------------------------------------------------------------------


class TestWorkerCpuOverride:
    """The worker CPU override must apply .options() only when explicitly set."""

    def test_recognizer_no_override(self, monkeypatch):
        from tide2.actors import recognizer

        mock_worker = MagicMock()
        monkeypatch.setattr(recognizer, "RecognizerWorker", mock_worker)

        recognizer.RecognizerSupervisor(worker_num_cpus=None)

        mock_worker.options.assert_not_called()
        mock_worker.remote.assert_called_once_with()

    def test_recognizer_with_override(self, monkeypatch):
        from tide2.actors import recognizer

        mock_worker = MagicMock()
        monkeypatch.setattr(recognizer, "RecognizerWorker", mock_worker)

        recognizer.RecognizerSupervisor(worker_num_cpus=0.75)

        mock_worker.options.assert_called_once_with(num_cpus=0.75)
        mock_worker.options.return_value.remote.assert_called_once_with()
        mock_worker.remote.assert_not_called()

    def test_anonymizer_no_override(self, monkeypatch):
        from tide2.actors import anonymizer

        mock_worker = MagicMock()
        monkeypatch.setattr(anonymizer, "AnonymizerWorker", mock_worker)

        Actor = anonymizer.create_anonymizer_actor(  # noqa: N806 # it's a type
            salt=b"\x00" * 32, key=b"\x11" * 32, worker_num_cpus=None
        )
        Actor()

        mock_worker.options.assert_not_called()
        mock_worker.remote.assert_called_once()

    def test_anonymizer_with_override(self, monkeypatch):
        from tide2.actors import anonymizer

        mock_worker = MagicMock()
        monkeypatch.setattr(anonymizer, "AnonymizerWorker", mock_worker)

        Actor = anonymizer.create_anonymizer_actor(  # noqa: N806 # it's a type
            salt=b"\x00" * 32, key=b"\x11" * 32, worker_num_cpus=0.5
        )
        Actor()

        mock_worker.options.assert_called_once_with(num_cpus=0.5)
        mock_worker.options.return_value.remote.assert_called_once()
        mock_worker.remote.assert_not_called()

    def test_llm_recognizer_no_override(self, monkeypatch):
        from tide2.actors import llm_recognizer

        mock_worker = MagicMock()
        monkeypatch.setattr(llm_recognizer, "LlmRecognizerWorker", mock_worker)

        llm_recognizer.LlmRecognizerSupervisor(project_id="proj", worker_num_cpus=None)

        mock_worker.options.assert_not_called()
        mock_worker.remote.assert_called_once()

    def test_llm_recognizer_with_override(self, monkeypatch):
        from tide2.actors import llm_recognizer

        mock_worker = MagicMock()
        monkeypatch.setattr(llm_recognizer, "LlmRecognizerWorker", mock_worker)

        llm_recognizer.LlmRecognizerSupervisor(project_id="proj", worker_num_cpus=0.0)

        mock_worker.options.assert_called_once_with(num_cpus=0.0)
        mock_worker.options.return_value.remote.assert_called_once()
        mock_worker.remote.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Checkpoint gating (_configure_checkpoint helper)
# ---------------------------------------------------------------------------


class TestConfigureCheckpoint:
    """enable gates the CheckpointConfig; id_column/checkpoint_path are correct."""

    def test_disabled_clears_config(self, tmp_path):
        ctx = SimpleNamespace(checkpoint_config="leftover")
        _configure_checkpoint(ctx, enable=False, output_dir=tmp_path / "out", id_column="text_hash")
        assert ctx.checkpoint_config is None

    def test_enabled_sets_config(self, tmp_path):
        ctx = SimpleNamespace(checkpoint_config=None)
        output_dir = tmp_path / "06_anonymizer_output"
        _configure_checkpoint(ctx, enable=True, output_dir=output_dir, id_column="text_hash")

        cfg = ctx.checkpoint_config
        assert cfg is not None
        assert cfg.id_column == "text_hash"
        # Sibling directory next to the output dir, suffixed _ray_checkpoint.
        assert cfg.checkpoint_path == str(output_dir.parent / "06_anonymizer_output_ray_checkpoint")
        assert cfg.delete_checkpoint_on_success is False

    def test_enabled_respects_id_column(self, tmp_path):
        """Anonymizer passes row_id when present; the helper must honour it."""
        ctx = SimpleNamespace(checkpoint_config=None)
        _configure_checkpoint(ctx, enable=True, output_dir=tmp_path / "out", id_column="row_id")
        assert ctx.checkpoint_config.id_column == "row_id"


# ---------------------------------------------------------------------------
# 3. Per-operator CPU reservations (read / write / flat_map)
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_ds(monkeypatch):
    """Mock ray.data.read_parquet and the resulting Dataset; capture call kwargs.

    Returns a dict that, after a stage runs, holds the kwargs passed to
    read_parquet (``read``), write_parquet (``write``) and flat_map (``flat_map``).
    The runner's Ray init and DataContext are stubbed so no cluster is needed.
    """
    captured: dict = {}

    fake_ds = MagicMock(name="dataset")
    fake_ds.map_batches.return_value = fake_ds
    fake_ds.count.return_value = 1  # non-zero so the anonymizer 0-row guard passes

    def fake_flat_map(*_args, **kwargs):
        captured["flat_map"] = kwargs
        return fake_ds

    fake_ds.flat_map.side_effect = fake_flat_map

    def fake_write_parquet(*_args, **kwargs):
        captured["write"] = kwargs

    fake_ds.write_parquet.side_effect = fake_write_parquet

    def fake_read_parquet(*_args, **kwargs):
        captured["read"] = kwargs
        return fake_ds

    monkeypatch.setattr(ray.data, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(lr, "configure_data_context", lambda **_kwargs: None)
    monkeypatch.setattr(
        ray.data.DataContext,
        "get_current",
        staticmethod(lambda: SimpleNamespace(checkpoint_config=None)),
    )
    monkeypatch.setattr(lr, "resolve_input_files", lambda _path: ["/fake/input.parquet"])

    return captured


def _make_runner(monkeypatch):
    runner = lr.LocalJobRunner()
    monkeypatch.setattr(runner, "_init_ray", lambda: None)
    return runner


class TestPerOperatorReservations:
    """read/write/flat_map carry the expected ray_remote_args num_cpus."""

    def test_recognition_read_write(self, monkeypatch, tmp_path, captured_ds):
        runner = _make_runner(monkeypatch)
        monkeypatch.setattr(lr, "detect_columns", lambda *_a, **_k: ["text_hash", "note_text", "patient_identifiers"])

        runner.run_recognition(
            input_path="in",
            output_path=str(tmp_path / "out"),
            num_actors=1,
            read_cpus=0.25,
            write_cpus=0.5,
            enable_checkpoint=False,
        )

        assert captured_ds["read"]["ray_remote_args"] == {"num_cpus": 0.25}
        assert captured_ds["write"]["ray_remote_args"] == {"num_cpus": 0.5}

    def test_anonymization_read_write(self, monkeypatch, tmp_path, captured_ds):
        runner = _make_runner(monkeypatch)
        monkeypatch.setattr(
            lr,
            "detect_columns",
            lambda *_a, **_k: ["text_hash", "note_text", "recognizer_results_json", "patient_uid"],
        )
        salt_file = tmp_path / "salt.bin"
        key_file = tmp_path / "key.bin"
        salt_file.write_text("00" * 32)
        key_file.write_text("11" * 32)

        runner.run_anonymization(
            input_path="in",
            output_path=str(tmp_path / "out"),
            salt_path=str(salt_file),
            key_path=str(key_file),
            num_actors=1,
            read_cpus=0.25,
            write_cpus=0.5,
            enable_checkpoint=False,
        )

        assert captured_ds["read"]["ray_remote_args"] == {"num_cpus": 0.25}
        assert captured_ds["write"]["ray_remote_args"] == {"num_cpus": 0.5}

    def test_llm_recognition_read_write(self, monkeypatch, tmp_path, captured_ds):
        """Option A: the LLM write must carry write_cpus (review H1)."""
        runner = _make_runner(monkeypatch)
        monkeypatch.setattr(lr, "detect_columns", lambda *_a, **_k: ["text_hash", "note_text"])

        runner.run_llm_recognition(
            input_path="in",
            output_path=str(tmp_path / "out"),
            project_id="proj",
            num_actors=1,
            read_cpus=0.25,
            write_cpus=0.5,
            enable_checkpoint=False,
        )

        assert captured_ds["read"]["ray_remote_args"] == {"num_cpus": 0.25}
        assert captured_ds["write"]["ray_remote_args"] == {"num_cpus": 0.5}

    def test_transformer_read_write_flat_map(self, monkeypatch, tmp_path, captured_ds):
        runner = _make_runner(monkeypatch)
        # Avoid ray.cluster_resources(): force CPU-only, single actor each.
        monkeypatch.setattr(runner, "_resolve_transformer_resources", lambda *_a, **_k: (0, True, 1, 1))
        from tide2.transformers import config as tconfig

        monkeypatch.setattr(tconfig, "load_model_config", lambda _name: {"CHUNK_SIZE": 512, "CHUNK_OVERLAP_SIZE": 40})
        from tide2 import actors

        monkeypatch.setattr(actors, "create_transformer_actor", lambda **_k: MagicMock())

        runner.run_transformer(
            input_path="in",
            output_path=str(tmp_path / "out"),
            model_name="fake-model",
            model_path=str(tmp_path / "model"),  # skip resolve_model_path download
            read_cpus=0.25,
            flat_map_cpus=0.3,
            write_cpus=0.5,
            enable_checkpoint=False,
        )

        assert captured_ds["read"]["ray_remote_args"] == {"num_cpus": 0.25}
        assert captured_ds["flat_map"]["num_cpus"] == 0.3
        assert captured_ds["write"]["ray_remote_args"] == {"num_cpus": 0.5}


# ---------------------------------------------------------------------------
# CLI surface: --write-cpus is advertised
# ---------------------------------------------------------------------------


def test_cli_help_lists_write_cpus(monkeypatch, capsys):
    from tide2.runner import cli

    monkeypatch.setattr("sys.argv", ["tide2-runner", "run", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "--write-cpus" in out


def test_cli_llm_recognizer_forwards_cpu_knobs(monkeypatch):
    """The llm-recognizer CLI path must forward worker_num_cpus and the
    --no-checkpoint flag into run_llm_recognition (review: Copilot)."""
    from tide2.runner import cli
    from tide2.runner.local_runner import LocalJobRunner

    captured: dict = {}

    def fake_run_llm_recognition(self, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(LocalJobRunner, "run_llm_recognition", fake_run_llm_recognition)
    monkeypatch.setattr(LocalJobRunner, "shutdown", lambda _self: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tide2-runner",
            "run",
            "llm-recognizer",
            "-i",
            "in",
            "-o",
            "out",
            "--project-id",
            "proj",
            "--worker-num-cpus",
            "0.0",
            "--write-cpus",
            "0.25",
            "--no-checkpoint",
        ],
    )

    cli.main()

    assert captured["worker_num_cpus"] == 0.0
    assert captured["write_cpus"] == 0.25
    assert captured["enable_checkpoint"] is False
