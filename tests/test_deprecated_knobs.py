"""Tests for deprecated knobs: batch_timeout, supervisor shims, and worker aliases."""

import contextlib

import pytest

from tide2.actors import AnonymizerActor
from tide2.actors import RecognizerActor
from tide2.actors.anonymizer import AnonymizerSupervisor
from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.anonymizer import AnonymizerWorkerActor
from tide2.actors.llm_recognizer import LlmRecognizerSupervisor
from tide2.actors.llm_recognizer import LlmRecognizerWorker
from tide2.actors.llm_recognizer import LlmRecognizerWorkerActor
from tide2.actors.recognizer import RecognizerSupervisor
from tide2.actors.recognizer import RecognizerWorker
from tide2.actors.recognizer import RecognizerWorkerActor
from tide2.runner.cli import main
from tide2.runner.local_runner import _resolve_slot_cpus


def test_actor_aliases():
    """RecognizerActor and AnonymizerActor re-point to plain worker classes."""
    assert RecognizerActor is RecognizerWorker
    assert AnonymizerActor is AnonymizerWorker
    # Plain classes, not remote actors
    assert not hasattr(RecognizerWorker, "remote")
    assert not hasattr(AnonymizerWorker, "remote")
    assert not hasattr(LlmRecognizerWorker, "remote")

    # Explicit ray.remote worker actor aliases preserve .remote()
    assert hasattr(RecognizerWorkerActor, "remote")
    assert hasattr(AnonymizerWorkerActor, "remote")
    assert hasattr(LlmRecognizerWorkerActor, "remote")


def test_supervisor_deprecated_shim():
    """Supervisors remain importable, emit DeprecationWarning, and wrap workers in-process."""
    from tide2.anonymizers.presidio_patches import unpatch_remove_duplicates

    try:
        with pytest.deprecated_call(match="RecognizerSupervisor is deprecated"):
            rec_sup = RecognizerSupervisor(batch_timeout=120)
        assert isinstance(rec_sup.worker, RecognizerWorker)

        with pytest.deprecated_call(match="AnonymizerSupervisor is deprecated"):
            anon_sup = AnonymizerSupervisor(salt=b"\x00" * 32, key=b"\x11" * 32)
        assert isinstance(anon_sup.worker, AnonymizerWorker)

        with pytest.deprecated_call(match="LlmRecognizerSupervisor is deprecated"):
            llm_sup = LlmRecognizerSupervisor(project_id="test-proj")
        assert isinstance(llm_sup.worker, LlmRecognizerWorker)
    finally:
        unpatch_remove_duplicates()


def test_resolve_slot_cpus_deprecation_warning():
    """_resolve_slot_cpus emits DeprecationWarning when num_cpus is provided."""
    with pytest.deprecated_call(match="`num_cpus` .* is deprecated in favor of `worker_num_cpus`"):
        slot_cpus, _, _ = _resolve_slot_cpus(num_cpus=2, worker_num_cpus=None)
    assert slot_cpus == 3.0


@pytest.mark.parametrize(
    ("num_cpus", "worker_num_cpus", "expected_slot_cpus"),
    [
        (0, 1.0, 1.0),
        (2, None, 3.0),
        (0, 0.25, 0.25),
        (2, 0.75, 2.75),
        (0, 0, 0.0),
    ],
)
def test_slot_cpu_additive_accounting(num_cpus, worker_num_cpus, expected_slot_cpus):
    slot_cpus, ray_remote_args, _ = _resolve_slot_cpus(num_cpus, worker_num_cpus)
    assert slot_cpus == expected_slot_cpus
    assert ray_remote_args["num_cpus"] == expected_slot_cpus


def test_cli_warns_deprecated_batch_timeout(monkeypatch, tmp_path):
    """CLI emits DeprecationWarning when --batch-timeout is passed."""
    import sys

    import pyarrow as pa
    import pyarrow.parquet as pq

    in_file = tmp_path / "in.parquet"
    pq.write_table(pa.table({"text_hash": ["h1"], "note_text": ["hello"]}), in_file)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tide2-runner",
            "run",
            "recognizer",
            "-i",
            str(in_file),
            "-o",
            str(tmp_path / "out"),
            "--dry-run",
            "--batch-timeout",
            "90",
        ],
    )
    with (
        pytest.deprecated_call(match="--batch-timeout is deprecated and ignored"),
        contextlib.suppress(SystemExit),
    ):
        main()
