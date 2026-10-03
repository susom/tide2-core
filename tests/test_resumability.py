"""Tests for the resumability contract: in-read row_id, URI-aware paths, and kill/restart."""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
import ray
import ray.data

import tide2.runner.local_runner as lr
import tide2.runner.utils as runner_utils
from tide2.actors.transformer import BIOAggregationActor
from tide2.actors.transformer import TransformerInferenceActor
from tide2.runner.local_runner import _configure_checkpoint
from tide2.runner.local_runner import _read_stage_source
from tide2.runner.local_runner import add_row_id
from tide2.runner.local_runner import normalize_source_batch

# ---------------------------------------------------------------------------
# Aggregation failures propagate instead of becoming zero detections
# ---------------------------------------------------------------------------


def _boom(*_args, **_kwargs):
    raise RuntimeError("malformed prediction")


def test_bio_aggregation_actor_propagates_aggregation_failure(monkeypatch):
    """A failing aggregation must raise, not emit a row with zero entities."""
    monkeypatch.setattr("tide2.actors.transformer.format_note_entities", _boom)
    actor = BIOAggregationActor("m", model_to_presidio_mapping={}, ignore_labels={"O"})
    batch = {"text_hash": ["h"], "note_text": ["John"], "predictions_raw_json": ["[]"], "patient_id": ["p"]}
    with pytest.raises(RuntimeError, match="malformed prediction"):
        actor(batch)


def test_transformer_actor_propagates_aggregation_failure():
    """The in-actor aggregation path must raise as well."""
    actor = object.__new__(TransformerInferenceActor)
    actor._aggregate_bio = True
    actor._log_gpu_mem = lambda *_a, **_k: None
    actor._run_inference_raw_with_oom_recovery = lambda texts: [[] for _ in texts]
    actor._format_note = _boom
    batch = {"text_hash": ["h"], "note_text": ["John"], "patient_id": ["p"]}
    with pytest.raises(RuntimeError, match="malformed prediction"):
        actor(batch)


# ---------------------------------------------------------------------------
# normalize_source_batch / add_row_id on empty tables
# ---------------------------------------------------------------------------


def test_normalize_source_batch_accepts_empty_table():
    """Ray passes an empty table to the read hook for schema inference."""
    empty = pa.table({"NOTE_TEXT": pa.array([], pa.string()), "patient_id": pa.array([], pa.string())})
    out = normalize_source_batch(empty)
    assert len(out) == 0
    assert {"note_text", "text_hash", "row_id"} <= set(out.column_names)


def test_add_row_id_accepts_empty_table():
    empty = pa.table({"text_hash": pa.array([], pa.string())})
    out = add_row_id(empty)
    assert len(out) == 0
    assert "row_id" in out.column_names


# ---------------------------------------------------------------------------
# URI-aware path helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_object_store(tmp_path, monkeypatch):
    """Back every ``gs://`` URI with a local directory."""
    root = tmp_path / "bucket"
    root.mkdir()
    store = pafs.SubTreeFileSystem(str(root), pafs.LocalFileSystem())
    monkeypatch.setattr(runner_utils, "_filesystem_and_path", lambda uri: (store, uri.split("://", 1)[1]))
    return root


def test_output_location_keeps_uri_and_resolves_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner_utils.output_location("gs://b/run/shard/") == "gs://b/run/shard"
    assert runner_utils.output_location("rel/out") == Path("rel/out")
    assert runner_utils.output_location("rel/out", resolve=True) == (tmp_path / "rel" / "out").resolve()


def test_make_output_dir_does_not_create_gs_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner_utils.make_output_dir(runner_utils.output_location("gs://b/run/shard", resolve=True))
    assert not (tmp_path / "gs:").exists()
    runner_utils.make_output_dir(runner_utils.output_location("local/out", resolve=True))
    assert (tmp_path / "local" / "out").is_dir()


def test_join_location():
    assert runner_utils.join_location("gs://b/run", "salt.bin") == "gs://b/run/salt.bin"
    assert runner_utils.join_location(Path("/x"), "salt.bin") == Path("/x/salt.bin")


@pytest.mark.parametrize("kind", ["local", "uri"])
def test_text_file_round_trip(kind, tmp_path, fake_object_store):
    """salt.bin / key.bin style files are written and read back for both path kinds."""
    location = tmp_path / "key.bin" if kind == "local" else "gs://run/shard/key.bin"
    if kind == "uri":
        (fake_object_store / "run" / "shard").mkdir(parents=True)
    runner_utils.write_text_file(location, "11" * 32)
    assert runner_utils.read_text_file(location) == "11" * 32


def test_load_key_reads_from_uri(fake_object_store):
    (fake_object_store / "run").mkdir()
    runner_utils.write_text_file("gs://run/key.bin", "11" * 32)
    assert lr.LocalJobRunner()._load_key("gs://run/key.bin") == bytes.fromhex("11" * 32)


def test_list_parquet_files_local_and_uri(tmp_path, fake_object_store):
    for base in (tmp_path / "local", fake_object_store / "run"):
        (base / "sub").mkdir(parents=True)
        pq.write_table(pa.table({"a": [1]}), base / "a.parquet")
        pq.write_table(pa.table({"a": [1]}), base / "sub" / "b.parquet")
        (base / "_marker").write_text("x")

    local = runner_utils.list_parquet_files(tmp_path / "local")
    assert [Path(f).name for f in local] == ["a.parquet", "b.parquet"]
    assert [Path(f).name for f in runner_utils.list_parquet_files(tmp_path / "local", recursive=False)] == ["a.parquet"]

    remote = runner_utils.list_parquet_files("gs://run")
    assert all(f.startswith("gs://") for f in remote)
    assert [Path(f).name for f in remote] == ["a.parquet", "b.parquet"]


def test_resolve_input_files_and_detect_columns_accept_uri(fake_object_store):
    (fake_object_store / "run").mkdir()
    pq.write_table(pa.table({"NOTE_TEXT": ["x"], "row_id": ["r"]}), fake_object_store / "run" / "a.parquet")

    assert runner_utils.resolve_input_files("gs://run") == ["gs://run/a.parquet"]
    assert runner_utils.resolve_input_files("gs://run/a.parquet") == ["gs://run/a.parquet"]
    assert runner_utils.resolve_input_files("gs://run/missing") == []
    assert runner_utils.detect_columns("gs://run/a.parquet", ["note_text"], ["row_id"]) == ["NOTE_TEXT", "row_id"]


# ---------------------------------------------------------------------------
# Checkpoint location
# ---------------------------------------------------------------------------


def test_configure_checkpoint_uri_and_local(tmp_path):
    ctx = SimpleNamespace(checkpoint_config=None)
    _configure_checkpoint(ctx, enable=True, output_dir="gs://b/run/shard", id_column="row_id")
    assert ctx.checkpoint_config.checkpoint_path == "gs://b/run/shard_ray_checkpoint"

    _configure_checkpoint(ctx, enable=True, output_dir=tmp_path / "shard", id_column="row_id")
    assert ctx.checkpoint_config.checkpoint_path == str(tmp_path / "shard_ray_checkpoint")


# ---------------------------------------------------------------------------
# Stage functions pass gs:// through unchanged
# ---------------------------------------------------------------------------


def _fake_ray_stage(monkeypatch, *, source_columns):
    """Stub Ray so a stage function runs without a cluster; return what the stage handed to Ray."""
    seen: dict = {}
    ctx = SimpleNamespace(checkpoint_config=None)
    fake_ds = MagicMock()
    fake_ds.map_batches.return_value = fake_ds

    def fake_write(path, **_kwargs):
        seen["write_path"] = path
        seen["checkpoint_path"] = ctx.checkpoint_config.checkpoint_path if ctx.checkpoint_config else None

    def fake_read(files, **kwargs):
        seen["read_kwargs"] = kwargs
        return fake_ds

    fake_ds.write_parquet.side_effect = fake_write
    monkeypatch.setattr(ray.data, "read_parquet", fake_read)
    monkeypatch.setattr(ray.data.DataContext, "get_current", staticmethod(lambda: ctx))
    monkeypatch.setattr(lr, "configure_data_context", lambda **_k: None)
    monkeypatch.setattr(lr, "detect_columns", lambda *_a, **_k: list(source_columns))
    monkeypatch.setattr(lr, "resolve_input_files", lambda _p: ["gs://b/in/a.parquet"])
    runner = lr.LocalJobRunner()
    monkeypatch.setattr(runner, "_init_ray", lambda: None)
    return runner, seen


def test_run_recognition_passes_gs_paths_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner, seen = _fake_ray_stage(monkeypatch, source_columns=["text_hash", "note_text"])

    runner.run_recognition(
        "gs://b/in", "gs://b/run/shard", num_actors=1, num_cpus=None, worker_num_cpus=1, enable_checkpoint=True
    )

    assert seen["write_path"] == "gs://b/run/shard"
    assert seen["checkpoint_path"] == "gs://b/run/shard_ray_checkpoint"
    assert not (tmp_path / "gs:").exists()


def test_run_recognition_local_relative_path_still_resolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner, seen = _fake_ray_stage(monkeypatch, source_columns=["text_hash", "note_text"])

    runner.run_recognition("in", "rel/shard", num_actors=1, num_cpus=None, worker_num_cpus=1, enable_checkpoint=True)

    assert seen["write_path"] == str((tmp_path / "rel" / "shard").resolve())
    assert seen["checkpoint_path"] == str((tmp_path / "rel" / "shard_ray_checkpoint").resolve())
    assert (tmp_path / "rel" / "shard").is_dir()


def _stub_pipeline_stages(runner, monkeypatch):
    calls: dict = {}
    monkeypatch.setattr(runner, "_init_ray", lambda: None)
    monkeypatch.setattr(runner, "_apply_pipeline_recommendations", lambda *_a, **_k: None)
    for name in ("run_transformer", "run_recognition", "run_anonymization"):
        monkeypatch.setattr(runner, name, lambda _name=name, **k: (calls.setdefault(_name, k), {})[1])
    return calls


def test_run_pipeline_with_gs_output_dir(tmp_path, monkeypatch, fake_object_store):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "in.parquet"
    pq.write_table(pa.table({"note_text": ["a note"]}), src)
    (fake_object_store / "run" / "shard").mkdir(parents=True)
    runner = lr.LocalJobRunner()
    calls = _stub_pipeline_stages(runner, monkeypatch)

    runner.run_pipeline(str(src), "gs://run/shard/", "model")

    assert calls["run_transformer"]["output_path"] == "gs://run/shard/02_transformer_output"
    assert calls["run_recognition"]["input_path"] == "gs://run/shard/02_transformer_output"
    assert calls["run_anonymization"]["input_path"] == "gs://run/shard/04_recognizer_output"
    assert calls["run_anonymization"]["salt_path"] == "gs://run/shard/salt.bin"
    assert (fake_object_store / "run" / "shard" / "key.bin").read_text() == "11" * 32
    assert not (tmp_path / "gs:").exists()


def test_run_pipeline_gs_output_dir_rejects_local_only_options(tmp_path, monkeypatch):
    src = tmp_path / "in.parquet"
    pq.write_table(pa.table({"note_text": ["a note"]}), src)
    runner = lr.LocalJobRunner()
    _stub_pipeline_stages(runner, monkeypatch)

    with pytest.raises(ValueError, match="local output_dir"):
        runner.run_pipeline(str(src), "gs://run/shard", "model", produce_visualizer_json=True)
    with pytest.raises(ValueError, match="local output_dir"):
        runner.run_pipeline(str(src), "gs://run/shard", "model", execution_mode="streamed")


# ---------------------------------------------------------------------------
# Derived row_id: where it is computed and what is logged
# ---------------------------------------------------------------------------


def test_read_stage_derives_ids_in_read_hook(monkeypatch):
    seen: dict = {}
    fake_ds = MagicMock()
    monkeypatch.setattr(ray.data, "read_parquet", lambda _files, **kw: (seen.update(kw), fake_ds)[1])

    _read_stage_source(["f.parquet"], columns=["note_text"], normalize=True, derive_ids_in_read=True)
    assert seen["_block_udf"] is normalize_source_batch
    fake_ds.map_batches.assert_not_called()

    seen.clear()
    _read_stage_source(["f.parquet"], columns=["text_hash"], normalize=False, derive_ids_in_read=True)
    assert seen["_block_udf"] is add_row_id

    seen.clear()
    _read_stage_source(["f.parquet"], columns=["note_text"], normalize=True)
    assert "_block_udf" not in seen
    fake_ds.map_batches.assert_called_once()


def test_derived_row_id_warning(monkeypatch, caplog):
    monkeypatch.setattr(ray.data, "read_parquet", lambda *_a, **_k: MagicMock())
    with caplog.at_level(logging.WARNING, logger="tide2.runner.local_runner"):
        _read_stage_source(["f"], columns=["note_text"], normalize=True, derive_ids_in_read=True)
    assert "row_id was not supplied" in caplog.text
    assert "does not deduplicate" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="tide2.runner.local_runner"):
        _read_stage_source(["f"], columns=["note_text", "ROW_ID"], normalize=True, derive_ids_in_read=True)
    assert "row_id was not supplied" not in caplog.text


# ---------------------------------------------------------------------------
# Contract with Ray's read hook (real Ray, both Parquet readers)
# ---------------------------------------------------------------------------


@pytest.fixture(params=[False, True], ids=["v1", "v2"])
def datasource_v2(request):
    ctx = ray.data.DataContext.get_current()
    previous = ctx.use_datasource_v2
    ctx.use_datasource_v2 = request.param
    yield request.param
    ctx.use_datasource_v2 = previous


def _write_source(path: Path, n: int = 20) -> None:
    pq.write_table(
        pa.table({"note_text": [f"note {i}" for i in range(n)], "patient_id": [f"p{i % 3}" for i in range(n)]}),
        path,
    )


@pytest.mark.integration
def test_in_read_ids_match_public_helper(tmp_path, datasource_v2):
    src = tmp_path / "in.parquet"
    _write_source(src)

    ds = _read_stage_source(
        [str(src)], columns=["note_text", "patient_id"], normalize=True, derive_ids_in_read=True, num_blocks=2
    )
    rows = ds.take_all()

    expected = normalize_source_batch(pq.read_table(src))
    assert sorted(r["row_id"] for r in rows) == sorted(expected["row_id"].to_pylist())
    assert sorted(r["text_hash"] for r in rows) == sorted(expected["text_hash"].to_pylist())


@pytest.mark.integration
def test_select_columns_after_hook_read_is_a_known_hazard(tmp_path, datasource_v2):
    """Projection pushdown hides derived columns from the hook; no checkpointed stage selects right after the read."""
    src = tmp_path / "in.parquet"
    _write_source(src)
    ds = _read_stage_source([str(src)], columns=["note_text", "patient_id"], normalize=True, derive_ids_in_read=True)
    with pytest.raises(Exception, match=r"row_id|does not exist in schema"):
        ds.select_columns(["row_id"]).take_all()


# ---------------------------------------------------------------------------
# Kill and restart: every source row is written exactly once
# ---------------------------------------------------------------------------

_DRIVER = textwrap.dedent(
    """
    import sys, time, uuid
    from pathlib import Path

    import ray
    import ray.data

    from tide2.runner.local_runner import _configure_checkpoint, _read_stage_source
    from tide2.runner.utils import output_location

    src, out, seen = sys.argv[1], output_location(sys.argv[2]), Path(sys.argv[3])
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    ctx = ray.data.DataContext.get_current()
    ctx.use_datasource_v2 = False
    _configure_checkpoint(ctx, enable=True, output_dir=out, id_column="row_id")


    def work(batch):
        (seen / f"{uuid.uuid4().hex}.txt").write_text("\\n".join(batch["row_id"].to_pylist()))
        time.sleep(0.1)
        return batch


    ds = _read_stage_source(
        [src], columns=["note_text", "patient_id"], normalize=True, derive_ids_in_read=True, num_blocks=40
    )
    ds = ds.map_batches(work, batch_format="pyarrow", batch_size=100, num_cpus=1)
    ds.write_parquet(str(out), compression="zstd")
    """
)

_GCS_PREFIX_ENV = "TIDE2_TEST_GCS_PREFIX"


def _processed_rows(seen: Path) -> list[str]:
    return [line for f in seen.glob("*.txt") for line in f.read_text().splitlines() if line]


def _kill_everything_under(*roots: Path) -> None:
    for root in roots:
        subprocess.run(["pkill", "-9", "-f", str(root)], check=False)  # noqa: S603, S607


def _list_files(location: str) -> list[str]:
    if not runner_utils.is_uri(location):
        return [str(f) for f in Path(location).rglob("*") if f.is_file()]
    filesystem, path = runner_utils._filesystem_and_path(location)
    selector = pafs.FileSelector(path, recursive=True, allow_not_found=True)
    return [i.path for i in filesystem.get_file_info(selector) if i.type == pafs.FileType.File]


def _read_output_row_ids(location: str) -> list[str]:
    files = runner_utils.list_parquet_files(location, recursive=False)
    if runner_utils.is_uri(location):
        filesystem, _ = runner_utils._filesystem_and_path(location)
        tables = [pq.read_table(f.split("://", 1)[1], filesystem=filesystem) for f in files]
    else:
        tables = [pq.read_table(f) for f in files]
    return pa.concat_tables(tables)["row_id"].to_pylist() if tables else []


def _delete_uri_dir(location: str) -> None:
    filesystem, path = runner_utils._filesystem_and_path(location)
    filesystem.delete_dir(path)


@pytest.fixture(params=["local", "gcs"])
def shard_location(request, tmp_path):
    """The output location of one shard: a local directory, or a unique ``gs://`` prefix when one is configured."""
    if request.param == "local":
        yield str(tmp_path / "shard")
        return
    prefix = os.environ.get(_GCS_PREFIX_ENV)
    if not prefix:
        pytest.skip(f"set {_GCS_PREFIX_ENV}=gs://bucket/prefix to run against object storage")
    run_root = f"{prefix.rstrip('/')}/tide2-test-{uuid.uuid4().hex}"
    yield f"{run_root}/shard"
    _delete_uri_dir(run_root)


@pytest.mark.integration
def test_sigkill_then_restart_writes_every_row_exactly_once(tmp_path, shard_location):
    # Ray's unix sockets need a short path, and a dedicated tmpdir scopes the cleanup to this test's processes.
    ray_tmp = Path(tempfile.mkdtemp(prefix="rt_", dir="/tmp"))
    try:
        _run_sigkill_then_restart(tmp_path, ray_tmp, shard_location)
    finally:
        _kill_everything_under(ray_tmp, tmp_path)
        shutil.rmtree(ray_tmp, ignore_errors=True)


def _run_sigkill_then_restart(tmp_path: Path, ray_tmp: Path, out: str) -> None:
    n_rows = 3000
    src = tmp_path / "in.parquet"
    pq.write_table(
        pa.table({"note_text": [f"note {i}" for i in range(n_rows)], "patient_id": [f"p{i}" for i in range(n_rows)]}),
        src,
    )
    seen = tmp_path / "seen"
    seen.mkdir()
    checkpoint_dir = f"{out}_ray_checkpoint"
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)
    env = {**os.environ, "RAY_TMPDIR": str(ray_tmp)}
    command = [sys.executable, str(driver), str(src), out, str(seen)]

    first = subprocess.Popen(command, env=env, start_new_session=True)  # noqa: S603
    try:
        deadline = time.time() + 180
        while len([f for f in _list_files(checkpoint_dir) if not f.endswith(".pending")]) < 4:
            if first.poll() is not None or time.time() > deadline:
                pytest.fail("first run ended before it could be killed mid-way")
            time.sleep(0.2)
    finally:
        _kill_everything_under(ray_tmp, tmp_path)
        first.wait()

    rows_after_kill = len(_read_output_row_ids(out))
    assert 0 < rows_after_kill < n_rows
    processed_first = len(_processed_rows(seen))

    subprocess.run(command, env=env, check=True, timeout=300)  # noqa: S603

    expected = normalize_source_batch(pq.read_table(src))["row_id"].to_pylist()
    assert sorted(_read_output_row_ids(out)) == sorted(expected)
    # Interrupted blocks are redone, but committed ones are not: less work is repeated than was done before the kill.
    total_processed = len(_processed_rows(seen))
    assert total_processed - n_rows < processed_first

    subprocess.run(command, env=env, check=True, timeout=300)  # noqa: S603
    assert len(_processed_rows(seen)) == total_processed
