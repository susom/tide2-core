"""Discrete vs streamed equivalence for ``run_pipeline``.

``execution_mode="streamed"`` changes orchestration only: no line under
``process_batch``, the recognizers, the anonymizers, span resolution, HIPS, FPE
or date jitter is touched. So the two modes must produce **equivalent** output,
which this asserts per ``row_id``:

- identical ``anonymized_note_text``, ``entity_count``,
  ``anonymizer_results_json`` (modulo key order) and ``processing_status``;
- **equal row sets**, not merely overlapping;
- matching entity totals by type.

Byte parity is not expected: ``processing_timestamp`` is wall-clock and row
order is not preserved (``preserve_order=False``, and streaming changes block
scheduling). Everything else is deterministic, because anonymization is keyed
off ``salt``/``key``/``patient_id`` and jitter is derived per patient.

The GPU transformer is replaced by a deterministic stub actor — both modes go
through ``tide2.actors.create_transformer_actor``, so the stub exercises the
same chaining without a model download. The recognizer and anonymizer stages
are the real ones.
"""

import json
import shutil
import sys
from collections import Counter
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest
import ray
from ray import cloudpickle

import tide2.actors
import tide2.transformers.config
import tide2.utils.gcs_resource_manager
from tide2.runner.local_runner import LocalJobRunner

pytestmark = pytest.mark.integration
KEEP_RAY_BETWEEN_TESTS = True

# Ray pickles actor classes by reference, and `tests/` is not importable inside a
# Ray worker. Serializing this module by value ships the stub actor with the task.
cloudpickle.register_pickle_by_value(sys.modules[__name__])

SALT_HEX = "ab" * 32
KEY_HEX = "cd" * 32
STUB_ENTITY = "John Doe"


class StubTransformerActor:
    """Deterministic stand-in for the GPU NER actor.

    Emits the same output shape as ``TransformerInferenceActor`` in its
    ``aggregate_bio=True`` mode (``num_agg_actors=0``), tagging every literal
    occurrence of ``STUB_ENTITY`` as a PERSON.
    """

    def __call__(self, batch):
        note_texts = [str(t) for t in batch["note_text"]]
        results, counts = [], []
        for text in note_texts:
            entities = []
            start = text.find(STUB_ENTITY)
            while start != -1:
                entities.append(
                    {
                        "entity_type": "PERSON",
                        "start": start,
                        "end": start + len(STUB_ENTITY),
                        "score": 0.99,
                        "analysis_explanation": None,
                        "recognition_metadata": {},
                    }
                )
                start = text.find(STUB_ENTITY, start + 1)
            results.append(json.dumps(entities))
            counts.append(len(entities))

        timestamp = datetime.now(tz=UTC).isoformat()
        statuses = [str(s) for s in batch.get("processing_status", ["success"] * len(note_texts))]
        failed = [s == "failed" for s in statuses]
        res = {
            "text_hash": [str(h) for h in batch["text_hash"]],
            "patient_id": [str(p) for p in batch["patient_id"]],
            "note_text": note_texts,
            "recognizer_results_json": [None if f else r for r, f in zip(results, failed, strict=True)],
            "entity_count": [None if f else c for c, f in zip(counts, failed, strict=True)],
            "processing_timestamp": [timestamp] * len(note_texts),
        }
        for col in ("patient_identifiers", "jitter", "row_id", "processing_status", "stage_status_json"):
            if col in batch:
                res[col] = list(batch[col])
        return res


@pytest.fixture(scope="module", autouse=True)
def shared_ray():
    """One Ray cluster for the whole module: starting and stopping Ray is most of a run's cost."""
    ray.init(num_cpus=8, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LocalJobRunner, "shutdown", lambda _self: None)
        yield
    ray.shutdown()


@pytest.fixture(scope="module")
def stub_transformer():
    """Point both execution modes at the stub actor and skip the model download."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tide2.actors, "create_transformer_actor", lambda **_k: StubTransformerActor)
        patch.setattr(
            tide2.transformers.config,
            "load_model_config",
            lambda _name: {"MODEL_MAX_LENGTH": 512, "CHUNK_OVERLAP_SIZE": 40},
        )
        patch.setattr(tide2.utils.gcs_resource_manager, "resolve_model_path", lambda **_k: "/nonexistent/model")
        yield


def _rows_with_failures(*, with_results):
    """Six rows; the even ones arrive marked failed by an earlier stage."""
    failed_stage = json.dumps({"transformer": {"status": "failed", "reason": "forced"}})
    ok_stage = json.dumps({"transformer": {"status": "success"}})
    rows = []
    for i in range(6):
        failed = i % 2 == 0
        row = {
            "text_hash": f"h{i}",
            "note_text": f"Patient {STUB_ENTITY} seen {i}.",
            "patient_id": f"p{i}",
            "row_id": f"r{i}",
            "processing_status": "failed" if failed else "success",
            "stage_status_json": failed_stage if failed else ok_stage,
        }
        if with_results:
            row["recognizer_results_json"] = json.dumps([{"entity_type": "PERSON", "start": 8, "end": 16, "score": 1}])
            row["entity_count"] = 1
        rows.append(row)
    return rows


def read_output(output_dir):
    files = list((output_dir / "anonymizer_output").glob("**/*.parquet"))
    assert files, f"no anonymizer output under {output_dir}"
    return pads.dataset(files).to_table().to_pandas()


def run_mode(tmp_path, notes, mode, *, checkpoint=False, name=None):
    in_file = tmp_path / f"notes_{len(notes)}.parquet"
    if not in_file.exists():
        notes.to_parquet(in_file, index=False)
    out = tmp_path / (name or mode)
    runner = LocalJobRunner(hardware_autotune=False)
    try:
        manifest = runner.run_pipeline(
            input_path=str(in_file),
            output_dir=str(out),
            model_name="stub-model",
            salt_hex=SALT_HEX,
            key_hex=KEY_HEX,
            execution_mode=mode,
            transformer_kwargs={"num_gpus": 0, "num_transformer_actors": 1, "num_agg_actors": 0},
            recognizer_kwargs={"num_actors": 2, "enable_checkpoint": checkpoint},
            anonymizer_kwargs={"num_actors": 2, "enable_checkpoint": checkpoint},
        )
    finally:
        runner.shutdown()
    return manifest, read_output(out)


def canonical_items(raw):
    """Anonymizer items with dict key order normalized away."""
    return [json.dumps(item, sort_keys=True) for item in json.loads(raw)]


def entity_totals(df):
    totals = Counter()
    for raw in df["anonymizer_results_json"]:
        for item in json.loads(raw):
            totals[item["entity_type"]] += 1
    return totals


@pytest.fixture(scope="module")
def shared_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("parity")


@pytest.fixture(scope="module")
def equivalence(shared_dir, notes, stub_transformer):
    """One discrete and one streamed run, shared by the tests that compare or inspect them."""
    return {mode: run_mode(shared_dir, notes, mode) for mode in ("discrete", "streamed")}


class TestDiscreteStreamedEquivalence:
    def test_equivalent_output(self, notes, equivalence):
        discrete_manifest, discrete = equivalence["discrete"]
        streamed_manifest, streamed = equivalence["streamed"]

        assert discrete_manifest["execution_mode"] == "discrete"
        assert streamed_manifest["execution_mode"] == "streamed"
        assert streamed_manifest["input_rows"] == len(notes)
        assert streamed_manifest["dropped_rows"] == 0

        # Equal row sets, not merely overlapping.
        assert set(discrete["row_id"]) == set(streamed["row_id"])
        assert len(discrete) == len(streamed) == len(notes)

        d = discrete.set_index("row_id").sort_index()
        s = streamed.set_index("row_id").sort_index()
        for col in ("anonymized_note_text", "entity_count", "processing_status"):
            pd.testing.assert_series_equal(d[col], s[col], check_names=False)
        for row_id in d.index:
            assert canonical_items(d.loc[row_id, "anonymizer_results_json"]) == canonical_items(
                s.loc[row_id, "anonymizer_results_json"]
            )

        assert entity_totals(discrete) == entity_totals(streamed)

    def test_streamed_writes_no_intermediates(self, shared_dir, equivalence):
        out = shared_dir / "streamed"
        assert (out / "anonymizer_output").exists()
        assert not (out / "transformer_output").exists()
        assert not (out / "recognizer_output").exists()

    def test_raw_note_text_never_reaches_the_sink(self, equivalence):
        _, streamed = equivalence["streamed"]
        assert "note_text" not in streamed.columns
        assert "anonymized_note_text" in streamed.columns


@pytest.fixture(scope="module")
def checkpointed(shared_dir, notes, stub_transformer):
    return run_mode(shared_dir, notes, "streamed", checkpoint=True, name="checkpointed")


class TestStreamedCheckpoint:
    def test_checkpointed_streamed_run_matches_discrete(self, shared_dir, equivalence, checkpointed):
        manifest, streamed = checkpointed
        _, discrete = equivalence["discrete"]

        assert manifest["dropped_rows"] == 0
        assert set(discrete["row_id"]) == set(streamed["row_id"])
        assert (shared_dir / "checkpointed" / "anonymizer_output_ray_checkpoint").exists()

    def test_rerun_into_the_same_folder_adds_no_rows(self, shared_dir, notes, checkpointed):
        _, first = checkpointed
        manifest, second = run_mode(shared_dir, notes, "streamed", checkpoint=True, name="checkpointed")

        assert manifest["output_rows"] == len(notes)
        assert second["row_id"].is_unique
        assert set(first["row_id"]) == set(second["row_id"])

    def test_rerun_with_more_rows_processes_only_the_new_ones(self, shared_dir, notes, stub_transformer):
        """A larger input stands in for a run that stopped after the first rows."""
        run_mode(shared_dir, notes.iloc[:3], "streamed", checkpoint=True, name="resume")
        before = {f.name for f in (shared_dir / "resume" / "anonymizer_output").glob("**/*.parquet")}

        manifest, resumed = run_mode(shared_dir, notes, "streamed", checkpoint=True, name="resume")

        after = {f.name for f in (shared_dir / "resume" / "anonymizer_output").glob("**/*.parquet")}
        assert before <= after and before != after
        assert manifest["output_rows"] == len(notes)
        assert resumed["row_id"].is_unique
        assert len(resumed) == len(notes)

    def test_rerun_without_a_checkpoint_is_refused_after_it_appends(self, shared_dir, notes, equivalence):
        shutil.copytree(shared_dir / "streamed", shared_dir / "refused")
        with pytest.raises(RuntimeError, match="fresh output_dir"):
            run_mode(shared_dir, notes, "streamed", checkpoint=False, name="refused")


class TestRowReconciliation:
    def test_total_failure_is_a_hard_error(self, tmp_path, notes, stub_transformer, monkeypatch):
        """A short write must not look like success."""

        class ExplodingAnonymizer:
            def __init__(self, *_a, **_k):
                raise RuntimeError("deliberate failure")

        monkeypatch.setattr(tide2.actors, "create_anonymizer_actor_class", lambda **_k: ExplodingAnonymizer)
        with pytest.raises(Exception, match=r"0 rows|deliberate failure|failed"):
            run_mode(tmp_path, notes, "streamed")


class TestUpstreamFailuresSurviveStreamedSource:
    """A row marked failed upstream must stay failed when the first stage reads it from the source."""

    @pytest.mark.parametrize("first_stage", ["recognizer", "anonymizer"])
    def test_stage_jobs_keep_failed_rows_failed(self, tmp_path, first_stage):
        rows = _rows_with_failures(with_results=first_stage == "anonymizer")
        in_file = tmp_path / "in.parquet"
        pd.DataFrame(rows).to_parquet(in_file, index=False)

        runner = LocalJobRunner()
        try:
            if first_stage == "recognizer":
                runner.run_recognition(str(in_file), str(tmp_path / "out"), num_actors=2, enable_checkpoint=False)
            else:
                (tmp_path / "salt.bin").write_text(SALT_HEX)
                (tmp_path / "key.bin").write_text(KEY_HEX)
                runner.run_anonymization(
                    str(in_file),
                    str(tmp_path / "out"),
                    salt_path=str(tmp_path / "salt.bin"),
                    key_path=str(tmp_path / "key.bin"),
                    num_actors=2,
                    enable_checkpoint=False,
                )
        finally:
            runner.shutdown()

        files = list((tmp_path / "out").glob("**/*.parquet"))
        table = pads.dataset(files).to_table().to_pandas().set_index("row_id").sort_index()
        statuses = table["processing_status"].to_dict()
        assert statuses == {f"r{i}": "failed" if i % 2 == 0 else "success" for i in range(6)}

    def test_failed_rows_match_between_discrete_and_streamed(self, tmp_path, stub_transformer):
        in_file = tmp_path / "in.parquet"
        pd.DataFrame(_rows_with_failures(with_results=False)).to_parquet(in_file, index=False)

        statuses = {}
        for mode in ("discrete", "streamed"):
            runner = LocalJobRunner(hardware_autotune=False)
            try:
                runner.run_pipeline(
                    input_path=str(in_file),
                    output_dir=str(tmp_path / mode),
                    model_name="stub-model",
                    salt_hex=SALT_HEX,
                    key_hex=KEY_HEX,
                    execution_mode=mode,
                    transformer_kwargs={"num_gpus": 0, "num_transformer_actors": 1, "num_agg_actors": 0},
                    recognizer_kwargs={"num_actors": 2, "enable_checkpoint": False},
                    anonymizer_kwargs={"num_actors": 2, "enable_checkpoint": False},
                )
            finally:
                runner.shutdown()
            table = read_output(tmp_path / mode).set_index("row_id").sort_index()
            statuses[mode] = table["processing_status"].to_dict()

        assert statuses["streamed"] == statuses["discrete"]
        assert statuses["streamed"]["r0"] == "failed"
        assert statuses["streamed"]["r1"] == "success"


def _write_recognizer_rows(folder, notes, name):
    folder.mkdir(parents=True, exist_ok=True)
    results = json.dumps([{"entity_type": "PERSON", "start": 8, "end": 16, "score": 0.9}])
    pq.write_table(
        pa.table(
            {
                "row_id": [f"r{i}" for i in range(len(notes))],
                "text_hash": [f"h{i}" for i in range(len(notes))],
                "note_text": list(notes["note_text"]),
                "patient_id": list(notes["patient_id"]),
                "recognizer_results_json": [results] * len(notes),
            }
        ),
        folder / f"{name}.parquet",
    )


class TestMergeRerun:
    """A rerun of merge mode replaces merged_recognizer_output; it never appends to it."""

    @pytest.fixture
    def merge_run(self, tmp_path, notes, monkeypatch):
        in_file = tmp_path / "in.parquet"
        notes.to_parquet(in_file, index=False)
        out = tmp_path / "out"
        state = {"write_sources": True}

        def fake_stage(name):
            def run(_self, **kwargs):
                if state["write_sources"] and name in ("regex", "llm"):
                    _write_recognizer_rows(Path(kwargs["output_path"]), notes, name)
                return {}

            return run

        for method, name in [
            ("run_transformer", "transformer"),
            ("run_recognition", "regex"),
            ("run_llm_recognition", "llm"),
            ("run_anonymization", "anonymizer"),
        ]:
            monkeypatch.setattr(LocalJobRunner, method, fake_stage(name))

        def run():
            return LocalJobRunner(hardware_autotune=False).run_pipeline(
                input_path=str(in_file),
                output_dir=str(out),
                model_name="stub-model",
                llm_recognizer_mode="merge",
                llm_recognizer_kwargs={"project_id": "p"},
            )

        return out, run, state

    @staticmethod
    def _merged_files(out):
        return sorted((out / "merged_recognizer_output").glob("**/*.parquet"))

    def test_rerun_replaces_the_merged_rows(self, notes, merge_run):
        out, run, state = merge_run
        run()
        pq.write_table(pa.table({"row_id": ["stray"]}), out / "merged_recognizer_output" / "stray.parquet")
        state["write_sources"] = False

        run()

        merged = pads.dataset(self._merged_files(out)).to_table().to_pandas()
        assert len(merged) == len(notes)
        assert merged["row_id"].is_unique
        assert "stray" not in set(merged["row_id"])

    def test_missing_source_leaves_the_merged_folder_alone(self, merge_run):
        out, run, state = merge_run
        run()
        before = self._merged_files(out)
        for f in (out / "llm_recognizer_output").glob("**/*.parquet"):
            f.unlink()
        state["write_sources"] = False

        with pytest.raises(RuntimeError, match="needs output from both"):
            run()

        assert self._merged_files(out) == before
