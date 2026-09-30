"""Tests for ``execution_mode="streamed"`` plan construction.

Everything here is driver-side: the column contracts, the CPU admission check,
the fallback rules, and the shape of the chained operator sequence. No Ray
cluster and no model are involved — ``ray.data.from_pandas`` is replaced by a
``Dataset`` double that records the ``map_batches`` calls made against it and
raises if anything tries to execute the plan.
"""

from typing import Any
from typing import cast
from unittest.mock import MagicMock

import pandas as pd
import pytest
import ray

import tide2.runner.local_runner as lr
from tide2.runner.local_runner import ANONYMIZER_STAGE_COLUMNS
from tide2.runner.local_runner import LLM_RECOGNIZER_STAGE_COLUMNS
from tide2.runner.local_runner import RECOGNIZER_STAGE_COLUMNS
from tide2.runner.local_runner import TRANSFORMER_STAGE_COLUMNS
from tide2.runner.local_runner import check_streamed_admission
from tide2.runner.local_runner import validate_stage_columns

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class PlanDataset:
    """Lazy ``Dataset`` double. Records operators; explodes if executed."""

    def __init__(self, state: dict[str, Any] | None = None):
        self.state: dict[str, Any] = (
            state if state is not None else {"ops": [], "executed": False, "sink": None, "select": None}
        )

    @property
    def ops(self):
        return self.state["ops"]

    def repartition(self, num_blocks):
        self.state["num_blocks"] = num_blocks
        return PlanDataset(self.state)

    def map_batches(self, fn, **kwargs):
        self.state["ops"].append({"fn": fn, **kwargs})
        return PlanDataset(self.state)

    def select_columns(self, cols):
        self.state["select"] = list(cols)
        return PlanDataset(self.state)

    def write_parquet(self, path, **kwargs):
        self.state["sink"] = (path, kwargs)
        # Materialize a stand-in output so the row reconciliation has something
        # to count — it uses pyarrow, deliberately never ds.count().
        import pathlib

        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = self.state.get("rows", 0)
        pq.write_table(
            pa.table({"row_id": [str(i) for i in range(rows)]}),
            str(pathlib.Path(path) / "part-0.parquet"),
        )

    def schema(self):
        self.state["executed"] = True
        raise AssertionError("schema() executes upstream operators; the plan must not call it")

    def count(self):
        self.state["executed"] = True
        raise AssertionError("count() re-executes the plan")

    def stats(self):
        return "fake stats"


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "note_text": ["a note", "another note"],
            "text_hash": ["h1", "h2"],
            "patient_id": ["p1", "p2"],
            "patient_uid": ["p1", "p2"],
            "row_id": ["r1", "r2"],
        }
    )


@pytest.fixture
def streamed_env(monkeypatch, tmp_path):
    """Patch out Ray, the model, and the output counting for a plan-only run."""
    state = {}

    def fake_from_pandas(frame):
        state["source_columns"] = list(frame.columns)
        state["rows"] = len(frame)
        ds = PlanDataset()
        ds.state["rows"] = len(frame)
        state["ds"] = ds
        return ds

    monkeypatch.setattr(ray.data, "from_pandas", fake_from_pandas)
    monkeypatch.setattr(lr, "configure_data_context", lambda **_k: None)
    monkeypatch.setattr(
        ray.data.DataContext,
        "get_current",
        staticmethod(lambda: MagicMock(checkpoint_config="leftover")),
    )
    monkeypatch.setattr(lr, "_alive_node_cpus", lambda: [16.0])
    monkeypatch.setattr(lr.LocalJobRunner, "_init_ray", lambda _self: None)
    monkeypatch.setattr(lr.LocalJobRunner, "_apply_pipeline_recommendations", lambda *_a, **_k: None)

    transformer_actor = MagicMock(name="TransformerActor")
    monkeypatch.setattr(
        lr.LocalJobRunner,
        "_resolve_streamed_transformer_actor",
        lambda _self, _model_name, _t_kw: (transformer_actor, 3, 0, {"num_gpus": 0.33, "num_cpus": 1.0}),
    )
    monkeypatch.setattr(lr, "create_anonymizer_actor_class", MagicMock(return_value=MagicMock()), raising=False)
    state["transformer_actor"] = transformer_actor
    state["output_dir"] = tmp_path
    return state


def run_streamed(runner, df, output_dir, **overrides):
    """Invoke the streamed path with sensible defaults for the plan tests."""
    kwargs = {
        "df_input": df,
        "output_path": output_dir,
        "model_name": "some-model",
        "run_transformer": True,
        "run_recognizer": True,
        "run_anonymizer": True,
        "salt_hex": "00" * 32,
        "key_hex": "11" * 32,
        "transformer_kwargs": {},
        "recognizer_kwargs": {},
        "anonymizer_kwargs": {},
        "llm_recognizer_mode": "off",
        "llm_recognizer_kwargs": {},
        "hardware_autotune": False,
        "start_time": 0.0,
    }
    kwargs.update(overrides)
    return runner._run_pipeline_streamed(**kwargs)


# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------


class TestColumnContracts:
    def test_full_chain_validates(self):
        validate_stage_columns(
            ["note_text", "text_hash", "patient_uid", "row_id"],
            [
                ("transformer", TRANSFORMER_STAGE_COLUMNS),
                ("recognizer", RECOGNIZER_STAGE_COLUMNS),
                ("anonymizer", ANONYMIZER_STAGE_COLUMNS),
            ],
        )

    def test_missing_source_column_raises_naming_it(self):
        with pytest.raises(ValueError, match=r"'transformer'.*note_text"):
            validate_stage_columns(["text_hash"], [("transformer", TRANSFORMER_STAGE_COLUMNS)])

    def test_anonymizer_without_a_recognizer_stage_raises(self):
        """Skipping every recognizer stage leaves recognizer_results_json unavailable."""
        with pytest.raises(ValueError, match="recognizer_results_json"):
            validate_stage_columns(["note_text", "text_hash"], [("anonymizer", ANONYMIZER_STAGE_COLUMNS)])

    def test_anonymizer_accepts_precomputed_results(self):
        validate_stage_columns(
            ["note_text", "text_hash", "recognizer_results_json"],
            [("anonymizer", ANONYMIZER_STAGE_COLUMNS)],
        )

    def test_llm_only_chain_validates(self):
        validate_stage_columns(
            ["note_text", "text_hash"],
            [("llm_recognizer", LLM_RECOGNIZER_STAGE_COLUMNS), ("anonymizer", ANONYMIZER_STAGE_COLUMNS)],
        )

    def test_final_columns_exclude_raw_note_text(self):
        assert "note_text" not in lr.FINAL_OUTPUT_COLUMNS
        assert ANONYMIZER_STAGE_COLUMNS.produces | {"row_id"} >= lr.FINAL_OUTPUT_COLUMNS


# ---------------------------------------------------------------------------
# Admission check
# ---------------------------------------------------------------------------


class TestAdmissionCheck:
    def test_accepts_a_fitting_plan(self, monkeypatch):
        monkeypatch.setattr(lr, "_alive_node_cpus", lambda: [16.0])
        assert check_streamed_admission({"read": 0.25, "transformer": 1.0, "recognizer": 2.0}) == 16.0

    def test_raises_naming_the_operators_when_oversubscribed(self, monkeypatch):
        monkeypatch.setattr(lr, "_alive_node_cpus", lambda: [16.0])
        with pytest.raises(ValueError, match=r"recognizer=14\.0.*transformer=12\.0") as exc:
            check_streamed_admission({"transformer": 12.0, "recognizer": 14.0, "anonymizer": 14.0})
        assert "discrete" in str(exc.value)

    def test_refuses_small_boxes(self, monkeypatch):
        monkeypatch.setattr(lr, "_alive_node_cpus", lambda: [4.0])
        with pytest.raises(ValueError, match="small boxes deadlock"):
            check_streamed_admission({"read": 0.25})

    def test_refuses_multi_node(self, monkeypatch):
        monkeypatch.setattr(lr, "_alive_node_cpus", lambda: [16.0, 16.0])
        with pytest.raises(ValueError, match="single-node only"):
            check_streamed_admission({"read": 0.25})

    def test_uses_largest_node_not_cluster_total(self, monkeypatch):
        """Cluster totals sum across machines; actor placement is per node."""
        monkeypatch.setattr(lr, "_alive_node_cpus", lambda: [8.0])
        with pytest.raises(ValueError, match=r"usable budget 7\.0"):
            check_streamed_admission({"a": 4.0, "b": 4.0})


# ---------------------------------------------------------------------------
# Plan shape
# ---------------------------------------------------------------------------


class TestPlanShape:
    def test_full_chain_is_one_plan_with_a_single_sink(self, streamed_env, df):
        runner = lr.LocalJobRunner()
        result = run_streamed(runner, df, streamed_env["output_dir"])
        ds = streamed_env["ds"]

        assert [type(op["fn"]).__name__ or op["fn"] for op in ds.ops]  # three operators were chained
        assert len(ds.ops) == 3
        assert ds.ops[0]["fn"] is streamed_env["transformer_actor"]
        assert ds.state["sink"][0].endswith("06_anonymizer_output")
        assert result["execution_mode"] == "streamed"
        assert result["input_rows"] == 2

    def test_sink_projection_drops_note_text(self, streamed_env, df):
        runner = lr.LocalJobRunner()
        run_streamed(runner, df, streamed_env["output_dir"])
        assert "note_text" not in streamed_env["ds"].state["select"]
        assert "anonymized_note_text" in streamed_env["ds"].state["select"]
        # row_id is a pass-through no stage lists under `produces`; it must
        # still survive to the sink so downstream joins keep working.
        assert "row_id" in streamed_env["ds"].state["select"]

    def test_checkpointing_is_cleared_for_the_whole_plan(self, streamed_env, df, monkeypatch):
        ctx = MagicMock(checkpoint_config="leftover")
        monkeypatch.setattr(ray.data.DataContext, "get_current", staticmethod(lambda: ctx))
        run_streamed(lr.LocalJobRunner(), df, streamed_env["output_dir"])
        assert ctx.checkpoint_config is None

    def test_skipping_the_recognizer_chains_transformer_to_anonymizer(self, streamed_env, df):
        run_streamed(lr.LocalJobRunner(), df, streamed_env["output_dir"], run_recognizer=False)
        assert len(streamed_env["ds"].ops) == 2

    def test_llm_only_replaces_transformer_and_recognizer(self, streamed_env, df):
        from tide2.actors import LlmRecognizerActor

        run_streamed(
            lr.LocalJobRunner(),
            df,
            streamed_env["output_dir"],
            llm_recognizer_mode="only",
            llm_recognizer_kwargs={"project_id": "proj"},
        )
        ops = streamed_env["ds"].ops
        assert len(ops) == 2
        assert ops[0]["fn"] is LlmRecognizerActor
        # Network-bound: reserves CPU but barely uses it.
        assert ops[0]["num_cpus"] == 0.25

    def test_anonymizer_skipped_sinks_to_the_recognizer_output(self, streamed_env, df):
        run_streamed(lr.LocalJobRunner(), df, streamed_env["output_dir"], run_anonymizer=False)
        assert streamed_env["ds"].state["sink"][0].endswith("04_recognizer_output")

    def test_source_is_projected_to_the_first_stage_contract(self, streamed_env, df):
        df = df.assign(unused_column=["x", "y"])
        run_streamed(lr.LocalJobRunner(), df, streamed_env["output_dir"])
        assert "unused_column" not in streamed_env["source_columns"]
        assert "note_text" in streamed_env["source_columns"]

    def test_plan_is_never_executed_during_construction(self, streamed_env, df):
        """schema()/count() would execute upstream operators; both explode in the double."""
        run_streamed(lr.LocalJobRunner(), df, streamed_env["output_dir"])
        assert streamed_env["ds"].state["executed"] is False

    def test_missing_columns_raise_before_any_operator_is_built(self, streamed_env):
        bad = pd.DataFrame({"note_text": ["a"]})  # no text_hash
        with pytest.raises(ValueError, match="text_hash"):
            run_streamed(lr.LocalJobRunner(), bad, streamed_env["output_dir"])
        assert "ds" not in streamed_env

    def test_over_subscribed_node_raises_instead_of_hanging(self, streamed_env, df, monkeypatch):
        monkeypatch.setattr(lr, "_alive_node_cpus", lambda: [6.0])
        with pytest.raises(ValueError, match="reserves a minimum"):
            run_streamed(
                lr.LocalJobRunner(),
                df,
                streamed_env["output_dir"],
                recognizer_kwargs={"num_cpus": 2, "num_actors": 8},
            )


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


class TestFallbacks:
    @pytest.fixture
    def discrete_spy(self, monkeypatch):
        """Make the discrete path a no-op that records it was reached."""
        seen = {}

        def fake_prepare(input_data):
            return pd.DataFrame(
                {
                    "note_text": ["a"],
                    "text_hash": ["h"],
                    "patient_id": ["p"],
                    "patient_uid": ["p"],
                    "row_id": ["r"],
                }
            )

        monkeypatch.setattr(lr.LocalJobRunner, "_prepare_pipeline_input", staticmethod(fake_prepare))
        monkeypatch.setattr(lr.LocalJobRunner, "_init_ray", lambda _self: None)
        monkeypatch.setattr(lr.LocalJobRunner, "_apply_pipeline_recommendations", lambda *_a, **_k: None)
        monkeypatch.setattr(
            lr.LocalJobRunner, "run_transformer", lambda _self, **k: seen.setdefault("transformer", k) or {}
        )
        monkeypatch.setattr(
            lr.LocalJobRunner, "run_recognition", lambda _self, **k: seen.setdefault("recognizer", k) or {}
        )
        monkeypatch.setattr(
            lr.LocalJobRunner, "run_anonymization", lambda _self, **k: seen.setdefault("anonymizer", k) or {}
        )
        monkeypatch.setattr(
            lr.LocalJobRunner, "run_llm_recognition", lambda _self, **k: seen.setdefault("llm", k) or {}
        )
        monkeypatch.setattr(
            lr.LocalJobRunner,
            "_run_pipeline_streamed",
            lambda _self, **k: (seen.setdefault("streamed", k), {"execution_mode": "streamed"})[1],
        )
        return seen

    @pytest.mark.parametrize(
        ("overrides", "expected_in_warning"),
        [
            ({"anonymizer_kwargs": {"enable_checkpoint": True}}, "enable_checkpoint=True"),
            ({"llm_recognizer_mode": "merge", "llm_recognizer_kwargs": {"project_id": "p"}}, "merge"),
            ({"produce_visualizer_json": True}, "produce_visualizer_json=True"),
        ],
    )
    def test_falls_back_to_discrete_with_a_corrected_manifest(
        self, discrete_spy, tmp_path, caplog, overrides, expected_in_warning
    ):
        with caplog.at_level("WARNING"):
            result = lr.LocalJobRunner().run_pipeline(
                input_data=pd.DataFrame({"note_text": ["a"]}),
                output_dir=str(tmp_path),
                model_name="m",
                execution_mode="streamed",
                **overrides,
            )
        assert "streamed" not in discrete_spy
        assert result["execution_mode"] == "discrete"
        assert expected_in_warning in caplog.text
        assert "Falling back to discrete execution" in caplog.text

    def test_no_fallback_for_a_plain_streamed_run(self, discrete_spy, tmp_path):
        result = lr.LocalJobRunner().run_pipeline(
            input_data=pd.DataFrame({"note_text": ["a"]}),
            output_dir=str(tmp_path),
            model_name="m",
            execution_mode="streamed",
        )
        assert "streamed" in discrete_spy
        assert result["execution_mode"] == "streamed"

    def test_explicit_enable_checkpoint_false_does_not_fall_back(self, discrete_spy, tmp_path):
        lr.LocalJobRunner().run_pipeline(
            input_data=pd.DataFrame({"note_text": ["a"]}),
            output_dir=str(tmp_path),
            model_name="m",
            execution_mode="streamed",
            recognizer_kwargs={"enable_checkpoint": False},
        )
        assert "streamed" in discrete_spy

    def test_discrete_is_the_default_and_is_recorded(self, discrete_spy, tmp_path):
        result = lr.LocalJobRunner().run_pipeline(
            input_data=pd.DataFrame({"note_text": ["a"]}), output_dir=str(tmp_path), model_name="m"
        )
        assert result["execution_mode"] == "discrete"
        assert "streamed" not in discrete_spy

    def test_unknown_mode_raises(self, discrete_spy, tmp_path):
        with pytest.raises(ValueError, match="execution_mode"):
            lr.LocalJobRunner().run_pipeline(
                input_data=pd.DataFrame({"note_text": ["a"]}),
                output_dir=str(tmp_path),
                model_name="m",
                execution_mode=cast(Any, "fused"),
            )
