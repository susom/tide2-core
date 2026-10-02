"""Tests for the ``build_*_stage`` seam on ``LocalJobRunner``.

The builders were extracted from the four ``run_*`` methods as pure code
motion, so the contract under test is *parity*: each builder must call
``Dataset.map_batches`` with exactly the arguments the wrapper used to pass
(actor class, batch size, pool strategy, ``ray_remote_args``). A regression
here is a discrete-path regression, which is the one thing the streamed-mode
work must not produce.

No live Ray cluster is needed — the ``Dataset`` is a double that records the
``map_batches`` calls made against it.
"""

from unittest.mock import MagicMock

import pytest
import ray

import tide2.runner.local_runner as lr


class FakeDataset:
    """Records ``map_batches`` calls and chains like a lazy ``Dataset``."""

    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []

    def map_batches(self, fn, **kwargs):
        self.calls.append({"fn": fn, **kwargs})
        return FakeDataset(self.calls)


@pytest.fixture
def runner():
    return lr.LocalJobRunner()


def pool_of(call):
    """Return (min_size, max_size) for the ActorPoolStrategy in a recorded call."""
    strategy = call["compute"]
    return strategy.min_size, strategy.max_size


class TestPoolStrategy:
    """``pool_min_size=None`` must reproduce the discrete fixed pool exactly."""

    def test_none_gives_fixed_pool(self, runner):
        strategy = runner._actor_pool(14, None)
        assert (strategy.min_size, strategy.max_size) == (14, 14)

    def test_min_size_gives_autoscaling_pool(self, runner):
        strategy = runner._actor_pool(10, 2)
        assert (strategy.min_size, strategy.max_size) == (2, 10)

    def test_min_size_clamped_to_max(self, runner):
        """A minimum above the pool cap must not produce an invalid pool."""
        strategy = runner._actor_pool(1, 4)
        assert (strategy.min_size, strategy.max_size) == (1, 1)


class TestRecognizerBuilder:
    def test_discrete_parity(self, runner):
        from tide2.actors import RecognizerActor

        ds = FakeDataset()
        runner.build_recognizer_stage(
            ds, batch_size=150, num_actors=14, ray_remote_args={"num_cpus": 3.0, "max_restarts": -1}
        )
        (call,) = ds.calls
        assert call["fn"] is RecognizerActor
        assert call["batch_size"] == 150
        assert call["num_cpus"] == 3.0
        assert call["max_restarts"] == -1
        assert pool_of(call) == (14, 14)

    def test_streamed_pool(self, runner):
        ds = FakeDataset()
        runner.build_recognizer_stage(
            ds, batch_size=150, num_actors=10, ray_remote_args={"num_cpus": 1.0}, pool_min_size=2
        )
        assert pool_of(ds.calls[0]) == (2, 10)


class TestLlmRecognizerBuilder:
    def test_discrete_parity(self, runner):
        from tide2.actors import LlmRecognizerActor

        ds = FakeDataset()
        ctor = {"project_id": "p", "model_name": "gemini-2.5-flash"}
        runner.build_llm_recognizer_stage(
            ds, batch_size=10, num_actors=4, ray_remote_args={"num_cpus": 1.0}, fn_constructor_kwargs=ctor
        )
        (call,) = ds.calls
        assert call["fn"] is LlmRecognizerActor
        assert call["batch_size"] == 10
        assert call["fn_constructor_kwargs"] == ctor
        assert pool_of(call) == (4, 4)


class TestAnonymizerBuilder:
    def test_discrete_parity(self, runner):
        ds = FakeDataset()
        actor_cls = MagicMock(name="AnonymizerActor")
        runner.build_anonymizer_stage(
            ds, actor_cls=actor_cls, batch_size=200, num_actors=14, ray_remote_args={"num_cpus": 3.0}
        )
        (call,) = ds.calls
        assert call["fn"] is actor_cls
        assert call["batch_size"] == 200
        assert call["num_cpus"] == 3.0
        assert pool_of(call) == (14, 14)


class TestTransformerBuilder:
    def test_single_operator_when_aggregating_in_actor(self, runner):
        ds = FakeDataset()
        actor = MagicMock(name="TransformerActor")
        runner.build_transformer_stage(
            ds,
            transformer_actor=actor,
            model_name="m",
            batch_size=8,
            num_transformer_actors=3,
            ray_remote_args_transformer={"num_gpus": 0.33},
            num_agg_actors=0,
            agg_num_cpus=1.0,
        )
        (call,) = ds.calls
        assert call["fn"] is actor
        assert call["batch_format"] == "numpy"
        assert call["num_gpus"] == 0.33
        assert pool_of(call) == (1, 3)

    def test_chains_aggregation_operator(self, runner):
        from tide2.actors import BIOAggregationActor

        ds = FakeDataset()
        runner.build_transformer_stage(
            ds,
            transformer_actor=MagicMock(),
            model_name="my-model",
            batch_size=8,
            num_transformer_actors=3,
            ray_remote_args_transformer={"num_gpus": 1},
            num_agg_actors=5,
            agg_num_cpus=0.5,
        )
        assert len(ds.calls) == 2
        agg = ds.calls[1]
        assert agg["fn"] is BIOAggregationActor
        assert agg["fn_constructor_kwargs"] == {"model_name": "my-model"}
        assert agg["num_cpus"] == 0.5
        assert pool_of(agg) == (5, 5)


class TestBuildersAreSideEffectFree:
    """Builders must not touch the DataContext, checkpoints, or the filesystem."""

    def test_no_data_context_access(self, runner, monkeypatch):
        def boom(*_a, **_k):
            raise AssertionError("builder touched the DataContext")

        monkeypatch.setattr(ray.data.DataContext, "get_current", staticmethod(boom))
        monkeypatch.setattr(lr, "configure_data_context", boom)

        ds = FakeDataset()
        runner.build_recognizer_stage(ds, batch_size=1, num_actors=1, ray_remote_args={})
        runner.build_anonymizer_stage(ds, actor_cls=MagicMock(), batch_size=1, num_actors=1, ray_remote_args={})
