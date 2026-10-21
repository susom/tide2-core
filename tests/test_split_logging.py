"""Tests for tide2.utils.logging (stdout/stderr split of Ray and tide2 logs)."""

import logging

import pytest

from tide2.utils import logging as tl


@pytest.fixture
def isolated_loggers(monkeypatch):
    """Snapshot/restore ray loggers and the env vars the module touches."""
    monkeypatch.delenv(tl.SPLIT_LOG_STREAMS_ENV, raising=False)
    monkeypatch.delenv(tl.RAY_DATA_LOGGING_CONFIG_ENV, raising=False)
    saved = {}
    for name in ("ray", "ray.data"):
        lg = logging.getLogger(name)
        saved[name] = (list(lg.handlers), lg.propagate)
    yield
    for name, (handlers, propagate) in saved.items():
        lg = logging.getLogger(name)
        lg.handlers[:] = handlers
        lg.propagate = propagate


def _logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.handlers.clear()
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    lg.addHandler(tl.SplitStreamHandler())
    return lg


def test_levels_split_between_streams(capsys):
    lg = _logger("tide2_test_split")
    try:
        lg.debug("d")
        lg.info("i")
        lg.warning("w")
        lg.error("e")
        lg.critical("c")
    finally:
        lg.handlers.clear()
    out, err = capsys.readouterr()
    assert out.split() == ["d", "i"]
    assert err.split() == ["w", "e", "c"]


def test_configure_twice_one_handler(isolated_loggers):
    tl.configure_ray_loggers()
    tl.configure_ray_loggers()
    for name in ("ray", "ray.data"):
        handlers = [h for h in logging.getLogger(name).handlers if isinstance(h, tl.SplitStreamHandler)]
        assert len(handlers) == 1
        assert not any(type(h).__name__ == "PlainRayHandler" for h in logging.getLogger(name).handlers)


def test_user_ray_data_config_respected(isolated_loggers, monkeypatch, tmp_path):
    monkeypatch.setenv(tl.RAY_DATA_LOGGING_CONFIG_ENV, str(tmp_path / "custom.yaml"))
    before = list(logging.getLogger("ray.data").handlers)
    tl.configure_ray_loggers()
    assert logging.getLogger("ray.data").handlers == before


def test_opt_out(monkeypatch):
    assert tl.split_streams_enabled(True)
    assert not tl.split_streams_enabled(False)
    monkeypatch.setenv(tl.SPLIT_LOG_STREAMS_ENV, "0")
    assert not tl.split_streams_enabled(True)


def test_opt_out_leaves_ray_handlers(isolated_loggers):
    before = {n: list(logging.getLogger(n).handlers) for n in ("ray", "ray.data")}
    env = tl.setup_driver_logging({"env_vars": {"A": "1"}}, configure_logging=False)
    assert env == {"env_vars": {"A": "1"}}
    assert {n: list(logging.getLogger(n).handlers) for n in before} == before


def test_runtime_env_merge():
    env = tl.with_logging_runtime_env({"pip": ["x"], "env_vars": {"A": "1"}})
    assert env["pip"] == ["x"]
    assert env["env_vars"]["A"] == "1"
    assert env["worker_process_setup_hook"] == tl.WORKER_SETUP_HOOK
    assert tl.RAY_DATA_LOGGING_CONFIG_ENV in env["env_vars"]


def test_packaged_yaml_exists():
    assert tl.RAY_DATA_LOGGING_CONFIG_PATH.is_file()


@pytest.mark.integration
def test_actor_info_reaches_driver_stdout(capfd):
    import ray

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=1, include_dashboard=False, runtime_env=tl.with_logging_runtime_env({}))
    try:

        @ray.remote
        class A:
            def go(self):
                logging.getLogger("tide2.test_actor").info("ACTOR_INFO_MARKER")
                logging.getLogger("tide2.test_actor").warning("ACTOR_WARN_MARKER")

        a = A.remote()
        ray.get(a.go.remote())
        import time

        time.sleep(2)
    finally:
        ray.shutdown()
    out, err = capfd.readouterr()
    assert "ACTOR_INFO_MARKER" in out
    assert "ACTOR_INFO_MARKER" not in err
    assert "ACTOR_WARN_MARKER" in err
