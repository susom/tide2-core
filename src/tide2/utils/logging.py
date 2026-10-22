"""Route INFO-level tide2/Ray logs to stdout and WARNING+ to stderr.

GCP Batch (and similar platforms) derive log severity from the output stream
(stdout -> INFO, stderr -> ERROR). Ray's default handlers write everything to
stderr, so routine INFO lines show up as errors. :func:`configure_logging`
replaces those console handlers with :class:`SplitStreamHandler`.

Opt out with ``configure_logging=False`` on ``LocalJobRunner`` / ``init_ray_local``
or by setting ``TIDE2_SPLIT_LOG_STREAMS=0``.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

SPLIT_LOG_STREAMS_ENV = "TIDE2_SPLIT_LOG_STREAMS"
RAY_DATA_LOGGING_CONFIG_ENV = "RAY_DATA_LOGGING_CONFIG"
RAY_DATA_LOGGING_CONFIG_PATH = Path(__file__).resolve().parent.parent / "resources" / "ray_data_logging.yaml"
RAY_LOG_FORMAT = "%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s"
TIDE2_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
WORKER_SETUP_HOOK = "tide2.utils.logging.worker_process_setup_hook"

_RAY_LOGGERS = ("ray", "ray.data")


class SplitStreamHandler(logging.Handler):
    """Handler sending records below WARNING to stdout and WARNING+ to stderr.

    The streams are resolved when each record is emitted (like
    ``logging._StderrHandler``), so captured/swapped ``sys.stdout``/``sys.stderr``
    are honoured. In Ray worker processes only the message is written, matching Ray's
    own worker-side output (the driver adds the process prefix).
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Write the formatted record to stdout or stderr depending on its level."""
        try:
            stream = sys.stdout if record.levelno < logging.WARNING else sys.stderr
            text = logging.Formatter("%(message)s").format(record) if _is_worker() else self.format(record)
            stream.write(text + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)


def split_streams_enabled(configure_logging: bool = True) -> bool:
    """Return whether split-stream logging is enabled.

    Args:
        configure_logging: Programmatic switch; False disables it.

    Returns:
        False if disabled by argument or by ``TIDE2_SPLIT_LOG_STREAMS`` set to
        ``0``/``false``/``no``/``off``.
    """
    if not configure_logging:
        return False
    return os.environ.get(SPLIT_LOG_STREAMS_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _is_worker() -> bool:
    try:
        import ray._private.worker as ray_worker
    except Exception:
        return False
    else:
        return ray_worker.global_worker.mode == ray_worker.WORKER_MODE


def _replace_handler(logger: logging.Logger) -> None:
    """Swap the logger's console handler for a single ``SplitStreamHandler``.

    File handlers are left untouched. Level, filters and formatter of the replaced
    console handler are carried over. Idempotent.
    """
    existing = [h for h in logger.handlers if isinstance(h, SplitStreamHandler)]
    if existing:
        for extra in existing[1:]:
            logger.removeHandler(extra)
        return

    replaced = [h for h in logger.handlers if type(h).__name__ == "PlainRayHandler"]
    new = SplitStreamHandler()
    if replaced:
        new.setLevel(replaced[0].level)
        for flt in replaced[0].filters:
            new.addFilter(flt)
        new.setFormatter(replaced[0].formatter)
    else:
        new.setFormatter(logging.Formatter(RAY_LOG_FORMAT))
    for handler in replaced:
        logger.removeHandler(handler)
    logger.addHandler(new)


def _ray_data_config_is_ours() -> bool:
    configured = os.environ.get(RAY_DATA_LOGGING_CONFIG_ENV)
    return configured is None or Path(configured) == RAY_DATA_LOGGING_CONFIG_PATH


def configure_ray_loggers() -> None:
    """Install the split handler on the ``ray`` and ``ray.data`` loggers of this process.

    Ray Data reapplies its config lazily (and clears handlers), so the packaged
    config is also registered via ``RAY_DATA_LOGGING_CONFIG`` unless the user set one.
    """
    if RAY_DATA_LOGGING_CONFIG_ENV not in os.environ:
        os.environ[RAY_DATA_LOGGING_CONFIG_ENV] = str(RAY_DATA_LOGGING_CONFIG_PATH)

    import ray  # noqa: F401  # ensure Ray's default logging config has been generated

    _replace_handler(logging.getLogger("ray"))
    if _ray_data_config_is_ours():
        _replace_handler(logging.getLogger("ray.data"))


def worker_process_setup_hook() -> None:
    """Ray ``worker_process_setup_hook``: configure logging inside worker processes."""
    configure_ray_loggers()
    root = logging.getLogger()
    if not root.handlers:
        handler = SplitStreamHandler()
        handler.setFormatter(logging.Formatter(TIDE2_LOG_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)


def configure_cli_logging(level: int | str = logging.INFO) -> None:
    """Replacement for ``logging.basicConfig`` in the CLI using the split handler.

    Honours the opt-out env var by falling back to ``logging.basicConfig``.
    """
    if not split_streams_enabled():
        logging.basicConfig(level=level, format=TIDE2_LOG_FORMAT)
        return
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    handler = SplitStreamHandler()
    handler.setFormatter(logging.Formatter(TIDE2_LOG_FORMAT))
    root.addHandler(handler)


def with_logging_runtime_env(runtime_env: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a copy of ``runtime_env`` with the logging worker setup hook added.

    Existing keys (including ``env_vars``) are preserved. A caller-provided
    ``worker_process_setup_hook`` is not overwritten, and the Ray Data logging
    config env var is forwarded to workers (the user's value wins over ours).
    """
    merged: dict[str, Any] = dict(runtime_env or {})
    merged.setdefault("worker_process_setup_hook", WORKER_SETUP_HOOK)
    env_vars = dict(merged.get("env_vars") or {})
    env_vars.setdefault(
        RAY_DATA_LOGGING_CONFIG_ENV,
        os.environ.get(RAY_DATA_LOGGING_CONFIG_ENV, str(RAY_DATA_LOGGING_CONFIG_PATH)),
    )
    merged["env_vars"] = env_vars
    return merged


def setup_driver_logging(runtime_env: dict[str, Any], configure_logging: bool = True) -> dict[str, Any]:
    """Configure driver loggers (if enabled) and return the Ray ``runtime_env`` to use."""
    if not split_streams_enabled(configure_logging):
        return runtime_env
    configure_ray_loggers()
    return with_logging_runtime_env(runtime_env)
