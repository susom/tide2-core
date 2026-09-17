"""
Unit tests for tide2.runner.pipeline's process-lifecycle helpers.

Covers the worker-termination/signal-handling logic used by main()'s multi-worker
path to avoid orphaning GPU worker processes on Ctrl-C/SIGTERM (see DEV_TESTING.md's
"Stopping a running job" section for the operational context this fixes).
"""

from unittest.mock import MagicMock

import pytest

from tide2.runner.pipeline import _handle_sigterm
from tide2.runner.pipeline import _terminate_workers


class TestTerminateWorkers:
    """Test cases for _terminate_workers."""

    def test_terminates_and_does_not_kill_a_process_that_exits_promptly(self):
        """A worker that dies from terminate() before the join timeout should never be killed."""
        proc = MagicMock()
        proc.is_alive.side_effect = [True, False]  # alive before terminate(), dead after join()
        _terminate_workers([proc])
        proc.terminate.assert_called_once()
        proc.join.assert_called_once_with(timeout=10)
        proc.kill.assert_not_called()

    def test_escalates_to_kill_for_a_process_that_ignores_terminate(self):
        """A worker still alive after the join timeout should be force-killed and re-joined."""
        proc = MagicMock()
        proc.is_alive.side_effect = [True, True]  # alive before terminate(), still alive after join()
        _terminate_workers([proc])
        proc.terminate.assert_called_once()
        proc.join.assert_any_call(timeout=10)
        proc.kill.assert_called_once()
        assert proc.join.call_count == 2

    def test_skips_terminate_for_an_already_dead_process(self):
        """A worker that already exited on its own should not be terminated or killed."""
        proc = MagicMock()
        proc.is_alive.return_value = False
        _terminate_workers([proc])
        proc.terminate.assert_not_called()
        proc.join.assert_called_once_with(timeout=10)
        proc.kill.assert_not_called()

    def test_handles_a_mix_of_prompt_and_stuck_workers(self):
        """Each worker is evaluated independently, not short-circuited by another's state."""
        prompt = MagicMock()
        prompt.is_alive.side_effect = [True, False]
        stuck = MagicMock()
        stuck.is_alive.side_effect = [True, True]

        _terminate_workers([prompt, stuck])

        prompt.kill.assert_not_called()
        stuck.kill.assert_called_once()


class TestHandleSigterm:
    """Test cases for _handle_sigterm."""

    def test_raises_keyboard_interrupt(self):
        """SIGTERM must be converted to KeyboardInterrupt so it hits the same cleanup path as Ctrl-C."""
        with pytest.raises(KeyboardInterrupt):
            _handle_sigterm(15, None)
