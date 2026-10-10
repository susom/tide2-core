"""Tests for fault tolerance and hang detection configuration."""

import pytest

from tide2.runner.fault_tolerance import configure_data_context


def test_configure_data_context_hang_guard():
    """configure_data_context sets execution_no_progress_timeout_s on DataContext."""
    ctx = configure_data_context(no_progress_timeout_s=300.0)
    assert ctx.execution_no_progress_timeout_s == 300.0


def test_configure_data_context_disables_with_minus_one():
    """Passing -1 to no_progress_timeout_s disables the guard."""
    ctx = configure_data_context(no_progress_timeout_s=-1)
    assert ctx.execution_no_progress_timeout_s == -1


def test_configure_data_context_rejects_zero():
    """Passing 0 raises ValueError naming -1."""
    with pytest.raises(ValueError, match=r"no_progress_timeout_s=0 is invalid\. Pass -1"):
        configure_data_context(no_progress_timeout_s=0)


def test_configure_data_context_aborts_on_errored_blocks_by_default():
    """Ray drops errored blocks silently, so the default tolerates none."""
    assert configure_data_context().max_errored_blocks == 0
    assert configure_data_context(max_errored_blocks=5).max_errored_blocks == 5
    configure_data_context()
