"""
Fault tolerance configuration for Ray batch processing.

This module provides configuration for Ray's native fault tolerance features
and helper utilities for graceful shutdown and DataContext configuration.

Works identically across all execution modes (local, VM, cluster).

Native Ray Features Used:
- max_restarts: Actor restarts on crash (node failure, OOM, etc.)
- max_task_retries: Retry tasks on actor failure
- retry_exceptions: Specify which exceptions trigger retry

Usage:
    # In map_batches call (unpack args directly for actors)
    ds.map_batches(
        MyActor,
        compute=ray.data.ActorPoolStrategy(size=num_actors),
        **get_ray_remote_args_cpu(),
    )

    # Configure DataContext before processing
    configure_data_context()
"""

import logging
import signal
import threading
from typing import Any

import ray.data

logger = logging.getLogger(__name__)


# =============================================================================
# Ray Remote Args Configuration
# =============================================================================

# Default fault tolerance settings for CPU actors
# Note: Only includes options valid for Ray Data actor-based map_batches
# Valid options: max_restarts, max_task_retries, max_pending_calls, max_concurrency
CPU_FAULT_TOLERANCE_CONFIG = {
    # Actor restarts on crash (node failure, segfault, etc.)
    # 5 restarts is generous but bounded
    "max_restarts": 5,
    # Retry tasks on failure (bounded to prevent infinite loops)
    # Combined with max_restarts=5, this gives up to 15 total attempts
    "max_task_retries": 3,
}

# Fault tolerance settings for GPU actors
# More conservative due to GPU memory management complexity
GPU_FAULT_TOLERANCE_CONFIG = {
    "max_restarts": 3,  # GPU actors more expensive to restart
    "max_task_retries": 2,  # Bounded retries to prevent infinite loops
}


def get_ray_remote_args_cpu(**overrides) -> dict[str, Any]:
    """
    Get ray_remote_args for CPU-based actors with fault tolerance.

    Args:
        **overrides: Override any default settings.

    Returns:
        Dictionary suitable for ray_remote_args parameter in map_batches.

    Example:
        ds.map_batches(
            RecognizerActor,
            compute=ray.data.ActorPoolStrategy(size=num_actors),
            **get_ray_remote_args_cpu(),
        )
    """
    config = CPU_FAULT_TOLERANCE_CONFIG.copy()
    config.update(overrides)
    return config


def get_ray_remote_args_gpu(num_gpus: int = 1, **overrides) -> dict[str, Any]:
    """
    Get ray_remote_args for GPU-based actors with fault tolerance.

    Includes num_gpus resource requirement by default. When num_gpus=0,
    no GPU resource is requested, allowing CPU-only execution.

    Args:
        num_gpus: Number of GPUs per actor (default: 1). Set to 0 for CPU-only mode.
        **overrides: Override any default settings.

    Returns:
        Dictionary suitable for ray_remote_args parameter in map_batches.

    Example:
        # GPU mode (default)
        ds.map_batches(
            TransformerActor,
            compute=ray.data.ActorPoolStrategy(size=num_gpus),
            **get_ray_remote_args_gpu(),
        )

        # CPU-only mode (no GPU resources requested)
        ds.map_batches(
            TransformerActor,
            compute=ray.data.ActorPoolStrategy(size=1),
            **get_ray_remote_args_gpu(num_gpus=0),
        )
    """
    config = GPU_FAULT_TOLERANCE_CONFIG.copy()
    if num_gpus > 0:
        config["num_gpus"] = num_gpus  # Each GPU actor gets specified GPUs
    # When num_gpus=0, don't add GPU resource requirement (CPU-only mode)
    config.update(overrides)
    return config


# =============================================================================
# DataContext Configuration
# =============================================================================


def configure_data_context(
    verbose_progress: bool = True,
    preserve_order: bool = False,
    target_max_block_size_mb: int = 128,
    target_min_block_size_mb: int = 1,
    read_op_min_num_blocks: int = 2000,
    max_errored_blocks: int = 100,
) -> ray.data.DataContext:
    """
    Configure Ray Data context for optimal batch processing.

    This configures native Ray Data features for progress tracking,
    memory management, and execution behavior.

    Args:
        verbose_progress: Show detailed progress bars per operator.
        preserve_order: Maintain block order (slower but deterministic).
        target_max_block_size_mb: Maximum block size in MB (default 128).
        target_min_block_size_mb: Minimum block size in MB (default 1).
        read_op_min_num_blocks: Minimum number of read output blocks (default 2000).
            Higher values increase read parallelism for large datasets.
        max_errored_blocks: Maximum number of blocks that can error before
            aborting the dataset execution (default 100). Prevents node-level
            OOM kills from aborting the entire job.

    Returns:
        Configured DataContext instance.
    """
    ctx = ray.data.DataContext.get_current()

    # Progress tracking
    ctx.execution_options.verbose_progress = verbose_progress

    # Order preservation (False = better performance)
    ctx.execution_options.preserve_order = preserve_order

    # Block size tuning for large datasets
    ctx.target_max_block_size = target_max_block_size_mb * 1024 * 1024
    ctx.target_min_block_size = target_min_block_size_mb * 1024 * 1024

    # Read parallelism: default is 200, far too low for large clusters.
    # With 1900 actors and hundreds of input files, we need many more
    # read output blocks to keep actors fed.
    ctx.read_op_min_num_blocks = read_op_min_num_blocks

    # Tolerate some errored blocks from node-level OOM kills instead of
    # aborting the entire job. Failed blocks are skipped in the output.
    ctx.max_errored_blocks = max_errored_blocks

    logger.info(
        f"DataContext configured: verbose_progress={verbose_progress}, "
        f"block_size={target_min_block_size_mb}-{target_max_block_size_mb}MB, "
        f"read_op_min_num_blocks={read_op_min_num_blocks}, "
        f"max_errored_blocks={max_errored_blocks}"
    )

    return ctx


# =============================================================================
# Graceful Shutdown Handler
# =============================================================================


class GracefulShutdown:
    """
    Handle SIGTERM/SIGINT for clean shutdown of batch processing.

    This allows the processing loop to complete the current segment
    and checkpoint before exiting, preventing data loss.

    Usage:
        shutdown = GracefulShutdown()

        for segment in segments:
            if shutdown.requested:
                logger.info("Shutdown requested, saving checkpoint")
                break
            process_segment(segment)

    Attributes:
        requested: True if shutdown has been requested.
    """

    def __init__(self):
        """Initialize and register signal handlers."""
        self._shutdown_event = threading.Event()
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)

        # Register handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.debug("Graceful shutdown handler registered")

    def _handle_signal(self, signum: int, frame):
        """Handle shutdown signal."""
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name}, initiating graceful shutdown...")
        self._shutdown_event.set()

        # If we get a second signal, use original handler (force quit)
        signal.signal(signal.SIGTERM, self._original_sigterm)
        signal.signal(signal.SIGINT, self._original_sigint)

    @property
    def requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """
        Wait for shutdown signal.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if shutdown was requested, False if timeout.
        """
        return self._shutdown_event.wait(timeout=timeout)

    def reset(self):
        """Reset shutdown state (use with caution)."""
        self._shutdown_event.clear()

    def restore_handlers(self):
        """Restore original signal handlers."""
        signal.signal(signal.SIGTERM, self._original_sigterm)
        signal.signal(signal.SIGINT, self._original_sigint)


# =============================================================================
# Utility Functions
# =============================================================================


def chunked(iterable, size: int):
    """
    Split an iterable into chunks of specified size.

    Args:
        iterable: Any iterable to chunk.
        size: Maximum size of each chunk.

    Yields:
        Lists of items, each with at most `size` items.
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
