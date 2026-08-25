"""
Ray Actors for transformer NER inference.

This module provides:
- TransformerInferenceActor: GPU actor for raw token inference (no BIO aggregation)
- BIOAggregationActor: CPU actor for BIO token aggregation and serialization

The two actors form a streaming pipeline where GPU inference and CPU
post-processing run concurrently via Ray Data's streaming executor:

    ReadParquet → FlatMap(chunk) → MapBatches(GPU inference) → MapBatches(BIO aggregation) → Write

Thread/Process Safety:
    Each Actor maintains its own state. The GPU actor loads a transformer model
    on a dedicated GPU. The CPU actor is stateless.

Usage with Ray Data:
    ds.map_batches(
        TransformerInferenceActor,
        batch_size=512,
        num_gpus=1,
        compute=ray.data.ActorPoolStrategy(size=num_gpus),
        fn_constructor_kwargs={"model_name": "StanfordAIMI/stanford-deidentifier-v2"},
    )
    ds.map_batches(
        BIOAggregationActor,
        batch_size=512,
        compute=ray.data.ActorPoolStrategy(size=num_agg_actors),
    )
"""

import json
import logging
import os
from pathlib import Path
from typing import Any
from typing import NamedTuple

import numpy as np
import torch

from tide2.transformers import TransformerCore
from tide2.utils.text_processing import aggregate_bio_tokens

logger = logging.getLogger(__name__)

# Nominal starting batch size — exists only so the actor runs out of the box. A
# real run supplies ``--gpu-batch-size`` sized to the load; if a batch still OOMs,
# the forward halves and retries (safety net, not a tuner). See
# :meth:`TransformerInferenceActor._forward_windows`.
_DEFAULT_GPU_BATCH_SIZE = 64

# Default token overlap between adjacent windows when a single chunk exceeds the
# model's per-window token budget. Matches the upstream char chunker's
# ``CHUNK_OVERLAP_SIZE`` (40 tokens) so an entity straddling a window boundary is
# still seen whole in at least one window; the existing downstream dedup
# (BIO raw-token tuples + reassembly IoU) collapses the overlap-region duplicates.
_DEFAULT_WINDOW_OVERLAP = 40


class _Window(NamedTuple):
    """One token-space window carved from a single input chunk.

    Windowing (not truncation) is how the actor covers a chunk whose tokenized
    length exceeds the model's per-window budget. Each window carries the source
    chunk's index (``owner``) plus the content token ids and their chunk-relative
    character offsets for this slice — all sourced from a single tokenization, so
    offsets never need re-basing when window predictions are merged back.
    """

    owner: int
    content_ids: list[int]
    offsets: list[tuple[int, int]]
    text: str
    token_start: int  # index of this window's first content token within the chunk
    token_end: int  # exclusive index of its last content token within the chunk


def _numpy_default(obj: Any) -> Any:
    """json.dumps default handler for numpy scalar types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class TransformerInferenceActor:
    """
    Ray Actor for GPU-based transformer inference (raw tokens only).

    This actor is designed to be used with Ray Data's map_batches() with
    ActorPoolStrategy. Each actor loads a transformer model on a GPU and
    performs token classification inference on text chunks.

    The actor returns raw BIO tokens (not aggregated) so that the CPU-heavy
    BIO aggregation can run in a separate BIOAggregationActor, allowing GPU
    inference and CPU post-processing to overlap via Ray Data streaming.

    The actor handles:
    - Model loading with GPU placement (via TransformerCore)
    - Token-accurate windowing of over-budget chunks (never truncation)
    - Batch inference with automatic OOM recovery
    - JSON serialization of raw token predictions

    Inference flow (single tokenization, window-don't-truncate, OOM-safe retry):

    1. Tokenize every input chunk **once** (ragged, no truncation, char offsets).
    2. Any chunk whose token length exceeds the per-window budget
       (``model_max_length`` minus the tokenizer's special tokens) is sliced into
       overlapping ≤-budget token windows — the earlier truncating path silently
       dropped the tail of dense chunks, so PHI there was never redacted.
    3. Windows are length-sorted (within this one ``__call__``) and forwarded at a
       fixed batch size, halving on OOM. Each forward releases every CUDA tensor on
       any exit; on CUDA OOM the forward halves the batch and retries over the
       **same** tokenized windows (never re-tokenizing).
    4. Window predictions are merged back per input chunk and emitted as the same
       ``predictions_raw_json`` contract as before; overlap-region duplicates are
       handled by the existing downstream dedup (BIO raw-token tuples; reassembly
       IoU).
    """

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        bucket_name: str | None = None,
        project_id: str | None = None,
        gpu_batch_size: int | None = None,
        allow_huggingface_download: bool = True,
        window_overlap: int = _DEFAULT_WINDOW_OVERLAP,
    ) -> None:
        """
        Initialize the actor with a transformer model on GPU.

        Args:
            model_name: Name of the model configuration to load.
            model_path: Optional explicit model path (overrides GCS resolution).
            bucket_name: Optional GCS bucket name for model loading.
            project_id: Optional GCP project ID for model loading.
            gpu_batch_size: Number of token windows per GPU forward, independent
                of the Ray Data batch size. Size it for the load; if a batch still
                OOMs, the forward halves and retries. If None, defaults to
                ``_DEFAULT_GPU_BATCH_SIZE`` (a nominal value that only needs to run
                out of the box, not to be optimal).
            allow_huggingface_download: If True, fall back to HuggingFace Hub
                when local cache and GCS both miss.
            window_overlap: Token overlap between adjacent windows when a chunk
                exceeds the per-window budget (default 40, matching the upstream
                char chunker's overlap). Clamped to ``[0, budget - 1]``. Overlap
                duplicates are removed downstream (BIO dedup + reassembly IoU).
        """
        self.model_name = model_name
        self._window_overlap = max(0, window_overlap)

        # Count of CUDA OOMs this actor caught and recovered from (by shrinking
        # the forward batch). Read by the GPU OOM-recovery test to prove the
        # handled-OOM path was actually exercised, not skipped.
        self._handled_oom_count = 0

        # Determine device for explicit GPU placement
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            device = f"cuda:{device_idx}"
        else:
            device = "cpu"
            logger.warning("No GPU detected, using CPU (performance will be degraded)")

        # CPU branch only: cap torch threads to the CPUs Ray reserved for this
        # actor so that actors x threads <= total CPUs (no oversubscription).
        # GPU branch keeps torch's default thread behavior untouched.
        if device == "cpu":
            import ray

            assigned = ray.get_runtime_context().get_assigned_resources()
            n = int(assigned.get("CPU", 1)) or 1
            torch.set_num_threads(n)
            logger.info("CPU inference: capping torch to %d thread(s) per Ray allocation", n)

        # Create core inference engine with explicit device and immediate loading
        self._core = TransformerCore(
            model_name=model_name,
            model_path=model_path,
            bucket_name=bucket_name,
            project_id=project_id,
            device=device,
            load_immediately=True,  # Load model immediately on actor init
            local_files_only=True,  # Use cached models only
            allow_huggingface_download=allow_huggingface_download,
        )

        # Store model path and seq_len for backwards compatibility and batch sizing
        self.model_path = self._core.model_path
        self._seq_len = self._core.model_max_length

        # Per-window content-token budget: the model's max length minus the
        # special tokens the tokenizer adds (e.g. [CLS]/[SEP]). Chunks longer than
        # this are windowed, not truncated. Clamp the window overlap below the
        # budget so the window step stays >= 1 (guaranteed forward progress).
        self._num_special_tokens = self._core.num_special_tokens
        self._token_budget = max(1, self._seq_len - self._num_special_tokens)
        self._window_overlap = min(self._window_overlap, self._token_budget - 1)

        # Windows per GPU forward. Operator-supplied for real runs; the nominal
        # default only needs to run out of the box (a sized run never OOMs; if one
        # slips through, _forward_windows halves and retries).
        self._gpu_batch_size = gpu_batch_size or _DEFAULT_GPU_BATCH_SIZE

        logger.info(
            f"TransformerInferenceActor initialized: model={model_name}, "
            f"device={self._core.get_device_info()}, gpu_batch_size={self._gpu_batch_size}, "
            f"token_budget={self._token_budget}, window_overlap={self._window_overlap}"
        )

    @property
    def model_pipeline(self) -> Any:
        """Get the model pipeline (for backwards compatibility)."""
        return self._core.pipeline

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """
        Process a batch of text chunks through transformer inference (raw tokens).

        This method is called by Ray Data's map_batches() with batches in
        columnar format (dict of column name -> list of values).

        Returns raw BIO tokens (not aggregated). BIO aggregation is handled
        by the downstream BIOAggregationActor for GPU/CPU overlap.

        Args:
            batch: Dictionary with columnar data:
                - chunk_text: List of chunk text strings
                - text_hash: List of document text hashes
                - chunk_id: List of chunk identifiers within each document
                - char_offset_start: List of character offsets in original document
                - patient_id: List of patient identifiers (passed through)

        Returns:
            Dictionary with inference results in columnar format:
                - text_hash: Document text hashes (passed through)
                - chunk_id: Chunk identifiers (passed through)
                - char_offset_start: Character offsets (passed through)
                - patient_id: Patient identifiers (passed through)
                - chunk_text: Chunk text (passed through for BIO aggregation)
                - predictions_raw_json: JSON-serialized list of raw BIO token dicts
        """
        chunk_texts = batch["chunk_text"]
        text_hashes = batch["text_hash"]
        chunk_ids = batch["chunk_id"]
        char_offsets = batch["char_offset_start"]
        patient_ids = batch.get("patient_id", [""] * len(chunk_texts))
        chunk_uids = batch.get("chunk_uid", [""] * len(chunk_texts))

        batch_size = len(chunk_texts)

        # Handle empty batches
        if batch_size == 0:
            return {
                "text_hash": [],
                "chunk_id": [],
                "chunk_uid": [],
                "char_offset_start": [],
                "patient_id": [],
                "chunk_text": [],
                "predictions_raw_json": [],
            }

        # Filter out None/empty texts
        chunk_texts = list(chunk_texts)
        valid_indices = [i for i, t in enumerate(chunk_texts) if t]
        if not valid_indices:
            return {
                "text_hash": list(text_hashes),
                "chunk_id": list(chunk_ids),
                "chunk_uid": list(chunk_uids),
                "char_offset_start": list(char_offsets),
                "patient_id": list(patient_ids),
                "chunk_text": chunk_texts,
                "predictions_raw_json": ["[]"] * batch_size,
            }

        valid_texts = [chunk_texts[i] for i in valid_indices]

        # Run raw inference with OOM recovery (no BIO aggregation)
        self._log_gpu_mem(f"before __call__ (n={len(valid_texts)})")
        raw_results = self._run_inference_raw_with_oom_recovery(valid_texts)
        self._log_gpu_mem(f"after __call__ (n={len(valid_texts)})")

        # Map predictions back to original indices and serialize to JSON
        predictions_raw_json_list = ["[]"] * batch_size
        for idx, preds in zip(valid_indices, raw_results, strict=False):
            try:
                predictions_raw_json_list[idx] = json.dumps(preds, ensure_ascii=False, default=_numpy_default)
            except Exception:
                logger.exception(f"Error serializing raw predictions for chunk {chunk_ids[idx]}")

        return {
            "text_hash": list(text_hashes),
            "chunk_id": list(chunk_ids),
            "chunk_uid": list(chunk_uids),
            "char_offset_start": list(char_offsets),
            "patient_id": list(patient_ids),
            "chunk_text": chunk_texts,
            "predictions_raw_json": predictions_raw_json_list,
        }

    def _log_gpu_mem(self, stage: str) -> None:
        """Log per-``__call__`` GPU memory when ``TIDE2_LOG_GPU_MEM`` is set.

        Diagnostics-only hook for the GPU OOM verification harness
        (``tests/oom_verification.py``): the driver cannot see an actor's VRAM, so
        the actor logs its own ``memory_allocated``/``memory_reserved``. A no-op
        unless the env var is truthy, so it adds no overhead in production.
        """
        if not os.environ.get("TIDE2_LOG_GPU_MEM"):
            return
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        logger.info(
            "GPU mem [%s]: allocated=%.1fMB reserved=%.1fMB peak=%.1fMB",
            stage,
            allocated,
            reserved,
            peak,
        )

    def _record_handled_oom(self) -> None:
        """Record a handled CUDA OOM to a file when ``TIDE2_OOM_COUNT_FILE`` is set.

        Diagnostics-only hook for the Ray-driven GPU OOM-recovery test: the
        driver cannot read a Ray actor's in-process ``_handled_oom_count`` (the
        actor runs in a separate worker process), so the actor appends a marker
        line to a shared file. With a single GPU actor there is exactly one
        writer, so no locking is needed. A no-op unless the env var is set, so it
        adds no overhead in production.
        """
        path = os.environ.get("TIDE2_OOM_COUNT_FILE")
        if not path:
            return
        try:
            with Path(path).open("a") as f:
                f.write("1\n")
        except OSError:
            logger.warning("could not record handled OOM to %s", path, exc_info=True)

    def _run_inference_raw_with_oom_recovery(self, texts: list[str]) -> list[list[dict]]:
        """Tokenize once, window long chunks, length-sort, forward, merge results.

        This is the actor's whole inference path. It tokenizes every chunk exactly
        once (no truncation), windows any chunk over the per-window token budget
        instead of dropping its tail, sorts the windows by token length within this
        one ``__call__`` (so multi-slice batches don't pad short windows to the
        batch max), forwards them at a fixed batch size that halves on OOM (never
        re-tokenizing), and merges each chunk's window predictions back into one
        list. The emitted per-chunk contract is unchanged; overlap-region
        duplicates are removed downstream (BIO dedup + reassembly IoU).

        Args:
            texts: List of chunk text strings to process (assumed non-empty; the
                caller filters empties).

        Returns:
            List of raw token lists (one per input chunk), aligned to ``texts``.
            A chunk that tokenizes to zero tokens yields an empty list.

        Raises:
            RuntimeError: If OOM persists at a single window, or for any non-OOM
                error (re-raised unchanged).
        """
        n = len(texts)
        results: list[list[dict]] = [[] for _ in range(n)]
        if n == 0:
            return results

        # 1. Tokenize once (ragged, no truncation, char offsets).
        encoded = self._core.tokenize_ragged(texts)
        input_ids = encoded["input_ids"]
        offset_mapping = encoded["offset_mapping"]

        # 2. Window (don't truncate) any chunk over budget.
        windows = self._plan_windows(texts, input_ids, offset_mapping)
        if not windows:
            return results

        # 3. Sort by token length within THIS __call__ so multi-slice batches
        #    (batch_size > gpu_batch_size) don't pad short windows to the batch max.
        #    Merge keys on window.owner, so no un-sort is needed.
        windows.sort(key=lambda w: len(w.content_ids))

        # 4. Forward (halve-and-retry on OOM), then merge each window onto its
        #    owner. Offsets are chunk-relative, so this is a plain concatenation;
        #    order across windows does not matter (downstream aggregation sorts by
        #    start position).
        for window, preds in zip(windows, self._forward_windows(windows), strict=True):
            results[window.owner].extend(preds)

        return results

    def _forward_windows(self, windows: list[_Window]) -> list[list[dict]]:
        """Forward all windows in fixed-size slices; halve the batch on CUDA OOM.

        On CUDA OOM the batch is halved and the whole set re-forwarded over the
        **same** pre-tokenized windows (never re-tokenizing). Halve-and-redo is the
        simplest correct recovery; a sized production run never OOMs, so the wasted
        recompute on the rare recovery is fine. The discarded partial ``out`` holds
        only detached CPU arrays.

        Args:
            windows: The windows to forward, aligned to the returned predictions.

        Returns:
            Raw token lists aligned to ``windows``.

        Raises:
            RuntimeError: If OOM persists at a single window, or for any non-OOM
                error (re-raised unchanged).
        """
        batch_size = max(1, min(self._gpu_batch_size, len(windows)))
        while True:
            try:
                out: list[list[dict]] = []
                for start in range(0, len(windows), batch_size):
                    chunk = windows[start : start + batch_size]
                    out.extend(self._core.forward_windows([(w.content_ids, w.offsets, w.text) for w in chunk]))
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise  # non-OOM: propagate unchanged
                if batch_size <= 1:
                    raise RuntimeError("CUDA OOM on a single token window") from e
            else:
                return out
                # Count the handled OOM so tests can prove the recovery path ran.
                # Defensive getattr: actors built via __new__ in unit tests skip
                # __init__ and so never set _handled_oom_count.
                self._handled_oom_count = getattr(self, "_handled_oom_count", 0) + 1
                self._record_handled_oom()
                batch_size = max(1, batch_size // 2)  # power-of-two reduction; terminates at 1
                logger.warning("CUDA OOM; halving GPU batch to %d and re-forwarding", batch_size)
                # forward_windows (Fix A1) already freed the failed forward's
                # tensors, so this reclaims real VRAM before the smaller retry.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _plan_windows(
        self,
        texts: list[str],
        input_ids: list[list[int]],
        offset_mapping: list[list[Any]],
    ) -> list[_Window]:
        """Slice each chunk's tokens into ≤-budget windows (never truncating).

        A chunk within budget becomes a single window; an over-budget chunk is
        sliced into windows of ``_token_budget`` content tokens that step forward
        by ``budget - overlap`` (>= 1), so consecutive windows overlap by
        ``_window_overlap`` and together cover every token with no gaps. Char
        offsets are carried straight from the single tokenization.
        """
        budget = self._token_budget
        step = max(1, budget - self._window_overlap)

        windows: list[_Window] = []
        for owner, text in enumerate(texts):
            ids = input_ids[owner]
            offs = offset_mapping[owner]
            n = len(ids)
            if n == 0:
                continue
            if n <= budget:
                windows.append(_Window(owner, list(ids), [tuple(o) for o in offs], text, 0, n))
                continue
            start = 0
            while start < n:
                end = min(start + budget, n)
                windows.append(
                    _Window(owner, list(ids[start:end]), [tuple(o) for o in offs[start:end]], text, start, end)
                )
                if end == n:
                    break
                start += step
        return windows


class BIOAggregationActor:
    """
    Stateless CPU actor for BIO token aggregation and serialization.

    Takes raw BIO tokens from TransformerInferenceActor and aggregates them
    into entity spans using aggregate_bio_tokens(). This separates CPU-heavy
    post-processing from GPU inference so they can run concurrently via
    Ray Data streaming.

    Input columns:
        - text_hash, chunk_id, char_offset_start, patient_id: passed through
        - chunk_text: original text (needed for BIO aggregation)
        - predictions_raw_json: JSON-serialized raw BIO token dicts

    Output columns:
        - text_hash, chunk_id, char_offset_start, patient_id: passed through
        - predictions_json: JSON-serialized aggregated entity spans
    """

    def __call__(self, batch: dict[str, Any]) -> dict[str, list[Any]]:
        """Aggregate raw BIO tokens into entity spans."""
        chunk_texts = batch["chunk_text"]
        raw_json_list = batch["predictions_raw_json"]
        text_hashes = batch["text_hash"]
        chunk_ids = batch["chunk_id"]
        char_offsets = batch["char_offset_start"]
        patient_ids = batch.get("patient_id", [""] * len(chunk_texts))
        chunk_uids = batch.get("chunk_uid", [""] * len(chunk_texts))

        batch_size = len(chunk_texts)

        if batch_size == 0:
            return {
                "text_hash": [],
                "chunk_id": [],
                "chunk_uid": [],
                "char_offset_start": [],
                "patient_id": [],
                "predictions_json": [],
                "chunk_status": [],
            }

        predictions_json_list = []
        chunk_statuses = []
        for i in range(batch_size):
            try:
                raw_tokens = json.loads(raw_json_list[i])
                text = chunk_texts[i] if chunk_texts[i] else ""

                if not raw_tokens or not text:
                    predictions_json_list.append("[]")
                    chunk_statuses.append("success")
                    continue

                # Remove duplicates (can occur from chunking)
                raw_tokens = [dict(t) for t in {tuple(d.items()) for d in raw_tokens}]

                aggregated = aggregate_bio_tokens(raw_tokens, text)
                predictions_json_list.append(json.dumps(aggregated, ensure_ascii=False))
                chunk_statuses.append("success")
            except Exception:
                logger.exception(f"Error aggregating predictions for chunk {chunk_ids[i]}")
                predictions_json_list.append("[]")
                chunk_statuses.append("failed")

        return {
            "text_hash": list(text_hashes),
            "chunk_id": list(chunk_ids),
            "chunk_uid": list(chunk_uids),
            "char_offset_start": list(char_offsets),
            "patient_id": list(patient_ids),
            "predictions_json": predictions_json_list,
            "chunk_status": chunk_statuses,
        }


def create_transformer_actor(
    model_name: str,
    model_path: str | None = None,
    bucket_name: str | None = None,
    project_id: str | None = None,
    gpu_batch_size: int | None = None,
    allow_huggingface_download: bool = True,
    window_overlap: int = _DEFAULT_WINDOW_OVERLAP,
) -> type[TransformerInferenceActor]:
    """
    Factory function to create a TransformerInferenceActor class with specific config.

    This unified factory works for both local/batch processing and cluster modes.
    Ray Data's map_batches with ActorPoolStrategy requires a class that can be
    instantiated without arguments, which this factory provides.

    Args:
        model_name: Name of the model configuration to load.
        model_path: Optional explicit model path (overrides GCS resolution).
        bucket_name: Optional GCS bucket name for model loading.
        project_id: Optional GCP project ID for model loading.
        gpu_batch_size: Windows per GPU forward (None = nominal default).
        allow_huggingface_download: If True, fall back to HuggingFace Hub
            when local cache and GCS both miss.
        window_overlap: Token overlap between adjacent windows for over-budget
            chunks (default 40).

    Returns:
        A class that can be used with Ray Data's map_batches().

    Examples:
        # Basic usage
        Actor = create_transformer_actor("StanfordAIMI/stanford-deidentifier-v2")

        # With explicit model path
        Actor = create_transformer_actor("my_model", model_path="/models/ner")
    """

    class ConfiguredTransformerActor(TransformerInferenceActor):
        """Pre-configured TransformerInferenceActor with captured model settings."""

        def __init__(self):
            super().__init__(
                model_name=model_name,
                model_path=model_path,
                bucket_name=bucket_name,
                project_id=project_id,
                gpu_batch_size=gpu_batch_size,
                allow_huggingface_download=allow_huggingface_download,
                window_overlap=window_overlap,
            )

    return ConfiguredTransformerActor


# Backwards compatibility alias
create_transformer_actor_class = create_transformer_actor
