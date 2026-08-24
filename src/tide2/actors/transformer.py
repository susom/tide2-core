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
from typing import cast

import numpy as np
import torch

from tide2.transformers import TransformerCore
from tide2.utils.text_processing import aggregate_bio_tokens

logger = logging.getLogger(__name__)

# VRAM thresholds (GB) for short-sequence budget tiers.
# Maps to NVIDIA product lines: L4/RTX (<=24), A6000/A100-40 (24-80), H100/A100-80 (>=80).
_VRAM_TIER_HIGH_GB = 80
_VRAM_TIER_MID_GB = 24

# Ceiling for the *worst-case* (max-seq-length) auto-computed GPU batch size. On
# very large GPUs the free-memory estimate can suggest thousands of texts per
# forward; capping the worst-case base keeps the initial forward modest so OOM
# recovery grows into headroom by splitting down, rather than starting oversized
# and cascading. Recovery still splits below this when needed.
#
# Note this bounds the max-seq base only, not the final per-forward window count:
# ``_batch_cap_for_tokens`` scales this base *up* for short windows, so short-window
# batches can exceed 512 (see that method for why that stays memory-safe).
_MAX_INITIAL_GPU_BATCH = 512

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
    3. Windows are bucketed into length-homogeneous forward batches (from the
       per-sample memory model, fed exact token counts) to bound padding waste and
       make per-forward memory predictable.
    4. Each bucket runs one forward, releasing every CUDA tensor on any exit.
    5. On CUDA OOM the forward retries at a smaller window-batch over the **same**
       tokenized windows (never re-tokenizing).
    6. Window predictions are merged back per input chunk and emitted as the same
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
        short_seq_budget: float | None = None,
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
            gpu_batch_size: Batch size for HuggingFace pipeline inference.
                Controls how many texts are fed to the GPU at once, independent
                of the Ray Data batch size. If None, auto-computed from model
                config and available GPU memory.
            short_seq_budget: Memory budget fraction for short sequences
                (shorter than half the model sequence length). If None,
                auto-computed from total GPU VRAM. Higher values use more
                GPU memory for short-text batches.
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

        # Store total VRAM for adaptive short-sequence budget
        model_device = next(self._core.pipeline.model.parameters()).device
        if model_device.type == "cuda":
            self._total_vram_bytes = torch.cuda.get_device_properties(model_device).total_memory
        else:
            self._total_vram_bytes = 0

        # Store user-provided short_seq_budget override (None = auto)
        self._short_seq_budget_override = short_seq_budget

        # Compute GPU batch size (worst case: all texts at max seq_len)
        estimated = self._estimate_gpu_batch_size()
        if gpu_batch_size is not None:
            self._gpu_batch_size = gpu_batch_size
            if gpu_batch_size < estimated:
                logger.warning(
                    f"gpu_batch_size={gpu_batch_size} is below the estimated maximum of {estimated}. "
                    f"GPU may be underutilized. Remove gpu_batch_size to auto-compute."
                )
        else:
            self._gpu_batch_size = estimated

        logger.info(
            f"TransformerInferenceActor initialized: model={model_name}, "
            f"device={self._core.get_device_info()}, gpu_batch_size={self._gpu_batch_size}, "
            f"short_seq_budget={self._short_seq_budget():.2f}, "
            f"token_budget={self._token_budget}, window_overlap={self._window_overlap}"
        )

    @property
    def model_pipeline(self) -> Any:
        """Get the model pipeline (for backwards compatibility)."""
        return self._core.pipeline

    def _short_seq_budget(self) -> float:
        """Return the memory budget fraction for short sequences.

        When texts are shorter than half the model sequence length, fixed
        per-sample costs (logits, embeddings) that are not captured by
        _per_sample_bytes become significant. The budget compensates for
        this gap. On GPUs with more VRAM these fixed costs are a smaller
        fraction of total memory, so the budget can be higher.

        Returns a value between 0.6 and 0.8 based on total GPU VRAM,
        or the user-provided override if set.
        """
        if self._short_seq_budget_override is not None:
            return self._short_seq_budget_override
        total_gb = self._total_vram_bytes / (1024**3)
        if total_gb >= _VRAM_TIER_HIGH_GB:
            return 0.8
        if total_gb > _VRAM_TIER_MID_GB:
            return 0.7
        return 0.6

    @staticmethod
    def _per_sample_bytes(
        num_heads: int, seq_len: int, hidden_size: int, intermediate_size: int, dtype_bytes: int
    ) -> int:
        """Per-sample activation memory for a single transformer layer.

        Based on EleutherAI's Transformer Math (inference, single layer peak):
            attention scores + FFN intermediate + hidden I/O

        Ref: https://blog.eleuther.ai/transformer-math/
        """
        attention = num_heads * seq_len * seq_len * dtype_bytes
        ffn = intermediate_size * seq_len * dtype_bytes
        hidden_io = 2 * hidden_size * seq_len * dtype_bytes
        return attention + ffn + hidden_io

    def _estimate_gpu_batch_size(self) -> int:
        """Estimate max GPU batch size from model config and free GPU memory.

        Budgets from the driver's *actual* free memory via
        ``torch.cuda.mem_get_info`` (which accounts for the loaded model, other
        processes, reserved-but-unallocated blocks and fragmentation) rather than
        ``total - memory_allocated`` (which only sees PyTorch's own live
        allocations and so overestimates on a busy or fragmented GPU). Uses 90% of
        free memory to leave allocator headroom, then caps the result at
        ``_MAX_INITIAL_GPU_BATCH`` so the worst-case (max-seq) first forward stays
        modest; OOM recovery splits below this as needed. The cap bounds this
        max-seq base only — ``_batch_cap_for_tokens`` scales it up for shorter
        windows, so the actual per-forward window count can exceed the cap.

        Falls back to 64 on CPU or if model config is unavailable.
        """
        model = self._core.pipeline.model
        device = next(model.parameters()).device
        if device.type != "cuda":
            return 64

        config = model.config
        num_heads = getattr(config, "num_attention_heads", 12)
        hidden_size = getattr(config, "hidden_size", 768)
        intermediate_size = getattr(config, "intermediate_size", 4 * hidden_size)
        dtype_bytes = 2 if self._core.dtype == torch.float16 else 4

        per_sample = self._per_sample_bytes(num_heads, self._seq_len, hidden_size, intermediate_size, dtype_bytes)

        # Actual free memory on the device (bytes), not total - allocated.
        free, _total = torch.cuda.mem_get_info(device)

        max_batch = max(1, int(free * 0.9 / per_sample))
        capped = min(max_batch, _MAX_INITIAL_GPU_BATCH)

        logger.info(
            f"GPU batch size auto-computed: {capped} "
            f"(estimate={max_batch}, cap={_MAX_INITIAL_GPU_BATCH}, free={free / 1024**3:.1f}GB, "
            f"per_sample={per_sample / 1024**2:.1f}MB, seq_len={self._seq_len}, "
            f"heads={num_heads}, hidden={hidden_size})"
        )
        return capped

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

    def _batch_cap_for_tokens(self, max_tokens: int) -> int:
        """Max windows per forward when the longest window has ``max_tokens`` tokens.

        HF pads every window in a forward to the batch's longest member, so
        per-forward memory is set by that longest window. When it is shorter than
        ``seq_len``, per-sample memory drops and more windows fit. Fed the
        **exact** token count from the single tokenization (no chars/4 proxy), it
        scales the max-seq base batch by the ``_per_sample_bytes`` ratio between
        worst-case and actual sequence length.

        The result may exceed ``_MAX_INITIAL_GPU_BATCH`` for short windows: that
        cap bounds the *worst-case* (max-seq) base, which this scales *up* by
        ``per_sample_worst / per_sample_actual``. This stays within the free-memory
        budget by construction — the un-capped base already equals the free-memory
        limit at max seq length, so the capped base (<= the cap) times the scale is
        strictly below the limit at the (shorter) actual length. If that model is
        ever optimistic, the OOM window-batch shrink is the backstop.

        This is the length-only cap used by window bucketing;
        :meth:`_forward_windows_with_shrink` clamps it to the windows in hand.
        """
        # Account for the special tokens added at forward time so the sizing
        # matches the padded sequence length actually forwarded.
        effective_seq = min(max(max_tokens + self._num_special_tokens, 1), self._seq_len)

        if effective_seq >= self._seq_len:
            return self._gpu_batch_size

        config = self._core.pipeline.model.config
        num_heads = getattr(config, "num_attention_heads", 12)
        hidden_size = getattr(config, "hidden_size", 768)
        intermediate_size = getattr(config, "intermediate_size", 4 * hidden_size)
        dtype_bytes = 2 if self._core.dtype == torch.float16 else 4

        per_sample_worst = self._per_sample_bytes(num_heads, self._seq_len, hidden_size, intermediate_size, dtype_bytes)
        per_sample_actual = self._per_sample_bytes(
            num_heads, effective_seq, hidden_size, intermediate_size, dtype_bytes
        )

        scale = per_sample_worst / max(per_sample_actual, 1)
        # At long sequences the formula overestimates (attention dominates), so
        # we can use more memory. At short sequences, fixed per-sample costs
        # (logits, embeddings) dominate, so use a tighter budget.
        budget = 0.9 if effective_seq > self._seq_len // 2 else self._short_seq_budget()
        adjusted = int(self._gpu_batch_size * scale * budget / 0.9)
        return max(1, adjusted)

    def _run_inference_raw_with_oom_recovery(self, texts: list[str]) -> list[list[dict]]:
        """Tokenize once, window long chunks, forward in buckets, merge results.

        This is the actor's whole inference path. It tokenizes every chunk exactly
        once (no truncation), windows any chunk over the per-window token budget
        instead of dropping its tail, buckets the windows into memory-predictable
        forward batches, forwards each bucket with an OOM-safe window-batch shrink
        (never re-tokenizing), and merges each chunk's window predictions back into
        one list. The emitted per-chunk contract is unchanged; overlap-region
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

        # 3+4+5. Bucket windows by real token length and forward each bucket with
        # an OOM-safe window-batch shrink over the same (already tokenized) windows.
        for group in self._bucket_windows(windows):
            group_windows = [windows[i] for i in group]
            group_results = self._forward_windows_with_shrink(group_windows)
            # 6. Merge each window's predictions back onto its owner chunk. Offsets
            # are already chunk-relative, so this is a plain concatenation; order
            # across windows does not matter (downstream aggregation sorts by
            # start position).
            for window, preds in zip(group_windows, group_results, strict=True):
                results[window.owner].extend(preds)

        return results

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

    def _bucket_windows(self, windows: list[_Window]) -> list[list[int]]:
        """Partition window indices into length-homogeneous groups (ascending).

        Orders windows by content-token length, then greedily grows each group
        until adding the next (longer) window would exceed the memory-safe batch
        cap at that longer length (:meth:`_batch_cap_for_tokens`). Merging uses the
        carried owner index, so output is unaffected by this reordering.
        """
        m = len(windows)
        order = sorted(range(m), key=lambda i: len(windows[i].content_ids))

        groups: list[list[int]] = []
        pos = 0
        while pos < m:
            end = pos + 1
            while end < m and (end - pos + 1) <= self._batch_cap_for_tokens(len(windows[order[end]].content_ids)):
                end += 1
            groups.append(order[pos:end])
            pos = end
        return groups

    def _forward_windows_with_shrink(self, windows: list[_Window]) -> list[list[dict]]:
        """Forward one length-homogeneous window group, shrinking on CUDA OOM.

        The group is sized to fit at its longest window's length, so the first
        forward should succeed; the shrink is the backstop if the memory model was
        optimistic. Each sub-batch is one forward over pre-tokenized windows
        (:meth:`TransformerCore.forward_windows`), so a mid-group OOM discards no
        already-computed results and — crucially — the retry never re-tokenizes.
        The retry size is derived from the sub-batch that actually OOMed, so every
        retry is strictly smaller and makes progress.

        Args:
            windows: A length-homogeneous group of windows.

        Returns:
            Raw token lists aligned to ``windows``.

        Raises:
            RuntimeError: If OOM persists at a single window, or for any non-OOM
                error (re-raised unchanged).
        """
        m = len(windows)
        out: list[list[dict] | None] = [None] * m
        if m == 0:
            return []

        max_tokens = max(len(w.content_ids) for w in windows)
        batch_size = max(1, min(self._batch_cap_for_tokens(max_tokens), m))

        start = 0
        while start < m:
            chunk = windows[start : start + batch_size]
            payload = [(w.content_ids, w.offsets, w.text) for w in chunk]
            try:
                chunk_results = self._core.forward_windows(payload)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise  # non-OOM error: propagate unchanged

                if len(chunk) <= 1:
                    raise RuntimeError("CUDA OOM on a single token window") from e

                # Count the handled OOM so tests can prove the recovery path ran.
                # Defensive getattr: actors built via __new__ in unit tests skip
                # __init__ and so never set _handled_oom_count.
                self._handled_oom_count = getattr(self, "_handled_oom_count", 0) + 1
                self._record_handled_oom()

                # Derive the retry size from the sub-batch that actually OOMed, not
                # the stored batch_size, so a too-large batch_size cannot retry the
                # same slice unchanged.
                batch_size = max(1, len(chunk) // 2)
                logger.warning(
                    "CUDA OOM on %d windows; shrinking window batch to %d and retrying", len(chunk), batch_size
                )
                # A1 already released the failed forward's tensors, so this
                # reclaims real VRAM before the smaller retry.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            for offset, preds in enumerate(chunk_results):
                out[start + offset] = preds
            start += len(chunk)

        return cast(list[list[dict]], out)


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
    short_seq_budget: float | None = None,
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
        gpu_batch_size: Batch size for HF pipeline inference (None = auto-compute).
        short_seq_budget: Memory budget fraction for short sequences (None = auto).
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
                short_seq_budget=short_seq_budget,
                allow_huggingface_download=allow_huggingface_download,
                window_overlap=window_overlap,
            )

    return ConfiguredTransformerActor


# Backwards compatibility alias
create_transformer_actor_class = create_transformer_actor
