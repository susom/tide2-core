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
import math
import os
from typing import Any
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
# Note this bounds the max-seq base only, not the final per-forward text count:
# ``_effective_batch_size`` scales this base *up* for short texts, so short-text
# batches can exceed 512 (see that method for why that stays memory-safe).
_MAX_INITIAL_GPU_BATCH = 512

# Length-homogeneity bound for in-actor length bucketing. A group may only grow
# while its longest member is within this factor of its shortest, capping the
# worst-case in-group padding waste (HF pads every member of a forward to the
# group's longest). Without it, when the memory cap already fits the whole batch
# (cap >= batch size) every text lands in one group and short texts are padded up
# to the batch's single longest text — the degenerate case that defeats bucketing.
# 1.5 (tolerate <=50% padding) was the knee on a real mixed-length SHIELD batch:
# it cut padded-token waste from ~2.2x to ~1.3x and sped the forward ~1.7x, while
# tighter factors only added forward-launch overhead for no further gain.
_MAX_GROUP_LENGTH_SPAN = 1.5

# Below this many characters the span bound is not enforced: padding waste on very
# short texts is negligible in absolute terms, so grouping them freely avoids
# over-splitting a crowd of tiny texts into many tiny forwards.
_GROUP_SPAN_MIN_ANCHOR_CHARS = 128


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
    - Batch inference with automatic OOM recovery
    - JSON serialization of raw token predictions

    Batching is owned by the actor, not by Ray Data's map_batches(batch_size=N):
    each received batch is length-sorted and partitioned into length-homogeneous
    groups (see :meth:`_run_inference_raw_with_oom_recovery`) so every forward
    pads to a near-uniform length instead of to the batch's single longest text.
    Each group runs as its own forward that halves its batch size on CUDA OOM and
    retries, recovering without discarding already-computed results (see
    :meth:`_infer_group_with_batch_shrink`).
    """

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        bucket_name: str | None = None,
        project_id: str | None = None,
        gpu_batch_size: int | None = None,
        short_seq_budget: float | None = None,
        tokenizer_workers: int | None = None,
        allow_huggingface_download: bool = True,
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
            tokenizer_workers: Size of the fast tokenizer's Rust (rayon) thread
                pool for CPU-side tokenization. If None, it is left at the
                library default (all cores) unless Ray assigned this actor a CPU
                floor (``transformer_cpus``), in which case it is pinned to that
                floor so concurrent GPU actors do not oversubscribe the CPUs.
                Set explicitly to give tokenization a fixed number of cores.
            allow_huggingface_download: If True, fall back to HuggingFace Hub
                when local cache and GCS both miss.
        """
        self.model_name = model_name

        # CPUs Ray reserved for this actor (0 when GPU-pinned without a floor).
        # get_assigned_resources() is only valid inside a Ray worker; this actor is
        # also constructed in-process (direct inference / the recognizer path / GPU
        # verification), where it raises — treat that as "no reservation info".
        import ray

        try:
            assigned = ray.get_runtime_context().get_assigned_resources()
            assigned_cpus = self._cpu_floor(assigned.get("CPU", 0))
        except Exception:
            assigned_cpus = 0

        # Determine device for explicit GPU placement
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            device = f"cuda:{device_idx}"
        else:
            device = "cpu"
            logger.warning("No GPU detected, using CPU (performance will be degraded)")

        # CPU branch: cap torch threads to the CPUs Ray reserved for this actor
        # so that actors x threads <= total CPUs (no oversubscription).
        if device == "cpu":
            n = assigned_cpus or 1
            torch.set_num_threads(n)
            logger.info("CPU inference: capping torch to %d thread(s) per Ray allocation", n)

        # Tokenization is CPU work on this (often GPU-pinned) actor. Give the fast
        # tokenizer's rayon pool a bounded, configurable core count so several GPU
        # actors on one node don't each grab every core. Only pins when asked
        # explicitly or when a CPU floor was assigned — otherwise the library
        # default (all cores) is left intact so single-actor boxes don't regress.
        if tokenizer_workers is not None:
            workers = tokenizer_workers
        elif assigned_cpus >= 1:
            workers = assigned_cpus
        else:
            workers = None
        if workers is not None:
            self._configure_tokenizer_parallelism(workers)

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
            f"short_seq_budget={self._short_seq_budget():.2f}"
        )

    @staticmethod
    def _cpu_floor(assigned_cpu: float | None) -> int:
        """Round a Ray CPU reservation up to a whole tokenizer-worker floor.

        Ray can assign *fractional* CPUs (e.g. ``transformer_cpus=0.25`` to pack
        several GPU actors onto one node). Truncating with ``int()`` would map any
        fraction below 1.0 to ``0`` and leave the fast tokenizer's rayon pool at
        all cores, so co-located GPU actors oversubscribe the CPUs. ``math.ceil``
        instead counts any positive reservation as at least one worker; ``0`` (a
        GPU-pinned actor with no CPU floor) stays ``0``.

        Args:
            assigned_cpu: The ``CPU`` value from Ray's assigned resources (may be
                fractional, ``0``, or ``None``).

        Returns:
            The whole-number CPU floor: ``ceil(assigned_cpu)`` for positive
            values, else ``0``.
        """
        return math.ceil(assigned_cpu or 0)

    @staticmethod
    def _configure_tokenizer_parallelism(workers: int) -> None:
        """Pin the fast tokenizer's rayon pool and enable HF parallelism.

        The HuggingFace fast tokenizer parallelizes over a Rust ``rayon`` thread
        pool sized from ``RAYON_NUM_THREADS`` (read once, at first tokenize) and
        gated by ``TOKENIZERS_PARALLELISM``. Uses ``setdefault`` so an
        operator-set value always wins. A no-op for ``workers < 1``.
        """
        if workers < 1:
            return
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        os.environ.setdefault("RAYON_NUM_THREADS", str(workers))
        logger.info("Tokenizer parallelism pinned to %d rayon thread(s)", workers)

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
        max-seq base only — ``_effective_batch_size`` scales it up for shorter
        texts, so the actual per-forward text count can exceed the cap.

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

    def _batch_cap_for_seq(self, max_chars: int) -> int:
        """Max texts per forward at a given text length, ignoring how many exist.

        HF pads all texts to the longest in the batch, so per-forward memory is
        set by the batch's *longest* member. When that is shorter than seq_len,
        per-sample memory drops and more texts fit. Uses chars/4 as a cheap token
        estimate, then scales the base batch by the ``_per_sample_bytes`` ratio
        between worst-case and actual seq length.

        The result may exceed ``_MAX_INITIAL_GPU_BATCH`` for short texts: that cap
        bounds the *worst-case* (max-seq) base, which this scales *up* by
        ``per_sample_worst / per_sample_actual``. This stays within the free-memory
        budget by construction — the un-capped base already equals the free-memory
        limit at max seq length, so the capped base (<= the cap) times the scale is
        strictly below the limit at the (shorter) actual length. If that model is
        ever optimistic, the batch-shrink recovery is the backstop.

        This is the length-only cap used by length bucketing; ``_effective_batch_size``
        clamps it to the number of texts actually in hand.
        """
        effective_seq = min(max(max_chars // 4, 1), self._seq_len)

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

    def _effective_batch_size(self, texts: list[str]) -> int:
        """Batch size adapted to the batch's longest text, clamped to ``len(texts)``.

        Thin wrapper over :meth:`_batch_cap_for_seq` for callers that hand over a
        concrete list of texts.
        """
        max_chars = max(len(t) for t in texts)
        return min(len(texts), self._batch_cap_for_seq(max_chars))

    def _run_inference_raw_with_oom_recovery(self, texts: list[str]) -> list[list[dict]]:
        """Run raw inference with in-actor length bucketing and OOM recovery.

        HF pads every text in a forward to the batch's longest member, so a
        single long text in an otherwise-short batch inflates padding for the
        whole batch (measured 2.3-5.8x throughput loss, review §4/§6f). To remove
        that waste, this:

        1. orders the texts by length (a cheap ``len`` proxy);
        2. partitions the length-sorted order into groups bounded by *both* the
           memory-safe batch cap for the group's longest member
           (:meth:`_batch_cap_for_seq`) *and* a length span
           (``_MAX_GROUP_LENGTH_SPAN``). The span bound matters when the cap alone
           already admits the whole batch (cap >= batch size): without it every
           text would land in one group and short texts would be padded up to the
           batch's single longest, the degenerate case that defeats bucketing
           (measured ~1.85x forward slowdown on a mixed-length SHIELD batch). With
           both bounds each forward pads to a near-uniform length and per-forward
           memory is predictable (an OOM-safety bonus);
        3. runs each group as its own forward(s), halving the batch on CUDA OOM;
        4. reassembles outputs into the original input order via a carried index
           map.

        Bucketing is entirely in-actor — no global Ray Data sort/shuffle (which
        would risk the small-box deadlock). It is also the **single owner of
        sub-batching**: each group runs one forward at its sized batch, so no
        already-computed sub-results are discarded, and the batch-shrink on OOM
        composes with A1's source-level tensor release (the failed forward pins no
        VRAM, so ``empty_cache()`` after a handled OOM actually reclaims).

        Args:
            texts: List of text strings to process (assumed all non-empty; the
                caller filters empties).

        Returns:
            List of raw token lists (one per input text), aligned to ``texts``.

        Raises:
            RuntimeError: If OOM persists at batch size 1, or for any non-OOM
                error (re-raised unchanged).
        """
        n = len(texts)
        results: list[list[dict] | None] = [None] * n
        if n == 0:
            return cast(list[list[dict]], results)

        # Length-sorted view of the input indices (ascending). Reassembly uses the
        # carried original index, so output order is unaffected by this reordering.
        order = sorted(range(n), key=lambda i: len(texts[i]))

        pos = 0
        while pos < n:
            # Grow a length-homogeneous group from the sorted order. Stop once
            # adding the next (longer) text would either exceed the memory-safe
            # batch cap at that longer length, or stretch the group's length span
            # past ``_MAX_GROUP_LENGTH_SPAN`` (which would pad the group's shorter
            # members too aggressively). Since ``order`` is ascending, the group's
            # shortest member is ``order[pos]`` and the candidate ``order[end]`` is
            # always the group's new longest member. The span bound is relaxed for
            # very short groups, where padding waste is negligible.
            span_anchor = max(len(texts[order[pos]]), _GROUP_SPAN_MIN_ANCHOR_CHARS)
            end = pos + 1
            while (
                end < n
                and (end - pos + 1) <= self._batch_cap_for_seq(len(texts[order[end]]))
                and len(texts[order[end]]) <= span_anchor * _MAX_GROUP_LENGTH_SPAN
            ):
                end += 1

            group_idx = order[pos:end]
            group_texts = [texts[i] for i in group_idx]
            group_results = self._infer_group_with_batch_shrink(group_texts)
            for gi, preds in zip(group_idx, group_results, strict=True):
                results[gi] = preds
            pos = end

        # Every index was filled by exactly one successful group member.
        return cast(list[list[dict]], results)

    def _infer_group_with_batch_shrink(self, texts: list[str]) -> list[list[dict]]:
        """Run one length-homogeneous group, halving the batch on CUDA OOM.

        The group is already sized to fit at its longest member's length, so the
        first forward should succeed; the shrink is the backstop if the memory
        model was optimistic. Each chunk is a single forward (``infer_raw_direct``
        with ``batch_size == len(chunk)``), so no already-computed results are
        discarded on a mid-group OOM.

        Args:
            texts: A length-homogeneous group of non-empty texts.

        Returns:
            Raw token lists aligned to ``texts``.

        Raises:
            RuntimeError: If OOM persists at a single text, or for any non-OOM
                error (re-raised unchanged).
        """
        n = len(texts)
        out: list[list[dict] | None] = [None] * n
        batch_size = n

        start = 0
        while start < n:
            chunk = texts[start : start + batch_size]
            try:
                chunk_results = self._core.infer_raw_direct(chunk, batch_size=len(chunk))
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise  # non-OOM error: propagate unchanged

                if len(chunk) <= 1:
                    raise RuntimeError("CUDA OOM on a single text chunk") from e

                batch_size = max(1, batch_size // 2)
                logger.warning(
                    "CUDA OOM on %d texts; halving GPU batch size to %d and retrying", len(chunk), batch_size
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
    tokenizer_workers: int | None = None,
    allow_huggingface_download: bool = True,
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
        tokenizer_workers: Rayon thread-pool size for CPU tokenization
            (None = library default unless a CPU floor is assigned).
        allow_huggingface_download: If True, fall back to HuggingFace Hub
            when local cache and GCS both miss.

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
                tokenizer_workers=tokenizer_workers,
                allow_huggingface_download=allow_huggingface_download,
            )

    return ConfiguredTransformerActor


# Backwards compatibility alias
create_transformer_actor_class = create_transformer_actor
