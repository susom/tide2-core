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
from typing import Any

import numpy as np
import torch

from tide2.transformers import TransformerCore
from tide2.utils.text_processing import aggregate_bio_tokens

logger = logging.getLogger(__name__)

# VRAM thresholds (GB) for short-sequence budget tiers.
# Maps to NVIDIA product lines: L4/RTX (<=24), A6000/A100-40 (24-80), H100/A100-80 (>=80).
_VRAM_TIER_HIGH_GB = 80
_VRAM_TIER_MID_GB = 24


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

    Batch size is controlled by Ray Data's map_batches(batch_size=N) — the actor
    processes whatever batch it receives without internal sub-batching.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        bucket_name: str | None = None,
        project_id: str | None = None,
        compile_model: bool | None = None,
        compile_cache_path: str | None = None,
        gpu_batch_size: int | None = None,
        short_seq_budget: float | None = None,
        allow_huggingface_download: bool = True,
    ) -> None:
        """
        Initialize the actor with a transformer model on GPU.

        Args:
            model_name: Name of the model configuration to load.
            model_path: Optional explicit model path (overrides GCS resolution).
            bucket_name: Optional GCS bucket name for model loading.
            project_id: Optional GCP project ID for model loading.
            compile_model: If True, apply torch.compile with mega-cache.
            compile_cache_path: Path to compiled cache .bin file.
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
        """
        self.model_name = model_name

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
            compile_model=compile_model,
            compile_cache_path=compile_cache_path,
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

        Uses 80% of free memory to leave room for CUDA allocator fragmentation;
        OOM recovery handles the rest.

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

        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        free = total - allocated

        max_batch = max(1, int(free * 0.9 / per_sample))

        logger.info(
            f"GPU batch size auto-computed: {max_batch} "
            f"(free={free / 1024**3:.1f}GB, per_sample={per_sample / 1024**2:.1f}MB, "
            f"seq_len={self._seq_len}, heads={num_heads}, hidden={hidden_size})"
        )
        return max_batch

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
        raw_results = self._run_inference_raw_with_oom_recovery(valid_texts)

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

    def _effective_batch_size(self, texts: list[str]) -> int:
        """Compute batch size adapted to actual text lengths.

        HF pipeline pads all texts to the longest in the batch. When texts are
        shorter than seq_len, per-sample memory drops and we can fit more.

        Uses chars/4 as a cheap token count estimate, then scales using
        _per_sample_bytes ratio between worst-case and actual seq length.
        """
        max_chars = max(len(t) for t in texts)
        effective_seq = min(max(max_chars // 4, 1), self._seq_len)

        if effective_seq >= self._seq_len:
            return min(len(texts), self._gpu_batch_size)

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
        return max(1, min(len(texts), adjusted))

    def _run_inference_raw_with_oom_recovery(self, texts: list[str]) -> list[list[dict]]:
        """
        Run raw inference with OOM recovery.

        Passes all texts to TransformerCore.infer_raw(). If CUDA OOM
        occurs, splits the batch in half and retries each half separately.

        Returns raw BIO tokens (not aggregated).

        Args:
            texts: List of text strings to process.

        Returns:
            List of raw token lists (one per input text).

        Raises:
            RuntimeError: If OOM persists even with single-item batches.
        """
        try:
            return self._core.infer_raw_direct(texts, batch_size=self._effective_batch_size(texts))
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(texts) <= 1:
                raise RuntimeError("CUDA OOM on a single text chunk") from e

            mid = len(texts) // 2
            logger.warning(f"CUDA OOM on {len(texts)} texts, splitting into {mid} + {len(texts) - mid}")
            left = self._run_inference_raw_with_oom_recovery(texts[:mid])
            right = self._run_inference_raw_with_oom_recovery(texts[mid:])
            return left + right


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
    compile_model: bool | None = None,
    compile_cache_path: str | None = None,
    gpu_batch_size: int | None = None,
    short_seq_budget: float | None = None,
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
        compile_model: If True, apply torch.compile with mega-cache.
        compile_cache_path: Path to compiled cache .bin file.
        gpu_batch_size: Batch size for HF pipeline inference (None = auto-compute).
        short_seq_budget: Memory budget fraction for short sequences (None = auto).
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
                compile_model=compile_model,
                compile_cache_path=compile_cache_path,
                gpu_batch_size=gpu_batch_size,
                short_seq_budget=short_seq_budget,
                allow_huggingface_download=allow_huggingface_download,
            )

    return ConfiguredTransformerActor


# Backwards compatibility alias
create_transformer_actor_class = create_transformer_actor
