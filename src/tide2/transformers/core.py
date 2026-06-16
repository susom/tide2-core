"""
Core transformer inference engine.

This module provides the TransformerCore class that encapsulates the shared logic
for transformer-based NER inference, used by both the Presidio recognizer and
the Ray actor.
"""

import logging
import threading
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import pipeline

from tide2.utils.gcs_resource_manager import resolve_model_path
from tide2.utils.text_processing import aggregate_bio_tokens

from .config import load_model_config

logger = logging.getLogger(__name__)

# Fixed schema of a raw BIO token prediction (see infer_raw). Used to build a
# stable dedupe key that does not depend on dict insertion order.
_RAW_PRED_KEYS = ("entity", "score", "start", "end", "word", "index")


class TransformerCore:
    """
    Core transformer inference engine used by both Presidio and Ray wrappers.

    This class handles:
    - Model configuration loading
    - Model path resolution (local or GCS)
    - Pipeline loading with device placement options
    - Raw inference (returns BIO tokens)
    - BIO token aggregation into entity spans

    Thread Safety:
        Pipeline loading is protected by a lock for thread-safe lazy loading.
        Inference is thread-safe once the pipeline is loaded.

    Args:
        model_name: Name of the model configuration to load
        model_path: Optional explicit path to model (overrides GCS resolution)
        bucket_name: Optional GCS bucket name for model loading
        project_id: Optional GCP project ID for model loading
        device: Device placement strategy:
            - "auto": Use accelerate's device_map="auto" (recommended for single-text)
            - "cuda:N": Explicit GPU placement (recommended for batch/actors)
            - "cpu": Force CPU placement
            - None: Auto-detect (cuda:0 if available, else cpu)
        dtype: Model dtype (default: torch.float16 for memory efficiency)
        load_immediately: If True, load pipeline in __init__. If False, lazy load.
        local_files_only: If True, don't download from HuggingFace (for cached models)
        compile_model: Controls torch.compile behavior:
            - None (default): Auto-detect. If compiled_cache.bin exists alongside
              model weights, compile automatically.
            - True: Require compilation. Raises FileNotFoundError if cache missing.
            - False: Skip compilation even if cache file exists.
        compile_cache_path: Override path to mega-cache .bin file. If None, looks
            for compiled_cache.bin in the resolved model directory.
        allow_huggingface_download: If True (default), fall back to downloading
            from HuggingFace Hub when local cache and GCS both miss.

    Example:
        # For Presidio (lazy loading, auto device)
        core = TransformerCore(model_name="stanford_deidentifier", device="auto")

        # For Ray actor (immediate loading, explicit GPU)
        core = TransformerCore(
            model_name="stanford_deidentifier",
            device="cuda:0",
            load_immediately=True,
            local_files_only=True,
        )
    """

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        bucket_name: str | None = None,
        project_id: str | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
        load_immediately: bool = False,
        local_files_only: bool = False,
        compile_model: bool | None = None,
        compile_cache_path: str | None = None,
        allow_huggingface_download: bool = True,
    ) -> None:
        """Initialize the transformer core.

        Args:
            model_name: Key in ``bert_transformer_configuration.json``.
            model_path: Local path override for the model directory.
            bucket_name: GCS bucket for auto-download.
            project_id: GCP project for GCS access.
            device: Device string (``"cpu"``, ``"cuda"``, or ``"auto"``).
            dtype: Torch dtype for model weights.
            load_immediately: If True, load the pipeline during init.
            local_files_only: Restrict HuggingFace to local files only.
            compile_model: Whether to use a compiled model cache.
            compile_cache_path: Path to the compiled ``.bin`` cache file.
            allow_huggingface_download: If True, fall back to HuggingFace Hub
                when local cache and GCS both miss.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.compile_model = compile_model
        self.compile_cache_path = compile_cache_path

        # Load configuration
        self._config = load_model_config(model_name)

        # Resolve model path
        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = resolve_model_path(
                model_name=model_name,
                bucket_name=bucket_name,
                project_id=project_id,
                allow_huggingface_download=allow_huggingface_download,
            )
            logger.info(f"Resolved model path: {self.model_path}")

        # Pipeline state
        self._pipeline: Any | None = None
        self._pipeline_lock = threading.Lock()

        # Load immediately if requested
        if load_immediately:
            self._load_pipeline()

    @property
    def config(self) -> dict[str, Any]:
        """Model configuration from bert_transformer_configuration.json."""
        return self._config

    @property
    def ignore_labels(self) -> list[str]:
        """Labels to ignore during inference (e.g., ["O"])."""
        return self._config.get("LABELS_TO_IGNORE", ["O"])

    @property
    def supported_entities(self) -> list[str]:
        """Presidio-compatible entity types supported by this model."""
        return self._config.get("PRESIDIO_SUPPORTED_ENTITIES", [])

    @property
    def is_loaded(self) -> bool:
        """Check if the pipeline is loaded."""
        return self._pipeline is not None

    @property
    def pipeline(self) -> Any:
        """Get the pipeline, loading it if not already loaded."""
        return self._ensure_pipeline_loaded()

    def _ensure_pipeline_loaded(self) -> Any:
        """Lazy-load the pipeline on first use with thread-safe initialization.

        Returns:
            The loaded pipeline instance
        """
        if self._pipeline is None:
            with self._pipeline_lock:
                # Double-check pattern
                if self._pipeline is None:
                    self._load_pipeline()

        return self._pipeline

    def _load_pipeline(self) -> None:
        """Load the transformer pipeline with the configured device placement."""
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Loading NER pipeline from {self.model_path}")

        if self.device == "auto":
            # Use accelerate's device_map for automatic placement
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=False,
                dtype=self.dtype,
                local_files_only=self.local_files_only,
            )
            model.eval()
            device_for_pipeline = None  # Let pipeline infer from model

        elif self.device == "cpu":
            # Force CPU placement
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
                local_files_only=self.local_files_only,
            )
            model.eval()
            device_for_pipeline = -1

        elif self.device is not None and self.device.startswith("cuda"):
            # Explicit GPU placement
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
                dtype=self.dtype,
                local_files_only=self.local_files_only,
            )
            model = model.to(self.device)
            model.eval()
            # Extract device index for pipeline
            device_for_pipeline = int(self.device.split(":")[1])

        # Auto-detect: use CUDA if available
        elif torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            device_str = f"cuda:{device_idx}"
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
                dtype=self.dtype,
                local_files_only=self.local_files_only,
            )
            model = model.to(device_str)
            model.eval()
            device_for_pipeline = device_idx
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
                local_files_only=self.local_files_only,
            )
            model.eval()
            device_for_pipeline = -1

        # Apply torch.compile with mega-cache
        cache_path = self._resolve_compile_cache_path()
        if cache_path is not None:
            logger.info(f"[{thread_name}] Loading compile cache from {cache_path}")
            torch.compiler.load_cache_artifacts(cache_path.read_bytes())
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            logger.info(f"[{thread_name}] Model compiled with fullgraph=True, mode=reduce-overhead")

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=self.local_files_only)

        # Build pipeline kwargs
        # Note: transformers 5.x removed the `framework` argument from pipeline()
        # (TensorFlow/Flax support was dropped, so everything is PyTorch).
        pipeline_kwargs: dict[str, Any] = {
            "task": "token-classification",
            "model": model,
            "tokenizer": tokenizer,
            "aggregation_strategy": "none",  # Return raw BIO tokens
            "ignore_labels": self.ignore_labels,
        }

        # Add device only if we have an explicit one (not for device_map="auto")
        if device_for_pipeline is not None:
            pipeline_kwargs["device"] = device_for_pipeline

        self._pipeline = pipeline(**pipeline_kwargs)

        # Store direct references for infer_raw_direct (bypasses HF pipeline dispatch)
        self._model = model
        self._tokenizer = tokenizer
        self._id2label = model.config.id2label
        self._ignore_labels_set = set(self.ignore_labels)

        # Log device info
        model_device = next(model.parameters()).device
        logger.info(f"[{thread_name}] Pipeline loaded on device: {model_device}")

    def _resolve_compile_cache_path(self) -> Path | None:
        """Resolve the path to the compiled cache .bin file.

        Behavior depends on self.compile_model:
            - None: Auto-detect. Return path if compiled_cache.bin exists, else None.
            - True: Require cache. Raise FileNotFoundError if missing.
            - False: Skip compilation. Return None immediately.

        The cache file is expected alongside the model weights at
        <model_path>/compiled_cache.bin, or at compile_cache_path if overridden.

        Returns:
            Path to the cache file, or None to skip compilation.

        Raises:
            FileNotFoundError: If compile_model is True and the cache file is missing.
        """
        if self.compile_model is False:
            return None

        if self.compile_cache_path is not None:
            path = Path(self.compile_cache_path)
        else:
            path = Path(self.model_path) / "compiled_cache.bin"

        if path.is_file():
            return path

        if self.compile_model is True:
            raise FileNotFoundError(
                f"Compiled cache file not found at {path}. "
                f"Generate it with: python scripts/compile_model.py save --output {path}"
            )

        # compile_model is None (auto-detect) and file not found — skip
        return None

    def infer_raw(self, texts: list[str], batch_size: int | None = None) -> list[list[dict]]:
        """Run raw inference on texts, returning BIO tokens.

        This method runs the transformer pipeline on a batch of texts and returns
        the raw predictions without BIO aggregation.

        Args:
            texts: List of text strings to process
            batch_size: Optional batch size for pipeline (default: process all at once)

        Returns:
            List of prediction lists, one per input text. Each prediction is a dict:
            {
                "entity": "B-PERSON",
                "score": 0.95,
                "start": 0,
                "end": 4,
                "word": "John",
                "index": 1,
            }
        """
        pipeline_instance = self._ensure_pipeline_loaded()

        if not texts:
            return []

        # Run pipeline
        if batch_size is not None:
            results = pipeline_instance(texts, batch_size=batch_size)
        else:
            results = pipeline_instance(texts)

        # Handle single-text case (pipeline returns list of dicts, not list of lists)
        if len(texts) == 1 and results and isinstance(results[0], dict):
            return [results]

        return results

    def infer_raw_direct(self, texts: list[str], batch_size: int | None = None) -> list[list[dict]]:
        """Run inference bypassing the HF pipeline dispatch loop.

        Tokenizes the entire batch in one call, runs a single GPU forward pass
        (or sub-batched forward passes if batch_size < len(texts)), and extracts
        raw token predictions using offset_mapping. This avoids the per-text
        preprocess/postprocess Python loops in HuggingFace's ChunkPipeline.

        Output format matches infer_raw(): list of lists of dicts with keys
        {entity, score, start, end, word, index}.

        Args:
            texts: List of text strings to process.
            batch_size: Max texts per GPU forward pass. If None, process all at once.

        Returns:
            List of prediction lists, one per input text.
        """
        self._ensure_pipeline_loaded()

        if not texts:
            return []

        if batch_size is None:
            batch_size = len(texts)

        all_results: list[list[dict]] = []

        for start in range(0, len(texts), batch_size):
            sub_texts = texts[start : start + batch_size]
            sub_results = self._forward_batch_direct(sub_texts)
            all_results.extend(sub_results)

        return all_results

    def _forward_batch_direct(self, texts: list[str]) -> list[list[dict]]:
        """Single batch: tokenize → GPU forward → extract predictions."""
        model = self._model
        tokenizer = self._tokenizer
        device = next(model.parameters()).device

        # Batch tokenize
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        offset_mapping = encoded.pop("offset_mapping")  # (batch, seq_len, 2) — keep on CPU
        special_tokens_mask = encoded.pop("special_tokens_mask")  # (batch, seq_len) — keep on CPU

        # Move input tensors to GPU
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Single forward pass
        with torch.no_grad():
            logits = model(**encoded).logits  # (batch, seq_len, num_labels)

        # Softmax + argmax on GPU, then transfer to CPU
        probs = torch.softmax(logits, dim=-1)
        scores_max, label_ids = probs.max(dim=-1)  # (batch, seq_len)

        scores_np = scores_max.cpu().numpy()
        label_ids_np = label_ids.cpu().numpy()
        offset_np = offset_mapping.numpy()
        special_np = special_tokens_mask.numpy()

        # Extract per-text predictions
        id2label = self._id2label
        ignore = self._ignore_labels_set
        results: list[list[dict]] = []

        for i, text in enumerate(texts):
            preds: list[dict] = []
            for j in range(scores_np.shape[1]):
                if special_np[i, j]:
                    continue
                label = id2label[label_ids_np[i, j]]
                if label in ignore:
                    continue
                s, e = int(offset_np[i, j, 0]), int(offset_np[i, j, 1])
                preds.append(
                    {
                        "entity": label,
                        "score": float(scores_np[i, j]),
                        "start": s,
                        "end": e,
                        "word": text[s:e],
                        "index": j,
                    }
                )
            results.append(preds)

        return results

    def infer_single_raw(self, text: str) -> list[dict]:
        """Run raw inference on a single text.

        Args:
            text: Text to process

        Returns:
            List of raw BIO token predictions
        """
        pipeline_instance = self._ensure_pipeline_loaded()

        if not text:
            return []

        return pipeline_instance(text)

    def infer_aggregated(self, text: str) -> list[dict]:
        """Run inference on a single text with BIO aggregation.

        This method runs inference and aggregates consecutive BIO tokens into
        entity spans.

        Args:
            text: Text to process

        Returns:
            List of aggregated entity predictions:
            {
                "entity_group": "PERSON",
                "score": 0.95,
                "start": 0,
                "end": 10,
                "word": "John Smith",
            }
        """
        raw_predictions = self.infer_single_raw(text)

        if not raw_predictions:
            return []

        # Remove duplicates (can occur from chunking at caller level)
        raw_predictions = [dict(t) for t in {tuple(d.items()) for d in raw_predictions}]

        # Aggregate BIO tokens
        return aggregate_bio_tokens(raw_predictions, text)

    def infer_batch_aggregated(self, texts: list[str], batch_size: int | None = None) -> list[list[dict]]:
        """Run inference on a batch of texts with BIO aggregation.

        Args:
            texts: List of texts to process
            batch_size: Optional batch size for pipeline

        Returns:
            List of aggregated entity prediction lists, one per input text
        """
        raw_results = self.infer_raw(texts, batch_size=batch_size)

        aggregated_results = []
        for text, raw_preds in zip(texts, raw_results, strict=True):
            if not raw_preds:
                aggregated_results.append([])
                continue

            # Remove duplicates. Build the dedupe key from the fixed prediction
            # schema in a stable order so it does not depend on dict insertion
            # order (O(k) per dict, no per-key sort).
            deduped_preds = [
                dict(zip(_RAW_PRED_KEYS, key, strict=True))
                for key in {tuple(d[k] for k in _RAW_PRED_KEYS) for d in raw_preds}
            ]

            # Aggregate BIO tokens
            aggregated = aggregate_bio_tokens(deduped_preds, text)
            aggregated_results.append(aggregated)

        return aggregated_results

    @property
    def model_max_length(self) -> int:
        """Maximum input length for the tokenizer."""
        pipeline_instance = self._ensure_pipeline_loaded()
        return getattr(pipeline_instance.tokenizer, "model_max_length", 512)

    def get_device_info(self) -> str:
        """Get current device information."""
        if not self.is_loaded:
            return "not loaded"

        try:
            model = self._pipeline.model
            device = next(model.parameters()).device
            if device.type == "cuda":
                device_name = torch.cuda.get_device_name(device.index)
                return f"{device} ({device_name})"
            return str(device)
        except Exception:
            return "unknown"
