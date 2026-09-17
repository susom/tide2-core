"""
Core transformer inference engine.

This module provides the TransformerCore class that encapsulates the shared logic
for transformer-based NER inference, used by the Presidio recognizer.
"""

import logging
import os
import shutil
import socket
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from filelock import FileLock

# transformers exposes these via a lazy module __getattr__, which ty can't resolve statically.
from transformers import AutoModelForTokenClassification  # ty: ignore[unresolved-import]
from transformers import AutoTokenizer  # ty: ignore[unresolved-import]
from transformers import pipeline  # ty: ignore[unresolved-import]

from .config import load_model_config

logger = logging.getLogger(__name__)

# Weight file names that indicate a complete, usable model directory.
_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _get_cache_dir() -> Path:
    """Get the TIDE model cache directory ($TIDE_CACHE_DIR or ~/.cache/tide2)."""
    cache_dir_str = os.getenv("TIDE_CACHE_DIR")
    cache_dir = Path(cache_dir_str).expanduser().resolve() if cache_dir_str else Path.home() / ".cache" / "tide2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _validate_model_directory(model_dir: Path) -> bool:
    """Check that a model directory has config.json and at least one weight file."""
    if not (model_dir / "config.json").is_file():
        return False
    return any((model_dir / f).is_file() for f in _WEIGHT_FILES)


def _model_resolution_lock(model_name: str) -> FileLock:
    """Per-model inter-process lock file, separate from the model directory itself
    (which gets deleted/recreated under the lock) so the lock file always survives.
    """
    lock_dir = _get_cache_dir() / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_dir / f"{model_name.replace('/', '__')}.lock"))


def _resolve_model_path(model_name: str, allow_huggingface_download: bool) -> str:
    """
    Resolve a model name to a local directory.

    Checks the local TIDE cache first, then falls back to downloading from
    HuggingFace Hub (using model_name as the repo id) when allowed.

    Guarded by an inter-process file lock keyed on model_name: independent
    processes (e.g. the pipeline runner's spawned GPU workers) can call this
    concurrently for the same model, and without a lock one process can see
    another's still-downloading directory as invalid (incomplete) and
    shutil.rmtree() it out from under the in-progress download.

    Args:
        model_name: HuggingFace repo id or cached model directory name.
        allow_huggingface_download: If True, download from HuggingFace Hub
            when the model is not already cached locally.

    Returns:
        Local path to the model directory.

    Raises:
        ValueError: If the model isn't cached locally and downloads are disabled,
            or if the downloaded model directory is incomplete.
    """
    local_model_path = _get_cache_dir() / "resources" / "models" / model_name

    with _model_resolution_lock(model_name):
        if local_model_path.exists() and local_model_path.is_dir():
            if _validate_model_directory(local_model_path):
                logger.info(f"Found model locally: {local_model_path}")
                return str(local_model_path)
            logger.warning(
                f"Cached model directory {local_model_path} is incomplete "
                f"(missing weight files or config.json). Re-downloading..."
            )
            shutil.rmtree(local_model_path)

        if not allow_huggingface_download:
            raise ValueError(
                f"Model '{model_name}' not found locally at {local_model_path} and "
                f"HuggingFace Hub downloads are disabled (allow_huggingface_download=False)."
            )

        logger.info(f"Downloading model '{model_name}' from HuggingFace Hub...")
        from huggingface_hub import snapshot_download

        local_model_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model_name, local_dir=str(local_model_path))
        if not _validate_model_directory(local_model_path):
            raise ValueError(
                f"Downloaded model '{model_name}' from HuggingFace Hub is incomplete at "
                f"{local_model_path}. Missing weight files (model.safetensors or "
                f"pytorch_model.bin) or config.json. Check your network connection or "
                f"try: huggingface-cli login"
            )
        logger.info(f"Successfully downloaded model to: {local_model_path}")
        return str(local_model_path)


class TransformerCore:
    """
    Core transformer inference engine used by the Presidio recognizer.

    This class handles:
    - Model configuration loading
    - Model path resolution (local cache or HuggingFace Hub)
    - Pipeline loading with device placement options
    - Raw inference (returns BIO tokens)
    - BIO token aggregation into entity spans

    Thread Safety:
        Pipeline loading is protected by a lock for thread-safe lazy loading.
        Inference is thread-safe once the pipeline is loaded.

    Args:
        model_name: Name of the model configuration to load
        model_path: Optional explicit path to model (overrides cache/Hub resolution)
        device: Device placement strategy:
            - "auto": Use accelerate's device_map="auto" (recommended for single-text)
            - "cuda:N": Explicit GPU placement (recommended for batch inference)
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
            from HuggingFace Hub when the model is not in the local cache.

    Example:
        # For Presidio (lazy loading, auto device)
        core = TransformerCore(model_name="stanford_deidentifier", device="auto")

        # For batch inference (immediate loading, explicit GPU)
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
            device: Device string (``"cpu"``, ``"cuda"``, or ``"auto"``).
            dtype: Torch dtype for model weights.
            load_immediately: If True, load the pipeline during init.
            local_files_only: Restrict HuggingFace to local files only.
            compile_model: Whether to use a compiled model cache.
            compile_cache_path: Path to the compiled ``.bin`` cache file.
            allow_huggingface_download: If True, fall back to HuggingFace Hub
                when the model is not in the local cache.
        """
        # Treat local_files_only=True as an offline / no-network mode. It already
        # stops transformers.from_pretrained from reaching the Hub, but the
        # name-only branch below calls _resolve_model_path, which would still attempt
        # a HuggingFace snapshot_download when allow_huggingface_download is set.
        # Disable that here so both code paths honor the offline contract and every
        # caller gets coherent behavior.
        if local_files_only and allow_huggingface_download:
            logger.warning(
                "local_files_only=True implies offline mode; disabling "
                "allow_huggingface_download=True so no HuggingFace Hub download is "
                "attempted. Set local_files_only=False to permit downloads."
            )
            allow_huggingface_download = False

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
            # An absolute path is unambiguously a local model dir (HF repo ids are
            # never absolute). If it's missing on this node, fail fast with an
            # actionable message instead of transformers' misleading "Repo id must be
            # in the form..." error. Bare repo ids (e.g.
            # "StanfordAIMI/stanford-deidentifier-v2") are left alone so name-only
            # loading from the HF cache keeps working.
            if Path(model_path).is_absolute() and not Path(model_path).is_dir():
                raise ValueError(
                    f"model_path {model_path!r} is an absolute path but does not exist or "
                    f"is not a directory on this node ({socket.gethostname()}). Ensure the "
                    f"model volume is mounted, or pass a HuggingFace repo id (e.g. "
                    f"'StanfordAIMI/stanford-deidentifier-v2') to load from the local HF "
                    f"cache instead."
                )
            self.model_path = model_path
        else:
            self.model_path = _resolve_model_path(
                model_name=model_name,
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

    def infer_raw_direct(self, texts: list[str], batch_size: int | None = None) -> list[list[dict]]:
        """Run inference bypassing the HF pipeline dispatch loop.

        Tokenizes the entire batch in one call, runs a single GPU forward pass
        (or sub-batched forward passes if batch_size < len(texts)), and extracts
        raw token predictions using offset_mapping. This avoids the per-text
        preprocess/postprocess Python loops in HuggingFace's ChunkPipeline.

        Output format: list of lists of dicts with keys
        {entity, score, start, end, word, index}, one list per input text.

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
            all_results.extend(self._forward_batch_direct(sub_texts))

        return all_results

    def _tokenize_batch(self, texts: list[str]) -> dict[str, Any]:
        """Tokenize one sub-batch."""
        tokenizer = self._tokenizer
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

    def _forward_batch_direct(self, texts: list[str]) -> list[list[dict]]:
        """Single batch: tokenize -> GPU forward -> extract predictions."""
        encoded = self._tokenize_batch(texts)
        return self._forward_and_postprocess(texts, encoded)

    def _forward_and_postprocess(self, texts: list[str], encoded: dict[str, Any]) -> list[list[dict]]:
        """GPU forward pass + prediction extraction for an already-tokenized sub-batch."""
        model = self._model
        device = next(model.parameters()).device

        encoded = dict(encoded)  # don't mutate the caller's dict via pop() below
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
        special_np = special_tokens_mask.numpy().astype(bool)

        # Vectorized label lookup + masking. The previous implementation walked
        # every (text, padded-token) pair in a pure-Python double loop - dead time
        # that doesn't touch the GPU and scales with padding rather than actual
        # content, which was enough to visibly stall GPU utilization on real
        # batches. Filtering with numpy first means the remaining Python loop only
        # runs once per surviving (non-special, non-ignored) prediction.
        id2label = self._id2label
        label_names = np.array([id2label[i] for i in range(len(id2label))])
        labels_np = label_names[label_ids_np]

        ignore = self._ignore_labels_set
        ignore_mask = np.isin(labels_np, list(ignore)) if ignore else np.zeros_like(labels_np, dtype=bool)
        valid_mask = ~special_np & ~ignore_mask

        # np.nonzero on a 2D mask returns (row, col) pairs in row-major order,
        # i.e. the same (text, token) order the original nested loop produced.
        batch_idx, seq_idx = np.nonzero(valid_mask)
        results: list[list[dict]] = [[] for _ in texts]
        for b, j in zip(batch_idx.tolist(), seq_idx.tolist(), strict=True):
            text = texts[b]
            s, e = int(offset_np[b, j, 0]), int(offset_np[b, j, 1])
            results[b].append(
                {
                    "entity": str(labels_np[b, j]),
                    "score": float(scores_np[b, j]),
                    "start": s,
                    "end": e,
                    "word": text[s:e],
                    "index": j,
                }
            )

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

    @property
    def model_max_length(self) -> int:
        """Maximum input length for the tokenizer."""
        pipeline_instance = self._ensure_pipeline_loaded()
        return getattr(pipeline_instance.tokenizer, "model_max_length", 512)

    def get_device_info(self) -> str:
        """Get current device information."""
        pipeline_instance = self._pipeline
        if pipeline_instance is None:
            return "not loaded"

        try:
            model = pipeline_instance.model
            device = next(model.parameters()).device
            if device.type == "cuda":
                device_name = torch.cuda.get_device_name(device.index)
                return f"{device} ({device_name})"
            return str(device)
        except Exception:
            return "unknown"
