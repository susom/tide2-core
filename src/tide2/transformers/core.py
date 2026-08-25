"""
Core transformer inference engine.

This module provides the TransformerCore class that encapsulates the shared logic
for transformer-based NER inference, used by both the Presidio recognizer and
the Ray actor.
"""

import logging
import socket
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import pipeline

from tide2.utils.gcs_resource_manager import resolve_model_path
from tide2.utils.text_processing import aggregate_bio_tokens

from .config import load_model_config

logger = logging.getLogger(__name__)

# Fixed schema of a raw BIO token prediction (see infer_single_raw /
# forward_windows). Used to build a stable dedupe key that does not depend on dict
# insertion order.
_RAW_PRED_KEYS = ("entity", "score", "start", "end", "word", "index")


def _dedupe_raw_predictions(raw_predictions: list[dict]) -> list[dict]:
    """Remove duplicate raw BIO predictions (can occur from chunking).

    The dedupe key is built from the fixed prediction schema in a stable order
    so it does not depend on dict insertion order (O(k) per dict, no per-key
    sort).
    """
    return [
        dict(zip(_RAW_PRED_KEYS, key, strict=True))
        for key in {tuple(d[k] for k in _RAW_PRED_KEYS) for d in raw_predictions}
    ]


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
            allow_huggingface_download: If True, fall back to HuggingFace Hub
                when local cache and GCS both miss.
        """
        # Treat local_files_only=True as an offline / no-network mode. It already
        # stops transformers.from_pretrained from reaching the Hub, but the
        # name-only branch below calls resolve_model_path, which would still attempt
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

        # Load configuration
        self._config = load_model_config(model_name)

        # Per-model dtype override from the config, e.g. {"DTYPE": "float32"}. Some
        # architectures are not half-precision-safe: DeBERTa-v1's disentangled
        # attention raises "expected scalar type Float but found Half/BFloat16" under
        # fp16/bf16, so its config pins float32. Absent key => keep the constructor
        # dtype (float16 default).
        _cfg_dtype = self._config.get("DTYPE")
        if _cfg_dtype:
            self.dtype = getattr(torch, _cfg_dtype)

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
                    f"is not a directory on this node ({socket.gethostname()}). Under Ray, "
                    f"ensure the model volume is mounted on all worker nodes, or pass a "
                    f"HuggingFace repo id (e.g. 'StanfordAIMI/stanford-deidentifier-v2') "
                    f"to load from the local HF cache instead."
                )
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

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=self.local_files_only)

        # Per-model max sequence length from the config, e.g. {"MODEL_MAX_LENGTH": 512}.
        # Some tokenizers ship a sentinel model_max_length (e.g. XLM-R/SentencePiece's
        # 1e30) that exceeds the model's real position-embedding limit, so tide2's
        # chunking/truncation never bounds sequences and the model raises
        # "expanded size of the tensor (N) must match the existing size (512)".
        # Pinning it here makes truncation (below) and the model_max_length property
        # honor the model's true limit. Absent key => keep the tokenizer's value.
        _cfg_mml = self._config.get("MODEL_MAX_LENGTH")
        if _cfg_mml:
            tokenizer.model_max_length = int(_cfg_mml)

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

        # Store direct references for windowed inference (bypasses HF pipeline dispatch)
        self._model = model
        self._tokenizer = tokenizer
        self._id2label = model.config.id2label
        self._ignore_labels_set = set(self.ignore_labels)

        # Log device info
        model_device = next(model.parameters()).device
        logger.info(f"[{thread_name}] Pipeline loaded on device: {model_device}")

    def tokenize_ragged(self, texts: list[str]) -> Any:
        """Tokenize a batch **once**, with no truncation, no padding, and offsets.

        This is the single tokenization per input batch. It returns the *content*
        tokens only (``add_special_tokens=False``) so the caller can window in
        token space against the model's real budget (``model_max_length`` minus
        :attr:`num_special_tokens`) and add the special tokens per window at
        forward time (:meth:`forward_windows`). Because tokenization happens here
        exactly once, windowing and OOM retries reuse this output and never
        re-tokenize.

        Args:
            texts: List of text strings to tokenize, in the caller's order.

        Returns:
            A ``BatchEncoding`` whose ``input_ids`` and ``offset_mapping`` are
            **ragged** Python lists — one list per input text (no tensors, since
            padding is off). ``input_ids[i]`` are the content token ids of
            ``texts[i]`` and ``offset_mapping[i]`` the matching ``(start, end)``
            character spans into ``texts[i]``. Requires a fast tokenizer (for
            ``offset_mapping``); tide2's models all ship one.
        """
        self._ensure_pipeline_loaded()
        return self._tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_offsets_mapping=True,
        )

    def forward_windows(self, windows: list[tuple[list[int], list[tuple[int, int]], str]]) -> list[list[dict]]:
        """GPU forward + prediction extraction for pre-tokenized token windows.

        Each window is ``(content_ids, offsets, text)`` where ``content_ids`` are
        the token ids of one window **without** special tokens and ``offsets`` are
        the matching ``(start, end)`` character spans into ``text`` — both taken
        directly from a single :meth:`tokenize_ragged` call, so they are already
        chunk-relative and never re-based. This method adds the model's special
        tokens, right-pads every window to the batch's longest member, runs one
        forward pass, and extracts raw BIO predictions.

        Releases every CUDA-resident tensor in a ``finally`` (Fix A1) so that on
        OOM no exception traceback can pin the failed forward's VRAM — this is
        what lets the caller's ``empty_cache()`` reclaim before retrying at a
        smaller window-batch. Because the windows are supplied pre-tokenized, that
        retry never re-tokenizes. See
        :meth:`tide2.actors.transformer.TransformerInferenceActor._forward_windows`.

        Args:
            windows: List of ``(content_ids, offsets, text)`` tuples, one per
                window, in the order the caller will consume predictions.

        Returns:
            One prediction list per window, aligned to ``windows``. Each dict has
            keys ``{entity, score, start, end, word, index}``; ``index`` is the
            token's position in the padded, special-token-bearing sequence, the
            same convention as the rest of the pipeline.
        """
        if not windows:
            return []

        model = self._model
        device = next(model.parameters()).device
        input_ids, attention_mask, special_np, offset_np, texts = self._build_window_batch(windows)

        # Track every CUDA-resident tensor so ``finally`` can free it all, even if
        # ``.to(device)`` or the forward raises CUDA OOM.
        input_ids_gpu = attention_mask_gpu = logits = probs = scores_max = label_ids = None
        try:
            input_ids_gpu = input_ids.to(device)
            attention_mask_gpu = attention_mask.to(device)

            with torch.inference_mode():
                logits = model(input_ids=input_ids_gpu, attention_mask=attention_mask_gpu).logits

            probs = torch.softmax(logits, dim=-1)
            scores_max, label_ids = probs.max(dim=-1)  # (batch, seq_len)

            scores_np = scores_max.cpu().numpy()
            label_ids_np = label_ids.cpu().numpy()
        finally:
            # Drop this frame's references to every GPU tensor. Without this an
            # exception traceback would keep them alive and empty_cache() could not
            # reclaim; the CPU numpy copies above are already detached from CUDA.
            del input_ids_gpu, attention_mask_gpu, logits, probs, scores_max, label_ids

        return self._extract_window_predictions(scores_np, label_ids_np, special_np, offset_np, texts)

    def _special_token_affixes(self) -> tuple[list[int], list[int]]:
        """The special token ids the tokenizer wraps a single sequence with.

        Returns ``(prefix_ids, suffix_ids)`` — e.g. ``([CLS], [SEP])`` for BERT or
        ``([<s>], [</s>])`` for RoBERTa. Derived once by diffing a probe encoding
        with and without special tokens, so it is architecture- and version-robust
        (transformers 5.x dropped ``build_inputs_with_special_tokens`` /
        ``prepare_for_model`` and only implements ``get_special_tokens_mask`` for
        already-formatted sequences). Cached on the instance.
        """
        cached = getattr(self, "_special_affixes", None)
        if cached is not None:
            return cached

        tokenizer = self._tokenizer
        probe = "hello"
        with_sp = tokenizer(probe, add_special_tokens=True)["input_ids"]
        without = tokenizer(probe, add_special_tokens=False)["input_ids"]
        prefix: list[int] = []
        suffix: list[int] = []
        span = len(without)
        for i in range(len(with_sp) - span + 1):
            if with_sp[i : i + span] == without:
                prefix = list(with_sp[:i])
                suffix = list(with_sp[i + span :])
                break

        self._special_affixes = (prefix, suffix)
        return prefix, suffix

    def _build_window_batch(
        self, windows: list[tuple[list[int], list[tuple[int, int]], str]]
    ) -> tuple[Any, Any, Any, Any, list[str]]:
        """Add special tokens and right-pad windows into a single forward batch.

        Returns ``(input_ids, attention_mask, special_np, offset_np, texts)``:
        two CPU ``torch`` tensors for the model plus numpy ``special``/``offset``
        arrays aligned to the padded sequence for extraction. Padding positions are
        marked special (so they are skipped) and carry a ``(0, 0)`` offset.
        """
        tokenizer = self._tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        prefix, suffix = self._special_token_affixes()
        n_pre, n_suf = len(prefix), len(suffix)
        pre_offsets = [(0, 0)] * n_pre
        suf_offsets = [(0, 0)] * n_suf
        pre_special = [1] * n_pre
        suf_special = [1] * n_suf

        n = len(windows)
        full_ids_list: list[list[int]] = []
        special_list: list[list[int]] = []
        offset_list: list[list[tuple[int, int]]] = []
        texts: list[str] = []
        for content_ids, offsets, text in windows:
            ids = list(content_ids)
            # Wrap the content tokens with the model's special-token affixes and
            # align the special mask + offsets: special positions carry a (0, 0)
            # placeholder and are skipped at extraction; content positions keep the
            # window's char offsets straight from the single tokenization.
            full_ids_list.append(prefix + ids + suffix)
            special_list.append(pre_special + [0] * len(ids) + suf_special)
            offset_list.append(pre_offsets + [tuple(o) for o in offsets] + suf_offsets)
            texts.append(text)

        max_len = max(len(ids) for ids in full_ids_list)

        input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        special_np = np.ones((n, max_len), dtype=bool)  # padding => special => skipped
        offset_np = np.zeros((n, max_len, 2), dtype=np.int64)
        for i, ids in enumerate(full_ids_list):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1
            special_np[i, :length] = np.asarray(special_list[i], dtype=bool)
            offset_np[i, :length] = np.asarray(offset_list[i], dtype=np.int64)

        return input_ids, attention_mask, special_np, offset_np, texts

    def _extract_window_predictions(
        self, scores_np: Any, label_ids_np: Any, special_np: Any, offset_np: Any, texts: list[str]
    ) -> list[list[dict]]:
        """Turn per-position scores/labels into raw BIO predictions per window.

        Skips special/padding positions and ignored labels; ``index`` is the
        token's position in the padded, special-token-bearing sequence.
        """
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
        raw_predictions = _dedupe_raw_predictions(raw_predictions)

        # Aggregate BIO tokens
        return aggregate_bio_tokens(raw_predictions, text)

    @property
    def model_max_length(self) -> int:
        """Maximum input length for the tokenizer."""
        pipeline_instance = self._ensure_pipeline_loaded()
        return getattr(pipeline_instance.tokenizer, "model_max_length", 512)

    @property
    def num_special_tokens(self) -> int:
        """Special tokens the tokenizer adds around a single sequence.

        For a single sequence this is typically 2 (e.g. BERT's ``[CLS]``/``[SEP]``
        or RoBERTa's ``<s>``/``</s>``). Callers subtract it from
        :attr:`model_max_length` to get the per-window content-token budget used
        for token-space windowing.
        """
        self._ensure_pipeline_loaded()
        return int(self._tokenizer.num_special_tokens_to_add(pair=False))

    def get_device_info(self) -> str:
        """Get current device information."""
        if not self.is_loaded:
            return "not loaded"

        try:
            model = self.pipeline.model
            device = next(model.parameters()).device
            if device.type == "cuda":
                device_name = torch.cuda.get_device_name(device.index)
                return f"{device} ({device_name})"
            return str(device)
        except Exception:
            return "unknown"
