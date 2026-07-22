"""
Presidio EntityRecognizer wrapper for transformer-based NER.

This module provides TransformersRecognizer, a Presidio-compatible recognizer
that uses transformer models for Named Entity Recognition.
"""

import copy
import logging
import time
from typing import Any

from presidio_analyzer import AnalysisExplanation
from presidio_analyzer import EntityRecognizer
from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from tide2.transformers import TransformerCore
from tide2.transformers import format_transformer_recognizer_name
from tide2.transformers import get_available_models
from tide2.transformers import load_model_config
from tide2.utils.text_processing import split_text_to_word_chunks

logger = logging.getLogger(__name__)


class TransformersRecognizer(EntityRecognizer):
    """
    Thread-safe wrapper for a transformers model for use within Presidio Analyzer.

    The class loads models hosted on HuggingFace and loads the model and tokenizer
    into a TokenClassification pipeline. The pipeline is lazily loaded on first use
    and shared across all threads with proper locking to ensure thread safety.

    Samples are split into short text chunks, ideally shorter than max_length input_ids
    of the individual model, to avoid truncation by the Tokenizer and loss of information.

    Configuration is automatically loaded from the bert_transformer_configuration.json
    resource file based on the model_name provided during initialization.

    Thread Safety: Pipeline initialization is locked, but inference runs in parallel.
    PyTorch models in eval mode are thread-safe for concurrent inference.

    Batch Processing: For large-scale processing, use the Ray-based TransformerInferenceActor
    which provides efficient batch inference with GPU support.

    Args:
        model_name: Name of the model configuration to load from
            ``bert_transformer_configuration.json``.
        model_path: Custom path to the model directory, overrides
            the default from configuration.
        bucket_name: GCS bucket name for model loading.
        project_id: GCP project ID for model loading.

    Example::

        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from tide2.recognizers.transformers_recognizer import TransformersRecognizer

        transformers_recognizer = TransformersRecognizer(model_name="STANFORD_CONFIGURATION")

        registry = RecognizerRegistry()
        registry.add_recognizer(transformers_recognizer)
        analyzer = AnalyzerEngine(registry=registry)
        results = analyzer.analyze("My name is Christopher.", language="en")
    """

    def load(self) -> None:
        """Load method required by EntityRecognizer interface."""
        pass

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of available model configurations.

        Returns:
            List of available model names that can be used with TransformersRecognizer
        """
        return get_available_models()

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        bucket_name: str | None = None,
        project_id: str | None = None,
        allow_huggingface_download: bool = True,
    ):
        """Initialize the recognizer with a named transformer model.

        Args:
            model_name: Key in ``bert_transformer_configuration.json``.
            model_path: Local path override for the model directory.
            bucket_name: GCS bucket for model download (if model_path is None).
            project_id: GCP project for GCS access.
            allow_huggingface_download: If True, fall back to HuggingFace Hub
                when local cache and GCS both miss.
        """
        # Load configuration from the resource file
        config = load_model_config(model_name)

        # Extract supported entities from config
        supported_entities = config.get("PRESIDIO_SUPPORTED_ENTITIES", [])

        super().__init__(
            supported_entities=supported_entities,
            name=format_transformer_recognizer_name(model_name),
        )

        self.model_name = model_name

        # Create core inference engine (lazy loading with auto device placement)
        self._core = TransformerCore(
            model_name=model_name,
            model_path=model_path,
            bucket_name=bucket_name,
            project_id=project_id,
            device="auto",  # Use accelerate's device_map for automatic placement
            load_immediately=False,  # Lazy load on first use
            allow_huggingface_download=allow_huggingface_download,
        )

        # Store model path for backwards compatibility
        self.model_path = self._core.model_path

        # Load Presidio-specific configuration parameters
        self.ignore_labels: list[str] = config.get("LABELS_TO_IGNORE", ["O"])
        self.model_to_presidio_mapping: dict[str, str] = config.get("MODEL_TO_PRESIDIO_MAPPING", {})
        self.default_explanation: str = config.get("DEFAULT_EXPLANATION", "N/A")
        self.text_overlap_length: int = config.get("CHUNK_OVERLAP_SIZE", 40)
        self.chunk_length: int = config.get("CHUNK_SIZE", 600)
        self.id_entity_name: str = config.get("ID_ENTITY_NAME", "ID")
        self.id_score_reduction: float = config.get("ID_SCORE_MULTIPLIER", 0.5)

    @property
    def pipeline(self) -> Any:
        """Get the pipeline, loading it if not already loaded."""
        return self._core.pipeline

    @property
    def is_loaded(self) -> bool:
        """Check if the pipeline is loaded."""
        return self._core.is_loaded

    @is_loaded.setter
    def is_loaded(self, value: bool) -> None:
        """Setter for is_loaded to maintain compatibility with parent class."""
        pass  # We manage loading state via _core, so this is a no-op

    def get_supported_entities(self) -> list[str]:
        """Return supported entities by this model."""
        return self.supported_entities

    def analyze(
        self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None
    ) -> list[RecognizerResult]:
        """
        Analyze text using transformers model to produce NER tagging.

        Args:
            text: The text for analysis.
            entities: The list of entities this recognizer is able to detect.
            nlp_artifacts: Not used by this recognizer.

        Returns:
            List of Presidio RecognizerResult constructed from the recognized
            transformers detections.
        """
        import threading

        start_time = time.time()
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] TransformersRecognizer starting analysis")

        results = []

        # Run model inference with chunking
        ner_start = time.time()
        ner_results = self._get_ner_results_for_text(text)
        ner_time = time.time() - ner_start
        logger.info(f"[{thread_name}] NER inference took {ner_time:.3f}s")

        # Post-process results: map entities and convert to RecognizerResult
        for res in ner_results:
            res["entity_group"] = self._check_label_transformer(res["entity_group"])
            if not res["entity_group"] or res["entity_group"] not in entities:
                continue

            if res["entity_group"] == self.id_entity_name:
                logger.debug(f"ID entity found, multiplying score by {self.id_score_reduction}")
                res["score"] = res["score"] * self.id_score_reduction

            textual_explanation = self.default_explanation.format(res["entity_group"])
            explanation = self.build_transformers_explanation(
                float(round(res["score"], 2)), textual_explanation, res["word"]
            )
            transformers_result = self._convert_to_recognizer_result(res, explanation)
            results.append(transformers_result)

        elapsed_time = time.time() - start_time
        logger.info(
            f"[{thread_name}] TransformersRecognizer completed analysis in {elapsed_time:.3f}s, found {len(results)} entities"
        )

        return results

    def _get_ner_results_for_text(self, text: str) -> list[dict]:
        """Run model inference on the provided text with chunking support.

        The text is split into chunks with n overlapping characters.
        The results are then aggregated and duplications are removed.

        Args:
            text: The text to run inference on.

        Returns:
            List of entity predictions on the word level.
        """

        # Get model max length for chunking decisions
        model_max_length = self._core.model_max_length

        # Estimate token count for BERT-based tokenizers (roughly 4 chars per token)
        estimated_tokens = int(len(text) / 4)
        text_length = estimated_tokens

        # Process text in chunks if needed
        if text_length <= model_max_length:
            inference_start = time.time()
            # Use core for single text, then aggregate
            raw_preds = self._core.infer_single_raw(text)
            inference_time = time.time() - inference_start
            logger.debug(f"Model inference (single chunk) took {inference_time:.3f}s")

            # Remove duplicates
            predictions = [dict(t) for t in {tuple(d.items()) for d in raw_preds}]

            # Use core's aggregation (imported from text_processing)
            from tide2.utils.text_processing import aggregate_bio_tokens

            return aggregate_bio_tokens(predictions, text)
        logger.info(f"Splitting text into chunks, length {text_length} > {model_max_length}")
        predictions = []
        chunk_indexes = split_text_to_word_chunks(text_length, self.chunk_length, self.text_overlap_length)

        # Iterate over text chunks and run inference
        total_inference_time = 0.0
        for idx, (chunk_start, chunk_end) in enumerate(chunk_indexes):
            chunk_text = text[chunk_start:chunk_end]

            chunk_inference_start = time.time()
            chunk_preds = self._core.infer_single_raw(chunk_text)
            chunk_inference_time = time.time() - chunk_inference_start
            total_inference_time += chunk_inference_time
            logger.debug(f"Chunk {idx + 1}/{len(chunk_indexes)} inference took {chunk_inference_time:.3f}s")

            # Align indexes to match the original text
            aligned_predictions = []
            for prediction in chunk_preds:
                prediction_tmp = copy.deepcopy(prediction)
                prediction_tmp["start"] += chunk_start
                prediction_tmp["end"] += chunk_start
                aligned_predictions.append(prediction_tmp)

            predictions.extend(aligned_predictions)

        logger.info(f"Model inference (chunked, {len(chunk_indexes)} chunks) took {total_inference_time:.3f}s")

        # Remove duplicates
        predictions = [dict(t) for t in {tuple(d.items()) for d in predictions}]

        # Apply BIO token aggregation
        from tide2.utils.text_processing import aggregate_bio_tokens

        return aggregate_bio_tokens(predictions, text)

    @staticmethod
    def _convert_to_recognizer_result(prediction_result: dict, explanation: AnalysisExplanation) -> RecognizerResult:
        """Convert NER model predictions into a RecognizerResult format.

        Args:
            prediction_result: A single entity prediction dict.
            explanation: Textual representation of model prediction.

        Returns:
            A RecognizerResult for evaluation.
        """
        return RecognizerResult(
            entity_type=prediction_result["entity_group"],
            start=prediction_result["start"],
            end=prediction_result["end"],
            score=float(round(prediction_result["score"], 2)),
            analysis_explanation=explanation,
        )

    def build_transformers_explanation(
        self, original_score: float, explanation: str, pattern: str
    ) -> AnalysisExplanation:
        """Create explanation for why this result was detected.

        Args:
            original_score: Score given by this recognizer.
            explanation: Explanation string.
            pattern: Pattern used for detection.

        Returns:
            Structured explanation and scores of a NER model prediction.
        """
        return AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=float(original_score),
            textual_explanation=explanation,
            pattern=pattern,
        )

    def _check_label_transformer(self, label: str) -> str | None:
        """Validate the predicted label is identified by Presidio and map to Presidio representation.

        Args:
            label: Predicted label by the model.

        Returns:
            The adjusted entity name, or None if the label should be ignored.
        """
        # Convert model label to presidio label
        entity = self.model_to_presidio_mapping.get(label, None)

        if entity in self.ignore_labels:
            return None

        if entity is None:
            logger.warning(f"Found unrecognized label {label}, returning entity as is")
            return label

        if entity not in self.supported_entities:
            logger.warning(f"Found entity {entity} which is not supported by Presidio")
            return entity

        return entity
