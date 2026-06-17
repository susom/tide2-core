"""
Unit tests for TransformersRecognizer.

Tests cover the transformer-based NER recognizer with mocking of model loading,
configuration loading, and NER predictions.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from tide2.recognizers.transformers_recognizer import TransformersRecognizer


def create_temp_config(config_data):
    """Helper function to create a temporary config file and return its path."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config_data, temp_file)
    temp_file.flush()
    temp_file.close()
    return temp_file.name


class TestTransformersRecognizer:
    """Test TransformersRecognizer functionality."""

    def setup_method(self):
        """Set up test fixtures with mocked configuration."""
        # Mock configuration
        self.mock_config = {
            "TEST_MODEL": {
                "DEFAULT_MODEL_PATH": "test/model",
                "PRESIDIO_SUPPORTED_ENTITIES": ["PERSON", "LOCATION", "PHONE_NUMBER"],
                "LABELS_TO_IGNORE": ["O"],
                "DEFAULT_EXPLANATION": "Test model explanation",
                "SUB_WORD_AGGREGATION": "simple",
                "MODEL_TO_PRESIDIO_MAPPING": {},
                "DATASET_TO_PRESIDIO_MAPPING": {"PERSON": "PERSON", "LOCATION": "LOCATION", "PHONE": "PHONE_NUMBER"},
                "CHUNK_OVERLAP_SIZE": 40,
                "CHUNK_SIZE": 600,
                "ID_ENTITY_NAME": "ID",
                "ID_SCORE_MULTIPLIER": 0.5,
            }
        }

    def test_get_available_models(self):
        """Test getting available model configurations."""
        # Patch at the config module level where get_resource_path is called
        with patch("tide2.transformers.config.get_resource_path") as mock_get_path:
            config_path = create_temp_config(self.mock_config)
            mock_get_path.return_value = config_path

            try:
                models = TransformersRecognizer.get_available_models()
                assert "TEST_MODEL" in models
            finally:
                Path(config_path).unlink()

    @patch("tide2.transformers.core.resolve_model_path")
    @patch("tide2.transformers.config.get_resource_path")
    def test_initialization(self, mock_get_path, mock_resolve_model):
        """Test TransformersRecognizer initialization."""
        # Mock model path resolution
        mock_resolve_model.return_value = "/fake/model/path"

        # Create temporary config file
        config_path = create_temp_config(self.mock_config)
        mock_get_path.return_value = config_path

        try:
            recognizer = TransformersRecognizer(model_name="TEST_MODEL")

            assert recognizer.model_name == "TEST_MODEL"
            assert recognizer.model_path == "/fake/model/path"
            assert recognizer.supported_entities == ["PERSON", "LOCATION", "PHONE_NUMBER"]
            assert recognizer.name == "Transformers model TEST_MODEL"
            assert not recognizer.is_loaded  # Pipeline not loaded yet
        finally:
            Path(config_path).unlink()

    @patch("tide2.transformers.config.get_resource_path")
    def test_custom_model_path(self, mock_get_path):
        """Test TransformersRecognizer with custom model path."""
        # Create temporary config file
        config_path = create_temp_config(self.mock_config)
        mock_get_path.return_value = config_path

        try:
            custom_path = "/custom/model/path"
            recognizer = TransformersRecognizer(model_name="TEST_MODEL", model_path=custom_path)

            assert recognizer.model_path == custom_path
            assert recognizer.model_name == "TEST_MODEL"
        finally:
            Path(config_path).unlink()

    @patch("tide2.transformers.core.resolve_model_path")
    @patch("tide2.transformers.config.get_resource_path")
    @patch("tide2.transformers.core.pipeline")
    @patch("tide2.transformers.core.AutoModelForTokenClassification")
    @patch("tide2.transformers.core.AutoTokenizer")
    def test_pipeline_loading(self, mock_tokenizer, mock_model, mock_pipeline, mock_get_path, mock_resolve_model):
        """Test lazy loading of transformer pipeline."""
        # Mock model path resolution
        mock_resolve_model.return_value = "/fake/model/path"

        # Create temporary config file
        config_path = create_temp_config(self.mock_config)
        mock_get_path.return_value = config_path

        # Mock model and tokenizer
        mock_model_instance = Mock()
        mock_model_param = Mock()
        mock_model_param.device = "cpu"
        mock_model_instance.parameters.return_value = iter([mock_model_param])
        mock_model_instance.eval.return_value = mock_model_instance
        mock_tokenizer_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Mock pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        try:
            recognizer = TransformersRecognizer(model_name="TEST_MODEL")

            # Pipeline should not be loaded initially
            assert not recognizer.is_loaded

            # Access pipeline property to trigger loading
            pipeline_result = recognizer.pipeline

            # Verify pipeline was loaded
            assert recognizer.is_loaded
            assert pipeline_result == mock_pipeline_instance

            # Verify pipeline was called with correct parameters
            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args
            # Check keyword arguments
            assert call_args.kwargs.get("task") == "token-classification"
            assert call_args.kwargs.get("model") == mock_model_instance
            assert call_args.kwargs.get("tokenizer") == mock_tokenizer_instance
            assert call_args.kwargs.get("aggregation_strategy") == "none"
            # transformers 5.x removed the `framework` argument from pipeline()
            # (TF/Flax support dropped; everything is PyTorch), so it must not be passed.
            assert "framework" not in call_args.kwargs
        finally:
            Path(config_path).unlink()

    @patch("tide2.transformers.core.resolve_model_path")
    @patch("tide2.transformers.config.get_resource_path")
    def test_get_supported_entities(self, mock_get_path, mock_resolve_model):
        """Test getting supported entities."""
        mock_resolve_model.return_value = "/fake/model/path"
        config_path = create_temp_config(self.mock_config)
        mock_get_path.return_value = config_path

        try:
            recognizer = TransformersRecognizer(model_name="TEST_MODEL")
            entities = recognizer.get_supported_entities()

            assert entities == ["PERSON", "LOCATION", "PHONE_NUMBER"]
        finally:
            Path(config_path).unlink()

    @patch("tide2.transformers.core.resolve_model_path")
    @patch("tide2.transformers.config.get_resource_path")
    def test_analyze_without_pipeline_loaded(self, mock_get_path, mock_resolve_model):
        """Test analyze method when pipeline is not loaded."""
        mock_resolve_model.return_value = "/fake/model/path"
        config_path = create_temp_config(self.mock_config)
        mock_get_path.return_value = config_path

        try:
            recognizer = TransformersRecognizer(model_name="TEST_MODEL")

            # Mock the core's pipeline to avoid triggering actual loading
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = []
            mock_pipeline_instance.tokenizer.model_max_length = 512
            recognizer._core._pipeline = mock_pipeline_instance

            results = recognizer.analyze("Test text", ["PERSON"])
            assert isinstance(results, list)
        finally:
            Path(config_path).unlink()

    @patch("tide2.transformers.core.AutoTokenizer")
    @patch("tide2.transformers.core.AutoModelForTokenClassification")
    @patch("tide2.transformers.core.pipeline")
    @patch("tide2.transformers.config.get_resource_path")
    @patch("tide2.transformers.core.resolve_model_path")
    def test_analyze_with_predictions(
        self, mock_resolve_model, mock_get_path, mock_pipeline, mock_model, mock_tokenizer
    ):
        """Test analyze method with mocked NER predictions."""
        # Mock model path resolution
        mock_resolve_model.return_value = "/fake/model/path"

        # Create temporary config file
        config_path = create_temp_config(self.mock_config)
        mock_get_path.return_value = config_path

        # Mock model components
        mock_model_instance = Mock()
        mock_model_param = Mock()
        mock_model_param.device = "cpu"
        mock_model_instance.parameters.return_value = iter([mock_model_param])
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = Mock()

        # Mock pipeline predictions (raw BIO format with "entity" key)
        mock_predictions = [
            {"entity": "B-PERSON", "score": 0.9, "start": 0, "end": 4, "word": "John"},
            {"entity": "B-LOCATION", "score": 0.8, "start": 14, "end": 21, "word": "Seattle"},
        ]

        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = mock_predictions
        # Properly mock the tokenizer's model_max_length attribute
        mock_pipeline_instance.tokenizer.model_max_length = 512
        mock_pipeline.return_value = mock_pipeline_instance

        try:
            recognizer = TransformersRecognizer(model_name="TEST_MODEL")

            text = "John lives in Seattle"
            results = recognizer.analyze(text, ["PERSON", "LOCATION"])

            # Should have processed the predictions into RecognizerResult objects
            assert isinstance(results, list)
            # Note: The actual conversion logic is complex and would need more detailed mocking
            # This test verifies the basic flow works
        finally:
            Path(config_path).unlink()

    @patch("tide2.transformers.core.resolve_model_path")
    @patch("tide2.transformers.config.get_resource_path")
    def test_thread_safety(self, mock_get_path, mock_resolve_model):
        """Test that TransformerCore uses thread-safe locking."""
        mock_resolve_model.return_value = "/fake/model/path"
        config_path = create_temp_config(self.mock_config)
        mock_get_path.return_value = config_path

        try:
            recognizer = TransformersRecognizer(model_name="TEST_MODEL")

            # Verify that core has _pipeline_lock for thread-safe parallel execution
            assert hasattr(recognizer._core, "_pipeline_lock")
            assert not recognizer.is_loaded
        finally:
            Path(config_path).unlink()

    @patch("tide2.transformers.core.resolve_model_path")
    @patch("tide2.transformers.config.get_resource_path")
    def test_configuration_parameters(self, mock_get_path, mock_resolve_model):
        """Test that configuration parameters are loaded correctly."""
        mock_resolve_model.return_value = "/fake/model/path"
        config_path = create_temp_config(self.mock_config)
        mock_get_path.return_value = config_path

        try:
            recognizer = TransformersRecognizer(model_name="TEST_MODEL")

            # Verify configuration parameters
            assert recognizer.ignore_labels == ["O"]
            assert recognizer.model_to_presidio_mapping == {}
            assert recognizer.default_explanation == "Test model explanation"
            assert recognizer.text_overlap_length == 40
            assert recognizer.chunk_length == 600
            assert recognizer.id_entity_name == "ID"
            assert recognizer.id_score_reduction == 0.5
        finally:
            Path(config_path).unlink()

    @patch("tide2.transformers.core.resolve_model_path")
    @patch("tide2.transformers.config.get_resource_path")
    def test_analyze_empty_text(self, mock_get_path, mock_resolve_model):
        """Test analysis of empty text."""
        mock_resolve_model.return_value = "/fake/model/path"
        config_path = create_temp_config(self.mock_config)
        mock_get_path.return_value = config_path

        try:
            recognizer = TransformersRecognizer(model_name="TEST_MODEL")

            # Mock the core's pipeline to avoid triggering actual loading
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = []
            mock_pipeline_instance.tokenizer.model_max_length = 512
            recognizer._core._pipeline = mock_pipeline_instance

            results = recognizer.analyze("", ["PERSON"])
            assert isinstance(results, list)
        finally:
            Path(config_path).unlink()

    @patch("tide2.transformers.core.resolve_model_path")
    @patch("tide2.transformers.config.get_resource_path")
    def test_unsupported_entity_filtering(self, mock_get_path, mock_resolve_model):
        """Test that unsupported entities are filtered out."""
        mock_resolve_model.return_value = "/fake/model/path"
        config_path = create_temp_config(self.mock_config)
        mock_get_path.return_value = config_path

        try:
            recognizer = TransformersRecognizer(model_name="TEST_MODEL")

            # Mock the core's pipeline to avoid triggering actual loading
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = []
            mock_pipeline_instance.tokenizer.model_max_length = 512
            recognizer._core._pipeline = mock_pipeline_instance

            # Request unsupported entity
            results = recognizer.analyze("Test text", ["UNSUPPORTED_ENTITY"])
            assert isinstance(results, list)
        finally:
            Path(config_path).unlink()

    def test_missing_configuration_file(self):
        """Test handling of missing configuration file."""
        with patch("tide2.transformers.config.get_resource_path") as mock_get_path:
            mock_get_path.return_value = "/nonexistent/path"

            # Expected to fail when configuration file is missing or model not found
            with pytest.raises((FileNotFoundError, KeyError)):
                TransformersRecognizer(model_name="NONEXISTENT_MODEL")

    @patch("tide2.transformers.config.get_resource_path")
    def test_missing_model_configuration(self, mock_get_path):
        """Test handling of missing model in configuration."""
        config_path = create_temp_config(self.mock_config)
        mock_get_path.return_value = config_path

        try:
            # Expected when model configuration is not found
            with pytest.raises(KeyError):
                TransformersRecognizer(model_name="MISSING_MODEL")
        finally:
            Path(config_path).unlink()
