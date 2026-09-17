"""
Transformer module for NER inference.

This module provides the core infrastructure for transformer-based Named Entity
Recognition (NER), used by the Presidio recognizer.

Classes:
    TransformerCore: Core inference engine with model loading and prediction

Functions:
    load_model_config: Load model configuration from config file
    get_available_models: List available model configurations
    format_transformer_recognizer_name: Canonical Presidio recognizer_name
        for transformer NER results

Example:
    from tide2.transformers import TransformerCore

    # Create core with explicit device placement for batch inference
    core = TransformerCore(model_name="StanfordAIMI/stanford-deidentifier-v2", device="cuda:0", load_immediately=True)

    # Run inference, returning raw BIO tokens per text
    raw_predictions = core.infer_raw_direct(["John Smith is a patient."])
"""

from .config import format_transformer_recognizer_name
from .config import get_available_models
from .config import load_model_config
from .core import TransformerCore

__all__ = [
    "TransformerCore",
    "format_transformer_recognizer_name",
    "get_available_models",
    "load_model_config",
]
