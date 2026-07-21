"""
Transformer module for NER inference.

This module provides the core infrastructure for transformer-based Named Entity
Recognition (NER), used by both the Presidio recognizer and Ray actor wrappers.

Classes:
    TransformerCore: Core inference engine with model loading and prediction

Functions:
    load_model_config: Load model configuration from config file
    get_available_models: List available model configurations
    format_transformer_recognizer_name: Canonical Presidio recognizer_name
        for transformer NER results

Example:
    from tide2.transformers import TransformerCore

    # Create core with auto device placement
    core = TransformerCore(model_name="StanfordAIMI/stanford-deidentifier-v2")

    # Run inference with BIO aggregation
    entities = core.infer_aggregated("John Smith is a patient.")
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
