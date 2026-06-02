"""
Configuration loading utilities for transformer models.

This module provides functions to load and access transformer model configurations
from the BERT transformer configuration file.
"""

import json
import logging
from pathlib import Path
from typing import Any

from tide2.utils.resource_utils import BERT_TRANSFORMER_CONFIG_FILE
from tide2.utils.resource_utils import get_resource_path

logger = logging.getLogger(__name__)


def load_model_config(model_name: str) -> dict[str, Any]:
    """Load model configuration from the BERT transformer configuration file.

    Args:
        model_name: Name of the model configuration to load

    Returns:
        Dictionary containing the model configuration

    Raises:
        KeyError: If the model_name is not found in the configuration file
        FileNotFoundError: If the configuration file doesn't exist
    """
    config_path_str = get_resource_path(BERT_TRANSFORMER_CONFIG_FILE)
    config_path = Path(config_path_str)

    with config_path.open(encoding="utf-8") as f:
        all_configs = json.load(f)

    if model_name not in all_configs:
        available_models = list(all_configs.keys())
        raise KeyError(f"Model '{model_name}' not found in configuration. Available models: {available_models}")

    return all_configs[model_name]


def get_available_models() -> list[str]:
    """Get list of available model configurations.

    Returns:
        List of available model names that can be used with TransformerCore
    """
    config_path_str = get_resource_path(BERT_TRANSFORMER_CONFIG_FILE)
    config_path = Path(config_path_str)

    with config_path.open(encoding="utf-8") as f:
        all_configs = json.load(f)

    return list(all_configs.keys())
