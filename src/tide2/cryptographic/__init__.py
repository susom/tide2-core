"""
Cryptographic module for TIDE 2.0.

This module provides cryptographic utilities for secure anonymization,
including format-preserving encryption and key management tools.

Available utilities:
- FPE (Format Preserving Encryption) for strings
- Key management utilities
- String selection and encryption tools
- Date jitter derivation for deterministic anonymization
"""

from .date_jitter import derive_date_jitter
from .date_jitter import derive_date_jitter_batch
from .date_jitter import validate_jitter_parameters
from .fpe_strings import FormatPreservingEncryption
from .keys_utils import derive_key
from .keys_utils import generate_salt
from .keys_utils import key_from_hex_string
from .keys_utils import load_key
from .keys_utils import load_salt
from .keys_utils import save_key
from .keys_utils import save_salt
from .string_selector import clear_selector_cache
from .string_selector import get_cache_info
from .string_selector import secure_string_selector

__all__ = [
    "FormatPreservingEncryption",
    "clear_selector_cache",
    "derive_date_jitter",
    "derive_date_jitter_batch",
    "derive_key",
    "generate_salt",
    "get_cache_info",
    "key_from_hex_string",
    "load_key",
    "load_salt",
    "save_key",
    "save_salt",
    "secure_string_selector",
    "validate_jitter_parameters",
]
