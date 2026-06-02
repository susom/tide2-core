"""Shared configuration for span and token level evaluation.

This module contains constants and utilities shared between span_eval.py
and token_eval.py to avoid code duplication.
"""

from __future__ import annotations

# =============================================================================
# RECOGNIZER CATEGORY DEFINITIONS FOR ABLATION STUDY
# =============================================================================
# Maps recognizer_name patterns to categories (uses prefix matching)
#
# Recognizer names from RecognizerActor (actors/recognizer.py):
#   - Presidio: DateRecognizer
#   - Custom regex: AccessionRecognizer, AddressRecognizer, EmailRecognizer,
#     HarRecognizer, MrnRecognizer, PhoneRecognizer, SsnRecognizer, UrlRecognizer,
#     Base64ImageRecognizer, GeneticSequenceRecognizer
#   - Ad-hoc: KnownValuesRecognizer, CachedResultsTransformerRecognizer
#
# Transformer results use format: TransformersRecognizer[model_name]
#
# Note: LlmJsonRecognizer is NOT used in the RecognizerActor pipeline.
RECOGNIZER_CATEGORIES = {
    # Regex/rule-based recognizers (from Presidio and custom)
    "regex": [
        "AccessionRecognizer",
        "AddressRecognizer",  # Uses usaddress library
        "DateRecognizer",  # Presidio's DateRecognizer
        "EmailRecognizer",
        "HarRecognizer",
        "MrnRecognizer",
        "PhoneRecognizer",
        "SsnRecognizer",
        "UrlRecognizer",
        "Base64ImageRecognizer",
        "GeneticSequenceRecognizer",
    ],
    # Known values (Aho-Corasick based - patient-specific PHI)
    "known_values": [
        "KnownValuesRecognizer",
    ],
    # Transformer-based NER models (prefix match handles model name suffix)
    # Format: TransformersRecognizer[model_name] or CachedResultsTransformerRecognizer
    "transformers": [
        "TransformersRecognizer",  # Matches TransformersRecognizer[...]
        "CachedResultsTransformerRecognizer",
    ],
    # LLM-based recognizers (JSON output from LLM inference)
    # Format: LlmJsonRecognizer or LlmJsonRecognizer-model_name
    "llm": [
        "LlmJsonRecognizer",  # Matches LlmJsonRecognizer, LlmJsonRecognizer-google/gemini-2.5-flash-lite, etc.
    ],
}

# Incremental ablation configurations
# Each tuple: (config_name, list_of_categories_to_include)
# Note: The "all" config includes all categories actually used in the pipeline.
# LLM recognizers are not currently used in RecognizerActor.
ABLATION_CONFIGS = [
    ("regex+known_values", ["regex", "known_values"]),
    ("transformers_only", ["transformers"]),
    ("all", ["regex", "known_values", "transformers"]),
]

# =============================================================================
# UNIFIED LABEL MAP FOR EVALUATION
# =============================================================================
# Maps fine-grained gold and ML labels to a common coarser taxonomy so that
# evaluation compares like-with-like.  For example, gold "PATIENT" and ML "HCW"
# both map to "PERSON"; gold "US_SSN", "HAR", "MRN", "ACC_NUM" and ML "ID" all
# map to "ID".  Labels not present in the map are dropped when
# drop_unmapped=True, which is the recommended default for evaluation.
UNIFIED_LABEL_MAP = {
    # Gold labels
    "PATIENT": "PERSON",
    "PHONE": "PHONE",
    "EMAIL_ADDRESS": "WEB",
    "LOCATION": "LOCATION",
    "US_SSN": "ID",
    "HAR": "ID",
    "MRN": "ID",
    "ACC_NUM": "ID",
    # ML-only labels mapped to common taxonomy
    "HCW": "PERSON",
    "HOSPITAL": "LOCATION",
    "VENDOR": "PERSON",
    "ID": "ID",
    "URL": "WEB",
    "DATE": "DATE",
    "DATE_TIME": "DATE",
    "BASE64_IMAGE": "OTHER",
    "GENETIC_SEQUENCE": "OTHER",
}


def get_recognizer_category(recognizer_name: str, category_map: dict[str, list[str]] | None = None) -> str | None:
    """Map a recognizer_name to its category using prefix matching.

    Args:
        recognizer_name: Name of the recognizer (e.g., "TransformersRecognizer[model]")
        category_map: Optional custom category map. Defaults to RECOGNIZER_CATEGORIES.

    Returns:
        Category name (e.g., "transformers") or None if no match.
    """
    if category_map is None:
        category_map = RECOGNIZER_CATEGORIES

    if not recognizer_name or recognizer_name == "unknown":
        return None
    for category, recognizer_prefixes in category_map.items():
        for prefix in recognizer_prefixes:
            if recognizer_name.startswith(prefix):
                return category
    return None
