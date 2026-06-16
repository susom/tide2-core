"""
Unified Ray actors for batch processing.

This module provides Ray actors for recognition, anonymization, and transformer
inference that work across all execution modes: local, VM, and cluster.

Actors:
    RecognizerActor: PII/PHI recognition using Presidio AnalyzerEngine
    AnonymizerActor: Anonymization using Presidio AnonymizerEngine with HIPS
    TransformerInferenceActor: GPU-based transformer NER inference

Factory Functions:
    create_anonymizer_actor: Create AnonymizerActor with keys (bytes or file paths)
    create_transformer_actor: Create TransformerInferenceActor with model config

Example:
    from tide2.actors import RecognizerActor, create_anonymizer_actor

    # Use directly with Ray Data
    ds.map_batches(RecognizerActor, batch_size=100, ...)

    # Create configured actor with factory
    AnonymizerActorClass = create_anonymizer_actor("/path/to/private.key", "/path/to/public.key")
    ds.map_batches(AnonymizerActorClass, batch_size=100, ...)
"""

from tide2.actors.anonymizer import AnonymizerActor
from tide2.actors.anonymizer import create_anonymizer_actor
from tide2.actors.anonymizer import create_anonymizer_actor_class  # Backwards compatibility
from tide2.actors.reassembly import ReassemblyActor
from tide2.actors.recognizer import NoOpContextEnhancer
from tide2.actors.recognizer import RecognizerActor


def __getattr__(name: str):
    """Lazy import for actors with heavy/optional dependencies.

    Transformer actors pull in torch; the LLM recognizer pulls in the provider
    SDKs from the optional ``[llm]`` extra (openai/anthropic/google-genai).
    Importing them lazily keeps ``import tide2.actors`` working for non-LLM,
    non-transformer jobs even when those extras are not installed.
    """
    _transformer_exports = {
        "BIOAggregationActor",
        "TransformerInferenceActor",
        "create_transformer_actor",
        "create_transformer_actor_class",
    }
    if name in _transformer_exports:
        from tide2.actors import transformer as _mod

        return getattr(_mod, name)
    if name == "LlmRecognizerActor":
        from tide2.actors.llm_recognizer import LlmRecognizerActor as _LlmRecognizerActor

        return _LlmRecognizerActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnonymizerActor",
    "BIOAggregationActor",
    "LlmRecognizerActor",
    "NoOpContextEnhancer",
    "ReassemblyActor",
    "RecognizerActor",
    "TransformerInferenceActor",
    "create_anonymizer_actor",
    "create_anonymizer_actor_class",
    "create_transformer_actor",
    "create_transformer_actor_class",
]
