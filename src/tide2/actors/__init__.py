"""
Unified Ray actors for batch processing.

This module provides Ray actors and worker classes for recognition,
anonymization, and transformer inference that work across all execution modes:
local, VM, and cluster.

Ray Data UDFs:
    RecognizerActor: Plain RecognizerWorker class passed to map_batches()
    AnonymizerActor: Plain AnonymizerWorker class passed to map_batches()
    LlmRecognizerActor: Plain LlmRecognizerWorker class passed to map_batches()
    TransformerInferenceActor: GPU-based transformer NER inference

Note on .remote():
    RecognizerActor, AnonymizerActor, and LlmRecognizerActor are plain callable
    classes driven directly by Ray Data map_batches(). Calling .remote() on them
    is not supported; use RecognizerWorkerActor, AnonymizerWorkerActor, or
    LlmRecognizerWorkerActor if direct Ray remote actor spawning is required.

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
        try:
            from tide2.actors.llm_recognizer import LlmRecognizerActor as _LlmRecognizerActor
        except ModuleNotFoundError as exc:
            from tide2._optional import reraise_missing_llm_sdk

            reraise_missing_llm_sdk("LlmRecognizerActor", exc)
            raise  # real internal import failure: propagate unchanged

        return _LlmRecognizerActor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnonymizerActor",
    "BIOAggregationActor",
    "LlmRecognizerActor",
    "NoOpContextEnhancer",
    "RecognizerActor",
    "TransformerInferenceActor",
    "create_anonymizer_actor",
    "create_anonymizer_actor_class",
    "create_transformer_actor",
    "create_transformer_actor_class",
]
