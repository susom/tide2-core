"""
Recognizers module for TIDE 2.0.

This module provides entity recognition and detection tools for identifying
personally identifiable information (PII) in text data.

Available recognizers:
- TransformersRecognizer: Machine learning-based entity recognition using Transformers
- AccessionNumberRecognizer: Medical accession number recognition (original)
- AccessionRecognizer: High-performance accession number detection (23x faster)
- HarRecognizer: HAR (Healthcare Associated Record) recognition
- MrnRecognizer: Medical Record Number recognition
- KnownValuesRecognizer: High-performance known values detection using Aho-Corasick (50-100x faster)
- PassthroughRecognizer: No-op recognizer for ablation studies
- CachedResultsTransformerRecognizer: Thread-safe ad-hoc recognizer for pre-computed batch results
- create_cached_recognizer: Factory function to create CachedResultsTransformerRecognizer instances
- create_recognizers_for_patient: Factory function to create KnownValuesRecognizer instances

High-performance recognizers (faster regex-only, used as default in workflows):
- PhoneRecognizer: High-performance phone number detection (59x faster than Presidio)
- UrlRecognizer: High-performance URL and IP address detection (35x faster than Presidio)
  - Detects URLs with protocols (http://, https://, ftp://, file://)
  - Detects www. prefix URLs
  - Detects bare domains (example.com)
  - Detects IPv4 addresses (e.g., 192.168.1.1)
  - Detects IPv6 addresses in all formats (e.g., 2001:db8::1, ::1, fe80::1)
- EmailRecognizer: High-performance email detection (1.2x faster than Presidio)
- AccessionRecognizer: High-performance accession number detection (23x faster)
- SsnRecognizer: High-performance US SSN detection (10-20x faster than Presidio)
- AddressRecognizer: High-performance US address detection using usaddress library
- InstitutionRecognizer: Institution-specific PHI detection (ships with Stanford Health Care patterns)

Note: For batch processing, use the runner module:
    from tide2.runner import LocalJobRunner
    runner = LocalJobRunner()
    runner.run_recognition(input_path, output_path)

Note: Conflict resolution and deduplication is now handled in tide2.utils.span_metrics
using O(n log n) algorithms. See resolve_conflicts() and resolve_recognizer_results().
"""

from .accession_recognizer import AccessionRecognizer
from .address_recognizer import AddressRecognizer
from .base64_image_recognizer import Base64ImageRecognizer
from .cached_results_transformers_recognizer import CachedResultsTransformerRecognizer
from .cached_results_transformers_recognizer import create_cached_recognizer
from .email_recognizer import EmailRecognizer
from .genetic_sequence_recognizer import GeneticSequenceRecognizer
from .har_recognizer import HarRecognizer
from .institution_recognizer import InstitutionRecognizer
from .known_values import KnownValuesRecognizer
from .known_values import create_recognizers_for_patient
from .llm_json_recognizer import LlmJsonRecognizer
from .mrn_recognizer import MrnRecognizer
from .passthrough_recognizer import PassthroughRecognizer
from .phone_recognizer import PhoneRecognizer
from .ssn_recognizer import SsnRecognizer
from .url_recognizer import UrlRecognizer


def __getattr__(name: str):
    """Lazy import for torch-dependent recognizers."""
    if name == "TransformersRecognizer":
        from .transformers_recognizer import TransformersRecognizer

        return TransformersRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AccessionRecognizer",
    "AddressRecognizer",
    "Base64ImageRecognizer",
    "CachedResultsTransformerRecognizer",
    "EmailRecognizer",
    "GeneticSequenceRecognizer",
    "HarRecognizer",
    "InstitutionRecognizer",
    "KnownValuesRecognizer",
    "LlmJsonRecognizer",
    "MrnRecognizer",
    "PassthroughRecognizer",
    "PhoneRecognizer",
    "SsnRecognizer",
    "TransformersRecognizer",
    "UrlRecognizer",
    "create_cached_recognizer",
    "create_recognizers_for_patient",
]
