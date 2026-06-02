"""
Anonymizers module for TIDE 2.0.

This module provides various anonymization strategies and operators for
de-identifying different types of sensitive data.

Available anonymizers:
- AccessionNumberHashAnonymizer: SHA256-based hashing for accession numbers (BQ compatible)
- FakerAnonymizer: Uses Faker library for realistic fake data generation
- AgeGroupAnonymizer: Groups ages into ranges for anonymization
- DateJitterAnonymizer: Adds random noise to dates
- HipsAlphaNumericAnonymizer: HIPS-based alphanumeric anonymization
- HipsLocationAnonymizer: HIPS-based location anonymization
- HipsNamesAnonymizer: HIPS-based name anonymization

All anonymizers use class-level caching for optimal performance.
For Ray-based batch processing, use the runner module:
    from tide2.runner import AnonymizerActor, run_anonymization_simple
"""

from .accession_number_hash import AccessionNumberHashAnonymizer
from .age_grouping import AgeGroupAnonymizer
from .date_jitter import DateJitterAnonymizer
from .faker import FakerAnonymizer
from .hips_alphanumeric import HipsAlphaNumericAnonymizer
from .hips_locations import HipsLocationAnonymizer
from .hips_names import HipsNamesAnonymizer
from .masking import MaskingAnonymizer

__all__ = [
    "AccessionNumberHashAnonymizer",
    "AgeGroupAnonymizer",
    "DateJitterAnonymizer",
    "FakerAnonymizer",
    "HipsAlphaNumericAnonymizer",
    "HipsLocationAnonymizer",
    "HipsNamesAnonymizer",
    "MaskingAnonymizer",
]
