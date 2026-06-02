"""
String parsers module for TIDE 2.0.

This module provides text parsing and format detection utilities for
processing and analyzing text data before anonymization.

Available parsers:
- Address parsers for location data
- Format detectors for identifying text patterns
- Name parsers for personal name processing
- Name tokenizer for name token classification
"""

from .address_parsers import *
from .format_detector import FormatDetector
from .format_detector import FormatType
from .name_parsers import NameParser
from .name_parsers import NameType
from .name_parsers import ParsedToken
from .name_tokenizer import NameToken
from .name_tokenizer import NameTokenizer
from .name_tokenizer import TokenType

__all__ = [
    "FormatDetector",
    "FormatType",
    # Name parsing exports
    "NameParser",
    "ParsedToken",
    "NameType",
    "NameTokenizer",
    "NameToken",
    "TokenType",
]
