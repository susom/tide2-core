"""
Centralized regex patterns and utility functions for format detection and validation.

This module provides a consistent interface for accessing and using regex patterns
to detect various data formats like phone numbers, SSNs, credit cards, etc.
"""

import re
from dataclasses import dataclass
from enum import Enum


class FormatType(Enum):
    """Enumeration of supported format types."""

    PHONE_US = "phone_us"
    PHONE_INTL = "phone_intl"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    LICENSE_PLATE = "license_plate"
    ACCOUNT_NUMBER = "account_number"
    ROUTING_NUMBER = "routing_number"
    UUID = "uuid"
    HEX_STRING = "hex_string"
    EMAIL = "email"
    ZIP_CODE = "zip_code"
    DATE_MDY = "date_mdy"
    DATE_YMD = "date_ymd"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    URL = "url"
    URL_HTTP = "url_http"
    URL_HTTPS = "url_https"
    URL_FTP = "url_ftp"
    URL_WITH_PORT = "url_with_port"
    URL_WITH_PARAMS = "url_with_params"
    URL_LOCALHOST = "url_localhost"
    URL_IP_BASED = "url_ip_based"
    URL_WITH_AUTH = "url_with_auth"
    URL_INCOMPLETE = "url_incomplete"
    URL_DOMAIN_ONLY = "url_domain_only"
    URL_WWW_PREFIX = "url_www_prefix"
    URL_LOCALHOST_NO_PROTOCOL = "url_localhost_no_protocol"
    URL_IP_NO_PROTOCOL = "url_ip_no_protocol"
    URL_MALFORMED = "url_malformed"
    URL_INCOMPLETE_PORT = "url_incomplete_port"
    URL_LEADING_DOT = "url_leading_dot"
    URL_TRAILING_DOT = "url_trailing_dot"
    URL_DOUBLE_DOT = "url_double_dot"
    AGE_WITH_UNITS = "age_with_units"
    AGE_NUMERIC_ONLY = "age_numeric_only"
    AGE_GESTATIONAL = "age_gestational"
    AGE_WRITTEN_NUMBERS = "age_written_numbers"
    # Basic alphabet categories for FPE
    DIGITS = "digits"
    LOWERCASE = "lowercase"
    UPPERCASE = "uppercase"
    HEX_LOWER = "hex_lower"
    HEX_UPPER = "hex_upper"
    ALPHANUMERIC_LOWER = "alphanumeric_lower"
    ALPHANUMERIC_UPPER = "alphanumeric_upper"
    ALPHANUMERIC_MIXED = "alphanumeric_mixed"
    # Leading whitespace patterns for FPE
    LEADING_SPACE_DIGITS = "leading_space_digits"
    LEADING_SPACE_ALPHANUMERIC = "leading_space_alphanumeric"
    # Trailing whitespace patterns for FPE
    TRAILING_SPACE_DIGITS = "trailing_space_digits"
    TRAILING_SPACE_ALPHANUMERIC = "trailing_space_alphanumeric"
    UNKNOWN = "unknown"


@dataclass
class PatternInfo:
    """Information about a regex pattern."""

    pattern: str
    description: str
    format_type: FormatType
    case_sensitive: bool = False
    examples: list[str] | None = None

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


class RegexPatterns:
    """Registry of regex patterns for various data formats."""

    # Patterns ordered by specificity (most specific first)
    _patterns: dict[FormatType, PatternInfo] = {
        # Basic alphabet categories first for FPE priority
        FormatType.DIGITS: PatternInfo(
            pattern=r"^\d+$",
            description="Pure numeric string (digits only)",
            format_type=FormatType.DIGITS,
            examples=["123", "456789", "12345678"],
        ),
        FormatType.LOWERCASE: PatternInfo(
            pattern=r"^[a-z]+$",
            description="Lowercase letters only",
            format_type=FormatType.LOWERCASE,
            case_sensitive=True,
            examples=["hello", "world", "abcdef"],
        ),
        FormatType.UPPERCASE: PatternInfo(
            pattern=r"^[A-Z]+$",
            description="Uppercase letters only",
            format_type=FormatType.UPPERCASE,
            case_sensitive=True,
            examples=["HELLO", "WORLD", "ABCDEF"],
        ),
        FormatType.HEX_LOWER: PatternInfo(
            pattern=r"^[0-9a-f]+$",
            description="Lowercase hexadecimal string",
            format_type=FormatType.HEX_LOWER,
            case_sensitive=True,
            examples=["abc123", "deadbeef", "ff00aa"],
        ),
        FormatType.HEX_UPPER: PatternInfo(
            pattern=r"^[0-9A-F]+$",
            description="Uppercase hexadecimal string",
            format_type=FormatType.HEX_UPPER,
            case_sensitive=True,
            examples=["ABC123", "DEADBEEF", "FF00AA"],
        ),
        FormatType.ALPHANUMERIC_LOWER: PatternInfo(
            pattern=r"^[0-9a-z]+$",
            description="Alphanumeric with lowercase letters",
            format_type=FormatType.ALPHANUMERIC_LOWER,
            case_sensitive=True,
            examples=["abc123", "hello123", "test456"],
        ),
        FormatType.ALPHANUMERIC_UPPER: PatternInfo(
            pattern=r"^[0-9A-Z]+$",
            description="Alphanumeric with uppercase letters",
            format_type=FormatType.ALPHANUMERIC_UPPER,
            case_sensitive=True,
            examples=["ABC123", "HELLO123", "TEST456"],
        ),
        FormatType.ALPHANUMERIC_MIXED: PatternInfo(
            pattern=r"^[0-9a-zA-Z]+$",
            description="Alphanumeric with mixed case letters",
            format_type=FormatType.ALPHANUMERIC_MIXED,
            case_sensitive=True,
            examples=["AbC123", "HelloWorld", "Test456"],
        ),
        # Leading whitespace patterns for FPE
        FormatType.LEADING_SPACE_DIGITS: PatternInfo(
            pattern=r"^\s+\d+$",
            description="Leading whitespace followed by digits only",
            format_type=FormatType.LEADING_SPACE_DIGITS,
            examples=[" 123456789", "  987654321", "\t123456"],
        ),
        FormatType.LEADING_SPACE_ALPHANUMERIC: PatternInfo(
            pattern=r"^\s+[0-9a-zA-Z]+$",
            description="Leading whitespace followed by alphanumeric characters",
            format_type=FormatType.LEADING_SPACE_ALPHANUMERIC,
            examples=[" abc123def", "  Hello123", "\tTest456"],
        ),
        # Trailing whitespace patterns for FPE
        FormatType.TRAILING_SPACE_DIGITS: PatternInfo(
            pattern=r"^\d+\s+$",
            description="Digits followed by trailing whitespace",
            format_type=FormatType.TRAILING_SPACE_DIGITS,
            examples=["123456789 ", "987654321  ", "123456\t"],
        ),
        FormatType.TRAILING_SPACE_ALPHANUMERIC: PatternInfo(
            pattern=r"^[0-9a-zA-Z]+\s+$",
            description="Alphanumeric characters followed by trailing whitespace",
            format_type=FormatType.TRAILING_SPACE_ALPHANUMERIC,
            examples=["abc123def ", "Hello123  ", "Test456\t"],
        ),
        # Most specific patterns first
        FormatType.PHONE_US: PatternInfo(
            pattern=r"^\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$",
            description="US phone number with optional formatting",
            format_type=FormatType.PHONE_US,
            examples=["(555) 123-4567", "555-123-4567", "5551234567", "555.123.4567"],
        ),
        FormatType.PHONE_INTL: PatternInfo(
            pattern=r"^\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}$",
            description="International phone number",
            format_type=FormatType.PHONE_INTL,
            examples=["+1-555-123-4567", "+44 20 7946 0958", "+33 1 42 86 83 26"],
        ),
        FormatType.SSN: PatternInfo(
            pattern=r"^\d{3}-\d{2}-\d{4}$",
            description="Social Security Number (with dashes)",
            format_type=FormatType.SSN,
            examples=["123-45-6789"],
        ),
        # Age patterns - ordered by specificity (most specific first)
        FormatType.AGE_GESTATIONAL: PatternInfo(
            pattern=r"^(?:\d{1,2}w(?:\d{1,2}d)?|\d{1,2}[-\s]?week(?:s)?(?:[-\s]?\d{1,2}[-\s]?day(?:s)?)?\s*old|\d{1,2}[-\s]?day[-\s]?old(?:\s+\d{1,2}w\d{1,2}d)?)$",
            description="Gestational age (weeks/days format like 29w2d, 5-week old, 10-day old 29w2d)",
            format_type=FormatType.AGE_GESTATIONAL,
            case_sensitive=False,
            examples=["29w2d", "37w2d", "5-week old", "4-week old", "10-day old 29w2d"],
        ),
        FormatType.AGE_WRITTEN_NUMBERS: PatternInfo(
            pattern=r"^(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[-\s]?)?(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\s+(?:year|month|day)(?:s)?\s*(?:old)?$",
            description="Written number ages (twenty-seven year old, six month old)",
            format_type=FormatType.AGE_WRITTEN_NUMBERS,
            case_sensitive=False,
            examples=["twenty-seven year old", "six month old", "sixty-two year old", "seventy-seven year old"],
        ),
        FormatType.AGE_WITH_UNITS: PatternInfo(
            pattern=r"^(?:(?:age\s*:?\s*)?[\(\[\"\s,]*>?\s*)?(?:\d{1,3}(?:\.\d{1,2})?(?:\s*1/2)?)\s*(?:[-\s]?(?:yo|y\.?o\.?|y/o|year(?:s)?(?:\s*of\s*age)?|yr(?:s)?|Y|month(?:s)?|mo|m\.o|day(?:s)?|Years?))?(?:\s*[-\s]?old)?(?:\s*Age\s*Units?\s*:\s*Years?)?[\)\]\"\s]*$|^\d{1,3}(?:yo|y|Y)$|^[\(\[\"\s,]*\d{1,3}[-\s]?[yY](?:\s+[yY])?[\)\]\"\s]*$|^S\d{1,3}y$|^\d{1,3}M$",
            description="Age with various units (18 yo, 32 Y, 12 mo, 75 year old, 16yo, 64y, age 10, Age: 33 Age Units: Years, S79y, 53M)",
            format_type=FormatType.AGE_WITH_UNITS,
            case_sensitive=False,
            examples=[
                "18 yo",
                "32 Y",
                "12 mo",
                "75 year old",
                "56-year-old",
                "49 year old",
                "age 10",
                "Age: 33",
                "Age Units: Years",
                "61 yr",
                "28 y/o",
                "15 mo",
                "47 Years",
                "2 1/2",
                "18 years of age",
                "65 years of age",
                "82 yrs",
                "16yo",
                "70yo",
                "2yo",
                "64y",
                "47y",
                "83y",
                "54Y",
                "75yo",
                "59yo",
                "30yo",
                "61yo",
                "71yo",
                "S79y",
                "53M",
                "57M",
            ],
        ),
        FormatType.AGE_NUMERIC_ONLY: PatternInfo(
            pattern=r"^(?:[\(\[\"\s,]*)?(?:\d{1,3})(?:[\)\]\"\s,]*)?$",
            description="Numeric age only (no units specified)",
            format_type=FormatType.AGE_NUMERIC_ONLY,
            case_sensitive=False,
            examples=["21", "49", "50", "68", "33", "25", "75", "42", "53", "74", "84", "17", "20", "22", "44", "6"],
        ),
        FormatType.CREDIT_CARD: PatternInfo(
            pattern=r"^\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}$",
            description="Credit card number",
            format_type=FormatType.CREDIT_CARD,
            examples=["1234 5678 9012 3456", "1234-5678-9012-3456", "1234567890123456"],
        ),
        FormatType.UUID: PatternInfo(
            pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            description="UUID (Universally Unique Identifier)",
            format_type=FormatType.UUID,
            case_sensitive=False,
            examples=["550e8400-e29b-41d4-a716-446655440000", "6ba7b810-9dad-11d1-80b4-00c04fd430c8"],
        ),
        FormatType.EMAIL: PatternInfo(
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            description="Email address",
            format_type=FormatType.EMAIL,
            examples=["user@example.com", "john.doe+label@company.org"],
        ),
        FormatType.IP_ADDRESS: PatternInfo(
            pattern=r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
            description="IPv4 address",
            format_type=FormatType.IP_ADDRESS,
            examples=["192.168.1.1", "10.0.0.1", "255.255.255.255"],
        ),
        FormatType.MAC_ADDRESS: PatternInfo(
            pattern=r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$",
            description="MAC address",
            format_type=FormatType.MAC_ADDRESS,
            case_sensitive=False,
            examples=["00:1B:63:84:45:E6", "00-1B-63-84-45-E6"],
        ),
        # URL patterns - ordered by specificity (most specific first)
        FormatType.URL_WITH_AUTH: PatternInfo(
            pattern=r"^(?:https?|ftp)://[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+@(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})?(?::\d{1,5})?(?:/[^\s]*)?$",
            description="URL with username and password authentication",
            format_type=FormatType.URL_WITH_AUTH,
            case_sensitive=False,
            examples=["https://user:pass@example.com", "ftp://admin:secret@ftp.example.org:21/files"],
        ),
        FormatType.URL_HTTPS: PatternInfo(
            pattern=r"^https://(?:[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+@)?(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})?(?::\d{1,5})?(?:/[^\s]*)?$",
            description="HTTPS URL",
            format_type=FormatType.URL_HTTPS,
            case_sensitive=False,
            examples=["https://www.example.com", "https://api.example.org/v1/users", "https://example.com:8443/path"],
        ),
        FormatType.URL_HTTP: PatternInfo(
            pattern=r"^http://(?:[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+@)?(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})?(?::\d{1,5})?(?:/[^\s]*)?$",
            description="HTTP URL",
            format_type=FormatType.URL_HTTP,
            case_sensitive=False,
            examples=["http://www.example.com", "http://localhost:8080", "http://example.org/api/v1"],
        ),
        FormatType.URL_FTP: PatternInfo(
            pattern=r"^ftp://(?:[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+@)?(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})?(?::\d{1,5})?(?:/[^\s]*)?$",
            description="FTP URL",
            format_type=FormatType.URL_FTP,
            case_sensitive=False,
            examples=["ftp://ftp.example.com", "ftp://files.example.org:21/documents"],
        ),
        FormatType.URL_WITH_PORT: PatternInfo(
            pattern=r"^https?://(?:[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+@)?(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})?:\d{1,5}(?:/[^\s]*)?$",
            description="URL with explicit port number",
            format_type=FormatType.URL_WITH_PORT,
            case_sensitive=False,
            examples=["http://example.com:8080", "https://api.example.org:3000/v1", "http://localhost:5000"],
        ),
        FormatType.URL_WITH_PARAMS: PatternInfo(
            pattern=r"^(?:https?|ftp)://(?:[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+@)?(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})?(?::\d{1,5})?/[^\s]*\?[^\s]*$",
            description="URL with query parameters",
            format_type=FormatType.URL_WITH_PARAMS,
            case_sensitive=False,
            examples=["https://example.com/search?q=test", "http://api.example.org/users?limit=10&offset=20"],
        ),
        FormatType.URL_LOCALHOST: PatternInfo(
            pattern=r"^https?://(?:[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+@)?localhost(?::\d{1,5})?(?:/[^\s]*)?$",
            description="Localhost URL",
            format_type=FormatType.URL_LOCALHOST,
            case_sensitive=False,
            examples=["http://localhost", "https://localhost:3000", "http://localhost:8080/api"],
        ),
        FormatType.URL_IP_BASED: PatternInfo(
            pattern=r"^https?://(?:[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+@)?(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?::\d{1,5})?(?:/[^\s]*)?$",
            description="URL with IP address",
            format_type=FormatType.URL_IP_BASED,
            case_sensitive=False,
            examples=["http://192.168.1.1", "https://10.0.0.1:8443", "http://127.0.0.1:3000/api"],
        ),
        FormatType.URL: PatternInfo(
            pattern=r"^(?:https?|ftp)://(?:[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+@)?(?:(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})?|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|localhost)(?::\d{1,5})?(?:/[^\s]*)?$",
            description="General URL (HTTP/HTTPS/FTP, supports domains, IPs, localhost, and authentication)",
            format_type=FormatType.URL,
            case_sensitive=False,
            examples=[
                "https://www.example.com",
                "http://192.168.1.1:8080",
                "ftp://user:pass@files.example.org",
                "http://localhost:3000",
            ],
        ),
        # Incomplete URL patterns (without protocol)
        FormatType.URL_WWW_PREFIX: PatternInfo(
            pattern=r"^www\.(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?::\d{1,5})?(?:/[^\s]*)?$",
            description="URL starting with www (no protocol)",
            format_type=FormatType.URL_WWW_PREFIX,
            case_sensitive=False,
            examples=["www.example.com", "www.google.com/search", "www.api.example.org:8080"],
        ),
        FormatType.URL_LOCALHOST_NO_PROTOCOL: PatternInfo(
            pattern=r"^localhost(?::\d{1,5})?(?:/[^\s]*)?$",
            description="Localhost without protocol",
            format_type=FormatType.URL_LOCALHOST_NO_PROTOCOL,
            case_sensitive=False,
            examples=["localhost", "localhost:3000", "localhost:8080/api"],
        ),
        FormatType.URL_IP_NO_PROTOCOL: PatternInfo(
            pattern=r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?::\d{1,5})?(?:/[^\s]*)?$",
            description="IP address without protocol",
            format_type=FormatType.URL_IP_NO_PROTOCOL,
            case_sensitive=False,
            examples=["192.168.1.1", "10.0.0.1:8080", "127.0.0.1:3000/api"],
        ),
        FormatType.URL_DOMAIN_ONLY: PatternInfo(
            pattern=r"^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?::\d{1,5})?(?:/[^\s]*)?$",
            description="Domain name without protocol (must have at least one dot and TLD)",
            format_type=FormatType.URL_DOMAIN_ONLY,
            case_sensitive=False,
            examples=["example.com", "api.example.org", "subdomain.example.co.uk:8080/path"],
        ),
        FormatType.URL_INCOMPLETE: PatternInfo(
            pattern=r"^(?:(?:www\.)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|localhost)(?::\d{1,5})?(?:/[^\s]*)?$",
            description="General incomplete URL (domain, IP, or localhost without protocol)",
            format_type=FormatType.URL_INCOMPLETE,
            case_sensitive=False,
            examples=["example.com", "www.example.com", "192.168.1.1:8080", "localhost:3000"],
        ),
        # Malformed URL patterns (edge cases)
        FormatType.URL_INCOMPLETE_PORT: PatternInfo(
            pattern=r"^(?:https?://)?(?:(?:www\.)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|localhost):$",
            description="URL with incomplete port (ends with colon)",
            format_type=FormatType.URL_INCOMPLETE_PORT,
            case_sensitive=False,
            examples=["https://example.com:", "localhost:", "www.example.com:"],
        ),
        FormatType.URL_LEADING_DOT: PatternInfo(
            pattern=r"^\.(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?::\d{1,5})?(?:/[^\s]*)?$",
            description="URL with leading dot",
            format_type=FormatType.URL_LEADING_DOT,
            case_sensitive=False,
            examples=[".example.com", ".www.example.org"],
        ),
        FormatType.URL_TRAILING_DOT: PatternInfo(
            pattern=r"^(?:https?://)?(?:www\.)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\.(?::\d{1,5})?(?:/[^\s]*)?$",
            description="URL with trailing dot after domain",
            format_type=FormatType.URL_TRAILING_DOT,
            case_sensitive=False,
            examples=["example.com.", "https://www.example.org.", "api.example.com./v1"],
        ),
        FormatType.URL_DOUBLE_DOT: PatternInfo(
            pattern=r"^(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.\.+[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})?(?::\d{1,5})?(?:/[^\s]*)?$",
            description="URL with double dots in domain",
            format_type=FormatType.URL_DOUBLE_DOT,
            case_sensitive=False,
            examples=["example..com", "www..example.org", "https://api..example.com"],
        ),
        FormatType.URL_MALFORMED: PatternInfo(
            pattern=r"^(?:(?:https?://)?(?:(?:www\.)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|localhost):(?:/[^\s]*)?$|^\.(?:[a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+\.[a-zA-Z]{2,}|^(?:https?://)?(?:www\.)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\.|^(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.\.+[a-zA-Z0-9-]+)(?::\d{1,5})?(?:/[^\s]*)?$",
            description="General malformed URL (various edge cases)",
            format_type=FormatType.URL_MALFORMED,
            case_sensitive=False,
            examples=["https://example.com:", ".example.com", "example.com.", "example..com", "localhost:"],
        ),
        FormatType.DATE_MDY: PatternInfo(
            pattern=r"^(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-]\d{4}$",
            description="Date in MM/DD/YYYY or M/D/YYYY format",
            format_type=FormatType.DATE_MDY,
            examples=["12/31/2023", "1/1/2024", "03-15-2023"],
        ),
        FormatType.DATE_YMD: PatternInfo(
            pattern=r"^\d{4}[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])$",
            description="Date in YYYY/MM/DD or YYYY/M/D format",
            format_type=FormatType.DATE_YMD,
            examples=["2023/12/31", "2024-01-01", "2023/3/15"],
        ),
        FormatType.ZIP_CODE: PatternInfo(
            pattern=r"^\d{5}(-\d{4})?$",
            description="US ZIP code",
            format_type=FormatType.ZIP_CODE,
            examples=["12345", "12345-6789"],
        ),
        FormatType.ROUTING_NUMBER: PatternInfo(
            pattern=r"^\d{9}$",
            description="Bank routing number (9 digits)",
            format_type=FormatType.ROUTING_NUMBER,
            examples=["123456789"],
        ),
        # Basic alphabet categories for FPE (ordered by specificity - most specific first)
        FormatType.DIGITS: PatternInfo(
            pattern=r"^\d+$",
            description="Pure numeric string (digits only)",
            format_type=FormatType.DIGITS,
            examples=["123", "456789", "12345678"],
        ),
        FormatType.HEX_STRING: PatternInfo(
            pattern=r"^[0-9a-f]{8,}$",
            description="Hexadecimal string (8+ characters)",
            format_type=FormatType.HEX_STRING,
            case_sensitive=False,
            examples=["deadbeef", "123abc456def", "ff00ff00ff00"],
        ),
        FormatType.ACCOUNT_NUMBER: PatternInfo(
            pattern=r"^\d{8,16}$",
            description="Bank account number",
            format_type=FormatType.ACCOUNT_NUMBER,
            examples=["12345678", "1234567890123456"],
        ),
        FormatType.LICENSE_PLATE: PatternInfo(
            pattern=r"^[A-Z0-9]{2,8}$",
            description="License plate number",
            format_type=FormatType.LICENSE_PLATE,
            case_sensitive=False,
            examples=["ABC123", "12345AB", "CUSTOM"],
        ),
    }

    @classmethod
    def get_pattern(cls, format_type: FormatType) -> PatternInfo | None:
        """Get pattern information for a specific format type.

        Args:
            format_type: The format type to look up.

        Returns:
            The PatternInfo for the format type, or None if not registered.
        """
        return cls._patterns.get(format_type)

    @classmethod
    def get_pattern_string(cls, format_type: FormatType) -> str | None:
        """Get the regex pattern string for a specific format type.

        Args:
            format_type: The format type to look up.

        Returns:
            The regex pattern string, or None if the format type is not registered.
        """
        pattern_info = cls._patterns.get(format_type)
        return pattern_info.pattern if pattern_info else None

    @classmethod
    def get_all_patterns(cls) -> dict[FormatType, PatternInfo]:
        """Get all registered patterns.

        Returns:
            A copy of the internal patterns dictionary.
        """
        return cls._patterns.copy()

    @classmethod
    def add_pattern(cls, format_type: FormatType, pattern_info: PatternInfo) -> None:
        """Add or update a pattern in the global registry.

        Args:
            format_type: The format type key.
            pattern_info: The pattern definition to register.
        """
        cls._patterns[format_type] = pattern_info

    @classmethod
    def remove_pattern(cls, format_type: FormatType) -> bool:
        """Remove a pattern from the global registry.

        Args:
            format_type: The format type to remove.

        Returns:
            True if the pattern existed and was removed, False otherwise.
        """
        return cls._patterns.pop(format_type, None) is not None

    @classmethod
    def get_format_types(cls) -> list[FormatType]:
        """Get all available format types.

        Returns:
            List of all registered FormatType keys.
        """
        return list(cls._patterns.keys())


class FormatDetector:
    """Utility class for detecting data formats using regex patterns."""

    def __init__(self, custom_patterns: dict[FormatType, PatternInfo] | None = None):
        """
        Initialize the detector with optional custom patterns.

        Args:
            custom_patterns: Additional patterns to register beyond defaults
        """
        self._compiled_patterns: dict[FormatType, re.Pattern] = {}
        self._load_patterns(custom_patterns)

    def _load_patterns(self, custom_patterns: dict[FormatType, PatternInfo] | None = None):
        """Load and compile all patterns."""
        all_patterns = RegexPatterns.get_all_patterns()

        if custom_patterns:
            all_patterns.update(custom_patterns)

        for format_type, pattern_info in all_patterns.items():
            flags = 0 if pattern_info.case_sensitive else re.IGNORECASE
            try:
                self._compiled_patterns[format_type] = re.compile(pattern_info.pattern, flags)
            except re.error as e:
                print(f"Warning: Invalid regex pattern for {format_type}: {e}")

    def detect_format(self, text: str) -> tuple[FormatType | None, str | None]:
        """
        Detect the format of input text.

        Args:
            text: The text to analyze

        Returns:
            Tuple of (detected_format_type, pattern_string) or (None, None) if no match
        """
        # First check for whitespace patterns without stripping
        whitespace_patterns = [
            FormatType.LEADING_SPACE_DIGITS,
            FormatType.LEADING_SPACE_ALPHANUMERIC,
            FormatType.TRAILING_SPACE_DIGITS,
            FormatType.TRAILING_SPACE_ALPHANUMERIC,
        ]

        for format_type in whitespace_patterns:
            compiled_pattern = self._compiled_patterns.get(format_type)
            if compiled_pattern and compiled_pattern.match(text):
                pattern_info = RegexPatterns.get_pattern(format_type)
                return format_type, pattern_info.pattern if pattern_info else None

        # Then check other patterns with stripped text
        text_stripped = text.strip()

        for format_type, compiled_pattern in self._compiled_patterns.items():
            # Skip whitespace patterns as they were already checked
            if format_type in whitespace_patterns:
                continue

            if compiled_pattern.match(text_stripped):
                pattern_info = RegexPatterns.get_pattern(format_type)
                return format_type, pattern_info.pattern if pattern_info else None

        return None, None

    def detect_all_matches(self, text: str) -> list[tuple[FormatType, str]]:
        """
        Detect all formats that match the input text.

        Args:
            text: The text to analyze

        Returns:
            List of (format_type, pattern_string) tuples for all matches
        """
        text = text.strip()
        matches = []

        for format_type, compiled_pattern in self._compiled_patterns.items():
            if compiled_pattern.match(text):
                pattern_info = RegexPatterns.get_pattern(format_type)
                if pattern_info:
                    matches.append((format_type, pattern_info.pattern))

        return matches

    def validate_format(self, text: str, expected_format: FormatType) -> bool:
        """
        Validate that text matches a specific format.

        Args:
            text: The text to validate
            expected_format: The expected format type

        Returns:
            True if text matches the expected format
        """
        compiled_pattern = self._compiled_patterns.get(expected_format)
        if not compiled_pattern:
            return False

        return bool(compiled_pattern.match(text.strip()))

    def extract_matches(self, text: str, format_type: FormatType) -> list[str]:
        """
        Extract all occurrences of a specific format from text.

        Args:
            text: The text to search
            format_type: The format type to find

        Returns:
            List of matched strings
        """
        compiled_pattern = self._compiled_patterns.get(format_type)
        if not compiled_pattern:
            return []

        return compiled_pattern.findall(text)

    def get_pattern_info(self, format_type: FormatType) -> PatternInfo | None:
        """Get information about a specific pattern.

        Args:
            format_type: The format type to look up.

        Returns:
            The PatternInfo for the format type, or None if not registered.
        """
        return RegexPatterns.get_pattern(format_type)

    def add_custom_pattern(self, format_type: FormatType, pattern_info: PatternInfo) -> None:
        """Add a custom pattern to this detector instance.

        Registers the pattern in the global registry and compiles it for
        use by this detector.

        Args:
            format_type: The format type key.
            pattern_info: The pattern definition to register and compile.
        """
        RegexPatterns.add_pattern(format_type, pattern_info)
        flags = 0 if pattern_info.case_sensitive else re.IGNORECASE
        try:
            self._compiled_patterns[format_type] = re.compile(pattern_info.pattern, flags)
        except re.error as e:
            print(f"Warning: Invalid regex pattern for {format_type}: {e}")


# Convenience functions for common operations
def detect_format(text: str) -> tuple[FormatType | None, str | None]:
    """
    Convenience function to detect format using default detector.

    Args:
        text: The text to analyze

    Returns:
        Tuple of (detected_format_type, pattern_string) or (None, None) if no match
    """
    detector = FormatDetector()
    return detector.detect_format(text)


def validate_format(text: str, expected_format: FormatType) -> bool:
    """
    Convenience function to validate format using default detector.

    Args:
        text: The text to validate
        expected_format: The expected format type

    Returns:
        True if text matches the expected format
    """
    detector = FormatDetector()
    return detector.validate_format(text, expected_format)


def get_pattern_for_format(format_type: FormatType) -> str | None:
    """
    Convenience function to get pattern string for a format.

    Args:
        format_type: The format type

    Returns:
        Pattern string or None if format not found
    """
    return RegexPatterns.get_pattern_string(format_type)


# Create a default global detector instance
_default_detector = FormatDetector()


def get_default_detector() -> FormatDetector:
    """Get the default global detector instance."""
    return _default_detector
