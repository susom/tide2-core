"""
Secure String Selector - Cryptographically secure deterministic string selection

Optimized version using SHA256 for fast, deterministic string selection.
Uses LRU caching for performance with worker-local cache safety for Ray.
"""

import hashlib
from functools import lru_cache


@lru_cache(maxsize=16)
def _get_combined_key(salt: bytes, key: bytes) -> bytes:
    """
    Cache the combined key derivation.

    Args:
        salt: 32-byte salt
        key: 32-byte key

    Returns:
        Combined key bytes for hashing
    """
    return salt + key


@lru_cache(maxsize=10000)
def _cached_select_index(combined_key: bytes, input_string: str, list_len: int) -> int:
    """
    Cached index selection using SHA256.

    Args:
        combined_key: Combined salt + key
        input_string: Input string to hash
        list_len: Length of the target list

    Returns:
        Selected index into the list
    """
    # Use SHA256 for fast, deterministic index selection
    digest = hashlib.sha256(combined_key + input_string.encode("utf-8")).digest()
    # Use first 8 bytes for index calculation (provides 2^64 range)
    index_value = int.from_bytes(digest[:8], byteorder="big")
    return index_value % list_len


def secure_string_selector(salt: bytes, key: bytes, string_list: list[str], input_string: str) -> str:
    """
    Securely and deterministically select a string from a list using SHA256 hashing.

    This function provides deterministic, cryptographically-derived string selection.
    The same inputs will always produce the same output, but the selection cannot
    be predicted without knowing the keys.

    Args:
        salt: 32-byte salt
        key: 32-byte key (derived from input)
        string_list: List of strings to select from (must be non-empty)
        input_string: Input string for deterministic selection (will be converted to str)

    Returns:
        The selected string from the list

    Note:
        This function uses LRU caching for performance. The cache is process-local,
        which is appropriate for Ray worker-local usage patterns.
    """
    # Convert input to string if not already (defensive, avoids validation overhead)
    input_str = str(input_string) if not isinstance(input_string, str) else input_string

    # Get cached combined key
    combined_key = _get_combined_key(salt, key)

    # Get cached index selection
    list_len = len(string_list)
    if list_len == 0:
        raise ValueError("String list must be non-empty")

    selected_index = _cached_select_index(combined_key, input_str, list_len)

    return string_list[selected_index]


def clear_selector_cache() -> None:
    """
    Clear the string selector caches.

    Call this when changing keys or when memory pressure is a concern.
    Useful for testing or when switching between different key sets.
    """
    _get_combined_key.cache_clear()
    _cached_select_index.cache_clear()


def get_cache_info() -> dict:
    """
    Get cache statistics for monitoring.

    Returns:
        Dictionary with cache info for both internal caches
    """
    return {
        "combined_key_cache": _get_combined_key.cache_info()._asdict(),
        "index_cache": _cached_select_index.cache_info()._asdict(),
    }
