"""Key generation, serialization, and loading utilities.

Provides functions for creating, saving, and loading cryptographic keys
used by the FPE and HMAC-based anonymization components.
"""

import hashlib
import os
import secrets
import stat


def generate_salt() -> bytes:
    """
    Generate a cryptographically secure 256-bit random salt.

    Returns:
        bytes: A 32-byte random salt
    """
    return secrets.token_bytes(32)


def derive_key(input_string: str) -> bytes:
    """
    Derive a deterministic 32-byte key from an input string.

    Args:
        input_string (str): The input string to derive the key from

    Returns:
        bytes: A 32-byte derived key
    """
    if not isinstance(input_string, str):
        raise TypeError("Input string must be of type str")

    # Use SHA-256 to derive a deterministic key from the input string
    return hashlib.sha256(input_string.encode("utf-8")).digest()


def save_salt(salt: bytes, filename: str) -> None:
    """
    Save a salt to a file in hexadecimal text format with restricted permissions.

    Args:
        salt (bytes): 32-byte salt to save
        filename (str): Path to the file where the salt will be saved

    Raises:
        ValueError: If salt is invalid
        IOError: If file cannot be written
    """
    if not isinstance(salt, bytes) or len(salt) != 32:
        raise ValueError("Salt must be 32 bytes")

    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")

    try:
        # Convert salt to hexadecimal string and write to file in text mode
        with open(filename, "w", encoding="utf-8") as f:
            f.write(salt.hex())

        # Set restrictive permissions (owner read/write only) for security
        os.chmod(filename, stat.S_IRUSR | stat.S_IWUSR)

    except Exception as e:
        raise OSError(f"Failed to save salt to {filename}: {e!s}")


def save_key(key: bytes, filename: str) -> None:
    """
    Save a key to a file in hexadecimal text format.

    Args:
        key (bytes): 32-byte key to save
        filename (str): Path to the file where the key will be saved

    Raises:
        ValueError: If key is invalid
        IOError: If file cannot be written
    """
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("Key must be 32 bytes")

    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")

    try:
        # Convert key to hexadecimal string and write to file in text mode
        with open(filename, "w", encoding="utf-8") as f:
            f.write(key.hex())

        # Set normal file permissions for key
        os.chmod(filename, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    except Exception as e:
        raise OSError(f"Failed to save key to {filename}: {e!s}")


def load_salt(filename: str) -> bytes:
    """
    Load a salt from a file in hexadecimal text format.

    Args:
        filename (str): Path to the file containing the salt

    Returns:
        bytes: The 32-byte salt

    Raises:
        ValueError: If the loaded salt is invalid
        IOError: If file cannot be read
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")

    if not os.path.exists(filename):
        raise OSError(f"Salt file {filename} does not exist")

    try:
        with open(filename, encoding="utf-8") as f:
            hex_key = f.read().strip()

        # Convert hexadecimal string back to bytes
        salt = bytes.fromhex(hex_key)

        # Validate the loaded salt
        if not isinstance(salt, bytes) or len(salt) != 32:
            raise ValueError(f"Invalid salt in file {filename}: must be 32 bytes")

        return salt

    except ValueError as e:
        if "non-hexadecimal" in str(e):
            raise ValueError(f"Invalid hexadecimal format in salt file {filename}")
        raise e
    except Exception as e:
        raise OSError(f"Failed to load salt from {filename}: {e!s}")


def load_key(filename: str) -> bytes:
    """
    Load a key from a file in hexadecimal text format.

    Args:
        filename (str): Path to the file containing the key

    Returns:
        bytes: The 32-byte key

    Raises:
        ValueError: If the loaded key is invalid
        IOError: If file cannot be read
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string")

    if not os.path.exists(filename):
        raise OSError(f"Key file {filename} does not exist")

    try:
        with open(filename, encoding="utf-8") as f:
            hex_key = f.read().strip()

        # Convert hexadecimal string back to bytes
        key = bytes.fromhex(hex_key)

        # Validate the loaded key
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError(f"Invalid key in file {filename}: must be 32 bytes")

        return key

    except ValueError as e:
        if "non-hexadecimal" in str(e):
            raise ValueError(f"Invalid hexadecimal format in key file {filename}")
        raise e
    except Exception as e:
        raise OSError(f"Failed to load key from {filename}: {e!s}")


def key_from_hex_string(hex_string: str) -> bytes:
    """
    Convert a hexadecimal string to a 32-byte key.

    Args:
        hex_string: 64-character hexadecimal string representing a 32-byte key

    Returns:
        bytes: The 32-byte key

    Raises:
        ValueError: If the hex string is invalid or wrong length
    """
    if not isinstance(hex_string, str):
        raise TypeError("hex_string must be a string")

    hex_string = hex_string.strip()

    try:
        key = bytes.fromhex(hex_string)
    except ValueError as e:
        raise ValueError(f"Invalid hexadecimal string: {e}") from e

    if len(key) != 32:
        raise ValueError(f"Key must be 32 bytes (64 hex characters), got {len(key)} bytes")

    return key
