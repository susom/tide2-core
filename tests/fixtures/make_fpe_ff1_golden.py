"""Regenerate tests/fixtures/fpe_ff1_golden.json (not collected by pytest).

Run only when pseudonyms are meant to change: `uv run python tests/fixtures/make_fpe_ff1_golden.py`.
"""

import json
from pathlib import Path

from tide2.cryptographic.fpe_strings import FormatPreservingEncryption

SALT = b"\x01" * 32
KEY = b"\x02" * 32
OUTPUT = Path(__file__).with_name("fpe_ff1_golden.json")

_DIGITS = "1234567890" * 6
_ALNUM = "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE3"


def build_inputs() -> list[str]:
    numeric = [_DIGITS[:n] for n in (*range(1, 9), 19, 57)]
    alphanumeric = [_ALNUM[:n] for n in (*range(1, 7), 33)]
    zero_led = ["0", "00", "0123", "012345", "0123456789"]
    separators = [
        "(555) 123-4567",
        "123-45-6789",
        "MRN-12345678",
        "AB-9021-X7",
        "-123456",
        "-0",
        "9223372036854775808",
        "-9223372036854775808",
        "9223372036854775807",
        "9223372036854775809",
    ]
    return numeric + alphanumeric + zero_led + separators


def main() -> None:
    fpe = FormatPreservingEncryption(SALT, KEY)
    cases = [{"input": text, "output": fpe.encrypt(text)[0]} for text in build_inputs()]
    document = {"salt_hex": SALT.hex(), "key_hex": KEY.hex(), "cases": cases}
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
