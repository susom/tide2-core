"""Regenerate tests/fixtures/fpe_relaxed_int64_golden.json (not collected by pytest).

Run only when relaxed pseudonyms are meant to change:
`uv run python tests/fixtures/make_fpe_relaxed_int64_golden.py`.
"""

import json
import random
from pathlib import Path

from tide2.cryptographic.fpe_strings import INT64_MAX
from tide2.cryptographic.fpe_strings import INT64_MIN
from tide2.cryptographic.fpe_strings import FormatPreservingEncryption

SALT = b"\x01" * 32
KEY = b"\x02" * 32
OUTPUT = Path(__file__).with_name("fpe_relaxed_int64_golden.json")


def build_inputs() -> list[str]:
    rng = random.Random(12)  # noqa: S311
    edges = [0, 1, -1, 2, -2, INT64_MAX, INT64_MAX - 1, INT64_MIN, INT64_MIN + 1]
    sample = list(range(-2000, 2001, 37))
    digit_widths = [rng.randint(10 ** (n - 1), 10**n - 1) for n in range(6, 20)]
    negatives = [-v for v in digit_widths]
    ints = [str(v) for v in [*edges, *sample, *digit_widths, *negatives]]
    ff1_path = ["0123", "-0", "123-45-6789", "MRN-12345678", "9223372036854775808", "-9223372036854775809"]
    return ints + ff1_path


def main() -> None:
    fpe = FormatPreservingEncryption(SALT, KEY, "relaxed")
    cases = [{"input": text, "output": fpe.encrypt(text)[0]} for text in build_inputs()]
    document = {"salt_hex": SALT.hex(), "key_hex": KEY.hex(), "cases": cases}
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
