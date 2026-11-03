"""Tests for the relaxed (keyed FarmHash Feistel) FPE mode for INT64 values."""

import json
import os
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from farmhash import Fingerprint128  # ty: ignore[unresolved-import]

from tide2.anonymizers import cell
from tide2.anonymizers.hips_alphanumeric import HipsAlphaNumericAnonymizer
from tide2.cryptographic.fpe_strings import INT64_MAX
from tide2.cryptographic.fpe_strings import INT64_MIN
from tide2.cryptographic.fpe_strings import FormatPreservingEncryption

SALT = b"\x01" * 32
KEY = b"\x02" * 32
FIXTURES = Path(__file__).parent / "fixtures"
EDGES = [INT64_MIN, INT64_MIN + 1, -2, -1, 0, 1, 2, INT64_MAX - 1, INT64_MAX]


@pytest.fixture(scope="module")
def relaxed() -> FormatPreservingEncryption:
    return FormatPreservingEncryption(SALT, KEY, "relaxed")


@pytest.fixture(scope="module")
def strict() -> FormatPreservingEncryption:
    return FormatPreservingEncryption(SALT, KEY, "strict")


def _random_int64(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)  # noqa: S311
    return [rng.randint(INT64_MIN, INT64_MAX) for _ in range(count)]


def test_relaxed_golden(relaxed: FormatPreservingEncryption) -> None:
    golden = json.loads((FIXTURES / "fpe_relaxed_int64_golden.json").read_text())
    assert bytes.fromhex(golden["salt_hex"]) == SALT
    assert bytes.fromhex(golden["key_hex"]) == KEY
    assert golden["cases"]
    for case in golden["cases"]:
        assert relaxed.encrypt(case["input"])[0] == case["output"], case["input"]


def test_strict_unchanged(strict: FormatPreservingEncryption) -> None:
    golden = json.loads((FIXTURES / "fpe_ff1_golden.json").read_text())
    operator = HipsAlphaNumericAnonymizer()
    for case in golden["cases"]:
        assert strict.encrypt(case["input"])[0] == case["output"], case["input"]
        assert operator.operate(case["input"], {"salt": SALT, "key": KEY}) == case["output"], case["input"]


def test_relaxed_exhaustive_small_width(relaxed: FormatPreservingEncryption) -> None:
    """A 22-bit signed domain (21-bit magnitude, 10/11 split) is a permutation that keeps the sign."""
    width = 21
    low = -(1 << width)
    size = 1 << (width + 1)
    seen = bytearray(size)
    outputs = []
    for v in range(low, low + size):
        o = relaxed._relaxed_int64(v, width=width)
        assert low <= o < low + size
        assert (v >= 0) == (o >= 0)
        assert not seen[o - low]
        seen[o - low] = 1
        outputs.append(o)
    domain = range(low, low + size)
    assert all(relaxed._relaxed_int64(o, decrypt=True, width=width) == v for v, o in zip(domain, outputs, strict=True))


def test_relaxed_int64_samples(relaxed: FormatPreservingEncryption) -> None:
    values = sorted({*EDGES, *range(-2000, 2001), *_random_int64(200_000, 3)})
    outputs = [relaxed.encrypt_int(v) for v in values]

    assert len(set(outputs)) == len(values)
    assert all(INT64_MIN <= o <= INT64_MAX for o in outputs)
    assert all((v >= 0) == (o >= 0) for v, o in zip(values, outputs, strict=True))
    assert [relaxed.decrypt_int(o) for o in outputs] == values


def test_relaxed_sequential_ids(relaxed: FormatPreservingEncryption) -> None:
    outputs = [relaxed.encrypt_int(v) for v in range(1_000_000)]
    assert len(set(outputs)) == 1_000_000
    assert min(outputs) >= 0
    assert all(str(o)[0] != "0" for o in outputs if o)


def test_sign_halves_independent(relaxed: FormatPreservingEncryption) -> None:
    values = [*range(0, 500), *_random_int64(500, 4)]
    non_negative = [abs(v) if v != INT64_MIN else 0 for v in values]
    assert all(relaxed.encrypt_int(-1 - v) != -1 - relaxed.encrypt_int(v) for v in non_negative)


@pytest.mark.parametrize(
    "text",
    ["0", "1", "123", "-5", "123456", str(INT64_MAX), str(INT64_MIN), "9182736455463728191", "-9182736455463728191"],
)
def test_relaxed_eligible(relaxed: FormatPreservingEncryption, text: str) -> None:
    out, format_type = relaxed.encrypt(text)
    assert format_type == FormatPreservingEncryption.FORMAT_INT64
    assert out == str(relaxed.encrypt_int(int(text)))
    assert relaxed.decrypt(out) == text


@pytest.mark.parametrize(
    "text",
    [
        "0123",
        "-0",
        "00",
        "123-45",
        "12a",
        " 12",
        "+12",
        str(INT64_MAX + 1),
        str(INT64_MIN - 1),
        "12345678901234567890",
        "-12345678901234567890",
        "1" * 5000,
    ],
)
def test_ff1_path_for_values_that_are_not_an_int64(
    relaxed: FormatPreservingEncryption, strict: FormatPreservingEncryption, text: str
) -> None:
    out, format_type = relaxed.encrypt(text)
    assert format_type != FormatPreservingEncryption.FORMAT_INT64
    assert (out, format_type) == strict.encrypt(text)
    assert relaxed.decrypt(out) == text


def test_relaxed_cross_path_injective(relaxed: FormatPreservingEncryption) -> None:
    rng = random.Random(9)  # noqa: S311
    eligible = {str(v) for v in [*EDGES, *range(-300, 300), *_random_int64(3000, 5)]}
    others = {
        *(f"0{rng.randint(1, 10**9)}" for _ in range(500)),
        *(str(rng.randint(INT64_MAX + 1, 10**20)) for _ in range(500)),
        *(str(-rng.randint(-INT64_MIN + 1, 10**20)) for _ in range(500)),
        *(f"{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}" for _ in range(500)),
        *(f"AB{rng.randint(1, 10**8)}" for _ in range(500)),
        "-0",
        "0000",
        str(-INT64_MIN),
    }
    assert not eligible & others
    inputs = sorted(eligible | others)
    outputs = [relaxed.encrypt(t)[0] for t in inputs]

    assert len(set(outputs)) == len(inputs)
    assert [relaxed.decrypt(o) for o in outputs] == inputs
    other_outputs = {relaxed.encrypt(t)[0] for t in others}
    eligible_outputs = {relaxed.encrypt(t)[0] for t in eligible}
    assert all(relaxed.encrypt(o)[1] != FormatPreservingEncryption.FORMAT_INT64 for o in other_outputs)
    assert all(relaxed.encrypt(o)[1] == FormatPreservingEncryption.FORMAT_INT64 for o in eligible_outputs)


@pytest.mark.parametrize("value", [*EDGES, 48291037, -48291037, 604821])
def test_relaxed_int_string_parity(value: int) -> None:
    from_int = cell.anonymize_cell(value, "MRN", SALT, KEY)
    assert from_int == cell.anonymize_cell(str(value), "MRN", SALT, KEY)
    assert from_int == str(FormatPreservingEncryption(SALT, KEY, "relaxed").encrypt_int(value))


def test_relaxed_keyed(relaxed: FormatPreservingEncryption) -> None:
    other_key = FormatPreservingEncryption(SALT, b"\x03" * 32, "relaxed")
    other_salt = FormatPreservingEncryption(b"\x04" * 32, KEY, "relaxed")
    values = [1, 42, 604821, 48291037, -7, INT64_MAX]
    base = [relaxed.encrypt_int(v) for v in values]
    assert base != [other_key.encrypt_int(v) for v in values]
    assert base != [other_salt.encrypt_int(v) for v in values]
    assert all(relaxed.encrypt_int(v) != Fingerprint128(str(v).encode()) & (2**63 - 1) for v in values if v >= 0)


def test_relaxed_differs_from_strict(relaxed: FormatPreservingEncryption, strict: FormatPreservingEncryption) -> None:
    assert relaxed.encrypt("48291037") != strict.encrypt("48291037")


SPANS = [{"entity_type": "MRN", "start": 4, "end": 12, "score": 1.0}]


def test_notes_stay_strict(strict: FormatPreservingEncryption) -> None:
    out = cell.anonymize_cell("MRN 48291037 seen", "FREE_TEXT", SALT, KEY, recognizer_results=SPANS, fpe_mode="relaxed")
    assert out == f"MRN {strict.encrypt('48291037')[0]} seen"


def test_fpe_mode_validation() -> None:
    with pytest.raises(ValueError, match="Unknown FPE mode"):
        FormatPreservingEncryption(SALT, KEY, "fast")
    with pytest.raises(ValueError, match="Unknown fpe_mode"):
        HipsAlphaNumericAnonymizer().validate({"entity_type": "MRN", "salt": SALT, "key": KEY, "fpe_mode": "fast"})
    assert cell.anonymize_cell("123456", "MRN", SALT, KEY, fpe_mode="fast") == "[MRN]"


_DETERMINISM_CHILD = """
import json
from tide2.cryptographic.fpe_strings import FormatPreservingEncryption

fpe = FormatPreservingEncryption(b"\\x01" * 32, b"\\x02" * 32, "relaxed")
values = ["0", "7", "-42", "123456", "48291037", "9223372036854775807", "-9223372036854775808", "0123", "AB-9021-X7"]
print(json.dumps([fpe.encrypt(v)[0] for v in values]))
"""


def test_relaxed_determinism_across_hash_seeds() -> None:
    results = []
    for seed in ("1", "7"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _DETERMINISM_CHILD], capture_output=True, text=True, env=env, check=False
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert results[0] == results[1]


def test_relaxed_determinism_across_threads() -> None:
    values = [*EDGES, *_random_int64(300, 6), *range(100_000, 100_100)]
    expected = [FormatPreservingEncryption(SALT, KEY, "relaxed").encrypt_int(v) for v in values]

    shared = FormatPreservingEncryption(SALT, KEY, "relaxed")
    barrier = threading.Barrier(32)

    def run(_: int) -> list[int]:
        barrier.wait()
        return [shared.encrypt_int(v) for v in values]

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(run, range(32)))
    assert all(r == expected for r in results)
