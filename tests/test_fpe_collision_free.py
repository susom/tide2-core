"""Tests for the collision-free FF1 format-preserving encryption in tide2.cryptographic.fpe_strings."""

import importlib.util
import itertools
import json
import os
import random
import string
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastfpe import ff1

from tide2.anonymizers import cell
from tide2.cryptographic import fpe_strings
from tide2.cryptographic.fpe_strings import FormatPreservingEncryption

SALT = b"\x01" * 32
KEY = b"\x02" * 32
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "fpe_ff1_golden.json"
SRC = Path(__file__).resolve().parents[1] / "src" / "tide2"

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

_K128 = "2B7E151628AED2A6ABF7158809CF4F3C"
_K192 = _K128 + "EF4359D8D580AA4F"
_K256 = _K192 + "7F036D6F04FC6A94"
_D10 = "0123456789"
_D36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_T10 = "39383736353433323130"
_T36 = "3737373770717273373737"

# NIST SP 800-38G FF1 samples 1-9: (key, tweak, alphabet, plaintext, ciphertext)
NIST_FF1_SAMPLES = [
    (_K128, "", _D10, "0123456789", "2433477484"),
    (_K128, _T10, _D10, "0123456789", "6124200773"),
    (_K128, _T36, _D36, "0123456789abcdefghi", "a9tv40mll9kdu509eum"),
    (_K192, "", _D10, "0123456789", "2830668132"),
    (_K192, _T10, _D10, "0123456789", "2496655549"),
    (_K192, _T36, _D36, "0123456789abcdefghi", "xbj3kv35jrawxv32ysr"),
    (_K256, "", _D10, "0123456789", "6657667009"),
    (_K256, _T10, _D10, "0123456789", "1001623463"),
    (_K256, _T36, _D36, "0123456789abcdefghi", "xs8a0azh2avyalyzuwd"),
]


@pytest.fixture(scope="module")
def fpe() -> FormatPreservingEncryption:
    return FormatPreservingEncryption(SALT, KEY)


def _klass(content: str) -> str:
    if not content.isdigit():
        return "A"
    if content == "0":
        return "Z"
    return "S" if content[0] == "0" else "N"


def _assert_permutation(fpe: FormatPreservingEncryption, inputs: list[str]) -> None:
    """Assert one-to-one, class and length kept, and round trip for distinct plain inputs."""
    assert len(set(inputs)) == len(inputs)
    outputs = [fpe.encrypt(text)[0] for text in inputs]
    assert len(set(outputs)) == len(inputs), "two inputs share an output"
    assert all(len(o) == len(i) and _klass(o) == _klass(i) for i, o in zip(inputs, outputs, strict=True))
    assert [fpe.decrypt(o) for o in outputs] == inputs


def _digit_strings(max_length: int) -> list[str]:
    return ["".join(p) for n in range(1, max_length + 1) for p in itertools.product(string.digits, repeat=n)]


def _alnum_strings(max_length: int) -> list[str]:
    alphabet = FormatPreservingEncryption.ALPHABET_ALPHANUMERIC
    return [
        s
        for n in range(1, max_length + 1)
        for s in map("".join, itertools.product(alphabet, repeat=n))
        if not s.isdigit()
    ]


def test_ff1_nist_vectors() -> None:
    for key, tweak, alphabet, plaintext, ciphertext in NIST_FF1_SAMPLES:
        assert ff1.encrypt(key, tweak, alphabet, plaintext) == ciphertext
        assert ff1.decrypt(key, tweak, alphabet, ciphertext) == plaintext


def test_fpe_golden(fpe: FormatPreservingEncryption) -> None:
    golden = json.loads(GOLDEN_PATH.read_text())
    assert bytes.fromhex(golden["salt_hex"]) == SALT
    assert bytes.fromhex(golden["key_hex"]) == KEY
    assert golden["cases"]
    for case in golden["cases"]:
        assert fpe.encrypt(case["input"])[0] == case["output"], case["input"]


def test_fpe_cf_small_domains(fpe: FormatPreservingEncryption) -> None:
    _assert_permutation(fpe, _digit_strings(4))
    _assert_permutation(fpe, _alnum_strings(2))


@pytest.mark.slow
def test_fpe_cf_exhaustive_digits(fpe: FormatPreservingEncryption) -> None:
    inputs = _digit_strings(6)
    assert len(inputs) == 1_111_110
    _assert_permutation(fpe, inputs)


@pytest.mark.slow
def test_fpe_cf_exhaustive_alnum(fpe: FormatPreservingEncryption) -> None:
    inputs = _alnum_strings(3)
    assert len(inputs) == 241_124
    _assert_permutation(fpe, inputs)


def test_fpe_cf_samples(fpe: FormatPreservingEncryption) -> None:
    rng = random.Random(4)  # noqa: S311
    alphabet = FormatPreservingEncryption.ALPHABET_ALPHANUMERIC

    def alnum(n: int) -> str:
        while True:
            s = "".join(rng.choice(alphabet) for _ in range(n))
            if not s.isdigit():
                return s

    def digits(n: int, leading_zero: bool) -> str:
        first = "0" if leading_zero else rng.choice("123456789")
        return first + "".join(rng.choice(string.digits) for _ in range(n - 1))

    below = [str(rng.randint(10**18, INT64_MAX)) for _ in range(3000)] + [str(INT64_MAX), "1000000000000000000"]
    above = [str(rng.randint(INT64_MAX + 1, 10**19 - 1)) for _ in range(3000)] + [
        str(INT64_MAX + 1),
        "9999999999999999999",
    ]
    groups = {
        "alnum_4_to_8": list({alnum(n) for n in range(4, 9) for _ in range(1500)}),
        "int64_below": below,
        "int64_above": above,
        "long_digits": list({digits(n, False) for n in range(60, 201, 5) for _ in range(20)}),
        "long_alnum": list({alnum(n) for n in range(40, 121, 5) for _ in range(20)}),
        "zero_led": list({digits(n, True) for n in range(6, 21) for _ in range(100)}),
    }
    for inputs in groups.values():
        _assert_permutation(fpe, inputs)

    assert all(int(fpe.encrypt(t)[0]) <= INT64_MAX for t in below)
    assert all(int(fpe.encrypt(t)[0]) > INT64_MAX for t in above)
    assert all(not fpe.encrypt(t)[0].startswith("0") for t in groups["long_digits"])
    assert all(any(c.isalpha() for c in fpe.encrypt(t)[0]) for t in groups["long_alnum"])


def test_fpe_cf_separators(fpe: FormatPreservingEncryption) -> None:
    inputs = [
        "(555) 123-4567",
        "123-45-6789",
        "MRN-12345678",
        "AB-9021-X7",
        "1 2 3 4 5 6 7",
        "+1.234.567.8901",
        "a/b/c/d/e/f",
        "007-00-7007",
    ]
    outputs = [fpe.encrypt(t)[0] for t in inputs]

    assert len(set(outputs)) == len(inputs)
    assert [fpe.decrypt(o) for o in outputs] == inputs
    for plain, enc in zip(inputs, outputs, strict=True):
        assert [(i, c) for i, c in enumerate(plain) if not c.isalnum()] == [
            (i, c) for i, c in enumerate(enc) if not c.isalnum()
        ]


def test_fpe_cf_zero_is_a_fixed_point(fpe: FormatPreservingEncryption) -> None:
    assert fpe.encrypt("0")[0] == "0"
    assert fpe.encrypt("-0")[0] == "-0"
    assert fpe.decrypt("-0") == "-0"


def test_fpe_cf_no_length_limit(fpe: FormatPreservingEncryption) -> None:
    for text in ("7" * 5000, "Ab" * 2500):
        encrypted, _ = fpe.encrypt(text)
        assert len(encrypted) == len(text)
        assert fpe.decrypt(encrypted) == text


def test_fpe_cf_int64(fpe: FormatPreservingEncryption) -> None:
    rng = random.Random(11)  # noqa: S311
    extremes = [INT64_MIN, INT64_MIN + 1, -1, 0, 1, INT64_MAX - 1, INT64_MAX]
    values = sorted(set(range(-2000, 2001)) | set(extremes) | {rng.randint(INT64_MIN, INT64_MAX) for _ in range(5000)})

    outputs = [fpe.encrypt_int(v) for v in values]

    assert len(set(outputs)) == len(values)
    assert all(INT64_MIN <= o <= INT64_MAX for o in outputs)
    assert all(
        (v > 0) == (o > 0) and (v < 0) == (o < 0) and (v == 0) == (o == 0) for v, o in zip(values, outputs, strict=True)
    )
    assert all(len(str(abs(v))) == len(str(abs(o))) for v, o in zip(values, outputs, strict=True))
    assert [fpe.decrypt_int(o) for o in outputs] == values
    assert all(o == int(fpe.encrypt(str(v))[0]) for v, o in zip(values, outputs, strict=True) if v != INT64_MIN)
    assert fpe.encrypt_int(INT64_MIN) == INT64_MIN
    assert fpe.decrypt_int(INT64_MIN) == INT64_MIN


@pytest.mark.parametrize("bad", [True, 1.5, "12", None])
def test_fpe_cf_int64_rejects_non_int(fpe: FormatPreservingEncryption, bad: Any) -> None:
    with pytest.raises(TypeError):
        fpe.encrypt_int(bad)
    with pytest.raises(TypeError):
        fpe.decrypt_int(bad)


@pytest.mark.parametrize("bad", [INT64_MAX + 1, INT64_MIN - 1])
def test_fpe_cf_int64_rejects_out_of_range(fpe: FormatPreservingEncryption, bad: int) -> None:
    with pytest.raises(ValueError, match="INT64"):
        fpe.encrypt_int(bad)
    with pytest.raises(ValueError, match="INT64"):
        fpe.decrypt_int(bad)


@pytest.mark.parametrize("value", ["48291037", "7", "42", "00482910", "AB-9021-X7", "(555) 014-8821", "900-12-3456"])
def test_fpe_cf_entity_type_independent(value: str) -> None:
    outputs = {entity_type: cell.anonymize_cell(value, entity_type, SALT, KEY) for entity_type in cell.FPE_ENTITY_TYPES}
    assert len(set(outputs.values())) == 1, outputs


def test_fpe_cf_cycle_walking_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    class StuckCipher:
        @staticmethod
        def encrypt(_key: str, _tweak: str, _alphabet: str, text: str) -> str:
            return "0" * len(text)

    monkeypatch.setattr(fpe_strings, "ff1", StuckCipher)
    stuck = FormatPreservingEncryption(SALT, KEY)
    stuck.MAX_WALK_STEPS = 5
    with pytest.raises(RuntimeError, match="did not terminate"):
        stuck.encrypt("1234567")


_DETERMINISM_CHILD = """
import json
from tide2.cryptographic.fpe_strings import FormatPreservingEncryption

fpe = FormatPreservingEncryption(b"\\x01" * 32, b"\\x02" * 32)
values = ["7", "42", "901", "12345", "123456", "0123456", "ab1", "AB-9021-X7", "(555) 014-8821", "9223372036854775808"]
print(json.dumps([fpe.encrypt(v)[0] for v in values]))
"""


def test_fpe_cf_determinism_across_hash_seeds() -> None:
    results = []
    for seed in ("1", "7"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _DETERMINISM_CHILD], capture_output=True, text=True, env=env, check=False
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert results[0] == results[1]


def test_fpe_cf_determinism_across_threads() -> None:
    values = _digit_strings(4)[::50] + _alnum_strings(2)[::50] + [f"{n}" for n in range(100_000, 100_100)]
    expected = [FormatPreservingEncryption(SALT, KEY).encrypt(v)[0] for v in values]

    shared = FormatPreservingEncryption(SALT, KEY)
    barrier = threading.Barrier(32)

    def run(_: int) -> list[str]:
        barrier.wait()
        return [shared.encrypt(v)[0] for v in values]

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(run, range(32)))
    assert all(r == expected for r in results)


def test_ff3_removed() -> None:
    assert importlib.util.find_spec("ff3") is None
    assert "ff3" not in sys.modules
    offenders = [
        str(path)
        for path in SRC.rglob("*.py")
        if any(line.strip().startswith(("import ff3", "from ff3")) for line in path.read_text().splitlines())
    ]
    assert offenders == []
    assert not hasattr(FormatPreservingEncryption, "suppress_short_input_warnings")
