# FPE collision-freeness: tests and verification results

This guide records how the format-preserving encryption (FPE) in
`tide2.cryptographic.fpe_strings` is shown to be collision-free, including for
`INT64` values, which tests cover it, and the results of an independent check.
Use it to re-run the check after any change to `fpe_strings.py`, the key
derivation, or the `fastfpe` dependency.

## What is guaranteed

Two different inputs must never produce the same pseudonym, because a collision
silently merges two people or two records in the de-identified data.

`FormatPreservingEncryption` keeps the non-alphanumeric characters (the
template) where they are and permutes only the alphanumeric content. Each piece
of content belongs to one class, and the mapping is a permutation of each
(class, length) set:

| Class | Content | Notes |
|---|---|---|
| `N` | digits, no leading zero | The output never starts with `0`, so `int()` is safe. At 19 digits, values up to `2**63 - 1` form `N`; values above `2**63` form the separate class `H`. |
| `M` | exactly `9223372036854775808` (`2**63`) | Maps to itself, so `-9223372036854775808` (INT64 minimum) stays in range as a `STRING` and as an `INT64`. No other value maps onto it. |
| `Z` | exactly `0` | Maps to itself, so a negative value never becomes `-0`. |
| `S` | two or more digits, starting with `0` | |
| `A` | content that contains a letter | The output contains a letter. |

Content shorter than FF1's minimum domain (fewer than 6 digits or 4
alphanumeric characters) is permuted with keyed tables. Longer content uses NIST
FF1 with cycle-walking back into the input's class. Both are one-to-one, so the
whole mapping is one-to-one and reversible. Because the template is preserved
and the content map is a bijection, two different strings cannot share an
output.

### INT64

`encrypt_int(x)` equals `int(encrypt(str(x)))`, so a value gets the same
pseudonym as a `STRING` and as an `INT64`. It stays inside the INT64 range, and
it is one-to-one on all of INT64, for these reasons:

- The sign is part of the template, so positive and negative values map to
  positive and negative values. A positive and a negative value cannot collide.
- `0` is its own class and maps to `0`.
- The magnitudes `1` to `2**63 - 1` are permuted inside their digit length. At
  19 digits the class `N` ends at `2**63 - 1`, so the result never exceeds the
  INT64 maximum.
- The magnitude `2**63` exists only as `-2**63` (INT64 minimum). It is its own
  class `M` and maps to itself in the string mapping, so `encrypt_int` needs no
  special case. No other value can map to it: class `H` is cycle-walked without
  it, and every other output magnitude is at most `2**63 - 1`.

## Tests in the repository

`tests/test_fpe_collision_free.py`:

| Test | What it checks |
|---|---|
| `test_ff1_nist_vectors` | `fastfpe` FF1 reproduces the NIST sample vectors. |
| `test_fpe_golden` | Fixed key and salt give the recorded outputs (`tests/fixtures/fpe_ff1_golden.json`). |
| `test_fpe_cf_small_domains` | Every digit string up to 4 characters and every non-numeric alphanumeric string up to 2 characters is one-to-one, keeps class and length, and round-trips. |
| `test_fpe_cf_exhaustive_digits` (slow) | All 1,111,110 digit strings of length 1 to 6: no two share an output, class and length are kept, decrypt returns the input. |
| `test_fpe_cf_exhaustive_alnum` (slow) | All 241,124 non-numeric alphanumeric strings of length 1 to 3, same checks. |
| `test_fpe_cf_samples` | Random samples that are one-to-one and keep class and length: alphanumeric of 4 to 8 characters, 19-digit values below and above `INT64_MAX` (outputs stay below or above it), long digit strings (no leading zero in the output), long alphanumeric strings (a letter in the output) and zero-led digit strings. |
| `test_fpe_cf_separators` | Inputs such as `(555) 123-4567`, `123-45-6789` and `AB-9021-X7` keep every non-alphanumeric character at its position, stay distinct and round-trip. |
| `test_fpe_cf_zero_is_a_fixed_point` | `0` and `-0` map to themselves. |
| `test_fpe_cf_no_length_limit` | 5,000-character inputs encrypt and decrypt. |
| `test_fpe_cf_int64` | Values -2000 to 2000, the extremes (`INT64_MIN`, `INT64_MIN + 1`, `-1`, `0`, `1`, `INT64_MAX - 1`, `INT64_MAX`) and 5,000 random INT64 values: all outputs distinct, in range, sign and digit count kept, round-trip, and equal to `int(encrypt(str(v)))`. `INT64_MIN` maps to itself. |
| `test_fpe_cf_int64_min_string_and_cell_agree`, `test_fpe_cf_int64_min_magnitude_is_reserved` | `-9223372036854775808` maps to itself through `encrypt`, `decrypt` and `anonymize_cell`; sampled 19-digit values above `2**63` and below it never map onto `2**63`, stay on their side of it, and round-trip. |
| `test_fpe_cf_int64_rejects_non_int`, `test_fpe_cf_int64_rejects_out_of_range` | `bool`, float, string, `None` raise `TypeError`; values outside INT64 raise `ValueError`. |
| `test_fpe_cf_entity_type_independent` | The same value gets the same pseudonym for every FPE entity type. |
| `test_fpe_cf_cycle_walking_cap` | A cipher that never returns to the class makes cycle-walking raise `RuntimeError` instead of returning the plaintext. |
| `test_fpe_cf_determinism_across_hash_seeds`, `test_fpe_cf_determinism_across_threads` | The output does not depend on `PYTHONHASHSEED` or on threads building the lookup tables. |

Run them with:

```bash
uv run pytest tests/test_fpe_collision_free.py            # includes the slow exhaustive tests
uv run pytest tests/test_fpe_collision_free.py -m "not slow"
```

## Independent INT64 check

The tests above sample INT64 sparsely. This check goes further for INT64: it
encrypts a large exhaustive range near zero, boundary values, a random sample
across every digit length, and the edges around the INT64 maximum, then asserts
the same properties for each value.

For every value `x` the check asserts that:

1. `encrypt_int(x)` is inside the INT64 range;
2. the sign and zero are kept (a positive stays positive, a negative stays
   negative, zero stays zero);
3. the digit count is kept (except `INT64_MIN`, which maps to itself);
4. `decrypt_int(encrypt_int(x)) == x`;
5. no earlier value in the same set produced the same output.

Condition 4 alone proves the mapping is injective on every tested value.
Condition 5 checks the outputs directly as well.

### Result

Run on 2026-10-28 against commit `79a787e` (`feat/fpe-collision-free`) with
`fastfpe` 0.2.1, a synthetic 32-byte salt (`b"s" * 32`) and key (`b"k" * 32`):

| Set | Values | Collisions | Range, sign, digit count, round trip |
|---|---|---|---|
| Every integer from -1,500,000 to 1,500,000 | 3,000,001 | 0 | all kept |
| Boundary values (see script) | 13 | 0 | all kept |
| Random sample: 300,000 uniform over INT64 plus 5,000 per digit length for lengths 1 to 19 | 381,175 | 0 | all kept |
| The 20,001 values ending at `INT64_MAX` | 20,001 | 0 | all kept |
| The same 20,001 values, negated | 20,001 | 0 | all kept |

The exhaustive range took about a minute. The other sets took a few seconds
each.

### Limits of this evidence

The set of INT64 values is $2^{64}$, so this is a large sample and not a
complete enumeration. The guarantee for values outside the tested sets comes
from the construction described above (a permutation on each class and length,
with the sign kept apart); the check confirms that the implementation behaves as
that construction predicts at the sizes and boundaries where mistakes usually
occur (the 6-digit FF1 minimum, the 19-digit class boundary and `INT64_MIN`).

### Re-running the check

Save this as a script outside the repository and run it from the repository
root with `uv run python <script>`. Use only synthetic keys.

```python
import random
import time

from tide2.cryptographic.fpe_strings import INT64_MAX
from tide2.cryptographic.fpe_strings import INT64_MIN
from tide2.cryptographic.fpe_strings import FormatPreservingEncryption

fpe = FormatPreservingEncryption(salt=b"s" * 32, key=b"k" * 32)


def check(values, label):
    start = time.time()
    seen = {}
    for x in values:
        y = fpe.encrypt_int(x)
        assert INT64_MIN <= y <= INT64_MAX, (x, y)
        assert (y > 0) == (x > 0) and (y < 0) == (x < 0), (x, y)
        assert len(str(abs(y))) == len(str(abs(x))) or x == INT64_MIN, (x, y)
        assert fpe.decrypt_int(y) == x, (x, y)
        assert y not in seen, ("collision", x, seen[y], y)
        seen[y] = x
    print(f"{label}: {len(seen):,} values, 0 collisions, {time.time() - start:.0f}s")


check(range(-1_500_000, 1_500_001), "exhaustive -1.5M..1.5M")

boundaries = {
    INT64_MAX, INT64_MAX - 1, INT64_MIN, INT64_MIN + 1, INT64_MIN + 2,
    10**18, 10**18 - 1, -(10**18),
    9_223_372_036_854_775_800, 9_223_372_036_854_775_806, -9_223_372_036_854_775_806,
    999_999, 1_000_000, 100_000, 99_999,
}
check(sorted(boundaries), "boundaries")

rng = random.Random(7)
values = {rng.randint(INT64_MIN, INT64_MAX) for _ in range(300_000)}
for digits in range(1, 20):
    values.update(
        rng.randint(10 ** (digits - 1), min(10**digits - 1, INT64_MAX)) * rng.choice((1, -1))
        for _ in range(5000)
    )
check(sorted(values), "random across every digit length")

top = list(range(INT64_MAX - 20_000, INT64_MAX + 1))
check(top, "top 20,001 values below INT64_MAX")
check([-x for x in top], "mirror (negative) top 20,001")
```

Note that `INT64_MIN + 1` equals `-INT64_MAX`. A value listed twice maps to the
same output twice, which the script would report as a collision, so keep the
boundary values in a set.
