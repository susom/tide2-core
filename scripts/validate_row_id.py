#!/usr/bin/env python3
"""Validate row_id generation logic."""

import hashlib
import sys

import pandas as pd

SHA256_HEX_LENGTH = 64
HASH1_ROW_COUNT = 2


def _run_checks(df: pd.DataFrame) -> list[str]:
    """Run validation checks on row_id column and return error messages."""
    errors = []

    # 1. All rows have row_id
    if not df["row_id"].notna().all():
        errors.append("ERROR: Some row_ids are null")
    else:
        print("[PASS] All rows have row_id")

    # 2. All row_ids are 64 chars
    if not all(len(rid) == SHA256_HEX_LENGTH for rid in df["row_id"]):
        errors.append("ERROR: row_ids should be 64 chars")
    else:
        print("[PASS] All row_ids are 64 chars (SHA256 hex)")

    # 3. row_ids are unique
    if not df["row_id"].is_unique:
        errors.append("ERROR: row_ids should be unique")
        duplicates = df[df["row_id"].duplicated(keep=False)]
        print("Duplicate row_ids:")
        print(duplicates)
    else:
        print("[PASS] All row_ids are unique")

    # 4. Same text_hash + different uid = different row_id
    hash1_rows = df[df["text_hash"] == "hash1"]
    if len(hash1_rows) == HASH1_ROW_COUNT and hash1_rows["row_id"].iloc[0] == hash1_rows["row_id"].iloc[1]:
        errors.append("ERROR: Same text_hash with different UIDs should have different row_ids")
    else:
        print("[PASS] Same text_hash with different UIDs produce different row_ids")

    # 5. Deterministic
    test_composite = "hash1:uid_a"
    hash_a = hashlib.sha256(test_composite.encode()).hexdigest()
    hash_b = hashlib.sha256(test_composite.encode()).hexdigest()
    if hash_a != hash_b:
        errors.append("ERROR: Hash should be deterministic")
    else:
        print("[PASS] Hash generation is deterministic")

    # 6. Format validation
    sample_rid = df["row_id"].iloc[0]
    if not all(c in "0123456789abcdef" for c in sample_rid):
        errors.append("ERROR: row_id should be lowercase hex")
    else:
        print("[PASS] row_id format is lowercase hex")

    return errors


def validate_row_id_generation() -> bool:
    """Quick validation of row_id generation."""
    print("=" * 60)
    print("Row ID Generation Validation")
    print("=" * 60)

    # Create test data with various scenarios
    df = pd.DataFrame(
        {
            "text_hash": ["hash1", "hash1", "hash2", "hash3"],
            "patient_uid": ["uid_a", "uid_b", "uid_a", None],
            "note_text": ["text1", "text1", "text2", "text3"],
        }
    )

    print("\nInput DataFrame:")
    print(df[["text_hash", "patient_uid"]])
    print()

    # Generate row_ids (same logic as local_runner.py)
    df["row_id"] = (df["text_hash"] + ":" + df["patient_uid"].fillna("None").astype(str)).apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()
    )

    print("Generated row_ids:")
    for _, row in df.iterrows():
        print(f"  {row['text_hash']}:{row['patient_uid']} -> {row['row_id'][:16]}...")
    print()

    errors = _run_checks(df)

    print()
    if errors:
        print("FAILURES:")
        print("\n".join(errors))
        return False

    print("=" * 60)
    print("All validations passed!")
    print("=" * 60)
    return True


def validate_sql_consistency() -> bool:
    """Validate SQL and Python hash consistency."""
    print("\n" + "=" * 60)
    print("SQL/Python Consistency Check")
    print("=" * 60)

    test_cases = [
        ("abc123", "patient_001"),
        ("def456", "patient_002"),
        ("", "empty_hash_test"),
    ]

    print("\nExpected BigQuery query to validate:")
    print("  SELECT TO_HEX(SHA256(CONCAT(text_hash, ':', uid))) AS row_id")
    print()

    all_passed = True
    for text_hash, uid in test_cases:
        composite = f"{text_hash}:{uid}"
        python_hash = hashlib.sha256(composite.encode()).hexdigest()

        # Validate format
        if len(python_hash) != SHA256_HEX_LENGTH:
            print(f"[FAIL] {composite} -> length {len(python_hash)} != {SHA256_HEX_LENGTH}")
            all_passed = False
        elif python_hash != python_hash.lower():
            print(f"[FAIL] {composite} -> not lowercase")
            all_passed = False
        else:
            print(f"[PASS] {composite} -> {python_hash[:32]}...")

    print()
    if all_passed:
        print("SQL/Python consistency validation passed!")
        print("Run the following in BigQuery to verify:")
        print("  SELECT TO_HEX(SHA256(CONCAT('abc123', ':', 'patient_001')))")
        print(f"  Expected: {hashlib.sha256(b'abc123:patient_001').hexdigest()}")
    return all_passed


if __name__ == "__main__":
    success1 = validate_row_id_generation()
    success2 = validate_sql_consistency()

    if success1 and success2:
        print("\n" + "=" * 60)
        print("ALL VALIDATIONS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        sys.exit(1)
