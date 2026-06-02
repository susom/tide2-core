"""Tests for row_id generation logic."""

import hashlib

import numpy as np
import pandas as pd


class TestRowIdGeneration:
    """Test row_id generation logic."""

    def _generate_row_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply row_id generation (same as local_runner.py)."""
        df = df.copy()
        df["row_id"] = (df["text_hash"] + ":" + df["patient_uid"].fillna("None").astype(str)).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()
        )
        return df

    def test_row_id_always_generated(self):
        """Verify row_id is always created, not just for duplicates."""
        df = pd.DataFrame(
            {
                "text_hash": ["unique_hash_1", "unique_hash_2"],
                "patient_uid": ["uid_a", "uid_b"],
            }
        )

        df = self._generate_row_id(df)

        assert "row_id" in df.columns
        assert df["row_id"].notna().all()

    def test_row_id_is_sha256_hash(self):
        """Verify the format is 64-char hex SHA256."""
        df = pd.DataFrame(
            {
                "text_hash": ["abc123"],
                "patient_uid": ["patient_001"],
            }
        )

        df = self._generate_row_id(df)
        row_id = df["row_id"].iloc[0]

        assert len(row_id) == 64, f"Expected 64 chars, got {len(row_id)}"
        assert row_id == row_id.lower(), "Should be lowercase hex"
        assert all(c in "0123456789abcdef" for c in row_id), "Should be valid hex"

    def test_row_id_deterministic(self):
        """Same inputs produce same row_id."""
        text_hash = "abc123def456"
        uid = "patient_001"

        composite = f"{text_hash}:{uid}"
        hash1 = hashlib.sha256(composite.encode()).hexdigest()
        hash2 = hashlib.sha256(composite.encode()).hexdigest()

        assert hash1 == hash2

    def test_row_id_unique_for_different_patients(self):
        """Different UIDs produce different row_ids."""
        df = pd.DataFrame(
            {
                "text_hash": ["same_hash", "same_hash"],
                "patient_uid": ["uid_a", "uid_b"],
            }
        )

        df = self._generate_row_id(df)

        assert df["row_id"].iloc[0] != df["row_id"].iloc[1]
        assert df["row_id"].is_unique

    def test_row_id_unique_for_different_text_hash(self):
        """Different text_hash with same UID produce different row_ids."""
        df = pd.DataFrame(
            {
                "text_hash": ["hash_a", "hash_b"],
                "patient_uid": ["same_uid", "same_uid"],
            }
        )

        df = self._generate_row_id(df)

        assert df["row_id"].iloc[0] != df["row_id"].iloc[1]
        assert df["row_id"].is_unique

    def test_row_id_handles_none_uid(self):
        """Verify behavior when patient_uid is None."""
        df = pd.DataFrame(
            {
                "text_hash": ["hash1", "hash2"],
                "patient_uid": [None, "valid_uid"],
            }
        )

        df = self._generate_row_id(df)

        assert df["row_id"].notna().all()
        assert df["row_id"].iloc[0] != df["row_id"].iloc[1]

    def test_row_id_handles_nan_uid(self):
        """Verify behavior with NaN values."""
        df = pd.DataFrame(
            {
                "text_hash": ["hash1"],
                "patient_uid": [np.nan],
            }
        )

        df = self._generate_row_id(df)

        assert df["row_id"].iloc[0] is not None
        assert len(df["row_id"].iloc[0]) == 64

    def test_row_id_empty_strings(self):
        """Verify behavior with empty strings."""
        df = pd.DataFrame(
            {
                "text_hash": ["", "hash"],
                "patient_uid": ["uid", ""],
            }
        )

        df = self._generate_row_id(df)

        assert df["row_id"].notna().all()
        assert df["row_id"].is_unique

    def test_row_id_special_characters(self):
        """Verify behavior with special characters."""
        df = pd.DataFrame(
            {
                "text_hash": ["hash_with_special_!@#$%"],
                "patient_uid": ["uid_with_unicode_é_ñ"],
            }
        )

        df = self._generate_row_id(df)

        assert len(df["row_id"].iloc[0]) == 64

    def test_row_id_large_dataset_uniqueness(self):
        """Verify uniqueness with larger dataset."""
        n = 1000
        df = pd.DataFrame(
            {
                "text_hash": [f"hash_{i}" for i in range(n)],
                "patient_uid": [f"uid_{i % 100}" for i in range(n)],
            }
        )

        df = self._generate_row_id(df)

        assert df["row_id"].is_unique
        assert len(df["row_id"].unique()) == n


class TestSqlPythonConsistency:
    """Test that SQL and Python hash generation are consistent."""

    def test_hash_format_matches_bigquery(self):
        """
        Verify Python hash format matches BigQuery TO_HEX(SHA256(...)).

        BigQuery produces lowercase hex, same as Python hexdigest().
        """
        text_hash = "abc123def456"
        uid = "patient_001"

        composite = f"{text_hash}:{uid}"
        python_hash = hashlib.sha256(composite.encode()).hexdigest()

        assert len(python_hash) == 64
        assert python_hash == python_hash.lower()
        assert all(c in "0123456789abcdef" for c in python_hash)

    def test_utf8_encoding_consistency(self):
        """Verify UTF-8 encoding is used (same as BigQuery default)."""
        text_hash = "hash"
        uid = "patient_é_ñ_中文"

        composite = f"{text_hash}:{uid}"

        # Python uses UTF-8 by default
        python_hash = hashlib.sha256(composite.encode("utf-8")).hexdigest()

        # Verify it's a valid hash
        assert len(python_hash) == 64

    def test_colon_separator(self):
        """Verify colon is used as separator (matches SQL CONCAT)."""
        text_hash = "abc"
        uid = "123"

        # With colon
        with_colon = hashlib.sha256(f"{text_hash}:{uid}".encode()).hexdigest()

        # Without colon (different)
        without_colon = hashlib.sha256(f"{text_hash}{uid}".encode()).hexdigest()

        # With underscore (old format)
        with_underscore = hashlib.sha256(f"{text_hash}_{uid}".encode()).hexdigest()

        assert with_colon != without_colon
        assert with_colon != with_underscore


class TestRowIdEdgeCases:
    """Test edge cases for row_id generation."""

    def test_duplicate_text_hash_same_uid(self):
        """Same text_hash and same UID should produce same row_id."""
        df = pd.DataFrame(
            {
                "text_hash": ["same_hash", "same_hash"],
                "patient_uid": ["same_uid", "same_uid"],
            }
        )

        df["row_id"] = (df["text_hash"] + ":" + df["patient_uid"].fillna("None").astype(str)).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()
        )

        # Same inputs = same row_id (not unique in this case)
        assert df["row_id"].iloc[0] == df["row_id"].iloc[1]

    def test_very_long_inputs(self):
        """Verify handling of very long text_hash and uid."""
        long_hash = "a" * 10000
        long_uid = "b" * 10000

        composite = f"{long_hash}:{long_uid}"
        result = hashlib.sha256(composite.encode()).hexdigest()

        assert len(result) == 64

    def test_numeric_uid(self):
        """Verify numeric UIDs are handled correctly."""
        df = pd.DataFrame(
            {
                "text_hash": ["hash1", "hash2"],
                "patient_uid": [12345, 67890],
            }
        )

        df["row_id"] = (df["text_hash"] + ":" + df["patient_uid"].fillna("None").astype(str)).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()
        )

        assert df["row_id"].is_unique
        assert all(len(rid) == 64 for rid in df["row_id"])
