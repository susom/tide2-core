"""
Unit tests for AccessionNumberHashAnonymizer.

Tests cover deterministic accession number hashing, BigQuery SQL parity,
parameter validation, and Presidio AnonymizerEngine integration.
"""

from hashlib import sha256

import pytest
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.entities import RecognizerResult
from presidio_anonymizer.operators import OperatorType

from tide2.anonymizers.accession_number_hash import AccessionNumberHashAnonymizer


def _expected_bq_hash(salt: str | None, study_id: str | None, entity: str | None, identifier: str) -> str:
    """Compute expected hash matching BigQuery identifier_hashing_algorithm."""
    s = salt.strip().upper() if salt is not None else "[S]"
    u = study_id.strip().upper() if study_id is not None else "[U]"
    e = entity.strip().upper() if entity is not None else "[E]"
    i = identifier.strip().upper() if identifier else ""
    concat = f"{s}|{u}|{e}|{i}"
    return sha256(concat.encode("utf-8")).hexdigest().upper()[:16]


class TestAccessionNumberHashAnonymizer:
    """Test suite for AccessionNumberHashAnonymizer."""

    def setup_method(self):
        """Instantiate operator."""
        self.anonymizer = AccessionNumberHashAnonymizer()

    def test_operator_metadata(self):
        """Verify operator name and type."""
        assert self.anonymizer.operator_name() == "accession_number_hash"
        assert self.anonymizer.operator_type() == OperatorType.Anonymize

    def test_validate_allowed_entities(self):
        """Pass validation when entity_type is in allowed set."""
        for ent in ["DEFAULT", "ACC_NUM", "ACCESSION_NUMBER"]:
            self.anonymizer.validate({"entity_type": ent, "patient_uid": "pat_123"})

    def test_validate_unsupported_entity(self):
        """Raise ValueError when entity_type is unsupported."""
        with pytest.raises(ValueError, match="Entity type 'INVALID' is not supported"):
            self.anonymizer.validate({"entity_type": "INVALID"})

    def test_per_patient_uniqueness(self):
        """Ensure different patient_uids produce distinct hashes for identical text."""
        params_a = {"salt": "mysalt", "study_id": "study1", "patient_uid": "patient_A"}
        params_b = {"salt": "mysalt", "study_id": "study1", "patient_uid": "patient_B"}
        hash_a = self.anonymizer.operate("ACC12345", params_a)
        hash_b = self.anonymizer.operate("ACC12345", params_b)

        assert hash_a != hash_b
        assert len(hash_a) == 16
        assert len(hash_b) == 16

    def test_determinism(self):
        """Same parameters and text produce identical hash."""
        params = {"salt": "mysalt", "study_id": "study1", "patient_uid": "pat_001"}
        hash_1 = self.anonymizer.operate("ACC12345", params)
        hash_2 = self.anonymizer.operate("ACC12345", params)
        assert hash_1 == hash_2

    def test_bigquery_parity_exact(self):
        """Hash matches the BigQuery SQL algorithm precisely."""
        salt = "test_salt"
        study_id = "study_99"
        patient_uid = "pat_xyz"
        acc_num = "ACC-987654"

        expected = _expected_bq_hash(salt, study_id, patient_uid, acc_num)
        actual = self.anonymizer.operate(
            acc_num,
            {"salt": salt, "study_id": study_id, "patient_uid": patient_uid},
        )
        assert actual == expected

    def test_coalesce_default_values(self):
        """None values resolve to default tokens [S], [U], and [E]."""
        expected_defaults = _expected_bq_hash(None, None, None, "ACC99")
        actual_defaults = self.anonymizer.operate("ACC99", {})
        assert actual_defaults == expected_defaults

    def test_nan_patient_uid_resolves_to_default(self):
        """Float NaN resolves to default token [E] matching SQL COALESCE."""
        expected = _expected_bq_hash("salt", "study", None, "ACC99")
        actual = self.anonymizer.operate(
            "ACC99",
            {"salt": "salt", "study_id": "study", "patient_uid": float("nan")},
        )
        assert actual == expected

    def test_numeric_patient_uid(self):
        """Numeric patient_uid is converted to string for hashing."""
        expected = _expected_bq_hash("salt", "study", "12345", "ACC99")
        actual = self.anonymizer.operate(
            "ACC99",
            {"salt": "salt", "study_id": "study", "patient_uid": 12345},
        )
        assert actual == expected

    def test_coalesce_empty_string_not_replaced(self):
        """Empty string is not None and remains empty string per SQL COALESCE."""
        actual = self.anonymizer.operate(
            "ACC99",
            {"salt": "", "study_id": "", "patient_uid": ""},
        )
        expected = _expected_bq_hash("", "", "", "ACC99")
        assert actual == expected

    def test_case_and_whitespace_normalization(self):
        """Whitespace is trimmed and text is upper-cased before hashing."""
        hash_lower = self.anonymizer.operate(
            "  acc12345  ",
            {"salt": " salt ", "study_id": " study ", "patient_uid": " pat "},
        )
        hash_upper = self.anonymizer.operate(
            "ACC12345",
            {"salt": "SALT", "study_id": "STUDY", "patient_uid": "PAT"},
        )
        assert hash_lower == hash_upper

    def test_presidio_anonymizer_engine_regression_guard(self):
        """Regression test: verify patient_uid survives Presidio AnonymizerEngine."""
        engine = AnonymizerEngine()
        engine.add_anonymizer(AccessionNumberHashAnonymizer)

        text = "Accession is ACC8821 for patient."
        analyzer_results = [
            RecognizerResult(entity_type="ACC_NUM", start=13, end=20, score=1.0),
        ]

        operators_pat_1 = {
            "ACC_NUM": OperatorConfig(
                "accession_number_hash",
                {
                    "salt": "s1",
                    "study_id": "u1",
                    "patient_uid": "pat_1",
                },
            ),
        }
        operators_pat_2 = {
            "ACC_NUM": OperatorConfig(
                "accession_number_hash",
                {
                    "salt": "s1",
                    "study_id": "u1",
                    "patient_uid": "pat_2",
                },
            ),
        }

        res_1 = engine.anonymize(text=text, analyzer_results=analyzer_results, operators=operators_pat_1)
        res_2 = engine.anonymize(text=text, analyzer_results=analyzer_results, operators=operators_pat_2)

        expected_hash_1 = _expected_bq_hash("s1", "u1", "pat_1", "ACC8821")
        expected_hash_2 = _expected_bq_hash("s1", "u1", "pat_2", "ACC8821")

        assert res_1.items[0].text == expected_hash_1
        assert res_2.items[0].text == expected_hash_2
        assert res_1.items[0].text != res_2.items[0].text
