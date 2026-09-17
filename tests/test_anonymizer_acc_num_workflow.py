"""
Workflow execution tests for ACC_NUM anonymization with per-note patient_uid.

Tests verify that AnonymizerWorker processes accession numbers using per-note
patient_uid correctly, ensuring per-patient uniqueness, determinism, and chunked
boundary handling.
"""

from hashlib import sha256

import orjson
import pytest
import ray

from tide2.actors.anonymizer import MAX_ANON_CHUNK_SIZE
from tide2.actors.anonymizer import AnonymizerWorker
from tide2.anonymizers.accession_number_hash import AccessionNumberHashAnonymizer


def _compute_expected_hash(salt: str | None, study_id: str | None, entity: str | None, identifier: str) -> str:
    """Compute expected hash matching BigQuery identifier_hashing_algorithm."""
    s = salt.strip().upper() if salt is not None else "[S]"
    u = study_id.strip().upper() if study_id is not None else "[U]"
    e = entity.strip().upper() if entity is not None else "[E]"
    i = identifier.strip().upper() if identifier else ""
    concat = f"{s}|{u}|{e}|{i}"
    return sha256(concat.encode("utf-8")).hexdigest().upper()[:16]


def _get_worker_instance(acc_num_salt: str = "test_salt", acc_num_study_id: str = "test_study"):
    """Instantiate underlying worker class directly without Ray cluster."""
    worker_cls = AnonymizerWorker.__ray_metadata__.modified_class
    return worker_cls(
        salt=b"0" * 32,
        key=b"1" * 32,
        acc_num_salt=acc_num_salt,
        acc_num_study_id=acc_num_study_id,
    )


class TestAnonymizerWorkerAccNumDirect:
    """Unit tests for AnonymizerWorker.process_note accession hashing."""

    def test_per_patient_uniqueness_short_path(self):
        """Different patient_uids yield distinct hashes for identical accession text."""
        worker = _get_worker_instance()
        note = "Accession is ACC1234567 for scan."
        recognizer_json = orjson.dumps([{"entity_type": "ACC_NUM", "start": 13, "end": 23, "score": 1.0}]).decode(
            "utf-8"
        )

        res_a = worker.process_note(
            note_text=note,
            original_text_hash="hash_a",
            recognizer_results_json=recognizer_json,
            patient_uid="patient_AAA",
            jitter=10,
        )
        res_b = worker.process_note(
            note_text=note,
            original_text_hash="hash_b",
            recognizer_results_json=recognizer_json,
            patient_uid="patient_BBB",
            jitter=10,
        )

        items_a = orjson.loads(res_a["anonymizer_results_json"])
        items_b = orjson.loads(res_b["anonymizer_results_json"])

        hash_a = items_a[0]["text"]
        hash_b = items_b[0]["text"]

        assert hash_a != hash_b
        assert len(hash_a) == 16
        assert len(hash_b) == 16
        assert hash_a in res_a["anonymized_note_text"]
        assert hash_b in res_b["anonymized_note_text"]

    def test_determinism_short_path(self):
        """Identical inputs produce identical anonymized text and hash."""
        worker = _get_worker_instance()
        note = "Accession is ACC1234567 for scan."
        recognizer_json = orjson.dumps([{"entity_type": "ACC_NUM", "start": 13, "end": 23, "score": 1.0}]).decode(
            "utf-8"
        )

        res_1 = worker.process_note(
            note_text=note,
            original_text_hash="hash_1",
            recognizer_results_json=recognizer_json,
            patient_uid="patient_AAA",
            jitter=10,
        )
        res_2 = worker.process_note(
            note_text=note,
            original_text_hash="hash_1",
            recognizer_results_json=recognizer_json,
            patient_uid="patient_AAA",
            jitter=10,
        )

        assert res_1["anonymized_note_text"] == res_2["anonymized_note_text"]
        assert res_1["anonymizer_results_json"] == res_2["anonymizer_results_json"]

    def test_matches_direct_operator_result(self):
        """Worker output matches direct AccessionNumberHashAnonymizer calculation."""
        salt = "worker_salt"
        study_id = "worker_study"
        patient_uid = "pat_789"
        accession_text = "ACC998877"

        worker = _get_worker_instance(acc_num_salt=salt, acc_num_study_id=study_id)
        note = f"Check {accession_text}."
        start = note.index(accession_text)
        end = start + len(accession_text)
        recognizer_json = orjson.dumps([{"entity_type": "ACC_NUM", "start": start, "end": end, "score": 1.0}]).decode(
            "utf-8"
        )

        res = worker.process_note(
            note_text=note,
            original_text_hash="h1",
            recognizer_results_json=recognizer_json,
            patient_uid=patient_uid,
            jitter=10,
        )

        expected_hash = AccessionNumberHashAnonymizer().operate(
            accession_text,
            {"salt": salt, "study_id": study_id, "patient_uid": patient_uid},
        )
        items = orjson.loads(res["anonymizer_results_json"])
        assert items[0]["text"] == expected_hash

    def test_none_patient_uid_uses_default(self):
        """None patient_uid yields default [E] component without error."""
        salt = "salt_x"
        study_id = "study_y"
        worker = _get_worker_instance(acc_num_salt=salt, acc_num_study_id=study_id)

        note = "Accession: ACC55555."
        start = note.index("ACC55555")
        end = start + len("ACC55555")
        recognizer_json = orjson.dumps([{"entity_type": "ACC_NUM", "start": start, "end": end, "score": 1.0}]).decode(
            "utf-8"
        )

        res = worker.process_note(
            note_text=note,
            original_text_hash="h_none",
            recognizer_results_json=recognizer_json,
            patient_uid=None,
            jitter=10,
        )

        expected_hash = _compute_expected_hash(salt, study_id, None, "ACC55555")
        items = orjson.loads(res["anonymizer_results_json"])
        assert items[0]["text"] == expected_hash

    def test_nan_patient_uid_uses_default(self):
        """NaN patient_uid from nullable float column resolves to default [E] token."""
        salt = "salt_x"
        study_id = "study_y"
        worker = _get_worker_instance(acc_num_salt=salt, acc_num_study_id=study_id)

        note = "Accession: ACC55555."
        start = note.index("ACC55555")
        end = start + len("ACC55555")
        recognizer_json = orjson.dumps([{"entity_type": "ACC_NUM", "start": start, "end": end, "score": 1.0}]).decode(
            "utf-8"
        )

        res = worker.process_note(
            note_text=note,
            original_text_hash="h_nan",
            recognizer_results_json=recognizer_json,
            patient_uid=float("nan"),
            jitter=10,
        )

        expected_hash = _compute_expected_hash(salt, study_id, None, "ACC55555")
        items = orjson.loads(res["anonymizer_results_json"])
        assert items[0]["text"] == expected_hash

    def test_numeric_patient_uid_worker(self):
        """Numeric patient_uid from integer column is converted to string for hashing."""
        salt = "salt_x"
        study_id = "study_y"
        worker = _get_worker_instance(acc_num_salt=salt, acc_num_study_id=study_id)

        note = "Accession: ACC55555."
        start = note.index("ACC55555")
        end = start + len("ACC55555")
        recognizer_json = orjson.dumps([{"entity_type": "ACC_NUM", "start": start, "end": end, "score": 1.0}]).decode(
            "utf-8"
        )

        res = worker.process_note(
            note_text=note,
            original_text_hash="h_int",
            recognizer_results_json=recognizer_json,
            patient_uid=123456,
            jitter=10,
        )

        expected_hash = _compute_expected_hash(salt, study_id, "123456", "ACC55555")
        items = orjson.loads(res["anonymizer_results_json"])
        assert items[0]["text"] == expected_hash

    def test_chunked_path_per_patient_uniqueness(self):
        """Chunked processing branch (> MAX_ANON_CHUNK_SIZE) scopes hash by patient_uid."""
        worker = _get_worker_instance()

        prefix = "A" * (MAX_ANON_CHUNK_SIZE + 500)
        accession_text = "ACC112233"
        note = f"{prefix} Report: {accession_text} end."
        start = note.index(accession_text)
        end = start + len(accession_text)
        assert len(note) > MAX_ANON_CHUNK_SIZE

        recognizer_json = orjson.dumps([{"entity_type": "ACC_NUM", "start": start, "end": end, "score": 1.0}]).decode(
            "utf-8"
        )

        res_a = worker.process_note(
            note_text=note,
            original_text_hash="h_chunk_a",
            recognizer_results_json=recognizer_json,
            patient_uid="patient_CHUNK_A",
            jitter=10,
        )
        res_b = worker.process_note(
            note_text=note,
            original_text_hash="h_chunk_b",
            recognizer_results_json=recognizer_json,
            patient_uid="patient_CHUNK_B",
            jitter=10,
        )

        items_a = orjson.loads(res_a["anonymizer_results_json"])
        items_b = orjson.loads(res_b["anonymizer_results_json"])

        hash_a = items_a[0]["text"]
        hash_b = items_b[0]["text"]

        assert hash_a != hash_b
        assert hash_a in res_a["anonymized_note_text"]
        assert hash_b in res_b["anonymized_note_text"]


@pytest.mark.integration
class TestAnonymizerWorkerRayWorkflow:
    """Ray remote actor workflow execution tests."""

    @classmethod
    def setup_class(cls):
        """Initialize Ray cluster for integration test."""
        if not ray.is_initialized():
            ray.init(num_cpus=1, ignore_reinit_error=True, include_dashboard=False)

    @classmethod
    def teardown_class(cls):
        """Shut down Ray cluster."""
        if ray.is_initialized():
            ray.shutdown()

    def test_remote_worker_execution(self):
        """Remote Ray actor processes note and preserves patient_uid scoping."""
        worker = AnonymizerWorker.remote(
            salt=b"0" * 32,
            key=b"1" * 32,
            acc_num_salt="ray_salt",
            acc_num_study_id="ray_study",
        )

        note = "Order accession: ACC776655."
        start = note.index("ACC776655")
        end = start + len("ACC776655")
        recognizer_json = orjson.dumps([{"entity_type": "ACC_NUM", "start": start, "end": end, "score": 1.0}]).decode(
            "utf-8"
        )

        ref_a = worker.process_note.remote(
            note_text=note,
            original_text_hash="ray_hash_a",
            recognizer_results_json=recognizer_json,
            patient_uid="ray_pat_a",
            jitter=10,
        )
        ref_b = worker.process_note.remote(
            note_text=note,
            original_text_hash="ray_hash_b",
            recognizer_results_json=recognizer_json,
            patient_uid="ray_pat_b",
            jitter=10,
        )

        res_a = ray.get(ref_a)
        res_b = ray.get(ref_b)

        items_a = orjson.loads(res_a["anonymizer_results_json"])
        items_b = orjson.loads(res_b["anonymizer_results_json"])

        assert items_a[0]["text"] != items_b[0]["text"]
        assert len(items_a[0]["text"]) == 16
