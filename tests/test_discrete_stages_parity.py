"""Tests for discrete sequential stages and performance parity fixes.

Covers:
1. DeduplicateLogFilter suppressing repeated warnings.
2. Vectorized NumPy extraction in TransformerCore._extract_window_predictions.
3. In-actor aggregation in TransformerInferenceActor (aggregate_bio=True).
4. Window length-sorting and order-restoration in TransformerInferenceActor.
5. RecognizerWorker patient_identifiers parsing (dicts and strings).
6. RecognizerWorker columnar passthrough for Stage 3 (note_text, patient_uid, row_id).
7. AnonymizerWorker patient_uid normalization (ignoring 'nan', 'none', etc.).
8. Fractional GPU resource resolution in LocalJobRunner.
9. override_num_blocks propagation across stages.
10. CLI support for fractional --num-gpus and --override-num-blocks.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import ray

import tide2.runner.local_runner as lr
from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.recognizer import RecognizerWorker
from tide2.actors.recognizer import _DeduplicateLogFilter
from tide2.actors.transformer import TransformerInferenceActor
from tide2.runner.cli import main as cli_main
from tide2.transformers.core import TransformerCore

# ---------------------------------------------------------------------------
# 1. DeduplicateLogFilter
# ---------------------------------------------------------------------------


def test_deduplicate_log_filter():
    """_DeduplicateLogFilter only lets each (level, msg) through once."""
    log_filter = _DeduplicateLogFilter(max_entries=10)
    record1 = logging.LogRecord("test", logging.WARNING, "path", 1, "Repeated warning", (), None)
    record2 = logging.LogRecord("test", logging.WARNING, "path", 1, "Repeated warning", (), None)
    record3 = logging.LogRecord("test", logging.ERROR, "path", 1, "Repeated warning", (), None)
    record4 = logging.LogRecord("test", logging.WARNING, "path", 1, "Different warning", (), None)

    assert log_filter.filter(record1) is True
    assert log_filter.filter(record2) is False  # Duplicate suppressed
    assert log_filter.filter(record3) is True  # Different level allowed
    assert log_filter.filter(record4) is True  # Different message allowed


# ---------------------------------------------------------------------------
# 2. Vectorized NumPy extraction in TransformerCore
# ---------------------------------------------------------------------------


def test_vectorized_window_predictions():
    """_extract_window_predictions uses vectorized filtering to extract entities."""
    core = TransformerCore.__new__(TransformerCore)
    core._id2label = {0: "O", 1: "B-PATIENT", 2: "I-PATIENT", 3: "B-DATE"}
    core._ignore_labels_set = {"O"}
    core._is_ignored_arr = None

    scores_np = np.array([[0.1, 0.95, 0.85, 0.1], [0.99, 0.1, 0.9, 0.1]], dtype=np.float32)
    label_ids_np = np.array([[0, 1, 2, 0], [3, 0, 1, 0]], dtype=np.int64)
    # Window 0 has special token at position 0, Window 1 has special token at position 3
    special_np = np.array([[True, False, False, False], [False, False, False, True]], dtype=bool)
    offset_np = np.array(
        [[[0, 0], [0, 4], [5, 8], [0, 0]], [[0, 4], [0, 0], [5, 9], [0, 0]]],
        dtype=np.int64,
    )
    texts = ["John Doe extra", "2024 Jane"]

    preds = core._extract_window_predictions(scores_np, label_ids_np, special_np, offset_np, texts)

    assert len(preds) == 2
    # Window 0: position 0 is special, position 1 is B-PATIENT (0..4), position 2 is I-PATIENT (5..8)
    assert len(preds[0]) == 2
    assert preds[0][0]["entity"] == "B-PATIENT"
    assert preds[0][0]["start"] == 0
    assert preds[0][0]["end"] == 4
    assert preds[0][0]["word"] == "John"
    assert preds[0][0]["index"] == 1

    assert preds[0][1]["entity"] == "I-PATIENT"
    assert preds[0][1]["start"] == 5
    assert preds[0][1]["end"] == 8
    assert preds[0][1]["word"] == "Doe"

    # Window 1: position 0 is B-DATE (0..4), position 2 is B-PATIENT (5..9)
    assert len(preds[1]) == 2
    assert preds[1][0]["entity"] == "B-DATE"
    assert preds[1][0]["word"] == "2024"
    assert preds[1][1]["entity"] == "B-PATIENT"
    assert preds[1][1]["word"] == "Jane"


# ---------------------------------------------------------------------------
# 3. In-actor aggregation in TransformerInferenceActor
# ---------------------------------------------------------------------------


def test_transformer_actor_in_actor_aggregation():
    """TransformerInferenceActor with aggregate_bio=True emits recognizer_results_json directly."""
    actor = TransformerInferenceActor.__new__(TransformerInferenceActor)
    actor._aggregate_bio = True
    actor._recognizer_name = "StanfordAIMI/stanford-deidentifier-v2"
    actor._log_gpu_mem = lambda _s: None
    actor._handled_oom_count = 0

    # Mock raw inference results: one note with two entities
    fake_preds = [
        [
            {"entity": "B-PATIENT", "score": 0.99, "start": 0, "end": 4, "word": "John", "index": 1},
            {"entity": "I-PATIENT", "score": 0.95, "start": 5, "end": 8, "word": "Doe", "index": 2},
        ]
    ]
    actor._run_inference_raw_with_oom_recovery = lambda _texts: fake_preds

    batch = {
        "text_hash": ["h123"],
        "patient_id": ["pid1"],
        "patient_uid": ["puid1"],
        "row_id": ["row1"],
        "jitter": [14],
        "note_text": ["John Doe is a patient."],
    }

    out = actor(batch)

    assert "recognizer_results_json" in out
    assert "predictions_raw_json" not in out
    assert out["entity_count"] == [1]
    assert out["patient_uid"] == ["puid1"]
    assert out["row_id"] == ["row1"]
    assert out["jitter"] == [14]
    assert "PATIENT" in out["recognizer_results_json"][0]


# ---------------------------------------------------------------------------
# 4. Window length-sorting and unsorting
# ---------------------------------------------------------------------------


def test_transformer_actor_length_sorting_preserves_order():
    """_run_inference_raw_with_oom_recovery sorts by window length and unsorts back to document order."""
    actor = TransformerInferenceActor.__new__(TransformerInferenceActor)
    actor._core = MagicMock()
    # 2 notes: Note 0 has a long window (len 10), Note 1 has a short window (len 3)
    actor._core.tokenize_ragged.return_value = {
        "input_ids": [[1] * 10, [2] * 3],
        "offset_mapping": [[(0, 1)] * 10, [(0, 1)] * 3],
    }

    w0 = MagicMock(owner=0, content_ids=[1] * 10)
    w1 = MagicMock(owner=1, content_ids=[2] * 3)
    actor._plan_windows = lambda _texts, _ids, _offs: [w0, w1]

    # _forward_windows receives sorted windows [w1 (len 3), w0 (len 10)]
    forwarded_order = []

    def mock_forward(windows):
        for w in windows:
            forwarded_order.append(len(w.content_ids))
        # Return predictions aligned to sorted windows: for w1 returns ["pred1"], for w0 returns ["pred0"]
        return [["pred1"], ["pred0"]]

    actor._forward_windows = mock_forward

    results = actor._run_inference_raw_with_oom_recovery(["text0", "text1"])

    # Forward executed shorter window first (length bucketing)
    assert forwarded_order == [3, 10]
    # Results mapped correctly back to owners
    assert results[0] == ["pred0"]
    assert results[1] == ["pred1"]


# ---------------------------------------------------------------------------
# 5. RecognizerWorker patient_identifiers parsing
# ---------------------------------------------------------------------------


def test_recognizer_worker_dict_patient_identifiers(monkeypatch):
    """RecognizerWorker._build_ad_hoc_recognizers parses dict and orjson string identifiers."""
    worker_cls = getattr(RecognizerWorker, "__ray_actor_class__", RecognizerWorker)
    worker = worker_cls.__new__(worker_cls)

    # With pre-parsed dict
    dict_phi = {"person": ["Alice Smith"]}
    recs = worker._build_ad_hoc_recognizers(cached_results=None, patient_identifiers=dict_phi, text_hash="h1")
    assert len(recs) > 0

    # With JSON string
    json_phi = '{"person": ["Bob Jones"]}'
    recs_json = worker._build_ad_hoc_recognizers(cached_results=None, patient_identifiers=json_phi, text_hash="h2")
    assert len(recs_json) > 0

    # With invalid / None
    recs_none = worker._build_ad_hoc_recognizers(cached_results=None, patient_identifiers=None, text_hash="h3")
    assert len(recs_none) == 0


# ---------------------------------------------------------------------------
# 6. RecognizerWorker process_batch passthrough columns
# ---------------------------------------------------------------------------


def test_recognizer_worker_process_batch_passthrough():
    """RecognizerWorker.process_batch passes note_text, patient_uid, row_id, jitter forward for Stage 3."""
    worker_cls = getattr(RecognizerWorker, "__ray_actor_class__", RecognizerWorker)
    worker = worker_cls.__new__(worker_cls)

    def mock_process_note(note_text, text_hash, cached_results, patient_identifiers):
        del note_text, cached_results, patient_identifiers
        return {
            "text_hash": text_hash,
            "recognizer_results_json": "[]",
            "entity_count": 0,
            "processing_status": "success",
            "error_message": None,
        }

    worker.process_note = mock_process_note

    batch = {
        "text_hash": ["h1"],
        "note_text": ["Patient note text"],
        "patient_uid": ["P123"],
        "row_id": ["R456"],
        "jitter": [5],
    }

    res = worker.process_batch(batch)
    assert res["note_text"] == ["Patient note text"]
    assert res["patient_uid"] == ["P123"]
    assert res["row_id"] == ["R456"]
    assert res["jitter"] == [5]
    assert res["text_hash"] == ["h1"]


# ---------------------------------------------------------------------------
# 7. AnonymizerWorker patient_uid normalization
# ---------------------------------------------------------------------------


def test_anonymizer_worker_patient_uid_normalization():
    """AnonymizerWorker normalizes patient_uid, discarding 'nan', 'none', and empty strings."""
    worker_cls = getattr(AnonymizerWorker, "__ray_actor_class__", AnonymizerWorker)
    worker = worker_cls.__new__(worker_cls)
    worker.salt = b"\x00" * 32
    worker.key = b"\x11" * 32
    worker.acc_num_salt = "salt"
    worker.acc_num_study_id = "study"
    worker._base_operators = {}

    # Valid patient_uid
    ops_valid = worker._create_operators_for_note(date_jitter=10, patient_uid="P999")
    assert ops_valid["ACC_NUM"].params["patient_uid"] == "P999"

    # 'nan' string should be normalized to None
    ops_nan = worker._create_operators_for_note(date_jitter=10, patient_uid="nan")
    assert ops_nan["ACC_NUM"].params["patient_uid"] is None

    # 'None' string should be normalized to None
    ops_none = worker._create_operators_for_note(date_jitter=10, patient_uid="None")
    assert ops_none["ACC_NUM"].params["patient_uid"] is None

    # float NaN should be normalized to None
    ops_float_nan = worker._create_operators_for_note(date_jitter=10, patient_uid=float("nan"))
    assert ops_float_nan["ACC_NUM"].params["patient_uid"] is None


# ---------------------------------------------------------------------------
# 8. Fractional GPU resource resolution
# ---------------------------------------------------------------------------


def test_resolve_transformer_resources_fractional_gpu(monkeypatch):
    """LocalJobRunner._resolve_transformer_resources handles fractional GPUs properly."""
    runner = lr.LocalJobRunner()
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 16, "GPU": 1})

    # Requesting 0.33 GPU: should multiplex 3 actors and set num_agg_actors to 0 (in-actor aggregation)
    num_gpus, cpu_only, num_actors, num_agg = runner._resolve_transformer_resources(
        num_gpus=0.33,
        num_transformer_actors=None,
        num_agg_actors=None,
    )

    assert num_gpus == 0.33
    assert cpu_only is False
    assert num_actors == 3
    assert num_agg == 0


# ---------------------------------------------------------------------------
# 9. override_num_blocks propagation across stages
# ---------------------------------------------------------------------------


def test_override_num_blocks_propagates(monkeypatch, tmp_path):
    """override_num_blocks is passed to ray.data.read_parquet across stages."""
    runner = lr.LocalJobRunner()
    monkeypatch.setattr(runner, "_init_ray", lambda: None)
    monkeypatch.setattr(lr, "configure_data_context", lambda **_k: None)

    captured_read = {}

    def fake_read_parquet(_files, **kwargs):
        captured_read.update(kwargs)
        fake_ds = MagicMock()
        fake_ds.map_batches.return_value = fake_ds
        fake_ds.count.return_value = 1
        return fake_ds

    monkeypatch.setattr(ray.data, "read_parquet", fake_read_parquet)
    monkeypatch.setattr(lr, "resolve_input_files", lambda _p: ["/fake.parquet"])
    monkeypatch.setattr(lr, "detect_columns", lambda *_a, **_k: ["text_hash", "note_text", "patient_identifiers"])
    monkeypatch.setattr(
        ray.data.DataContext,
        "get_current",
        staticmethod(lambda: SimpleNamespace(checkpoint_config=None)),
    )

    # 1. Recognizer with override_num_blocks=32
    runner.run_recognition(
        input_path="in",
        output_path=str(tmp_path / "rec"),
        num_actors=14,
        override_num_blocks=32,
        enable_checkpoint=False,
    )
    assert captured_read["override_num_blocks"] == 32

    # 2. Anonymizer with override_num_blocks=32
    salt_file = tmp_path / "salt.bin"
    key_file = tmp_path / "key.bin"
    salt_file.write_text("00" * 32)
    key_file.write_text("11" * 32)
    runner.run_anonymization(
        input_path="in",
        output_path=str(tmp_path / "anon"),
        salt_path=str(salt_file),
        key_path=str(key_file),
        num_actors=14,
        override_num_blocks=32,
        enable_checkpoint=False,
    )
    assert captured_read["override_num_blocks"] == 32

    # 3. Transformer with override_num_blocks=16
    monkeypatch.setattr(runner, "_resolve_transformer_resources", lambda *_a, **_k: (0.33, False, 3, 0))
    from tide2.transformers import config as tconfig

    monkeypatch.setattr(tconfig, "load_model_config", lambda _name: {"CHUNK_OVERLAP_SIZE": 40, "MODEL_MAX_LENGTH": 512})
    from tide2 import actors

    monkeypatch.setattr(actors, "create_transformer_actor", lambda **_k: MagicMock())

    runner.run_transformer(
        input_path="in",
        output_path=str(tmp_path / "trans"),
        model_name="fake-model",
        model_path=str(tmp_path / "model"),
        override_num_blocks=16,
        enable_checkpoint=False,
    )
    assert captured_read["override_num_blocks"] == 16


# ---------------------------------------------------------------------------
# 10. CLI flags for fractional GPU and override_num_blocks
# ---------------------------------------------------------------------------


def test_cli_flags_fractional_gpu_and_blocks(monkeypatch):
    """CLI accepts fractional --num-gpus and --override-num-blocks."""
    captured: dict = {}

    def fake_run_transformer(self, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(lr.LocalJobRunner, "run_transformer", fake_run_transformer)
    monkeypatch.setattr(lr.LocalJobRunner, "shutdown", lambda _self: None)

    monkeypatch.setattr(
        "sys.argv",
        [
            "tide2-runner",
            "run",
            "transformer",
            "-i",
            "in",
            "-o",
            "out",
            "--model",
            "test-model",
            "--num-gpus",
            "0.33",
            "--override-num-blocks",
            "16",
        ],
    )

    cli_main()
    assert captured["num_gpus"] == 0.33
    assert captured["override_num_blocks"] == 16
