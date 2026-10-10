"""Tests for the failure contract: one row out per row in, a status per stage, a mask per entity."""

import json
import logging

import pyarrow as pa
import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.actors.recognizer import RecognizerWorker
from tide2.actors.transformer import BIOAggregationActor
from tide2.actors.transformer import TransformerInferenceActor
from tide2.anonymizers import presidio_patches
from tide2.anonymizers.guarded import guarded
from tide2.anonymizers.guarded import start_note
from tide2.anonymizers.guarded import summarize
from tide2.runner.local_runner import _resolve_merged_batch
from tide2.utils.stage_status import NoteError
from tide2.utils.stage_status import append_status
from tide2.utils.stage_status import failure_reason
from tide2.utils.stage_status import load_stage_status
from tide2.utils.stage_status import merge_stage_status

SALT = b"\x00" * 32
KEY = b"\x11" * 32


@pytest.fixture(scope="module", autouse=True)
def _restore_presidio_patches():
    """The real workers patch Presidio globally; undo it for the other test modules."""
    yield
    presidio_patches.unpatch_remove_duplicates()
    presidio_patches.enable_whitespace_merging()
    presidio_patches.unpatch_conflict_resolution()


def _stages(value: str | None) -> dict:
    return json.loads(value) if value else {}


def _py(column) -> list:
    """Column values as a list; an all-null result column is a typed Arrow array."""
    return column.to_pylist() if hasattr(column, "to_pylist") else list(column)


# ---------------------------------------------------------------------------
# stage_status helpers
# ---------------------------------------------------------------------------


def test_append_status_rolls_up_the_worst_status():
    stage_json, status = append_status(None, "transformer", "success")
    assert status == "success"
    stage_json, status = append_status(stage_json, "anonymizer", "degraded", "AGE")
    assert status == "degraded"
    stage_json, status = append_status(stage_json, "recognizer", "failed", "NoteError:cached_results")
    assert status == "failed"
    assert _stages(stage_json) == {
        "transformer": {"status": "success"},
        "anonymizer": {"status": "degraded", "reason": "AGE"},
        "recognizer": {"status": "failed", "reason": "NoteError:cached_results"},
    }


@pytest.mark.parametrize("previous", [None, float("nan"), "", "not json", "[1, 2]", 5])
def test_malformed_previous_status_counts_as_empty(previous):
    assert load_stage_status(previous) == {}
    stage_json, status = append_status(previous, "recognizer", "success")
    assert _stages(stage_json) == {"recognizer": {"status": "success"}}
    assert status == "success"


def test_merge_stage_status_unions_branches():
    regex_json, _ = append_status(None, "recognizer", "failed", "KeyError")
    llm_json, _ = append_status(None, "llm_recognizer", "success")
    merged_json, status = merge_stage_status(regex_json, llm_json)
    assert set(_stages(merged_json)) == {"recognizer", "llm_recognizer"}
    assert status == "failed"
    assert merge_stage_status(None, None) == ("{}", "success")


def test_failure_reason_never_contains_the_exception_message():
    assert failure_reason(ValueError("John Smith 555-1234")) == "ValueError"
    assert failure_reason(NoteError("invalid_span")) == "NoteError:invalid_span"


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def recognizer():
    return RecognizerWorker()


def _recognizer_batch(**overrides):
    batch = {
        "text_hash": ["good", "bad"],
        "note_text": ["Patient seen on 03/04/2020.", "Another note."],
        "patient_id": ["p1", "p2"],
        "row_id": ["r1", "r2"],
        "jitter": [5, 6],
    }
    batch.update(overrides)
    return batch


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("recognizer_results_json", "not json"),
        ("recognizer_results_json", '{"a": 1}'),
        ("patient_identifiers", "not json"),
        ("patient_identifiers", "[1, 2]"),
        ("patient_identifiers", {"person": [1234]}),
    ],
)
def test_recognizer_emits_a_failed_row_for_malformed_input(recognizer, column, bad_value):
    good = "[]" if column == "recognizer_results_json" else None
    out = recognizer.process_batch(_recognizer_batch(**{column: [good, bad_value]}))

    assert out["text_hash"] == ["good", "bad"]
    assert out["row_id"] == ["r1", "r2"]
    assert out["processing_status"] == ["success", "failed"]
    assert out["recognizer_results_json"][1] is None
    assert out["entity_count"][1] is None
    assert out["recognizer_results_json"][0] is not None
    assert "recognizer" in _stages(out["stage_status_json"][1])
    assert _stages(out["stage_status_json"][0]) == {"recognizer": {"status": "success"}}


def test_recognizer_passes_a_failed_upstream_row_through(recognizer):
    failed_json, _ = append_status(None, "transformer", "failed", "RuntimeError")
    out = recognizer.process_batch(
        _recognizer_batch(
            recognizer_results_json=[None, None],
            processing_status=["failed", "success"],
            stage_status_json=[failed_json, None],
        )
    )
    assert out["processing_status"] == ["failed", "success"]
    assert out["recognizer_results_json"][0] is None
    assert _stages(out["stage_status_json"][0]) == _stages(failed_json)
    assert out["recognizer_results_json"][1] is not None


def test_recognizer_row_id_derivation_matches_add_row_id(recognizer):
    from tide2.runner.local_runner import add_row_id

    patient_ids = [float("nan"), None, 123.0, 1.5]
    hashes = [f"h{i}" for i in range(len(patient_ids))]
    out = recognizer.process_batch({"text_hash": hashes, "note_text": ["x"] * len(hashes), "patient_id": patient_ids})
    derived = add_row_id(pa.table({"text_hash": hashes, "patient_id": pa.array(patient_ids, pa.float64())}))
    assert out["row_id"] == derived["row_id"].to_pylist()


# ---------------------------------------------------------------------------
# Transformer and BIO aggregation
# ---------------------------------------------------------------------------


def _fused_actor(**attrs):
    actor = object.__new__(TransformerInferenceActor)
    actor._aggregate_bio = True
    actor._log_gpu_mem = lambda *_a, **_k: None
    actor._run_inference_raw_with_oom_recovery = lambda texts: [[] for _ in texts]
    for name, value in attrs.items():
        setattr(actor, name, value)
    return actor


def test_transformer_fails_one_note_when_its_aggregation_fails():
    def format_note(_preds, note_text):
        if "BAD" in note_text:
            raise RuntimeError("malformed prediction for John Smith")
        return "[]", 0

    actor = _fused_actor(_format_note=format_note)
    out = actor({"text_hash": ["a", "b"], "note_text": ["fine", "BAD note"], "patient_id": ["p", "q"]})

    assert out["processing_status"] == ["success", "failed"]
    assert out["recognizer_results_json"] == ["[]", None]
    assert out["entity_count"] == [0, None]
    assert out["text_hash"] == ["a", "b"]
    assert "John Smith" not in out["stage_status_json"][1]
    assert _stages(out["stage_status_json"][1])["transformer"] == {"status": "failed", "reason": "RuntimeError"}


def test_transformer_fails_every_note_in_the_batch_when_the_forward_fails():
    def forward(_texts):
        raise RuntimeError("CUDA OOM on a single token window")

    actor = _fused_actor(_run_inference_raw_with_oom_recovery=forward, _format_note=lambda *_a: ("[]", 0))
    out = actor({"text_hash": ["a", "b", "c"], "note_text": ["one", "two", ""], "patient_id": [1, 2, 3]})

    assert out["processing_status"] == ["failed", "failed", "success"]
    assert out["recognizer_results_json"] == [None, None, "[]"]
    assert len(out["text_hash"]) == 3


def test_raw_transformer_path_reports_status_too():
    actor = _fused_actor(_aggregate_bio=False)
    out = actor({"text_hash": ["a"], "note_text": ["one"], "patient_id": [1]})
    assert out["processing_status"] == ["success"]
    assert out["predictions_raw_json"] == ["[]"]


def test_bio_aggregation_fails_a_note_and_passes_upstream_failures_through(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("malformed prediction")

    monkeypatch.setattr("tide2.actors.transformer.format_note_entities", boom)
    actor = BIOAggregationActor("m", model_to_presidio_mapping={}, ignore_labels={"O"})
    failed_json, _ = append_status(None, "transformer", "failed", "RuntimeError")
    out = actor(
        {
            "text_hash": ["a", "b"],
            "note_text": ["John", "Mary"],
            "predictions_raw_json": ["[]", None],
            "patient_id": ["p", "q"],
            "processing_status": ["success", "failed"],
            "stage_status_json": [None, failed_json],
        }
    )
    assert out["processing_status"] == ["failed", "failed"]
    assert _py(out["recognizer_results_json"]) == [None, None]
    assert _stages(out["stage_status_json"][0])["bio_aggregation"]["reason"] == "RuntimeError"
    assert _stages(out["stage_status_json"][1]) == _stages(failed_json)


# ---------------------------------------------------------------------------
# Anonymizer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anonymizer():
    return AnonymizerWorker(salt=SALT, key=KEY)


def _spans(*items):
    return json.dumps([{"entity_type": t, "start": s, "end": e, "score": 1.0} for t, s, e in items])


def _note(anonymizer, text, spans, **kwargs):
    return anonymizer.process_note(
        note_text=text,
        original_text_hash="h" * 64,
        recognizer_results_json=spans,
        patient_id="p1",
        jitter=kwargs.pop("jitter", 5),
        **kwargs,
    )


TEXT = "Pt John Smith seen 03/04/2020 MRN 12345678."
NAME = (3, 13)
DATE = (19, 29)
MRN = (34, 42)


@pytest.mark.parametrize(
    "bad_results",
    [
        None,
        float("nan"),
        "",
        "not json",
        '{"a": 1}',
        '["PERSON"]',
        '[{"entity_type": "PERSON"}]',
        '[{"entity_type": "PERSON", "start": 3}]',
        '[{"entity_type": "PERSON", "start": "3", "end": "13"}]',
        '[{"entity_type": "PERSON", "start": 3.0, "end": 13.0}]',
        '[{"entity_type": "PERSON", "start": -1, "end": 13}]',
        '[{"entity_type": "PERSON", "start": 13, "end": 3}]',
        '[{"entity_type": "PERSON", "start": 3, "end": 5000}]',
    ],
)
def test_anonymizer_fails_the_note_when_results_are_unusable(anonymizer, bad_results):
    out = anonymizer.process_batch(
        {
            "text_hash": ["h"],
            "note_text": [TEXT],
            "recognizer_results_json": [bad_results],
            "patient_id": ["p1"],
            "row_id": ["r1"],
        }
    )
    assert out["processing_status"] == ["failed"]
    assert _py(out["anonymized_note_text"]) == [None]
    assert _py(out["anonymizer_results_json"]) == [None]
    assert out["row_id"] == ["r1"]
    assert "John Smith" not in out["stage_status_json"][0]


def test_anonymizer_accepts_empty_results_and_empty_spans(anonymizer):
    for results in ("[]", _spans(("PERSON", 5, 5))):
        out = _note(anonymizer, TEXT, results)
        assert out["anonymized_note_text"] == TEXT
        assert out["stage_status"] == "success"


def test_an_invalid_jitter_masks_only_the_dates(anonymizer):
    out = _note(anonymizer, TEXT, _spans(("PERSON", *NAME), ("DATE", *DATE), ("MRN", *MRN)), jitter=1)

    assert out["stage_status"] == "failed"
    assert out["stage_reason"] == "DATE:ValueError"
    text = out["anonymized_note_text"]
    assert "[DATE]" in text
    assert "03/04/2020" not in text
    assert "John Smith" not in text
    assert "12345678" not in text
    assert "[PERSON]" not in text and "[MRN]" not in text


def test_a_missing_required_jitter_masks_the_dates_and_fails_the_note():
    worker = AnonymizerWorker(salt=SALT, key=KEY, jitter_required=True)
    out = worker.process_note(
        note_text=TEXT,
        original_text_hash="h" * 64,
        recognizer_results_json=_spans(("PERSON", *NAME), ("DATE", *DATE)),
        patient_id="p1",
        jitter=None,
    )
    assert out["stage_status"] == "failed"
    assert out["stage_reason"] == "DATE:jitter_required"
    assert "[DATE]" in out["anonymized_note_text"]
    assert "John Smith" not in out["anonymized_note_text"]


@pytest.mark.parametrize("age", ["ninety-five", "one hundred and two", "95+", "95", "aged 9x", "elderly"])
def test_unrecognized_ages_become_a_visible_fallback(anonymizer, age):
    text = f"Patient is {age} today."
    start = text.index(age)
    out = _note(anonymizer, text, _spans(("AGE", start, start + len(age))))
    assert out["anonymized_note_text"] == "Patient is [AGE] today."
    assert out["stage_status"] == "degraded"
    assert out["stage_reason"] == "AGE"


def test_recognized_ages_are_capped_or_rewritten_without_a_fallback(anonymizer):
    text = "Seen at 102 y/o."
    out = _note(anonymizer, text, _spans(("AGE", 8, 15)))
    assert out["anonymized_note_text"] == "Seen at 89 y/o."
    assert out["stage_status"] == "success"


def test_unknown_entity_type_gets_a_placeholder_not_a_deletion(anonymizer):
    out = _note(anonymizer, "Code FOO123 here", _spans(("FOO", 5, 11)))
    assert out["anonymized_note_text"] == "Code [FOO] here"
    assert (out["stage_status"], out["stage_reason"]) == ("degraded", "FOO")


@pytest.mark.parametrize("entity_type", ["OTHER", "BASE64_IMAGE"])
def test_configured_redactions_keep_a_placeholder(anonymizer, entity_type):
    out = _note(anonymizer, "see XXXXXX end", _spans((entity_type, 4, 10)))
    assert out["anonymized_note_text"] == f"see [{entity_type}] end"
    assert out["stage_status"] == "success"


def test_unparsable_date_uses_the_existing_default_and_is_degraded(anonymizer):
    out = _note(anonymizer, "Seen on 13/45/2020 only", _spans(("DATE", 8, 18)))
    assert out["anonymized_note_text"] == "Seen on [DATE] only"
    assert out["stage_status"] == "degraded"


def test_deliberate_pass_throughs_stay_successful(anonymizer):
    out = _note(anonymizer, "Born in 2016 here", _spans(("DATE", 8, 12)))
    assert out["anonymized_note_text"] == "Born in 2016 here"
    assert out["stage_status"] == "success"


def test_an_operator_error_masks_one_entity_and_keeps_the_others(anonymizer, monkeypatch):
    from tide2.cryptographic.fpe_strings import FormatPreservingEncryption

    def boom(self, content, format_type):
        raise RuntimeError(f"cipher failed for {content}")

    monkeypatch.setattr(FormatPreservingEncryption, "_encrypt_content", boom)
    out = _note(anonymizer, TEXT, _spans(("PERSON", *NAME), ("MRN", *MRN)))

    assert out["stage_status"] == "failed"
    assert out["stage_reason"] == "MRN:RuntimeError"
    assert "[MRN]" in out["anonymized_note_text"]
    assert "12345678" not in out["anonymized_note_text"]
    assert "John Smith" not in out["anonymized_note_text"]
    assert "[PERSON]" not in out["anonymized_note_text"]


def test_fpe_encrypt_raises_instead_of_returning_the_plaintext(monkeypatch):
    from tide2.cryptographic.fpe_strings import FormatPreservingEncryption

    fpe = FormatPreservingEncryption(SALT, KEY)
    monkeypatch.setattr(FormatPreservingEncryption, "_encrypt_content", lambda *_a: (_ for _ in ()).throw(ValueError()))
    with pytest.raises(ValueError):
        fpe.encrypt("MRN-12345678")


def test_guarded_operator_masks_empty_output_and_records_it():
    from presidio_anonymizer.operators import Operator
    from presidio_anonymizer.operators import OperatorType

    class Deleting(Operator):
        def operate(self, text, params=None):
            return ""

        def validate(self, params=None):
            pass

        def operator_name(self):
            return "deleting"

        def operator_type(self):
            return OperatorType.Anonymize

    events = start_note()
    assert guarded(Deleting)().operate("secret", {"entity_type": "PERSON"}) == "[PERSON]"
    assert summarize(events) == ("degraded", "PERSON")


def test_guarded_operators_are_inert_outside_a_note():
    from tide2.anonymizers.age_grouping import AgeGroupAnonymizer

    assert AgeGroupAnonymizer().operate("elderly", {"entity_type": "AGE"}) == "[AGE]"


def test_anonymizer_passes_failed_upstream_rows_through(anonymizer):
    failed_json, _ = append_status(None, "recognizer", "failed", "NoteError:cached_results")
    out = anonymizer.process_batch(
        {
            "text_hash": ["h"],
            "note_text": [TEXT],
            "recognizer_results_json": [None],
            "patient_id": ["p1"],
            "row_id": ["r1"],
            "processing_status": ["failed"],
            "stage_status_json": [failed_json],
        }
    )
    assert out["processing_status"] == ["failed"]
    assert _py(out["anonymized_note_text"]) == [None]
    assert _stages(out["stage_status_json"][0]) == _stages(failed_json)


def test_anonymizer_batch_keeps_one_row_per_input_row_and_never_logs_text(anonymizer, caplog):
    caplog.set_level(logging.DEBUG)
    out = anonymizer.process_batch(
        {
            "text_hash": ["a", "b", "c"],
            "note_text": [TEXT, TEXT, TEXT],
            "recognizer_results_json": ["[]", "not json", _spans(("AGE", 3, 8))],
            "patient_id": ["p1", "p2", "p3"],
            "row_id": ["r1", "r2", "r3"],
        }
    )
    assert out["row_id"] == ["r1", "r2", "r3"]
    assert out["processing_status"] == ["success", "failed", "degraded"]
    assert "John Smith" not in caplog.text


# ---------------------------------------------------------------------------
# Merge mode
# ---------------------------------------------------------------------------


def test_merge_takes_the_worst_branch_status_and_nulls_failed_rows():
    regex_ok, _ = append_status(None, "recognizer", "success")
    llm_failed, _ = append_status(None, "llm_recognizer", "failed", "LlmError")
    batch = pa.table(
        {
            "row_id": ["r1", "r2"],
            "text_hash_regex": ["h1", "h2"],
            "results_regex": ["[]", "[]"],
            "results_llm": ["[]", None],
            "processing_status_regex": ["success", "success"],
            "processing_status_llm": ["success", "failed"],
            "stage_status_json_regex": [regex_ok, regex_ok],
            "stage_status_json_llm": [regex_ok, llm_failed],
        }
    )
    out = _resolve_merged_batch(batch).to_pydict()

    assert out["processing_status"] == ["success", "failed"]
    assert out["recognizer_results_json"] == ["[]", None]
    assert out["entity_count"] == [0, None]
    assert set(_stages(out["stage_status_json"][1])) == {"recognizer", "llm_recognizer"}
    assert not any(name.endswith(("_regex", "_llm")) and name.startswith(("processing", "stage")) for name in out)


# ---------------------------------------------------------------------------
# Column types stay stable when a whole batch fails
# ---------------------------------------------------------------------------


def test_all_null_result_columns_keep_their_arrow_type():
    from tide2.utils.batch_columns import type_all_null_columns

    res = type_all_null_columns(
        {
            "recognizer_results_json": [None, None],
            "entity_count": [None, None],
            "anonymized_note_text": [None, "x"],
            "patient_id": [None, None],
        }
    )
    assert res["recognizer_results_json"].type == pa.string()
    assert res["entity_count"].type == pa.int64()
    assert res["anonymized_note_text"] == [None, "x"]
    assert res["patient_id"].type == pa.string()


def test_all_null_passthrough_columns_keep_their_arrow_type():
    from tide2.utils.batch_columns import type_all_null_columns

    res = type_all_null_columns(
        {
            "patient_identifiers": [None, None],
            "patient_id": [None, None],
            "jitter": [None, None],
            "row_id": [None, None],
        }
    )
    assert res["patient_identifiers"].type == pa.string()
    assert res["patient_id"].type == pa.string()
    assert res["jitter"].type == pa.int64()
    assert res["row_id"].type == pa.string()

    partial = type_all_null_columns({"jitter": [None, 3], "patient_id": ["p1", None]})
    assert partial == {"jitter": [None, 3], "patient_id": ["p1", None]}


@pytest.mark.parametrize("null_block_first", [True, False])
def test_directory_with_an_all_null_passthrough_block_reads_back(tmp_path, null_block_first):
    """Blocks with and without patient_id and jitter must read as one dataset in any file order."""
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq
    from ray.data.block import BlockAccessor

    from tide2.utils.batch_columns import type_all_null_columns

    def block(patient_id, jitter, patient_identifiers):
        batch = type_all_null_columns(
            {
                "row_id": ["r1", "r2"],
                "patient_id": patient_id,
                "jitter": jitter,
                "patient_identifiers": patient_identifiers,
            }
        )
        return BlockAccessor.batch_to_block(batch)

    blocks = [block([None, None], [None, None], [None, None]), block(["p1", "p2"], [3, 4], ["{}", "{}"])]
    if not null_block_first:
        blocks.reverse()
    out = tmp_path / "out"
    out.mkdir()
    for i, tbl in enumerate(blocks):
        pq.write_table(tbl, out / f"{i}.parquet")

    for read in (lambda: pq.read_table(out), lambda: pads.dataset(out).to_table()):
        table = read()
        assert table.num_rows == 4
        assert sorted(table["row_id"].to_pylist()) == ["r1", "r1", "r2", "r2"]
        assert table["patient_id"].null_count == 2
        assert table["jitter"].null_count == 2


@pytest.mark.integration
def test_a_fully_failed_batch_still_writes_a_readable_parquet_directory(tmp_path):
    """Ray writes one file per block; a null-typed file must not break reading the directory."""
    import pyarrow.dataset as pads
    import ray
    import ray.data

    rows = [
        {
            "text_hash": f"h{i}",
            "note_text": "Patient seen.",
            "patient_id": f"p{i}",
            "row_id": f"r{i}",
            "recognizer_results_json": "not json" if i < 10 else "[]",
        }
        for i in range(20)
    ]
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    try:
        ds = ray.data.from_items(rows, override_num_blocks=4).map_batches(
            RecognizerWorker,
            batch_size=5,
            batch_format="numpy",
            compute=ray.data.ActorPoolStrategy(size=1),
        )
        ds.write_parquet(str(tmp_path / "out"))
        table = pads.dataset(str(tmp_path / "out")).to_table()
    finally:
        ray.shutdown()

    assert table.num_rows == 20
    assert table["recognizer_results_json"].null_count == 10
    assert sorted(table["processing_status"].to_pylist()).count("failed") == 10
