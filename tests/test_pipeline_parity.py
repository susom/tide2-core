"""Discrete vs streamed equivalence for ``run_pipeline``.

``execution_mode="streamed"`` changes orchestration only: no line under
``process_batch``, the recognizers, the anonymizers, span resolution, HIPS, FPE
or date jitter is touched. So the two modes must produce **equivalent** output,
which this asserts per ``row_id``:

- identical ``anonymized_note_text``, ``entity_count``,
  ``anonymizer_results_json`` (modulo key order) and ``processing_status``;
- **equal row sets**, not merely overlapping;
- matching entity totals by type.

Byte parity is not expected: ``processing_timestamp`` is wall-clock and row
order is not preserved (``preserve_order=False``, and streaming changes block
scheduling). Everything else is deterministic, because anonymization is keyed
off ``salt``/``key``/``patient_id`` and jitter is derived per patient.

The GPU transformer is replaced by a deterministic stub actor — both modes go
through ``tide2.actors.create_transformer_actor``, so the stub exercises the
same chaining without a model download. The recognizer and anonymizer stages
are the real ones.
"""

import json
import sys
from collections import Counter
from datetime import UTC
from datetime import datetime

import pandas as pd
import pyarrow.dataset as pads
import pytest
from ray import cloudpickle

import tide2.actors
import tide2.transformers.config
import tide2.utils.gcs_resource_manager
from tide2.runner.local_runner import LocalJobRunner

pytestmark = pytest.mark.integration

# Ray pickles actor classes by reference, and `tests/` is not importable inside a
# Ray worker. Serializing this module by value ships the stub actor with the task.
cloudpickle.register_pickle_by_value(sys.modules[__name__])

SALT_HEX = "ab" * 32
KEY_HEX = "cd" * 32
STUB_ENTITY = "John Doe"


class StubTransformerActor:
    """Deterministic stand-in for the GPU NER actor.

    Emits the same output shape as ``TransformerInferenceActor`` in its
    ``aggregate_bio=True`` mode (``num_agg_actors=0``), tagging every literal
    occurrence of ``STUB_ENTITY`` as a PERSON.
    """

    def __call__(self, batch):
        note_texts = [str(t) for t in batch["note_text"]]
        results, counts = [], []
        for text in note_texts:
            entities = []
            start = text.find(STUB_ENTITY)
            while start != -1:
                entities.append(
                    {
                        "entity_type": "PERSON",
                        "start": start,
                        "end": start + len(STUB_ENTITY),
                        "score": 0.99,
                        "analysis_explanation": None,
                        "recognition_metadata": {},
                    }
                )
                start = text.find(STUB_ENTITY, start + 1)
            results.append(json.dumps(entities))
            counts.append(len(entities))

        timestamp = datetime.now(tz=UTC).isoformat()
        res = {
            "text_hash": [str(h) for h in batch["text_hash"]],
            "patient_id": [str(p) for p in batch["patient_id"]],
            "note_text": note_texts,
            "recognizer_results_json": results,
            "entity_count": counts,
            "processing_timestamp": [timestamp] * len(note_texts),
        }
        for col in ("patient_identifiers", "jitter", "row_id"):
            if col in batch:
                res[col] = list(batch[col])
        return res


@pytest.fixture
def stub_transformer(monkeypatch):
    """Point both execution modes at the stub actor and skip the model download."""
    monkeypatch.setattr(tide2.actors, "create_transformer_actor", lambda **_k: StubTransformerActor)
    monkeypatch.setattr(
        tide2.transformers.config,
        "load_model_config",
        lambda _name: {"MODEL_MAX_LENGTH": 512, "CHUNK_OVERLAP_SIZE": 40},
    )
    monkeypatch.setattr(tide2.utils.gcs_resource_manager, "resolve_model_path", lambda **_k: "/nonexistent/model")


@pytest.fixture
def notes():
    """20 notes with repeated patients, so per-patient jitter is exercised."""
    rows = []
    for i in range(20):
        rows.append(
            {
                "note_text": (
                    f"Patient {STUB_ENTITY} (MRN 1234{i:02d}) was seen on 03/1{i % 9}/2021 "
                    f"at 555-010{i % 9}-2020. Follow-up scheduled."
                ),
                "patient_id": f"patient-{i % 5}",
            }
        )
    return pd.DataFrame(rows)


def read_output(output_dir):
    files = list((output_dir / "06_anonymizer_output").glob("**/*.parquet"))
    assert files, f"no anonymizer output under {output_dir}"
    return pads.dataset(files).to_table().to_pandas()


def run_mode(tmp_path, notes, mode):
    runner = LocalJobRunner()
    try:
        manifest = runner.run_pipeline(
            input_data=notes,
            output_dir=str(tmp_path / mode),
            model_name="stub-model",
            salt_hex=SALT_HEX,
            key_hex=KEY_HEX,
            hardware_autotune=False,
            execution_mode=mode,
            transformer_kwargs={"num_gpus": 0, "num_transformer_actors": 1, "num_agg_actors": 0},
            recognizer_kwargs={"num_actors": 2, "enable_checkpoint": False},
            anonymizer_kwargs={"num_actors": 2, "enable_checkpoint": False},
        )
    finally:
        runner.shutdown()
    return manifest, read_output(tmp_path / mode)


def canonical_items(raw):
    """Anonymizer items with dict key order normalized away."""
    return [json.dumps(item, sort_keys=True) for item in json.loads(raw)]


def entity_totals(df):
    totals = Counter()
    for raw in df["anonymizer_results_json"]:
        for item in json.loads(raw):
            totals[item["entity_type"]] += 1
    return totals


class TestDiscreteStreamedEquivalence:
    def test_equivalent_output(self, tmp_path, notes, stub_transformer):
        discrete_manifest, discrete = run_mode(tmp_path, notes, "discrete")
        streamed_manifest, streamed = run_mode(tmp_path, notes, "streamed")

        assert discrete_manifest["execution_mode"] == "discrete"
        assert streamed_manifest["execution_mode"] == "streamed"
        assert streamed_manifest["input_rows"] == len(notes)
        assert streamed_manifest["dropped_rows"] == 0

        # Equal row sets, not merely overlapping.
        assert set(discrete["row_id"]) == set(streamed["row_id"])
        assert len(discrete) == len(streamed) == len(notes)

        d = discrete.set_index("row_id").sort_index()
        s = streamed.set_index("row_id").sort_index()
        for col in ("anonymized_note_text", "entity_count", "processing_status"):
            pd.testing.assert_series_equal(d[col], s[col], check_names=False)
        for row_id in d.index:
            assert canonical_items(d.loc[row_id, "anonymizer_results_json"]) == canonical_items(
                s.loc[row_id, "anonymizer_results_json"]
            )

        assert entity_totals(discrete) == entity_totals(streamed)

    def test_streamed_writes_no_intermediates(self, tmp_path, notes, stub_transformer):
        run_mode(tmp_path, notes, "streamed")
        out = tmp_path / "streamed"
        assert (out / "06_anonymizer_output").exists()
        assert not (out / "01_transformer_input.parquet").exists()
        assert not (out / "02_transformer_output").exists()
        assert not (out / "04_recognizer_output").exists()

    def test_raw_note_text_never_reaches_the_sink(self, tmp_path, notes, stub_transformer):
        _, streamed = run_mode(tmp_path, notes, "streamed")
        assert "note_text" not in streamed.columns
        assert "anonymized_note_text" in streamed.columns


class TestRowReconciliation:
    def test_total_failure_is_a_hard_error(self, tmp_path, notes, stub_transformer, monkeypatch):
        """A short write must not look like success."""

        class ExplodingAnonymizer:
            def __init__(self, *_a, **_k):
                raise RuntimeError("deliberate failure")

        monkeypatch.setattr(tide2.actors, "create_anonymizer_actor_class", lambda **_k: ExplodingAnonymizer)
        with pytest.raises(Exception, match=r"0 rows|deliberate failure|failed"):
            run_mode(tmp_path, notes, "streamed")
