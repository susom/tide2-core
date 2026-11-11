"""Tests for score thresholds: spans below the threshold of their recognizer type stay clear text."""

import json
from unittest.mock import MagicMock

import pytest

from tide2.actors.anonymizer import AnonymizerWorker
from tide2.anonymizers import cell
from tide2.anonymizers import free_text
from tide2.anonymizers.cell import AnonymizationPolicy
from tide2.runner import cli
from tide2.runner import local_runner

SALT = b"\x01" * 32
KEY = b"\x02" * 32
TRANSFORMER = {"recognizer_name": "TransformersRecognizer[SOME_MODEL]"}
CACHED = {"recognizer_name": "CachedResultsTransformerRecognizer"}
REGEX = {"recognizer_name": "PatternRecognizer"}

# Synthetic text only.
TEXT = "Patient Rhea Orleans, MRN 48291037, seen in Marlowe Clinic."
PERSON = (8, 20)
MRN = (26, 34)


def _span(entity_type: str, bounds: tuple[int, int], score: float, meta: dict | None) -> dict:
    span = {"entity_type": entity_type, "start": bounds[0], "end": bounds[1], "score": score}
    if meta is not None:
        span["recognition_metadata"] = meta
    return span


def _run(policy: AnonymizationPolicy, spans: list[dict], text: str = TEXT) -> str:
    worker = AnonymizerWorker(SALT, KEY, policy=policy)
    result = worker.process_note(text, worker.compute_text_hash(text), json.dumps(spans), "FAKE-P0001", 7)
    return result["anonymized_note_text"]


@pytest.fixture(autouse=True)
def _fresh_workers():
    free_text._worker.cache_clear()
    yield
    free_text._worker.cache_clear()


@pytest.mark.parametrize("bad", [-0.1, 1.1, True, "0.5", [0.5]])
@pytest.mark.parametrize("name", ["transformer_score_threshold", "recognizer_score_threshold"])
def test_policy_rejects_bad_threshold(name: str, bad: object) -> None:
    with pytest.raises(ValueError, match=name):
        AnonymizationPolicy(**{name: bad})


def test_policy_threshold_values() -> None:
    policy = AnonymizationPolicy()
    assert (policy.transformer_score_threshold, policy.recognizer_score_threshold) == (0.0, 0.0)
    policy = AnonymizationPolicy.from_mapping({"transformer_score_threshold": 1, "recognizer_score_threshold": 0.75})
    assert (policy.transformer_score_threshold, policy.recognizer_score_threshold) == (1.0, 0.75)
    assert AnonymizationPolicy.from_mapping({"transformer_score_threshold": None}).transformer_score_threshold == 0.0


def test_default_anonymizes_every_span() -> None:
    spans = [_span("PERSON", PERSON, 0.01, TRANSFORMER), _span("MRN", MRN, 0.01, REGEX)]
    out = _run(AnonymizationPolicy(), spans)
    assert "Rhea Orleans" not in out
    assert "48291037" not in out


@pytest.mark.parametrize("meta", [TRANSFORMER, CACHED])
def test_transformer_threshold_drops_low_score(meta: dict) -> None:
    out = _run(AnonymizationPolicy(transformer_score_threshold=0.6), [_span("PERSON", PERSON, 0.4, meta)])
    assert out == TEXT


def test_transformer_threshold_keeps_score_at_or_above() -> None:
    policy = AnonymizationPolicy(transformer_score_threshold=0.6)
    assert "Rhea Orleans" not in _run(policy, [_span("PERSON", PERSON, 0.8, TRANSFORMER)])
    assert "Rhea Orleans" not in _run(policy, [_span("PERSON", PERSON, 0.6, TRANSFORMER)])


def test_recognizer_threshold_drops_low_score_regex() -> None:
    policy = AnonymizationPolicy(recognizer_score_threshold=0.7)
    assert _run(policy, [_span("MRN", MRN, 0.5, REGEX)]) == TEXT
    assert "48291037" not in _run(policy, [_span("MRN", MRN, 0.9, REGEX)])


def test_span_without_metadata_counts_as_recognizer() -> None:
    spans = [_span("MRN", MRN, 0.5, None)]
    assert _run(AnonymizationPolicy(transformer_score_threshold=0.9), spans) != TEXT
    assert _run(AnonymizationPolicy(recognizer_score_threshold=0.9), spans) == TEXT


def test_thresholds_are_independent() -> None:
    spans = [_span("PERSON", PERSON, 0.85, TRANSFORMER), _span("MRN", MRN, 0.6, REGEX)]
    out = _run(AnonymizationPolicy(transformer_score_threshold=0.8), spans)
    assert "Rhea Orleans" not in out
    assert "48291037" not in out
    out = _run(AnonymizationPolicy(recognizer_score_threshold=0.9), spans)
    assert "Rhea Orleans" not in out
    assert "48291037" in out


def test_dropped_transformer_span_does_not_hide_overlapping_regex_span() -> None:
    # The longer, low-score transformer span would win "longest_wins" and then be left clear.
    spans = [_span("PERSON", (26, 40), 0.3, TRANSFORMER), _span("MRN", MRN, 0.8, REGEX)]
    out = _run(AnonymizationPolicy(transformer_score_threshold=0.5), spans)
    assert "48291037" not in out
    assert "MRN" in out


def test_free_text_cell_obeys_thresholds() -> None:
    spans = json.dumps([_span("PERSON", PERSON, 0.4, TRANSFORMER), _span("MRN", MRN, 0.9, REGEX)])
    policy = AnonymizationPolicy(transformer_score_threshold=0.6)
    kwargs = {"patient_id": "FAKE-P0001", "jitter": 7, "recognizer_results": spans, "policy": policy}
    out = cell.anonymize_cell(TEXT, "FREE_TEXT", SALT, KEY, **kwargs)
    assert out is not None
    assert "Rhea Orleans" in out
    assert "48291037" not in out
    assert out == _run(policy, json.loads(spans))


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    runner = MagicMock()
    runner.run_anonymization.return_value = {}
    runner.run_pipeline.return_value = {}
    monkeypatch.setattr(local_runner, "LocalJobRunner", lambda **_kw: runner)
    return runner


def test_cli_flags_forwarded(fake_runner: MagicMock) -> None:
    flags = ["--transformer-score-threshold", "0.7", "--recognizer-score-threshold", "0.4"]
    cli.main(["run", "anonymizer", "--input", "in", "--output", "out", "--salt", "s", "--key", "k", *flags])
    kwargs = fake_runner.run_anonymization.call_args.kwargs
    assert (kwargs["transformer_score_threshold"], kwargs["recognizer_score_threshold"]) == (0.7, 0.4)
    cli.main(["run", "pipeline", "--input", "in", "--output", "out", "--model", "m", *flags])
    a_kw = fake_runner.run_pipeline.call_args.kwargs["anonymizer_kwargs"]
    assert (a_kw["transformer_score_threshold"], a_kw["recognizer_score_threshold"]) == (0.7, 0.4)


def test_cli_flags_default_to_policy_default(fake_runner: MagicMock) -> None:
    cli.main(["run", "anonymizer", "--input", "in", "--output", "out", "--salt", "s", "--key", "k"])
    kwargs = fake_runner.run_anonymization.call_args.kwargs
    assert kwargs["transformer_score_threshold"] is None
    assert kwargs["recognizer_score_threshold"] is None
    policy = AnonymizationPolicy.from_mapping(kwargs)
    assert (policy.transformer_score_threshold, policy.recognizer_score_threshold) == (0.0, 0.0)


@pytest.mark.parametrize("bad", ["-0.1", "1.5"])
def test_run_anonymization_rejects_bad_threshold_before_ray(bad: str) -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        local_runner.LocalJobRunner.run_anonymization(
            MagicMock(), "in", "out", salt_path="s", key_path="k", transformer_score_threshold=float(bad)
        )
