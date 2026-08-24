"""Tests for the deberta/openmed PII model configs and the config-driven
TransformerCore behaviors they exercise.

Covers:
- C1: MODEL_TO_PRESIDIO_MAPPING for both new models is populated, every raw
  label maps to a supported entity, and the mapping+anonymization path works
  end-to-end on a synthetic note (mocked pipeline labels -> RecognizerResults ->
  presidio anonymization) for one label per operator class.
- C3: the single-text inference path enforces the tokenizer's model_max_length.
- H4 / L12: per-model DTYPE reaches ``from_pretrained`` on both GPU and CPU load
  paths (float32 pin honored on CPU, never a blind float16 default), and
  MODEL_MAX_LENGTH pins the tokenizer's max length used for truncation.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from tide2.actors.recognizer import ALL_SUPPORTED_ENTITIES
from tide2.recognizers.transformers_recognizer import TransformersRecognizer
from tide2.transformers.core import TransformerCore
from tide2.utils.resource_utils import BERT_TRANSFORMER_CONFIG_FILE
from tide2.utils.resource_utils import get_resource_path

DEBERTA = "lakshyakh93/deberta_finetuned_pii"
OPENMED = "OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1"


def _load_real_configs() -> dict:
    """Load the shipped bert_transformer_configuration.json."""
    with Path(get_resource_path(BERT_TRANSFORMER_CONFIG_FILE)).open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# C1: mapping completeness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", [DEBERTA, OPENMED])
def test_model_mapping_is_populated(model_name):
    """Both new models must ship a non-empty MODEL_TO_PRESIDIO_MAPPING (regression
    for the empty-mapping bug that dropped every label)."""
    cfg = _load_real_configs()[model_name]
    mapping = cfg["MODEL_TO_PRESIDIO_MAPPING"]
    assert mapping, f"{model_name} MODEL_TO_PRESIDIO_MAPPING must not be empty"


@pytest.mark.parametrize("model_name", [DEBERTA, OPENMED])
def test_every_raw_label_maps_to_supported_entity(model_name):
    """Every raw label resolves to a non-None entity that is in the runner
    allowlist, so nothing is silently dropped downstream."""
    cfg = _load_real_configs()[model_name]
    mapping = cfg["MODEL_TO_PRESIDIO_MAPPING"]

    for raw_label, target in mapping.items():
        assert target is not None, f"{model_name}:{raw_label} maps to None"
        assert target in ALL_SUPPORTED_ENTITIES, (
            f"{model_name}:{raw_label} -> {target} is not in ALL_SUPPORTED_ENTITIES"
        )


@pytest.mark.parametrize("model_name", [DEBERTA, OPENMED])
def test_every_supported_label_has_a_mapping(model_name):
    """Every PRESIDIO_SUPPORTED_ENTITIES raw label has a mapping entry (no label
    the model can emit is left unmapped)."""
    cfg = _load_real_configs()[model_name]
    mapping = cfg["MODEL_TO_PRESIDIO_MAPPING"]
    for raw_label in cfg["PRESIDIO_SUPPORTED_ENTITIES"]:
        assert raw_label in mapping, f"{model_name}: raw label {raw_label} has no mapping entry"


# ---------------------------------------------------------------------------
# C1: end-to-end mapping + anonymization path (mocked pipeline labels)
# ---------------------------------------------------------------------------


def _raw_pred(text: str, substring: str, label: str, score: float = 0.99) -> dict:
    """Build a single raw BIO token prediction located in ``text``."""
    start = text.index(substring)
    return {
        "entity": label,
        "score": score,
        "start": start,
        "end": start + len(substring),
        "word": substring,
        "index": 1,
    }


# Each case pairs a model with a synthetic note and a list of tuples holding a
# substring, its raw BIO label, and the expected presidio entity. One label is
# included per operator class: PERSON / LOCATION / DATE_TIME / ID / OTHER.
_E2E_CASES = [
    (
        DEBERTA,
        "John lives in Paris on 2020-01-01 ip 1.2.3.4 pw secretword",
        [
            ("John", "B-FIRSTNAME", "PERSON"),
            ("Paris", "B-CITY", "LOCATION"),
            ("2020-01-01", "B-DATE", "DATE_TIME"),
            ("1.2.3.4", "B-IP", "ID"),
            ("secretword", "B-PASSWORD", "OTHER"),
        ],
    ),
    (
        OPENMED,
        "Mary lives in Boston on 2019-05-05 acct 987654 pw hunter2pass",
        [
            ("Mary", "B-first_name", "PERSON"),
            ("Boston", "B-city", "LOCATION"),
            ("2019-05-05", "B-date", "DATE_TIME"),
            ("987654", "B-account_number", "ID"),
            ("hunter2pass", "B-password", "OTHER"),
        ],
    ),
]


@pytest.mark.parametrize(("model_name", "note", "cases"), _E2E_CASES)
def test_mapping_and_anonymization_path(model_name, note, cases):
    """Mocked pipeline labels flow through the real config mapping into
    RecognizerResults with the correct presidio entity types, and each span is
    anonymized end-to-end."""
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    from presidio_anonymizer.entities import RecognizerResult as AnonRecognizerResult

    raw_preds = [_raw_pred(note, sub, label) for sub, label, _ in cases]

    with patch("tide2.transformers.core.resolve_model_path", return_value="/fake/model/path"):
        recognizer = TransformersRecognizer(model_name=model_name)

        mock_pipeline = Mock()
        mock_pipeline.return_value = raw_preds
        mock_pipeline.tokenizer.model_max_length = 512
        recognizer._core._pipeline = mock_pipeline

        results = recognizer.analyze(note, ALL_SUPPORTED_ENTITIES)

    found_types = {r.entity_type for r in results}
    expected_types = {expected for _, _, expected in cases}
    assert expected_types <= found_types, f"{model_name}: expected {expected_types}, got {found_types}"
    # Every operator class we asked for must be represented.
    assert {"PERSON", "LOCATION", "DATE_TIME", "ID", "OTHER"} <= found_types

    engine = AnonymizerEngine()
    operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})}
    anon_results = [AnonRecognizerResult(r.entity_type, r.start, r.end, r.score) for r in results]
    anonymized = engine.anonymize(text=note, analyzer_results=anon_results, operators=operators).text

    for substring, _, _ in cases:
        assert substring not in anonymized, f"{model_name}: {substring!r} not anonymized in {anonymized!r}"


# ---------------------------------------------------------------------------
# C3: single-text truncation
# ---------------------------------------------------------------------------


def _make_core(model_name: str = DEBERTA) -> TransformerCore:
    """Build a TransformerCore without touching the network/filesystem model."""
    with patch("tide2.transformers.core.resolve_model_path", return_value="/fake/model/path"):
        return TransformerCore(model_name=model_name, device="cpu", load_immediately=False)


class _RealLikePipeline:
    """Mimics transformers>=5 TokenClassificationPipeline: its
    _sanitize_parameters does NOT accept truncation and has no **kwargs."""

    def __init__(self):
        self.calls = []
        self.tokenizer = SimpleNamespace(model_max_length=512)

    def _sanitize_parameters(
        self,
        ignore_labels=None,
        aggregation_strategy=None,
        offset_mapping=None,
        is_split_into_words=False,
        stride=None,
        delimiter=None,
    ):
        return {}, {}, {}

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return []


class _TruncationAwarePipeline:
    """A pipeline whose _sanitize_parameters explicitly accepts truncation and
    max_length (older-style / capable pipeline)."""

    def __init__(self):
        self.calls = []
        self.tokenizer = SimpleNamespace(model_max_length=512)

    def _sanitize_parameters(self, ignore_labels=None, truncation=None, max_length=None):
        return {}, {}, {}

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return []


def test_single_text_truncation_requested_when_supported():
    """When the pipeline accepts it, infer_single_raw requests truncation bounded
    by the resolved model_max_length."""
    core = _make_core()
    pipe = _TruncationAwarePipeline()
    core._pipeline = pipe

    core.infer_single_raw("some text")

    assert len(pipe.calls) == 1
    _, kwargs = pipe.calls[0]
    assert kwargs.get("truncation") is True
    assert kwargs.get("max_length") == 512


def test_single_text_no_unsupported_kwarg_on_transformers5():
    """transformers>=5 pipelines truncate internally and reject a truncation
    kwarg; infer_single_raw must not pass one (would raise TypeError)."""
    core = _make_core()
    pipe = _RealLikePipeline()
    core._pipeline = pipe

    core.infer_single_raw("some text")

    assert len(pipe.calls) == 1
    _, kwargs = pipe.calls[0]
    assert "truncation" not in kwargs
    assert "max_length" not in kwargs


def test_single_text_empty_returns_early():
    core = _make_core()
    pipe = _TruncationAwarePipeline()
    core._pipeline = pipe
    assert core.infer_single_raw("") == []
    assert pipe.calls == []


# ---------------------------------------------------------------------------
# H4 / L12: DTYPE on load paths + MODEL_MAX_LENGTH pin
# ---------------------------------------------------------------------------


def _create_temp_config(config_data: dict) -> str:
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config_data, temp_file)
    temp_file.flush()
    temp_file.close()
    return temp_file.name


def _base_model_config(**overrides) -> dict:
    cfg = {
        "PRESIDIO_SUPPORTED_ENTITIES": ["PERSON"],
        "LABELS_TO_IGNORE": ["O"],
        "DEFAULT_EXPLANATION": "test",
        "MODEL_TO_PRESIDIO_MAPPING": {"PER": "PERSON"},
        "CHUNK_OVERLAP_SIZE": 40,
        "CHUNK_SIZE": 512,
        "ID_ENTITY_NAME": "ID",
        "ID_SCORE_MULTIPLIER": 0.5,
    }
    cfg.update(overrides)
    return cfg


def _run_load_pipeline(model_cfg: dict, device: str | None, ctor_dtype=torch.float16, cuda_available: bool = False):
    """Load a pipeline with everything mocked, returning the from_pretrained call
    kwargs and the tokenizer mock so tests can assert dtype / model_max_length."""
    config_data = {"TEST_MODEL": model_cfg}
    config_path = _create_temp_config(config_data)

    mock_model_instance = Mock()
    mock_param = Mock()
    mock_param.device = "cpu"
    mock_model_instance.parameters.return_value = iter([mock_param])
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.to.return_value = mock_model_instance

    mock_tokenizer_instance = Mock()

    try:
        with (
            patch("tide2.transformers.config.get_resource_path", return_value=config_path),
            patch("tide2.transformers.core.resolve_model_path", return_value="/fake/model/path"),
            patch("tide2.transformers.core.pipeline") as mock_pipeline,
            patch("tide2.transformers.core.AutoModelForTokenClassification") as mock_model,
            patch("tide2.transformers.core.AutoTokenizer") as mock_tokenizer,
            patch("tide2.transformers.core.torch.cuda.is_available", return_value=cuda_available),
        ):
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            mock_pipeline.return_value = Mock()

            core = TransformerCore(
                model_name="TEST_MODEL",
                device=device,
                dtype=ctor_dtype,
                load_immediately=True,
            )
            assert core.is_loaded
            return mock_model.from_pretrained.call_args, mock_tokenizer_instance
    finally:
        Path(config_path).unlink()


def test_cpu_dtype_uses_config_dtype_when_set():
    """H4: on the explicit CPU path, a configured DTYPE reaches from_pretrained."""
    call_args, _ = _run_load_pipeline(_base_model_config(DTYPE="bfloat16"), device="cpu")
    assert call_args.kwargs.get("dtype") == torch.bfloat16


def test_cpu_dtype_defaults_to_float32_when_unset():
    """H4: on the explicit CPU path with no DTYPE, from_pretrained gets float32
    (never the float16 default, which CPU kernels can't run)."""
    call_args, _ = _run_load_pipeline(_base_model_config(), device="cpu", ctor_dtype=torch.float16)
    assert call_args.kwargs.get("dtype") == torch.float32


def test_autodetect_cpu_dtype_uses_config_dtype_when_set():
    """H4: on the auto-detected CPU fallback (no CUDA), a configured DTYPE
    reaches from_pretrained."""
    call_args, _ = _run_load_pipeline(
        _base_model_config(DTYPE="float32"), device=None, ctor_dtype=torch.float16, cuda_available=False
    )
    assert call_args.kwargs.get("dtype") == torch.float32


def test_autodetect_cpu_dtype_defaults_to_float32_when_unset():
    """H4: auto-detected CPU fallback with no DTYPE loads as float32."""
    call_args, _ = _run_load_pipeline(_base_model_config(), device=None, ctor_dtype=torch.float16, cuda_available=False)
    assert call_args.kwargs.get("dtype") == torch.float32


def test_explicit_gpu_dtype_reaches_from_pretrained():
    """L12: on the explicit cuda:N path, the configured DTYPE reaches
    from_pretrained. Fully mocked so it runs without a real GPU."""
    call_args, _ = _run_load_pipeline(_base_model_config(DTYPE="float32"), device="cuda:0")
    assert call_args.kwargs.get("dtype") == torch.float32


def test_autodetect_gpu_dtype_reaches_from_pretrained():
    """L12: on the auto-detected CUDA path, the configured DTYPE reaches
    from_pretrained. Uses a patched cuda.is_available so no real GPU is needed."""
    with patch("tide2.transformers.core.torch.cuda.current_device", return_value=0):
        call_args, _ = _run_load_pipeline(_base_model_config(DTYPE="float32"), device=None, cuda_available=True)
    assert call_args.kwargs.get("dtype") == torch.float32


def test_model_max_length_pins_tokenizer_when_set():
    """L12: MODEL_MAX_LENGTH pins the tokenizer's model_max_length (the value the
    pipeline uses to bound truncation)."""
    _, tokenizer = _run_load_pipeline(_base_model_config(MODEL_MAX_LENGTH=512), device="cpu")
    assert tokenizer.model_max_length == 512


def test_model_max_length_absent_leaves_tokenizer_untouched():
    """L12: without MODEL_MAX_LENGTH, the tokenizer's own max length is kept."""
    sentinel = object()
    config_data = {"TEST_MODEL": _base_model_config()}
    config_path = _create_temp_config(config_data)

    mock_model_instance = Mock()
    mock_param = Mock()
    mock_param.device = "cpu"
    mock_model_instance.parameters.return_value = iter([mock_param])
    mock_model_instance.eval.return_value = mock_model_instance

    mock_tokenizer_instance = Mock()
    mock_tokenizer_instance.model_max_length = sentinel

    try:
        with (
            patch("tide2.transformers.config.get_resource_path", return_value=config_path),
            patch("tide2.transformers.core.resolve_model_path", return_value="/fake/model/path"),
            patch("tide2.transformers.core.pipeline", return_value=Mock()),
            patch("tide2.transformers.core.AutoModelForTokenClassification") as mock_model,
            patch("tide2.transformers.core.AutoTokenizer") as mock_tokenizer,
            patch("tide2.transformers.core.torch.cuda.is_available", return_value=False),
        ):
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            TransformerCore(model_name="TEST_MODEL", device="cpu", load_immediately=True)
        assert mock_tokenizer_instance.model_max_length is sentinel
    finally:
        Path(config_path).unlink()
