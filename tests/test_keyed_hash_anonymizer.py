"""Unit tests for KeyedHashAnonymizer."""

import re

import pytest
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.entities import RecognizerResult
from presidio_anonymizer.operators import OperatorType

from tide2.anonymizers.keyed_hash import KeyedHashAnonymizer

SALT = b"\x01" * 32
KEY = b"\x02" * 32
PARAMS = {"salt": SALT, "key": KEY}
GOLDEN = "817533D5ED213D8014F41CCBD96BF2F4A751032A564189EFA94EE24078B31169"


@pytest.fixture
def operator() -> KeyedHashAnonymizer:
    return KeyedHashAnonymizer()


def test_metadata(operator: KeyedHashAnonymizer) -> None:
    assert operator.operator_name() == "keyed_hash"
    assert operator.operator_type() == OperatorType.Anonymize


def test_golden(operator: KeyedHashAnonymizer) -> None:
    assert operator.operate("MRN123", PARAMS) == GOLDEN


def test_format(operator: KeyedHashAnonymizer) -> None:
    assert re.fullmatch(r"[0-9A-F]{64}", operator.operate("MRN123", PARAMS))


def test_deterministic(operator: KeyedHashAnonymizer) -> None:
    assert operator.operate("MRN123", PARAMS) == operator.operate("MRN123", PARAMS)


def test_keyed(operator: KeyedHashAnonymizer) -> None:
    base = operator.operate("MRN123", PARAMS)
    assert base != operator.operate("MRN124", PARAMS)
    assert base != operator.operate("MRN123", {"salt": b"\x03" * 32, "key": KEY})
    assert base != operator.operate("MRN123", {"salt": SALT, "key": b"\x03" * 32})


def test_normalized(operator: KeyedHashAnonymizer) -> None:
    assert operator.operate(" mrn123 ", PARAMS) == operator.operate("MRN123", PARAMS)


@pytest.mark.parametrize("name", ["salt", "key"])
@pytest.mark.parametrize("bad", [None, b"", "text"])
def test_validate(operator: KeyedHashAnonymizer, name: str, bad: object) -> None:
    operator.validate(PARAMS)
    with pytest.raises(ValueError, match=name):
        operator.validate({**PARAMS, name: bad})
    params = dict(PARAMS)
    del params[name]
    with pytest.raises(ValueError, match=name):
        operator.validate(params)


def test_engine(operator: KeyedHashAnonymizer) -> None:
    engine = AnonymizerEngine()
    engine.add_anonymizer(KeyedHashAnonymizer)
    result = engine.anonymize(
        "MRN 123456 seen",
        [RecognizerResult("MRN", 4, 10, 1.0)],
        {"MRN": OperatorConfig("keyed_hash", PARAMS)},
    )
    assert result.text == f"MRN {operator.operate('123456', PARAMS)} seen"
