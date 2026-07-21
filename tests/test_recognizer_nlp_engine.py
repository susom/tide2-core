"""
Regression tests for ``_BlankSpacyNlpEngine``.

These guard the fix that stops ``RecognizerWorker`` actors from crashing with
``OSError: [E050] Can't find model 'en_core_web_lg'`` on newer, eager-loading
presidio builds. The engine must construct without calling
``SpacyNlpEngine.__init__`` (which eagerly loads ``en_core_web_lg``) while still
driving regex-based recognition through a blank spaCy tokenizer.
"""

import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer import RecognizerRegistry

from tide2.actors.recognizer import _BlankSpacyNlpEngine
from tide2.recognizers import PhoneRecognizer


class TestBlankSpacyNlpEngine:
    """Test _BlankSpacyNlpEngine construction and behavior."""

    def test_construction_and_regex_recognition(self):
        """Engine builds, reports loaded, and drives regex recognition.

        Wires the blank engine into an AnalyzerEngine exactly as
        RecognizerWorker.__init__ does, then confirms a regex entity is found —
        proving the blank tokenizer path still recognizes entities.
        """
        blank_nlp = spacy.blank("en")
        blank_nlp.max_length = 2_000_000
        nlp_engine = _BlankSpacyNlpEngine(loaded_spacy_model=blank_nlp)

        assert nlp_engine.is_loaded() is True

        registry = RecognizerRegistry()
        registry.add_recognizer(PhoneRecognizer())
        registry.remove_recognizer("SpacyRecognizer")

        analyzer = AnalyzerEngine(
            registry=registry,
            nlp_engine=nlp_engine,
            supported_languages=["en"],
        )

        results = analyzer.analyze("Call 555-123-4567", language="en")
        assert any(r.entity_type == "PHONE" for r in results)

    def test_does_not_call_super_init(self, monkeypatch):
        """Construction survives an eager-loading presidio build.

        Simulate the newer presidio whose SpacyNlpEngine.__init__ eagerly calls
        spacy.load("en_core_web_lg") by making that constructor raise the E050
        OSError. Because _BlankSpacyNlpEngine no longer calls super().__init__(),
        it must still construct.
        """

        def _boom(self, *args, **kwargs):
            raise OSError("[E050] Can't find model 'en_core_web_lg'.")

        monkeypatch.setattr(
            "presidio_analyzer.nlp_engine.SpacyNlpEngine.__init__",
            _boom,
        )

        engine = _BlankSpacyNlpEngine(loaded_spacy_model=spacy.blank("en"))
        assert engine.is_loaded() is True
