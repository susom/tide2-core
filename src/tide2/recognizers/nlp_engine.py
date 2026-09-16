"""
Minimal spaCy-backed NLP engine for regex-only Presidio recognition.

``AnalyzerEngine`` requires an ``NlpEngine`` even when no ML/NER recognizers are
registered. ``_BlankSpacyNlpEngine`` provides tokenization only (via
``spacy.blank``), skipping the NER/POS/DEP pipelines of a full spaCy model.
"""

from presidio_analyzer.nlp_engine import NerModelConfiguration
from presidio_analyzer.nlp_engine import SpacyNlpEngine


class _BlankSpacyNlpEngine(SpacyNlpEngine):
    """SpacyNlpEngine backed by a pre-loaded ``spacy.blank`` model (tokenization only).

    Sets the required attributes directly instead of calling ``super().__init__()``.
    Newer presidio versions eagerly call ``spacy.load("en_core_web_lg")`` inside
    ``SpacyNlpEngine.__init__``; that model is neither needed (we only tokenize) nor
    shipped in our images, so calling the parent constructor raises
    ``OSError: [E050] Can't find model 'en_core_web_lg'``. Bypassing it keeps this
    engine correct on both lazy- and eager-loading presidio builds.
    """

    def __init__(self, loaded_spacy_model):
        self.nlp = {"en": loaded_spacy_model}
        self.models = [{"lang_code": "en", "model_name": "blank"}]
        self.ner_model_configuration = NerModelConfiguration()
