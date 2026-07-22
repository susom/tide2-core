"""
Regression tests for the Presidio monkeypatches in
``tide2.anonymizers.presidio_patches``.

These lock in three things that a future presidio bump could silently break:

1. **Attribute-existence guards** — each presidio internal we patch still exists
   (under whichever name the installed version uses). This would have caught the
   2.2.363 rename of ``_merge_entities_with_whitespace_between`` ->
   ``_merge_entities_with_spaces_between`` in CI instead of at runtime.
2. **Round-trip** — every disable/enable and patch/unpatch pair restores the
   original method exactly.
3. **Behavioral equivalence** — two adjacent same-type entities separated by a
   single space stay UNMERGED with the patch/parameter and MERGE without it. This
   proves the no-merge behavior is genuinely exercised (not a no-op that happens
   to match) and pins the equivalence between the monkeypatch path and the public
   ``anonymize(merge_entities_with_spaces=False)`` parameter on every future bump.
"""

import inspect
from typing import ClassVar

import pytest
from presidio_analyzer import EntityRecognizer
from presidio_analyzer import RecognizerResult as AnalyzerRecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.entities import RecognizerResult

from tide2.anonymizers import presidio_patches


@pytest.fixture(autouse=True)
def _restore_patch_state():
    """Ensure every test starts and ends with all patches disabled."""
    if presidio_patches.is_whitespace_merging_disabled():
        presidio_patches.enable_whitespace_merging()
    if presidio_patches.is_remove_duplicates_patched():
        presidio_patches.unpatch_remove_duplicates()
    if presidio_patches.is_conflict_resolution_patched():
        presidio_patches.unpatch_conflict_resolution()
    yield
    if presidio_patches.is_whitespace_merging_disabled():
        presidio_patches.enable_whitespace_merging()
    if presidio_patches.is_remove_duplicates_patched():
        presidio_patches.unpatch_remove_duplicates()
    if presidio_patches.is_conflict_resolution_patched():
        presidio_patches.unpatch_conflict_resolution()


class TestAttributeExistenceGuards:
    """Guards that each patched presidio internal still exists."""

    def test_whitespace_merge_method_exists_under_some_name(self):
        """The whitespace-merge method exists under one of the known names."""
        assert any(hasattr(AnonymizerEngine, name) for name in presidio_patches._MERGE_METHOD_NAMES)
        # The resolver must return that name without raising.
        resolved = presidio_patches._resolve_merge_method_name()
        assert resolved in presidio_patches._MERGE_METHOD_NAMES
        assert hasattr(AnonymizerEngine, resolved)

    def test_remove_duplicates_exists(self):
        """EntityRecognizer.remove_duplicates still exists (patch 2 target)."""
        assert hasattr(EntityRecognizer, "remove_duplicates")

    def test_conflict_resolution_method_exists(self):
        """The conflict-resolution method still exists (patch 3 target)."""
        assert hasattr(AnonymizerEngine, "_remove_conflicts_and_get_text_manipulation_data")

    def test_public_merge_parameter_exists(self):
        """anonymize() accepts the public merge_entities_with_spaces parameter."""
        params = inspect.signature(AnonymizerEngine.anonymize).parameters
        assert "merge_entities_with_spaces" in params


class TestImportSafety:
    """The patch module must import cleanly and never touch attrs at import time."""

    def test_no_original_captured_at_import(self):
        """Whitespace-merge original is resolved lazily, not at import."""
        # Before any disable call, nothing is captured.
        if presidio_patches.is_whitespace_merging_disabled():
            presidio_patches.enable_whitespace_merging()
        assert presidio_patches._original_merge_method is None
        assert presidio_patches._resolved_merge_method_name is None


class TestWhitespaceMergingRoundTrip:
    """disable/enable whitespace merging restores the original method."""

    def test_disable_then_enable_restores_original(self):
        resolved = presidio_patches._resolve_merge_method_name()
        original = getattr(AnonymizerEngine, resolved)

        presidio_patches.disable_whitespace_merging()
        assert presidio_patches.is_whitespace_merging_disabled()
        assert getattr(AnonymizerEngine, resolved) is presidio_patches._no_merge

        presidio_patches.enable_whitespace_merging()
        assert not presidio_patches.is_whitespace_merging_disabled()
        assert getattr(AnonymizerEngine, resolved) is original

    def test_context_manager_restores_state(self):
        resolved = presidio_patches._resolve_merge_method_name()
        original = getattr(AnonymizerEngine, resolved)
        with presidio_patches.no_whitespace_merging():
            assert presidio_patches.is_whitespace_merging_disabled()
        assert not presidio_patches.is_whitespace_merging_disabled()
        assert getattr(AnonymizerEngine, resolved) is original

    def test_double_disable_is_idempotent(self):
        presidio_patches.disable_whitespace_merging()
        presidio_patches.disable_whitespace_merging()
        assert presidio_patches.is_whitespace_merging_disabled()
        presidio_patches.enable_whitespace_merging()
        assert not presidio_patches.is_whitespace_merging_disabled()


class TestRemoveDuplicatesRoundTrip:
    """patch/unpatch remove_duplicates restores the original method."""

    def test_patch_then_unpatch_restores_original(self):
        original = EntityRecognizer.remove_duplicates
        presidio_patches.patch_remove_duplicates()
        assert presidio_patches.is_remove_duplicates_patched()
        # Patched version is a passthrough no-op.
        sample = [
            AnalyzerRecognizerResult(entity_type="PERSON", start=0, end=4, score=0.99),
            AnalyzerRecognizerResult(entity_type="PERSON", start=5, end=9, score=0.99),
        ]
        assert EntityRecognizer.remove_duplicates(sample) is sample

        presidio_patches.unpatch_remove_duplicates()
        assert not presidio_patches.is_remove_duplicates_patched()
        assert EntityRecognizer.remove_duplicates == original


class TestConflictResolutionRoundTrip:
    """patch/unpatch conflict resolution restores the original method."""

    def test_patch_then_unpatch_restores_original(self):
        original = AnonymizerEngine._remove_conflicts_and_get_text_manipulation_data
        presidio_patches.patch_conflict_resolution()
        assert presidio_patches.is_conflict_resolution_patched()

        presidio_patches.unpatch_conflict_resolution()
        assert not presidio_patches.is_conflict_resolution_patched()
        assert AnonymizerEngine._remove_conflicts_and_get_text_manipulation_data == original


def _fingerprint(result) -> tuple:
    """Normalized (text, sorted item spans/types) fingerprint of an anonymize result."""
    items = sorted((it.start, it.end, it.entity_type) for it in result.items)
    return (result.text, tuple(items))


class TestBehavioralEquivalence:
    """
    Two adjacent same-type entities separated by one space:
    - default merge  -> a single merged replacement,
    - no-merge (patch or public parameter) -> two separate replacements.

    Both the monkeypatch and the public parameter must produce the same
    (identical) no-merge fingerprint, and it must differ from the default.
    """

    # "AAAA BBBB went home" — two adjacent PERSON entities separated by one space.
    TEXT = "AAAA BBBB went home"
    OPERATORS: ClassVar[dict] = {"PERSON": OperatorConfig("replace", {"new_value": "<X>"})}

    def _anonymize(self, engine, **kwargs):
        return engine.anonymize(
            text=self.TEXT,
            analyzer_results=[
                RecognizerResult(entity_type="PERSON", start=0, end=4, score=0.99),
                RecognizerResult(entity_type="PERSON", start=5, end=9, score=0.99),
            ],
            operators=self.OPERATORS,
            **kwargs,
        )

    def test_default_merges_adjacent_entities(self):
        """Sanity: without any no-merge, the two entities merge into one span."""
        engine = AnonymizerEngine()
        default_fp = _fingerprint(self._anonymize(engine))
        # Merged -> single replacement over the combined span.
        assert default_fp[0] == "<X> went home"

    def test_public_parameter_disables_merge(self):
        engine = AnonymizerEngine()
        no_merge_fp = _fingerprint(self._anonymize(engine, merge_entities_with_spaces=False))
        assert no_merge_fp[0] == "<X> <X> went home"

    def test_monkeypatch_disables_merge(self):
        engine = AnonymizerEngine()
        with presidio_patches.no_whitespace_merging():
            patched_fp = _fingerprint(self._anonymize(engine))
        assert patched_fp[0] == "<X> <X> went home"

    def test_monkeypatch_and_parameter_are_equivalent(self):
        """The monkeypatch path and the public-parameter path are byte-identical."""
        engine = AnonymizerEngine()
        param_fp = _fingerprint(self._anonymize(engine, merge_entities_with_spaces=False))
        with presidio_patches.no_whitespace_merging():
            patch_fp = _fingerprint(self._anonymize(engine))
        assert param_fp == patch_fp

    def test_no_merge_differs_from_default(self):
        """The knob is real: no-merge output differs from default-merge output."""
        engine = AnonymizerEngine()
        default_fp = _fingerprint(self._anonymize(engine))
        no_merge_fp = _fingerprint(self._anonymize(engine, merge_entities_with_spaces=False))
        assert default_fp != no_merge_fp
