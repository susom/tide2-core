"""
Monkey patches for Presidio AnalyzerEngine and AnonymizerEngine.

This module provides patches to modify Presidio's default behavior:

1. Whitespace merging (AnonymizerEngine):
    Presidio merges adjacent entities of the same type separated by whitespace.
    This causes inconsistent encryption for identical values in different formats.
    Patch: disable _merge_entities_with_whitespace_between.

2. Remove duplicates (AnalyzerEngine):
    Presidio's EntityRecognizer.remove_duplicates() uses an O(n²) algorithm that
    becomes catastrophic with thousands of entities (e.g., 7,800 entities → 32M
    comparisons → 28s per note). Deduplication is handled downstream on the
    anonymizer side, so we skip it entirely on the recognizer side.
    Patch: replace remove_duplicates with a no-op passthrough.

Usage:
    from tide2.anonymizers import presidio_patches
    presidio_patches.disable_whitespace_merging()
    presidio_patches.patch_remove_duplicates()
"""

from contextlib import contextmanager

from presidio_analyzer import EntityRecognizer
from presidio_anonymizer import AnonymizerEngine

# Store original method for restoration
_original_merge_entities_with_whitespace_between = AnonymizerEngine._merge_entities_with_whitespace_between

# Flag to track if patch is applied
_patch_applied = False


def _no_merge(self, text: str, analyzer_results: list) -> list:
    """
    Replacement method that performs no merging.

    Simply returns the input results unchanged, preventing Presidio from
    merging adjacent entities of the same type.

    Args:
        text: The original text (unused)
        analyzer_results: List of RecognizerResult objects

    Returns:
        The same list of RecognizerResult objects, unmodified
    """
    return analyzer_results


def disable_whitespace_merging() -> None:
    """
    Disable Presidio's whitespace-based entity merging globally.

    After calling this function, adjacent entities of the same type will
    NOT be merged, even if separated only by whitespace. This ensures
    each entity is processed individually by anonymizers.

    This is a global patch that affects all AnonymizerEngine instances.

    Example:
        >>> from tide2.anonymizers import presidio_patches
        >>> presidio_patches.disable_whitespace_merging()
        >>> # Now all anonymizations will process entities individually
    """
    global _patch_applied
    if not _patch_applied:
        AnonymizerEngine._merge_entities_with_whitespace_between = _no_merge
        _patch_applied = True


def enable_whitespace_merging() -> None:
    """
    Re-enable Presidio's default whitespace-based entity merging.

    Restores the original `_merge_entities_with_whitespace_between` method.

    Example:
        >>> from tide2.anonymizers import presidio_patches
        >>> presidio_patches.disable_whitespace_merging()
        >>> # ... do work without merging ...
        >>> presidio_patches.enable_whitespace_merging()
        >>> # Back to default Presidio behavior
    """
    global _patch_applied
    if _patch_applied:
        AnonymizerEngine._merge_entities_with_whitespace_between = _original_merge_entities_with_whitespace_between
        _patch_applied = False


def is_whitespace_merging_disabled() -> bool:
    """
    Check if whitespace merging is currently disabled.

    Returns:
        True if the patch is applied (merging disabled), False otherwise.
    """
    return _patch_applied


@contextmanager
def no_whitespace_merging():
    """
    Context manager to temporarily disable whitespace merging.

    Disables merging for the duration of the context, then restores
    the previous state (whether merging was enabled or disabled).

    Example:
        >>> from tide2.anonymizers import presidio_patches
        >>> with presidio_patches.no_whitespace_merging():
        ...     result = anonymizer.anonymize(text, results, operators)
        >>> # Merging state is restored after the context
    """
    was_disabled = is_whitespace_merging_disabled()
    try:
        disable_whitespace_merging()
        yield
    finally:
        if not was_disabled:
            enable_whitespace_merging()


# ============================================================================
# Patch 2: Disable O(n²) remove_duplicates in AnalyzerEngine
# ============================================================================

# Store original method for restoration
_original_remove_duplicates = EntityRecognizer.remove_duplicates

# Flag to track if patch is applied
_remove_duplicates_patched = False


def patch_remove_duplicates() -> None:
    """
    Replace Presidio's O(n²) remove_duplicates with a no-op passthrough.

    Presidio's EntityRecognizer.remove_duplicates() compares every result against
    every other result to find duplicates and contained spans. With thousands of
    entities this becomes catastrophic (e.g., 7,800 entities → 32M comparisons).

    Deduplication is handled on the anonymizer side, so we skip it here.

    This is a global patch that affects all AnalyzerEngine instances.
    """
    global _remove_duplicates_patched
    if not _remove_duplicates_patched:
        EntityRecognizer.remove_duplicates = staticmethod(lambda results: results)
        _remove_duplicates_patched = True


def unpatch_remove_duplicates() -> None:
    """Restore Presidio's original remove_duplicates method."""
    global _remove_duplicates_patched
    if _remove_duplicates_patched:
        EntityRecognizer.remove_duplicates = _original_remove_duplicates
        _remove_duplicates_patched = False


def is_remove_duplicates_patched() -> bool:
    """Check if remove_duplicates is currently patched."""
    return _remove_duplicates_patched


# ============================================================================
# Patch 3: Disable O(n²) conflict resolution in AnonymizerEngine
# ============================================================================

# Store original method for restoration
_original_remove_conflicts = AnonymizerEngine._remove_conflicts_and_get_text_manipulation_data

# Flag to track if patch is applied
_conflict_resolution_patched = False


def patch_conflict_resolution() -> None:
    """
    Replace Presidio's O(n²) _remove_conflicts_and_get_text_manipulation_data
    with a no-op passthrough.

    Presidio's conflict resolution iterates every entity against every other
    entity twice (merge same-type + check conflicts). With 44K entities this
    takes hours. Conflict resolution is already handled upstream via
    resolve_recognizer_results() in the anonymizer actor, so we skip it here.

    This is a global patch that affects all AnonymizerEngine instances.
    """
    global _conflict_resolution_patched
    if not _conflict_resolution_patched:
        AnonymizerEngine._remove_conflicts_and_get_text_manipulation_data = lambda self, results, conflict_resolution: (
            results
        )
        _conflict_resolution_patched = True


def unpatch_conflict_resolution() -> None:
    """Restore Presidio's original conflict resolution method."""
    global _conflict_resolution_patched
    if _conflict_resolution_patched:
        AnonymizerEngine._remove_conflicts_and_get_text_manipulation_data = _original_remove_conflicts
        _conflict_resolution_patched = False


def is_conflict_resolution_patched() -> bool:
    """Check if conflict resolution is currently patched."""
    return _conflict_resolution_patched
