"""
Monkey patches for Presidio AnalyzerEngine and AnonymizerEngine.

This module provides patches to modify Presidio's default behavior:

1. Whitespace merging (AnonymizerEngine):
    Presidio merges adjacent entities of the same type separated by whitespace.
    This causes inconsistent encryption for identical values in different formats.
    Patch: disable the private whitespace-merge method.

    Presidio renamed this private method between releases:
    ``_merge_entities_with_whitespace_between`` (<= 2.2.362) became
    ``_merge_entities_with_spaces_between`` (>= 2.2.363). The patch resolves the
    method by either name, lazily, at call time -- never at import -- so a future
    rename cannot crash the import of this module (pdoc imports every module for
    the docs build, so import-time safety is a hard constraint).

    The presidio-blessed, rename-proof way to disable merging is the public
    parameter ``AnonymizerEngine.anonymize(..., merge_entities_with_spaces=False)``
    (added in 2.2.363); call sites use it directly. This monkeypatch remains as
    back-compat defense-in-depth for any code path that constructs an
    AnonymizerEngine without going through those call sites.

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

# Presidio renamed this private method in 2.2.363. Try the new name first, then
# the old one. Resolution is deferred to call time (see _resolve_merge_method);
# we do NOT touch these attributes at import, since a missing attribute would
# crash the import of this module (and thus the whole docs build via pdoc).
_MERGE_METHOD_NAMES = (
    "_merge_entities_with_spaces_between",  # presidio >= 2.2.363
    "_merge_entities_with_whitespace_between",  # presidio <= 2.2.362
)

# Original method + the name it was resolved under, captured lazily on first
# disable_whitespace_merging() call so enable_whitespace_merging() can restore it.
_original_merge_method = None
_resolved_merge_method_name: str | None = None

# Flag to track if patch is applied
_patch_applied = False


def _resolve_merge_method_name() -> str:
    """
    Resolve the name of Presidio's private whitespace-merge method, lazily.

    Presidio renamed this method between releases (see _MERGE_METHOD_NAMES). This
    resolver is only ever called at patch-apply time, never at import, so a rename
    that removes both known names fails with an actionable error at call time
    instead of crashing the import of this module.

    Returns:
        The attribute name present on AnonymizerEngine.

    Raises:
        RuntimeError: If none of the known method names exist on the installed
            presidio version (names it in the message).
    """
    for name in _MERGE_METHOD_NAMES:
        if hasattr(AnonymizerEngine, name):
            return name

    try:
        import importlib.metadata as _md

        version = _md.version("presidio-anonymizer")
    except Exception:  # pragma: no cover - version lookup is best-effort
        version = "unknown"
    raise RuntimeError(
        f"Cannot disable whitespace merging: none of {_MERGE_METHOD_NAMES} exist on "
        f"AnonymizerEngine (installed presidio-anonymizer=={version}). Presidio may have "
        "renamed the private merge method again; update _MERGE_METHOD_NAMES in "
        "presidio_patches.py. Note that call sites should prefer the public "
        "anonymize(merge_entities_with_spaces=False) parameter."
    )


def _no_merge(self, text: str, analyzer_results: list) -> list:  # noqa: ARG001  # signature must match presidio's method it replaces
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

    The method name is resolved lazily here (never at import), tolerating the
    presidio 2.2.363 rename. Prefer the public
    ``anonymize(merge_entities_with_spaces=False)`` parameter at call sites; this
    patch is back-compat defense-in-depth.

    Raises:
        RuntimeError: If no known whitespace-merge method exists on the installed
            presidio version (see _resolve_merge_method_name).

    Example:
        >>> from tide2.anonymizers import presidio_patches
        >>> presidio_patches.disable_whitespace_merging()
        >>> # Now all anonymizations will process entities individually
    """
    # Module-level globals (not object state): these patches must apply on every
    # Ray worker process, where re-imported module globals are the reliable shared
    # state; instance-level patching would not propagate across the actor pool.
    global _patch_applied, _original_merge_method, _resolved_merge_method_name  # noqa: PLW0603
    if not _patch_applied:
        _resolved_merge_method_name = _resolve_merge_method_name()
        _original_merge_method = getattr(AnonymizerEngine, _resolved_merge_method_name)
        setattr(AnonymizerEngine, _resolved_merge_method_name, _no_merge)
        _patch_applied = True


def enable_whitespace_merging() -> None:
    """
    Re-enable Presidio's default whitespace-based entity merging.

    Restores the original whitespace-merge method against whichever name it was
    resolved under when the patch was applied.

    Example:
        >>> from tide2.anonymizers import presidio_patches
        >>> presidio_patches.disable_whitespace_merging()
        >>> # ... do work without merging ...
        >>> presidio_patches.enable_whitespace_merging()
        >>> # Back to default Presidio behavior
    """
    # Module-level globals (not object state): these patches must apply on every
    # Ray worker process, where re-imported module globals are the reliable shared
    # state; instance-level patching would not propagate across the actor pool.
    global _patch_applied, _original_merge_method, _resolved_merge_method_name  # noqa: PLW0603
    if _patch_applied:
        setattr(AnonymizerEngine, _resolved_merge_method_name, _original_merge_method)
        _original_merge_method = None
        _resolved_merge_method_name = None
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
    global _remove_duplicates_patched  # noqa: PLW0603  # shared across Ray workers via module globals
    if not _remove_duplicates_patched:
        EntityRecognizer.remove_duplicates = staticmethod(lambda results: results)
        _remove_duplicates_patched = True


def unpatch_remove_duplicates() -> None:
    """Restore Presidio's original remove_duplicates method."""
    global _remove_duplicates_patched  # noqa: PLW0603  # shared across Ray workers via module globals
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
    global _conflict_resolution_patched  # noqa: PLW0603  # shared across Ray workers via module globals
    if not _conflict_resolution_patched:
        AnonymizerEngine._remove_conflicts_and_get_text_manipulation_data = lambda self, results, conflict_resolution: (  # noqa: ARG005  # lambda must match presidio's method signature positionally
            results
        )
        _conflict_resolution_patched = True


def unpatch_conflict_resolution() -> None:
    """Restore Presidio's original conflict resolution method."""
    global _conflict_resolution_patched  # noqa: PLW0603  # shared across Ray workers via module globals
    if _conflict_resolution_patched:
        AnonymizerEngine._remove_conflicts_and_get_text_manipulation_data = _original_remove_conflicts
        _conflict_resolution_patched = False


def is_conflict_resolution_patched() -> bool:
    """Check if conflict resolution is currently patched."""
    return _conflict_resolution_patched
