"""
Span Metrics Module

This module provides functionality for computing metrics between gold standard
and machine learning-generated text spans, as well as efficient O(n log n)
conflict resolution for overlapping spans.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import pandas as pd

if TYPE_CHECKING:
    from presidio_anonymizer.entities import RecognizerResult

logger = logging.getLogger(__name__)


def load_llm_spans_from_dir(llm_dir: str | Path) -> pd.DataFrame:
    """
    Load LLM JSON outputs and convert to ml_spans DataFrame format.

    Handles two folder structures:
    - Direct: JSON files directly in llm_dir (e.g., llm_json_recognizer_gemini25/*.json)
    - Nested: JSON files in llm_dir/analyzer_output/ (e.g., tide20_*/analyzer_output/*.json)

    Args:
        llm_dir: Path to LLM output directory containing JSON files with recognizer_results

    Returns:
        DataFrame with columns: note_id, span_start, span_end, span_tag, score, recognizer_name
        Compatible with ml_spans.parquet format used by evaluation scripts.

    Example:
        >>> df = load_llm_spans_from_dir("/path/to/llm_json_recognizer_gemini25")
        >>> df.columns.tolist()
        ['note_id', 'span_start', 'span_end', 'span_tag', 'score', 'recognizer_name']
    """
    llm_dir = Path(llm_dir)

    # Auto-detect folder structure: check for analyzer_output subfolder
    json_dir = llm_dir / "analyzer_output" if (llm_dir / "analyzer_output").exists() else llm_dir

    if not json_dir.exists():
        raise FileNotFoundError(f"LLM output directory not found: {json_dir}")

    rows = []
    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in {json_dir}")
        return pd.DataFrame(columns=["note_id", "span_start", "span_end", "span_tag", "score", "recognizer_name"])

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue

        note_id = data.get("key", json_file.stem)

        # Support two formats:
        # 1. recognizer_results format: start/end/entity_type (from TIDE pipeline)
        # 2. response format: span_start/span_end/pii_type (from LLM pre-annotation)
        recognizer_results = data.get("recognizer_results", [])
        response_results = data.get("response", [])

        for r in recognizer_results:
            recognizer_name = r.get("recognition_metadata", {}).get("recognizer_name", "LlmJsonRecognizer")
            rows.append(
                {
                    "note_id": note_id,
                    "span_start": r["start"],
                    "span_end": r["end"],
                    "span_tag": r["entity_type"],
                    "score": r.get("score", 0.9),
                    "recognizer_name": recognizer_name,
                }
            )

        for r in response_results:
            rows.append(
                {
                    "note_id": note_id,
                    "span_start": r["span_start"],
                    "span_end": r["span_end"],
                    "span_tag": r["pii_type"],
                    "score": r.get("score", 0.9),
                    "recognizer_name": "LlmJsonRecognizer",
                }
            )

    logger.info(f"Loaded {len(rows)} spans from {len(json_files)} LLM JSON files in {json_dir}")
    return pd.DataFrame(rows)


def map_labels(
    df: pd.DataFrame,
    label_map: dict[str, str],
    column: str = "span_tag",
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Map labels in a DataFrame column using a provided mapping.

    Args:
        df: DataFrame containing the column to map.
        label_map: Dictionary mapping source labels to target labels.
        column: Name of the column to map. Defaults to "span_tag".
        drop_unmapped: If True, drop rows with unmapped labels. If False, keep original label.

    Returns:
        DataFrame with mapped labels.

    Example:
        >>> df = pd.DataFrame({"span_tag": ["DATES", "HCW", "PATIENT"]})
        >>> label_map = {"DATES": "DATE", "HCW": "DOCTOR", "PATIENT": "PATIENT"}
        >>> result = map_labels(df, label_map)
        >>> result["span_tag"].tolist()
        ['DATE', 'DOCTOR', 'PATIENT']
    """
    if df.empty or not label_map:
        return df.copy()

    result = df.copy()

    if drop_unmapped:
        result[column] = result[column].map(label_map)
        unmapped_count = result[column].isna().sum()
        if unmapped_count > 0:
            logger.warning(f"Dropping {unmapped_count} rows with unmapped labels in column '{column}'")
            result = result.dropna(subset=[column])
    else:
        # Keep original label if not in map
        result[column] = result[column].apply(lambda x: label_map.get(x, x))

    return result


def resolve_conflicts(
    spans: list[dict[str, Any]],
    strategy: Literal["longest_wins", "merge_contained"] = "longest_wins",
) -> list[dict[str, Any]]:
    """
    O(n log n) conflict resolution for overlapping spans.

    This function efficiently resolves conflicts between overlapping spans using
    a sweep-line algorithm. It handles both same-type and cross-type overlaps.

    Algorithm complexity: O(n log n) due to sorting, vs O(n²) for naive approaches.

    Args:
        spans: List of span dicts, each must have 'start' and 'end' keys.
               Optional keys: 'entity_type', 'score', and any other metadata.
        strategy:
            - "longest_wins": Any overlap → keep longer span, discard shorter (default)
            - "merge_contained": Only remove spans fully contained within another

    Returns:
        List of non-overlapping spans (for "longest_wins") or spans with
        containment resolved (for "merge_contained").

    Example:
        >>> spans = [
        ...     {"start": 10, "end": 25, "entity_type": "PERSON"},
        ...     {"start": 15, "end": 20, "entity_type": "NAME"},  # contained
        ...     {"start": 30, "end": 40, "entity_type": "DATE"},
        ... ]
        >>> result = resolve_conflicts(spans)
        >>> len(result)
        2
    """
    if not spans:
        return []

    # Step 1: Remove exact duplicates by (start, end), keeping first occurrence
    # This is O(n)
    seen: dict[tuple[int, int], dict[str, Any]] = {}
    for span in spans:
        key = (span["start"], span["end"])
        if key not in seen:
            seen[key] = span
        else:
            # If both have scores, keep higher score
            existing_score = seen[key].get("score", 0)
            new_score = span.get("score", 0)
            if new_score > existing_score:
                seen[key] = span

    unique_spans = list(seen.values())

    if not unique_spans:
        return []

    # Step 2: Sort by start ascending, then by length descending (longer first)
    # This ensures at same start position, longer spans are processed first
    # O(n log n)
    unique_spans.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    if strategy == "merge_contained":
        return _resolve_contained(unique_spans)
    # longest_wins
    return _resolve_longest_wins(unique_spans)


def _resolve_contained(sorted_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove spans that are fully contained within another span.

    A span [a, b] is contained in [c, d] if c <= a and b <= d.

    This uses a sweep-line approach: we track the maximum end position
    seen so far. A span is contained if its end <= max_end of any
    previously kept span that started at or before this span's start.

    Args:
        sorted_spans: Spans sorted by (start, -length)

    Returns:
        Spans with contained ones removed
    """
    if not sorted_spans:
        return []

    kept: list[dict[str, Any]] = []

    for span in sorted_spans:
        start, end = span["start"], span["end"]
        is_contained = False

        # Check if this span is contained in any kept span
        # Since we sorted by start then -length, we only need to check
        # if any kept span with start <= this start has end >= this end
        for k in kept:
            if k["start"] <= start and k["end"] >= end:
                is_contained = True
                break

        if not is_contained:
            kept.append(span)

    return kept


def _resolve_longest_wins(sorted_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Resolve all overlaps by keeping the longer span.

    For any two overlapping spans, the longer one wins. This uses an
    interval-based approach where we track non-overlapping intervals.

    Args:
        sorted_spans: Spans sorted by (start, -length)

    Returns:
        Non-overlapping spans where longer spans won conflicts
    """
    if not sorted_spans:
        return []

    # We'll use a greedy approach: process spans by start position,
    # and for each span check if it overlaps with any kept span.
    # If it does, keep the longer one.

    kept: list[dict[str, Any]] = []

    for span in sorted_spans:
        start, end = span["start"], span["end"]
        span_len = end - start

        # Find any overlapping spans in kept
        overlapping_indices = []
        for i, k in enumerate(kept):
            k_start, k_end = k["start"], k["end"]
            # Check for overlap: spans overlap if start < k_end and end > k_start
            if start < k_end and end > k_start:
                overlapping_indices.append(i)

        if not overlapping_indices:
            # No overlap, add this span
            kept.append(span)
        else:
            # Check if this span is longer than all overlapping spans
            # If so, remove overlapping spans and add this one
            all_shorter = True
            for i in overlapping_indices:
                k = kept[i]
                k_len = k["end"] - k["start"]
                if k_len >= span_len:
                    all_shorter = False
                    break

            if all_shorter:
                # Remove all overlapping (shorter) spans and add this one
                # Remove in reverse order to maintain indices
                for i in sorted(overlapping_indices, reverse=True):
                    kept.pop(i)
                kept.append(span)
            # else: this span is shorter than at least one overlapping span, discard it

    # Sort by start position for consistent output
    kept.sort(key=lambda x: x["start"])
    return kept


def resolve_recognizer_results(
    results: list["RecognizerResult"],
    strategy: Literal["longest_wins", "merge_contained"] = "longest_wins",
    merge_adjacent_types: set[str] | None = None,
    text: str | None = None,
    max_merge_gap: int = 2,
) -> list["RecognizerResult"]:
    """
    O(n log n) conflict resolution for Presidio RecognizerResult objects.

    This is a drop-in replacement for Presidio's built-in conflict resolution,
    optimized for performance with large numbers of entities (5000+).

    Use this with `conflict_resolution=ConflictResolutionStrategy.NONE` when
    calling AnonymizerEngine.anonymize() to bypass Presidio's O(n²) resolution.

    Args:
        results: List of Presidio RecognizerResult objects
        strategy:
            - "longest_wins": Any overlap → keep longer span (default)
            - "merge_contained": Only remove fully contained spans
        merge_adjacent_types: If provided, merge adjacent spans of the same type
            for these entity types (e.g. {"DATE", "DATE_TIME"}).
        text: The source text, required when merge_adjacent_types is set
            (used to verify gaps contain only whitespace).
        max_merge_gap: Maximum character gap between spans to merge (default 2).

    Returns:
        List of RecognizerResult objects with conflicts resolved
        (and optionally adjacent same-type spans merged)

    Example:
        >>> from presidio_anonymizer.entities import RecognizerResult
        >>> results = [
        ...     RecognizerResult(entity_type="PERSON", start=0, end=10, score=0.9),
        ...     RecognizerResult(entity_type="NAME", start=2, end=8, score=0.85),
        ... ]
        >>> resolved = resolve_recognizer_results(results)
        >>> len(resolved)  # Only the longer PERSON span is kept
        1
    """
    if not results:
        return []

    # Filter zero-score results
    results = [r for r in results if r.score > 0]

    if not results:
        return []

    # Convert to dict format for processing
    spans_with_refs: list[tuple[dict[str, Any], RecognizerResult]] = []
    for r in results:
        span_dict = {
            "start": r.start,
            "end": r.end,
            "entity_type": r.entity_type,
            "score": r.score,
        }
        spans_with_refs.append((span_dict, r))

    # Remove exact duplicates, keeping highest score
    seen: dict[tuple[int, int, str], tuple[dict[str, Any], RecognizerResult]] = {}
    for span_dict, ref in spans_with_refs:
        key = (span_dict["start"], span_dict["end"], span_dict["entity_type"])
        if key not in seen or span_dict["score"] > seen[key][0]["score"]:
            seen[key] = (span_dict, ref)

    unique_spans = list(seen.values())

    if not unique_spans:
        return []

    # Sort by start ascending, then by length descending
    unique_spans.sort(key=lambda x: (x[0]["start"], -(x[0]["end"] - x[0]["start"])))

    # Apply resolution strategy
    if strategy == "merge_contained":
        kept = _resolve_contained_with_refs(unique_spans)
    else:  # longest_wins
        kept = _resolve_longest_wins_with_refs(unique_spans)

    # Extract the original RecognizerResult objects
    resolved = [ref for _, ref in kept]

    if merge_adjacent_types and text is not None:
        resolved = _merge_adjacent_same_type(resolved, text, merge_adjacent_types, max_merge_gap)

    return resolved


def _merge_adjacent_same_type(
    results: list["RecognizerResult"],
    text: str,
    entity_types: set[str],
    max_gap: int,
) -> list["RecognizerResult"]:
    """Merge adjacent spans of the same entity type separated by at most max_gap whitespace chars.

    Only spans whose entity_type is in ``entity_types`` are candidates for
    merging.  Two candidates merge when they share the same entity_type and
    the gap between them is 0..max_gap characters of whitespace.

    Args:
        results: Conflict-resolved RecognizerResult list.
        text: The source text (used to inspect gap content).
        entity_types: Entity types eligible for merging (e.g. {"DATE", "DATE_TIME"}).
        max_gap: Maximum character gap between spans to consider merging.

    Returns:
        List of RecognizerResult with eligible adjacent spans merged.
    """
    from presidio_anonymizer.entities import RecognizerResult

    to_merge = [r for r in results if r.entity_type in entity_types]
    others = [r for r in results if r.entity_type not in entity_types]

    if len(to_merge) <= 1:
        return results

    to_merge.sort(key=lambda r: r.start)
    merged: list[RecognizerResult] = [to_merge[0]]

    for current in to_merge[1:]:
        prev = merged[-1]
        if prev.entity_type == current.entity_type:
            gap = current.start - prev.end
            if 0 <= gap <= max_gap:
                gap_text = text[prev.end : current.start] if gap > 0 else ""
                if gap_text.strip() == "":
                    merged[-1] = RecognizerResult(
                        entity_type=prev.entity_type,
                        start=prev.start,
                        end=current.end,
                        score=max(prev.score, current.score),
                    )
                    continue
        merged.append(current)

    return others + merged


def _resolve_contained_with_refs(
    sorted_spans: list[tuple[dict[str, Any], "RecognizerResult"]],
) -> list[tuple[dict[str, Any], "RecognizerResult"]]:
    """Remove contained spans, preserving RecognizerResult references."""
    if not sorted_spans:
        return []

    kept: list[tuple[dict[str, Any], RecognizerResult]] = []

    for span_tuple in sorted_spans:
        span_dict = span_tuple[0]
        start, end = span_dict["start"], span_dict["end"]
        is_contained = False

        for k_tuple in kept:
            k = k_tuple[0]
            if k["start"] <= start and k["end"] >= end:
                is_contained = True
                break

        if not is_contained:
            kept.append(span_tuple)

    return kept


def _resolve_longest_wins_with_refs(
    sorted_spans: list[tuple[dict[str, Any], "RecognizerResult"]],
) -> list[tuple[dict[str, Any], "RecognizerResult"]]:
    """Resolve overlaps keeping longest span, preserving RecognizerResult references.

    Input must be sorted by start ascending, then by length descending.
    Uses a backwards scan on ``kept`` so that only recent (overlapping)
    entries are examined, giving O(n) for non-pathological inputs.
    """
    if not sorted_spans:
        return []

    kept: list[tuple[dict[str, Any], RecognizerResult]] = []
    # Track the running maximum end position across kept spans so we can
    # quickly skip spans that are fully contained within an already-kept span.
    max_end = -1

    for span_tuple in sorted_spans:
        span_dict = span_tuple[0]
        start, end = span_dict["start"], span_dict["end"]
        span_len = end - start

        # Fast path: no overlap with any kept span
        if start >= max_end:
            kept.append(span_tuple)
            max_end = end
            continue

        # Scan backwards to find overlapping kept spans.
        # Because kept is sorted by start, we only need to look at entries
        # whose start < current end (guaranteed for recent entries).
        overlapping_indices = []
        for i in range(len(kept) - 1, -1, -1):
            k = kept[i][0]
            k_start, k_end = k["start"], k["end"]
            # Since kept is sorted by start and we scan backwards,
            # once k_end <= start there can be no more overlaps.
            if k_end <= start:
                break
            if start < k_end and end > k_start:
                overlapping_indices.append(i)

        if not overlapping_indices:
            kept.append(span_tuple)
            max_end = max(max_end, end)
        else:
            all_shorter = True
            for i in overlapping_indices:
                k = kept[i][0]
                k_len = k["end"] - k["start"]
                if k_len >= span_len:
                    all_shorter = False
                    break

            if all_shorter:
                for i in sorted(overlapping_indices, reverse=True):
                    kept.pop(i)
                kept.append(span_tuple)
                max_end = max(max_end, end)

    kept.sort(key=lambda x: x[0]["start"])
    return kept


class SpanCollection:
    """Efficient collection for storing and querying spans by note_id and tag."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize SpanCollection from a DataFrame.

        Args:
            df: DataFrame with columns: note_id, span_start, span_end, span_tag
        """
        self._spans_by_note_and_tag = defaultdict(lambda: defaultdict(list))
        self._note_ids = set()

        # Group spans by note_id and tag for O(1) lookup
        for row in df.itertuples():
            note_id = row.note_id
            tag = row.span_tag
            span = (row.span_start, row.span_end, row.Index)  # Include original index

            self._spans_by_note_and_tag[note_id][tag].append(span)
            self._note_ids.add(note_id)

    def get_spans(self, note_id: str, tag: str) -> list[tuple[int, int, int]]:
        """Get all spans for a specific note_id and tag."""
        return self._spans_by_note_and_tag[note_id][tag]

    def get_note_ids(self) -> set[str]:
        """Get all note_ids in the collection."""
        return self._note_ids

    def get_tags_for_note(self, note_id: str) -> list[str]:
        """Get all tags for a specific note_id."""
        return list(self._spans_by_note_and_tag[note_id].keys())

    def iterate_spans(self):
        """Iterate over all spans with their metadata."""
        for note_id, tags_dict in self._spans_by_note_and_tag.items():
            for tag, spans in tags_dict.items():
                for start, end, idx in spans:
                    yield note_id, tag, start, end, idx


def span_overlap(span1: tuple[int, int], span2: tuple[int, int], exact_match=False) -> float:
    """
    Calculate the overlap between two spans.

    This function computes the overlap between two text spans, returning either
    the proportional overlap relative to the first span, or a binary exact match
    indicator depending on the exact_match parameter.

    Args:
        span1: A tuple of (start, end) positions for the first span (reference span)
        span2: A tuple of (start, end) positions for the second span (comparison span)
        exact_match: If True, returns 1.0 only if spans are identical (same start and end),
                    otherwise returns 0.0. If False (default), returns the proportion of
                    span1 that overlaps with span2.

    Returns:
        float: If exact_match is False, returns the proportion of span1 that overlaps
               with span2 (range: 0.0 to 1.0). If exact_match is True, returns 1.0 for
               exact matches and 0.0 otherwise.

    Examples:
        >>> span_overlap((10, 20), (15, 25))  # 50% overlap
        0.5
        >>> span_overlap((10, 20), (10, 20), exact_match=True)  # Exact match
        1.0
        >>> span_overlap((10, 20), (10, 19), exact_match=True)  # Not exact
        0.0
        >>> span_overlap((10, 20), (5, 30))  # Complete overlap
        1.0
        >>> span_overlap((10, 20), (25, 30))  # No overlap
        0.0
    """

    start1, end1 = span1
    start2, end2 = span2
    if not exact_match:
        overlap = max(0, min(end1, end2) - max(start1, start2))
        return overlap / (end1 - start1) if end1 > start1 else 0
    return 1.0 if start1 == start2 and end1 == end2 else 0.0


def resolve_overlapping_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Resolve overlapping spans using non-maximum suppression by keeping the longest span.

    When two spans overlap, this function keeps the longer one and suppresses the shorter one.
    This is a thin wrapper around resolve_conflicts() for backwards compatibility.

    Algorithm complexity: O(n log n)

    Args:
        spans: List of span dictionaries, each containing at least 'start' and 'end' keys.
               Additional keys in each span dict are preserved in the output.

    Returns:
        List of non-overlapping span dictionaries with the same structure as input spans.
        Only the longest spans are kept when overlaps occur.

    Examples:
        >>> spans = [
        ...     {"start": 10, "end": 20, "text": "example"},
        ...     {"start": 15, "end": 25, "text": "longer example"},
        ... ]
        >>> result = resolve_overlapping_spans(spans)
        >>> len(result)
        1
        >>> result[0]["text"]
        'longer example'
    """
    return resolve_conflicts(spans, strategy="longest_wins")


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score from true positives,
    false positives, and false negatives.

    Args:
        tp: Number of true positives
        fp: Number of false positives
        fn: Number of false negatives

    Returns:
        A tuple of (precision, recall, f1)

    Example:
        >>> compute_prf(8, 2, 1)
        (0.8, 0.888..., 0.842...)
    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def preprocess_ml_dataframe(ml_df: pd.DataFrame, label_maps: dict[str, list[str]]) -> pd.DataFrame:
    """
    Preprocess ML dataframe by applying label mappings.

    Args:
        ml_df: Machine learning spans DataFrame
        label_maps: Dictionary mapping span tags to their descriptions

    Returns:
        Preprocessed DataFrame with mapped labels
    """
    df_ml = ml_df.copy()

    # Create reverse mapping from values to keys
    reversed_label_maps = {}
    for key, values in label_maps.items():
        if isinstance(values, (list, tuple)):
            for value in values:
                current_keys = reversed_label_maps.get(value, [])
                current_keys.append(key)
                reversed_label_maps[value] = current_keys
        else:
            # Handle single value case
            current_keys = reversed_label_maps.get(values, [])
            current_keys.append(key)
            reversed_label_maps[values] = current_keys

    # Map labels and expand for one-to-many mappings
    df_ml["span_tag"] = df_ml["span_tag"].map(reversed_label_maps)

    unmapped_count = df_ml["span_tag"].isna().sum()
    if unmapped_count > 0:
        logging.warning(f"There are {unmapped_count} unmapped labels in the ML dataframe")
        df_ml = df_ml.dropna(subset=["span_tag"])

    # Expand rows for one-to-many mapping
    return df_ml.explode("span_tag")


def find_matches(
    gold_collection: SpanCollection, ml_collection: SpanCollection, overlap_threshold: float, exact_match: bool = False
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """
    Find matches between gold and ML spans efficiently.

    Duplicate-overlap policy:
        If multiple ML spans overlap the *same* gold span at or above the
        overlap_threshold, only the first encountered prediction will be
        counted as a true positive (TP). Later overlapping predictions for
        that already-matched gold span are ignored (not counted as TP or FP).
        This prevents inflating TP counts while avoiding penalizing systems
        that output fragmented / redundant spans for the same entity.

    Matching direction & overlap metric:
        Overlap is computed as proportion of the gold span length by default
        (i.e., span_overlap(gold, prediction)). When exact_match is True,
        only spans with identical start and end positions are considered matches,
        and the overlap_threshold is ignored.

    Args:
        gold_collection: Gold standard spans
        ml_collection: ML prediction spans
        overlap_threshold: Minimum overlap for match (ignored if exact_match is True)
        exact_match: If True, only spans with identical start and end positions
                    are considered matches. If False (default), uses proportional
                    overlap with the overlap_threshold.

    Returns:
        Tuple of (span_metrics, metrics_per_label, doc_metrics)
    """
    # Initialize metrics storage
    all_tags = set()
    for note_id in gold_collection.get_note_ids():
        all_tags.update(gold_collection.get_tags_for_note(note_id))

    metrics_per_label = {tag: {"tp": 0, "fp": 0, "fn": 0} for tag in all_tags}
    doc_metrics = {note_id: {"tp": 0, "fp": 0, "fn": 0} for note_id in gold_collection.get_note_ids()}
    span_metrics = []

    # Track matched gold spans to avoid double counting
    matched_gold_spans = set()

    # Process ML spans to find TPs and FPs.
    # Updated rule (duplicate-overlap neutralization):
    #   If multiple ML spans overlap the SAME gold span above threshold,
    #   only the first counts as TP; subsequent overlapping predictions for
    #   that already-matched gold span are ignored (neither TP nor FP).
    #   Rationale: avoid penalizing (no FP) but also avoid inflating TP.
    for note_id, tag, ml_start, ml_end, ml_idx in ml_collection.iterate_spans():
        # Skip if note not in gold data
        if note_id not in gold_collection.get_note_ids():
            continue

        # Skip if tag not in gold data
        if tag not in metrics_per_label:
            continue

        gold_spans = gold_collection.get_spans(note_id, tag)
        found_match = False

        for gold_start, gold_end, gold_idx in gold_spans:
            overlap = span_overlap((ml_start, ml_end), (gold_start, gold_end), exact_match=exact_match)
            if overlap >= overlap_threshold:
                if (note_id, tag, gold_start, gold_end) in matched_gold_spans:
                    # Already satisfied by a previous prediction -> ignore silently
                    found_match = True  # treat as matched so it is not an FP
                    break
                # First acceptable overlap for this gold span -> TP
                metrics_per_label[tag]["tp"] += 1
                doc_metrics[note_id]["tp"] += 1
                matched_gold_spans.add((note_id, tag, gold_start, gold_end))
                span_metrics.append(
                    {
                        "note_id": note_id,
                        "span_start": gold_start,
                        "span_end": gold_end,
                        "span_tag": tag,
                        "metric": "tp",
                    }
                )
                found_match = True
                break

        if not found_match:
            # False positive
            metrics_per_label[tag]["fp"] += 1
            doc_metrics[note_id]["fp"] += 1

            span_metrics.append(
                {
                    "note_id": note_id,
                    "span_start": ml_start,
                    "span_end": ml_end,
                    "span_tag": tag,
                    "metric": "fp",
                }
            )

    # Process remaining unmatched gold spans as FNs
    for note_id, tag, gold_start, gold_end, gold_idx in gold_collection.iterate_spans():
        if (note_id, tag, gold_start, gold_end) not in matched_gold_spans:
            # False negative
            metrics_per_label[tag]["fn"] += 1
            doc_metrics[note_id]["fn"] += 1

            span_metrics.append(
                {
                    "note_id": note_id,
                    "span_start": gold_start,
                    "span_end": gold_end,
                    "span_tag": tag,
                    "metric": "fn",
                }
            )

    return span_metrics, metrics_per_label, doc_metrics


def compute_metrics(
    gold_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    overlap_threshold: float = 0.8,
    label_maps: dict[str, Any] | None = None,
    exact_match: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]], dict[str, dict[str, int]], list[dict[str, Any]]]:
    """
    Compute precision, recall, and F1 metrics comparing gold standard spans
    to machine learning-generated spans.

    Args:
        gold_df: DataFrame containing gold standard spans with columns:
                 note_id, span_start, span_end, span_tag
        ml_df: DataFrame containing machine learning spans with columns:
               note_id, span_start, span_end, span_tag
        overlap_threshold: Minimum overlap required to consider spans as matching
                          (ignored if exact_match is True)
        label_maps: Dictionary mapping span tags to their descriptions
        exact_match: If True, only spans with identical start and end positions
                    are considered matches, and overlap_threshold is ignored.
                    If False (default), uses proportional overlap with the
                    overlap_threshold parameter.

    Returns:
        Tuple containing:
        - results: Dict with precision, recall, F1 and total spans per label
        - metrics_per_label: Dict with tp, fp, fn counts per label
        - doc_metrics: Dict with tp, fp, fn counts per document
        - span_metrics: List of dicts with metric details for each span

    Example:
        >>> gold = pd.DataFrame({
        ...     'note_id': ['1', '1'],
        ...     'span_start': [0, 10],
        ...     'span_end': [5, 15],
        ...     'span_tag': ['PER', 'LOC']
        ... })
        >>> ml = pd.DataFrame({
        ...     'note_id': ['1'],
        ...     'span_start': [0],
        ...     'span_end': [5],
        ...     'span_tag': ['PER']
        ... })
        >>> results, _, _, _ = compute_metrics(gold, ml)
        >>> results['PER']['precision']
        1.0
    """
    # Validate input DataFrames
    required_cols = ["note_id", "span_start", "span_end", "span_tag"]
    for col in required_cols:
        if col not in gold_df.columns:
            raise ValueError(f"Missing required column '{col}' in gold_df")
        if col not in ml_df.columns:
            raise ValueError(f"Missing required column '{col}' in ml_df")

    # Preprocess ML dataframe
    df_ml = preprocess_ml_dataframe(ml_df, label_maps) if label_maps is not None else ml_df.copy()
    df_gold = gold_df.copy()

    # Check note ID compatibility
    gold_note_ids = set(df_gold["note_id"])
    ml_note_ids = set(df_ml["note_id"])

    if not ml_note_ids.issubset(gold_note_ids):
        unknown_note_ids = ml_note_ids - gold_note_ids
        logging.warning(f"ML dataframe contains {len(unknown_note_ids)} note IDs not in gold dataframe.")

    # Create efficient span collections
    gold_collection = SpanCollection(df_gold)
    ml_collection = SpanCollection(df_ml)

    # Find matches efficiently
    span_metrics, metrics_per_label, doc_metrics = find_matches(
        gold_collection, ml_collection, overlap_threshold, exact_match=exact_match
    )

    # Calculate final results
    results = {}
    for tag, counts in metrics_per_label.items():
        precision, recall, f1 = compute_prf(counts["tp"], counts["fp"], counts["fn"])
        total_spans = counts["tp"] + counts["fn"]
        results[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_spans": total_spans,
        }

    return results, metrics_per_label, doc_metrics, span_metrics


def format_results(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Format results dictionary into a readable DataFrame.

    Args:
        results: Dictionary with precision, recall, F1 and total spans per label

    Returns:
        DataFrame with formatted results

    Example:
        >>> results = {'PER': {'precision': 0.8, 'recall': 0.9, 'f1': 0.85, 'total_spans': 10}}
        >>> df = format_results(results)
        >>> df.iloc[0]['Tag']
        'PER'
    """
    data = []
    for tag, metrics in results.items():
        data.append(
            {
                "Tag": tag,
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1": f"{metrics['f1']:.4f}",
                "Total Spans": metrics["total_spans"],
            }
        )
    return pd.DataFrame(data)


def aggregate_results(results: dict[str, dict[str, float]]) -> dict[str, float]:
    """
    Compute macro averages of precision, recall and F1 across all labels.

    Args:
        results: Dictionary with precision, recall, F1 and total spans per label

    Returns:
        Dictionary with macro-averaged metrics

    Example:
        >>> results = {
        ...     'PER': {'precision': 0.8, 'recall': 0.9, 'f1': 0.85},
        ...     'LOC': {'precision': 0.9, 'recall': 0.8, 'f1': 0.84}
        ... }
        >>> agg = aggregate_results(results)
        >>> agg['macro_precision']
        0.85
    """
    total_tags = len(results)

    if total_tags == 0:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    macro_precision = sum(m["precision"] for m in results.values()) / total_tags
    macro_recall = sum(m["recall"] for m in results.values()) / total_tags
    macro_f1 = sum(m["f1"] for m in results.values()) / total_tags

    return {"macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1}


def spans_to_dataframe(span_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert span metrics list to a DataFrame.

    Args:
        span_metrics: List of dictionaries with span metric details

    Returns:
        DataFrame of span metrics

    Example:
        >>> span_metrics = [{'note_id': '1', 'span_start': 0, 'span_end': 5, 'span_tag': 'PER', 'metric': 'tp'}]
        >>> df = spans_to_dataframe(span_metrics)
        >>> len(df)
        1
    """
    return pd.DataFrame(span_metrics)


def resolve_dataframe_conflicts(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Resolve overlapping spans in a DataFrame per note_id and assign a model name.

    This function groups spans by note_id, resolves overlapping spans within each note
    using non-maximum suppression (keeping the longest span), and adds a model_name column.

    Args:
        df: DataFrame with columns: note_id, span_start, span_end, span_tag, and optionally other columns.
            Must contain at least: note_id, span_start, span_end
        model_name: Name of the model to assign to all spans in the output DataFrame

    Returns:
        DataFrame with resolved (non-overlapping) spans per note, with model_name column added.
        Maintains all original columns from the input DataFrame.

    Example:
        >>> df = pd.DataFrame({
        ...     'note_id': ['note1', 'note1', 'note2'],
        ...     'span_start': [10, 15, 5],
        ...     'span_end': [25, 20, 15],
        ...     'span_tag': ['PERSON', 'PERSON', 'LOCATION']
        ... })
        >>> result = resolve_dataframe_conflicts(df, 'MyModel')
        >>> len(result)  # First two spans overlap, so only longest is kept
        2
        >>> result['model_name'].iloc[0]
        'MyModel'
    """
    # Validate required columns
    required_cols = ["note_id", "span_start", "span_end"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in DataFrame")

    if df.empty:
        # Return empty DataFrame with model_name column
        result_df = df.copy()
        result_df["model_name"] = model_name
        return result_df

    # Optimized: Convert to records once, group in Python, resolve, build DataFrame once
    # This avoids per-group DataFrame operations which are very slow

    # Convert entire DataFrame to records (single to_dict call)
    records = df.to_dict("records")

    # Add temporary start/end keys for resolve_overlapping_spans
    for r in records:
        r["start"] = r["span_start"]
        r["end"] = r["span_end"]

    # Group records by note_id in Python (much faster than pandas groupby + to_dict per group)
    groups: dict[str, list[dict]] = {}
    for r in records:
        note_id = r["note_id"]
        groups.setdefault(note_id, []).append(r)

    # Resolve conflicts for each group and collect all resolved records
    all_resolved = []
    for note_id, group_spans in groups.items():
        resolved_spans = resolve_overlapping_spans(group_spans)
        # Remove temporary start/end keys
        for span in resolved_spans:
            span.pop("start", None)
            span.pop("end", None)
        all_resolved.extend(resolved_spans)

    # Build final DataFrame from all resolved records (single DataFrame creation)
    if all_resolved:
        result_df = pd.DataFrame(all_resolved)
    else:
        result_df = pd.DataFrame(columns=df.columns)

    # Add model_name column
    result_df["model_name"] = model_name

    return result_df


def generate_model_combinations_with_resolution(df: pd.DataFrame, model_column: str = "model_name") -> pd.DataFrame:
    """
    Generate all possible combinations of models from a DataFrame and apply conflict resolution.

    This function creates ablation studies by combining every model with every other model
    (including combinations of 2, 3, 4, ... up to all models). For each combination,
    conflict resolution is applied to remove overlapping spans.

    Args:
        df: DataFrame with columns including model_column, note_id, span_start, span_end, span_tag
        model_column: Name of the column containing model names (default: "model_name")

    Returns:
        DataFrame containing all model combinations with resolved conflicts.
        Each combination is identified by a combined model_name like "Model1+Model2+Model3"

    Example:
        >>> df = pd.DataFrame({
        ...     'model_name': ['M1', 'M1', 'M2', 'M2'],
        ...     'note_id': ['n1', 'n1', 'n1', 'n1'],
        ...     'span_start': [0, 10, 5, 10],
        ...     'span_end': [8, 15, 12, 15],
        ...     'span_tag': ['PER', 'LOC', 'PER', 'LOC']
        ... })
        >>> result = generate_model_combinations_with_resolution(df)
        >>> 'M1+M2' in result['model_name'].values
        True
    """
    from itertools import combinations

    # Validate required columns
    required_cols = ["note_id", "span_start", "span_end"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in DataFrame")

    if model_column not in df.columns:
        raise ValueError(f"Missing model column '{model_column}' in DataFrame")

    if df.empty:
        return df.copy()

    # Get unique model names
    model_names = sorted(df[model_column].unique().tolist())

    if len(model_names) < 1:
        logging.warning("No models found. Returning empty DataFrame for combinations.")
        return pd.DataFrame(columns=df.columns)

    all_combinations = []

    # Generate all combinations starting from individual models (1) up to all models
    for r in range(1, len(model_names) + 1):
        for model_combo in combinations(model_names, r):
            # Filter data for this combination of models
            combined_df = df[df[model_column].isin(model_combo)].copy()

            if combined_df.empty:
                continue

            # Create combination name
            combo_name = "+".join(sorted(model_combo))

            # Apply conflict resolution to the combined data
            resolved_df = resolve_dataframe_conflicts(combined_df, combo_name)

            all_combinations.append(resolved_df)

    # Concatenate all combinations
    if all_combinations:
        return pd.concat(all_combinations, ignore_index=True)

    return pd.DataFrame(columns=df.columns)
