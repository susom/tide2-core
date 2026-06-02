"""
Text Processing Utilities for TIDE 2.0.

This module provides utilities for text chunking, token aggregation, and span
reconstruction for entity recognition workflows.

Key Functions:
- compute_text_hash: Compute SHA256 hash for text identification
- split_text_to_word_chunks: Split text into overlapping chunks with optional metadata
- sort_tokens_by_position: Sort tokens by start position for BIO aggregation
- aggregate_bio_tokens: Aggregate BIO-tagged tokens into continuous entity spans
- reconstruct_document_spans: Map chunk-local spans to document-global coordinates
- deduplicate_overlapping_entities: Remove duplicate entities using IoU threshold

Author: TIDE 2.0 Team
Updated: January 2026
"""

from __future__ import annotations

import hashlib


def compute_text_hash(text: str) -> str:
    """
    Compute SHA256 hash of the given text for consistent identification.

    This function is used throughout TIDE 2.0 for generating stable identifiers
    for text documents, enabling O(1) lookup of cached results.

    Args:
        text: Raw text to hash

    Returns:
        Hexadecimal SHA256 hash string

    Example:
        >>> compute_text_hash("Hello world")
        '64ec88ca00b268e5ba1a35678a1b5316d212f4f366b2477232534a8aeca37f3c'
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_text_to_word_chunks(
    input_length: int, chunk_length: int, overlap_length: int, return_metadata: bool = False
) -> list[list[int]] | list[dict[str, int]]:
    """
    Calculate chunks of text with size chunk_length and overlap_length for context.

    This function works in token space using the approximation: 1 token ≈ 4 characters.
    The chunk_length and overlap_length parameters are in tokens, and the function
    converts them to character positions for output.

    Args:
        input_length: Length of input text in characters
        chunk_length: Length of each chunk in tokens (e.g., 512 tokens)
        overlap_length: Number of overlapping tokens in each chunk (e.g., 40 tokens)
        return_metadata: If True, return list of dicts with chunk_id and offsets.
                        If False (default), return [[start, end]] for backward compatibility.

    Returns:
        If return_metadata=False: List of [start, end] character positions for individual text chunks
        If return_metadata=True: List of dicts with keys:
            - start: Start character position in original text
            - end: End character position in original text
            - chunk_id: Sequential chunk identifier
            - char_offset_start: Same as start (for compatibility)
            - char_offset_end: Same as end (for compatibility)

    Examples:
        >>> # Legacy mode (backward compatible)
        >>> # 400 chars ≈ 100 tokens, chunk_length=50 tokens, overlap=10 tokens
        >>> split_text_to_word_chunks(400, 50, 10)
        [[0, 200], [160, 360], [320, 400]]

        >>> # New mode with metadata
        >>> split_text_to_word_chunks(400, 50, 10, return_metadata=True)
        [
            {"start": 0, "end": 200, "chunk_id": 0, "char_offset_start": 0, "char_offset_end": 200},
            {"start": 160, "end": 360, "chunk_id": 1, "char_offset_start": 160, "char_offset_end": 360},
            {"start": 320, "end": 400, "chunk_id": 2, "char_offset_start": 320, "char_offset_end": 400}
        ]
    """
    # Convert character length to approximate token count (1 token ≈ 4 characters)
    input_length_tokens = input_length // 4

    # Convert chunk and overlap sizes from tokens to characters
    chunk_length_chars = chunk_length * 4
    overlap_length_chars = overlap_length * 4

    if input_length_tokens < chunk_length:
        if return_metadata:
            return [
                {
                    "start": 0,
                    "end": input_length,
                    "chunk_id": 0,
                    "char_offset_start": 0,
                    "char_offset_end": input_length,
                }
            ]
        return [[0, input_length]]

    if chunk_length <= overlap_length:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "overlap_length should be shorter than chunk_length, setting overlap_length to half of chunk_length"
        )
        overlap_length = chunk_length // 2
        overlap_length_chars = overlap_length * 4

    chunks = []

    # Calculate step size in characters (chunk_length - overlap_length) in token space → chars
    step_size_chars = (chunk_length - overlap_length) * 4

    for chunk_id, i in enumerate(range(0, input_length - overlap_length_chars, step_size_chars)):
        start = i
        end = min(i + chunk_length_chars, input_length)

        if return_metadata:
            chunks.append(
                {"start": start, "end": end, "chunk_id": chunk_id, "char_offset_start": start, "char_offset_end": end}
            )
        else:
            chunks.append([start, end])

    return chunks


def sort_tokens_by_position(tokens: list[dict]) -> list[dict]:
    """
    Sort tokens by their start position in the text.

    This method ensures tokens are processed in the correct order for BIO aggregation,
    as the input tokens may not be sorted by default.

    Args:
        tokens: List of token dictionaries with 'start' key

    Returns:
        Sorted list of tokens by start position

    Example:
        >>> tokens = [
        ...     {"start": 10, "end": 15, "entity": "PERSON"},
        ...     {"start": 5, "end": 8, "entity": "LOCATION"}
        ... ]
        >>> sorted_tokens = sort_tokens_by_position(tokens)
        >>> sorted_tokens[0]["start"]
        5
    """
    return sorted(tokens, key=lambda token: token["start"])


def _get_entity_type(entity_tag: str) -> str:
    """Extract entity type from BIO tag (remove B-/I- prefix)."""
    if entity_tag.startswith(("B-", "I-")):
        return entity_tag[2:]
    return entity_tag


def _is_word_start_token(token: dict) -> bool:
    """
    Check if a token represents the start of a new word.

    Different tokenizers use different conventions:
    - BPE (GPT, RoBERTa): 'Ġ' prefix indicates word start (space before word)
    - SentencePiece (BERT, DeBERTa): '▁' prefix indicates word start
    - WordPiece (BERT): '##' prefix indicates continuation (no prefix = word start)

    Args:
        token: Token dictionary with 'word' key

    Returns:
        True if this token starts a new word
    """
    word = token.get("word", "")
    if not word:
        return True  # Assume word start if no word info

    # BPE style (GPT, RoBERTa, ModernBERT): Ġ prefix = space = new word
    if word.startswith("Ġ"):
        return True

    # SentencePiece style (some BERT variants, DeBERTa): ▁ prefix = new word
    if word.startswith("▁"):
        return True

    # WordPiece style (BERT): ## prefix = continuation, no prefix = new word
    # But we need to be careful - if first token, it's always a word start
    if word.startswith("##"):
        return False

    return False  # Conservative default for continuation tokens


def _normalize_token_labels(
    tokens: list[dict],
    original_text: str,
    max_gap: int = 2,
    high_confidence_threshold: float = 0.8,
) -> list[dict]:
    """
    Normalize entity labels across contiguous token groups within word boundaries.

    This function addresses the issue where transformer models predict inconsistent
    entity types for consecutive subword tokens of the same word (e.g., predicting
    'Al' as HOSPITAL, 'n' as PATIENT, 'ion' as HOSPITAL for "Alnion").

    The normalization only applies when:
    1. Tokens are near-contiguous (gap <= max_gap)
    2. All tokens in the group have low confidence (< high_confidence_threshold)
    3. The next token is NOT a word-start token (detected via tokenizer prefixes)

    Args:
        tokens: Sorted list of token dictionaries
        original_text: Original text for checking word boundaries
        max_gap: Maximum character gap to consider tokens as part of the same group
        high_confidence_threshold: Tokens above this score are treated as confident
            predictions that shouldn't be overridden by label voting

    Returns:
        List of tokens with normalized entity labels
    """
    if not tokens:
        return []

    # Word boundary characters - if gap contains these, don't merge
    word_boundary_chars = {" ", "\n", "\t", "\r"}

    # Group tokens by near-contiguity within word boundaries
    groups = []
    current_group = [tokens[0]]

    for i in range(1, len(tokens)):
        prev_token = tokens[i - 1]
        curr_token = tokens[i]
        gap = curr_token["start"] - prev_token["end"]

        # Check if gap is small enough and doesn't cross word boundary
        should_merge = gap <= max_gap

        # Check for word boundary in gap text
        if should_merge and gap > 0:
            gap_text = original_text[prev_token["end"] : curr_token["start"]]
            if any(c in word_boundary_chars for c in gap_text):
                should_merge = False

        # Check for word-start token (indicates new word even if contiguous)
        if should_merge and _is_word_start_token(curr_token):
            should_merge = False

        if should_merge:
            # Near-contiguous within same word, add to current group
            current_group.append(curr_token)
        else:
            # Gap too large, crosses word boundary, or new word starts
            groups.append(current_group)
            current_group = [curr_token]

    # Don't forget the last group
    groups.append(current_group)

    # Normalize labels within each group
    normalized_tokens = []
    for group in groups:
        if len(group) == 1:
            # Single token, no normalization needed
            normalized_tokens.append(group[0])
            continue

        # Check if all tokens have the same entity type
        entity_types = {}
        has_high_confidence = False
        high_confidence_type = None

        for token in group:
            entity_type = _get_entity_type(token["entity"])
            if entity_type not in entity_types:
                entity_types[entity_type] = {"score": 0.0, "count": 0, "max_score": 0.0}
            entity_types[entity_type]["score"] += token["score"]
            entity_types[entity_type]["count"] += 1
            entity_types[entity_type]["max_score"] = max(entity_types[entity_type]["max_score"], token["score"])

            # Track if any token has high confidence
            if token["score"] >= high_confidence_threshold:
                has_high_confidence = True
                high_confidence_type = entity_type

        if len(entity_types) == 1:
            # All same type, no normalization needed
            normalized_tokens.extend(group)
            continue

        # Multiple entity types - determine winning type
        if has_high_confidence and high_confidence_type:
            # If there's a high-confidence token, use its type
            best_type = high_confidence_type
        else:
            # Otherwise, use highest average score to pick winner
            best_type = max(
                entity_types.keys(),
                key=lambda t: entity_types[t]["score"] / entity_types[t]["count"],
            )

        # Normalize all tokens in the group to the winning type
        for j, token in enumerate(group):
            current_type = _get_entity_type(token["entity"])
            if current_type != best_type:
                # Create a copy with normalized entity type
                new_token = token.copy()
                # Preserve B-/I- prefix pattern
                if token["entity"].startswith("B-"):
                    new_token["entity"] = f"B-{best_type}"
                elif token["entity"].startswith("I-"):
                    new_token["entity"] = f"I-{best_type}"
                else:
                    new_token["entity"] = best_type
                normalized_tokens.append(new_token)
            else:
                normalized_tokens.append(token)

    return normalized_tokens


def aggregate_bio_tokens(
    tokens: list[dict],
    original_text: str,
    max_gap: int = 2,
    normalize_labels: bool = True,
) -> list[dict]:
    """
    Aggregate BIO-tagged tokens into continuous spans with the same entity tag.

    This method processes tokens tagged with B-<tag>, I-<tag> format and groups
    consecutive tokens with the same entity type into single spans. It handles:
    - B-<tag> followed by I-<tag> tokens
    - Consecutive B-<tag> tokens if they are near-contiguous (gap <= max_gap)
    - Standalone I-<tag> tokens (starts new span)
    - Score averaging across aggregated tokens
    - Label normalization for inconsistent subword predictions (optional)

    Note: O tags are never provided as input to this method.

    Args:
        tokens: List of token dictionaries with 'entity', 'score', 'start', 'end' keys
        original_text: Original text to reconstruct aggregated spans
        max_gap: Maximum character gap to allow when merging tokens (default: 2).
            This handles cases where punctuation (like periods) creates small gaps
            between tokens that should be part of the same entity.
        normalize_labels: If True, normalize inconsistent entity labels within
            contiguous token groups using highest-score voting (default: True).
            This fixes issues where models predict different entity types for
            consecutive subword tokens of the same word.

    Returns:
        List of aggregated entity spans with keys:
            - entity_group: Entity type without B-/I- prefix
            - score: Average score across aggregated tokens
            - word: Extracted text from original text
            - start: Start position in original text
            - end: End position in original text

    Example:
        >>> tokens = [
        ...     {"entity": "B-PERSON", "score": 0.9, "start": 0, "end": 4},
        ...     {"entity": "I-PERSON", "score": 0.85, "start": 5, "end": 8}
        ... ]
        >>> aggregated = aggregate_bio_tokens(tokens, "John Doe")
        >>> aggregated[0]["entity_group"]
        'PERSON'
        >>> aggregated[0]["word"]
        'John Doe'
    """
    if not tokens:
        return []

    # Sort tokens by their start position to ensure correct processing order
    sorted_tokens = sort_tokens_by_position(tokens)

    # Normalize labels if requested (fixes inconsistent subword predictions)
    if normalize_labels:
        sorted_tokens = _normalize_token_labels(sorted_tokens, original_text, max_gap=max_gap)

    aggregated_spans = []
    current_span_tokens = []
    current_entity_type = None

    for token in sorted_tokens:
        entity_tag = token["entity"]

        # Parse BIO tag
        if entity_tag.startswith("B-"):
            entity_type = entity_tag[2:]  # Remove 'B-' prefix

            # Check if we can extend current span (same entity type and near-contiguous)
            gap = token["start"] - current_span_tokens[-1]["end"] if current_span_tokens else float("inf")
            if current_entity_type == entity_type and current_span_tokens and gap <= max_gap:
                # Extend current span with this B-tag token
                current_span_tokens.append(token)
            else:
                # Finalize previous span if exists
                if current_span_tokens:
                    aggregated_spans.append(_finalize_span(current_span_tokens, original_text))

                # Start new span
                current_span_tokens = [token]
                current_entity_type = entity_type

        elif entity_tag.startswith("I-"):
            entity_type = entity_tag[2:]  # Remove 'I-' prefix

            # If same entity type as current span and near-contiguous, add to it
            gap = token["start"] - current_span_tokens[-1]["end"] if current_span_tokens else float("inf")
            if current_entity_type == entity_type and current_span_tokens and gap <= max_gap:
                current_span_tokens.append(token)
            else:
                # Finalize previous span if exists
                if current_span_tokens:
                    aggregated_spans.append(_finalize_span(current_span_tokens, original_text))

                # Start new span with I-tag (as specified in requirements)
                current_span_tokens = [token]
                current_entity_type = entity_type

        # Handle tags without B- or I- prefix (treat as B-tag)
        else:
            # Check if we can extend current span (same entity type and near-contiguous)
            gap = token["start"] - current_span_tokens[-1]["end"] if current_span_tokens else float("inf")
            if current_entity_type == entity_tag and current_span_tokens and gap <= max_gap:
                # Extend current span
                current_span_tokens.append(token)
            else:
                # Finalize previous span if exists
                if current_span_tokens:
                    aggregated_spans.append(_finalize_span(current_span_tokens, original_text))

                # Start new span
                current_span_tokens = [token]
                current_entity_type = entity_tag

    # Finalize the last span if exists
    if current_span_tokens:
        aggregated_spans.append(_finalize_span(current_span_tokens, original_text))

    return aggregated_spans


def _finalize_span(span_tokens: list[dict], original_text: str) -> dict:
    """
    Finalize a span by aggregating token information.

    Args:
        span_tokens: List of tokens belonging to the same entity span
        original_text: Original text to extract the span text from

    Returns:
        Aggregated span dictionary with keys:
            - entity_group: Entity type without B-/I- prefix
            - score: Average score across tokens
            - word: Extracted text from original text
            - start: Start position
            - end: End position

    Raises:
        ValueError: If span_tokens is empty
    """
    if not span_tokens:
        raise ValueError("Cannot finalize empty span")

    # Calculate span boundaries
    start_pos = min(token["start"] for token in span_tokens)
    end_pos = max(token["end"] for token in span_tokens)

    # Extract text from original text using span boundaries
    span_text = original_text[start_pos:end_pos]

    # Calculate average score
    avg_score = sum(token["score"] for token in span_tokens) / len(span_tokens)

    # Get entity type (remove B- or I- prefix if present)
    entity_tag = span_tokens[0]["entity"]
    entity_group = entity_tag[2:] if entity_tag.startswith(("B-", "I-")) else entity_tag

    return {
        "entity_group": entity_group,
        "score": float(avg_score),
        "word": span_text,
        "start": start_pos,
        "end": end_pos,
    }


def reconstruct_document_spans(chunk_predictions: list[dict], original_text: str) -> list[dict]:
    """
    Reconstruct entity spans relative to original document positions.

    Takes entity predictions from individual chunks (with chunk-local offsets)
    and maps them back to document-global coordinates by adding chunk offsets.

    Args:
        chunk_predictions: List of dicts with keys:
            - chunk_id: Chunk identifier
            - char_offset_start: Start position of chunk in document
            - predictions: List of entities with chunk-local 'start' and 'end'
        original_text: Original document text for extracting entity text

    Returns:
        List of entities with document-global coordinates:
            - entity: Entity type
            - score: Confidence score
            - start: Global start position in document
            - end: Global end position in document
            - text: Extracted text from document
            - chunk_id: Source chunk identifier

    Example:
        >>> chunk_preds = [
        ...     {
        ...         "chunk_id": 0,
        ...         "char_offset_start": 0,
        ...         "predictions": [{"entity_group": "PERSON", "score": 0.9, "start": 0, "end": 4}]
        ...     },
        ...     {
        ...         "chunk_id": 1,
        ...         "char_offset_start": 100,
        ...         "predictions": [{"entity_group": "LOCATION", "score": 0.85, "start": 10, "end": 17}]
        ...     }
        ... ]
        >>> original_text = "John lives in Seattle"
        >>> entities = reconstruct_document_spans(chunk_preds, original_text)
        >>> entities[1]["start"]  # Location entity in second chunk
        110
    """
    all_entities = []

    for chunk_pred in chunk_predictions:
        char_offset = chunk_pred["char_offset_start"]

        for entity in chunk_pred["predictions"]:
            # Adjust spans from chunk-local to document-global coordinates
            global_start = int(entity["start"] + char_offset)
            global_end = int(entity["end"] + char_offset)

            all_entities.append(
                {
                    "entity": entity["entity_group"],
                    "score": entity["score"],
                    "start": global_start,
                    "end": global_end,
                    "text": original_text[global_start:global_end],
                    "chunk_id": chunk_pred["chunk_id"],
                }
            )

    return all_entities


def calculate_iou(span1: tuple[int, int], span2: tuple[int, int]) -> float:
    """
    Calculate Intersection over Union for two spans.

    Args:
        span1: Tuple of (start, end) for first span
        span2: Tuple of (start, end) for second span

    Returns:
        IoU value between 0.0 and 1.0

    Example:
        >>> calculate_iou((0, 10), (5, 15))
        0.5
        >>> calculate_iou((0, 10), (10, 20))
        0.0
    """
    start1, end1 = span1
    start2, end2 = span2

    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)

    return intersection / union if union > 0 else 0.0


def deduplicate_overlapping_entities(entities: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """
    Remove duplicate entities in overlapping regions using IoU threshold.

    When multiple entities overlap with IoU >= threshold, keeps the one
    with the highest confidence score.

    Uses interval tree for O(n log n) overlap detection instead of O(n²).

    Args:
        entities: List of entity dicts with 'start', 'end', 'score' keys
        iou_threshold: Minimum IoU to consider entities as overlapping (default: 0.5)

    Returns:
        Deduplicated list of entities

    Example:
        >>> entities = [
        ...     {"entity": "PERSON", "score": 0.9, "start": 0, "end": 10, "text": "John Doe"},
        ...     {"entity": "PERSON", "score": 0.7, "start": 5, "end": 10, "text": "Doe"}
        ... ]
        >>> deduped = deduplicate_overlapping_entities(entities, iou_threshold=0.3)
        >>> len(deduped)
        1
        >>> deduped[0]["score"]
        0.9
    """
    if not entities:
        return []

    from intervaltree import IntervalTree

    # Build interval tree with all entities.
    tree = IntervalTree()
    for i, entity in enumerate(entities):
        tree[entity["start"] : entity["end"]] = i

    skip_indices: set[int] = set()

    # Deterministic ordering: highest score wins; on ties prefer leftmost, then
    # narrowest, then alphabetical entity type. This tuple is chosen so the
    # equivalent SQL implementation can use ROW_NUMBER() OVER (ORDER BY
    # score DESC, start ASC, end ASC, entity ASC) and produce identical output.
    def _dedup_key(item: tuple[int, dict]) -> tuple[float, int, int, str]:
        _, e = item
        return (-e["score"], e["start"], e["end"], e["entity"])

    ordered = sorted(enumerate(entities), key=_dedup_key)

    for i, entity1 in ordered:
        if i in skip_indices:
            continue

        overlaps = tree[entity1["start"] : entity1["end"]]
        for interval in overlaps:
            j = interval.data
            if j == i or j in skip_indices:
                continue

            entity2 = entities[j]
            iou = calculate_iou((entity1["start"], entity1["end"]), (entity2["start"], entity2["end"]))
            if iou >= iou_threshold:
                skip_indices.add(j)

    return [entity for i, entity in enumerate(entities) if i not in skip_indices]
