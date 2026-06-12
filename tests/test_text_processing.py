"""Tests for utils/text_processing.py — chunking, BIO aggregation, span reconstruction."""

import pytest

from tide2.utils.text_processing import _finalize_span
from tide2.utils.text_processing import _get_entity_type
from tide2.utils.text_processing import _is_word_start_token
from tide2.utils.text_processing import _normalize_token_labels
from tide2.utils.text_processing import aggregate_bio_tokens
from tide2.utils.text_processing import calculate_iou
from tide2.utils.text_processing import compute_text_hash
from tide2.utils.text_processing import deduplicate_overlapping_entities
from tide2.utils.text_processing import reconstruct_document_spans
from tide2.utils.text_processing import sort_tokens_by_position
from tide2.utils.text_processing import split_text_to_word_chunks


class TestComputeTextHash:
    def test_deterministic(self):
        assert compute_text_hash("hello") == compute_text_hash("hello")

    def test_different_inputs(self):
        assert compute_text_hash("a") != compute_text_hash("b")

    def test_known_value(self):
        import hashlib

        expected = hashlib.sha256(b"test").hexdigest()
        assert compute_text_hash("test") == expected


class TestSplitTextToWordChunks:
    def test_short_text_single_chunk(self):
        # 100 chars < 50 tokens * 4 = 200 chars
        result = split_text_to_word_chunks(100, 50, 10)
        assert result == [[0, 100]]

    def test_short_text_metadata(self):
        result = split_text_to_word_chunks(100, 50, 10, return_metadata=True)
        assert len(result) == 1
        assert result[0]["start"] == 0
        assert result[0]["end"] == 100
        assert result[0]["chunk_id"] == 0

    def test_multiple_chunks(self):
        # 800 chars = 200 tokens, chunk_length=50, overlap=10 → step=40 tokens=160 chars
        result = split_text_to_word_chunks(800, 50, 10)
        assert len(result) > 1
        # First chunk starts at 0
        assert result[0][0] == 0

    def test_multiple_chunks_metadata(self):
        result = split_text_to_word_chunks(800, 50, 10, return_metadata=True)
        assert len(result) > 1
        for i, chunk in enumerate(result):
            assert chunk["chunk_id"] == i
            assert chunk["char_offset_start"] == chunk["start"]

    def test_overlap_warning(self):
        # overlap >= chunk_length should warn and adjust
        result = split_text_to_word_chunks(800, 50, 50)
        # Should still produce valid chunks
        assert len(result) > 0

    def test_chunks_cover_full_text(self):
        result = split_text_to_word_chunks(1000, 50, 10)
        # Last chunk should reach end of text
        assert result[-1][1] == 1000


class TestSortTokensByPosition:
    def test_sorts_by_start(self):
        tokens = [
            {"start": 10, "end": 15, "entity": "B-PERSON"},
            {"start": 0, "end": 5, "entity": "B-LOCATION"},
        ]
        sorted_t = sort_tokens_by_position(tokens)
        assert sorted_t[0]["start"] == 0
        assert sorted_t[1]["start"] == 10

    def test_empty_list(self):
        assert sort_tokens_by_position([]) == []


class TestGetEntityType:
    def test_b_prefix(self):
        assert _get_entity_type("B-PERSON") == "PERSON"

    def test_i_prefix(self):
        assert _get_entity_type("I-LOCATION") == "LOCATION"

    def test_no_prefix(self):
        assert _get_entity_type("DATE") == "DATE"


class TestIsWordStartToken:
    def test_bpe_prefix(self):
        assert _is_word_start_token({"word": "\u0120hello"}) is True

    def test_sentencepiece_prefix(self):
        assert _is_word_start_token({"word": "\u2581hello"}) is True

    def test_wordpiece_continuation(self):
        assert _is_word_start_token({"word": "##ing"}) is False

    def test_empty_word(self):
        assert _is_word_start_token({"word": ""}) is True

    def test_no_word_key(self):
        assert _is_word_start_token({}) is True

    def test_plain_subword(self):
        # No special prefix, no ## → conservative default False
        assert _is_word_start_token({"word": "ing"}) is False


class TestNormalizeTokenLabels:
    def test_empty_input(self):
        assert _normalize_token_labels([], "text") == []

    def test_single_token(self):
        tokens = [{"start": 0, "end": 3, "entity": "B-PERSON", "score": 0.9, "word": "abc"}]
        result = _normalize_token_labels(tokens, "abcdef")
        assert len(result) == 1

    def test_normalizes_mixed_labels_in_same_word(self):
        # Contiguous subword tokens with mixed labels → should normalize
        tokens = [
            {"start": 0, "end": 2, "entity": "B-HOSPITAL", "score": 0.5, "word": "Al"},
            {"start": 2, "end": 3, "entity": "B-PATIENT", "score": 0.4, "word": "n"},
            {"start": 3, "end": 6, "entity": "B-HOSPITAL", "score": 0.6, "word": "ion"},
        ]
        result = _normalize_token_labels(tokens, "Alnion")
        # All should be normalized to HOSPITAL (majority type by avg score)
        types = {_get_entity_type(t["entity"]) for t in result}
        assert len(types) == 1

    def test_word_boundary_prevents_merging(self):
        tokens = [
            {"start": 0, "end": 4, "entity": "B-PERSON", "score": 0.9, "word": "John"},
            {"start": 5, "end": 8, "entity": "B-LOCATION", "score": 0.9, "word": "NYC"},
        ]
        result = _normalize_token_labels(tokens, "John NYC")
        types = [_get_entity_type(t["entity"]) for t in result]
        assert "PERSON" in types
        assert "LOCATION" in types


class TestAggregateBioTokens:
    def test_empty_input(self):
        assert aggregate_bio_tokens([], "") == []

    def test_single_b_token(self):
        tokens = [{"entity": "B-PERSON", "score": 0.9, "start": 0, "end": 4}]
        result = aggregate_bio_tokens(tokens, "John Smith")
        assert len(result) == 1
        assert result[0]["entity_group"] == "PERSON"
        assert result[0]["word"] == "John"

    def test_b_followed_by_i(self):
        tokens = [
            {"entity": "B-PERSON", "score": 0.9, "start": 0, "end": 4},
            {"entity": "I-PERSON", "score": 0.85, "start": 5, "end": 10},
        ]
        result = aggregate_bio_tokens(tokens, "John Smith")
        assert len(result) == 1
        assert result[0]["word"] == "John Smith"
        assert result[0]["score"] == pytest.approx(0.875)

    def test_different_entity_types_create_separate_spans(self):
        tokens = [
            {"entity": "B-PERSON", "score": 0.9, "start": 0, "end": 4},
            {"entity": "B-LOCATION", "score": 0.8, "start": 13, "end": 21},
        ]
        result = aggregate_bio_tokens(tokens, "John went to Stanford", normalize_labels=False)
        assert len(result) == 2
        assert result[0]["entity_group"] == "PERSON"
        assert result[1]["entity_group"] == "LOCATION"

    def test_standalone_i_tag_starts_new_span(self):
        tokens = [
            {"entity": "I-PERSON", "score": 0.9, "start": 0, "end": 4},
        ]
        result = aggregate_bio_tokens(tokens, "John", normalize_labels=False)
        assert len(result) == 1
        assert result[0]["entity_group"] == "PERSON"

    def test_tag_without_prefix(self):
        tokens = [
            {"entity": "DATE", "score": 0.8, "start": 0, "end": 10},
        ]
        result = aggregate_bio_tokens(tokens, "2024-01-01", normalize_labels=False)
        assert len(result) == 1
        assert result[0]["entity_group"] == "DATE"

    def test_near_contiguous_b_tags_same_type_merge(self):
        # Gap of 1 char (space), should merge since gap <= max_gap (2)
        tokens = [
            {"entity": "B-PERSON", "score": 0.9, "start": 0, "end": 3},
            {"entity": "B-PERSON", "score": 0.8, "start": 4, "end": 9},
        ]
        result = aggregate_bio_tokens(tokens, "Dr. Smith", normalize_labels=False)
        assert len(result) == 1
        assert result[0]["word"] == "Dr. Smith"


class TestFinalizeSpan:
    def test_basic(self):
        tokens = [
            {"entity": "B-PERSON", "score": 0.9, "start": 0, "end": 4},
            {"entity": "I-PERSON", "score": 0.8, "start": 5, "end": 10},
        ]
        result = _finalize_span(tokens, "John Smith is here")
        assert result["entity_group"] == "PERSON"
        assert result["word"] == "John Smith"
        assert result["score"] == pytest.approx(0.85)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _finalize_span([], "text")


class TestReconstructDocumentSpans:
    def test_single_chunk(self):
        chunks = [
            {
                "chunk_id": 0,
                "char_offset_start": 0,
                "predictions": [
                    {"entity_group": "PERSON", "score": 0.9, "start": 0, "end": 4},
                ],
            }
        ]
        result = reconstruct_document_spans(chunks, "John lives in Seattle")
        assert len(result) == 1
        assert result[0]["start"] == 0
        assert result[0]["text"] == "John"

    def test_offset_applied(self):
        chunks = [
            {
                "chunk_id": 1,
                "char_offset_start": 100,
                "predictions": [
                    {"entity_group": "LOCATION", "score": 0.8, "start": 10, "end": 17},
                ],
            }
        ]
        # Need a text long enough for the global span
        text = "x" * 200
        result = reconstruct_document_spans(chunks, text)
        assert result[0]["start"] == 110
        assert result[0]["end"] == 117

    def test_empty_predictions(self):
        chunks = [{"chunk_id": 0, "char_offset_start": 0, "predictions": []}]
        assert reconstruct_document_spans(chunks, "text") == []


class TestCalculateIou:
    def test_full_overlap(self):
        assert calculate_iou((0, 10), (0, 10)) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert calculate_iou((0, 10), (10, 20)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # intersection=5, union span=15, IoU = 5/15 = 1/3
        assert calculate_iou((0, 10), (5, 15)) == pytest.approx(1 / 3)

    def test_contained(self):
        # intersection=6, union span=10, IoU = 6/10 = 0.6
        iou = calculate_iou((0, 10), (2, 8))
        assert iou == pytest.approx(0.6)


class TestDeduplicateOverlappingEntities:
    def test_empty_list(self):
        assert deduplicate_overlapping_entities([]) == []

    def test_no_overlap(self):
        entities = [
            {"entity": "PERSON", "score": 0.9, "start": 0, "end": 4},
            {"entity": "LOCATION", "score": 0.8, "start": 10, "end": 17},
        ]
        result = deduplicate_overlapping_entities(entities)
        assert len(result) == 2

    def test_overlapping_keeps_higher_score(self):
        entities = [
            {"entity": "PERSON", "score": 0.7, "start": 0, "end": 10},
            {"entity": "PERSON", "score": 0.9, "start": 5, "end": 10},
        ]
        result = deduplicate_overlapping_entities(entities, iou_threshold=0.3)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_low_iou_keeps_both(self):
        entities = [
            {"entity": "PERSON", "score": 0.9, "start": 0, "end": 10},
            {"entity": "PERSON", "score": 0.7, "start": 8, "end": 20},
        ]
        # IoU is low (2 / 20 = 0.1), threshold 0.5 → keep both
        result = deduplicate_overlapping_entities(entities, iou_threshold=0.5)
        assert len(result) == 2
