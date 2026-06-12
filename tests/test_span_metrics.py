"""
Unit tests for the span_metrics module.

This module tests the functionality for computing metrics between gold standard
and machine learning-generated text spans, as well as conflict resolution.
"""

import pandas as pd
import pytest

from tide2.utils.span_metrics import aggregate_results
from tide2.utils.span_metrics import compute_metrics
from tide2.utils.span_metrics import compute_prf
from tide2.utils.span_metrics import format_results
from tide2.utils.span_metrics import generate_model_combinations_with_resolution
from tide2.utils.span_metrics import resolve_conflicts
from tide2.utils.span_metrics import resolve_dataframe_conflicts
from tide2.utils.span_metrics import resolve_overlapping_spans
from tide2.utils.span_metrics import resolve_recognizer_results
from tide2.utils.span_metrics import span_overlap
from tide2.utils.span_metrics import spans_to_dataframe


class TestSpanOverlap:
    """Test cases for the span_overlap function."""

    def test_complete_overlap(self):
        """Test when spans completely overlap."""
        span1 = (10, 20)
        span2 = (10, 20)
        assert span_overlap(span1, span2) == 1.0

    def test_no_overlap(self):
        """Test when spans don't overlap at all."""
        span1 = (10, 20)
        span2 = (30, 40)
        assert span_overlap(span1, span2) == 0.0

    def test_partial_overlap_50_percent(self):
        """Test when spans have 50% overlap."""
        span1 = (10, 20)  # length 10
        span2 = (15, 25)  # overlap from 15-20 = 5, so 5/10 = 0.5
        assert span_overlap(span1, span2) == 0.5

    def test_partial_overlap_contained(self):
        """Test when first span is completely contained in second."""
        span1 = (15, 20)  # length 5
        span2 = (10, 25)  # completely contains span1
        assert span_overlap(span1, span2) == 1.0

    def test_partial_overlap_containing(self):
        """Test when second span is completely contained in first."""
        span1 = (10, 25)  # length 15
        span2 = (15, 20)  # length 5, contained in span1
        assert span_overlap(span1, span2) == 5 / 15  # 1/3

    def test_edge_overlap(self):
        """Test when spans touch at edges."""
        span1 = (10, 20)
        span2 = (20, 30)
        assert span_overlap(span1, span2) == 0.0


class TestComputePRF:
    """Test cases for the compute_prf function."""

    def test_perfect_precision_recall(self):
        """Test when all predictions are correct."""
        precision, recall, f1 = compute_prf(tp=10, fp=0, fn=0)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_zero_precision_recall(self):
        """Test when no correct predictions."""
        precision, recall, f1 = compute_prf(tp=0, fp=10, fn=10)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_mixed_results(self):
        """Test with mixed tp, fp, fn."""
        precision, recall, f1 = compute_prf(tp=6, fp=2, fn=3)
        # precision = 6/(6+2) = 0.75
        # recall = 6/(6+3) = 0.667
        # f1 = 2*0.75*0.667/(0.75+0.667) ≈ 0.706
        assert abs(precision - 0.75) < 0.001
        assert abs(recall - 6 / 9) < 0.001
        expected_f1 = 2 * 0.75 * (6 / 9) / (0.75 + 6 / 9)
        assert abs(f1 - expected_f1) < 0.001

    def test_only_false_positives(self):
        """Test when only false positives exist."""
        precision, recall, f1 = compute_prf(tp=0, fp=5, fn=0)
        assert precision == 0.0
        assert recall == 0.0  # No gold standard to miss
        assert f1 == 0.0

    def test_only_false_negatives(self):
        """Test when only false negatives exist."""
        precision, recall, f1 = compute_prf(tp=0, fp=0, fn=5)
        assert precision == 0.0  # No predictions made
        assert recall == 0.0
        assert f1 == 0.0


class TestComputeMetrics:
    """Test cases for the compute_metrics function."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Create sample gold standard data
        self.gold_df = pd.DataFrame(
            {
                "note_id": ["doc1", "doc1", "doc2", "doc2", "doc3"],
                "span_start": [10, 30, 50, 70, 90],
                "span_end": [20, 40, 60, 80, 100],
                "span_tag": ["PERSON", "LOCATION", "PERSON", "ORGANIZATION", "PERSON"],
            }
        )

        # Create sample ML predictions data
        self.ml_df = pd.DataFrame(
            {
                "note_id": ["doc1", "doc1", "doc2", "doc2"],
                "span_start": [12, 30, 50, 75],
                "span_end": [18, 40, 60, 85],
                "span_tag": ["PERSON", "LOCATION", "PERSON", "ORGANIZATION"],
            }
        )

    def test_perfect_match(self):
        """Test when ML predictions perfectly match gold standard."""
        gold_df = pd.DataFrame(
            {
                "note_id": ["doc1", "doc1"],
                "span_start": [10, 30],
                "span_end": [20, 40],
                "span_tag": ["PERSON", "LOCATION"],
            }
        )

        ml_df = pd.DataFrame(
            {
                "note_id": ["doc1", "doc1"],
                "span_start": [10, 30],
                "span_end": [20, 40],
                "span_tag": ["PERSON", "LOCATION"],
            }
        )

        results, _metrics, _doc_metrics, _span_metrics = compute_metrics(gold_df, ml_df)

        assert results["PERSON"]["precision"] == 1.0
        assert results["PERSON"]["recall"] == 1.0
        assert results["PERSON"]["f1"] == 1.0
        assert results["LOCATION"]["precision"] == 1.0
        assert results["LOCATION"]["recall"] == 1.0
        assert results["LOCATION"]["f1"] == 1.0

    def test_partial_overlap_above_threshold(self):
        """Test spans with overlap above threshold."""
        _results, metrics, _doc_metrics, _span_metrics = compute_metrics(
            self.gold_df, self.ml_df, overlap_threshold=0.6
        )

        # First PERSON span: gold (10,20), ml (12,18) - overlap 6/10 = 0.6 >= 0.6 -> TP
        # LOCATION span: perfect match -> TP
        # Second PERSON span: perfect match -> TP
        # ORGANIZATION span: gold (70,80), ml (75,85) - overlap 5/10 = 0.5 < 0.6 -> FP for ML, FN for gold
        # Third PERSON span in doc3: no ML prediction -> FN

        assert metrics["PERSON"]["tp"] == 2  # doc1 and doc2
        assert metrics["PERSON"]["fp"] == 0
        assert metrics["PERSON"]["fn"] == 1  # doc3 only

        assert metrics["LOCATION"]["tp"] == 1
        assert metrics["LOCATION"]["fp"] == 0
        assert metrics["LOCATION"]["fn"] == 0

        assert metrics["ORGANIZATION"]["tp"] == 0
        assert metrics["ORGANIZATION"]["fp"] == 1  # ML prediction didn't match
        assert metrics["ORGANIZATION"]["fn"] == 1  # Gold standard wasn't matched

    def test_duplicate_overlapping_predictions_single_gold(self):
        """Two ML spans overlapping the same gold span should yield one TP and no FP."""
        gold_df = pd.DataFrame({"note_id": ["docX"], "span_start": [100], "span_end": [110], "span_tag": ["PERSON"]})
        # Both predictions overlap fully with the gold span
        ml_df = pd.DataFrame(
            {
                "note_id": ["docX", "docX"],
                "span_start": [100, 101],
                "span_end": [110, 109],
                "span_tag": ["PERSON", "PERSON"],
            }
        )
        _results, metrics, _doc_metrics, span_metrics = compute_metrics(gold_df, ml_df, overlap_threshold=0.8)
        assert metrics["PERSON"]["tp"] == 1
        assert metrics["PERSON"]["fp"] == 0
        assert metrics["PERSON"]["fn"] == 0
        # Ensure only one TP span metric recorded (duplicate ignored)
        tp_records = [m for m in span_metrics if m["metric"] == "tp"]
        assert len(tp_records) == 1

    def test_note_id_mismatch_gold_only(self):
        """Test when gold has note_ids not in ML (should be FN)."""
        # ML data only has doc1 and doc2, gold has doc1, doc2, doc3
        _results, metrics, doc_metrics, _span_metrics = compute_metrics(self.gold_df, self.ml_df)

        # doc3 exists only in gold, so its span should be FN
        assert metrics["PERSON"]["fn"] >= 1  # At least one from doc3

        # Check that doc3 is in doc_metrics
        assert "doc3" in doc_metrics
        assert doc_metrics["doc3"]["fn"] > 0
        assert doc_metrics["doc3"]["tp"] == 0
        assert doc_metrics["doc3"]["fp"] == 0

    def test_note_id_mismatch_ml_only(self):
        """Test when ML has note_ids not in gold (should be ignored)."""
        # Add ML predictions for a document not in gold
        ml_df_extended = pd.concat(
            [
                self.ml_df,
                pd.DataFrame(
                    {
                        "note_id": ["doc4", "doc4"],
                        "span_start": [100, 120],
                        "span_end": [110, 130],
                        "span_tag": ["PERSON", "LOCATION"],
                    }
                ),
            ],
            ignore_index=True,
        )

        _results, _metrics, doc_metrics, _span_metrics = compute_metrics(self.gold_df, ml_df_extended)

        # doc4 should not appear in doc_metrics since it's not in gold
        assert "doc4" not in doc_metrics

        # The extra ML predictions should not affect the metrics
        # (they should be completely ignored)

    def test_different_tags(self):
        """Test when ML predictions have different tags than gold."""
        ml_df_wrong_tags = pd.DataFrame(
            {
                "note_id": ["doc1", "doc1"],
                "span_start": [10, 30],
                "span_end": [20, 40],
                "span_tag": ["ORGANIZATION", "PERSON"],  # Wrong tags
            }
        )

        _results, metrics, _doc_metrics, _span_metrics = compute_metrics(self.gold_df, ml_df_wrong_tags)

        # All ML predictions should be FP since tags don't match
        # All gold spans should be FN since they weren't matched
        assert metrics["PERSON"]["tp"] == 0
        assert metrics["PERSON"]["fp"] == 1  # ML predicts PERSON at (30,40) but gold has LOCATION there
        assert metrics["PERSON"]["fn"] == 3  # All 3 PERSON spans in gold are unmatched
        assert metrics["LOCATION"]["tp"] == 0
        assert metrics["LOCATION"]["fp"] == 0  # No ML predictions tagged as LOCATION
        assert metrics["LOCATION"]["fn"] == 1  # 1 LOCATION span in gold is unmatched

    def test_label_maps(self):
        """Test with label mapping."""
        label_maps = {"PERSON": ["PER", "INDIVIDUAL"], "LOCATION": ["LOC", "PLACE"]}

        ml_df_mapped = pd.DataFrame(
            {"note_id": ["doc1", "doc1"], "span_start": [10, 30], "span_end": [20, 40], "span_tag": ["PER", "LOC"]}
        )

        _results, metrics, _doc_metrics, _span_metrics = compute_metrics(
            self.gold_df, ml_df_mapped, label_maps=label_maps
        )

        # Should match after mapping
        assert metrics["PERSON"]["tp"] >= 1
        assert metrics["LOCATION"]["tp"] >= 1

    def test_empty_dataframes(self):
        """Test with empty dataframes."""
        empty_gold = pd.DataFrame(columns=["note_id", "span_start", "span_end", "span_tag"])
        empty_ml = pd.DataFrame(columns=["note_id", "span_start", "span_end", "span_tag"])

        results, metrics, doc_metrics, span_metrics = compute_metrics(empty_gold, empty_ml)

        assert len(results) == 0
        assert len(metrics) == 0
        assert len(doc_metrics) == 0
        assert len(span_metrics) == 0

    def test_threshold_sensitivity(self):
        """Test that overlap threshold affects results correctly."""
        # Test with high threshold (0.9)
        _results_high, metrics_high, _, _ = compute_metrics(self.gold_df, self.ml_df, overlap_threshold=0.9)

        # Test with low threshold (0.1)
        _results_low, metrics_low, _, _ = compute_metrics(self.gold_df, self.ml_df, overlap_threshold=0.1)

        # With lower threshold, should have more TPs and fewer FPs/FNs
        assert sum(metrics_low[tag]["tp"] for tag in metrics_low) >= sum(
            metrics_high[tag]["tp"] for tag in metrics_high
        )


class TestFormatResults:
    """Test cases for the format_results function."""

    def test_format_results(self):
        """Test formatting results into DataFrame."""
        results = {
            "PERSON": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "total_spans": 10},
            "LOCATION": {"precision": 0.9, "recall": 0.85, "f1": 0.875, "total_spans": 5},
        }

        df = format_results(results)

        assert len(df) == 2
        assert "Tag" in df.columns
        assert "Precision" in df.columns
        assert "Recall" in df.columns
        assert "F1" in df.columns
        assert "Total Spans" in df.columns

        # Check that precision values are formatted correctly
        person_row = df[df["Tag"] == "PERSON"].iloc[0]
        assert person_row["Precision"] == "0.8000"
        assert person_row["Total Spans"] == 10


class TestAggregateResults:
    """Test cases for the aggregate_results function."""

    def test_aggregate_results(self):
        """Test macro averaging of results."""
        results = {
            "PERSON": {"precision": 0.8, "recall": 0.6, "f1": 0.7},
            "LOCATION": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
        }

        aggregated = aggregate_results(results)

        assert abs(aggregated["macro_precision"] - 0.85) < 0.001  # (0.8 + 0.9) / 2
        assert abs(aggregated["macro_recall"] - 0.7) < 0.001  # (0.6 + 0.8) / 2
        assert abs(aggregated["macro_f1"] - 0.775) < 0.001  # (0.7 + 0.85) / 2

    def test_empty_results(self):
        """Test with empty results dictionary."""
        results = {}
        aggregated = aggregate_results(results)

        assert aggregated["macro_precision"] == 0
        assert aggregated["macro_recall"] == 0
        assert aggregated["macro_f1"] == 0


class TestSpansToDataframe:
    """Test cases for the spans_to_dataframe function."""

    def test_spans_to_dataframe(self):
        """Test converting span metrics to DataFrame."""
        span_metrics = [
            {"note_id": "doc1", "span_start": 10, "span_end": 20, "span_tag": "PERSON", "metric": "tp"},
            {"note_id": "doc1", "span_start": 30, "span_end": 40, "span_tag": "LOCATION", "metric": "fp"},
        ]

        df = spans_to_dataframe(span_metrics)

        assert len(df) == 2
        assert list(df.columns) == ["note_id", "span_start", "span_end", "span_tag", "metric"]
        assert df.iloc[0]["metric"] == "tp"
        assert df.iloc[1]["metric"] == "fp"


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    def test_realistic_scenario(self):
        """Test a realistic scenario with multiple documents and entity types."""
        # Create a realistic gold standard dataset
        gold_df = pd.DataFrame(
            {
                "note_id": ["patient_001"] * 4 + ["patient_002"] * 3 + ["patient_003"] * 2,
                "span_start": [15, 45, 78, 120, 25, 67, 95, 33, 88],
                "span_end": [25, 55, 88, 135, 40, 82, 110, 48, 103],
                "span_tag": [
                    "PERSON",
                    "MEDICATION",
                    "CONDITION",
                    "PERSON",
                    "PERSON",
                    "CONDITION",
                    "MEDICATION",
                    "PERSON",
                    "CONDITION",
                ],
            }
        )

        # Create ML predictions with some correct, some incorrect
        ml_df = pd.DataFrame(
            {
                "note_id": ["patient_001"] * 4 + ["patient_002"] * 2 + ["patient_003"] * 3,
                "span_start": [16, 45, 80, 122, 25, 95, 33, 90, 200],
                "span_end": [24, 55, 87, 133, 40, 110, 48, 102, 210],
                "span_tag": [
                    "PERSON",
                    "MEDICATION",
                    "CONDITION",
                    "PERSON",
                    "PERSON",
                    "MEDICATION",
                    "PERSON",
                    "CONDITION",
                    "PERSON",
                ],
            }
        )

        results, _metrics, doc_metrics, _span_metrics = compute_metrics(gold_df, ml_df, overlap_threshold=0.7)

        # Verify that all entity types are represented
        assert "PERSON" in results
        assert "MEDICATION" in results
        assert "CONDITION" in results

        # Verify that metrics make sense
        for entity_type in results:
            assert 0 <= results[entity_type]["precision"] <= 1
            assert 0 <= results[entity_type]["recall"] <= 1
            assert 0 <= results[entity_type]["f1"] <= 1

        # Verify document-level metrics
        assert len(doc_metrics) == 3  # Three patients
        for doc_id in ["patient_001", "patient_002", "patient_003"]:
            assert doc_id in doc_metrics
            total_metrics = doc_metrics[doc_id]["tp"] + doc_metrics[doc_id]["fp"] + doc_metrics[doc_id]["fn"]
            assert total_metrics > 0  # Should have some metrics for each doc

    def test_edge_case_single_character_spans(self):
        """Test with single-character spans."""
        gold_df = pd.DataFrame(
            {
                "note_id": ["doc1", "doc1"],
                "span_start": [10, 15],
                "span_end": [11, 16],
                "span_tag": ["PERSON", "LOCATION"],
            }
        )

        ml_df = pd.DataFrame(
            {
                "note_id": ["doc1", "doc1"],
                "span_start": [10, 14],
                "span_end": [11, 17],
                "span_tag": ["PERSON", "LOCATION"],
            }
        )

        _results, metrics, _doc_metrics, _span_metrics = compute_metrics(gold_df, ml_df)

        # First span should match exactly
        assert metrics["PERSON"]["tp"] == 1

        # Second span should have some overlap but may not meet threshold
        assert "LOCATION" in metrics


def test_span_overlap_zero_length_span():
    """A span with zero length (end == start) should yield 0 overlap safely."""
    assert span_overlap((10, 10), (10, 20)) == 0
    assert span_overlap((5, 5), (1, 100)) == 0


def test_span_overlap_inverted_or_empty():
    """Defensive check: if span has end <= start, treat as 0 length (overlap 0)."""
    # Although data pipeline should prevent this, ensure function is stable.
    assert span_overlap((20, 10), (10, 30)) == 0  # inverted
    assert span_overlap((15, 15), (15, 15)) == 0  # both zero-length


def test_ignore_predictions_from_unknown_note_ids():
    """Predictions for note_ids absent from gold are ignored (design choice)."""
    gold = pd.DataFrame({"note_id": ["doc1"], "span_start": [0], "span_end": [5], "span_tag": ["PERSON"]})
    ml = pd.DataFrame(
        {
            "note_id": ["doc1", "docX"],  # docX not in gold
            "span_start": [0, 10],
            "span_end": [5, 15],
            "span_tag": ["PERSON", "PERSON"],
        }
    )
    _results, metrics, doc_metrics, _span_metrics = compute_metrics(gold, ml)

    # Ensure docX did not inflate FP counts.
    assert metrics["PERSON"]["fp"] == 0, "Unknown note_id prediction should be ignored per current design"
    assert "docX" not in doc_metrics


def test_prediction_with_tag_not_in_gold_is_ignored():
    """Predicted tags absent from gold label set are ignored (not counted as FP)."""
    gold = pd.DataFrame({"note_id": ["doc1"], "span_start": [0], "span_end": [4], "span_tag": ["PERSON"]})
    ml = pd.DataFrame(
        {
            "note_id": ["doc1"],
            "span_start": [0],
            "span_end": [4],
            "span_tag": ["UNKNOWN"],  # tag not present among gold spans
        }
    )
    _results, metrics, _doc_metrics, _span_metrics = compute_metrics(gold, ml)

    # No TP or FP should be recorded because UNKNOWN tag is discarded early
    assert metrics["PERSON"]["tp"] == 0
    assert metrics["PERSON"]["fp"] == 0
    assert metrics["PERSON"]["fn"] == 1  # gold span remains unmatched


def test_label_maps_one_to_many_explosion_ignored_extra():
    """One predicted label mapping to multiple canonical tags only counts those in gold.

    If a mapped canonical tag doesn't exist in gold, its exploded prediction row is ignored.
    """
    gold = pd.DataFrame({"note_id": ["doc1"], "span_start": [0], "span_end": [6], "span_tag": ["PERSON"]})
    # Prediction uses HUM which maps to both PERSON and ALIAS, but ALIAS not in gold
    ml = pd.DataFrame({"note_id": ["doc1"], "span_start": [0], "span_end": [6], "span_tag": ["HUM"]})
    label_maps = {
        "PERSON": ["HUM"],
        "ALIAS": ["HUM"],  # extra canonical that isn't in gold
    }
    results, metrics, _doc_metrics, _span_metrics = compute_metrics(gold, ml, label_maps=label_maps)

    # PERSON should get the TP; ALIAS should be absent entirely
    assert metrics["PERSON"]["tp"] == 1
    assert "ALIAS" not in metrics
    assert "ALIAS" not in results


class TestResolveDataframeConflicts:
    """Test cases for the resolve_dataframe_conflicts function."""

    def test_basic_overlap_resolution(self):
        """Test basic overlapping spans are resolved correctly."""
        df = pd.DataFrame(
            {
                "note_id": ["note1", "note1", "note2"],
                "span_start": [10, 15, 5],
                "span_end": [25, 20, 15],
                "span_tag": ["PERSON", "PERSON", "LOCATION"],
            }
        )
        result = resolve_dataframe_conflicts(df, "TestModel")

        # First two spans overlap, longest should be kept
        assert len(result) == 2
        assert "model_name" in result.columns
        assert all(result["model_name"] == "TestModel")

        # Check note1 kept the longer span (10-25)
        note1_spans = result[result["note_id"] == "note1"]
        assert len(note1_spans) == 1
        assert note1_spans.iloc[0]["span_start"] == 10
        assert note1_spans.iloc[0]["span_end"] == 25

    def test_no_overlaps(self):
        """Test non-overlapping spans are all kept."""
        df = pd.DataFrame(
            {
                "note_id": ["note1", "note1", "note1"],
                "span_start": [0, 10, 20],
                "span_end": [5, 15, 25],
                "span_tag": ["PERSON", "LOCATION", "DATE"],
            }
        )
        result = resolve_dataframe_conflicts(df, "TestModel")

        assert len(result) == 3
        assert all(result["model_name"] == "TestModel")

    def test_multiple_notes(self):
        """Test resolution works independently per note."""
        df = pd.DataFrame(
            {
                "note_id": ["note1", "note1", "note2", "note2"],
                "span_start": [10, 15, 10, 15],
                "span_end": [25, 20, 25, 20],
                "span_tag": ["PERSON", "PERSON", "LOCATION", "LOCATION"],
            }
        )
        result = resolve_dataframe_conflicts(df, "TestModel")

        # Each note has overlapping spans, should have 2 total (1 per note)
        assert len(result) == 2
        assert len(result[result["note_id"] == "note1"]) == 1
        assert len(result[result["note_id"] == "note2"]) == 1

    def test_empty_dataframe(self):
        """Test empty DataFrame returns empty result with model_name column."""
        df = pd.DataFrame(columns=["note_id", "span_start", "span_end", "span_tag"])
        result = resolve_dataframe_conflicts(df, "TestModel")

        assert len(result) == 0
        assert "model_name" in result.columns

    def test_preserves_additional_columns(self):
        """Test that additional columns are preserved."""
        df = pd.DataFrame(
            {
                "note_id": ["note1"],
                "span_start": [10],
                "span_end": [20],
                "span_tag": ["PERSON"],
                "confidence": [0.95],
                "source": ["llm"],
            }
        )
        result = resolve_dataframe_conflicts(df, "TestModel")

        assert "confidence" in result.columns
        assert "source" in result.columns
        assert result.iloc[0]["confidence"] == 0.95
        assert result.iloc[0]["source"] == "llm"

    def test_missing_required_columns(self):
        """Test error is raised for missing required columns."""
        df = pd.DataFrame(
            {
                "note_id": ["note1"],
                "span_start": [10],
            }
        )
        with pytest.raises(ValueError, match="Missing required column 'span_end'"):
            resolve_dataframe_conflicts(df, "TestModel")


class TestGenerateModelCombinationsWithResolution:
    """Test cases for the generate_model_combinations_with_resolution function."""

    def test_two_models_combination(self):
        """Test combination of two models with conflict resolution."""
        df = pd.DataFrame(
            {
                "model_name": ["M1", "M1", "M2", "M2"],
                "note_id": ["n1", "n1", "n1", "n1"],
                "span_start": [0, 10, 5, 15],
                "span_end": [8, 15, 12, 20],
                "span_tag": ["PER", "LOC", "PER", "LOC"],
            }
        )
        result = generate_model_combinations_with_resolution(df)

        # Should have 3 combinations: M1, M2, M1+M2 (includes individual models)
        assert "M1+M2" in result["model_name"].values
        assert "M1" in result["model_name"].values
        assert "M2" in result["model_name"].values
        unique_combos = result["model_name"].unique()
        assert len(unique_combos) == 3

    def test_three_models_combinations(self):
        """Test all combinations of three models."""
        df = pd.DataFrame(
            {
                "model_name": ["M1", "M2", "M3"] * 2,
                "note_id": ["n1"] * 6,
                "span_start": [0, 5, 10, 20, 25, 30],
                "span_end": [4, 9, 14, 24, 29, 34],
                "span_tag": ["PER"] * 6,
            }
        )
        result = generate_model_combinations_with_resolution(df)

        # Should have 3 individual + 3 pairs + 1 triple = 7 combinations
        # (3 choose 1) + (3 choose 2) + (3 choose 3) = 3 + 3 + 1 = 7
        unique_combos = result["model_name"].unique()
        assert len(unique_combos) == 7
        assert "M1" in unique_combos
        assert "M2" in unique_combos
        assert "M3" in unique_combos
        assert "M1+M2" in unique_combos
        assert "M1+M3" in unique_combos
        assert "M2+M3" in unique_combos
        assert "M1+M2+M3" in unique_combos

    def test_conflict_resolution_applied(self):
        """Test that conflict resolution removes overlapping spans."""
        df = pd.DataFrame(
            {
                "model_name": ["M1", "M1", "M2", "M2"],
                "note_id": ["n1", "n1", "n1", "n1"],
                "span_start": [0, 5, 0, 10],  # First two overlap
                "span_end": [10, 8, 10, 15],
                "span_tag": ["PER", "PER", "PER", "LOC"],
            }
        )
        result = generate_model_combinations_with_resolution(df)

        # Filter for M1+M2 combination
        combo_df = result[result["model_name"] == "M1+M2"]

        # Should have 2 spans after resolution (overlapping PER spans reduced to 1, plus LOC)
        assert len(combo_df) == 2

    def test_multiple_notes(self):
        """Test combinations work correctly across multiple notes."""
        df = pd.DataFrame(
            {
                "model_name": ["M1", "M1", "M2", "M2"],
                "note_id": ["n1", "n2", "n1", "n2"],
                "span_start": [0, 0, 5, 5],
                "span_end": [10, 10, 15, 15],
                "span_tag": ["PER", "LOC", "PER", "LOC"],
            }
        )
        result = generate_model_combinations_with_resolution(df)

        # Should have M1+M2 combination
        combo_df = result[result["model_name"] == "M1+M2"]

        # After conflict resolution, overlapping spans per note are resolved
        # n1: M1 PER (0-10) overlaps with M2 PER (5-15), longer span wins -> 1 span
        # n2: M1 LOC (0-10) overlaps with M2 LOC (5-15), longer span wins -> 1 span
        # Total: 2 spans (1 per note)
        assert len(combo_df) == 2
        assert len(combo_df[combo_df["note_id"] == "n1"]) == 1
        assert len(combo_df[combo_df["note_id"] == "n2"]) == 1

    def test_empty_dataframe(self):
        """Test empty DataFrame returns empty result."""
        df = pd.DataFrame(columns=["model_name", "note_id", "span_start", "span_end", "span_tag"])
        result = generate_model_combinations_with_resolution(df)

        assert len(result) == 0

    def test_single_model(self):
        """Test single model returns its own resolved spans."""
        df = pd.DataFrame(
            {
                "model_name": ["M1", "M1"],
                "note_id": ["n1", "n1"],
                "span_start": [0, 10],
                "span_end": [5, 15],
                "span_tag": ["PER", "LOC"],
            }
        )
        result = generate_model_combinations_with_resolution(df)

        # Single model returns its own spans (no combinations, just the model itself)
        assert len(result) == 2
        assert all(result["model_name"] == "M1")

    def test_custom_model_column(self):
        """Test using a custom model column name."""
        df = pd.DataFrame(
            {
                "system": ["S1", "S1", "S2", "S2"],
                "note_id": ["n1", "n1", "n1", "n1"],
                "span_start": [0, 10, 5, 15],
                "span_end": [8, 15, 12, 20],
                "span_tag": ["PER", "LOC", "PER", "LOC"],
            }
        )
        result = generate_model_combinations_with_resolution(df, model_column="system")

        # Should have model_name column with combination name
        assert "model_name" in result.columns
        assert "S1+S2" in result["model_name"].values

    def test_missing_required_columns(self):
        """Test error is raised for missing required columns."""
        df = pd.DataFrame({"model_name": ["M1"], "note_id": ["n1"], "span_start": [0]})
        with pytest.raises(ValueError, match="Missing required column 'span_end'"):
            generate_model_combinations_with_resolution(df)

    def test_preserves_additional_columns(self):
        """Test that additional columns are preserved in combinations."""
        df = pd.DataFrame(
            {
                "model_name": ["M1", "M2"],
                "note_id": ["n1", "n1"],
                "span_start": [0, 10],
                "span_end": [5, 15],
                "span_tag": ["PER", "LOC"],
                "confidence": [0.9, 0.8],
                "source": ["llm", "rule"],
            }
        )
        result = generate_model_combinations_with_resolution(df)

        assert "confidence" in result.columns
        assert "source" in result.columns


class TestResolveConflicts:
    """Test cases for the resolve_conflicts function."""

    def test_empty_input(self):
        """Test with empty list returns empty list."""
        assert resolve_conflicts([]) == []

    def test_single_span(self):
        """Test single span is returned unchanged."""
        spans = [{"start": 10, "end": 20, "entity_type": "PERSON"}]
        result = resolve_conflicts(spans)
        assert len(result) == 1
        assert result[0]["start"] == 10
        assert result[0]["end"] == 20

    def test_no_overlaps(self):
        """Test non-overlapping spans are all kept."""
        spans = [
            {"start": 0, "end": 10, "entity_type": "PERSON"},
            {"start": 20, "end": 30, "entity_type": "DATE"},
            {"start": 40, "end": 50, "entity_type": "LOCATION"},
        ]
        result = resolve_conflicts(spans)
        assert len(result) == 3

    def test_longest_wins_same_type(self):
        """Test longer span wins when overlapping (same entity type)."""
        spans = [
            {"start": 10, "end": 30, "entity_type": "PERSON"},  # length 20
            {"start": 15, "end": 25, "entity_type": "PERSON"},  # length 10, overlaps
        ]
        result = resolve_conflicts(spans, strategy="longest_wins")
        assert len(result) == 1
        assert result[0]["start"] == 10
        assert result[0]["end"] == 30

    def test_longest_wins_cross_type(self):
        """Test longer span wins when overlapping (different entity types)."""
        spans = [
            {"start": 10, "end": 30, "entity_type": "PERSON"},  # length 20
            {"start": 15, "end": 25, "entity_type": "NAME"},  # length 10, overlaps
        ]
        result = resolve_conflicts(spans, strategy="longest_wins")
        assert len(result) == 1
        assert result[0]["entity_type"] == "PERSON"

    def test_merge_contained_only(self):
        """Test merge_contained only removes fully contained spans."""
        spans = [
            {"start": 10, "end": 30, "entity_type": "PERSON"},  # contains next
            {"start": 15, "end": 25, "entity_type": "NAME"},  # fully contained
        ]
        result = resolve_conflicts(spans, strategy="merge_contained")
        assert len(result) == 1
        assert result[0]["start"] == 10

    def test_merge_contained_partial_overlap_kept(self):
        """Test merge_contained keeps partially overlapping spans."""
        spans = [
            {"start": 10, "end": 25, "entity_type": "PERSON"},
            {"start": 20, "end": 35, "entity_type": "NAME"},  # partial overlap, not contained
        ]
        result = resolve_conflicts(spans, strategy="merge_contained")
        assert len(result) == 2

    def test_exact_duplicates_removed(self):
        """Test exact duplicates are removed, keeping higher score."""
        spans = [
            {"start": 10, "end": 20, "entity_type": "PERSON", "score": 0.8},
            {"start": 10, "end": 20, "entity_type": "PERSON", "score": 0.9},
        ]
        result = resolve_conflicts(spans)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_preserves_metadata(self):
        """Test that additional metadata is preserved."""
        spans = [
            {"start": 10, "end": 20, "entity_type": "PERSON", "source": "llm", "confidence": 0.95},
        ]
        result = resolve_conflicts(spans)
        assert len(result) == 1
        assert result[0]["source"] == "llm"
        assert result[0]["confidence"] == 0.95

    def test_multiple_overlaps_longest_wins(self):
        """Test when one span overlaps multiple others, longest wins."""
        spans = [
            {"start": 0, "end": 50, "entity_type": "PERSON"},  # longest, overlaps all
            {"start": 5, "end": 15, "entity_type": "NAME"},
            {"start": 20, "end": 30, "entity_type": "NAME"},
            {"start": 35, "end": 45, "entity_type": "NAME"},
        ]
        result = resolve_conflicts(spans, strategy="longest_wins")
        assert len(result) == 1
        assert result[0]["start"] == 0
        assert result[0]["end"] == 50

    def test_chain_of_overlaps(self):
        """Test chain of overlapping spans."""
        spans = [
            {"start": 0, "end": 20, "entity_type": "A"},  # overlaps B
            {"start": 15, "end": 35, "entity_type": "B"},  # overlaps A and C
            {"start": 30, "end": 50, "entity_type": "C"},  # overlaps B
        ]
        result = resolve_conflicts(spans, strategy="longest_wins")
        # All have same length (20), so ordering matters
        # With longest_wins, the first processed that doesn't overlap kept spans stays
        assert len(result) >= 1

    def test_sorted_output(self):
        """Test output is sorted by start position."""
        spans = [
            {"start": 30, "end": 40, "entity_type": "C"},
            {"start": 10, "end": 20, "entity_type": "A"},
            {"start": 50, "end": 60, "entity_type": "D"},
        ]
        result = resolve_conflicts(spans)
        starts = [s["start"] for s in result]
        assert starts == sorted(starts)


class TestResolveRecognizerResults:
    """Test cases for the resolve_recognizer_results function."""

    def test_empty_input(self):
        """Test with empty list returns empty list."""
        assert resolve_recognizer_results([]) == []

    def test_filters_zero_score(self):
        """Test that zero-score results are filtered out."""
        from presidio_anonymizer.entities import RecognizerResult

        results = [
            RecognizerResult(entity_type="PERSON", start=0, end=10, score=0.9),
            RecognizerResult(entity_type="NAME", start=5, end=15, score=0.0),  # zero score
        ]
        resolved = resolve_recognizer_results(results)
        assert len(resolved) == 1
        assert resolved[0].entity_type == "PERSON"

    def test_longest_wins_strategy(self):
        """Test longest_wins strategy keeps longer span."""
        from presidio_anonymizer.entities import RecognizerResult

        results = [
            RecognizerResult(entity_type="PERSON", start=0, end=20, score=0.8),  # length 20
            RecognizerResult(entity_type="NAME", start=5, end=15, score=0.9),  # length 10, contained
        ]
        resolved = resolve_recognizer_results(results, strategy="longest_wins")
        assert len(resolved) == 1
        assert resolved[0].entity_type == "PERSON"

    def test_merge_contained_strategy(self):
        """Test merge_contained only removes fully contained spans."""
        from presidio_anonymizer.entities import RecognizerResult

        results = [
            RecognizerResult(entity_type="PERSON", start=0, end=20, score=0.8),
            RecognizerResult(entity_type="NAME", start=5, end=15, score=0.9),  # contained
        ]
        resolved = resolve_recognizer_results(results, strategy="merge_contained")
        assert len(resolved) == 1
        assert resolved[0].entity_type == "PERSON"

    def test_partial_overlap_merge_contained(self):
        """Test merge_contained keeps partial overlaps."""
        from presidio_anonymizer.entities import RecognizerResult

        results = [
            RecognizerResult(entity_type="PERSON", start=0, end=15, score=0.8),
            RecognizerResult(entity_type="NAME", start=10, end=25, score=0.9),  # partial overlap
        ]
        resolved = resolve_recognizer_results(results, strategy="merge_contained")
        assert len(resolved) == 2

    def test_exact_duplicates_highest_score(self):
        """Test exact duplicates keep highest score."""
        from presidio_anonymizer.entities import RecognizerResult

        results = [
            RecognizerResult(entity_type="PERSON", start=0, end=10, score=0.7),
            RecognizerResult(entity_type="PERSON", start=0, end=10, score=0.9),
        ]
        resolved = resolve_recognizer_results(results)
        assert len(resolved) == 1
        assert resolved[0].score == 0.9

    def test_returns_recognizer_result_objects(self):
        """Test that output contains RecognizerResult objects."""
        from presidio_anonymizer.entities import RecognizerResult

        results = [
            RecognizerResult(entity_type="PERSON", start=0, end=10, score=0.9),
        ]
        resolved = resolve_recognizer_results(results)
        assert len(resolved) == 1
        assert isinstance(resolved[0], RecognizerResult)

    def test_many_entities_performance(self):
        """Test that large number of entities is handled efficiently."""
        import time

        from presidio_anonymizer.entities import RecognizerResult

        # Create 1000 non-overlapping spans
        results = [
            RecognizerResult(entity_type="PERSON", start=i * 20, end=i * 20 + 10, score=0.9) for i in range(1000)
        ]

        start_time = time.time()
        resolved = resolve_recognizer_results(results)
        elapsed = time.time() - start_time

        assert len(resolved) == 1000
        # Should complete in well under 1 second for O(n log n)
        assert elapsed < 1.0


class TestMergeAdjacentSpans:
    """Test merge_adjacent_types parameter of resolve_recognizer_results."""

    def test_adjacent_date_spans_merged(self):
        """Adjacent DATE spans separated by whitespace are merged."""
        from presidio_anonymizer.entities import RecognizerResult

        text = "Date: January 15, 2024 end"
        results = [
            RecognizerResult(entity_type="DATE", start=6, end=13, score=0.9),  # "January"
            RecognizerResult(entity_type="DATE", start=14, end=22, score=0.85),  # "15, 2024"
        ]
        resolved = resolve_recognizer_results(
            results,
            merge_adjacent_types={"DATE", "DATE_TIME"},
            text=text,
        )
        assert len(resolved) == 1
        assert resolved[0].start == 6
        assert resolved[0].end == 22
        assert resolved[0].score == 0.9

    def test_different_types_not_merged(self):
        """Adjacent spans of different entity types are not merged."""
        from presidio_anonymizer.entities import RecognizerResult

        text = "John Smith visited"
        results = [
            RecognizerResult(entity_type="DATE", start=0, end=4, score=0.9),
            RecognizerResult(entity_type="PERSON", start=5, end=10, score=0.9),
        ]
        resolved = resolve_recognizer_results(
            results,
            merge_adjacent_types={"DATE", "PERSON"},
            text=text,
        )
        assert len(resolved) == 2

    def test_gap_too_large(self):
        """Spans separated by more than max_merge_gap are not merged."""
        from presidio_anonymizer.entities import RecognizerResult

        text = "Date: Jan    15 end"
        results = [
            RecognizerResult(entity_type="DATE", start=6, end=9, score=0.9),  # "Jan"
            RecognizerResult(entity_type="DATE", start=13, end=15, score=0.9),  # "15"
        ]
        resolved = resolve_recognizer_results(
            results,
            merge_adjacent_types={"DATE"},
            text=text,
            max_merge_gap=2,
        )
        assert len(resolved) == 2

    def test_non_whitespace_gap_not_merged(self):
        """Spans separated by non-whitespace characters are not merged."""
        from presidio_anonymizer.entities import RecognizerResult

        text = "Date: Jan,15 end"
        results = [
            RecognizerResult(entity_type="DATE", start=6, end=9, score=0.9),  # "Jan"
            RecognizerResult(entity_type="DATE", start=10, end=12, score=0.9),  # "15"
        ]
        resolved = resolve_recognizer_results(
            results,
            merge_adjacent_types={"DATE"},
            text=text,
        )
        assert len(resolved) == 2

    def test_chain_merge(self):
        """Three consecutive same-type spans merge into one."""
        from presidio_anonymizer.entities import RecognizerResult

        text = "January 15, 2024"
        results = [
            RecognizerResult(entity_type="DATE", start=0, end=7, score=0.9),  # "January"
            RecognizerResult(entity_type="DATE", start=8, end=11, score=0.8),  # "15,"
            RecognizerResult(entity_type="DATE", start=12, end=16, score=0.85),  # "2024"
        ]
        resolved = resolve_recognizer_results(
            results,
            merge_adjacent_types={"DATE"},
            text=text,
        )
        assert len(resolved) == 1
        assert resolved[0].start == 0
        assert resolved[0].end == 16
        assert resolved[0].score == 0.9

    def test_mixed_mergeable_and_non_mergeable(self):
        """Only specified entity types are merged; others remain untouched."""
        from presidio_anonymizer.entities import RecognizerResult

        text = "John saw Jan 15 at clinic"
        results = [
            RecognizerResult(entity_type="PERSON", start=0, end=4, score=0.9),
            RecognizerResult(entity_type="DATE", start=9, end=12, score=0.9),  # "Jan"
            RecognizerResult(entity_type="DATE", start=13, end=15, score=0.85),  # "15"
            RecognizerResult(entity_type="LOCATION", start=19, end=25, score=0.8),
        ]
        resolved = resolve_recognizer_results(
            results,
            merge_adjacent_types={"DATE"},
            text=text,
        )
        # PERSON + merged DATE + LOCATION = 3
        assert len(resolved) == 3
        dates = [r for r in resolved if r.entity_type == "DATE"]
        assert len(dates) == 1
        assert dates[0].start == 9
        assert dates[0].end == 15

    def test_no_merge_without_param(self):
        """Without merge_adjacent_types, adjacent same-type spans are NOT merged."""
        from presidio_anonymizer.entities import RecognizerResult

        results = [
            RecognizerResult(entity_type="DATE", start=0, end=7, score=0.9),
            RecognizerResult(entity_type="DATE", start=8, end=16, score=0.85),
        ]
        resolved = resolve_recognizer_results(results)
        assert len(resolved) == 2


class TestResolveOverlappingSpansBackwardsCompat:
    """Test that resolve_overlapping_spans maintains backwards compatibility."""

    def test_basic_overlap_resolution(self):
        """Test basic overlapping spans are resolved correctly."""
        spans = [
            {"start": 10, "end": 25, "text": "longer span"},
            {"start": 15, "end": 20, "text": "short"},
        ]
        result = resolve_overlapping_spans(spans)
        assert len(result) == 1
        assert result[0]["text"] == "longer span"

    def test_no_overlaps_all_kept(self):
        """Test non-overlapping spans are all kept."""
        spans = [
            {"start": 0, "end": 10, "text": "first"},
            {"start": 20, "end": 30, "text": "second"},
        ]
        result = resolve_overlapping_spans(spans)
        assert len(result) == 2

    def test_empty_list(self):
        """Test empty list returns empty."""
        assert resolve_overlapping_spans([]) == []


if __name__ == "__main__":
    pytest.main([__file__])
