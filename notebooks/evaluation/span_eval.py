"""Span-level PHI evaluation functions.

Computes precision, recall, and F1 at the span level by comparing
gold standard annotations against model predictions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tide2.utils.span_metrics import aggregate_results
from tide2.utils.span_metrics import compute_metrics
from tide2.utils.span_metrics import format_results
from tide2.utils.span_metrics import resolve_dataframe_conflicts


def run_span_evaluation(
    df_gold: pd.DataFrame,
    df_ml: pd.DataFrame,
    ablation_configs: list[tuple[str, list[str]]],
    output_dir: Path,
    overlap_threshold: float = 0.8,
) -> pd.DataFrame:
    """Run span-level evaluation for each ablation configuration.

    Args:
        df_gold: Gold standard spans with columns: note_id, span_start, span_end, span_tag.
        df_ml: Model prediction spans with columns: note_id, span_start, span_end, span_tag, category.
        ablation_configs: List of (config_name, categories_list) tuples.
        output_dir: Directory to save per-config results.
        overlap_threshold: Minimum overlap fraction for span matching.

    Returns:
        DataFrame summarizing micro/macro P/R/F1 per ablation config.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for config_name, categories in ablation_configs:
        df_filtered = df_ml[df_ml["category"].isin(categories)].copy()

        if df_filtered.empty:
            print(f"  WARNING: No spans for config '{config_name}', skipping")
            continue

        df_resolved = resolve_dataframe_conflicts(df_filtered, "pipeline")

        results, metrics_per_label, _doc_metrics, _span_metrics = compute_metrics(
            gold_df=df_gold,
            ml_df=df_resolved,
            overlap_threshold=overlap_threshold,
        )

        agg = aggregate_results(results)
        total_tp = sum(m["tp"] for m in metrics_per_label.values())
        total_fp = sum(m["fp"] for m in metrics_per_label.values())
        total_fn = sum(m["fn"] for m in metrics_per_label.values())

        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        print(f"  {config_name}: Micro P/R/F1 = {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")

        all_results.append(
            {
                "config": config_name,
                "categories": "+".join(categories),
                "num_spans": len(df_resolved),
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "micro_precision": micro_p,
                "micro_recall": micro_r,
                "micro_f1": micro_f1,
                "macro_precision": agg["macro_precision"],
                "macro_recall": agg["macro_recall"],
                "macro_f1": agg["macro_f1"],
            }
        )

        # Save per-label breakdown
        config_dir = output_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        format_results(results).to_csv(config_dir / "per_label_results.csv", index=False)

    ablation_df = pd.DataFrame(all_results)
    if not ablation_df.empty:
        ablation_df.to_csv(output_dir / "ablation_summary.csv", index=False)
    return ablation_df
