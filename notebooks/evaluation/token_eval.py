"""Token-level PHI evaluation functions.

Computes precision, recall, and F1 at the token (character or subword) level
by comparing gold standard annotations against model predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from tide2.utils.span_metrics import resolve_dataframe_conflicts

# ---------------------------------------------------------------------------
# Span → token-label conversion
# ---------------------------------------------------------------------------


def spans_to_dict(df_spans: pd.DataFrame) -> dict[str, list[dict]]:
    """Convert spans DataFrame to dict keyed by text_id."""
    records = df_spans[["text_id", "span_start", "span_end", "span_tag"]].to_dict("records")
    spans_by_text: dict[str, list[dict]] = {}
    for r in records:
        text_id = r.pop("text_id")
        spans_by_text.setdefault(text_id, []).append(r)
    return spans_by_text


def _create_char_labels(text: str, spans: list[dict]) -> list[str]:
    labels = ["O"] * len(text)
    for span in spans:
        start, end, tag = span["span_start"], span["span_end"], span["span_tag"]
        for i in range(start, min(end, len(text))):
            labels[i] = tag
    return labels


def _labels_from_offsets(offset_mapping: list[tuple[int, int]], spans: list[dict]) -> list[str]:
    labels = ["O"] * len(offset_mapping)
    for token_idx, (start_char, end_char) in enumerate(offset_mapping):
        if start_char == end_char:
            continue
        for span in spans:
            if span["span_end"] <= start_char:
                continue
            if span["span_start"] >= end_char:
                break
            if start_char < span["span_end"] and end_char > span["span_start"]:
                labels[token_idx] = span["span_tag"]
                break
    return labels


def precompute_offsets(
    df_texts: pd.DataFrame,
    tokenizer=None,
) -> dict[str, list[tuple[int, int]] | int]:
    """Pre-tokenize all texts and return offset mappings (or text lengths for char-level)."""
    text_ids = df_texts["text_id"].tolist()
    texts = df_texts["text_content"].tolist()

    if tokenizer is not None:
        tokenized = tokenizer(
            texts,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        return dict(zip(text_ids, tokenized["offset_mapping"], strict=False))
    return {tid: len(t) for tid, t in zip(text_ids, texts, strict=False)}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_token_level(
    df_texts: pd.DataFrame,
    gold_spans: dict[str, list[dict]],
    pred_spans: dict[str, list[dict]],
    precomputed_offsets: dict | None = None,
) -> dict:
    """Evaluate predictions at token level and return per-label + aggregate metrics."""
    all_gold_labels: list[str] = []
    all_pred_labels: list[str] = []

    text_ids = df_texts["text_id"].tolist()
    text_contents = df_texts["text_content"].tolist()

    for text_id, text in zip(text_ids, text_contents, strict=False):
        gold = gold_spans.get(text_id, [])
        pred = pred_spans.get(text_id, [])

        offsets = precomputed_offsets.get(text_id) if precomputed_offsets else None

        if offsets is not None and isinstance(offsets, list):
            gold_labels = _labels_from_offsets(offsets, gold)
            pred_labels = _labels_from_offsets(offsets, pred)
        else:
            gold_labels = _create_char_labels(text, gold)
            pred_labels = _create_char_labels(text, pred)

        all_gold_labels.extend(gold_labels)
        all_pred_labels.extend(pred_labels)

    return compute_token_metrics(all_gold_labels, all_pred_labels)


def compute_token_metrics(gold_labels: list[str], pred_labels: list[str]) -> dict:
    """Compute per-label and micro/macro metrics from flat label lists."""
    labels_present = sorted(set(gold_labels) | set(pred_labels))
    entity_labels = [label for label in labels_present if label != "O"]

    precision, recall, f1, support = precision_recall_fscore_support(
        gold_labels,
        pred_labels,
        labels=labels_present,
        average=None,
        zero_division=0,
    )

    metrics: dict = {}
    for i, label in enumerate(labels_present):
        metrics[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    if entity_labels:
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            gold_labels,
            pred_labels,
            labels=entity_labels,
            average="micro",
            zero_division=0,
        )
        metrics["micro_avg"] = {
            "precision": float(micro_p),
            "recall": float(micro_r),
            "f1": float(micro_f1),
            "support": sum(metrics[label]["support"] for label in entity_labels),
        }
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            gold_labels,
            pred_labels,
            labels=entity_labels,
            average="macro",
            zero_division=0,
        )
        metrics["macro_avg"] = {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
            "support": metrics["micro_avg"]["support"],
        }

    return metrics


def _save_token_metrics(metrics: dict, output_path: Path, metadata: dict | None = None) -> None:
    """Save token metrics to JSON."""
    output = {"metrics": metrics, "metadata": metadata or {}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


def run_token_evaluation(
    df_gold: pd.DataFrame,
    df_ml: pd.DataFrame,
    df_texts: pd.DataFrame,
    ablation_configs: list[tuple[str, list[str]]],
    output_dir: Path,
    tokenizer_name: str | None = None,
) -> pd.DataFrame:
    """Run token-level evaluation for each ablation configuration.

    Args:
        df_gold: Gold spans with columns: text_id, span_start, span_end, span_tag.
        df_ml: Model spans with columns: text_id, span_start, span_end, span_tag, category.
        df_texts: Texts with columns: text_id, text_content.
        ablation_configs: List of (config_name, categories_list) tuples.
        output_dir: Directory to save per-config results.
        tokenizer_name: HuggingFace tokenizer name for subword eval; None = char-level.

    Returns:
        DataFrame summarizing micro/macro P/R/F1 per ablation config.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_spans = spans_to_dict(df_gold)

    tokenizer = None
    if tokenizer_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    precomputed_offsets = precompute_offsets(df_texts, tokenizer)

    all_results = []

    for config_name, categories in ablation_configs:
        df_filtered = df_ml[df_ml["category"].isin(categories)].copy()

        if df_filtered.empty:
            print(f"  WARNING: No spans for config '{config_name}', skipping")
            continue

        # resolve_dataframe_conflicts expects note_id
        df_for_resolve = df_filtered.rename(columns={"text_id": "note_id"})
        df_resolved = resolve_dataframe_conflicts(df_for_resolve, "pipeline")
        df_resolved = df_resolved.rename(columns={"note_id": "text_id"})

        pred_spans = spans_to_dict(df_resolved)

        metrics = evaluate_token_level(
            df_texts,
            gold_spans,
            pred_spans,
            precomputed_offsets=precomputed_offsets,
        )

        micro = metrics.get("micro_avg", {})
        macro = metrics.get("macro_avg", {})

        print(
            f"  {config_name}: Micro P/R/F1 = "
            f"{micro.get('precision', 0):.4f} / {micro.get('recall', 0):.4f} / {micro.get('f1', 0):.4f}"
        )

        all_results.append(
            {
                "config": config_name,
                "categories": "+".join(categories),
                "num_spans": len(df_resolved),
                "micro_precision": micro.get("precision", 0),
                "micro_recall": micro.get("recall", 0),
                "micro_f1": micro.get("f1", 0),
                "macro_precision": macro.get("precision", 0),
                "macro_recall": macro.get("recall", 0),
                "macro_f1": macro.get("f1", 0),
            }
        )

        config_dir = output_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        _save_token_metrics(
            metrics,
            config_dir / "token_metrics.json",
            metadata={
                "config_name": config_name,
                "categories": categories,
                "tokenizer": tokenizer_name,
            },
        )

    ablation_df = pd.DataFrame(all_results)
    if not ablation_df.empty:
        ablation_df.to_csv(output_dir / "token_ablation_summary.csv", index=False)
    return ablation_df
