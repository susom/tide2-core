# Evaluation

Span-level and token-level evaluation of TIDE 2.0 de-identification performance using the sample data.

## Workflow

### 1. Run the pipeline

Execute the pipeline notebook (`notebooks/tide2_pipeline.ipynb`) to produce model predictions. This generates:

```
notebooks/sample_data/pipeline_output/
├── ml_spans.parquet      # Model predictions (text_id, span_start, span_end, span_tag, recognizer_name)
└── texts.parquet         # Original texts (text_id, text_content)
```

### 2. Gold standard

The gold standard lives in `notebooks/sample_data/gold_standard/` as one JSON file per note. Each file contains a list of hand-verified span objects with exact character offsets. No generation step is required.

### 3. Run evaluation

```bash
python notebooks/evaluation/run_eval.py \
    --gold-dir notebooks/sample_data/gold_standard \
    --output-dir notebooks/sample_data/pipeline_output \
    --eval-type both
```

This runs ablation-based evaluation comparing gold spans against model predictions, producing span-level and token-level precision/recall/F1 metrics.

## Expected file layout

```
notebooks/sample_data/
├── text_files/             # Raw note text files (<note_id>.txt)
├── patient_phi/            # Planted PHI per note (<note_id>.json)
├── gold_standard/          # Gold standard spans (<note_id>.json)
└── pipeline_output/
    ├── ml_spans.parquet    # Model predictions
    └── texts.parquet       # Original texts
```

## Options

| Flag | Description |
|------|-------------|
| `--gold-dir` | Directory containing gold standard JSON files (one per note) |
| `--output-dir` | Directory containing `ml_spans.parquet` and `texts.parquet` |
| `--eval-type` | `span`, `token`, or `both` (default) |
| `--overlap-threshold` | Overlap threshold for span matching (default: 0.8) |
| `--tokenizer` | HuggingFace tokenizer for subword token eval (default: char-level) |
