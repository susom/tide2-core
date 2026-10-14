# Large-Scale Distributed Pipeline Guide

## 1. Overview

This guide explains how to run the TIDE 2.0 de-identification pipeline at large scale across distributed compute nodes.
In large-scale production environments, workloads execute in discrete stages across dedicated compute clusters.
GPU nodes execute transformer deep-learning models.
CPU nodes execute regex pattern matching, structured dictionary lookups, and cryptographic anonymization.

This document uses the ASD-STE100 standard for Simplified Technical English.

---

## 2. Pipeline Stage Architecture

The pipeline processes clinical notes in three discrete stages:

<table>
  <thead>
    <tr>
      <th align="center">Source Data</th>
      <th align="center"></th>
      <th align="center">Stage 1 (GPU Nodes)</th>
      <th align="center"></th>
      <th align="center">Stage 2 (CPU Nodes)</th>
      <th align="center"></th>
      <th align="center">Stage 3 (CPU Nodes)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">
        <b>Input Parquet</b><br>
        <code>note_text</code><br>
        <i>(row_id recommended)</i>
      </td>
      <td align="center">&#10132;</td>
      <td align="center">
        <b>Transformer Stage</b><br>
        <i>Writes:</i> <code>transformer_output/</code><br>
        <i>Adds:</i> <code>recognizer_results_json</code>
      </td>
      <td align="center">&#10132;</td>
      <td align="center">
        <b>Recognizer Stage</b><br>
        <i>Writes:</i> <code>recognizer_output/</code><br>
        <i>Adds:</i> Regex + Dictionary PHI
      </td>
      <td align="center">&#10132;</td>
      <td align="center">
        <b>Anonymizer Stage</b><br>
        <i>Writes:</i> <code>anonymizer_output/</code><br>
        <i>Emits:</i> <code>anonymized_note_text</code>
      </td>
    </tr>
  </tbody>
</table>

1. **Stage 1 (Transformer Inference)**: Tokenizes clinical notes, executes token-accurate windowing, performs neural NER, and aggregates BIO tokens in memory.
2. **Stage 2 (Pattern and Known-Values Recognition)**: Merges transformer predictions with regular expressions, checksums, and structured patient identifiers.
3. **Stage 3 (Cryptographic Anonymization)**: Applies HIPS date shifting, format-preserving encryption (FPE), surrogate generation, and masking. Deletes raw text and outputs anonymized notes.

---

## 3. Input and Output Contracts for Individual Stages

Each stage can execute as an independent Ray Data job on dedicated hardware fleets.

**Important Column Rule**: For input source datasets, **`note_text` is the only required column**.
When `text_hash` and `row_id` are absent from source Parquet files, Ray read tasks derive them automatically.
Downstream anonymization requires `recognizer_results_json` from upstream recognizers to identify entity spans.

---

### 3.1 Stage 1: Transformer Inference (`run_transformer`)

#### Purpose
Executes deep-learning named entity recognition (NER) using token-accurate windowing based on `MODEL_MAX_LENGTH`.

#### CLI Command
```bash
tide2-runner run transformer \
  -i /shared_fs/data/shards/shard_01.parquet \
  -o /shared_fs/runs/run_01/shard_01/transformer_output \
  --model stanford-med-hdr/tide2-sentry-clinical-ner
```

#### Python Equivalent
```python
from tide2.runner.local_runner import LocalJobRunner

runner = LocalJobRunner(num_gpus=1.0)
runner.run_transformer(
    input_path="/shared_fs/data/shards/shard_01.parquet",
    output_path="/shared_fs/runs/run_01/shard_01/transformer_output",
    model_name="stanford-med-hdr/tide2-sentry-clinical-ner",
)
```

#### Output Directory
- When run inside `run_pipeline`: Writes to `<output_dir>/transformer_output`.
- When run standalone: Writes directly to the path supplied in `-o` / `--output`.
- Checkpoint directory: `<output_path>_ray_checkpoint` (sibling directory).

#### Column Contract (`TRANSFORMER_STAGE_COLUMNS`)
| Type | Columns | Notes |
|---|---|---|
| **Required Input** | `note_text` | Only required column in source data. |
| **Optional Passthrough** | `row_id`, `text_hash`, `patient_id`, `patient_identifiers`, `jitter`, `processing_status`, `stage_status_json` | `row_id` and `text_hash` are derived if absent. `patient_id` (converted to string), `patient_identifiers`, and `jitter` pass through if present. `processing_status` and `stage_status_json` are updated for each note. |
| **Output Columns** | `text_hash`, `note_text`, `patient_id`, `row_id`, `recognizer_results_json`, `entity_count`, `processing_timestamp`, `processing_status`, `stage_status_json` | `recognizer_results_json` contains deep-learning entity spans in Presidio format. `entity_count` contains detected entity count. |

---

### 3.2 Stage 2: Pattern and Known-Values Recognizer (`run_recognition`)

#### Purpose
Combines deep-learning entity predictions with regex patterns, checksums, and structured patient identifiers.

#### CLI Command
```bash
tide2-runner run recognizer \
  -i /shared_fs/runs/run_01/shard_01/transformer_output \
  -o /shared_fs/runs/run_01/shard_01/recognizer_output
```

#### Python Equivalent
```python
from tide2.runner.local_runner import LocalJobRunner

runner = LocalJobRunner()
runner.run_recognition(
    input_path="/shared_fs/runs/run_01/shard_01/transformer_output",
    output_path="/shared_fs/runs/run_01/shard_01/recognizer_output",
)
```

#### Output Directory
- When run inside `run_pipeline`: Writes to `<output_dir>/recognizer_output`.
- When run standalone: Writes directly to the path supplied in `-o` / `--output`.
- Checkpoint directory: `<output_path>_ray_checkpoint` (sibling directory).

#### Column Contract (`RECOGNIZER_STAGE_COLUMNS`)
| Type | Columns | Notes |
|---|---|---|
| **Required Input** | `note_text` | Only required column. |
| **Optional Passthrough** | `recognizer_results_json`, `patient_identifiers`, `patient_id`, `row_id`, `text_hash`, `jitter`, `processing_status`, `stage_status_json` | Existing transformer spans in `recognizer_results_json` merge with regex and dictionary detections. `patient_id` and `jitter` pass through if present. `processing_status` and `stage_status_json` are updated. |
| **Output Columns** | `text_hash`, `note_text`, `patient_id`, `row_id`, `recognizer_results_json`, `entity_count`, `processing_timestamp`, `processing_status`, `stage_status_json` | `recognizer_results_json` contains all merged detections. `entity_count` contains total detection count. |

---

### 3.3 Stage 2 (Optional Alternative): LLM Recognizer (`run_llm_recognition`)

#### Purpose
Extracts entity spans using large language model prompts instead of, or in addition to, regex and transformer models.

#### CLI Command
```bash
tide2-runner run llm-recognizer \
  -i /shared_fs/data/shards/shard_01.parquet \
  -o /shared_fs/runs/run_01/shard_01/llm_recognizer_output \
  --project-id my-gcp-project \
  --model gemini-2.5-flash
```

#### Python Equivalent
```python
from tide2.runner.local_runner import LocalJobRunner

runner = LocalJobRunner()
runner.run_llm_recognition(
    input_path="/shared_fs/data/shards/shard_01.parquet",
    output_path="/shared_fs/runs/run_01/shard_01/llm_recognizer_output",
    project_id="my-gcp-project",
    model_name="gemini-2.5-flash",
)
```

#### Output Directory
- Standalone mode: Writes to `<output_dir>`.
- Pipeline `llm_recognizer_mode="only"`: Writes to `<output_dir>/llm_recognizer_output`.
- Pipeline `llm_recognizer_mode="merge"`: Merges regex and LLM detections on `row_id` and writes to `<output_dir>/merged_recognizer_output`.

#### Column Contract (`LLM_RECOGNIZER_STAGE_COLUMNS`)
| Type | Columns | Notes |
|---|---|---|
| **Required Input** | `note_text` | Only required column. |
| **Optional Passthrough** | `patient_identifiers`, `patient_id`, `jitter`, `row_id`, `text_hash`, `processing_status`, `stage_status_json` | Passed through to output if present. `processing_status` and `stage_status_json` are updated. |
| **Output Columns** | `text_hash`, `note_text`, `patient_id`, `row_id`, `recognizer_results_json`, `entity_count`, `processing_timestamp`, `processing_status`, `stage_status_json` | Contains LLM-extracted entity spans. |

---

### 3.4 Stage 3: Cryptographic Anonymizer (`run_anonymization`)

#### Purpose
Applies HIPS cryptographic date shifting, format-preserving encryption (FPE), surrogate generation, and masking.

#### CLI Command
```bash
tide2-runner run anonymizer \
  -i /shared_fs/runs/run_01/shard_01/recognizer_output \
  -o /shared_fs/runs/run_01/shard_01/anonymizer_output \
  --salt /path/to/salt.bin \
  --key /path/to/key.bin
```

#### Python Equivalent
```python
from tide2.runner.local_runner import LocalJobRunner

runner = LocalJobRunner()
runner.run_anonymization(
    input_path="/shared_fs/runs/run_01/shard_01/recognizer_output",
    output_path="/shared_fs/runs/run_01/shard_01/anonymizer_output",
    salt_path="/path/to/salt.bin",
    key_path="/path/to/key.bin",
)
```

#### Output Directory
- When run inside `run_pipeline`: Writes to `<output_dir>/anonymizer_output`.
- When run standalone: Writes directly to the path supplied in `-o` / `--output`.
- Checkpoint directory: `<output_path>_ray_checkpoint` (sibling directory).

#### Key Material Requirements
The anonymizer requires key material:
- **CLI (`tide2-runner run anonymizer`)**: Requires file paths `--salt <path>` and `--key <path>`. (The flags `--salt-hex` and `--key-hex` are accepted only by pipeline jobs: `tide2-runner run pipeline`).
- **Python API (`runner.run_anonymization`)**: Accepts either file paths (`salt_path` and `key_path`) or hex strings (`salt_hex` and `key_hex`).

#### Column Contract (`ANONYMIZER_STAGE_COLUMNS` and `FINAL_OUTPUT_COLUMNS`)
| Type | Columns | Notes |
|---|---|---|
| **Required Input** | `note_text`, `recognizer_results_json` | `recognizer_results_json` is produced by upstream recognizer stages. |
| **Optional Passthrough** | `patient_id`, `jitter`, `row_id`, `text_hash`, `processing_status`, `stage_status_json` | Used to compute dates and accession number hashes. Passed through if present. |
| **Output Columns** | `text_hash`, `patient_id`, `row_id`, `anonymized_note_text`, `anonymizer_results_json`, `entity_count`, `jitter`, `processing_timestamp`, `processing_status`, `stage_status_json` | **CRITICAL SECURITY INVARIANT**: Raw `note_text` is deleted. The output contains only `anonymized_note_text`. |

---

## 4. Best Practice: Pre-Generate `row_id` Before Pipeline Execution

**You must generate unique `row_id` values before submitting datasets to any stage.**

### Why Pre-Generation is Critical
1. If `row_id` is absent, tide2 derives the value during Parquet read tasks:
   $$\text{row\_id} = \text{sha256}(\text{text\_hash} : \text{patient\_id})$$
2. Notes with identical text and the same `patient_id` receive the same derived `row_id`.
3. The pipeline does not deduplicate notes during a run.
4. Ray Data checkpoints use `row_id` as the primary key.
5. If a job stops and restarts, the checkpoint marks committed `row_id` values as complete.
6. The restart skips all remaining notes that share an existing `row_id`.
7. This causes silent data loss on rerun for duplicate clinical notes.

### How to Pre-Generate `row_id`
When clinical notes can repeat (such as identical standard texts across encounters or patients), **you must assign a unique identifier per record (such as a UUID)** to guarantee each row has its own identity. Content-derived IDs cause duplicate texts to share an ID and collide on resume:

```python
import uuid
import pyarrow as pa
import pyarrow.parquet as pq

# RECOMMENDED: Assign a globally unique UUID string per source record and persist to disk.
# This guarantees no checkpoint collisions or skipped rows when notes repeat across retries.
table = pq.read_table("source_notes.parquet")
row_ids = [str(uuid.uuid4()) for _ in range(len(table))]
table_with_ids = table.append_column("row_id", pa.array(row_ids, type=pa.string()))
pq.write_table(table_with_ids, "prepared_notes.parquet")
```

If notes are already known to be unique per patient, you can derive deterministic content IDs using `normalize_source_batch`:

```python
import pyarrow.parquet as pq
from tide2.runner.local_runner import normalize_source_batch

# Content-derived IDs: derives sha256(text_hash:patient_id).
# Notes with identical text and patient will share this ID.
table = pq.read_table("source_notes.parquet")
table_with_ids = normalize_source_batch(table)
pq.write_table(table_with_ids, "prepared_notes.parquet")
```

---

## 5. Distributed Checkpointing and Fault Tolerance

TIDE 2.0 provides native row-level checkpointing and fault tolerance for distributed clusters.

### 5.1 Fault Handling and Recovery Mechanics

The system handles failures at three levels: worker crashes, GPU memory limits, and data validation errors.

#### 1. Worker Actor Crashes and Task Failures
- **CPU Actors (`RecognizerWorker`, `AnonymizerWorker`, `LlmRecognizerWorker`)**:
  - `max_restarts = 5`: Ray restarts a worker actor up to 5 times if it crashes (for example, on host node eviction or segfault).
  - `max_task_retries = 3`: Ray retries each failed batch task up to 3 times before raising an error.
- **GPU Actors (`TransformerInferenceActor`)**:
  - `max_restarts = 3` and `max_task_retries = 2`.
- When an actor restarts, Ray automatically resumes task processing without terminating the entire pipeline.

#### 2. Automatic CUDA Out-of-Memory (OOM) Recovery
- If a GPU actor experiences a CUDA OOM error during forward inference, recovery executes automatically:
  1. The actor intercepts the CUDA OOM exception.
  2. The actor calls `torch.cuda.empty_cache()` outside the exception frame to release pinned memory.
  3. The actor halves the window batch size (`gpu_batch_size // 2`) and recursively retries inference.
  4. The actor can halve batch sizes down to a single window.
  5. If forward inference fails on a single window, the actor marks all notes in that batch as `failed` with null results. The worker does not crash and the stage continues.

#### 3. Execution Hang Detection (`NoProgressGuard`)
- Ray Data monitors pipeline execution with `no_progress_timeout_s` (default: 600 seconds).
- If no operator moves a data block for 600 seconds (for example, on cluster deadlock), Ray Data raises `ExecutionTimeoutError` and aborts the job.
- **Ray Limitation**: Ray automatically disables `NoProgressGuard` if the execution topology includes an `AllToAllOperator` or `HashShufflingOperatorBase` (such as distributed shuffles). External schedulers (Airflow, Kubernetes, Slurm) should maintain task-level execution timeouts as a backstop.

#### 4. Zero Data Loss Contract (`max_errored_blocks = 0`)
- Ray Data's `max_errored_blocks` is set to `0`.
- Ray will not silently discard failed data blocks. If an unrecoverable batch error occurs, the job halts immediately.

#### 5. Per-Note Degradation and Failure Handling
- **Degraded Status (`processing_status = "degraded"`)**: If an entity cannot be parsed (e.g. invalid age format or unrecognized date pattern), the anonymizer replaces it with `[ENTITY_TYPE]` and anonymizes the rest of the note normally.
- **Failed Status (`processing_status = "failed"`)**: If an operator raises an exception or required jitter is missing, affected entities are masked. If note parsing fails completely, the note receives `null` output text. Raw clinical text is never emitted unmasked.
- **Upstream Failure Passthrough**: If an upstream stage marks a row as `failed`, all downstream stages pass that row through immediately without performing computation.

---

### 5.2 Checkpointing Architecture (Two-Phase Commit)

Every stage supports independent row-level checkpointing for resume capability:

- **Checkpoint Location**: Ray creates a checkpoint directory sibling to the output directory:
  $$\text{Checkpoint Path} = \langle\text{output\_dir}\rangle\text{\_ray\_checkpoint}$$
- **Configured Checkpoint Key**: Ray keys the checkpoint directory on the stage's configured ID column:
  - All stages (`run_pipeline`, `run_transformer`, `run_recognition`, `run_llm_recognition`, `run_anonymization`) default to `row_id`.
  - To checkpoint on `text_hash`, supply `--id-column text_hash` via CLI or `_id_column="text_hash"` in Python.
- **Two-Phase Commit Protocol**:
  1. The task writes a `.pending` checkpoint file.
  2. The task writes the Parquet output data file.
  3. The task renames the `.pending` file to committed status.
- **Startup Cleanup**: On job launch, Ray inspects `<output_dir>_ray_checkpoint`. Any orphaned `.pending` files and uncommitted output files from prior crashed runs are deleted.
- **Read-Time Filtering**: Ray loads committed checkpoint IDs once at startup. Read tasks skip all rows whose configured ID has already been committed.
- **Restarting Interrupted Jobs**: When a worker crashes, relaunch the identical command pointing to the same input and output paths. The job skips completed rows and processes only uncommitted rows.

---

### 5.3 Multi-Machine Configuration on a Shared Filesystem

Large-scale pipelines run across multiple machines connected to a single shared filesystem (such as Google Cloud Filestore, NFS, Lustre, or Cloud Storage via `gcsfuse` / native `gs://` URIs).

#### 1. Filesystem Mount Requirements
- Every compute node must mount the shared filesystem at the **exact same absolute directory path**.
  - Example: `/mnt/shared_storage/runs/...` must be identical on all GPU and CPU worker nodes.
- When using `gs://` URIs, all nodes must have authorized Google Cloud credentials with read and write access to the bucket.

#### 2. Shard Directory Structure
Divide the dataset into $N$ disjoint input shards.
Create an isolated directory structure per shard under the master run directory:

```text
/mnt/shared_storage/runs/run_2026_10/
├── shard_0001/
│   ├── transformer_output/               # Written by GPU Worker
│   ├── transformer_output_ray_checkpoint/
│   ├── recognizer_output/                # Written by CPU Worker
│   ├── recognizer_output_ray_checkpoint/
│   ├── anonymizer_output/                # Written by CPU Worker
│   └── anonymizer_output_ray_checkpoint/
├── shard_0002/
│   ├── ...
```

#### 3. Chaining Stages Across Fleets
Each fleet reads the output directory written by the previous fleet:

```bash
# Step 1: Run Transformer on GPU Node (Fleet 1)
tide2-runner run transformer \
  -i /mnt/shared_storage/inputs/shard_0001.parquet \
  -o /mnt/shared_storage/runs/run_2026_10/shard_0001/transformer_output \
  --model stanford-med-hdr/tide2-sentry-clinical-ner

# Step 2: Run Recognizer on CPU Node (Fleet 2)
tide2-runner run recognizer \
  -i /mnt/shared_storage/runs/run_2026_10/shard_0001/transformer_output \
  -o /mnt/shared_storage/runs/run_2026_10/shard_0001/recognizer_output

# Step 3: Run Anonymizer on CPU Node (Fleet 3)
tide2-runner run anonymizer \
  -i /mnt/shared_storage/runs/run_2026_10/shard_0001/recognizer_output \
  -o /mnt/shared_storage/runs/run_2026_10/shard_0001/anonymizer_output \
  --salt /mnt/shared_storage/keys/salt.bin \
  --key /mnt/shared_storage/keys/key.bin
```

---

### 5.4 The Four Orchestrator Guarantees

When external schedulers (Airflow, Kubernetes, Slurm) manage distributed workers on a shared filesystem, they must satisfy four guarantees:

| Rule # | Guarantee | Requirement | Why |
|---|---|---|---|
| **1** | **Disjoint Shards** | Ensure no input note exists in more than one shard. | tide2 does not deduplicate across shards. Duplicate notes in multiple shards will be processed multiple times. |
| **2** | **Stable Output Paths** | Reuse the exact same output path when retrying a failed shard. | Checkpoints live at `<output_dir>_ray_checkpoint`. Changing the output path causes the job to restart from zero. |
| **3** | **Single Live Attempt** | Run at most one live worker process per shard output path. Launch a retry only after the prior process has terminated. | Ray startup deletes `.pending` files. Two concurrent jobs using the same directory delete each other's checkpoints and cause race conditions. |
| **4** | **Immutable Parameters** | Keep input data, model, keys, and settings identical across retries. | Ray keys checkpoints on the configured ID column only (`row_id` or `text_hash`). If settings change, you must specify a new output path. |

---

## 6. Hardware Autotuning

The `tide2.runner.hardware` module detects physical and container resources (CPU affinity, cgroup quotas, RAM, GPUs via `nvidia-smi`) and applies optimal defaults defined in `src/tide2/resources/hardware_rules.yaml`.

- `gpu` (CPUs $> 4$, GPUs $\ge 1$): Allocates up to 3 transformer worker actors per GPU based on available VRAM (requires ~7 GB per actor).
- `cpu-stages` (CPUs $> 4$): Sets recognizer and anonymizer actors to approximately $87.5\%$ of host CPUs (for example, 14 actors on a 16-CPU host).
- `many-cores` (CPUs $\ge 64$): Sets CPU actors to `cpus - 2` to maximize core utilization on high-capacity nodes.
- `low-ram` (RAM $\le 24$ GB): Overrides actor counts to at most `ram_gb // 2` to prevent host memory exhaustion.
- `cpu-only` (CPUs $> 4$, GPUs $= 0$): Sets transformer actors to approximately $25\%$ of CPUs and aggregation actors to $30\%$ of CPUs.
- `tiny` (CPUs $\le 4$): Sets `enable_checkpoint: false` and allocates fractional CPUs to prevent Ray 2.55 reservation deadlocks on small VMs.
  - **Standalone Stage Requirement**: Automatic `tiny` sizing (fractional CPUs and disabled checkpoints) is applied by `run_pipeline`. Standalone stage jobs (`run_transformer`, `run_recognition`, `run_anonymization`) auto-size actors but retain default task reservations (`read_cpus`, `write_cpus`) and enable checkpointing. On machines with $\le 4$ CPUs, standalone CLI commands must explicitly pass fractional CPU flags and `--no-checkpoint` (for CPU stages: `--read-cpus 0.25 --write-cpus 0.25 --worker-num-cpus 0.5 --no-checkpoint`; for transformer: `--read-cpus 0.25 --write-cpus 0.25 --agg-num-cpus 0.5 --transformer-cpus 0.25 --no-checkpoint`). Unified knob forwarding across all standalone stages is tracked in issue #100.
- **User Overrides Always Win**: Any parameter specified on the CLI, via YAML config, or passed as a Python argument overrides the hardware autotune recommendations unconditionally.
- **Opt-out**: Pass `--no-hardware-autotune` or set `hardware_autotune: false` to use safe static defaults.

---

## 7. Citation

If you use TIDE 2.0 or the SHIELD benchmark in your research, please cite:

```bibtex
@article{posada2026tide2,
  title={TIDE 2.0: an open, model-agnostic engine for keyed de-identification of clinical notes},
  author={Posada, Jose D. and Datta, Somalee and Desai, Priya},
  journal={arXiv preprint arXiv:2610.07224},
  year={2026},
  url={https://arxiv.org/abs/2610.07224}
}

@article{posada2026shield,
  title={SHIELD: A Diverse Clinical Note Dataset and Distilled Small Language Models for Enterprise-Scale De-identification},
  author={Posada, Jose D. and Love, David and Datta, Somalee and Desai, Priya},
  journal={arXiv preprint arXiv:2605.03301},
  year={2026},
  url={https://arxiv.org/abs/2605.03301}
}
```
