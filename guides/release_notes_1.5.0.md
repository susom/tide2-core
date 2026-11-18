# Release Notes: TIDE 2.0 (v1.5.0)

## 1. About This Document

This document gives the release notes for `tide2-core` version 1.5.0.
Version 1.5.0 has all the new functions of version 1.4.0.
PyPI has no package for version 1.4.0.
The last package on PyPI is version 1.3.1.
Use this document to upgrade from version 1.3.1 to version 1.5.0.

The document has two parts:

- Part 1 (sections 3 to 9) gives the new pipeline functions from version 1.4.0.
- Part 2 (sections 10 to 18) gives the new anonymization functions from version 1.5.0.

This document is for operators who run the pipeline on many nodes.
The pipeline removes PHI from clinical notes.
This document follows ASD-STE100.

---

## 2. Before You Upgrade

Some items give different output or stop your workflow.
Read this section before you run the new version.

| Item | Effect | Step |
|---|---|---|
| The `reassembly` stage is not available. | The CLI rejects `run reassembly` with exit code `2`. | Remove all `reassembly` steps. |
| The `patient_uid` column is not available. | The pipeline gives `ValueError`. | Replace the column name with `patient_id`. |
| `run_pipeline` accepts Parquet paths only. | A `pandas.DataFrame` input gives `TypeError`. | Write the data to Parquet first. |
| The stage folders have new names. | The downstream tasks do not find `02_transformer_output`. | Use `transformer_output`, `recognizer_output` and `anonymizer_output`. |
| FPE uses NIST FF1 and not FF3. | The pseudonyms are different from earlier runs. | Run again from the source data. Do not join earlier output and new output. |
| Date output keeps the input format. | Some dates are different from earlier output. | Run again from the source data. |
| `ACC_NUM` uses HIPS alphanumeric encryption as the default. | The output is not a 16-character hash. | Use `--text-operators ACC_NUM=acc_num_hash` to keep the hash. |
| ZIP codes follow `--zip-mode`. | The default mode `relaxed` keeps 5 digits in large areas. | Use `--zip-mode strict` for HIPAA Safe Harbor. |
| Standalone stages use `row_id` as the checkpoint key. | Resume uses `row_id` and not `text_hash`. | Use `--id-column text_hash` for the earlier key. |
| `pandas` is an optional dependency. | The package does not include `pandas`. | Install the `evaluation` extra or the `dev` extra if you must have it. |
| The visualizer reads Parquet only. | The visualizer does not read BigQuery, CSV or JSON input. | Write your data to Parquet. |

---

## 3. Removal of the `reassembly` Stage (v1.4.0)

The pipeline does not have a `reassembly` stage.

### 3.1 New Function

The earlier version used three steps for transformer work:

1. Split long notes into chunks of the same number of characters.
2. Run transformer inference on each chunk.
3. Run a `reassembly` job with a Ray `groupby` operation to assemble the notes.

The new version has no `reassembly` job:

- The transformer stage (`run_transformer`) processes full notes.
- The stage uses token windows. The size of a window comes from `MODEL_MAX_LENGTH`.
- An actor in memory (`BIOAggregationActor`) collects the tokens. The actor resolves spans that overlap.
- The transformer stage writes the predictions for each full note.
- The pipeline does not write chunk files.

### 3.2 Operator Step

Do not run `tide2-runner run reassembly` in your workflows.
The CLI rejects this command with exit code `2`.
Remove all `reassembly` steps from your workflow scripts.

```bash
# Earlier workflow (DO NOT USE):
# tide2-runner run transformer -i notes.parquet -o /tmp/raw_ner --model <model>
# tide2-runner run reassembly -i /tmp/raw_ner -o /tmp/ner_out --model <model>

# New workflow:
tide2-runner run transformer -i notes.parquet -o /tmp/ner_out --model <model>
```

---

## 4. Removed Functions (v1.4.0)

The table shows each component, parameter and function that version 1.4.0 removes.

| Item | Earlier function | New function | Step |
|---|---|---|---|
| `reassembly` stage | It ran as a stage with `groupby`. | The transformer stage does the work. | Remove all `reassembly` tasks. |
| `patient_uid` column | The pipeline accepted it as an input column. | The pipeline gives `ValueError` and `DeprecationWarning`. | Replace the column name with `patient_id`. |
| `pandas.DataFrame` input | `run_pipeline` accepted it on the driver node. | The pipeline gives `TypeError`. | Write Parquet. Give file paths or directory paths. |
| Parquet files between stages | The pipeline wrote `01_` and `03_` files. | Ray Data streams the data. | Adjust the tasks that read the earlier paths. |
| `--chunk-size` flag | The flag gave the character size of the chunks. | The CLI rejects the flag. | Use `--chunk-overlap` for the token overlap. |
| `--batch-timeout` flag | The flag removed batches that did not move for 120 seconds. | The CLI rejects the flag. | Use `--no-progress-timeout`. |
| `--cpus-per-actor` flag | The flag gave the CPUs of each worker actor. | The CLI rejects the flag. | Use `--worker-num-cpus`. |
| Stage flags of `run_pipeline` | You could use `--no-transformer` and other flags of this type. | The CLI rejects the flags. | Run one stage command at a time. |
| `produce_visualizer_json` | The pipeline wrote the visualizer files. | The parameter is not available. | Run `tide2-runner visualizer-json -o <dir>` after the run. |
| Supervisor actors | Workers ran below a supervisor layer. | Workers connect directly to the actor pools of Ray Data. | None. |
| Stage folder names | The pipeline used names such as `02_transformer_output`. | The pipeline uses names such as `transformer_output` and `recognizer_output`. | Adjust the paths in downstream tasks. |

---

## 5. New Runner Parameters (v1.4.0)

Version 1.4.0 adds parameters to `LocalJobRunner` and `tide2-runner`.

#### `execution_mode` (values `"discrete"` and `"streamed"`, default `"discrete"`)

- `"discrete"` runs each stage as a Ray Data job.
  Each stage writes Parquet files before the next stage starts.
  Use this mode for clusters of many machines, shared filesystems and nodes with 4 CPUs or less.
- `"streamed"` connects all stages in one Ray run on one node.
  The stages send data in the Ray object store.
  Use this mode for local tests and benchmarks only.

#### `hardware_autotune` (Boolean, default `True`)

- When the value is `True`, the runner probes the CPUs, RAM and GPUs of the host.
  The runner then selects the worker pools and the batch sizes.
- To stop autotuning, use `--no-hardware-autotune` or write `hardware_autotune: false`.

#### `no_progress_timeout_s` and `--no-progress-timeout` (float, default `600.0`)

- This value sets the Ray Data `NoProgressGuard`.
- The runner stops the job when no operator processes data in this time.
- Use `-1` to stop the guard.

#### `worker_num_cpus` (float, default `1.0`)

- This value gives the CPU cores of each recognizer actor and each anonymizer actor.
- On small machines, use a smaller value, for example `0.5`, to prevent deadlocks in the scheduler.

#### `override_num_blocks` (integer, default `None`)

- This value gives the number of Ray Data blocks of the input dataset.
- Use it on machines with many cores to prevent core starvation from large blocks.

#### `write_cpus` (float, default `1.0`)

- This value gives the CPUs that each `write_parquet` task reserves.

#### Fractional `num_gpus` (float, default `1.0`)

- `num_gpus` accepts fractional values, for example `0.33` or `0.5`.
- More than one transformer actor can then use one GPU.

#### `object_store_gb` (integer, default `None`)

- This value gives the limit of the Ray object store in gigabytes.
- When you give no value, Ray selects a limit from the system.

```python
from pathlib import Path

from tide2.runner.local_runner import LocalJobRunner

# Load your own salt and key from protected storage.
# If you omit them, run_pipeline uses public default values. Never use those values for real data.
salt_hex = Path("/secure/keys/salt.hex").read_text().strip()
key_hex = Path("/secure/keys/key.hex").read_text().strip()

runner = LocalJobRunner(
    num_cpus=32,
    num_gpus=1.0,
    hardware_autotune=True,
    no_progress_timeout_s=600.0,
)

runner.run_pipeline(
    input_path="gs://my-bucket/shards/shard_01.parquet",
    output_dir="gs://my-bucket/runs/shard_01",
    model_name="stanford-med-hdr/tide2-sentry-clinical-ner",
    salt_hex=salt_hex,
    key_hex=key_hex,
    execution_mode="discrete",
)
```

---

## 6. Input and Output of Each Stage (v1.4.0)

In large pipelines, you run one stage at a time.
You can run the transformer on GPU nodes.
You can run the recognizer and the anonymizer on CPU nodes.

**Column rule.** `note_text` is the only input column that is necessary for source data.
When `text_hash` and `row_id` are not in the source Parquet files, the read tasks derive them.
The anonymizer stage must have `recognizer_results_json` from an earlier recognizer stage.

| Stage | Node type | Writes | Adds |
|---|---|---|---|
| Transformer | GPU | `transformer_output/` | `recognizer_results_json` |
| Recognizer | CPU | `recognizer_output/` | Regex and dictionary detections |
| Anonymizer | CPU | `anonymizer_output/` | `anonymized_note_text` |

### 6.1 Stage 1: Transformer (`run_transformer`)

The stage runs named entity recognition (NER) with a transformer model and token windows.

```bash
tide2-runner run transformer \
  -i /shared_fs/data/shards/shard_01.parquet \
  -o /shared_fs/runs/run_01/shard_01/transformer_output \
  --model stanford-med-hdr/tide2-sentry-clinical-ner
```

```python
from tide2.runner.local_runner import LocalJobRunner

runner = LocalJobRunner(num_gpus=1.0)
runner.run_transformer(
    input_path="/shared_fs/data/shards/shard_01.parquet",
    output_path="/shared_fs/runs/run_01/shard_01/transformer_output",
    model_name="stanford-med-hdr/tide2-sentry-clinical-ner",
)
```

- In `run_pipeline`, the stage writes to `<output_dir>/transformer_output`.
- When you run only this stage, it writes to the path in `-o`.
- The checkpoint directory is `<output_path>_ray_checkpoint`. This directory is next to the output directory.

| Type | Columns | Notes |
|---|---|---|
| Input that is necessary | `note_text` | This is the only input column that is necessary. |
| Optional passthrough | `row_id`, `text_hash`, `patient_id`, `patient_identifiers`, `jitter`, `birth_date`, `processing_status`, `stage_status_json` | The stage derives `row_id` and `text_hash` when they are not in the input. The stage writes `patient_id` as a string. |
| Output | `text_hash`, `note_text`, `patient_id`, `row_id`, `recognizer_results_json`, `entity_count`, `processing_timestamp`, `processing_status`, `stage_status_json` | `recognizer_results_json` has the entity spans in Presidio format. |

### 6.2 Stage 2: Recognizer (`run_recognition`)

The stage puts the transformer spans together with regex patterns, checksums and patient identifiers from your tables.

```bash
tide2-runner run recognizer \
  -i /shared_fs/runs/run_01/shard_01/transformer_output \
  -o /shared_fs/runs/run_01/shard_01/recognizer_output
```

```python
runner = LocalJobRunner()
runner.run_recognition(
    input_path="/shared_fs/runs/run_01/shard_01/transformer_output",
    output_path="/shared_fs/runs/run_01/shard_01/recognizer_output",
)
```

- In `run_pipeline`, the stage writes to `<output_dir>/recognizer_output`.
- The stage records each span, also a span with a low score.

| Type | Columns | Notes |
|---|---|---|
| Input that is necessary | `note_text` | This is the only input column that is necessary. |
| Optional passthrough | `recognizer_results_json`, `patient_identifiers`, `patient_id`, `row_id`, `text_hash`, `jitter`, `birth_date`, `processing_status`, `stage_status_json` | The stage merges the earlier transformer spans with the detections that it finds. |
| Output | `text_hash`, `note_text`, `patient_id`, `row_id`, `recognizer_results_json`, `entity_count`, `processing_timestamp`, `processing_status`, `stage_status_json` | `recognizer_results_json` has all the merged detections. |

### 6.3 Stage 2 (Optional): LLM Recognizer (`run_llm_recognition`)

The stage finds entity spans with large language model prompts.
You can use it with the other recognizer stages.
You can also use it without them.

```bash
tide2-runner run llm-recognizer \
  -i /shared_fs/data/shards/shard_01.parquet \
  -o /shared_fs/runs/run_01/shard_01/llm_recognizer_output \
  --project-id my-gcp-project \
  --model gemini-2.5-flash
```

```python
runner = LocalJobRunner()
runner.run_llm_recognition(
    input_path="/shared_fs/data/shards/shard_01.parquet",
    output_path="/shared_fs/runs/run_01/shard_01/llm_recognizer_output",
    project_id="my-gcp-project",
    model_name="gemini-2.5-flash",
)
```

- When you run only this stage, it writes to `<output_dir>`.
- In the pipeline with `llm_recognizer_mode="only"`, it writes to `<output_dir>/llm_recognizer_output`.
- In the pipeline with `llm_recognizer_mode="merge"`, it merges the regex detections and the LLM detections on `row_id`.
  It writes the result to `<output_dir>/merged_recognizer_output`.

### 6.4 Stage 3: Anonymizer (`run_anonymization`)

The stage applies HIPS date shifts, FPE (encryption that keeps the format), surrogate values and masks.

```bash
tide2-runner run anonymizer \
  -i /shared_fs/runs/run_01/shard_01/recognizer_output \
  -o /shared_fs/runs/run_01/shard_01/anonymizer_output \
  --salt /path/to/salt.bin \
  --key /path/to/key.bin
```

```python
runner = LocalJobRunner()
runner.run_anonymization(
    input_path="/shared_fs/runs/run_01/shard_01/recognizer_output",
    output_path="/shared_fs/runs/run_01/shard_01/anonymizer_output",
    salt_path="/path/to/salt.bin",
    key_path="/path/to/key.bin",
)
```

- In `run_pipeline`, the stage writes to `<output_dir>/anonymizer_output`.
- The CLI must have the file paths `--salt` and `--key`.
- Only pipeline jobs accept `--salt-hex` and `--key-hex`.
- The Python API accepts file paths (`salt_path`, `key_path`) or hex strings (`salt_hex`, `key_hex`).

| Type | Columns | Notes |
|---|---|---|
| Input that is necessary | `note_text`, `recognizer_results_json` | An earlier recognizer stage writes `recognizer_results_json`. |
| Optional passthrough | `patient_id`, `jitter`, `birth_date`, `row_id`, `text_hash`, `processing_status`, `stage_status_json` | The stage uses these columns for date shifts, the age cap and hashes. |
| Output | `text_hash`, `patient_id`, `row_id`, `anonymized_note_text`, `anonymizer_results_json`, `entity_count`, `jitter`, `processing_timestamp`, `processing_status`, `stage_status_json` | The output does not have `note_text` or `birth_date`. |

**PHI rule.** The anonymizer output does not have the `note_text` column.
The output has `anonymized_note_text` only.

---

## 7. Make a Unique `row_id` Before You Run a Stage (v1.4.0)

You must make a unique `row_id` for each row before you give a dataset to a stage.

### 7.1 Cause

1. When `row_id` is not in the input, tide2 derives it in the Parquet read task.
   The formula is `sha256(text_hash:patient_id)`.
2. Notes with the same text and the same `patient_id` get the same `row_id`.
3. The pipeline does not remove duplicate notes.
4. The Ray Data checkpoint uses `row_id` as the primary key.
5. After a restart, the checkpoint lists each `row_id` that the job wrote.
6. The restart skips all notes with a `row_id` in that list.
7. The pipeline then skips notes that it did not process, and it gives no error.

### 7.2 Procedure

When notes can be duplicates, give each row a unique identifier, for example a UUID.
Write the result to disk.

```python
import uuid
import pyarrow as pa
import pyarrow.parquet as pq

table = pq.read_table("source_notes.parquet")
row_ids = [str(uuid.uuid4()) for _ in range(len(table))]
table_with_ids = table.append_column("row_id", pa.array(row_ids, type=pa.string()))
pq.write_table(table_with_ids, "prepared_notes.parquet")
```

If each note is unique for its patient, you can derive content identifiers with `normalize_source_batch`.
Notes with the same text and the same patient get the same ID.

```python
import pyarrow.parquet as pq
from tide2.runner.local_runner import normalize_source_batch

table = pq.read_table("source_notes.parquet")
table_with_ids = normalize_source_batch(table)
pq.write_table(table_with_ids, "prepared_notes.parquet")
```

---

## 8. Checkpoints, Faults and Runs on Many Machines (v1.4.0 and v1.5.0)

Version 1.4.0 adds fault tolerance and row-level checkpoints for large clusters.

### 8.1 Faults and Errors

The system handles worker crashes, GPU memory limits and data errors.

#### Worker crashes and task failures

- CPU actors (`RecognizerWorker`, `AnonymizerWorker`, `LlmRecognizerWorker`) use `max_restarts = 5`.
  Ray restarts a crashed actor up to 5 times.
- CPU actors use `max_task_retries = 3`.
  Ray runs a batch that fails again up to 3 times before it gives an error.
- GPU actors (`TransformerInferenceActor`) use `max_restarts = 3` and `max_task_retries = 2`.
- When an actor restarts, Ray continues the task. The pipeline does not stop.

#### CUDA out-of-memory (OOM) errors

When a GPU actor gets a CUDA OOM error, it does these steps:

1. The actor catches the exception.
2. The actor uses `torch.cuda.empty_cache()` after the `except` block.
3. The actor halves the window batch size (`gpu_batch_size // 2`) and tries again.
4. The actor can halve the batch size down to one window.
5. If one window fails, the actor writes `failed` in `processing_status` and null results for all notes of the batch.
   The worker does not crash and the stage continues.

#### Hang detection (`NoProgressGuard`)

- Ray Data monitors the run with `no_progress_timeout_s` (default 600 seconds).
- When no operator moves a block for 600 seconds, Ray Data gives `ExecutionTimeoutError` and stops the job.
- Ray stops the guard when the job has an `AllToAllOperator` or a `HashShufflingOperatorBase`.
  External schedulers (Airflow, Kubernetes, Slurm) must keep a time limit for each task as a second control.

#### No discarded blocks (`max_errored_blocks = 0`)

- Ray Data does not discard blocks that fail.
- When a batch has an error that the retries do not clear, the job stops immediately.

#### Notes with `failed` or `degraded` in `processing_status`

- `processing_status = "degraded"`: The anonymizer cannot parse an entity, for example an unknown age format or date pattern.
  The anonymizer replaces the entity with `[ENTITY_TYPE]` and anonymizes the remaining text.
- `processing_status = "failed"`: An operator gives an exception, or a jitter that is necessary is not in the input.
  The anonymizer masks the entities that have the error.
  When the anonymizer cannot parse the note, the note gets null output text.
  After an anonymizer error, the output has no unmasked text.
- When an earlier stage writes `failed` for a row, the next stages send the row through with no change.

### 8.2 Checkpoint Protocol (Two-Phase Commit)

Each stage can use row-level checkpoints to resume.

- **Location.** Ray makes the checkpoint directory next to the output directory.
  The path is `<output_dir>_ray_checkpoint`.
- **Key.** Since version 1.5.0, all stages use `row_id` as the default key.
  This includes the standalone `run_transformer` command and `run_recognition` command.
  Use `--id-column text_hash` (or `_id_column="text_hash"` in Python) for the earlier key.
  A YAML config accepts `id_column`. When a CLI flag and the YAML value are different, the runner uses the CLI flag.
- **Protocol.**
  1. The task writes a `.pending` checkpoint file.
  2. The task writes the Parquet output file.
  3. The task renames the `.pending` file to show that the output is complete.
- **Cleanup at start.** When a job starts, Ray removes orphan `.pending` files and uncommitted output files from earlier crashed runs.
- **Read filter.** Ray loads the list of identifiers of the rows that the job wrote, one time at start. Read tasks skip each row that is in the list.
- **Restart.** After a crash, run the same command again with the same input paths and output paths.
  The job skips the completed rows and processes the other rows.

### 8.3 Runs on Many Machines with a Shared Filesystem

Large pipelines run on many machines that share one filesystem.
Examples are Google Cloud Filestore, NFS, Lustre, or Cloud Storage with `gcsfuse` or `gs://` URIs.

1. **Mount path.** Each node must mount the shared filesystem at the same absolute path.
   With `gs://` URIs, each node must have credentials with read access and write access to the bucket.
2. **Shard folders.** Divide the dataset into disjoint input shards.
   Give each shard a folder in the run folder.

```text
/mnt/shared_storage/runs/run_2026_10/
├── shard_0001/
│   ├── transformer_output/
│   ├── transformer_output_ray_checkpoint/
│   ├── recognizer_output/
│   ├── recognizer_output_ray_checkpoint/
│   ├── anonymizer_output/
│   └── anonymizer_output_ray_checkpoint/
├── shard_0002/
```

3. **Stage sequence.** Each group of nodes reads the output of the group before it.

```bash
# Step 1: GPU node
tide2-runner run transformer \
  -i /mnt/shared_storage/inputs/shard_0001.parquet \
  -o /mnt/shared_storage/runs/run_2026_10/shard_0001/transformer_output \
  --model stanford-med-hdr/tide2-sentry-clinical-ner

# Step 2: CPU node
tide2-runner run recognizer \
  -i /mnt/shared_storage/runs/run_2026_10/shard_0001/transformer_output \
  -o /mnt/shared_storage/runs/run_2026_10/shard_0001/recognizer_output

# Step 3: CPU node
tide2-runner run anonymizer \
  -i /mnt/shared_storage/runs/run_2026_10/shard_0001/recognizer_output \
  -o /mnt/shared_storage/runs/run_2026_10/shard_0001/anonymizer_output \
  --salt /mnt/shared_storage/keys/salt.bin \
  --key /mnt/shared_storage/keys/key.bin
```

### 8.4 The Four Orchestrator Rules

An external scheduler (Airflow, Kubernetes, Slurm) must obey four rules.

| Rule | Step | Cause |
|---|---|---|
| 1. Disjoint shards | Put each note in one shard only. | tide2 does not remove duplicates across shards. |
| 2. Stable output paths | Use the same output path for each retry of a shard. | The checkpoint is at `<output_dir>_ray_checkpoint`. A new path starts from zero. |
| 3. One live run | Run one worker process for each shard output path. Run a retry after the earlier process stops. | Ray removes `.pending` files at start. Two jobs on one path remove the checkpoints of each other. |
| 4. Same parameters | Keep the input data, model, keys and settings the same for each retry. | The checkpoint uses the ID column only. When a setting is different, use a new output path. |

---

## 9. Hardware Autotuning (v1.4.0)

Version 1.4.0 adds the `tide2.runner.hardware` module and `hardware_rules.yaml`.

### 9.1 Operation

1. The module finds these resources. It applies the cgroup limits.
   - CPU cores that you can use
   - available RAM
   - GPU count and GPU model names (from `nvidia-smi`)
   - GPU memory (VRAM) of each GPU
2. The module compares the values with the rules in `hardware_rules.yaml`.
3. When the model name agrees with a measured profile, the module applies the tested batch sizes.
   An example is `tide2-sentry-clinical-ner` on an NVIDIA L4.
4. A value from a CLI flag, a Python argument or YAML always replaces the value of the module.
   The module does not replace a value that you give.

### 9.2 Rule Profiles

| Profile | Condition | Effect |
|---|---|---|
| `tiny` | 4 CPUs or less | Sets `enable_checkpoint: false` and uses fractional CPUs. This prevents deadlocks in the Ray 2.55 reservation on small VMs. |
| `cpu-stages` | More than 4 CPUs | Sets recognizer actors and anonymizer actors to about 87.5% of the host CPUs. |
| `many-cores` | 64 CPUs or more | Sets CPU actors to the number of CPUs minus 2. |
| `low-ram` | 24 GB RAM or less | Limits CPU actors to `ram_gb // 2`. This prevents out-of-memory errors on the host. |
| `gpu` | More than 4 CPUs and 1 GPU or more | Starts up to 3 transformer actors for each GPU. Each actor uses about 7 GB of VRAM. |
| `cpu-only` | More than 4 CPUs and no GPU | Sets transformer actors to 25% of the CPUs. Sets the hang timeout to 1200 seconds. |

---

## 10. Anonymization Policy (v1.5.0)

`AnonymizationPolicy` is a frozen and checked set of anonymization options.
It is in `tide2.anonymizers.cell`.

- `AnonymizerWorker`, `create_anonymizer_actor`, `anonymize_free_text` and `anonymize_cell` accept it as `policy=`.
- The CLI, `run_anonymization` and `run_pipeline` examine the options one time with the policy.
- Notes, table cells and `FREE_TEXT` cells use the same options.
  `text_operators` applies to free text only.
- Secrets are not policy fields. They stay as the arguments `salt`, `key`, `acc_num_salt` and `acc_num_study_id`.

| Field | Default | CLI flag |
|---|---|---|
| `min_jitter_days`, `max_jitter_days` | 3, 180 | `--min-jitter-days`, `--max-jitter-days` |
| `max_age` | 90 | `--max-age` |
| `text_operators` (free text only) | none | `--text-operators` |
| `jitter_required` | off | `--jitter-required` |
| `fpe_mode` | `strict` | `--fpe-mode` |
| `zip_mode` | `relaxed` | `--zip-mode` |
| `transformer_score_threshold` | 0.0 | `--transformer-score-threshold` |
| `recognizer_score_threshold` | 0.0 | `--recognizer-score-threshold` |

```python
policy = AnonymizationPolicy(fpe_mode="relaxed", text_operators="DATE=keep")
worker = AnonymizerWorker(salt, key, policy=policy)
anonymize_cell(value, "MRN", salt, key, policy=policy)
```

A `max_age` that is invalid gives `ValueError` always.
The jitter limits and `max_age` accept floats with an integer value and numeric strings, for example YAML `3.0`.
A fractional value, a Boolean value or a value that is not a number gives `ValueError`.
A `text_operators` value with a malformed format also gives `ValueError`.
The CLI shows a configuration error.

---

## 11. Free-Text Operators (v1.5.0)

Use `--text-operators TYPE=METHOD,...` to select a method for each entity type in free text.
All other types keep their default anonymization.

| Method | Types | Effect |
|---|---|---|
| `keep` | Dates and other types that you can keep | Keeps the text as it is. |
| `hash` | `MRN`, `HAR`, `ACC_NUM`, `ID`, `CSN_ID`, `US_SSN`, `PHONE_NUMBER`, `PHONE`, `MEDICAL_LICENSE` | Replaces the value with a keyed hash. |
| `acc_num_hash` | `ACC_NUM` only | Replaces the value with the SHA256 hash that is compatible with BigQuery. |

### 11.1 The `keep` Method

- Example: `--text-operators DATE=keep,DATE_TIME=keep,AGE=keep`.
- `DATE` does not include `DATE_TIME`. Add `DATE_TIME` to the list if you must keep it.
- Text that you keep is PHI in the output.
  The operator who runs the job must record the types that the job keeps.
- The recognizers find each type.
  Kept spans are in `anonymizer_results_json` with the operator `keep`.
- When you use `--jitter-required`, a note with no jitter does not fail if you keep `DATE` and `DATE_TIME`.
  The job shifts no date in this condition.

### 11.2 The `hash` Method

- Example: `--text-operators MRN=hash,CSN_ID=hash`.
- The operator calculates `HMAC-SHA256(key, salt | value)`.
- The output has 64 uppercase hexadecimal characters. You cannot decrypt it.
- The same value, salt and key give the same hash in each table and each note of a run.
- The hash is the same for each letter case and for outer spaces in the value.
- `fpe_mode` has no effect on a hashed type.
- Presidio merges adjacent spans of the same type before it anonymizes them.
  An example is two `PHONE` spans that are 2 whitespace characters apart or less.
  The merged span gets one hash. This hash is different from the hash of each cell value.

### 11.3 The `acc_num_hash` Method

- The method must have `--acc-num-salt` and `--acc-num-study-id`.
  The method stops with an error when one of them is not in the input.
- Without this method, accession numbers use HIPS alphanumeric encryption, as for `MRN` and `HAR`.
  An example output is `ldH-GwAu-rUmJ6`.
- **Change in output.** Runs that join against the BigQuery hash must use `ACC_NUM=acc_num_hash`.
  This includes `FREE_TEXT` callers in `tide2-bq`.

### 11.4 Table Cells

A table cell has one entity type.
The caller selects the method with the `method` argument of `anonymize_cell`.
The argument accepts `None` (default), `"hash"` and `"acc_num_hash"`.
`keep` is not a cell method. To keep a column, do not send the column to anonymization.

---

## 12. Score Thresholds (v1.5.0)

A span with a score below the threshold of its recognizer type stays as clear text.
Use these options:

- `--transformer-score-threshold` for transformer spans.
- `--recognizer-score-threshold` for all other spans, such as regex spans and known-value spans.

Follow these rules:

- Each threshold is a number from 0.0 to 1.0. The default is 0.0, which anonymizes all spans.
- A span is a transformer span when its `recognition_metadata["recognizer_name"]` starts with `TransformersRecognizer`.
  The name `CachedResultsTransformerRecognizer` also counts as a transformer span.
- A span with no metadata uses the recognizer threshold.
- The filter runs in `AnonymizerWorker._parse_recognizer_results`.
  It runs after span validation and before overlap resolution.
  A transformer span that the filter drops cannot remove a regex span that overlaps it.
- Notes and `FREE_TEXT` cells use the thresholds. Table cells do not.
- The recognizer stage records each span.
- A span below the threshold is PHI in the output.

**Changes in overlap resolution.**
The anonymizer resolves overlaps one time, after the thresholds.
`RecognizerWorker` (long notes), `LlmRecognizerWorker` (long notes) and the merge step do not use `longest_wins`.
They use `dedupe_exact_spans`.
This function removes duplicate spans only.
Duplicate spans have the same type, the same start and end, and the same threshold class. The span with the highest score stays.
The function keeps spans that overlap, the input sequence and `recognition_metadata`.
The merge step also keeps `recognition_metadata`.
With the default thresholds, the anonymized text is the same.
The recognizer JSON and the merge JSON of a long note keep spans that overlap, as for a short note.
The sequence of the spans has no effect on the overlap resolver.
When a span has an overlap only with a removed span, the resolver keeps that span.

---

## 13. Date Shifts (v1.5.0)

### 13.1 Date Formats

The anonymizer shifts each date by the jitter of the patient, in full days.
The output keeps these parts of the input.

- numeric month or named month, full name or short name, letter case, zero padding, ordinal suffix and separators
- the text `of`, the `T` separator, the time of day and the time zone
- the text around the date

A weekday name turns by the same jitter. The weekday agrees with the shifted date.

**Change in output.** Earlier versions changed the format and shifted some inputs incorrectly.
Compact `yyyymmdd` dates gave an incorrect month and day.
`15 March 2023` gave a date with no day.
The output did not have the time, time zone, weekday and adjacent text.
A date with no year, or with a two-digit year, used the date of the run.
Output with `YYYY-MM-DD` is the same as before.

| Input | Jitter | Output | Note |
|---|---|---|---|
| `2023-03-15` | +5 | `2023-03-20` | The anonymizer shifts ISO dates accurately. |
| `2023-03-15T23:30:00-07:00` | +5 | `2023-03-20T23:30:00-07:00` | The time and the time zone stay. |
| `Wed 2023-03-15` | +5 | `Mon 2023-03-20` | The weekday turns. |
| `20231105` | +5 | `20231110` | Compact `yyyymmdd`. |
| `3/5/2023` | +5 | `3/10/2023` | The padding follows the input. |
| `15 March 2023` | +5 | `20 March 2023` | Day first, with a month name. |
| `March 15th, 2023` | +5 | `March 20th, 2023` | The suffix is in the output only when it is in the input. |
| `12/31/99` | +5 | `01/05/00` | Years 00 to 29 give 20xx. Years 30 to 99 give 19xx. |
| `Feb 29` | +5 | `Mar 5` | No year in the output. The shift uses a leap year. |
| `2023-03` | +112 | `2023-07` | A date with a month only shifts from the 15th. |
| `2023` | +5 | `2023` | A year does not shift if it is the only part. Safe Harbor lets you keep the year. |
| `13/45/2020` | +5 | `[DATE]` | This is not a date. The row is `degraded`. |

A numeric date that starts with 01 to 12 is month first.
A numeric date that starts with 13 to 31 is day first.
A shifted two-digit year keeps two digits.
A result can pass the 29/30 pivot and give a year in the other century.
Use the operator parameter `two_digit_year_pivot` to adjust the pivot.

### 13.2 Jitter Range

- `--min-jitter-days` (default 3, the lowest value) and `--max-jitter-days` (default 180) give the size of the date shift.
- A row with no patient ID gets a random shift in the same range.
- The range changes the derived shift. Keep the range the same in one dataset.
  If you use a different range, the same patient gets a new shift.
- The job uses a `jitter` value from the input as it is.
  A value below 3 days in absolute value masks the date.
- The `jitter` output column is a debug aid. Give it the same protection as `salt.bin` and `key.bin`.
  Remove it before you release the data.

### 13.3 Age Cap (Ages Above 89)

HIPAA Safe Harbor removes ages above 89 and dates that show them.
It lets you use one category for ages of 90 and above.

- Add a `birth_date` column to the input. Use a date or an ISO `YYYY-MM-DD` string.
- The anonymizer adds the jitter first.
  A shifted date after the shifted 90th birthday becomes that birthday.
  The birthday is the birth date plus 90 years plus the jitter.
- All events after the cap get the same date. The sequence and the intervals of the events do not stay.
- The birth date and the earlier dates shift only.
- `--max-age` (default 90, range 1 to 150) sets the birthday.
- The `birth_date` column goes through the stages as `jitter` does.
  The anonymizer output does not have this column.
- Without the column, or with a null value, dates shift with no cap and the job records a warning.
  Such a run does not follow Safe Harbor for patients above 89.

The cap is for each event. A run that uses the cap can show an age above 89.
A date with a year only and a date with no year do not get the cap.
A note with a shifted birth date and a year only can show an age above 89.
Ages in text, such as `92-year-old`, are not dates. The cap does not include them.

| Input | `birth_date` | Jitter | Output |
|---|---|---|---|
| `2023-03-15` | `1930-06-15` | +5 | `2020-06-20` |
| `March 10th, 2023` | `1930-06-15` | +5 | `June 20th, 2020` |
| `2019-01-10` | `1930-06-15` | +5 | `2019-01-15` |
| `2023` | `1930-06-15` | +5 | `2023` |

### 13.4 Day, Month and Year Columns

A table can keep a day, a month or a year in a column for that part.
You cannot shift a part without the full date. Day 31 plus 5 days gives a different result for each month and year.

- Use the entity types `DATE_DAY`, `DATE_MONTH` and `DATE_YEAR` with `anonymize_cell`.
- Give the full date of the row as `full_date`.
  The value can be a date, an ISO string or a timestamp.
- The cell value gives the format only.
  The result is that part of the shifted full date, with the same jitter and age cap.
- Related parts must have the same jitter.
  Give the same patient ID, not null, to each related function.
  Use the same salt, key and jitter range. You can also calculate one jitter for each row and give it as `jitter`.
  Without one of these, each function makes a random shift.
- A year column shifts, because the year can be different after 1 January and with the cap.
- A value that is not that part of `full_date` becomes `[DATE]`.
  A `full_date` that is null gives the same result.
- A `full_date` that is not a date becomes `[DATE_DAY]`, `[DATE_MONTH]` or `[DATE_YEAR]` for the type of the cell.
- These types are for table cells only. The note pipeline does not make them.

| Entity type | Value | `full_date` | Jitter | Output |
|---|---|---|---|---|
| `DATE_DAY` | `30` | `2023-03-30` | +5 | `4` |
| `DATE_MONTH` | `03` | `2023-03-30` | +5 | `04` |
| `DATE_MONTH` | `March` | `2023-03-30` | +5 | `April` |
| `DATE_YEAR` | `2023` | `2023-12-30` | +5 | `2024` |
| `DATE_DAY` | `29` | `2023-03-30` | +5 | `[DATE]` |

---

## 14. ZIP Codes (v1.5.0)

Use `--zip-mode` to select how the anonymizer decreases ZIP codes.
The option is available for anonymizer jobs and pipeline jobs.
In the YAML config, the name is `zip_mode`.

| Mode | Rule | Safe Harbor |
|---|---|---|
| `strict` | Keep the 3-digit prefix when all ZIP codes with that prefix have more than 20,000 persons. If not, use `000`. | Follows 45 CFR §164.514(b)(2)(i)(B). |
| `relaxed` (default) | Keep the 5-digit ZIP when its 3-digit area has more than 20,000 persons. If not, keep the 3-digit prefix of the ZIP. | Does not follow Safe Harbor. |

Use `relaxed` only with an expert determination or a data use agreement that lets you use 5-digit ZIP codes.
Use `--zip-mode strict` for Safe Harbor.

| Input | `strict` | `relaxed` |
|---|---|---|
| `94305-1234` (large area) | `943` | `94305` |
| `02139` (large area) | `021` | `02139` |
| `03601` (area of 20,000 persons or less) | `000` | `036` |
| `2139`, `ABCDE` | `[ZIP_CODE]` | `[ZIP_CODE]` |

- The anonymizer always removes the +4 part.
- The anonymizer masks a value that is not a 5-digit ZIP. The note gets `processing_status` `failed`.
  The anonymizer continues to anonymize the other entities of the note.
- The population data is in `src/tide2/resources/zip3_population_2020.tsv`.
  It has the 2020 Census population of each `ZCTA` (ZIP code area of the Census).
  The table adds the values together by the first three digits.
  The table has 894 prefixes. 18 of them have 20,000 persons or less.
  A prefix that is not in the table counts as 0 persons. The anonymizer uses it as an area of 20,000 persons or less.
- Handle ZIP codes as strings. `02139` keeps its zero.
  The anonymizer does not add zeros to a 3-digit value or a 4-digit value. It masks the value.
  For `anonymize_cell`, an integer ZIP gets zeros to 5 digits and the job records a warning.
- The labels `ZIP` and `ZIPCODE` use the same rule as `ZIP_CODE`.
- A ZIP in an address that the pipeline finds (`LOCATION`, `HOSPITAL`, `VENDOR`) follows the same mode.
  The anonymizer replaces the street, the number and the city. The state stays.
- **Change in output.** Before `--zip-mode`, the anonymizer masked a `ZIP_CODE` entity as `[ZIP_CODE]`.
  It replaced the ZIP in an address with a random ZIP.
  The two cases follow the mode.
- Python: `anonymize_zip("02139", "strict")`.
  For a table cell, use `anonymize_cell(value, "ZIP_CODE", salt, key, policy=AnonymizationPolicy(zip_mode=...))`.

---

## 15. FPE: Encryption That Keeps the Format (v1.5.0)

The ID entity types are `MRN`, `HAR`, `ID`, `CSN_ID`, `US_SSN`, `PHONE_NUMBER`, `PHONE` and `MEDICAL_LICENSE`.
The anonymizer pseudonymizes them with NIST FF1 (AES-256, from the `fastfpe` package).

- Characters that are not alphanumeric stay in position.
- The alphanumeric content changes position in its class and keeps its length.
- The mapping is one-to-one. You can decrypt it. It has no length limit.
  The entity type and the storage as `STRING` or `INT64` have no effect on it.

| Class | Content | Output |
|---|---|---|
| `N` | Digits that do not start with zero | Digits that do not start with zero. `int()` is safe. |
| `Z` | `0` only | `0` |
| `S` | A digit string that starts with `0` | A digit string that starts with `0` |
| `A` | Has a letter | Has a letter |

Content that is shorter than the minimum domain of FF1 uses keyed tables.
The minimum domain is 6 digits or 4 alphanumeric characters.
The job makes the tables when it first uses them.
Longer content uses FF1 and cycle-walks back into the same class.
The `FormatPreservingEncryption.encrypt_int` method and the `decrypt_int` method apply the same mapping to `INT64` values.
The [FPE test guide](fpe_collision_free_verification.md) gives the collision tests and the `INT64` results.

**Important.**
FF1 replaces FF3 because NIST removed FF3.
FPE pseudonyms are not compatible with releases that used FF3, in notes and in tables.
An earlier pseudonym can be the same as a new pseudonym of a different identifier.
Do not mix or join output from before and after this change. Such a join can link records of different persons.
Run again from the source data.

### 15.1 Mode `relaxed` for `INT64` Cells

`AnonymizationPolicy(fpe_mode="relaxed")` (CLI: `--fpe-mode relaxed`) is faster for `INT64` identifiers.
The default is `strict`.

- A cell in this mode uses a keyed permutation of the `INT64` range.
  The permutation is an 8-round Feistel network with FarmHash as the round function.
- The speed is about 2.7 microseconds for each value. FF1 uses 10 to 12 microseconds.
- An `INT64` is a Python `int`.
  An `INT64` can also be a digit string in the `INT64` range.
  The string has no zero at the start and no `-0`.
  Other values use FF1.
- Different `INT64` values always get different outputs.
- A value that is not negative stays not negative. A negative value stays negative.
  Zero does not stay as zero. Do not use `0` as a sentinel in this mode.
- The number of digits changes. About 89% of the outputs that are not negative have 19 digits.
  Use `strict` when the column must keep its length.
- Use one `fpe_mode` for the full dataset.
  A pseudonymized `INT64` ID links a table row to a note only when the two use the same mode.
- This mode is keyed pseudonymization. It is not NIST-approved encryption.
  It gives protection against a person who has the tables but not the key.
  It does not give protection against an attacker who has many known pairs of input and output.
  Use `strict` for that condition.

---

## 16. Cell-Level and Free-Text Anonymization (v1.5.0)

### 16.1 `anonymize_cell`

`anonymize_cell` anonymizes one table value.
It gives the same result as `AnonymizerWorker.process_note` gives for a note with one span.
It does not use a Presidio engine.

- The module loads without the package root of `tide2` and the package root of `tide2.anonymizers`.
  You can bundle it without other modules. The test `tests/test_cell_isolation.py` makes sure of this.
- The `ACC_NUM` type uses HIPS alphanumeric encryption as the default.
- The function also accepts the `hash` method and the `acc_num_hash` method.

### 16.2 `FREE_TEXT` Cells

The entity type `FREE_TEXT` anonymizes a free-text cell, such as a comment column.
It uses spans that earlier recognition stages found.
No recognizer runs for each cell.
`tide2-bq` (the BigQuery UDF repository) uses this path.

- Give the spans as `recognizer_results`. The value is a JSON string or a list.
  The spans index into `str(value)`.
- Dates in the text shift as the `DATE` cells of the row do.
  Use the same `patient_id`, `jitter` and `birth_date`.
- The text follows the same `policy` as a note.
- The result is `None` when the spans are null, are malformed or are not in the text.
  With `ERROR` logs, the job records the text hash and the exception class, and not the text.
  With `DEBUG` logs, the job also records more data about the error. This data can have input text.
  Do not use `DEBUG` logs with PHI.
- When an operator fails, the anonymizer masks its entity only, for example `[PERSON]`.
- The Faker entity types (`EMAIL_ADDRESS`, `WEB`, `URL`, `ORGANIZATION`, `GENETIC_SEQUENCE`) give random values.
  The output for such text is not deterministic.
- The anonymizer reads `entity_type`, `start`, `end` and `score` from a span.
  It also reads `recognizer_name` in `recognition_metadata` to select the score threshold.
  Keep `recognition_metadata`.
  If you remove it, the anonymizer uses `recognizer_score_threshold` for transformer spans.
  When the two thresholds are different, a span can stay as clear text.
  You can remove `analysis_explanation` to make the input smaller.
- The path does not load `presidio_analyzer`, spaCy, torch, pandas, pyarrow or Ray.
- Each process has one `AnonymizerWorker` for each set of settings (`functools.lru_cache(maxsize=4)`).
  There is no batch API.

---

## 17. Visualizer (v1.5.0)

The `tide2-visualizer` command shows the input text and the anonymized text side by side.
Run `tide2-visualizer [path]`.

- The visualizer reads Parquet only. It does not read BigQuery input, CSV input or JSON input.
- It finds the pipeline stage folders (`merged_recognizer_output`, `llm_recognizer_output`, `recognizer_output`, `anonymizer_output` and `transformer_output`).
  It aligns the rows by `row_id`.
- It also reads a Parquet directory or one Parquet file.
- The interface has `< Prev` and `Next >` buttons, a `Jump to note #` field and a `Find by Row ID / Hash` field.
- A table shows the PHI entities that the visualizer finds: type, text excerpt, character range, recognizer name and score.
  You can filter by entity type and recognizer.
- The `google-cloud-bigquery` dependency and the `db-dtypes` dependency are not in the package.

---

## 18. Dependencies, Documentation and Other Items (v1.5.0)

- **Dependencies.** `pandas` is not a core dependency.
  It is in the `evaluation` extra and the `dev` extra.
  The core TSV loaders use the Python `csv` module.
  The core modules import `pandas` only when they must have it.
- **Hospital and institution names.** The anonymizer keeps the letter case of the input text.
  It handles Title Case, ALL CAPS, lowercase and acronyms.
  It keeps the whitespace at the start and the end of the span.
- **FPE.** Section 15 gives the data about FF3 and FF1.
- **Documentation.** The README has a guide for runs on many machines and updated sections for each new function of version 1.5.0.
  The pipeline notebook uses `write_visualizer_json`.

---

## 19. Upgrade Steps

Do these steps before you run live pipelines with version 1.5.0.

- [ ] Remove `reassembly` tasks from all orchestrator workflows.
- [ ] Replace the input column `patient_uid` with `patient_id`.
- [ ] Save all input datasets in Parquet format.
- [ ] Make a unique `row_id` for each input row.
- [ ] Adjust the paths that have a stage folder name with `01_` or `02_`.
- [ ] Run again from the source data. Do not join FPE output from version 1.3.1 or earlier with new output.
- [ ] Examine the date output for formats that earlier versions shifted incorrectly.
- [ ] Select `--zip-mode strict` if you must follow HIPAA Safe Harbor.
- [ ] Add `--text-operators ACC_NUM=acc_num_hash` if you join against the BigQuery accession hash.
- [ ] Add a `birth_date` column if you must use the age cap.
- [ ] Use the same `fpe_mode`, `zip_mode`, jitter range and `max_age` for the full dataset.
- [ ] Use `--id-column text_hash` if you must keep the earlier checkpoint key.
- [ ] Remove the `jitter` column from the data before you release it.

---

## 20. List of Changes

| Pull request | Change | Section |
|---|---|---|
| #62 | Token windows for NER, actors with no supervisor and hardware autotuning | 3 to 9 |
| #93 | The pipeline notebook uses `write_visualizer_json` | 18 |
| #95 | Hospital and institution names keep their letter case | 18 |
| #97 | Documentation update and guide for runs on many machines | 18 |
| #101 | Checkpoint on `row_id` for standalone stages and the `--id-column` flag | 8 |
| #102 | Visualizer with Parquet input | 17 |
| #103 | `pandas` removed from the core dependencies | 18 |
| #106 | `anonymize_cell` | 16 |
| #107 | Date format correction and jitter range | 13 |
| #108 | FF1 FPE replaces FF3 | 15 |
| #111 | Date cap at the 90th birthday | 13 |
| #112 | Day, month and year columns | 13 |
| #113 | ZIP code anonymizer | 14 |
| #114 | `FREE_TEXT` cells with precomputed spans | 16 |
| #115 | You must select the `ACC_NUM` hash | 11 |
| #116 | Keep selected entity types (this is `text_operators` with `keep`) | 11 |
| #117 | Mode `relaxed` for `INT64` cells | 15 |
| #119 | Anonymization policy | 10 |
| #120 | Keyed hash method for ID types | 11 |
| #123 | Score thresholds | 12 |
