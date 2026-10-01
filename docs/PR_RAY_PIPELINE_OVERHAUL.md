# PR: `feat(runner): rework the ray pipeline for token-accurate ner, direct actors, and hardware autotuning`

Combined PR for the five-branch stack: `fix/transformer-actor-token-windowing` →
`perf/ray-discrete-stages` → `perf/direct-worker-actors` → `feat/hardware-aware-defaults` →
`perf/streamed-run-pipeline`. Base: `main`.

---

## Why

Three problems, each fixed by one part of this stack:

1. **Correctness.** Long notes were only partially de-identified — the chunker sized windows
   from a character approximation and the transformer truncated at ~512 tokens, so PHI in the
   tail of a note was never sent to the model. A CUDA OOM was also unrecoverable: recovery code
   sat after `return` and never ran, and the failed forward's tensors stayed pinned.
2. **Throughput.** The pipeline ran at ~220s on the 1,381-note SHIELD corpus, with a supervisor
   actor tier between Ray Data and every worker, per-batch `ray.get` timeouts that silently
   dropped batches, and redundant full-dataset materializations.
3. **Portability.** Every sizing knob was a literal tuned for one machine (16 CPU, 1×L4). A
   224-CPU node still got 14 recognizer actors; a 4-CPU box got a config that deadlocks at `0/1`.

## What changed

### 1. Token-accurate NER, and OOM recovery that works (`fix`)

- **One length authority.** `MODEL_MAX_LENGTH` (pinned per model) replaces three inconsistent
  sequence sizers — the tokenizer sentinel (`1e30`), RoBERTa's off-by-two `514`, and the
  `len(text)//4` character estimate. Resolution fails loudly instead of falling back.
- **Window, don't truncate.** Whole notes go to the GPU actor, which tokenizes → windows →
  forwards per note via a shared `TransformerCore` primitive (`plan_windows` /
  `tokenize_and_window`). The char-approximation `flat_map` chunker is deleted; a coverage guard
  raises (not `assert`, so `-O` can't strip it) if windows don't span the note. On a 1.98M-char
  note, entities are now detected out to the final character.
- **Reassembly folded away.** `BIOAggregationActor` aggregates → dedups overlapping spans →
  emits document-ready `recognizer_results_json`, so the reassembly stage, actor and groupby are
  gone. Dedup uses a fixed-schema key, so it is deterministic regardless of JSON key order.
- **OOM recovery.** Halve-and-retry moved into the `except` clause (it was unreachable);
  device tensors released in `try/finally`; `empty_cache()` runs outside the handler so CPython
  can't pin the failed forward's frames via `sys.exc_info()`; the recursive split is now an
  iterative work list. `reduce-overhead` `torch.compile` is removed from the product path.
- **Batch sizing model deleted.** With a working backstop, an analytical VRAM predictor can only
  be right (wasted) or wrong (backstop fixes it). Forward runs at an operator-set
  `--gpu-batch-size` over length-sorted windows.
- **Per-model config keys** `DTYPE` and `MODEL_MAX_LENGTH`, validated on the driver with
  actionable errors; DeBERTa-v3 and OpenMed-PII-SuperClinical configs added, with GDPR Art-9
  labels routed to `LABELS_TO_IGNORE`.

### 2. Discrete sequential stages with Ray parity (`perf`)

`run_pipeline` runs three strict stages — GPU transformer → CPU recognizer → CPU anonymizer —
each its own Ray Data execution with a Parquet boundary. Block starvation and a redundant
full-dataset `count()` pass are removed, `use_datasource_v2 = False` restores Parquet reader
concurrency, `override_num_blocks` is exposed through the CLI, and log dedup stops stdio
saturation across parallel workers. `patient_uid` null/NaN handling is hardened so the literals
`"nan"` / `"none"` can't leak into crypto salt or jitter derivation. **220s → 144.4s.**

### 3. Direct worker actors — the supervisor tier is gone (`perf`)

`RecognizerWorker`, `AnonymizerWorker` and `LlmRecognizerWorker` are plain callable classes that
Ray Data's `map_batches` drives directly; the three supervisor classes remain importable as
deprecating shims. Per-batch `ray.get(timeout=…)` — which killed a worker and dropped the whole
batch at 120s — is replaced by Ray Data's execution-level `NoProgressGuard` (600s default,
`--no-progress-timeout`), which fails the job loudly instead of silently discarding notes.
`--batch-timeout` becomes a deprecated no-op. Actor count per CPU stage: 28 → 14.
**144.4s → 123.0s** (recognizer −39%, anonymizer −22%), with row-, entity- and per-field parity
verified against the baseline.

### 4. Hardware-aware defaults (`feat`)

New `src/tide2/runner/hardware.py` is the single tuning-purpose reader of `ray.cluster_resources()`
/ `ray.nodes()` / `psutil`; it replaces four ad-hoc probes and the inline literal block in
`run_pipeline`.

- **A recommendation only ever fills a knob the caller left unset.** Nothing passed by kwarg, CLI
  flag or YAML is overridden, and the resolved table is logged per knob with `USER` / `auto` /
  `default` provenance.
- **Profiles on two axes**, matched on *node* shape (never cluster totals):

  | | `cpu ≤ 4` | `4 < cpu < 64` | `cpu ≥ 64` |
  |---|---|---|---|
  | **GPU** | `small-box-gpu` | `gpu-workstation` | `gpu-server` |
  | **No GPU** | `small-box-cpu` | `cpu-only` | `large-cpu` |

  Heterogeneous clusters classify as `unknown` and recommend *nothing*, rather than averaging two
  machine shapes into a config that fits neither.
- **Small-box pairing by construction:** fractional CPUs *and* `enable_checkpoint=False` are
  emitted from one function, because each alone is known not to clear the `0/1` deadlock.
- **Measured batch sizes only.** `MEASURED_MODELS` is keyed on `(model, gpu_family)`; unmeasured
  hardware falls back to the profile default rather than extrapolating a VRAM envelope, because
  guessing high OOMs mid-run. Model aliasing requires confirmed checkpoint identity, never name
  similarity (asserted by test).
- Opt out with `--no-hardware-autotune` / `hardware_autotune: false` / `hardware_autotune=False`.

On the reference box every recommended value is byte-identical to the literal it replaces
(asserted as an exact-equality test) — **121.9s, parity, which is the correct result.** A CPU-only
before/after run also matched knob-for-knob (316.7s vs 320.2s).

Four real defects surfaced and were fixed here: recommended `no_progress_timeout_s` was dead code
(and this silently discarded the pre-existing `--no-progress-timeout` flag); YAML
`hardware_autotune: false` was ignored; YAML could overwrite an explicitly typed negative flag
(same latent bug for `--no-checkpoint`, `--no-transformer`, `--no-recognizer`, `--no-anonymizer`);
and `run_pipeline` mutated the caller's kwargs dicts.

### 5. Opt-in streamed execution mode (`perf`)

`execution_mode="streamed"` chains the three stages into a *single* Ray Data execution, removing
the driver-side `01_transformer_input` write and the `02_` / `04_` Parquet round-trips.
**`discrete` remains the default and is behaviourally unchanged** — the shared stage-construction
code was extracted as pure code motion and is asserted identical by test.

The stages **pipeline; they do not fuse** — all three are actor pools, and Ray Data only fuses
`TaskPool→{TaskPool,ActorPool}`. The win is that the anonymizer consumes blocks the recognizer has
finished while the transformer is still working, with nothing serialized in between.

- Static `StageColumns` contracts validated up front instead of file sniffing (`ds.schema()`
  mid-plan would execute upstream operators and defeat the pipelining).
- `check_streamed_admission` sums each pool's *minimum* size against *per-node* capacity and
  raises with a breakdown, rather than letting the run park at `0/1`.
- Refusals vs fallbacks chosen by whether the user can be silently wrong: `enable_checkpoint`,
  `llm_recognizer_mode="merge"` and `produce_visualizer_json` **fall back** to discrete with a
  warning; multi-node clusters and ≤4-CPU nodes **raise**.
- Row reconciliation uses `pyarrow.dataset(...).count_rows()`, never `ds.count()` (which would
  re-execute the whole plan). Zero rows out of a non-empty input is a hard error.
- The sink ends in a projection that keeps raw `note_text` out of the output directory.

**PHI note:** in streamed mode `note_text` lives in the Ray object store for the run instead of
in Parquet between stages, and may spill. That is in scope only while the spill directory is
**local** storage — a network mount would take PHI off the host. Documented in the README's
Security Considerations.

**Streamed benchmarks are outstanding** (Plan §12: SHIELD, 3 reps, ≤120s merge gate). Since
streamed is opt-in and `discrete` is untouched, this does not block the rest of the stack.

### 6. Review Comments Resolution Updates

- **Pinned Base Image Digest:** Pinned base image to `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04@sha256:14d94b039cb94bbd5da559f303b46bc4b0d5d6c24ab1a9d7b186e566ed3400dc` in `Dockerfile`.
- **Model-to-Presidio Entity Mapping:** `format_note_entities` and `BIOAggregationActor` apply `MODEL_TO_PRESIDIO_MAPPING` and filter `LABELS_TO_IGNORE` so models with lower-case tags (e.g. OpenMed) emit canonical Presidio entity types.
- **Standalone TransformersRecognizer Parity & Narrow Fallback:** Refactored `TransformersRecognizer._get_ner_results_for_text` to eliminate character heuristic (`len(text) // 4`) and character-based chunking; inference now matches `TransformerInferenceActor` (token-accurate windowing, BIO aggregation, span IoU deduplication). Narrowed `_infer_raw_tokens` fallback so runtime/CUDA OOM errors from `forward_windows` propagate directly rather than being swallowed into the character fallback.
- **Row ID Semantics & Transformer Input Preservation:** In `_build_recognizer_input_from_transformer`, all transformer passthrough columns (`patient_identifiers`, `patient_uid`, `jitter`, `row_id`) are preserved, and fallback lookup for missing identifiers joins on `row_id` (`sha256(text_hash:patient_uid)`) rather than `text_hash` to keep distinct patient contexts separate.
- **Explicit Intercept on Removed Reassembly:** Running `tide2-runner run reassembly` halts immediately with an explicit, actionable error explaining the removal.
- **Clean Signatures and Explicit Deprecation Errors:** Dead supervisor arguments (`batch_timeout`, `timeout`, `worker_num_cpus`) and deprecated `run_transformer` kwargs (`chunk_size`, `flat_map_cpus`, `compile_model`, `compile_cache_path`, `pre_chunked`, `short_seq_budget`) are captured via `**kwargs`, emitting a `DeprecationWarning` and failing fast with `ValueError`.
- **LLM Recognizer Metadata Passthrough:** `LlmRecognizerWorker` retains `note_text` and copies passthrough metadata columns (`patient_uid`, `row_id`, `jitter`, `patient_identifiers`) to support downstream anonymizers in streamed mode.
- **Nullable Scalar Hardening (`pd.NA` Handling):** Reordered null checks in `LlmRecognizerWorker` (`llm_recognizer.py:217`) and `RecognizerWorker` (`recognizer.py:304, 365, 370`) to test `is_null(...)` prior to boolean truthiness, preventing `TypeError: boolean value of NA is ambiguous` on nullable Arrow/pandas columns.
- **Offline Mode Derived from Download Flag:** Configured `TransformerRayActor` to pass `local_files_only=not allow_huggingface_download` to `TransformerCore`, ensuring explicit download requests function on fresh hosts without cached models.
- **Out-of-Scope OOM Cache Clearing:** Deferred `torch.cuda.empty_cache()` execution outside the `except RuntimeError as e:` block in `_forward_windows_with_retry`, allowing CPython to release traceback frames and unpin failed forward tensors before cache reclamation.
- **Token-Bound GPU Family Matching:** Replaced substring containment in `extract_gpu_family` with word-boundary regex (`\b{family}\b`), preventing `NVIDIA L40` and `L40S` from misclassifying as `L4`.

## Compatibility

Breaking changes and behaviour changes worth knowing:

| Change | Before | After |
|---|---|---|
| Reassembly stage/job | Separate stage + groupby | **Removed.** Folded into transformer stage. `tide2-runner run reassembly` halts with explicit error. |
| Long-note coverage | Silently truncated past ~512 tokens | Full note windowed — **output differs, and that is the fix** |
| `batch_timeout` | 120s; kills worker, drops the batch, job continues | Deprecated; halts with explicit error explaining `NoProgressGuard` replacement |
| Deprecated kwargs | Silently ignored or passed to dead code | Captured via `**kwargs`; emits `DeprecationWarning` and raises `ValueError` |
| Unset sizing knobs | Literals tuned for one 16-CPU / 1×L4 box | Sized from the detected profile; identical on that box |
| `--chunk-size` | Char-based chunker | Deprecated; halts with error directing to `--chunk-overlap` |
| Supervisor classes | Ray actors | Deprecating in-process shims; `*WorkerActor` for `.remote()` callers |
| Startup logs | Resolved CPU config | `Detected:` / `Profile:` header + per-knob provenance table |

## Performance

SHIELD, 1,381 notes (~360k content tokens), 1×NVIDIA L4 / 16 CPU / 64 GB, `discrete` mode:

| | `main` | Discrete stages | Direct workers | Autotuned |
|---|---:|---:|---:|---:|
| Stage 1 (transformer) | — | 57.7s | 57.3s | 57.4s |
| Stage 2 (recognizer) | — | 38.6s | 23.5s | 23.5s |
| Stage 3 (anonymizer) | — | 30.4s | 23.7s | 22.8s |
| **Total** | **~220s** | **144.4s** | **123.0s** | **121.9s** |

**~220s → ~122s end to end (−45%).**

## Docs

README gains *Hardware autotuning*, *Execution modes: discrete vs streamed*, a Security
Considerations bullet on object-store residency, and an explicit note that the *"Why small boxes
deadlock"* section is about `discrete` mode. Google-style docstrings updated across the new public
API; `runner_config_example.yaml` documents `hardware_autotune`, `execution_mode` and
`no_progress_timeout_s`. A benchmark harness (`scripts/benchmark_stage_throughput.py`) times any
single stage through the real CLI path.

## Testing

- `uv run pytest`: **1002 passed**.
- `uv run pre-commit run --files …`: clean on every changed file.
- `uv run ty check`: clean on changed files; the 2 remaining `local_runner.py` diagnostics were
  verified by worktree comparison to pre-exist on `main`.
- `uv sync --extra docs --extra llm && uv run make docs`: builds cleanly.
- GPU-gated suites (`tests/oom_verification.py`, `test_transformer_oom_recovery_gpu.py`,
  `test_transformer_windowing_gpu.py`) verified on a real L4: 0 MB drift across passes and after
  an OOM-and-recovery cascade.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
