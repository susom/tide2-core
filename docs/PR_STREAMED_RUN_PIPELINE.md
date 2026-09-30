# PR: `perf(runner): add in-memory stage chaining to run_pipeline`

## PR Metadata

- **Title:** `perf(runner): add in-memory stage chaining to run_pipeline`
- **Head Branch:** `perf/streamed-run-pipeline`
- **Base Branch:** `feat/hardware-aware-defaults` *(Note: targeted against `feat/hardware-aware-defaults` instead of `main` because it builds on the unmerged Plan 1 + Plan 2 stack. Retarget down the stack as each one merges.)*
- **Commit Type:** `perf` (patch version bump, changelog-visible)
- **Breaking Changes:** None (no breaking-change notation; the new mode is opt-in and the default path is untouched).

---

## Description

This PR implements Plan 3 from [docs/plan_03_streamed_run_pipeline.md](docs/plan_03_streamed_run_pipeline.md). It adds an **opt-in** `execution_mode="streamed"` to `LocalJobRunner.run_pipeline` that chains transformer → recognizer → anonymizer into a *single* Ray Data execution, removing the driver-side `01_transformer_input.parquet` write and the `02_` / `04_` Parquet round-trips between stages.

Today `run_pipeline` in `discrete` mode runs each stage as its own Ray Data execution with a Parquet boundary between them. That boundary is load-bearing for production: it is what lets the GPU stage run on one machine and the CPU stages elsewhere as independent jobs, and it is what row-level resume (`enable_checkpoint`) is keyed to. But for a single-box run — development, benchmarking, a one-shot batch — the boundary is pure overhead: the driver serializes the whole input to Parquet, then each stage re-reads and re-writes the corpus.

The central contract is that **`discrete` stays the default and stays behaviourally unchanged.** Nothing in this PR alters the discrete code path's semantics; the stage-construction code it shares with the streamed path was extracted as pure code motion and is asserted identical by test.

### A note on terminology: the stages *pipeline*, they do not *fuse*

Ray Data only fuses `TaskPool→TaskPool` and `TaskPool→ActorPool` operators. All three stages here are **actor pools**, so no fusion occurs and none is claimed. What the streamed mode buys is *pipelining*: the anonymizer starts consuming blocks the recognizer has already finished while the transformer is still working on later blocks, and no block is ever serialized to Parquet in between. The word "fused" is deliberately absent from the code, the docs, and this writeup.

### Key Architectural Changes

1. **Four `build_*_stage` methods on `LocalJobRunner` (pure code motion)**:
   - `build_transformer_stage`, `build_recognizer_stage`, `build_llm_recognizer_stage`, `build_anonymizer_stage` each take a `Dataset` and return a `Dataset` with one (or, for the transformer with BIO aggregation, two) `map_batches` operators appended.
   - No new classes and no builder pattern — they are plain methods, so the discrete `run_*` wrappers call them and the streamed path calls them, from one definition.
   - A shared `_actor_pool(num_actors, pool_min_size)` helper encodes the one rule both modes need: `size=` (fixed) when no minimum is given, `min_size=min(pool_min_size, num_actors), max_size=num_actors` (autoscaling) when one is.
   - The builders never touch the `DataContext` — that is the caller's job, and it has to be, because `configure_data_context` is global and last-call-wins. A chained plan configures it **once**; the discrete path still configures per stage exactly as before.

2. **Static column contracts replace file-sniffing**:
   - `detect_columns` reads a Parquet footer to decide which optional columns exist. There is no file to sniff mid-plan, and calling `ds.schema()` on a lazy mid-plan `Dataset` can execute upstream operators and silently defeat the pipelining the mode exists for.
   - Instead, a frozen `StageColumns(requires, optional, produces)` per stage mirrors the actual actor outputs, and `validate_stage_columns` walks them in order against the **source** dataset's columns — once, before a single operator is chained — raising a `ValueError` that names the offending stage and the missing columns.
   - `available_after` models pass-through explicitly: `(upstream & (requires | optional)) | produces`. This matters — `row_id`, `patient_uid` and `jitter` are forwarded by the actors but are not listed under any stage's `produces`, and the sink projection is computed from the walk's result so they survive to the output.

3. **CPU admission check instead of a `0/1` hang**:
   - A chained plan holds every operator's pool resident at once. Over-subscribe the node and Ray's `ReservationOpResourceAllocator` parks the execution at `0/1` with `backpressured:tasks(ResourceBudget)` — a hang, not an error.
   - `check_streamed_admission` sums each pool's **minimum** size (the pools autoscale, so the minimum is what must be admitted) and compares it against **per-node** capacity from `ray.nodes()`, never `ray.cluster_resources()["CPU"]` — a 4×16-CPU cluster has 64 cluster CPUs and no node that can host the plan.
   - It raises, with the per-pool breakdown in the message, rather than letting the run hang. It is documented as a heuristic; `no_progress_timeout_s` remains the real backstop.

4. **Refusals and fallbacks, chosen by whether the user can be silently wrong**:

   | Condition | Behaviour | Why |
   |---|---|---|
   | `enable_checkpoint=True` | **Fall back** to discrete, warn, corrected manifest | Row-level resume is keyed to the stage boundaries streamed removes. Honouring the request means running discrete. |
   | `llm_recognizer_mode="merge"` | **Fall back** to discrete, warn | Merge needs both the transformer and LLM outputs joined on `row_id`; chaining would re-run GPU inference on a shared prefix. |
   | `produce_visualizer_json=True` | **Fall back** to discrete, warn | See *Deviations* below. |
   | Multi-node cluster | **Raise** | A chained plan is single-node by construction; running it anyway would be silently slow and non-obvious. |
   | Node with ≤4 CPUs | **Raise** | Three resident pools cannot fit; this is the `0/1` deadlock. `discrete` is fully supported there and always has been. |

5. **Row reconciliation with `pyarrow`, never `ds.count()`**:
   - After the sink writes, the streamed path counts output rows with `pyarrow.dataset(...).count_rows()` and compares against the input row count. `ds.count()` would **re-execute the entire chained plan** — the exact mistake that cost ~32s and was fixed in `86e0bb7`.
   - Zero rows out of a non-empty input is a hard `RuntimeError` (every batch failed). A drop of more than 1% warns.

6. **Sink hygiene**: the streamed plan ends in a `select_columns` projection computed as `FINAL_OUTPUT_COLUMNS & <columns available after the last stage>`. Raw `note_text` is deliberately absent from `FINAL_OUTPUT_COLUMNS`, so the projection is a guard that keeps plaintext out of the output even if an actor starts forwarding it.

7. **Opt-in on all three entry paths**: `--execution-mode {discrete,streamed}` (CLI, `default=None`), `execution_mode: discrete` (YAML, documented), `run_pipeline(execution_mode=...)` (Python). An unrecognized value raises rather than being coerced.

---

## Deviations from the Plan

Two, both deliberate and both documented in code:

1. **§9's mode-independent visualizer post-pass is not implemented.** It is not reachable without changing `AnonymizerActor.process_batch`, which is out of scope here: the anonymizer **drops** `recognizer_results_json` from its output, and the plan's suggested alternative — reconstructing recognizer spans from `anonymizer_results_json` — does not work, because those offsets are in the *anonymized* text, not the original. `produce_visualizer_json=True` therefore falls back to discrete with a warning and a corrected manifest, which produces the correct artifact by the supported route.

2. **Skip semantics raise through column-contract validation rather than a missing-directory check.** Asking for a stage whose inputs no upstream stage produces (e.g. recognizer-only with no transformer) fails in `validate_stage_columns` — earlier than the plan describes and with a message naming the stage and the missing columns, but from a different site.

---

## Benchmarks

**Not yet run.** The Plan §12 protocol (SHIELD corpus, 3 repetitions, medians, against the 144.4s Plan-2 baseline) and the resulting ≤120s merge / no-merge decision are outstanding. This PR should not merge until that gate is evaluated on the reference box.

The performance model this is built against, from the plan: baseline **144.4s**; merge threshold **≤120s**; hard floor **~73s** (the GPU stage is irreducible and the CPU stages cannot fully hide behind it).

---

## Compatibility

### Hard Breaks

None. `execution_mode` defaults to `"discrete"`, and every pre-existing call signature, CLI flag, and YAML key is accepted unchanged. A caller who never passes `execution_mode` gets byte-for-byte the pre-PR behaviour.

### Silent Semantic Changes

None in `discrete` mode — that is invariant 1 and a discrete regression blocks this PR.

Within `streamed` mode (all new behaviour, reachable only by opting in):

| Aspect | `discrete` | `streamed` |
|---|---|---|
| Intermediate artifacts | `01_transformer_input`, `02_transformer_output`, `04_recognizer_output`, `06_anonymizer_output` | `06_anonymizer_output` only (or `04_recognizer_output` when the anonymizer is skipped) |
| Resume (`enable_checkpoint`) | Supported | Not supported — falls back to discrete |
| Multi-machine split | Supported | Refused (raises) |
| ≲4-CPU boxes | Supported | Refused (raises) |
| `note_text` residency | Written to Parquet between stages | Held in the object store for the whole run; may spill to Ray's **local** spill directory |
| Manifest | Per-stage manifests | `{execution_mode, total_elapsed_seconds, input_rows, output_rows, dropped_rows, output_dir, operator_stats}` |
| Output values | — | Equivalent, not byte-identical: `processing_timestamp` and row order differ by construction |

### Unchanged

- All existing kwargs, CLI flags, and YAML keys.
- `scripts/benchmark_stage_throughput.py` needs no change — it drives the CLI by subprocess and reads no manifest keys.
- `runner/fault_tolerance.py` needs no change — `configure_data_context` already accepts every parameter the streamed path passes and already sets `use_datasource_v2 = False`.

---

## PHI / Security Considerations

The host is the trust boundary, and `streamed` moves work *inside* it, not across it — but it changes where plaintext lives during a run. Raw `note_text` stays in the Ray object store for the duration of the chained plan instead of being written to the output directory between stages, and under memory pressure Ray may spill those objects to its spill directory.

That is in scope as long as the spill directory is **local storage**. A network mount, NFS/SMB share, or object-store FUSE path takes PHI off the host, which the discrete path's Parquet boundaries never did implicitly. This is called out in the README's Security Considerations section.

The sink projection (point 6 above) is the second half of this: `note_text` cannot reach the output directory even if an actor starts forwarding it.

---

## Documentation

- **README**: new *"Execution modes: discrete (default) vs streamed"* section before the deadlock section, with the comparison table, the pipelining-vs-fusion explanation, the CPU-admission and memory notes, and a CLI example. The existing *"Why small boxes deadlock"* section is now explicitly prefixed as being about **`discrete` mode** — which is fully supported on ≲4-CPU boxes and always has been — since `streamed` is refused there outright. New Security Considerations bullet on object-store residency and local-only spill.
- **Docstrings**: `run_pipeline` gains a section on the two modes and their trade-offs; `validate_stage_columns`, `check_streamed_admission`, and the four builders document their contracts, including why the admission check is a heuristic and why the builders must not touch the `DataContext`.
- **`runner_config_example.yaml`**: documents `execution_mode` and why `discrete` is the production default.

---

## Testing & Quality Checklist

- [x] `uv run pytest tests/test_stage_builders.py`: 10 passed — every builder's `map_batches` kwargs asserted identical to the pre-extraction discrete call, the `_actor_pool` clamping rule, and that no builder touches the `DataContext`.
- [x] `uv run pytest tests/test_streamed_pipeline_plan.py`: 28 passed — column contracts, admission (over-subscribed / small-box / multi-node / largest-node-not-cluster-total), plan shape (3 operators, single sink, projection drops `note_text` and keeps `row_id`, checkpoint config cleared, skip semantics, LLM-only, source projection), and every fallback rule. The `Dataset` double's `schema()` and `count()` raise, which is what proves the plan is never executed during construction.
- [x] `uv run pytest tests/test_pipeline_parity.py` (integration, real Ray, stub transformer): 4 passed — discrete and streamed produce equal row sets and per-`row_id` equal `anonymized_note_text` / `entity_count` / `processing_status` / key-order-normalized `anonymizer_results_json`; matching entity totals by type; streamed writes no `01_` / `02_` / `04_` intermediates; `note_text` never reaches the sink; a total anonymizer failure is a hard error.
- [x] **Discrete regression suite** (`test_cpu_knobs`, `test_deprecated_knobs`, `test_discrete_stages_parity`, `test_hardware`, `test_row_id_generation`, `test_fault_tolerance`): 179 passed.
- [x] `uv run pytest`: **1002 passed**.
- [x] `uv run pre-commit run --files ...`: all hooks clean on every changed file.
- [x] `uv run ty check`: clean on the changed files. The 2 remaining diagnostics in `local_runner.py` (`run_transformer`, lines 1431 / 1507) were verified by stash comparison to pre-exist on the unmodified tree.
- [x] No breaking-change notation in title or body.
- [ ] **§12 benchmark protocol and the ≤120s merge decision — outstanding.**

### Defect found during verification

The sink projection was initially computed as `FINAL_OUTPUT_COLUMNS & ANONYMIZER_STAGE_COLUMNS.produces`, which silently dropped `row_id`: the actors forward it, but no stage lists it under `produces`, so it is invisible to a `produces`-only intersection. The parity test caught it as a `KeyError: 'row_id'`. `validate_stage_columns` now returns the column set its walk arrives at — which models pass-through correctly — and the sink projects against that. A regression assertion was added to `test_sink_projection_drops_note_text`.
