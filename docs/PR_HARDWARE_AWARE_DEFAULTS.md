# PR: `feat(runner): recommend per-stage settings from detected hardware`

## PR Metadata

- **Title:** `feat(runner): recommend per-stage settings from detected hardware`
- **Head Branch:** `feat/hardware-aware-defaults`
- **Base Branch:** `perf/direct-worker-actors` *(Note: targeted against `perf/direct-worker-actors` instead of `main` because it builds upon the 28 unmerged commits from Plan 1. Retarget to `main` once `perf/direct-worker-actors` merges.)*
- **Commit Type:** `feat` (minor version bump, changelog-visible)
- **Breaking Changes:** None (strict adherence to Conventional Commits; no breaking-change notation).

---

## Description

This PR implements Plan 2 from [docs/plan_02_settings_recommender.md](docs/plan_02_settings_recommender.md). It replaces the four independent, ad-hoc hardware probes scattered through `local_runner.py` with a single recommender module, [src/tide2/runner/hardware.py](src/tide2/runner/hardware.py), that detects the cluster shape once, matches it to a named profile, and returns per-stage setting recommendations.

Before this PR, the hardcoded literals in `run_pipeline` (`num_transformer_actors=3`, `num_gpus=0.33`, `batch_size=512`, `TARGET_NODE_CPU_ACTORS=14`, …) were tuned for exactly one machine — a 16-CPU box with a single NVIDIA L4. On any other shape they were silently wrong: a 224-CPU node still got 14 recognizer actors, and a 4-CPU box got a configuration that deadlocks at `0/1`. The recommender makes the machine shape an input rather than an assumption.

The central contract is that **a recommendation only ever fills a knob the caller left unset**. Nothing a user passes — by Python kwarg, CLI flag, or YAML key — is ever overridden, and the resolved table is logged with a `USER` / `auto` / `default` provenance marker per knob so a run is self-documenting.

### Key Architectural Changes

1. **Single Tuning Module (`runner/hardware.py`)**:
   - `hardware.py` is now the only place in the package that reads `ray.cluster_resources()`, `ray.nodes()`, or `psutil` for the purpose of sizing settings. (`runner/utils.py` still reads cluster resources, but purely to log them.)
   - Pure functions over frozen dataclasses: `NodeShape`, `HardwareFacts`, `MeasuredModelEntry`, `Recommendations`, `SettingEntry`, and `Applied`. Detection is the only impure step; `classify_profile`, `recommend_settings`, `apply_recommendations`, and `render_settings_table` are all deterministic given facts, which is what makes the profile matrix testable without a cluster.
   - Public API re-exported from `tide2.runner`: `detect_hardware`, `classify_profile`, `recommend_settings`, `apply_recommendations`, `render_settings_table`, `recommend_object_store_gb`, and the five dataclasses.

2. **Profile Matrix on Two Axes**:
   - `classify_profile` matches on `(gpu_present, cpu_class)` of the **node** shape, never on cluster totals. A 4-node cluster of 16-CPU machines is a `gpu-workstation`/`cpu-only` shape 4 times over, not one 64-CPU `large-cpu` box.
   - Per-node facts drive per-actor sizing (`worker_num_cpus`, `num_gpus`, `gpu_batch_size`); cluster totals drive counts (`num_actors`, `num_transformer_actors`), because a Ray Data actor pool is cluster-wide.
   - A heterogeneous cluster (or missing node facts) classifies as `unknown` and recommends **nothing at all**, rather than averaging two machine shapes into a configuration that fits neither.

   | | `cpu <= 4` | `4 < cpu < 64` | `cpu >= 64` |
   |---|---|---|---|
   | **GPU present** | `small-box-gpu` | `gpu-workstation` | `gpu-server` |
   | **No GPU** | `small-box-cpu` | `cpu-only` | `large-cpu` |

3. **Consolidation of the Four Detectors**:

   | Site | Before | After |
   |---|---|---|
   | `_init_ray` object store | inline `psutil` probe, `try/except ImportError: pass` | `recommend_object_store_gb(detect_hardware())` |
   | `_auto_num_actors` | `int(cluster_cpu * 0.45)` | profile value; `fraction=` override retained |
   | `_resolve_transformer_resources` | 3 separate `ray.cluster_resources()` reads | one `detect_hardware()`, recommendations with legacy fallbacks |
   | `run_pipeline` inline GPU block | `if available_gpus > 0:` + 7 `setdefault` literals, plus `TARGET_NODE_CPUS` comparisons for stages 2 and 3 | deleted; replaced by `apply_recommendations` |

   `TARGET_NODE_CPUS` and `TARGET_NODE_CPU_ACTORS` are deleted (nothing imported them).

4. **Small-Box Pairing Enforced by Construction**:
   - `small-box-cpu` / `small-box-gpu` emit fractional CPUs **and** `enable_checkpoint=False` together, because each alone is known not to clear the `0/1` deadlock (README → *"Why small boxes deadlock"*). They are emitted from one function so they cannot drift apart.
   - These profiles also raise `no_progress_timeout_s` from 600s to 1200s, since a cold model load on a small box can legitimately exceed the guard.

5. **Measured Batch Sizes Only**:
   - `MEASURED_MODELS` is keyed on `(model_name, gpu_family)` and today holds exactly one entry: the canonical model on an `L4` (`gpu_batch_size=64`, `batch_size=512`, `peak_vram_gb=6.77`, `min_vram_gb=7.77`).
   - A batch size is recommended only on an exact `(model, family)` hit that also clears the VRAM guard. Unmeasured hardware falls back to the profile default rather than extrapolating a VRAM envelope, because guessing high OOMs the GPU mid-run.
   - `extract_gpu_family` recognizes `L4`, `T4`, `A100`, `H100`, `V100`, and `A10G`. The header line reports `(measured, L4)` or `(unmeasured)` so the operator can see which path was taken.

6. **Resolved-Settings Table**:

   ```text
   Detected: 16 CPU | 1× NVIDIA L4 (22.0 GB) | 62.8 GB RAM | 1 node (homogeneous)
   Profile:  gpu-workstation   Model: 20260211_debertav3_finetuned (measured, L4)

    stage        knob                      value   source
    transformer  num_transformer_actors        3   auto
    transformer  num_gpus                   0.33   auto
    transformer  transformer_cpus            4.0   auto
    transformer  gpu_batch_size               64   auto
    transformer  num_agg_actors                0   auto
    transformer  override_num_blocks          16   auto
    transformer  batch_size                  512   auto
    recognizer   num_actors                   14   auto
    recognizer   worker_num_cpus             1.0   auto
    recognizer   override_num_blocks          32   auto
    anonymizer   num_actors                   14   auto
    anonymizer   worker_num_cpus             1.0   auto
    anonymizer   override_num_blocks          32   auto
    runner       no_progress_timeout_s       600   auto
    runner       object_store_gb            18.8   auto
   ```

7. **Opt-Out**: `--no-hardware-autotune` (CLI), `hardware_autotune: false` (YAML), or `run_pipeline(hardware_autotune=False)` (Python). With autotune off the pre-PR literals run unchanged and the table is still logged, with every row marked `default`.

---

## Section 6.1 Resolution: Model-Name Deprecation

`20260211_debertav3_finetuned` is deprecated in favour of `stanford-med-hdr/tide2-sentry-clinical-ner`. `load_model_config` emits a single `DeprecationWarning` naming the canonical key.

**Nothing is removed.** Both entries in `resources/bert_transformer_configuration.json` are kept intact, so the deprecated key keeps returning its own explanation text and output stays byte-identical for callers who do not switch.

The two registry entries were diffed: they are identical in every key except `DEFAULT_EXPLANATION` (`"Identified as {} by finetuned microsoft/deberta-v3-base model"` vs `"Identified as {} by the stanford-med-hdr/tide2-sentry-clinical-ner NER model"`). That confirmed checkpoint identity is the *only* reason the deprecated name is aliased into the `MEASURED_MODELS` lookup. The module docstring records the standard: **alias only on confirmed checkpoint identity, never on name similarity**, because aliasing on a similar-looking name silently applies one model's VRAM envelope to another and can OOM a GPU mid-run. A test asserts the two entries still differ in exactly that one key, so the alias breaks loudly if either entry is edited.

---

## Section 8: Benchmark Results

### Reference box — SHIELD dataset, 1,381 notes, autotune **on**

Evaluated on 1x NVIDIA L4 (22.0GB VRAM visible), 16 CPU cores, 62.8GB RAM, model `20260211_debertav3_finetuned`. Median of 3 runs against the Plan 1 baseline:

| Stage | Plan 1 (Direct Workers) | Plan 2 (Autotuned) | Delta |
|---|---:|---:|---|
| **Stage 1 (Transformer NER)** | 57.3s | 57.4s | +0.1s (within noise) |
| **Stage 2 (Recognizer)** | 23.5s | 23.5s | 0.0s |
| **Stage 3 (Anonymizer)** | 23.7s | 22.8s | -0.9s (within noise) |
| **Total Wall-Clock** | **123.0s** | **121.9s** | **-1.1s (-0.9%, within noise)** |

Individual runs: 121.6s / 122.9s / 121.9s.

On this box every recommended value is byte-identical to the literal it replaces (`3` / `0.33` / `4.0` / `64` / `512` / `16` / `0` / `14` / `1.0` / `32` / `600`), which is the point: **this machine *is* the reference box the old literals were tuned for, so parity is the correct result.** Parity is additionally asserted as an exact-equality unit test, so it cannot regress silently.

### CPU-only — GPUs hidden, 120 notes

Run with `CUDA_VISIBLE_DEVICES=""` on the same box, comparing `6e95581` (before) against this PR. This was measured explicitly because site 3's transformer actor count was the most likely value to move.

| Stage | Before (`6e95581`) | After (this PR) | Delta |
|---|---:|---:|---|
| **Stage 1 (Transformer NER)** | 260.7s | 259.5s | -1.2s (within noise) |
| **Stage 2 (Recognizer)** | 21.4s | 21.3s | -0.1s |
| **Stage 3 (Anonymizer)** | 20.4s | 18.5s | -1.9s (within noise) |
| **Total Wall-Clock** | **320.2s** | **316.7s** | **-3.5s (-1.1%, within noise)** |

Every resolved knob is identical across the two runs (`num_transformer_actors=4`, `num_agg_actors=4`, `override_num_blocks=16`; both CPU stages `num_actors=14`, `worker_num_cpus=1.0`, `override_num_blocks=32`). The actor count did **not** move on a 16-CPU box: the `cpu-only` profile yields 14, the same value the old `TARGET_NODE_CPU_ACTORS` constant hardcoded.

The "before" side was run from a temporary `git worktree` at `6e95581` rather than a stash, so no uncommitted work was ever at risk. The worktree has been removed.

### Verification Gate Status

- [x] **Reference-box parity:** every knob `auto` and byte-identical to the pre-PR literals, asserted by unit test and confirmed by three live pipeline runs.
- [x] **SHIELD benchmark within noise:** median 121.9s vs the Plan 1 baseline of 123.0s.
- [x] **CPU-only before/after measured:** 316.7s vs 320.2s, identical knobs.
- [x] **Single tuning module:** grep confirms `hardware.py` is the only tuning-purpose reader of `ray.cluster_resources()` / `ray.nodes()` / `psutil`. `runner/utils.py:181` is logging-only.
- [x] **Precedence honoured:** user-supplied values never overridden, on any of the three entry paths.
- [x] **Small-box pairing:** fractional CPUs and `enable_checkpoint=False` emitted together, asserted by test.
- [x] **Recommended keys are real:** a test introspects `LocalJobRunner.run_transformer` / `run_recognition` / `run_anonymization` signatures against every emitted key, so a recommendation cannot name a kwarg that does not exist.

---

## Defects Found and Fixed During Verification

Three genuine functional defects surfaced while testing the recommender, plus one that the fix for the second exposed. All predate or arise from this work and are fixed here.

| Defect | Symptom | Fix |
|---|---|---|
| Recommended `no_progress_timeout_s` was dead code | Every stage re-called `configure_data_context()` without it, resetting to the library default. **This also silently discarded the existing `--no-progress-timeout` CLI flag** — a pre-existing bug independent of this PR. | `no_progress_timeout_s` is now a sticky attribute on `LocalJobRunner`, threaded through all four `configure_data_context` call sites by a `_data_context_kwargs()` helper. |
| `hardware_autotune: false` in YAML was ignored | argparse `default=True` made `_apply_config` treat the key as already-set, so the YAML value was never backfilled. | `default=None` on `--no-hardware-autotune`, resolved as `is not False`, so "unset" is distinguishable from "explicitly on". |
| YAML could overwrite an explicitly typed negative flag | `_apply_config` compared *parsed values*; an explicit `--no-hardware-autotune` and an unset `store_true` flag both read as `False`. Exposed by the fix above. | New `_cli_supplied_dests` matches `argv` against each parser action's option strings. **This repairs the same latent bug for `--no-checkpoint`, `--no-transformer`, `--no-recognizer`, and `--no-anonymizer`.** |
| `run_pipeline` mutated the caller's kwargs dicts | `t_kw = transformer_kwargs or {}` aliased the caller's dict, which `apply_recommendations` then fills in place. A second `run_pipeline` call would read the first run's resolved values back as explicitly-supplied `USER` settings. | `dict(...)` copies, with a comment explaining why the copy is load-bearing. |

---

## Compatibility

### Hard Breaks

None. Every pre-existing call signature, CLI flag, and YAML key is accepted unchanged.

### Silent Semantic Changes

| Change | Before | After |
|---|---|---|
| Unset per-stage knobs | Hardcoded literals tuned for a 16-CPU / 1×L4 box, applied on every machine | Sized from the detected profile. **Identical on the reference box**; different (and correct) elsewhere. |
| `no_progress_timeout_s` | 600s, and silently reset by every stage even when explicitly set | 600s on GPU and `large-cpu` profiles, 1200s on small boxes and `cpu-only`. Now actually reaches the stages. |
| Object store sizing | `int(total_ram * 0.3)`, silently skipped if `psutil` missing | `recommend_object_store_gb`, same 30% ratio, withheld (not silently skipped) when RAM is unknown. |
| `_auto_num_actors` default | `int(cluster_cpu * 0.45)` | Profile value (~0.875 on a `gpu-workstation`). The `fraction=` argument still overrides. |
| `20260211_debertav3_finetuned` | Accepted silently | Accepted with a `DeprecationWarning`. Config entry and output unchanged. |
| Startup logs | Resolved CPU config only | Full `Detected:` / `Profile:` header plus a per-knob table with `USER` / `auto` / `default` provenance. |

### Unchanged

- All existing kwargs, CLI flags, and YAML keys (`num_actors`, `batch_size`, `cpus_per_actor`, `worker_num_cpus`, `override_num_blocks`, `read_cpus`, `write_cpus`, `enable_checkpoint`, `num_gpus`, `transformer_cpus`, `gpu_batch_size`, …) remain fully supported and now take strict precedence over every recommendation.
- Both model registry entries, including the deprecated key's explanation string.
- `docs/plan_02_settings_recommender.md` §9 compatibility requirements are met in full.

---

## Documentation

- **README**: new *"Hardware autotuning"* section covering the precedence rule, example table output, all three opt-out paths, the 2×3 profile matrix, the small-box pairing requirement, the extrapolated-profile caveat, heterogeneous → `unknown`, the measured-models-only rule with its model table, the deprecation, and the procedure for adding a measured model.
- **Docstrings**: `hardware.py` carries the module-level statement that it is the sole tuning-purpose hardware reader, the application rule, the cluster-vs-node distinction, and the model-aliasing standard. All 11 public symbols render on the pdoc page for `tide2.runner`.
- **`runner_config_example.yaml`**: documents `hardware_autotune`, and comments out the stale `num_actors: 100` / `cpus_per_actor: 2` pair — a supervisor-era pairing that over-subscribed the example's 224-CPU cluster.

---

## Testing & Quality Checklist

- [x] `uv run pytest tests/test_hardware.py --no-cov`: 125 passed (all 8 plan test requirements plus CLI/YAML precedence, signature introspection, and timeout stickiness).
- [x] `uv run pytest`: 955 passed (953 excluding integration).
- [x] `uv run pre-commit run --files ...`: all hooks passed cleanly on every changed file.
- [x] `uv run ty check`: clean on `hardware.py`. The 2 remaining diagnostics in `local_runner.py` were verified by worktree comparison to pre-exist on `6e95581`.
- [x] `uv sync --extra docs --extra llm && uv run make docs`: builds cleanly; all 11 new public symbols render.
- [x] No breaking-change notation in title or body.

### Known pre-existing issue (not introduced here)

The pdoc build emits `UserWarning: Found '<name>' in <module>.__all__, but it does not resolve` for five names in `tide2.actors` and two in `tide2.recognizers`. These predate this PR and are unrelated to it, but the published docs site currently has those gaps.
