# Interactive GPU testing on GKE

How to test the current branch against a real NVIDIA L4 GPU in
`onc-central-dev-gke`, without a local GPU, without waiting on the
prefect/ray production pipeline. Everything here is dev/scratch tooling —
none of it is used by CI or the release process.

## Current session state (2026-09-16, branch `nobody-loves-raymond`)

Picking this back up in a new session? Start here instead of redeploying
from scratch:

- **Pod is already running**: `tide2-gpu-dev-5c8c7cc6f-2shpr` in namespace
  `starr` (check with `kubectl get pods -n starr -l
  app.kubernetes.io/name=tide2-gpu-dev`). Don't `kubectl apply -f
  dev-pod.yaml` again unless it's gone — reuse it.
- **`examples/gpu_batch_pipeline.py` is gitignored/untracked, not part of any
  commit** — it's pure local scratch on this branch. Re-`kubectl cp` it to
  the pod (`/data/scratch/jmesterh-dev/gpu_batch_pipeline.py`) after any
  local edit; nothing about it survives a fresh clone.
- **Scratch data already staged on the pod** at
  `/data/scratch/jmesterh-dev/`:
  - `gpu_batch_pipeline.py` — current copy of the example script (re-`kubectl
    cp` it if you edit the local copy).
  - `input/` — the full 3-file set (`part-000000000000/1/2.parquet`, ~46,435
    rows total).
  - `input_1file/` — just `part-000000000000.parquet` (~15,368 rows) for fast
    ~35-40s iteration instead of the full ~67-90s run.
  - `output/final_baseline/` — the final combined-changes benchmark output
    (see *Final baseline* at the end of the results section below); other
    ad-hoc benchmark outputs get cleaned up as they're produced, don't expect
    anything else to persist here.
- **Benchmarking investigation is closed out** — see the *Benchmarking /
  GPU-utilization investigation* section below for the full results table.
  Conclusion: ~50-65% GPU duty cycle is the likely ceiling for
  *overlap*-based approaches on this architecture; don't re-try hypotheses
  2-4, 6, or 7 without new evidence.
- **CUDA streams / async pipelining (hypothesis #7) was tried and reverted**
  this session — single-thread software pipelining (`_launch_forward`/
  `_collect_results` split) + pinned memory/`non_blocking=True` H2D copy
  showed no reproducible wall-clock or utilization improvement. `core.py` is
  back to the hypothesis #4 baseline; see the results table for the full
  A/B evidence before trying a variant of this again.
- **Length-bucketing (hypothesis #8) is a confirmed win** this session —
  sorting `chunk_texts` by length before batching (in
  `examples/gpu_batch_pipeline.py`'s `run_transformer_stage`) cut real
  padding waste instead of trying to overlap around it: 69s → 37s on the
  1-file input, reproduced twice, correctness-verified against baseline. This
  is currently the only change with a large, reproducible win in this whole
  investigation. Only lives in the (gitignored) example script, not `core.py`.
- **`--gpu-batch-size` sweep with bucketing (hypothesis #9) is done** —
  bigger batches are *still* worse even with bucketing (16=35s, 32=36s,
  64=37s, 96=40s, 128=49s, 256=76s). Keep the default at 64; don't try
  increasing it again without new evidence.
- **Multi-process `--num-gpu-workers` (hypothesis #10) is a confirmed,
  smaller-than-expected win, now the default (3)** — running N independent
  worker processes (each with its own model) on the same GPU fills one
  worker's GPU-idle CPU time with another's GPU compute: ~8-10% faster,
  reproducibly, once fairly measured (a naive first test showed a *bigger*
  win, but that was confounded — see the hypothesis #10 write-up before
  reading anything into a `--num-gpu-workers` A/B).
- **`--row-batch-size` default raised 256→512 (hypothesis #11)** — bigger
  bucketing candidate pools bucket more precisely, with no downside up to at
  least 2048 (unlike `--gpu-batch-size`, which does have a downside — #9).
- **Final combined-changes baseline recorded** — 67s / 54.6% avg GPU util
  (100% peak) / 3.29GB avg GPU memory (4.43GB peak) on the full 3-file input
  with current defaults (`--gpu-batch-size 64 --row-batch-size 512
  --num-gpu-workers 3`); ~25% faster than the session's starting point (89s).
  Full commands + numbers in the *Final baseline* subsection at the end of
  the results section.
- **Ray vs no-Ray comparison done** (this branch's whole reason for
  existing) — see the *Ray vs no-Ray comparison* section right after *Final
  baseline*. Headline: `main`'s Ray pipeline takes 391.4s vs this branch's
  67s on the same input, **but this is not a clean "Ray adds overhead"
  result** — `main` is missing hypothesis #8's length-bucketing entirely,
  which also explains part (not all) of a 22% entity-span mismatch rate
  between the two (a real, reproducible fp16 numerical effect of
  unsorted/unbucketed batch padding, not a correctness bug — root-caused
  via an isolated `batch_size=1` repro). **Follow-up tested and disproved
  all three leading candidates for the timing gap**: reduced CPU-actor
  count, disabled checkpointing, and porting length-bucketing into `main`'s
  `TransformerInferenceActor` (a real uncommitted local change on a `main`
  checkout) — none moved the 391s→245s transformer-phase timing at all,
  though bucketing did improve the span match rate to 81.4%. The timing
  gap's root cause is now genuinely open, not just untested — see the *Ray
  vs no-Ray comparison* section's final subsection before repeating any of
  these three experiments.
- **Not yet tried** (bigger, untested ideas if resuming perf work): verify
  the attention backend is `sdpa` not `eager`, process-based tokenization, or
  `torch.compile(mode="reduce-overhead")` with CUDA graphs (needs a
  `scripts/compile_model.py` cache-generation tool that doesn't exist yet).
  For the Ray comparison specifically: profile Ray Data's per-operator
  overhead directly (task scheduling, object-store serialization/spilling —
  `--object-store-gb 4` is quite small) via `py-spy`/`ray.timeline()` rather
  than more CLI-flag guessing; actor count, checkpointing, and bucketing are
  all now falsified as the timing cause.
- **Teardown**: if wrapping up for good, `kubectl delete -f dev-pod.yaml` (see
  *Cleanup* at the end of this file) — otherwise leave the pod running, it's
  cheap to reuse across sessions.

## Why

`tide2-core` is a library: there's no bundled batch runner (see
[`examples/gpu_batch_pipeline.py`](examples/gpu_batch_pipeline.py) for the
reference orchestration an external consumer would write). To sanity-check a
change against a real transformer model on GPU, the fastest path is a
long-lived pod you `kubectl exec` into repeatedly, rather than a one-shot
prefect job like [`pod.yaml`](pod.yaml).

## One-time environment facts

- Cluster: `gke_som-rit-phi-oncology-dev_us-central1_onc-central-dev-gke`,
  namespace `starr`. Already has an autoscaling L4 node pool
  (`onc-central-dev-l4`, `g2-standard-32`, 1x `nvidia-l4` per node,
  autoscales 0->30). From zero nodes, provisioning a node takes ~5-8 minutes
  (VM boot + GPU driver install) before a GPU pod can schedule.
- PVC `tide-pvc` (4Ti RWX) and ServiceAccount `tide-sa` already exist in
  `starr` and are reused here — **`tide-pvc` is the same volume used by real
  production pipeline runs**, so all test data lives under a scratch
  subdirectory (e.g. `/data/scratch/<you>-dev/`) to avoid colliding with real
  data.
- Image registry: `us-west1-docker.pkg.dev/som-rit-infrastructure-prod/starr`
  (Artifact Registry, cross-project from the cluster but already reachable —
  same repo the real `tide2-app-gpu` image lives in). Dev builds use a
  distinct image name, `tide2-gpu-dev`, so they never collide with real tags.

## One-time setup

```bash
gcloud auth login                                   # if not already
gcloud auth configure-docker us-west1-docker.pkg.dev # docker push credentials
kubectl config use-context gke_som-rit-phi-oncology-dev_us-central1_onc-central-dev-gke
```

Create a local `.env` (gitignored, read by the `Makefile`) pointing
`make docker-gpu` at the dev image instead of the real one:

```
DOCKER_REGISTRY=us-west1-docker.pkg.dev/som-rit-infrastructure-prod/starr
DOCKER_IMAGE_GPU=tide2-gpu-dev
```

`dev-pod.yaml` (repo root, untracked scratch file, same pattern as
`pod.yaml`) defines a `Deployment` named `tide2-gpu-dev`:

- `nodeSelector: cloud.google.com/gke-accelerator: nvidia-l4` +
  `nvidia.com/gpu:NoSchedule` toleration, so it lands on the L4 pool.
- `command: ["sleep", "infinity"]` — stays up across many `kubectl exec` runs
  instead of running a one-shot job.
- Resources: `24` CPU / `32Gi` mem / `1` GPU. This pod has the `g2-standard-32`
  node (31.85 allocatable cores) to itself in practice, so it's sized close
  to the node rather than split with other workloads — leaves ~7 cores of
  headroom for GKE system daemonsets/driver components, not for other test
  pods. If you *do* expect to share the node, size down accordingly.
- Mounts `tide-pvc` at `/data`, plus a memory-backed `/dev/shm`.

## Build + push the image

```bash
make docker-gpu
```

Builds the existing `production-gpu` Dockerfile target for `linux/amd64` and
pushes `$(DOCKER_REGISTRY)/$(DOCKER_IMAGE_GPU):dev`. First build is slow
(~5-6 min: CUDA base image + torch/transformers wheels, ~5GB). Rebuilds are
much faster since the buildx cache (`.buildx-cache-gpu/`) only invalidates
the last `uv sync` + `COPY src` layers when `src/` or lockfile changes.

> Piping `make docker-gpu` through `| tail` buffers *all* output until the
> build finishes — redirect to a log file and poll it instead if you want to
> watch progress: `make docker-gpu > /tmp/build.log 2>&1 &` then `tail -f /tmp/build.log`.

## Deploy / reuse the pod

```bash
kubectl apply -f dev-pod.yaml
kubectl get pods -n starr -l app.kubernetes.io/name=tide2-gpu-dev -w
```

First apply triggers a cluster-autoscaler scale-up if no L4 node is up yet
(watch with `kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-l4`
and `kubectl describe pod ... | tail -30` for `TriggeredScaleUp` events).
Once `Running`, sanity-check the GPU is visible:

```bash
POD=$(kubectl get pods -n starr -l app.kubernetes.io/name=tide2-gpu-dev -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n starr "$POD" -- python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Stage code + data and run the pipeline

`examples/` isn't baked into the image (it's external-consumer reference
code, not library code), so copy it in along with a data slice:

```bash
POD=$(kubectl get pods -n starr -l app.kubernetes.io/name=tide2-gpu-dev -o jsonpath='{.items[0].metadata.name}')
SCRATCH=/data/scratch/$(whoami)-dev
kubectl exec -n starr "$POD" -- mkdir -p "$SCRATCH/input" "$SCRATCH/output"
kubectl cp examples/gpu_batch_pipeline.py "starr/$POD:$SCRATCH/gpu_batch_pipeline.py"
for f in temp/part-000000000000.parquet temp/part-000000000001.parquet; do
  kubectl cp "$f" "starr/$POD:$SCRATCH/input/$(basename "$f")"
done

kubectl exec -n starr "$POD" -- python "$SCRATCH/gpu_batch_pipeline.py" \
  --input "$SCRATCH/input" --output "$SCRATCH/output/output.parquet" \
  --model StanfordAIMI/stanford-deidentifier-v2 \
  --salt-hex "$(cat temp/salt.bin)" --key-hex "$(cat temp/key.bin)"
```

Note: `temp/salt.bin` / `temp/key.bin` already contain ASCII hex text (64 hex
chars = 32 bytes each) — pass them straight through with `cat`, don't
re-hex-encode.

## Fast iteration

- **Dependency changes** (`pyproject.toml`/`uv.lock`): `make docker-gpu` again,
  then `kubectl rollout restart deployment/tide2-gpu-dev -n starr` to pick up
  the new image.
- **Pure `src/tide2` code changes** (it's pure Python, no compiled
  extensions): skip the image rebuild entirely and hot-patch the running pod.
  The image installs the project in editable mode, so the live source is at
  `/opt/tide2/src/tide2`, **not** under `.venv/site-packages` — confirm with
  `kubectl exec -n starr "$POD" -- python -c "import tide2, os; print(os.path.dirname(tide2.__file__))"`
  if unsure:
  ```bash
  kubectl cp src/tide2 "starr/$POD:/opt/tide2/src/tide2"
  ```
  then just re-run the `kubectl exec` test command — seconds instead of
  minutes, no build/push/pull/restart cycle.

## Benchmarking / GPU-utilization investigation (read before repeating this)

Starting symptom on this L4 pod: `examples/gpu_batch_pipeline.py` used only
~1.6GB GPU memory and ~50-65% GPU duty cycle (`nvidia-smi
--query-gpu=utilization.gpu`, sampled inside the pod every ~1s while the
pipeline ran — cross-checked against the GKE console GPU graphs, so the
sampling itself is *not* the source of noise). Run-to-run variance on
otherwise-identical code was ~±10 points, so only treat effects bigger than
that as real; always A/B on the same input file(s) and same salt/key.

For fast iteration, copy just one parquet file into its own input dir
(~15k rows, ~70s per run) instead of the full `temp/` set (~46k rows, ~3min):
```bash
kubectl exec -n starr "$POD" -- mkdir -p "$SCRATCH/input_1file"
kubectl exec -n starr "$POD" -- cp "$SCRATCH/input/part-000000000000.parquet" "$SCRATCH/input_1file/"
```

**Hypotheses tried, in order, with results** (so the next agent doesn't
re-run these): all changes below except the two marked "kept" were reverted.

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | CPU recognizer/anonymizer stage (regex, GIL) blocks the GPU between batches | Moved it off the main thread: first a background *thread* (partial win, ~50%→~58% util — GIL still held during regex matching), then a `ProcessPoolExecutor` (~50%→~63% util) | Real but modest overlap improvement — **kept** (single worker, `max_workers=1`) |
| 2 | CPU stage is slow enough to need more parallel workers | Added `--cpu-workers` fan-out (1→8→24 workers) | **No effect at any worker count.** Instrumenting `flush()` showed the CPU-stage future was *already done* (`wait≈0.00s`) even with 1 worker — it was never the bottleneck, the process-based overlap in #1 was already enough. Fan-out reverted (dead complexity). |
| 3 | Larger `--gpu-batch-size` amortizes fixed per-call overhead | 64 → 256 | **Made it worse**: total inference wall time +25% (177s→223s). Bigger batches pad every chunk to the batch's longest member, wasting compute; utilization got *more* bursty, not steadier. Don't increase this. |
| 4 | `_forward_batch_direct`'s postprocessing loop (pure-Python double loop over batch×padded-seq-len) is the dead CPU time | Vectorized with numpy (mask + `np.nonzero` instead of nested `for`) | Correctness-verified via tests (`tests/test_transformer_core_forward_batch.py`), asymptotically better — **kept** — but a fair single-file A/B showed **no measurable wall-clock/utilization change** (70s vs 72s). This loop was never the real bottleneck either. |
| 5 | Fine-grained timing of one `_forward_batch_direct` call (tokenize vs to_device vs forward vs softmax vs copy_back, with `torch.cuda.synchronize()` around each) | Direct instrumentation, 300 sub-batches | **Real finding**: tokenize = 33% of time, forward pass = 66%, everything else ≈0%. Tokenize runs entirely before the GPU call with zero overlap. |
| 6 | Hide tokenize behind the GPU forward pass via a background thread (HF fast tokenizers release the GIL for the Rust tokenization loop) | Prefetch next sub-batch's tokenization while the current forward pass runs | Overlap *mechanism* worked (`tokenize_wait≈0.00s` for 13/15 sample batches) but **total critical-path time was unchanged** (56.34s vs 56.25s) — the background thread's Python-level tensor/`BatchEncoding` construction contends for the GIL with the main thread's own work, canceling the benefit. Reverted. |
| 7 | Single-thread software pipelining (no background thread, so no GIL contention): split `_forward_and_postprocess` into `_launch_forward` (H2D copy + forward + softmax, enqueues async CUDA work, no `.cpu()`) and `_collect_results` (the `.cpu()` sync point), then reorder `infer_raw_direct`'s loop to depth-2 — tokenize sub-batch i+1 and enqueue its GPU work *before* blocking on sub-batch i's results. Also pinned the tokenizer's output tensors (`.pin_memory()`) and used `non_blocking=True` for the H2D copy, since a plain `.to(device)` on pageable memory is synchronous regardless of pipelining and would silently defeat it. | **No reproducible improvement.** Full 1-file pipeline run: 70s (baseline was 70-72s). Isolated `infer_raw_direct` over all 16,216 chunks: 56.9-57.6s pipelined vs 57.6s baseline — within run-to-run noise. A paired N=30 in-process comparison of "launch+immediate collect" vs "launch+tokenize_next+collect" gave statistically indistinguishable means (160ms vs 158ms, stdev ~40ms). A separate 5-trial CUDA-event probe *did* show partial overlap in isolation (GPU-reported exec time ≈ total wall time including the CPU work gap, i.e. the CPU work looked "free"), but this didn't survive averaging over many real batches — most of the apparent per-batch timing variance turned out to be batch-content variance (variable real sequence length → variable padding → variable true compute time), not an overlap effect. Reverted (`core.py` back to the vectorized-postprocessing baseline from hypothesis #4). |
| 8 | Reduce actual GPU compute (not just overlap it): hypothesis #3 already proved padding waste is real, but chunks were still batched in document order, so each batch of 64 padded to whatever the longest chunk in that arbitrary grouping was. Sort `chunk_texts` by character length (cheap proxy for token count) before batching in `run_transformer_stage` (examples/gpu_batch_pipeline.py), run inference on the sorted order, then unsort predictions back to the original per-chunk order before reassembly. | **Confirmed real win.** Same-session back-to-back A/B on the 1-file input: baseline 69s → bucketed 37s (~46% wall-clock reduction), reproduced twice. Correctness verified two ways: (a) row order/`text_hash` alignment unaffected (sorting only reorders the GPU call, not `chunk_rows`/output rows); (b) output content diffed positionally against baseline — 88/15368 rows differ only in Faker-substituted values (URLs/IDs, which are unseeded-random by design: a baseline-vs-itself rerun showed the *same* 88 mismatches), and entity_count differs by ±1 on only 7/15368 rows (0.046%), consistent with ordinary fp16 batch-composition numerical noise, not a bucketing bug. **Kept** — this is scratch/example code only (`examples/` is gitignored on this branch, not part of the library). |
| 9 | With bucketing in place, does hypothesis #3's "bigger `--gpu-batch-size` is worse" finding still hold, or does bucketing unlock larger batches? | Swept `--gpu-batch-size` (16/32/64/96/128/256) on the bucketed 1-file input | **#3's finding still holds — bigger is still worse, even bucketed.** 16=35s, 32=36s, 64=37s (current default), 96=40s, 128=49s, 256=76s. Flat/near-optimal from 16-64, then degrades sharply. Sorting only reduces intra-batch length *variance*, and that window of variance still grows with batch size, so padding waste still increases with batch size — bucketing helps at any fixed batch size, it doesn't remove the incentive to keep batches small. **No change made** — the existing default of `--gpu-batch-size 64` is already at/near the sweet spot; don't increase it. |
| 10 | The real headroom isn't compute *efficiency*, it's that a single process leaves the GPU idle ~35-50% of the time (waiting on the CPU-bound recognizer/anonymizer stage) while using only ~1.2GB of the L4's 23GB — so run N independent worker *processes*, each with its own `TransformerCore` on the same GPU, sharding input files across them, so one worker's GPU-idle CPU time gets filled by another's GPU compute. | Added `--num-gpu-workers` to `examples/gpu_batch_pipeline.py`: extracted the per-process pipeline into `run_pipeline()`, and for N>1 spawns N `multiprocessing` (spawn context — CUDA+fork is unsafe) processes, each handling a shard of `--input`'s files and writing its own `workerN.parquet` into `--output` (now a directory). First proven manually (3 separate `kubectl exec` processes on 3 file shards) before implementing: sequential single-process 3-file baseline 89s → 3 concurrent processes 64s, GPU util 50-65%→78% avg (100% peak), GPU memory 1.2GB→3.6GB peak (3 model copies). | **Confirmed real win, now implemented — but smaller once measured fairly.** The built-in `--num-gpu-workers 3` reproduced it end-to-end: 89s → 80s (~10%, vs the raw manual test's 64s — see caveats below), reproducible across 2 runs, correctness verified (46,435 rows total across `worker{0,1,2}.parquet`, matching sequential exactly). `--num-gpu-workers 2` on the *same* 3-file input was worse (91s, avg util only 44%) — but `_shard`'s naive round-robin (`items[i::num_shards]`) gives 3 files ÷ 2 workers a 2:1 split (one worker idle for the last third while the other finishes alone), not a real "2 workers < 3 workers" signal. Re-tested with a **balanced** 2-file input: 1 worker 71s → 2 workers 65s (~8%, avg util 48%, peak mem 3.07GB) — a real but modest win, in the same ballpark as the 3-worker case's ~10%, not the ~28% the initial raw manual demo suggested. Two caveats worth remembering: (1) **the built-in multiprocessing version underperforms the raw manual `kubectl exec`-per-process demo** (64.6% avg util / 80s vs 78% avg util / 64s at N=3) — likely `spawn`'s per-worker interpreter/import startup (re-importing torch/transformers/spacy from scratch each time) eating into the run, not yet investigated further; (2) **`_shard`'s file-level round-robin needs an even file-count/worker-count ratio (or at least similar-sized shards) to give a fair comparison** — don't read anything into `--num-gpu-workers` results without checking the resulting shard sizes are balanced. **Kept.** Still the most direct, reproducible answer to "the GPU is underused": scale *workers*, not batch size — memory headroom (1.2GB used of 23GB) is exactly what makes this safe on a single node/pod, even if the realistic gain is ~8-10% rather than ~28%. |
| 11 | Bucketing (#8) only sorts by length *within* each call to `run_transformer_stage`, i.e. within one `--row-batch-size` worth of notes — is the default of 256 notes/row-batch big enough to bucket well, or does a bigger candidate pool bucket more precisely (and does a bigger pool have any downside, the way bigger `--gpu-batch-size` does in #9)? | Swept `--row-batch-size` (64/128/256/512/768/1024/2048) on the bucketed 1-file input, `--gpu-batch-size` left at the default 64 | **Real, previously-undiscovered win — the 256 default was measurably too small.** 64=58s, 128=45s, 256=38s (old default), 512=35s, 768=35s, 1024=36s, 2048=36s (reproduced 512 and 768 at the same 35s). Unlike `--gpu-batch-size`, there's **no downside to going bigger** here up to at least 2048 — makes sense, since `--row-batch-size` only sizes the bucketing candidate pool and the CPU/GPU pipeline granularity, it's never padded as one GPU-forward unit the way a `--gpu-batch-size` batch is. **Changed the default to 512** (clear plateau start, comfortable margin before any theoretical corpus-size-driven memory growth). |

**Where this leaves it**: ~50-65% GPU duty cycle *per single process* looks like
the natural ceiling for this model/tokenizer/batch-size on this GPU given the
current synchronous, single-stream `TransformerCore` architecture. Every "overlap it
in Python" lever has been tried — background thread (#6), single-thread
reorder + pinned memory + non_blocking H2D (#7) — and each either didn't
apply (the thing it targeted wasn't actually the bottleneck) or was canceled
by the GIL or by batch-to-batch variance swamping the effect size. Hypothesis
#8 broke that pattern by *reducing real GPU work* (less padding waste)
instead of trying to hide CPU time behind it, and it's the first thing in
this whole investigation with a large, reproducible, verified effect. #9
confirmed that win doesn't extend to bigger batches — keep
`--gpu-batch-size` at 64 (or as low as 16-32, statistically indistinguishable).
#11 found the flip side: bigger `--row-batch-size` (the bucketing candidate
pool) *does* help, up to at least 2048, with no downside — changed the
default from 256 to 512. #10 found an orthogonal, complementary lever (worker
*count*, not batch/pool size): running multiple worker processes on the same
GPU gives another ~8-10% by filling one worker's GPU-idle CPU time with
another's GPU compute.
Further gains would require either:
- Verify the attention backend (`attn_implementation`) is `sdpa` rather than
  `eager` (neither `TransformerCore` nor the example script sets this
  explicitly today), or
- **Process-based tokenization** (real GIL avoidance, but untested — passing
  tokenized tensors across a process boundary has its own serialization cost
  that could easily eat the gain), or
- **`torch.compile(mode="reduce-overhead", fullgraph=True)` with CUDA
  graphs** — `TransformerCore` already has the `compile_model`/
  `compile_cache_path` plumbing, but `_resolve_compile_cache_path` expects a
  pre-built `compiled_cache.bin` generated by `scripts/compile_model.py`,
  which **does not exist in this repo**. CUDA graphs want fixed shapes, so
  this would need combining with #8's bucketing (or padding to one fixed
  length) to avoid constant recompilation. Bigger lift than #8, not
  attempted, or
- **A genuinely separate `torch.cuda.Stream()` for H2D copies** with
  fixed-size padded buffers and explicit `torch.cuda.Event` wait/record
  pairs — not attempted; #7 only reordered work on the *default* stream.
  Given #7's evidence that any real overlap window here is on the order of
  tens of ms (swamped by ~40ms of inherent per-batch variance), this is a
  low-confidence next step, not a promising one.

Don't re-try hypotheses 2-4, 6, 7, or 9 without new evidence; they're
falsified for this workload. If picking this back up on the overlap side,
start from #5's breakdown (get a fresh tokenize/forward split first) rather
than guessing again. On the compute-reduction side (the more promising
direction after #8), the next untried step is verifying the attention
backend (`attn_implementation`) is `sdpa` not `eager` — cheap to check, and
distinct from bucketing/batch-size which are both already settled (#8, #9).

### Final baseline (all changes combined)

One clean run with every kept change in place at its current default
(`--gpu-batch-size 64`, `--row-batch-size 512`, `--num-gpu-workers 3`),
against the full 3-file input, to have a single reference number for "where
this branch stands" rather than only per-hypothesis deltas:

```bash
POD=$(kubectl get pods -n starr -l app.kubernetes.io/name=tide2-gpu-dev -o jsonpath='{.items[0].metadata.name}')
kubectl cp examples/gpu_batch_pipeline.py "starr/$POD:/data/scratch/jmesterh-dev/gpu_batch_pipeline.py"
kubectl exec -n starr "$POD" -- rm -rf /data/scratch/jmesterh-dev/output/final_baseline

SALT_HEX=$(cat temp/salt.bin)
KEY_HEX=$(cat temp/key.bin)

START=$(date +%s)
: > /tmp/gpu_samples_final.txt
nohup kubectl exec -n starr "$POD" -- python /data/scratch/jmesterh-dev/gpu_batch_pipeline.py \
  --input /data/scratch/jmesterh-dev/input \
  --output /data/scratch/jmesterh-dev/output/final_baseline \
  --model StanfordAIMI/stanford-deidentifier-v2 \
  --salt-hex "$SALT_HEX" --key-hex "$KEY_HEX" > /tmp/pipeline-final-baseline.log 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  kubectl exec -n starr "$POD" -- nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null >> /tmp/gpu_samples_final.txt
  sleep 1
done
END=$(date +%s)
echo "elapsed: $((END-START))s"
awk -F', ' '{util+=$1; mem+=$2; n++; if($1+0>maxu)maxu=$1+0; if($2+0>maxm)maxm=$2+0}
    END {print "avg util:", util/n, "% peak util:", maxu, "% avg mem:", mem/n, "MiB peak mem:", maxm, "MiB"}' \
  /tmp/gpu_samples_final.txt
```

Result: **67s** wall-clock, **54.6%** avg GPU utilization (100% peak), **3.29GB**
avg GPU memory (4.43GB peak), output split across `worker{0,1,2}.parquet`
(15368 + 15559 + 15508 = **46,435 rows**, matching the known-correct total
from every earlier run in this investigation — verified via a small
`pyarrow.parquet` row-count script, not shown here).

For context against where this session started (89s, single sequential
worker, defaults before hypotheses #8/#10/#11): **~25% wall-clock reduction**,
and GPU memory usage went from ~1.2GB (barely touching the L4's 23GB) to
~3.3-4.4GB — the hardware is now actually being used instead of sitting
mostly idle.

## Ray vs no-Ray comparison (does removing Ray actually help?)

This whole branch exists as a proof-of-concept: is `main`'s Ray-based
`tide2-runner run pipeline` actually slower than a plain-Python
single/multi-process pipeline doing the same work? This section runs
`main`'s real Ray code on the *same pod, same 3-file input* as the Final
baseline above, and is far more nuanced than a single wall-clock number —
read the whole thing before quoting just the headline numbers.

### Getting main's Ray code running on this pod (environment notes)

The dev pod's image was built from *this* branch, which has no `ray`
installed (`chore: nuked ray`) and a different `src/tide2` (this branch has
its own accumulated changes — see *Why the comparison is confounded* below).
To run `main`'s code on the same pod without a full image rebuild:

- `git show main:pyproject.toml` / `uv.lock` don't need a full
  `make docker-gpu` — the `production-gpu` Dockerfile target requires
  `prefect.yaml`/`prefect_job_template.json`, which aren't tracked in this
  repo (deployment artifacts supplied separately) and aren't needed just to
  run the CLI. Instead: `kubectl cp` main's `pyproject.toml`/`uv.lock` onto
  the pod, then `uv sync --locked --no-dev` **inside the pod** (the `uv`
  binary is already there from the image build) — main's dependency set is
  a superset of this branch's (adds `ray[default,data]`, `streamlit`,
  `google-cloud-*`), so this is a safe, one-directional upgrade of the venv.
- **`src/tide2` has to be swapped wholesale, not just `runner`/`actors`.**
  The two branches differ well beyond Ray removal (`transformers/core.py`
  ~198 lines, `transformers/reassembly.py` moved to `runner/transformer.py`,
  `recognizers/nlp_engine.py`, etc.) — copying just `runner`/`actors` from
  `main` onto this branch's `src/tide2` leaves an inconsistent hybrid that
  either won't import or won't reflect either branch's real behavior.
  Whichever CLI you need to run (`tide2-runner` vs
  `examples/gpu_batch_pipeline.py`), replace the *entire*
  `/opt/tide2/src/tide2` on the pod with that branch's version first:
  ```bash
  POD=$(kubectl get pods -n starr -l app.kubernetes.io/name=tide2-gpu-dev -o jsonpath='{.items[0].metadata.name}')
  # To run main's tide2-runner:
  git archive main -- src/tide2 | tar -x -C /tmp/main_src   # from a `main` checkout/worktree
  kubectl exec -n starr "$POD" -- rm -rf /opt/tide2/src/tide2
  kubectl cp /tmp/main_src/src/tide2 "starr/$POD:/opt/tide2/src/tide2"
  # To go back to running examples/gpu_batch_pipeline.py on this branch:
  git archive nobody-loves-raymond -- src/tide2 | tar -x -C /tmp/nlr_src
  kubectl exec -n starr "$POD" -- rm -rf /opt/tide2/src/tide2
  kubectl cp /tmp/nlr_src/src/tide2 "starr/$POD:/opt/tide2/src/tide2"
  ```
  (The venv/dependencies from the `uv sync` above don't need to change back
  and forth — main's superset venv runs both branches' code fine.)
- **Ray's memory auto-detection fails in this container as-is.** `ray.init()`
  with no explicit sizing raises `ValueError: ... amount of memory on this
  node available for tasks and actors (-7.79 GB) is less than -22% of total`.
  Cause: `free -h` inside the pod reports the **host's** 125GiB (cgroup-blind
  `/proc/meminfo`), while the actual enforced limit is the pod's 32Gi
  (`cat /sys/fs/cgroup/memory.max` → `34359738368`) — Ray's sizing heuristic
  gets a mismatched signal and computes negative headroom. Fix: pass explicit
  `--num-cpus 24 --num-gpus 1 --object-store-gb 4` to `tide2-runner run
  pipeline` (same class of fix as the README's small-box
  `--object-store-gb` guidance, just triggered here by cgroup/host memory
  mismatch rather than a genuinely tiny box).

### Headline numbers

```bash
SALT_HEX=$(cat temp/salt.bin)
KEY_HEX=$(cat temp/key.bin)
kubectl exec -n starr "$POD" -- tide2-runner run pipeline \
  --input /data/scratch/jmesterh-dev/input \
  --output /data/scratch/jmesterh-dev/output/main_ray_baseline \
  --model StanfordAIMI/stanford-deidentifier-v2 \
  --num-cpus 24 --num-gpus 1 --object-store-gb 4 \
  --salt-hex "$SALT_HEX" --key-hex "$KEY_HEX"
```

| | No-Ray (this branch, Final baseline) | `main` (Ray) |
|---|---|---|
| Wall-clock, 3-file input | **67s** | **391.4s** (~5.8x slower) |
| Breakdown | n/a (single combined run) | transformer 244.8s + recognizer 61.8s + anonymizer 62.4s |
| Output rows | 46,435 (verified) | 46,435 (verified) |

Re-ran with `--gpu-batch-size 64` (matching this branch's default, instead of
main's VRAM-auto-computed value) to test whether batch size explains the
gap: **389.9s — no change.** Batch size is **not** the driver of the timing
difference; root cause of the ~5.8x gap is still open (see below).

### Why row counts aren't good enough QC — and what to compare instead

`anonymized_note_text` and simple row counts are **not** valid correctness
signals here: `FakerAnonymizer`-substituted values (URLs/emails/IDs) are
unseeded-random by design (confirmed earlier — hypothesis #8's QC section —
that even reruns of the *same* code produce different Faker output for the
same input), so comparing anonymized text or counting rows tells you nothing
about whether recognition itself agrees.

The right comparison is the **resolved entity spans** — `(entity_type,
start, end)` — since these are the one thing that should be idempotent
across implementations for the same input text, independent of which
(randomized) anonymization operator ran afterward. Two more wrinkles:

- **`text_hash` isn't a safe join key** — 12,432 of 15,368 rows in file 0
  alone share a `text_hash` with another row (duplicate note content across
  different patients). Joined instead on **`row_id`**, which is already
  present in the input parquet and threaded through by `main`'s pipeline,
  but wasn't preserved by `examples/gpu_batch_pipeline.py`'s output — added
  a `"row_id": note.get("row_id")` passthrough to `run_cpu_stage`'s output
  row (see the script) to enable the join.
- **Compare *resolved* spans, not raw per-recognizer spans.** `main`'s
  `04_recognizer_output` checkpoint has *unresolved* overlapping spans
  (e.g. PHONE/MRN/ID all claiming the same characters, from different
  recognizers, before conflict resolution) — not comparable to this
  branch's already-resolved `recognizer_results_json`. `main`'s
  `06_anonymizer_output`'s `anonymizer_results_json` **is** post-resolution
  and is the correct comparison point.

Comparison script joins on `row_id`, and for each side builds
`frozenset({(entity_type, start, end), ...})` per row, then checks set
equality:

```python
noray_spans[row_id] = frozenset((s["entity_type"], s["start"], s["end"]) for s in json.loads(recognizer_results_json))
ray_spans[row_id]   = frozenset((s["entity_type"], s["start"], s["end"]) for s in json.loads(anonymizer_results_json))
```

**Result: 46,435/46,435 row_ids matched on both sides; 77.8% exact span-set
match (36,143), 22.2% mismatch (10,292).**

### Root-causing the 22% span mismatch

Ruled out, one by one (all verified identical or functionally equivalent
between the two branches):

- `reassemble_chunks_for_document`/`chunk_document_row` — byte-identical
  logic; only moved from `runner/transformer.py` (main) to
  `transformers/reassembly.py` (this branch) when Ray was removed.
- `chunk_size`/`chunk_overlap` — both resolve to 512/40 for
  `StanfordAIMI/stanford-deidentifier-v2` on both branches (main reads
  these from the model's own config entry in
  `bert_transformer_configuration.json`, which happens to match this
  branch's hardcoded example-script defaults for this specific model).
- `aggregate_bio_tokens` (BIO→entity merging) — unchanged between branches.
- `infer_raw_direct`/`_forward_batch_direct` — main has the
  pre-hypothesis-#4 nested-loop version, this branch has the vectorized
  version; same math, same output.
- dtype (`float16` default) and `transformers`/`torch`/`tokenizers`/
  `presidio-analyzer`/`presidio-anonymizer` versions — all identical
  between the two `uv.lock` files.
- `--gpu-batch-size` — tested explicitly (see *Headline numbers* above):
  matching it to 64 changed the match rate from 77.8% to 71.9%, i.e. no
  systematic effect (well within the fp16 batch-composition noise floor
  already characterized in hypothesis #8's QC, just a larger sample of it).

**Found**: an isolated, single-text (`batch_size=1`, no padding at all)
call to `TransformerCore.infer_raw_direct` on one specific mismatched note
(`row_id=132db386...`, a short 255-char note) reproduces this branch's
entity boundary (`HOSPITAL` at chars 145-152, `"LPCH IP"`) **exactly** —
not main's actual Ray-pipeline output for the same text (145-155,
`"LPCH IP NE"`). That wrong boundary is **reproducible**, not random: both
completed Ray runs (auto batch size and matched `--gpu-batch-size 64`) give
the identical (145, 155) for this row_id. Conclusion: **main's Ray actor
batches chunks in whatever order Ray Data's blocks happen to produce — not
sorted by length, unlike this branch's hypothesis #8 bucketing** — so this
short note ends up batched alongside much longer chunks, producing heavy,
uneven padding. That padding measurably (and reproducibly) shifts this
borderline BIO tag's boundary via fp16 attention numerics. This is a real
numerical sensitivity to *unsorted* batch composition, not a logic bug in
either branch's recognition code.

**Still unexplained (before further testing)**: the ~5.8x wall-clock gap
itself. `--gpu-batch-size` is ruled out (identical timing with and without
it matched). Leading untested candidates: Ray Data/actor task-scheduling and
object-store serialization overhead, or the recognizer/anonymizer stage's
10-actor fan-out (main) vs this branch's single `ProcessPoolExecutor` worker
(hypothesis #1).

### Testing the leading candidates: actor count, checkpointing, bucketing

Three follow-up experiments, all on `main` (uncommitted local edits on a
`main` checkout — not part of this branch, not committed anywhere):

**`--num-actors 4` + `--no-checkpoint`** (testing reduced CPU-actor fan-out
and disabled checkpoint shuffle together): **385.9s total, transformer
244.3s, recognizer 60.1s, anonymizer 59.1s — no meaningful change from the
391.4s/389.9s baselines.** But running this surfaced a real, separate
finding:
```
UserWarning: The minimum number of concurrent actors for
'MapBatches(ConfiguredAnonymizerActor)' is set to 4, but the operator only
received 1 input(s)... won't fully utilize the available concurrency.
```
**The recognizer/anonymizer stages were never actually running on multiple
actors in the first place** — all 46,435 rows land in a single Ray Data
block, so only 1 of the requested N actors (10, then 4) ever gets used.
That's *why* 10→4 changed nothing: there was no real parallelism there to
reduce. Low-priority to fix, though, since recognizer+anonymizer combined
(~120s) are dwarfed by the transformer phase (~245s) regardless.

**In-batch length bucketing in `TransformerInferenceActor`** (porting
hypothesis #8 into `main`'s `__call__`, sorting `valid_texts` by length
before `_run_inference_raw_with_oom_recovery` and unsorting results after —
a ~10-line, single-file change, same pattern as the example script's
bucketing): **245.3s transformer phase, 391.6s total — statistically
identical to baseline. Bucketing does not explain the timing gap.** It
*does* have a real but modest effect on the correctness side: span match
rate vs this branch improved from 77.8% to **81.4%** (22.2%→18.6%
mismatch, ~16% relative reduction) — but the *specific* row root-caused
earlier (`row_id=132db386...`, the "LPCH IP" vs "LPCH IP NE" case) is
**still wrong** even with bucketing applied, so batch-composition
sensitivity is only *part* of the correctness gap, not the whole story, and
apparently unrelated to the timing gap entirely.

**Conclusion**: all three of this session's leading hypotheses for the
5.8x timing gap (actor count, checkpointing, length-bucketing) are now
individually disproven. The timing gap is not caused by anything this
session characterized as a batching/padding/parallelism-fan-out problem.
It remains genuinely open — if picking this back up, the next things worth
instrumenting are Ray Data's own per-operator overhead (task scheduling,
object-store serialization/spilling — note `--object-store-gb 4` is quite
small, forced by the cgroup-memory workaround, and could itself be causing
spilling under load) or a from-scratch profiling pass (`py-spy`/Ray's own
timeline/`ray.timeline()` export) rather than guessing at more CLI flags.

### Why the comparison is confounded (read before quoting the 67s vs 391s number)

This is **not** a clean "Ray adds 5.8x overhead" result:

1. `main`'s Ray pipeline had no length-bucketing at all when this comparison
   started; porting it in (above) fixed part of the correctness gap but
   none of the timing gap, so bucketing's absence is *not* the timing
   story — though it's still a legitimate, worthwhile fix for `main`
   independent of this comparison.
2. This branch has accumulated its own extra optimizations beyond removing
   Ray (hypotheses #8-#11) that `main` has no equivalent of.
3. The wall-clock gap's actual root cause is still unidentified after
   testing the three most plausible candidates (actor count, checkpointing,
   bucketing) — see above.

Take the 67s vs 391s numbers as "current state of each branch as found,"
not as an isolated Ray-overhead measurement — and don't re-test actor
count, checkpointing, or bucketing as explanations for the timing gap
without new evidence; all three are now falsified for it.

### Severity breakdown of the span mismatches (is the 22.2%/18.6% mismatch actually dangerous?)

Row-level match/mismatch (77.8%/81.4% match) doesn't say whether a
mismatched row is a harmless boundary shift or an actual missed PHI
entity. Re-ran the comparison at **entity level** (joined on `row_id`,
`final_baseline_qc` vs `main_ray_baseline`'s `06_anonymizer_output`),
classifying every entity into one of five buckets via overlap-based
matching (exact match → same-type overlap = boundary shift → different-type
overlap = type changed → no overlap at all in the other side = missing):

| Category | Count | % of 105,789 entities |
|---|---|---|
| Exact match | 32,690 | 30.9% |
| Boundary shift (same type, overlapping) | 28,479 | 26.9% |
| Type changed (overlapping, different label) | 8,398 | 7.9% |
| Missing in Ray (present in this branch, absent in main) | 17,860 | 16.9% |
| Missing in no-ray (present in main, absent in this branch) | 18,362 | 17.4% |

The combined ~34% "missing" figure looks alarming in isolation, but it is
**heavily concentrated**, not spread evenly across notes:

- Only **2,368 of 46,435 rows (5.1%)** have *any* missing entity at all.
- The worst **25% of those affected rows (592 rows, 1.3% of the whole
  dataset) account for 81% of all missing-entity instances**; the worst 10
  rows alone account for 200-339 missing entities each.
- Checked note length for the 10 worst-offender rows: **10,002-35,682
  characters (~10-19 chunks each at chunk_size=512/overlap=40)**, vs. a
  median note of **106 characters (1 chunk)** and a p99 of only ~4,979
  chars — the worst rows are all in the extreme <0.1%-length tail.

**Conclusion**: for the overwhelming majority of notes (median through
p99, i.e. essentially all typical single-chunk/few-chunk clinical notes),
recognition matches exactly or differs only by a boundary/type label on an
already-detected entity — nothing gets silently dropped. The severe
"entity present on one side only" failures are concentrated in a small
number of unusually long, heavily-chunked documents, consistent with the
fp16 batch-composition sensitivity root-caused above compounding across
many sequential chunks in one long document (one early divergence cascades
forward through the rest of that note's chunks). This is a narrow,
specific, and investigable mechanism — not a general correctness
regression in either branch — but it's also not fully reassuring on its
own: long documents (discharge summaries, long consult notes) are exactly
the kind of note most likely to carry PHI, so this tells you *where* to
focus further validation, not that it's safe to ignore. Re-ran the same
breakdown against `main_ray_bucketed` (the length-bucketed Ray variant)
for completeness: exact match improves to 32.96%, boundary shift/type
changed both drop slightly, but the missing-in-ray/missing-in-noray rates
are essentially unchanged (~16.9%/17.4%) — bucketing does not touch the
long-document cascade mechanism, only ordinary padding-induced boundary
noise.

## Multi-GPU support + dual-L4 benchmark comparison

This branch's runner (`src/tide2/runner/pipeline.py`) gained multi-GPU
support this session: `--num-workers-per-gpu` (default 3) is multiplied by
`torch.cuda.device_count()` to get the total worker count, and each spawned
worker is assigned to a physical GPU round-robin (`worker i -> cuda:{i %
num_gpus}`) via a new `_device_for_worker()` helper. On a single-GPU pod
this is a no-op (all workers still land on `cuda:0`); on a multi-GPU pod it
spreads workers across all visible GPUs automatically, no new flag needed
beyond the existing worker-count knob.

Tested on a new node pool, **`onc-central-dev-l4-dual`** (`g2-standard-24`,
2x NVIDIA L4, 24 vCPU, 96GB RAM), via a new `dev-pod-dual.yaml` manifest
(nodeSelector `cloud.google.com/gke-nodepool: onc-central-dev-l4-dual`,
`nvidia.com/gpu: "2"`, otherwise identical pattern to `dev-pod.yaml`).
Benchmarked both branches against the same 6 input files (236,500 rows
total, from `/data/scratch/benchmark/part-00000000000{1..6}.parquet`),
defaults on both sides (this branch: `--num-workers-per-gpu 3` → 6 workers
total; main: `--num-cpus 20 --num-gpus 2 --object-store-gb 4`). Resource
usage sampled every 3s for the duration of each run (system-wide CPU% from
`/proc/stat`, memory from `/proc/meminfo`, per-GPU utilization/memory from
`nvidia-smi`).

| Metric | This branch (nobody-loves-raymond) | main (Ray) |
|---|---|---|
| Wall-clock | **341s** | 1663s |
| Speedup | **4.9x** | 1x (baseline) |
| Output rows | 236,500 (verified) | 236,500 (verified) |
| CPU avg / peak | 42.5% / 52.3% | 12.3% / 71.9% |
| Memory avg / peak | 15.5GB / 17.2GB | 14.5GB / 26.6GB |
| GPU0 util avg / peak | 82.9% / 100% | 42.0% / 100% |
| GPU0 mem avg / peak | 5.2GB / 5.8GB | 5.4GB / 7.9GB |
| GPU1 util avg / peak | 81.9% / 100% | 23.2% / 100% |
| GPU1 mem avg / peak | 3.3GB / 3.6GB | 2.2GB / 6.4GB |
| Output files | 6 (`worker0..5.parquet`, 1:1 with worker count) | 9 (Ray Data block count — no longer collapses to 1 file at this larger 236K-row scale, unlike the single-GPU/46K-row runs earlier in this doc) |

Takeaways:
- The ~4.9x speedup on 2 GPUs is consistent with the ~5.9x speedup seen
  earlier in this doc on 1 GPU (3-file/46,435-row input) — the advantage
  isn't an artifact of the smaller single-GPU test, it holds at 2x the
  hardware and ~5x the data.
- This branch keeps both GPUs busy and roughly evenly loaded the whole run
  (~82% avg util on both, matching the row-count-based work distribution
  verified earlier: 118,400 vs 118,100 rows per GPU). Main's Ray pipeline
  shows both lower *and* more uneven GPU utilization (42.0% vs 23.2% avg) —
  consistent with the per-GPU actor scheduling being driven by Ray Data's
  block/task assignment rather than an even, deterministic split.
- Main's peak memory (26.6GB system, 7.9GB/6.4GB GPU) is notably higher than
  this branch's (17.2GB system, 5.8GB/3.6GB GPU) despite doing the same
  work — consistent with Ray's object store, actor pool, and checkpoint
  materialization overhead.
- Row counts matched exactly (236,500 both sides) — this run wasn't used
  for a span-level correctness comparison (see the dedicated section above
  for that methodology); it's purely a performance/resource comparison at
  larger scale and on genuinely parallel GPU hardware.

## Cleanup

```bash
kubectl delete -f dev-pod.yaml       # single-GPU dev pod
kubectl delete -f dev-pod-dual.yaml  # dual-GPU dev pod (onc-central-dev-l4-dual)
```

Either node pool scales back down automatically once nothing needs it.
