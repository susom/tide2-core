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
- **Code is committed**: all of `core.py`'s vectorized postprocessing, the
  process-pool overlap in `examples/gpu_batch_pipeline.py`, and this file are
  in commit `d4b373e` ("chore: nuked ray, added dev test pod") on
  `nobody-loves-raymond`. Working tree is clean — nothing further needs to be
  re-copied to the pod unless you make new local edits (in which case,
  hot-patch per the *Fast iteration* section below).
- **Scratch data already staged on the pod** at
  `/data/scratch/jmesterh-dev/`:
  - `gpu_batch_pipeline.py` — current copy of the example script (re-`kubectl
    cp` it if you edit the local copy).
  - `input/` — the full 3-file set (`part-000000000000/1/2.parquet`, ~46,435
    rows total).
  - `input_1file/` — just `part-000000000000.parquet` (~15,368 rows) for fast
    ~70s iteration instead of the full ~3min run.
  - `output/` — currently empty (leftover benchmarking outputs were cleaned
    up); the last full clean run wrote to `output/output_final.parquet`
    (46,435 rows, since deleted) with no errors.
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
- **Not yet tried** (bigger, untested ideas if resuming perf work): verify
  the attention backend is `sdpa` not `eager`, process-based tokenization, or
  `torch.compile(mode="reduce-overhead")` with CUDA graphs (needs a
  `scripts/compile_model.py` cache-generation tool that doesn't exist yet).
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

## Cleanup

```bash
kubectl delete -f dev-pod.yaml
```

The L4 node pool scales back down automatically once nothing needs it.
