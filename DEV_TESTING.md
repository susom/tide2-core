# Interactive GPU testing on GKE

How to test the current branch against a real NVIDIA L4 GPU in
`onc-central-dev-gke`, without a local GPU, without waiting on the
prefect/ray production pipeline. Everything here is dev/scratch tooling —
none of it is used by CI or the release process.

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

**Where this leaves it**: ~50-65% GPU duty cycle looks like the natural
ceiling for this model/tokenizer/batch-size on this GPU given the current
synchronous, single-stream `TransformerCore` architecture. Every "overlap it
in Python" lever has been tried and either didn't apply (the thing it
targeted wasn't actually the bottleneck) or was canceled by the GIL. Further
gains would require either:
- **Process-based tokenization** (real GIL avoidance, but untested — passing
  tokenized tensors across a process boundary has its own serialization cost
  that could easily eat the gain), or
- **CUDA streams / async pipelining** at the torch level in `TransformerCore`
  — a much bigger architectural change, not attempted here.

Don't re-try hypotheses 2-4 or 6 without new evidence; they're falsified for
this workload. If picking this back up, start from #5's breakdown (get a
fresh tokenize/forward split first) rather than guessing again.

## Cleanup

```bash
kubectl delete -f dev-pod.yaml
```

The L4 node pool scales back down automatically once nothing needs it.
