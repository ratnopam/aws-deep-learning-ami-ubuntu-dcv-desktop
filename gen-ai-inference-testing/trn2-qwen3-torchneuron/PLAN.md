# Qwen3 on Trainium2 with vLLM + TorchNeuron — Implementation Plan

**Status:** plan + unverified scaffolding. Nothing in this directory has been executed on hardware yet.
**Author:** drafted 2026-11-06, ahead of a time-boxed trn2.48xlarge Capacity Block.
**Goal:** a runnable, reproducible example that serves Qwen3 models on a single `trn2.48xlarge`
through vLLM on the **native TorchNeuron** backend, and produces defensible SLO numbers
(TTFT, ITL, TPOT, throughput) — then lifts to EKS.

---

## 1. Why this is a new example and not a patch of the existing one

The repo already has `gen-ai-inference-testing/containers/openai_server-vllm-neuron`. It is **not**
a starting point for this work:

| | Existing container | This example |
|---|---|---|
| Base image | `...vllm-neuronx:0.13.0-neuronx-py312-sdk2.28.0-ubuntu24.04` | `...vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` |
| vLLM | 0.7.2 / 0.6.6.post1, **patched** (`patch/vllm-neuron-*.patch`) | 0.29.0 via the **`vllm-neuron` plugin**, no patches |
| Neuron backend | `torch-neuronx` + XLA (`PJRT_DEVICE=NEURON`) | **TorchNeuron native** (`VLLM_NEURON_BACKEND=neuron_native`) |
| Model impls | NxD Inference (`neuronx-distributed-inference`) | **In the plugin** — NxDI dropped as a dependency at 0.21.0 |
| Hardware | trn1 / inf2 era | trn2 / trn3 only |

The compilation model changed (XLA graph tracing → `torch.compile` to NEFF through TorchNeuron's
Dynamo backend), the dependency graph changed, and the patch files no longer apply. Treat the old
container as the *legacy* path and this as the forward path.

**A dependency trap to avoid:** the plugin depends on **`libtorch-neuronx-lite`**, which *vendors*
the TorchNeuron compile components into its own wheel instead of depending on `torch-neuronx`.
Installing `torch-neuronx` alongside it in the same environment is a conflict risk. Use
`torch-neuronx` only for standalone TorchNeuron work (Section 6), not inside the serving image.

---

## 2. Pinned stack (AWS Neuron SDK 2.33.0, released 2026-10-19)

| Component | Version | Notes |
|---|---|---|
| Neuron SDK | `2.33.0` | |
| TorchNeuron | `2.14.3.0` (Beta) | native PyTorch backend, `PrivateUse1` device |
| PyTorch | `2.14` (CPU wheel) | TorchNeuron rides on the CPU build |
| vLLM Neuron plugin | `0.29.0.1.2.0` (Beta) | targets vLLM `0.29.0` |
| Neuron Compiler (`neuronx-cc`) | `2.28.7631.0` | |
| Neuron Runtime | `2.35.30.0` | |
| Python | `3.14` in the DLC / DLAMI | plugin requires ≥ 3.11 |
| OS | Ubuntu 26.04 in the DLC | |

Both TorchNeuron and the vLLM plugin are **Beta**. Expect API churn and pin exactly.

**Primary references**
- TorchNeuron: `https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/index.html`
- TorchNeuron quickstart: `.../torch/get-started/quickstart-torch-neuron.html`
- Native PyTorch inference: `.../inference-native-pytorch.html`
- vLLM Neuron plugin: `.../vllm-neuron/docs/index.html`
- Setup guide: `.../vllm-neuron/docs/getting-started/setup-guide.html`
- Qwen3-32B recipe: `.../vllm-neuron/docs/model-recipes/qwen3-32b.html`
- Qwen3-32B tutorial: `.../vllm-neuron/docs/tutorials/tutorial-qwen3-32b.html`
- Precompile cache: `.../vllm-neuron/docs/guides/how-to-precompile-cache.html`
- Config reference: `.../vllm-neuron/docs/guides/reference-configuration.html`
- Benchmark methodology: `.../vllm-neuron/docs/tutorials/tutorial-prefix-caching-gpt-oss-benchmarking.html`
- Plugin source: `https://github.com/vllm-project/vllm-neuron` (branch `release-0.29.0.1.2.0`)
- Agent-friendly URL index: `https://awsdocs-neuron.readthedocs-hosted.com/llms.txt`

---

## 2a. How TorchNeuron and vLLM actually connect

The compile pipeline, end to end:

```
your model → TorchDynamo → FX graphs → AOT Autograd → Neuron FX passes
           → StableHLO MLIR (NO TorchInductor) → neuronx-cc → NEFF → Neuron Runtime
```

vLLM does not reimplement any of this. The plugin compiles each model *through*
TorchNeuron's `torch.compile` backend, so the serving stack and a standalone
`@torch.compile(backend="neuron")` script share one compiler path. That is why the Phase 1
smoke test is diagnostic: if `torch.compile(backend="neuron")` fails on a matmul, the fault is
TorchNeuron, not vLLM.

**`VLLM_NEURON_BACKEND` values** (two compilation routes):

| Value | Route | Lowering | Default opt level |
|---|---|---|---|
| unset | **native (default)** | AOTAutograd + torch-mlir → StableHLO → `neuronx-cc` | O2 |
| `neuron_native` | native | as above | O2 |
| `neuron` | native | as above | O2 |
| `vllm_neuron` | **XLA (deprecated)** | PyTorch/XLA → HLO → `neuronx-cc` | O1 |

Both routes compile identical `torch.compile` graphs onto the same runtime; only the lowering
differs. The XLA route is deprecated and slated for removal — use it only if the native route
produces results you cannot accept.

> Subtlety worth knowing: the native route is already the default, so setting
> `VLLM_NEURON_BACKEND=neuron_native` does not change the route. But the **precompile cache
> only activates when the variable is explicitly set**. Set it anyway.

### Three things this pipeline implies

**1. Dynamic shapes are not supported.** The backend forces `dynamic=False`; "each distinct
input shape requires separate compilation." This is the single most operationally important
fact in the whole stack: **buckets are not a tuning knob, they are the list of shapes that
exist.** A benchmark sweep over input lengths and concurrency must be covered by the server's
buckets or it recompiles mid-run and the latency numbers are garbage. Hence `SWEEP_MODE=1` in
`serve/run_qwen3_32b.sh`, kept deliberately separate from the published control config.

**2. Graph breaks degrade but do not fail.** When Dynamo hits an unsupported op or a
data-dependent branch it splits the graph and compiles each segment independently — narrower
optimisation scope, not an error. Trivial pass-through segments are skipped entirely.

**3. Caching is three-layered**, and one layer is explicitly designed for what the lab store
does:

| Layer | Keyed by | Portable? |
|---|---|---|
| Persistent NEFF cache | lowered HLO + compiler config (content-addressed) | **yes — "reusable across instances via shared storage"** |
| HLO cache | shared between eager and compiled modes | within a host |
| Dynamo precompile cache | embeds StableHLO bytes for warm reload | fingerprint-bound |

That middle column is why `labstore` keys on resolved versions and topology rather than on a
friendly name, and the quoted phrase is direct documentation support for putting the cache on
EFS/S3.

Also: only `torch.compile(mode="default")` is supported — other modes warn and fall back.

## 3. The hardware, and the core math that drives every config

`trn2.48xlarge`:

- **16** Trainium2 devices → `/dev/neuron0` … `/dev/neuron15`
- **8** NeuronCore-v3 per device → **128 physical cores**
- **96 GiB** HBM per device → **1,536 GiB** total HBM
- 192 vCPU, 2,048 GiB host RAM
- EFA-capable (the repo's CFN template enables it for `trn2.48xlarge`)

**Logical NeuronCore (LNC)** fuses physical cores into one logical unit:

| LNC | Logical cores/device | Logical cores/instance | Meaning |
|---|---|---|---|
| 1 | 8 | **128** | max parallelism, less memory+compute per unit |
| 2 | 4 | **64** | fewer units, more memory+compute each |

`NEURON_LOGICAL_NC_CONFIG=2` is what the published Qwen3-32B recipe uses.

**The consequence worth internalizing:** Qwen3-32B wants **TP8 at LNC2** = 8 logical cores =
**2 physical devices**. One `trn2.48xlarge` has 64 logical cores at LNC2, so it fits
**8 independent TP8 replicas**. That makes a single instance a legitimate throughput-scaling
testbed, not just a single-replica box — and the repo's existing `serve` script already builds
replica "core groups" automatically, so this is reachable.

> **O-1 RESOLVED on hardware 2026-11-07** (`i-02b1dc0805521506e`, trn2.48xlarge):
> `neuron-ls --json-output` reports **`"nc_count": 4`** — i.e. **logical** cores per device at
> LNC2, not physical. `neuron-ls` also confirms `logical-neuroncore-config: 2`, so **LNC2 is the
> instance default** and needs no explicit setting. 16 devices x 4 = 64 logical cores, core IDs
> `0-63`, 96 GB per device.
>
> Consequence: the repo's existing `serve` replica math (`len(devices) * nc_count`) is **correct
> as-is** on trn2 — no `NEURON_CORES_PER_DEVICE` override needed. With
> `TENSOR_PARALLEL_SIZE=8` it yields 8 core-groups of 8 logical cores, matching the 8-replica
> figure below.

**BF16 only on trn2.** The Qwen3-32B recipe offers MXFP8, but the docs state it needs **Trn3**;
Trn2 is BF16 for this model. Do not plan FP8 numbers on this instance.

---

## 4. Published Qwen3-32B recipe (the baseline to reproduce first)

Requires "at least eight available logical NeuronCores at LNC2".

```bash
export NEURON_LOGICAL_NC_CONFIG=2
export VLLM_NEURON_BACKEND=neuron_native
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200

vllm serve "$MODEL_PATH" \
  --served-model-name qwen3-32b \
  --dtype bfloat16 \
  --optimization-level 2 \
  --tensor-parallel-size 8 \
  --max-model-len 10240 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4
```

with `neuron_config` buckets:

```json
{"neuron_config": {
  "kv_segment_size_buckets":   [8192],
  "num_batched_tokens_buckets":[8192],
  "num_seqs_buckets":          [4]
}}
```

Checkpoint: `hf download Qwen/Qwen3-32B --revision "$BF16_REVISION" --local-dir "$BF16_MODEL"`

Validation per the tutorial: `curl localhost:8000/health`, then a completion for
"The capital of France is" expecting Paris, then 16 concurrent requests all returning non-empty.

**Reproduce this verbatim before changing a single flag.** It is the only configuration with
published AWS correctness + performance validation, so it is the control for everything after.

---

## 5. Phases

### Phase 0 — Prerequisites (off-instance, done)
- [x] trn2 capacity reserved (Capacity Block, us-east-2b)
- [x] DCV desktop stack armed to launch into the reservation
- [ ] `HF_TOKEN` available on the instance (Qwen3 is gated on some mirrors — verify)
- [ ] EFS or S3 path chosen for the **compile cache** (Phase 3) and benchmark artifacts

### Phase 1 — Verify the hardware and the native backend (~20 min on instance)
Cheap, fast, and it fails loudly if the driver stack is wrong. Do it before any model work.

1. `neuron-ls` and `neuron-ls --json-output` → confirm 16 devices, record `nc_count` (**O-1**).
2. `cat /proc/modules | grep neuron`, `modinfo neuron` → driver (`aws-neuronx-dkms`) loaded.
3. Standalone TorchNeuron smoke test in its own venv (**not** the serving image):
   ```bash
   python -m pip install "torch-neuronx==2.14.*" "neuronx-cc==2.*" nki \
     --extra-index-url https://pip.repos.neuron.amazonaws.com \
     --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   then the quickstart assertion: `torch.neuron.is_available()`, `torch.neuron.device_count()`,
   and a matmul on `.to("neuron")` compared against CPU with `torch.testing.assert_close`.
4. `@torch.compile(backend="neuron")` on the same matmul → exercises the Dynamo path that vLLM
   itself uses, so a failure here localises the problem to TorchNeuron rather than vLLM.

Deliverable: `notebooks/qwen3_32b_trn2_vllm.ipynb` §1–2.

### Phase 2 — Containers
**Primary — DLC base** (`containers/Dockerfile`): `FROM public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`,
then layer the serve/bench scripts. This is the low-risk path and the one to use inside the
capacity block.

**Reference — from source** (`containers/Dockerfile.from-source`): Ubuntu 26.04 + Python 3.14,
`neuronx-cc`, `nki`, `libtorch-neuronx-lite`, and the plugin built from
`git clone -b release-0.29.0.1.2.0 https://github.com/vllm-project/vllm-neuron.git` +
`pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .`.
This documents the full recipe and is what EKS hardening will eventually build from. **Build it
outside the block window** — dependency resolution is exactly the kind of thing that eats hours
of paid Trainium time.

Verify either image with:
```bash
python -c "import vllm; from vllm.platforms import current_platform; print(current_platform.device_name)"
# expect: neuron
```

Run flags (16 devices; the DLC docs enumerate each):
```bash
docker run -it --device=/dev/neuron0 ... --device=/dev/neuron15 \
  --cap-add SYS_ADMIN --cap-add IPC_LOCK -p 8000:8000 ...
```
The repo's `compose/neuron.yaml` already achieves this more simply with `privileged: true` and
`/dev:/dev` — prefer that for the single-instance example.

### Phase 3 — Compile cache strategy (the highest-leverage step)
First run traces on CPU and compiles; the docs default both timeouts to **1200s**, which tells you
the expected order of magnitude. On a time-boxed block this is the difference between collecting
data and watching a compiler.

- Set **before Python starts** (setting them in-process has no effect):
  `VLLM_NEURON_BACKEND=neuron_native`, `VLLM_CACHE_ROOT=/cache`.
- Cache path: `$VLLM_CACHE_ROOT/neuron/torch_compile/<fingerprint>/dp_<i>/rank_<r>`.
- Mount `/cache` to **EFS** (persists past instance termination; the CFN template provisions it)
  and sync to S3 as a backstop.
- Confirm warm start from startup logs: cold shows misses + saves > 0; warm shows Dynamo hits and
  **zero** saves.
- **The fingerprint invalidates on any version, compiler-flag, or config change.** So freeze the
  serve config *before* the sweep, precompile once per config you intend to benchmark, and treat
  each new bucket/TP combination as a fresh compile. Budget for this explicitly.
- Multi-node has no cache coordination → `TORCH_CACHING_PRECOMPILE=0` there. Also incompatible
  with encoder-disaggregated (EPD) serving.

`serve/precompile.sh` warms the cache for each planned config before any benchmarking starts.

### Phase 4 — Serve integration
`serve/run_qwen3_32b.sh` exports the Section 4 environment, generates the YAML the repo's
`/opt/program/serve` consumes, and launches. Mapping onto the repo's existing convention
(`launch.sh` → `scripts/${SERVER}-${ENGINE}-${DEVICE}.sh`, image `${SERVER}-${ENGINE}:${DEVICE}`,
run scripts from `${SERVER}/${ENGINE}/`):

| Convention slot | Value |
|---|---|
| `INFERENCE_SERVER` | `openai_server` |
| `INFERENCE_ENGINE` | `vllm_native` |
| `DEVICE` | `neuron` |
| build script | `scripts/openai_server-vllm_native-neuron.sh` |
| image | `openai_server-vllm_native:neuron` |
| run scripts | `openai_server/vllm_native/run.sh` |

Folding in requires `compose/neuron.yaml` to pass the new variables
(`NEURON_LOGICAL_NC_CONFIG`, `VLLM_NEURON_BACKEND`, `VLLM_CACHE_ROOT`,
`VLLM_NEURON_COMPILATION_TIMEOUT`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`). Extra env vars are
inert for the legacy image, so this is a non-breaking addition. Deferred until the standalone
example works — don't destabilise a working path for an unvalidated one. **Open item O-2.**

### Phase 5 — Benchmarking and SLO determination
Two harnesses, cross-validated.

**Source of truth — `vllm bench serve`** (built into vLLM 0.29). Reports `median_ttft_ms`,
`p99_ttft_ms`, ITL, TPOT, `output_throughput`; supports `--max-concurrency`, `--request-rate`,
`--ignore-eos`, `--save-result`.

Fix the token shape so input/output lengths are controlled rather than observed — `--ignore-eos`
is essential or output length becomes a function of the model's stopping behaviour and ITL
comparisons stop meaning anything.

Sweep (`bench/run_vllm_bench.sh`):
- Token shapes: 1k/128, 4k/256, 8k/1k input/output — spanning prefill- and decode-bound regimes
- Concurrency: 1, 2, 4, 8, 16, 32 (1 = latency floor; raise until TTFT degrades)
- `--request-rate inf` for saturation, plus fixed rates for steady-state
- `--num-prompts` ≥ 10x concurrency so percentiles mean something

Then `bench/parse_results.py` folds the JSON into one CSV/table.

**Cross-validation — Locust** (`bench/locustfile.py`), reusing the repo's `run_locust.sh` +
`modules/` prompt generators. Compute ITL from streamed chunk arrival timestamps. Agreement within
~10% on median TTFT and throughput validates both; disagreement means one harness is
mis-instrumented and the numbers are not yet publishable.

**Deriving the SLO** — the actual deliverable:
1. At concurrency 1, record the **floor**: best achievable TTFT and ITL. No SLO can beat this.
2. Sweep concurrency and plot TTFT-p99 and ITL-p99 against throughput.
3. Find the **knee** — the concurrency beyond which latency rises faster than throughput.
4. Set the SLO just inside the knee, e.g. "p99 TTFT ≤ X ms and p99 ITL ≤ Y ms at N concurrent
   requests per TP8 replica, 4k input / 256 output, BF16".
5. Record cost per million tokens at that point, derived from the block's hourly rate.
6. Re-run at 2 and 4 replicas to confirm the per-replica SLO holds under instance-level contention
   — this is what makes the number meaningful for EKS capacity planning rather than a lab curio.

An SLO without its token shape, concurrency, replica count and dtype attached is not reusable.
Record all five with every number.

### Phase 6 — The notebook
`notebooks/qwen3_32b_trn2_vllm.ipynb`, runnable top-to-bottom on the desktop over DCV:
1. Hardware + driver inventory (`neuron-ls`, LNC)
2. TorchNeuron smoke test (eager + `torch.compile`)
3. Build/pull the image
4. Download the checkpoint
5. Precompile (cache warm)
6. Serve + health + correctness ("capital of France", 16-way concurrency)
7. `vllm bench serve` sweep
8. Locust cross-check
9. Parse → SLO table + TTFT/ITL-vs-throughput plots
10. Teardown, and push cache + results to EFS/S3

### Phase 7 — The other two Qwen3 variants
Different architectures, so each needs its own pass. Sequence them after the dense model works —
they share the container and the harness, so marginal cost is low once Phase 5 is solid.

| Model | What it exercises | Benchmark caveat |
|---|---|---|
| **Qwen3-32B Dense** | baseline decoder, TP8/LNC2/BF16 | the control |
| **Qwen3-VL-32B** | vision encoder, M-RoPE, on-device encoder cache, block-packing attention | TTFT includes image encode; report text and image TTFT separately or the number is meaningless |
| **Qwen3-Embedding-8B** | pooling models, no autoregressive decode | TTFT/ITL **do not apply** — SLO is request latency and embeddings/sec vs batch size |

### Phase 8 — EKS (scoped later, not designed here)
What this phase should carry forward, noted now so Phase 1–6 artifacts don't block it:
- The **from-source Dockerfile** becomes the production image; push to private ECR.
- Neuron device plugin + scheduling: `aws.amazon.com/neuron` resource requests, node selectors for
  `trn2.48xlarge`, and the fact that **TP8 at LNC2 = 2 devices**, so a pod requests a slice, not
  the whole node — up to 8 pods per instance.
- The **compile cache must be a shared RWX volume** (EFS CSI) or baked into the image. Per-pod cold
  compiles at ~20 min would make autoscaling unusable. Baking it in is probably correct for prod.
- `TORCH_CACHING_PRECOMPILE=0` if any deployment spans nodes.
- Readiness probes must tolerate a multi-minute warm start; `/health` plus a generous
  `initialDelaySeconds`/`failureThreshold`.
- The repo's `ray-vllm-trn2` and `ai-on-eks` work next door are the likely integration points.

---

## 6. TorchNeuron direct usage (beyond serving)

Worth a short notebook section independent of vLLM, since it is the part of the stack the user
specifically wants to evaluate and it is where the "native PyTorch" claim is actually visible:

- Device: `torch.device("neuron")`; autoloads on `import torch` via `PrivateUse1` — no explicit
  `import torch_neuronx` needed for basic use.
- Port pattern: `.to('cuda')` → `.to('neuron')`; `torch.autocast(device_type="neuron")`.
- Compile: `@torch.compile(backend="neuron")`; drop CUDA-specific kwargs.
- Distributed: DDP, FSDP2, DTensor TP/SP, PP, CP all work through stock `torch.distributed`,
  sharding over NeuronCores. Relevant to the training side of this repo (`gen-ai-training-testing`),
  and a plausible follow-on.

---

## 7. Risks

| # | Risk | Mitigation |
|---|---|---|
| R-1 | Compile time consumes the block | Precompile first, persist cache to EFS, freeze configs before sweeping |
| R-2 | Both key components are **Beta**; APIs shift | Pin every version exactly; record `pip freeze` in results |
| R-3 | `nc_count` ambiguity breaks replica math (**O-1**) | Resolve in Phase 1 before any serving; set `NEURON_CORES_PER_DEVICE` explicitly |
| R-4 | `torch-neuronx` vs `libtorch-neuronx-lite` conflict | Separate venvs: standalone TorchNeuron vs serving image |
| R-5 | Benchmarks disagree | Treat as a blocker, not an average; `--ignore-eos` and fixed token shapes in both |
| R-6 | MXFP8 assumed available | It needs Trn3. BF16 only here. |
| R-7 | Block expires with artifacts on root EBS | Everything durable goes to EFS/S3 *during* the run, not at the end |
| R-8 | Folding into `launch.sh` breaks the legacy path | Keep standalone until validated; env additions are inert (**O-2**) |

## 8. Open items

- ~~**O-1** Does `neuron-ls --json-output` report `nc_count` as physical (8) or logical (4) per device at LNC2?~~
  **RESOLVED 2026-11-07: logical (4).** LNC2 is the instance default; existing replica math is correct. See Section 3.
- **O-2** When to fold into `launch.sh`/`compose/neuron.yaml` vs stay standalone.
- ~~**O-3** Is `Qwen/Qwen3-32B` gated? Confirm `HF_TOKEN` need and pin `BF16_REVISION` for reproducibility.~~
  **RESOLVED 2026-11-13: not gated.** Downloaded with no `HF_TOKEN` supplied. 62 GB on disk,
  **65,540,308,523 bytes / 84 objects / 18 safetensors shards** staged to
  `s3://dl-lab-store-489829964455-us-east-2/lab/models/qwen-qwen3-32b`.
  Pinning a revision SHA is still worth doing before publishing numbers; `main` was used here.
- **O-4** Exact `--optimization-level` semantics (recipe uses O2 single-server, O3 for disaggregated decode).
- ~~**O-5** Does a 1k-input shape need its own bucket config, given buckets are pinned at 8192?~~
  **RESOLVED from docs: yes.** The Neuron `torch.compile` backend forces `dynamic=False`, so
  every distinct shape compiles separately. Buckets must enumerate every shape the sweep will
  use. Handled by `SWEEP_MODE=1`, which is a separate fingerprint from the control config. See
  Section 2a.
- **O-6** Confirm the plugin's own `benchmarks/` directory contents — may supersede `bench/` here.
