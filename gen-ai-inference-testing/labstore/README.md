# labstore — durable session store for ephemeral accelerator instances

Trainium capacity arrives as Capacity Blocks measured in days. When a block expires the instance
is **terminated** and its root EBS volume deleted. This keeps the expensive artifacts somewhere
that outlives the host, and restores them onto a fresh instance in minutes.

Works for **GPU and Trainium alike**, and is **entirely opt-in** — nothing here runs unless you
run it, and no existing project file is modified.

## Three tiers

Putting an artifact in the wrong tier is the main way this gets *slower* rather than faster.

| Tier | Holds | Why |
|---|---|---|
| **S3** | weights, cache backups, results | durable, ~$0.023/GB-mo, fast parallel fan-out |
| **EFS** (Elastic) | compile caches, manifests, results | instantly mounted, no hydration step |
| **Local NVMe** | weights during a session | `trn2.48xlarge` has **7.6 TB** of it, unused by default |

Weights do **not** belong on EFS: bursting-mode throughput scales with *stored* data, EFS costs
~13x S3 at rest, and local NVMe beats NFS for large sequential reads.

## Commands

```bash
labstore.sh nvme-setup     # RAID0 + mount instance-store NVMe (idempotent, needs sudo)
labstore.sh init           # create the store layout, write a provenance snapshot
labstore.sh status         # what is warm vs cold -- and WHY, if cold
labstore.sh push-weights   # local checkpoint -> S3 (run on a cheap box, not Trainium)
labstore.sh resume         # restore compile cache + hydrate weights to local
labstore.sh save           # cache + manifest + results -> EFS and S3
```

`provenance.py` can be used on its own:

```bash
python3 provenance.py --model-id Qwen/Qwen3-32B --topology tp8-lnc2
python3 provenance.py --key cache      # just the cache key, for shell use
python3 provenance.py --json           # full manifest
```

## Why caches are keyed, not named

A compiled artifact is reusable only where the *relevant* environment matches. A Trainium NEFF
invalidates on Neuron SDK version, plugin version, torch version, LNC and TP degree; a CUDA
TensorRT engine invalidates on GPU model and driver. None of that is visible in a filename.

So entries are keyed by a **stack hash** over exactly those facts:

```
cache/<model-slug>/<accel-class>-<accel-model>/<stack-hash>/<topology>/
models/<model-slug>/                                  # no accelerator fields
```

Consequences, each verified by test:

- **Weights are shared across hardware** — `models/qwen-qwen3-32b` is identical on a GPU box and
  a Trainium box, so one download serves both.
- **Compile caches cannot collide** — `neuron-trainium2/...` and `cuda-h100/...` are distinct keys.
- **Host size does not invalidate** — `device_count` is excluded from the hash, so a TP8 cache
  built on a 16-device host still hits on a 4-device host. The parallelism degree lives in
  `topology`, which *is* hashed.
- **A version bump invalidates, and says so** — `status` diffs the stored `MANIFEST.json` against
  the live machine and names the field:
  ```
  frameworks.vllm-neuron: cached='0.29.0.1.2.0' live='0.29.0.1.3.0'
  ```
  A silent cache miss costs ~20 minutes and looks like a mystery; this is the whole point of the
  manifest.

## Typical use

**Before a block, on a cheap instance** — removes the download from accelerator time entirely:
```bash
MODEL_ID=Qwen/Qwen3-32B labstore.sh push-weights
```

**On a fresh Trainium instance:**
```bash
sudo -E labstore.sh nvme-setup
labstore.sh init
eval "$(labstore.sh resume | grep ^export)"   # sets VLLM_CACHE_ROOT and MODEL_PATH
# ... run the workload ...
RESULTS_DIR=./results labstore.sh save
```

`trn2-qwen3-torchneuron/bootstrap.sh` already does this automatically, falling back to a flat S3
prefix if labstore is absent.

## Bridging the existing harness (optional)

`launch.sh` and `compose/{cuda,neuron}.yaml` mount `$HOME/snapshots` and `$HOME/cache`, both of
which live on the root EBS volume today — so nothing persists. `link-harness.sh` redirects them
into the store by symlink, **without editing any harness file**:

```bash
link-harness.sh link      # $HOME/snapshots and $HOME/cache -> store
link-harness.sh status
link-harness.sh unlink    # revert to plain local directories
```

It migrates (never deletes) existing contents, preserving the original as
`*.pre-labstore.<timestamp>`. `$HOME/snapshots` maps to a shared location since the HF cache is
hardware-independent; `$HOME/cache` maps to a per-accelerator subdirectory so a NEFF is never
handed to a CUDA runtime.

Note that EFS is slower than local disk for large sequential reads, so for heavy weight loading
prefer `labstore resume`, which hydrates to NVMe.

## Infrastructure

Provisioned by `lab-store.yaml` at the repo root (stack `dl-lab-store`), which is **permanent and
independent of any desktop stack**:

| Resource | Value |
|---|---|
| EFS | `fs-090ea92bfa28de363` — `elastic`, encrypted, mount targets in **all 3 AZs** |
| S3 | `dl-lab-store-489829964455-us-east-2` |
| ECR | `489829964455.dkr.ecr.us-east-2.amazonaws.com/dl-lab-inference` |

Desktop stacks consume it by passing **only** `EFSFileSystemId` to the unmodified
`deep-learning-ubuntu-desktop.yaml`. Mount targets exist in every AZ because a Capacity Block
picks its own AZ — `cr-0ca15816889289192` landed in `us-east-2a`, not the `2b` used previously,
and an AZ without a mount target cannot reach the store at all.
