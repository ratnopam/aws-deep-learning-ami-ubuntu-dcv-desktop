# Durable Lab Store — surviving ephemeral accelerator instances

**Problem.** Trainium capacity arrives as Capacity Blocks measured in days. When the block
expires the instance is terminated and its root EBS volume deleted. Everything expensive —
downloaded weights, compiled NEFF artifacts, benchmark results — dies with it. The 2026-11-07
block produced **zero** surviving artifacts.

**Goal.** A new instance should reach "serving a model" in minutes rather than an hour, by
restoring prior work from durable storage.

**Constraint (explicit):** this must **not disrupt the existing workflow**, and must work for
**all hardware** — CUDA GPU desktops as well as Trainium. It is purely additive and opt-in.

---

## 1. Why "just use EFS" would make things slower

The obvious design — put everything on EFS — is wrong, for two measured reasons.

**EFS throughput scales with stored data.** `fs-0667c8d3d8eee3f52` was in `bursting` mode holding
12 KB. Bursting baseline is ~50 KiB/s per GiB stored, so a near-empty filesystem has
*essentially no throughput*. Reading a 65 GB checkpoint from it would be far slower than
re-downloading from HuggingFace.

**`trn2.48xlarge` has 7,600 GB of local NVMe** (4 x 1,900 GB SSD), plus 16x200 Gbit networking
and 10,000 MB/s to EBS. Local NVMe beats any network filesystem for large sequential reads.

So the answer is **three tiers**, and the main way this goes wrong is putting an artifact in the
wrong one:

| Tier | Role | Survives termination | Cost |
|---|---|---|---|
| **S3** | durable source of truth | yes | ~$0.023/GB-mo |
| **EFS** (Elastic mode) | warm cache, instantly mounted, no hydration step | yes | ~$0.30/GB-mo + $0.03/GB read |
| **Local NVMe** (7.6 TB) | hot working set for a session | **no** — by design | included |

### Placement

| Artifact | Size | Regeneration cost | Home |
|---|---|---|---|
| Model weights | 15-650 GB | 10-60 min download | **S3** -> hydrate to NVMe |
| Compile cache (NEFF / inductor / TRT engines) | GBs | **~20 min per config** | **EFS** + S3 backup |
| Converted / sharded checkpoints | ~weights | minutes-hours | S3 -> NVMe |
| Benchmark results, plots | MB | irreplaceable | EFS + S3 + git |
| Run manifests (versions, env, topology) | KB | irreplaceable | EFS + git |
| pip wheelhouse | 2-5 GB | 5-15 min | EFS |
| Container images | 10-30 GB | 10-20 min | **ECR** — not EFS |
| Accelerator profiles (NTFF / nsys) | MB-GB | rerun | EFS + S3 |

The compile cache is the crown jewel: best ratio of regeneration cost to size, and it is read at
process start, so instant availability genuinely matters. Weights on EFS would cost 13x S3
storage and read slower than NVMe.

---

## 2. Non-disruption: zero changes to existing templates

Verified facts that make this clean:

- `$HOME/snapshots` and `$HOME/cache` (mounted into containers by `compose/{cuda,neuron}.yaml`)
  live on the **root EBS volume**, not on EFS. The only EFS mention in
  `gen-ai-inference-testing/README.md` ("EFS-based model storage") is aspirational — nothing
  wires it. **So today nothing persists, and anything we add is additive.**
- `deep-learning-ubuntu-desktop.yaml` already has an `EFSFileSystemId` parameter. Supplying it
  sets `CreateNewEFSFileSystem: false`, so the template creates no EFS, no mount target and no
  EFS security group — it just mounts what you give it.
- The template's `DesktopSecurityGroupCIDREgress` rule allows **TCP 0-65535 to 0.0.0.0/0**, which
  covers NFS 2049. So a template-created desktop security group can already reach a shared EFS.

**Consequence:** if the shared EFS accepts NFS from the VPC CIDR, callers need to pass only
`EFSFileSystemId`. They do **not** need `DesktopSecurityGroupId`, and the desktop template is
**not modified at all**. GPU stacks that pass nothing keep behaving exactly as they do today.

```
                 ┌──────────────────────────────┐
                 │  lab-store stack (permanent) │
                 │  EFS(Elastic) + MTs in 3 AZs │
                 │  S3 prefixes + ECR repo      │
                 └──────────────┬───────────────┘
                                │ exports EfsId
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  trn2 desktop            g6 GPU desktop          p5 GPU desktop
  (ephemeral)              (ephemeral)             (ephemeral)
  EFSFileSystemId=…        EFSFileSystemId=…       (omit → old behaviour)
```

---

## 3. Hardware-agnostic by construction

One API, different payloads. The cache key encodes what the artifact actually depends on:

```
<artifact-class>/<model-slug>/<accel-class>-<accel-model>/<stack-hash>/<topology>/
```

| Field | Trainium example | CUDA example |
|---|---|---|
| `accel-class` | `neuron` | `cuda` |
| `accel-model` | `trainium2` | `h100` / `l40s` |
| `stack-hash` | sdk 2.33.0 + plugin 0.29.0.1.2.0 + torch 2.14 | driver 560 + cuda 12.6 + trtllm x.y |
| `topology` | `tp8-lnc2` | `tp8` |

This generalises correctly. TensorRT-LLM engines are notoriously GPU-architecture-specific, so the
same discipline that stops a Trainium NEFF being reused across SDK versions stops a TRT engine
being reused across GPU models. Weights and results are hardware-independent and keyed without
the accelerator fields, so a GPU box and a Trainium box **share one weight cache**.

---

## 4. The real correctness risk: silent cache misses

The `torch_compile` cache fingerprint invalidates on **any** change to versions, compiler flags,
TP, LNC, or bucket config. A cache that looks warm can silently miss, and you lose 20 minutes
wondering why.

Mitigation: every cache entry carries a `MANIFEST.json` recording resolved versions, topology,
flags and model revision. `labstore restore` **diffs** the manifest against the live environment
and reports *which field* changed — "cold: plugin 0.29.0.1.2.0 -> 0.29.0.1.3.0" — instead of
failing quietly. Never report a restore as warm without verifying the fingerprint inputs match.

---

## 5. Implementation phases

### Phase 1 — persistent layer (CloudFormation, off-block, ~1h, a few dollars)
New stack `lab-store.yaml`, independent of any desktop stack:
- EFS, **`elastic`** throughput, encrypted, `DeletionPolicy: Retain`
- **Mount targets in all three AZs.** Mount targets are per-AZ and free; the old EFS had one, in
  `us-east-2b` only. Nothing guarantees the next Capacity Block lands in 2b, and a missing mount
  target means no EFS at all.
- EFS security group: ingress TCP 2049 from the VPC CIDR
- S3 bucket with versioning and an IA lifecycle rule
- ECR repository for built images
- Exported outputs: `EfsId`, `BucketName`, `EcrRepoUri`

Replaces `fs-0667c8d3d8eee3f52` (bursting, empty). Verify by mounting from a cheap `m5.large` and
measuring read throughput before declaring it done.

### Phase 2 — provenance and cache contract
`lab/provenance.py` — detect accelerator class/model, resolve framework and SDK versions, emit a
stable `stack-hash`. `MANIFEST.json` schema. `lab/cache.py` — `save` / `restore` / `explain`.

### Phase 3 — session lifecycle
`labstore init | resume | save | status`. `resume` mounts EFS, reads state, hydrates NVMe from S3
in parallel, verifies manifests, prints a warm/cold table. Includes **NVMe setup** — the four
local disks need RAID0 and a mount, which nothing currently does.

### Phase 4 — pre-stage from a cheap instance
Populate S3 with weights from an `m5`/`inf2` box so accelerator time is never spent downloading.
Biggest single win, and it needs no Trainium at all.

### Phase 5 — integrate, additively
`bootstrap.sh` calls the session API instead of its ad-hoc cache handling. For the existing
harness, provide an **opt-in** hook: when `LAB_STORE_ROOT` is set, `$HOME/snapshots` and
`$HOME/cache` are symlinked into the store; when unset, behaviour is byte-for-byte unchanged.
No edits to `launch.sh`, `compose/*.yaml`, or the desktop templates.

### Phase 6 — prove it
Two consecutive blocks: first cold, second resumed. Metric: **time from instance running to first
token served.** Target under 10 min warm versus ~60-90 min cold. Also prove the GPU path by
restoring the same weight cache onto a `g6` desktop.

---

## 6. Scope

Start with the models the vLLM Neuron plugin actually supports (recipe index, SDK 2.33.0):

| Model | Exercises |
|---|---|
| **Qwen3-32B dense** | baseline decoder, TP8/LNC2/BF16 |
| **Llama-3-70B** | large dense, higher TP |
| **GPT-OSS-120B** | MoE path |

**DeepSeek and GLM are not supported** by the plugin — absent from the recipe index. Running them
means onboarding a new architecture (there is a documented guide, but it is a development project,
not a config change). Two further realities: the NKI library already ships DeepSeek-V3 Router and
V3.2 Sparse Indexer kernels, so AWS is building toward it; and DeepSeek-V3/R1 at 671B BF16 is
~1,342 GB against 1,536 GB of HBM, leaving ~194 GB for KV cache — marginal on a single node even
once supported. The store is model-agnostic and will be ready; the models are the blocker.

---

## 7. Risks

| # | Risk | Mitigation |
|---|---|---|
| L-1 | EFS left in `bursting` mode -> unusably slow | `elastic` enforced in the template; verify throughput in Phase 1 |
| L-2 | Capacity Block lands in an AZ with no mount target | mount targets in all three AZs from the start |
| L-3 | Silent cache miss after a version bump | manifest diffing; `explain` reports the changed field |
| L-4 | Weights on EFS -> slow and 13x the storage cost | placement table is normative; weights go S3 -> NVMe |
| L-5 | Changes break the GPU workflow | additive only; opt-in via `LAB_STORE_ROOT`; no template edits |
| L-6 | NVMe assumed present | instance-store varies by type; detect and fall back to EBS |
| L-7 | NFS open to whole VPC CIDR | acceptable in a single-tenant dev account; tighten to SG-to-SG if that changes |
| L-8 | Store grows without bound | S3 lifecycle to IA; prune caches whose stack-hash is stale |

## 8. Cost

EFS Elastic at ~$0.30/GB-month plus $0.03/GB read; ~50 GB of caches is ~$15/month. S3 at 500 GB
is ~$12/month. Against **$858/day** for a trn2 Capacity Block, the layer repays itself the first
time it saves half an hour.
