#!/usr/bin/env bash
# Unattended pre-staging for a trn2 Capacity Block session.
#
# Runs ON the trn2 desktop, detached, triggered over SSM right after the
# CloudFormation stack completes. Its job is to burn through everything slow
# BEFORE a human sits down, so the paid Trainium window is spent on actual
# testing rather than on apt, a ~65 GB download and a cold compile.
#
# Because a Capacity Block is prepaid for its whole window, instance time
# before you start costs nothing extra. This script monetises that.
#
# Phases: wait -> clone -> cache-restore -> image -> checkpoint -> precompile
#         -> serve -> validate -> cache-backup
#
# Progress is written to $STATUS so a remote poller can report phase without
# streaming the whole log. Everything durable goes to EFS and S3 as it is
# produced, not at the end -- the host is reclaimed when the block expires.
#
# UNVERIFIED end-to-end. Individual commands mirror the validated recipe in
# PLAN.md but this orchestration has not yet run on hardware.
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/ratnopam/aws-deep-learning-ami-ubuntu-dcv-desktop.git}"
BRANCH="${BRANCH:-trn2-qwen3-torchneuron-vllm}"
S3_BUCKET="${S3_BUCKET:-genai-trn2-489829964455-us-east-2}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
BF16_REVISION="${BF16_REVISION:-main}"
DLC="${DLC:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04}"
IMAGE_TAG="${IMAGE_TAG:-openai_server-vllm_native:neuron}"

HOME_DIR="${HOME_DIR:-/home/ubuntu}"
EFS="${EFS:-$HOME_DIR/efs}"
WORK="$EFS/trn2-qwen3"
CACHE_DIR="$WORK/compile-cache"
MODEL_DIR="$WORK/models/$(basename "$MODEL_ID")"
REPO_DIR="$HOME_DIR/aws-deep-learning-ami-ubuntu-dcv-desktop"
STATUS="$HOME_DIR/bootstrap-status"
LOG="$HOME_DIR/bootstrap.log"

exec >>"$LOG" 2>&1

phase() { echo "PHASE=$1 AT=$(date -u +%Y-%m-%dT%H:%M:%SZ) ${2:-}" | tee -a "$STATUS"; }
fail()  { echo "PHASE=FAILED AT=$(date -u +%Y-%m-%dT%H:%M:%SZ) WHERE=$1" | tee -a "$STATUS"; exit 1; }

echo "=============================================================="
echo "bootstrap start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="
: >"$STATUS"

# --- 1. wait for the instance to actually be usable ----------------------
# cloud-init installs the Neuron driver via DKMS and reboots; /dev/neuron*
# does not exist until after that. Docker may also still be installing.
phase WAIT "waiting for neuron devices + docker"
for i in $(seq 1 90); do
  NODES=$(ls -1 /dev/neuron* 2>/dev/null | wc -l | tr -d ' ')
  DOCKER_OK=$(docker info >/dev/null 2>&1 && echo yes || echo no)
  if [[ "$NODES" -gt 0 && "$DOCKER_OK" == yes ]]; then
    phase WAIT_OK "devices=$NODES docker=ready after ${i}0s"
    break
  fi
  sleep 10
done
NODES=$(ls -1 /dev/neuron* 2>/dev/null | wc -l | tr -d ' ')
[[ "$NODES" -gt 0 ]] || fail "no neuron devices after 15min"
docker info >/dev/null 2>&1 || fail "docker not ready after 15min"

/opt/aws/neuron/bin/neuron-ls 2>&1 | head -8 || true
mkdir -p "$WORK" "$CACHE_DIR" "$MODEL_DIR" || fail "cannot write to EFS at $WORK"

# --- 2. clone the branch -------------------------------------------------
phase CLONE
if [[ -d "$REPO_DIR/.git" ]]; then
  git -C "$REPO_DIR" fetch origin "$BRANCH" && git -C "$REPO_DIR" checkout "$BRANCH" \
    && git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
else
  git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR" || fail clone
fi
EX="$REPO_DIR/gen-ai-inference-testing/trn2-qwen3-torchneuron"
[[ -d "$EX" ]] || fail "example dir missing after clone"
chmod +x "$EX"/serve/*.sh "$EX"/bench/*.sh 2>/dev/null || true

# --- 3. durable store: NVMe, then restore cache and weights --------------
# Delegated to labstore (gen-ai-inference-testing/labstore), which keys the
# compile cache by accelerator class/model + stack hash + topology, so a
# restore is only reported warm when the fingerprint inputs actually match.
# If labstore is unavailable this falls back to the old flat S3 prefix, so
# the bootstrap still works standalone.
LABSTORE="$REPO_DIR/gen-ai-inference-testing/labstore/labstore.sh"
export LAB_EFS_ROOT="${LAB_EFS_ROOT:-$EFS/lab}"
export LAB_S3_BUCKET="${LAB_S3_BUCKET:-$S3_BUCKET}"
export MODEL_ID MODEL_REVISION="$BF16_REVISION"
export TOPOLOGY="${TOPOLOGY:-tp8-lnc2}"

if [[ -x "$LABSTORE" ]]; then
  # 7.6 TB of instance-store NVMe on trn2.48xlarge is unused by default;
  # weights belong there, not on EFS or the 1 TB root volume.
  phase NVME
  sudo -E "$LABSTORE" nvme-setup 2>&1 | tail -3 || phase NVME_WARN "continuing on EBS"
  if mountpoint -q /mnt/nvme 2>/dev/null; then
    export LAB_LOCAL_ROOT=/mnt/nvme/lab
    phase NVME_OK "$(df -h /mnt/nvme | tail -1 | awk '{print $2" usable"}')"
  else
    export LAB_LOCAL_ROOT="$HOME_DIR/lab-local"
    phase NVME_SKIP "no instance store; using $LAB_LOCAL_ROOT"
  fi

  phase STORE_INIT
  "$LABSTORE" init 2>&1 | tail -4

  phase STORE_RESUME "restoring cache + hydrating weights"
  RESUME_OUT=$("$LABSTORE" resume 2>&1) || true
  echo "$RESUME_OUT"
  # resume emits the exports it wants us to use; adopt them.
  eval "$(printf '%s\n' "$RESUME_OUT" | grep -E '^export (VLLM_CACHE_ROOT|MODEL_PATH)=')" || true
  CACHE_DIR="${VLLM_CACHE_ROOT:-$CACHE_DIR}"
  MODEL_DIR="${MODEL_PATH:-$MODEL_DIR}"
  phase STORE_RESUME_OK "cache=$CACHE_DIR model=$MODEL_DIR"
else
  phase CACHE_RESTORE "labstore unavailable; flat S3 fallback"
  if aws s3 ls "s3://$S3_BUCKET/trn2-qwen3/compile-cache/" >/dev/null 2>&1; then
    aws s3 sync "s3://$S3_BUCKET/trn2-qwen3/compile-cache/" "$CACHE_DIR/" --only-show-errors \
      && phase CACHE_RESTORE_OK "files=$(find "$CACHE_DIR" -type f | wc -l | tr -d ' ')"
  else
    phase CACHE_RESTORE_SKIP "no prior cache in s3"
  fi
fi
mkdir -p "$CACHE_DIR" "$MODEL_DIR"

# --- 4. container image -------------------------------------------------
# Prefer the pre-built, off-block-validated image from ECR. It was built and
# pushed on 2026-11-13 from this same Dockerfile, and its vLLM import was
# confirmed working (it failed only on "No neuron device available", which is
# the correct failure on a non-accelerator host). Pulling it skips a ~4.5 min
# build and removes build failure from the critical path.
ECR_IMAGE="${ECR_IMAGE:-489829964455.dkr.ecr.us-east-2.amazonaws.com/dl-lab-inference:qwen3-trn2-validated}"
ECR_REGISTRY="${ECR_IMAGE%%/*}"
phase IMAGE "trying prebuilt $ECR_IMAGE"
if aws ecr get-login-password --region "${AWS_REGION:-us-east-2}" 2>/dev/null \
     | docker login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null 2>&1 \
   && docker pull "$ECR_IMAGE" >/dev/null 2>&1; then
  docker tag "$ECR_IMAGE" "$IMAGE_TAG" || fail "docker tag"
  phase IMAGE_PREBUILT "pulled validated image from ECR"
else
  phase IMAGE_BUILD "ECR unavailable -- building from $DLC"
  docker pull "$DLC" || fail "docker pull"
  # buildx may not be present on every base; fall back to classic build.
  if docker buildx version >/dev/null 2>&1; then
    ( cd "$EX" && docker buildx build -t "$IMAGE_TAG" -f containers/Dockerfile . ) || fail "docker buildx build"
  else
    ( cd "$EX" && docker build -t "$IMAGE_TAG" -f containers/Dockerfile . ) || fail "docker build"
  fi
fi
# Confirm the plugin reports the neuron platform. Needs devices, hence
# --privileged, and it is cheap insurance before a 20-minute compile.
PLAT=$(docker run --rm --privileged -v /dev:/dev "$IMAGE_TAG" \
  python -c "import vllm;from vllm.platforms import current_platform;print(current_platform.device_name)" 2>&1 | tail -1)
phase IMAGE_OK "platform=$PLAT"
[[ "$PLAT" == "neuron" ]] || fail "platform probe returned '$PLAT', expected 'neuron'"

# --- 5. checkpoint ------------------------------------------------------
# Onto EFS so it survives the block and a later run skips the download.
phase CHECKPOINT "downloading $MODEL_ID -> $MODEL_DIR"
if [[ -f "$MODEL_DIR/config.json" ]]; then
  phase CHECKPOINT_SKIP "already present ($(du -sh "$MODEL_DIR" | cut -f1))"
else
  MODEL_ID="$MODEL_ID" BF16_REVISION="$BF16_REVISION" DEST="$MODEL_DIR" \
    bash "$EX/serve/download_model.sh" || fail "checkpoint download"
  phase CHECKPOINT_OK "$(du -sh "$MODEL_DIR" | cut -f1)"
fi

# Seed S3 with the checkpoint so EVERY FUTURE block skips the HuggingFace
# download entirely -- `labstore resume` then hydrates from S3 in-region,
# which is far faster and cannot be rate-limited by a third party. This runs
# inside the prepaid pre-stage window, so it costs no testing time.
if [[ -x "$LABSTORE" ]]; then
  phase WEIGHTS_PUSH "seeding S3 for future sessions"
  if WEIGHTS_SRC="$MODEL_DIR" "$LABSTORE" push-weights 2>&1 | tail -3; then
    phase WEIGHTS_PUSH_OK
  else
    phase WEIGHTS_PUSH_WARN "push failed; weights still on local/EFS"
  fi
fi

# --- 6. precompile ------------------------------------------------------
# The single highest-leverage step. VLLM_NEURON_BACKEND and VLLM_CACHE_ROOT
# must be in the environment before Python starts.
phase PRECOMPILE "this is the long one"
docker run --rm --privileged \
  -v /dev:/dev \
  -v "$MODEL_DIR:/model:ro" \
  -v "$CACHE_DIR:/cache:rw" \
  -e MODEL_PATH=/model \
  -e VLLM_CACHE_ROOT=/cache \
  -e VLLM_NEURON_BACKEND=neuron_native \
  -e NEURON_LOGICAL_NC_CONFIG=2 \
  --shm-size 32g \
  "$IMAGE_TAG" bash /opt/program/serve/precompile.sh || fail precompile
phase PRECOMPILE_OK "cache files=$(find "$CACHE_DIR" -type f | wc -l | tr -d ' ')"

# --- 7. back the cache up immediately -----------------------------------
# Before serving, not after. If anything later goes wrong, the expensive
# artifact is already safe. labstore also writes a MANIFEST.json recording
# the exact versions and topology, without which the next run cannot tell
# whether the cache is reusable.
phase CACHE_BACKUP
if [[ -x "$LABSTORE" ]]; then
  VLLM_CACHE_ROOT="$CACHE_DIR" "$LABSTORE" save 2>&1 | tail -5 \
    && phase CACHE_BACKUP_OK "manifest written, synced to S3" \
    || phase CACHE_BACKUP_WARN "labstore save failed; cache still on EFS"
else
  aws s3 sync "$CACHE_DIR/" "s3://$S3_BUCKET/trn2-qwen3/compile-cache/" --only-show-errors \
    && phase CACHE_BACKUP_OK || phase CACHE_BACKUP_WARN "s3 sync failed, cache still on EFS"
fi

# --- 8. serve (warm start) ----------------------------------------------
phase SERVE
docker rm -f qwen3-trn2 >/dev/null 2>&1 || true
docker run -d --name qwen3-trn2 --privileged \
  -v /dev:/dev \
  -v "$MODEL_DIR:/model:ro" \
  -v "$CACHE_DIR:/cache:rw" \
  -v "$WORK/results:/results:rw" \
  -e MODEL_PATH=/model \
  -e VLLM_CACHE_ROOT=/cache \
  -e VLLM_NEURON_BACKEND=neuron_native \
  -e NEURON_LOGICAL_NC_CONFIG=2 \
  --shm-size 32g -p 8000:8000 \
  "$IMAGE_TAG" bash /opt/program/serve/run_qwen3_32b.sh || fail "docker run serve"

for i in $(seq 1 120); do
  curl -sf http://localhost:8000/health >/dev/null 2>&1 && { phase SERVE_OK "healthy after ${i}0s"; break; }
  docker inspect -f '{{.State.Running}}' qwen3-trn2 2>/dev/null | grep -q true || fail "server container exited"
  sleep 10
done
curl -sf http://localhost:8000/health >/dev/null 2>&1 || fail "server never became healthy"

# --- 9. correctness -----------------------------------------------------
# A fast wrong answer is worthless; prove it before anyone benchmarks it.
phase VALIDATE
ANS=$(curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":20,"temperature":0}' \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "answer: $ANS"
if echo "$ANS" | grep -qi paris; then
  phase VALIDATE_OK "answer mentions Paris"
else
  phase VALIDATE_WARN "unexpected answer: $ANS"
fi

phase READY "server up on :8000, cache warm, checkpoint on EFS"
echo "=============================================================="
echo "bootstrap complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  model      : $MODEL_DIR"
echo "  cache      : $CACHE_DIR (also s3://$S3_BUCKET/trn2-qwen3/compile-cache/)"
echo "  results dir: $WORK/results"
echo "  next       : bash $EX/bench/run_vllm_bench.sh"
echo "=============================================================="
