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

# --- 3. restore a previous compile cache if one exists -------------------
# This is what makes a second block cheap: a warm cache skips ~20 min of
# cold compile per config. Fingerprints invalidate on version/config change,
# so a restore may still miss -- that is fine, it just recompiles.
phase CACHE_RESTORE
if aws s3 ls "s3://$S3_BUCKET/trn2-qwen3/compile-cache/" >/dev/null 2>&1; then
  aws s3 sync "s3://$S3_BUCKET/trn2-qwen3/compile-cache/" "$CACHE_DIR/" --only-show-errors \
    && phase CACHE_RESTORE_OK "files=$(find "$CACHE_DIR" -type f | wc -l | tr -d ' ')"
else
  phase CACHE_RESTORE_SKIP "no prior cache in s3"
fi

# --- 4. container image -------------------------------------------------
phase IMAGE "pulling $DLC"
docker pull "$DLC" || fail "docker pull"
( cd "$EX" && docker buildx build -t "$IMAGE_TAG" -f containers/Dockerfile . ) || fail "docker build"
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
# artifact is already safe.
phase CACHE_BACKUP
aws s3 sync "$CACHE_DIR/" "s3://$S3_BUCKET/trn2-qwen3/compile-cache/" --only-show-errors \
  && phase CACHE_BACKUP_OK || phase CACHE_BACKUP_WARN "s3 sync failed, cache still on EFS"

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
