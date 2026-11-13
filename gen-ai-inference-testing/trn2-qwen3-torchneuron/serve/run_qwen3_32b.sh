#!/usr/bin/env bash
# Serve Qwen/Qwen3-32B on trn2.48xlarge via vLLM on the native TorchNeuron
# backend. Defaults reproduce the published AWS Trn2 recipe verbatim --
# reproduce the baseline before changing any flag, since it is the only
# config with published correctness + performance validation.
#
# UNVERIFIED: not yet run on hardware.
#
# Usage:
#   ./run_qwen3_32b.sh                       # TP8 baseline, port 8000
#   TENSOR_PARALLEL_SIZE=16 ./run_qwen3_32b.sh
#   PORT=8001 ./run_qwen3_32b.sh             # second replica
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/snapshots/Qwen3-32B}"
SERVED_NAME="${SERVED_NAME:-qwen3-32b}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# --- published Trn2 BF16 recipe ------------------------------------------
# MXFP8 is NOT available here: the docs require Trn3 for it. BF16 only.
DTYPE="${DTYPE:-bfloat16}"
OPT_LEVEL="${OPT_LEVEL:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

# --- buckets: why this matters more than it looks -----------------------
# torch.compile on Neuron DOES NOT SUPPORT DYNAMIC SHAPES -- the backend
# forces dynamic=False, so "each distinct input shape requires separate
# compilation". Buckets are the mechanism that pins the shapes which get
# compiled. Any shape outside them means a recompile (or worse, mid-benchmark
# stalls that corrupt your latency numbers).
#
# Two distinct configurations, deliberately NOT mixed:
#
#   SWEEP_MODE=0 (default) -- the published AWS Trn2 recipe, verbatim. This is
#   the CONTROL: the only config with AWS correctness+performance validation.
#   Valid for concurrency <= 4 only.
#
#   SWEEP_MODE=1 -- widened buckets so one compile covers a whole SLO sweep
#   across concurrency and input length. This is a DIFFERENT fingerprint from
#   the control, so it gets its own compile cache entry and its numbers are
#   not directly comparable to the published recipe.
SWEEP_MODE="${SWEEP_MODE:-0}"
if [[ "$SWEEP_MODE" == "1" ]]; then
  # max-num-seqs must be >= the largest num_seqs bucket, so raise it too.
  MAX_NUM_SEQS="${MAX_NUM_SEQS_SWEEP:-32}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN_SWEEP:-10240}"
  KV_SEGMENT_BUCKETS="${KV_SEGMENT_BUCKETS:-[2048,8192]}"
  BATCHED_TOKENS_BUCKETS="${BATCHED_TOKENS_BUCKETS:-[2048,8192]}"
  NUM_SEQS_BUCKETS="${NUM_SEQS_BUCKETS:-[1,2,4,8,16,32]}"
  echo "SWEEP_MODE=1: widened buckets. NOT the published control config."
  echo "  Every bucket combination compiles separately -- expect a longer"
  echo "  first compile, then a sweep with no mid-run recompiles."
else
  KV_SEGMENT_BUCKETS="${KV_SEGMENT_BUCKETS:-[8192]}"
  BATCHED_TOKENS_BUCKETS="${BATCHED_TOKENS_BUCKETS:-[8192]}"
  NUM_SEQS_BUCKETS="${NUM_SEQS_BUCKETS:-[$MAX_NUM_SEQS]}"
fi

# --- environment ---------------------------------------------------------
# These MUST be exported before Python starts. Setting them from inside
# Python has no effect (per the precompile-cache docs).
export NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-2}"
export VLLM_NEURON_BACKEND="${VLLM_NEURON_BACKEND:-neuron_native}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache}"
export VLLM_NEURON_COMPILATION_TIMEOUT="${VLLM_NEURON_COMPILATION_TIMEOUT:-1200}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: MODEL_PATH '$MODEL_PATH' not found. Run serve/download_model.sh first." >&2
  exit 1
fi
mkdir -p "$VLLM_CACHE_ROOT"

echo "=== config ==="
echo "  model            : $MODEL_PATH"
echo "  dtype            : $DTYPE"
echo "  TP               : $TENSOR_PARALLEL_SIZE  (LNC=$NEURON_LOGICAL_NC_CONFIG)"
echo "  max-model-len    : $MAX_MODEL_LEN"
echo "  max-num-seqs     : $MAX_NUM_SEQS"
echo "  backend          : $VLLM_NEURON_BACKEND"
echo "  cache root       : $VLLM_CACHE_ROOT"
echo "  port             : $PORT"
# At LNC2 a trn2.48xlarge exposes 64 logical cores, so TP8 uses 2 of 16
# devices and up to 8 such replicas fit on one instance.
echo "  logical cores used: $((TENSOR_PARALLEL_SIZE)) of 64 (LNC2)"
echo

ADDITIONAL_CONFIG=$(cat <<EOF
{"neuron_config": {
  "kv_segment_size_buckets": ${KV_SEGMENT_BUCKETS},
  "num_batched_tokens_buckets": ${BATCHED_TOKENS_BUCKETS},
  "num_seqs_buckets": ${NUM_SEQS_BUCKETS}
}}
EOF
)

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --optimization-level "$OPT_LEVEL" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --additional-config "$ADDITIONAL_CONFIG"
