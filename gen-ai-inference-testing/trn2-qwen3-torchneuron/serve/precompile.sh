#!/usr/bin/env bash
# Warm the TorchNeuron precompile cache, then exit.
#
# This is the highest-leverage step when running inside a time-boxed Capacity
# Block. First-run compilation traces on CPU and compiles to NEFF; the docs
# default the timeout to 1200s, which indicates the order of magnitude. Run
# this once per config you intend to benchmark, BEFORE collecting any numbers.
#
# The cache fingerprint invalidates on any version, compiler-flag or config
# change, so each distinct TP/bucket combination is a separate cold compile.
# Freeze your sweep matrix before starting.
#
# UNVERIFIED: not yet run on hardware.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/snapshots/Qwen3-32B}"
PORT="${PORT:-8000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"   # generous: cold compile
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache}"
export VLLM_NEURON_BACKEND="${VLLM_NEURON_BACKEND:-neuron_native}"
mkdir -p "$VLLM_CACHE_ROOT"

echo "=== precompile: warming $VLLM_CACHE_ROOT ==="
echo "Cache layout: \$VLLM_CACHE_ROOT/neuron/torch_compile/<fingerprint>/dp_<i>/rank_<r>"
BEFORE=$(find "$VLLM_CACHE_ROOT" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "files in cache before: $BEFORE"

LOG=$(mktemp /tmp/precompile-XXXXXX.log)
echo "server log -> $LOG"

PORT="$PORT" MODEL_PATH="$MODEL_PATH" "$SCRIPT_DIR/run_qwen3_32b.sh" >"$LOG" 2>&1 &
SERVER_PID=$!
# shellcheck disable=SC2064
trap "kill $SERVER_PID 2>/dev/null || true" EXIT

echo "waiting up to ${HEALTH_TIMEOUT}s for first compile + startup ..."
START=$(date +%s)
READY=0
while (( $(date +%s) - START < HEALTH_TIMEOUT )); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server exited during compile. Tail of log:" >&2
    tail -40 "$LOG" >&2
    exit 1
  fi
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 10
done

ELAPSED=$(( $(date +%s) - START ))
if (( READY == 0 )); then
  echo "ERROR: not ready after ${ELAPSED}s. Tail of log:" >&2
  tail -40 "$LOG" >&2
  exit 1
fi
echo "ready after ${ELAPSED}s"

AFTER=$(find "$VLLM_CACHE_ROOT" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "files in cache after: $AFTER (added $((AFTER - BEFORE)))"

# Cold start logs cache misses and saves > 0; a warm start shows Dynamo hits
# and zero saves. Surface whichever happened.
echo "=== cache signal from startup log ==="
grep -iE "cache (hit|miss|save)|dynamo|precompile" "$LOG" | tail -20 || \
  echo "(no cache lines matched -- inspect $LOG directly)"

echo
echo "Precompile complete. Restart the server for a warm start and confirm"
echo "startup logs show Dynamo hits with zero saves before benchmarking."
echo "Persist the cache off-instance now -- the Capacity Block reclaims this host:"
echo "  aws s3 sync $VLLM_CACHE_ROOT s3://<bucket>/neuron-compile-cache/qwen3-32b-tp8/"
