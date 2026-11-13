#!/usr/bin/env bash
# SLO sweep using `vllm bench serve` -- the source-of-truth harness.
#
# Emits one JSON per (token-shape, concurrency) cell into $RESULTS_DIR, which
# bench/parse_results.py folds into an SLO table.
#
# --ignore-eos is NOT optional. Without it, output length becomes a function
# of the model's stopping behaviour, so ITL and throughput stop being
# comparable across cells.
#
# UNVERIFIED: not yet run on hardware.
#
# Usage:
#   ./run_vllm_bench.sh
#   CONCURRENCIES="1 4 16" SHAPES="4096:256" ./run_vllm_bench.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-qwen3-32b}"
RESULTS_DIR="${RESULTS_DIR:-./results}"

# "input:output" token shapes -- spanning prefill-bound to decode-bound.
#
# IMPORTANT: Neuron's torch.compile backend forces dynamic=False, so each
# distinct shape needs a compiled bucket. These shapes and concurrencies MUST
# be covered by the server's neuron_config buckets, or the sweep triggers
# recompiles mid-run and the latency numbers become meaningless.
#
# Start the server with SWEEP_MODE=1 (see serve/run_qwen3_32b.sh) so one
# compile covers the whole matrix below. If the server is running the
# published control config (num_seqs_buckets [4]), restrict this sweep to
# CONCURRENCIES="1 2 4".
SHAPES="${SHAPES:-1024:128 4096:256 8192:1024}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16 32}"
# Enough prompts for percentiles to mean something: >= 10x concurrency.
PROMPT_MULTIPLIER="${PROMPT_MULTIPLIER:-10}"
MIN_PROMPTS="${MIN_PROMPTS:-20}"
REQUEST_RATE="${REQUEST_RATE:-inf}"

mkdir -p "$RESULTS_DIR"

if ! curl -sf "${BASE_URL}/health" >/dev/null; then
  echo "ERROR: no healthy server at ${BASE_URL}. Start it with serve/run_qwen3_32b.sh." >&2
  exit 1
fi

# Capture the environment alongside results -- both TorchNeuron and the vLLM
# plugin are Beta, so resolved versions are part of the measurement.
{
  echo "timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "base_url: $BASE_URL"
  echo "model: $MODEL"
  echo "NEURON_LOGICAL_NC_CONFIG: ${NEURON_LOGICAL_NC_CONFIG:-unset}"
  echo "VLLM_NEURON_BACKEND: ${VLLM_NEURON_BACKEND:-unset}"
  command -v neuron-ls >/dev/null && neuron-ls 2>&1 | head -30
  python -m pip freeze 2>/dev/null | grep -iE "vllm|neuron|^torch" || true
} > "$RESULTS_DIR/run-metadata.txt"
echo "metadata -> $RESULTS_DIR/run-metadata.txt"

for shape in $SHAPES; do
  IN="${shape%%:*}"
  OUT="${shape##*:}"
  for c in $CONCURRENCIES; do
    n=$(( c * PROMPT_MULTIPLIER ))
    (( n < MIN_PROMPTS )) && n=$MIN_PROMPTS
    name="vllmbench_in${IN}_out${OUT}_c${c}"
    out_file="$RESULTS_DIR/${name}.json"

    if [[ -f "$out_file" ]]; then
      echo "--- skip ${name} (exists) ---"
      continue
    fi

    echo "=== ${name}: in=${IN} out=${OUT} concurrency=${c} prompts=${n} ==="
    # `random` dataset with fixed lengths gives controlled token shapes
    # rather than observed ones.
    if ! vllm bench serve \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$IN" \
        --random-output-len "$OUT" \
        --num-prompts "$n" \
        --max-concurrency "$c" \
        --request-rate "$REQUEST_RATE" \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,99 \
        --save-result \
        --result-filename "$out_file"; then
      echo "WARN: ${name} failed; continuing sweep" >&2
    fi
    sleep 5
  done
done

echo
echo "=== sweep complete: $RESULTS_DIR ==="
echo "Next: python bench/parse_results.py --results-dir $RESULTS_DIR"
echo "Copy results off-instance before the Capacity Block expires."
