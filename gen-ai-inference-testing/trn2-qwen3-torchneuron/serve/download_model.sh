#!/usr/bin/env bash
# Download a Qwen3 checkpoint to local disk (or EFS) before serving.
#
# Pin BF16_REVISION for reproducibility -- benchmark numbers are not
# comparable across checkpoint revisions. Open item O-3 in PLAN.md: confirm
# whether Qwen/Qwen3-32B is gated and an HF_TOKEN is required.
#
# UNVERIFIED: not yet run on hardware.
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
BF16_REVISION="${BF16_REVISION:-main}"
DEST="${DEST:-/snapshots/$(basename "$MODEL_ID")}"

mkdir -p "$DEST"
echo "=== downloading $MODEL_ID @ $BF16_REVISION -> $DEST ==="
[[ -n "${HF_TOKEN:-}" ]] && echo "HF_TOKEN is set" || echo "HF_TOKEN not set (fine if the repo is public)"

hf download "$MODEL_ID" --revision "$BF16_REVISION" --local-dir "$DEST"

echo
echo "=== checkpoint manifest ==="
# Hash the files so the exact checkpoint can be verified across environments
# (the AWS tutorial recommends this) and attached to benchmark results.
find "$DEST" -type f \( -name '*.safetensors' -o -name '*.json' \) -print0 \
  | sort -z | xargs -0 shasum -a 256 | tee "$DEST/../$(basename "$DEST").sha256"

du -sh "$DEST"
echo "done. Set MODEL_PATH=$DEST when serving."
