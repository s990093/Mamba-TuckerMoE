#!/usr/bin/env bash
Mamba3-XR Chat Demo launcher
# Usage:
#   ./inference/run_chat_demo.sh                          # stable preset (safe, 8-bit, stochastic)
#   STREAM_PRESET=best ./inference/run_chat_demo.sh       # speed preset (4-bit, greedy)
#   ./inference/run_chat_demo.sh --port 8080              # custom port
#   CHECKPOINT=path/to.pt ./inference/run_chat_demo.sh    # custom checkpoint
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-checkpoints/checkpoint_sft_s27510_model_only.pt}"
STREAM_PRESET="${STREAM_PRESET:-bset}"
# STREAM_PRESET="${STREAM_PRESET:-stable}"
PORT="${PORT:-7860}"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          Mamba3-XR · Chat Demo Launcher             ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Preset    : ${STREAM_PRESET}"
echo "║  Checkpoint: ${CHECKPOINT}"
echo "║  Port      : ${PORT}"
echo "║  URL       : http://localhost:${PORT}"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

PY=(.venv/bin/python inference/chat_demo.py)

if [[ "$STREAM_PRESET" == "best" ]]; then
  exec "${PY[@]}" \
    --checkpoint "$CHECKPOINT" \
    --inference-type throughput \
    --dtype bf16 \
    --kv-dtype auto \
    --quantize 4 \
    --tucker-einsum-fuse \
    --tucker-scalar-fuse \
    --fused-mamba-mixer \
    --fast-sample \
    --full-decode-compile \
    --no-materialize-caches \
    --router-temp 0.5 \
    --port "$PORT" \
    "$@"
else
  exec "${PY[@]}" \
    --checkpoint "$CHECKPOINT" \
    --inference-type safe \
    --materialize-caches \
    --dtype bf16 \
    --kv-dtype bf16 \
    --quantize 8 \
    --no-tucker-einsum-fuse \
    --no-fast-sample \
    --temp 0.3 \
    --top_p 0.9 \
    --top_k 40 \
    --min_p 0.05 \
    --port "$PORT" \
    "$@"
fi
