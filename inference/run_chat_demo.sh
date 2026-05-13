#!/usr/bin/env bash
# Mamba3-XR Chat Demo launcher
# Usage:
#   ./inference/run_chat_demo.sh                          # stable preset (safe, 8-bit, stochastic)
#   STREAM_PRESET=best ./inference/run_chat_demo.sh       # speed preset (4-bit, greedy)
#   ./inference/run_chat_demo.sh --port 8080              # custom port
#   CHECKPOINT=path/to.pt ./inference/run_chat_demo.sh    # custom checkpoint
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-/Users/hungwei/Desktop/Proj/Mamba3-XR/checkpoints/latest_sft_cot_model.npz}"
STREAM_PRESET="${STREAM_PRESET:-bset}"   # NOTE: looks like a typo of "best"; left unchanged to preserve current default (falls into the "stable" else branch).
# STREAM_PRESET="${STREAM_PRESET:-stable}"
PORT="${PORT:-7860}"
# Max tokens that the slider can request.  KV cache memory scales linearly
# with this — keep it modest (4096–8192) on 8 GB Macs, bump to 20480 only
# when you really need long generations.
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20480}"
# Hard cap on how many distinct system prompts we keep KV-primed at once.
# Each entry holds a padded KV cache, so the LRU bound directly caps memory.
PRIME_MAX_ENTRIES="${PRIME_MAX_ENTRIES:-2}"
# Max tokens the model may emit *inside* `<think>` before we force-stop the
# turn.  Set to 0 to disable the safety net (not recommended).
REASONING_BUDGET="${REASONING_BUDGET:-512}"
# Inference middleware knobs (see inference/cot_middleware.py).  The
# middleware bundles: <|im_start|> logit ban, ramped </think> close-bias,
# reasoning-budget watchdog, and the multi-stage <final>\n injection.
#   FORMAT_GUARD       — master switch  (on|off, default: on)
#   CLOSE_BIAS         — *initial* bonus added to </think>'s first id
#                        once close-bias engages.
#   CLOSE_BIAS_MAX     — *peak* bonus reached at the reasoning budget
#                        (linear ramp from CLOSE_BIAS to CLOSE_BIAS_MAX).
#   CLOSE_BIAS_START   — tokens-inside-<think> at which the ramp begins.
#                        0 = auto: REASONING_BUDGET // 2.
#   FORCE_FINAL_INJECT — when the model emits </think>, encode "<final>\n"
#                        and run it through the model so the cache commits
#                        to the structural transition (on|off, default: on).
FORMAT_GUARD="${FORMAT_GUARD:-on}"
CLOSE_BIAS="${CLOSE_BIAS:-4.0}"
CLOSE_BIAS_MAX="${CLOSE_BIAS_MAX:-16.0}"
CLOSE_BIAS_START="${CLOSE_BIAS_START:-0}"
FORCE_FINAL_INJECT="${FORCE_FINAL_INJECT:-on}"

GUARD_FLAGS=(
  --close-bias "$CLOSE_BIAS"
  --close-bias-max "$CLOSE_BIAS_MAX"
  --close-bias-start "$CLOSE_BIAS_START"
)
if [[ "$FORMAT_GUARD" == "off" || "$FORMAT_GUARD" == "0" || "$FORMAT_GUARD" == "false" ]]; then
  GUARD_FLAGS=(--no-format-guard "${GUARD_FLAGS[@]}")
else
  GUARD_FLAGS=(--format-guard "${GUARD_FLAGS[@]}")
fi
if [[ "$FORCE_FINAL_INJECT" == "off" || "$FORCE_FINAL_INJECT" == "0" || "$FORCE_FINAL_INJECT" == "false" ]]; then
  GUARD_FLAGS=("${GUARD_FLAGS[@]}" --no-force-final-inject)
else
  GUARD_FLAGS=("${GUARD_FLAGS[@]}" --force-final-inject)
fi
MOCK_FLAG=()
if [[ "${MOCK:-}" == "1" || "${MOCK:-}" == "true" ]]; then
  MOCK_FLAG=(--mock)
fi
# Safe expansion for `set -u`: empty arrays are otherwise treated as unbound (esp. macOS bash 3.2).
MOCK_ARGS=( ${MOCK_FLAG[@]+"${MOCK_FLAG[@]}"} )

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
    ${MOCK_ARGS[@]+"${MOCK_ARGS[@]}"} \
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
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --prime-max-entries "$PRIME_MAX_ENTRIES" \
    --reasoning-budget "$REASONING_BUDGET" \
    "${GUARD_FLAGS[@]}" \
    --port "$PORT" \
    "$@"
else
  exec "${PY[@]}" \
    ${MOCK_ARGS[@]+"${MOCK_ARGS[@]}"} \
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
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --prime-max-entries "$PRIME_MAX_ENTRIES" \
    --reasoning-budget "$REASONING_BUDGET" \
    "${GUARD_FLAGS[@]}" \
    --port "$PORT" \
    "$@"
fi
