#!/usr/bin/env bash
# 生成質量測試腳本 — 顯示 streaming 輸出
#
# Usage:
#   # 跑內建 3 組測試 prompt (emotion / burnout / travel)
#   ./inference/run_quality_check.sh
#
#   # 單一自訂 prompt
#   ./inference/run_quality_check.sh -p "你的問題"
#
#   # 安靜模式：只顯示 token stream，不印 banner
#   QUIET=1 ./inference/run_quality_check.sh
#
#   # 調 sampling 參數
#   TEMP=0.5 TOP_K=50 ./inference/run_quality_check.sh
#
#   # 比較用：速度優先 (greedy, 4-bit)
#   MODE=fast ./inference/run_quality_check.sh -p "..."
if [ -z "${BASH_VERSION-}" ] || [ -n "${POSIXLY_CORRECT-}" ]; then
  exec /usr/bin/env bash --noprofile --norc "$0" "$@"
fi
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── 設定值 ────────────────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-checkpoints/latest_sft_cot_model.npz}"
MODE="${MODE:-quality}"          # quality | fast
TEMP="${TEMP:-0.3}"
TOP_K="${TOP_K:-40}"
TOP_P="${TOP_P:-0.9}"
MIN_P="${MIN_P:-0.05}"
MAX_TOKENS="${MAX_TOKENS:-512}"
QUIET="${QUIET:-0}"

CUSTOM_PROMPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prompt) CUSTOM_PROMPT="$2"; shift 2 ;;
    -q|--quiet)  QUIET=1; shift ;;
    *) shift ;;
  esac
done

# ── 內建測試 prompts ──────────────────────────────────────────────────────────
PROMPTS=(
  "Every morning I wake up and I already feel exhausted. I don't know how much longer I can keep doing this job."
  "I am traveling to a country where I do not speak the language. What is the most efficient way to handle basic communication without relying on internet translation apps?"
  "A train leaves station A at 60 km/h and another leaves station B at 80 km/h toward each other. They are 280 km apart. When do they meet?"
)

# ── Banner ────────────────────────────────────────────────────────────────────
if [[ "$QUIET" == "0" ]]; then
  echo ""
  echo "╔═══════════════════════════════════════════════════════════╗"
  echo "║          Mamba3-XR · 生成質量 Streaming 測試             ║"
  echo "╠═══════════════════════════════════════════════════════════╣"
  printf "║  模式      : %-44s ║\n" "$MODE"
  printf "║  Checkpoint: %-44s ║\n" "$(basename "$CHECKPOINT")"
  printf "║  Temp/TopK : %-44s ║\n" "temp=${TEMP}  top_k=${TOP_K}  top_p=${TOP_P}"
  printf "║  Max tokens: %-44s ║\n" "$MAX_TOKENS"
  echo "╚═══════════════════════════════════════════════════════════╝"
  echo ""
fi

# ── 共用旗標 ──────────────────────────────────────────────────────────────────
PY=(.venv/bin/python inference/stream_mlx.py)

QUALITY_FLAGS=(
  --checkpoint "$CHECKPOINT"
  --inference-type safe
  --dtype bf16
  --kv-dtype bf16
  --quantize 8
  --no-tucker-einsum-fuse
  --materialize-caches
  --temp "$TEMP"
  --top_k "$TOP_K"
  --top_p "$TOP_P"
  --min_p "$MIN_P"
  --enable-penalties
  --rep_pen 1.1
  --stop-on-eos
  --max-new-tokens "$MAX_TOKENS"
  --plain-output
  --warmup 1
)

FAST_FLAGS=(
  --checkpoint "$CHECKPOINT"
  --inference-type throughput
  --dtype bf16
  --kv-dtype auto
  --quantize 4
  --tucker-einsum-fuse
  --no-materialize-caches
  --fast-sample
  --no-penalties
  --full-decode-compile
  --stop-on-eos
  --max-new-tokens "$MAX_TOKENS"
  --plain-output
  --warmup 2
)

if [[ "$MODE" == "fast" ]]; then
  FLAGS=( "${FAST_FLAGS[@]}" )
else
  FLAGS=( "${QUALITY_FLAGS[@]}" )
fi

# ── 執行函式 ──────────────────────────────────────────────────────────────────
run_prompt() {
  local idx="$1"
  local prompt="$2"

  if [[ "$QUIET" == "0" ]]; then
    local label
    case "$idx" in
      1) label="[Prompt 1/3] Emotion · 職業倦怠" ;;
      2) label="[Prompt 2/3] Travel · 語言障礙" ;;
      3) label="[Prompt 3/3] Math · 追及問題" ;;
      *) label="[Custom prompt]" ;;
    esac
    echo "────────────────────────────────────────────────────────────"
    echo "  $label"
    echo "  Q: $(echo "$prompt" | head -c 80)..."
    echo "────────────────────────────────────────────────────────────"
    echo ""
  fi

  "${PY[@]}" "${FLAGS[@]}" --prompt "$prompt"
  echo ""
}

# ── 主流程 ────────────────────────────────────────────────────────────────────
if [[ -n "$CUSTOM_PROMPT" ]]; then
  run_prompt 0 "$CUSTOM_PROMPT"
else
  run_prompt 1 "${PROMPTS[0]}"
  run_prompt 2 "${PROMPTS[1]}"
  run_prompt 3 "${PROMPTS[2]}"
fi
