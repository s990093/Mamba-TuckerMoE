#!/usr/bin/env bash
# 極速版本推論腳本 (High-Speed Stream Script for batch=1)
# 自動啟用所有最高效能的 Metal 優化，並支援從檔案讀取 Prompt
if [ -z "${BASH_VERSION-}" ] || [ -n "${POSIXLY_CORRECT-}" ]; then
  exec /usr/bin/env bash --noprofile --norc "$0" "$@"
fi

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-checkpoints/checkpoint_sft_s27510_model_only.pt}"
PROMPT_FILE=""
PROMPT_TEXT=""

# 解析參數
extras=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file)
      PROMPT_FILE="$2"
      shift 2
      ;;
    -p|--prompt)
      PROMPT_TEXT="$2"
      shift 2
      ;;
    *)
      extras+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$PROMPT_FILE" ]]; then
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Error: Prompt file '$PROMPT_FILE' not found!" >&2
    exit 1
  fi
  # 讀取檔案內容作為 prompt
  PROMPT_TEXT="$(<"$PROMPT_FILE")"
fi

if [[ -z "$PROMPT_TEXT" && ${#extras[@]} -eq 0 ]]; then
  echo "Usage: ./inference/run_fast_stream.sh -f <prompt_file.txt> [other args]"
  echo "   or: ./inference/run_fast_stream.sh -p \"Your prompt\" [other args]"
  exit 1
fi

print_launch_banner() {
  local w="${1:-76}" i
  i="$(printf '─%.0s' $(seq 1 "$w"))"
  printf '\n╭%s╮\n' "$i"
  printf '│ %-*s │\n' "$((w - 2))" "Mamba3-XR · run_fast_stream (ULTIMATE SPEED)"
  printf '├%s┤\n' "$i"
  printf '│ %-*s │\n' "$((w - 2))" "Checkpoint  : ${CHECKPOINT}"
  printf '│ %-*s │\n' "$((w - 2))" "Prompt File : ${PROMPT_FILE:-<None>}"
  printf '│ %-*s │\n' "$((w - 2))" "Core flags  : throughput │ bf16 │ kv auto │ full-decode-compile"
  printf '│ %-*s │\n' "$((w - 2))" "              --tucker-scalar-fuse │ --fused-mamba-mixer"
  printf '│ %-*s │\n' "$((w - 2))" "Sampling    : --fast-sample --no-penalties (greedy, max tok/s)"
  printf '╰%s╯\n\n' "$i"
}

print_launch_banner 76

PY=(.venv/bin/python inference/stream_mlx.py)
COMMON=(
  --checkpoint "$CHECKPOINT"
  --inference-type throughput
  --dtype bf16
  --kv-dtype auto
  --quantize 8
  --tucker-scalar-fuse
  --fused-mamba-mixer
  --fast-sample
  --full-decode-compile
  --no-materialize-caches
)

if [[ -n "$PROMPT_TEXT" ]]; then
  COMMON+=(--prompt "$PROMPT_TEXT")
fi

if [[ ${#extras[@]} -gt 0 ]]; then
  exec "${PY[@]}" "${COMMON[@]}" "${extras[@]}"
else
  exec "${PY[@]}" "${COMMON[@]}"
fi
