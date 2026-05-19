#!/usr/bin/env bash
# chat.sh — single-turn chat with default sampling settings
# Usage: bash mamba3_mlx/chat.sh "Your question here" [extra run.py flags]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash mamba3_mlx/chat.sh \"<prompt>\" [--max_tokens N] [--temp F] ..."
  exit 1
fi

PROMPT="$1"; shift

cd "${REPO_ROOT}"
exec "${PYTHON}" "${REPO_ROOT}/mamba3_mlx/run.py" \
  --prompt "${PROMPT}" \
  --checkpoint "${REPO_ROOT}/checkpoints/latest_sft_cot_model.npz" \
  --tokenizer  "${REPO_ROOT}/checkpoints/tokenizer" \
  --max_tokens 512 \
  --min_tokens 16 \
  --temp       0.8 \
  --top_k      40 \
  --top_p      0.9 \
  --min_p      0.05 \
  --rep_pen    1.1 \
  --freq_pen   0.02 \
  --dtype      bf16 \
  --prime-think \
  --benchmark \
  "$@"
