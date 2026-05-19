#!/usr/bin/env bash
# chat_creative.sh — higher temperature, more diverse output
# Usage: bash mamba3_mlx/chat_creative.sh "Write a haiku about the moon."
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash mamba3_mlx/chat_creative.sh \"<prompt>\" [extra flags]"
  exit 1
fi

PROMPT="$1"; shift

cd "${REPO_ROOT}"
exec "${PYTHON}" "${REPO_ROOT}/mamba3_mlx/run.py" \
  --prompt "${PROMPT}" \
  --checkpoint "${REPO_ROOT}/checkpoints/latest_sft_cot_model.npz" \
  --tokenizer  "${REPO_ROOT}/checkpoints/tokenizer" \
  --max_tokens 400 \
  --temp       1.1 \
  --top_k      80 \
  --top_p      0.95 \
  --min_p      0.02 \
  --rep_pen    1.15 \
  --freq_pen   0.05 \
  --dtype      bf16 \
  --benchmark \
  "$@"
