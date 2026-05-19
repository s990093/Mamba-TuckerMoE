#!/usr/bin/env bash
# chat_precise.sh — low temperature, near-greedy, best for factual queries
# Usage: bash mamba3_mlx/chat_precise.sh "What is the boiling point of water?"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash mamba3_mlx/chat_precise.sh \"<prompt>\" [extra flags]"
  exit 1
fi

PROMPT="$1"; shift

cd "${REPO_ROOT}"
exec "${PYTHON}" "${REPO_ROOT}/mamba3_mlx/run.py" \
  --prompt "${PROMPT}" \
  --checkpoint "${REPO_ROOT}/checkpoints/latest_sft_cot_model.npz" \
  --tokenizer  "${REPO_ROOT}/checkpoints/tokenizer" \
  --max_tokens 256 \
  --temp       0.3 \
  --top_k      20 \
  --top_p      0.85 \
  --min_p      0.1 \
  --rep_pen    1.2 \
  --freq_pen   0.1 \
  --dtype      bf16 \
  --benchmark \
  "$@"
