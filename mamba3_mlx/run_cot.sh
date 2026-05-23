#!/usr/bin/env bash
# run_cot.sh — exercise the CoT middleware (mv/cot_*.py) on Mamba3 MLX.
# Read-only with respect to the original inference path — invokes
# mamba3_mlx/run_cot.py without touching run.py / generator.py.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python3}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3)"
fi

PROMPT="${PROMPT:-Explain why the sky appears blue in one short paragraph.}"
MODE="${MODE:-self}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMP="${TEMP:-0.8}"
FINAL_MIN_TOKENS="${FINAL_MIN_TOKENS:-16}"
# No --reasoning-budget — the model decides when to close <think>.

echo "=== run_cot.sh ==="
echo "PYTHON           = $PYTHON"
echo "PROMPT           = $PROMPT"
echo "MODE             = $MODE"
echo "MAX_TOKENS       = $MAX_TOKENS"
echo "FINAL_MIN_TOKENS = $FINAL_MIN_TOKENS"
echo "------------------"

cd "$REPO_ROOT"
exec "$PYTHON" -m mamba3_mlx.run_cot \
  --prompt "$PROMPT" \
  --mode "$MODE" \
  --max_tokens "$MAX_TOKENS" \
  --temp "$TEMP" \
  --final-min-tokens "$FINAL_MIN_TOKENS" \
  "$@"
