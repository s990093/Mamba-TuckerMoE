#!/usr/bin/env bash
# Launch the Mamba chat server using the project .venv
# Usage:
#   ./mamba3_mlx/serve.sh              # real model, default checkpoint
#   ./mamba3_mlx/serve.sh --mock       # mock mode (no GPU)
#   MOCK=1 ./mamba3_mlx/serve.sh       # same via env
set -e
cd "$(dirname "$0")/../.."

VENV=".venv/bin/python"
if [ ! -f "$VENV" ]; then
  echo "ERROR: .venv not found. Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

PORT="${PORT:-8000}"

# Kill any existing process on the target port before binding
if command -v lsof >/dev/null 2>&1; then
  _pid=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
  if [ -n "$_pid" ]; then
    echo "Killing existing process(es) on port $PORT: $_pid"
    kill "$_pid" 2>/dev/null || true
    sleep 0.5
  fi
fi

MOCK_FLAG=""
if [ "${MOCK:-0}" = "1" ]; then MOCK_FLAG="--mock"; fi

exec "$VENV" -m mamba3_mlx.server \
  ${MOCK_FLAG} \
  --checkpoint "${CHECKPOINT:-checkpoints/latest_sft_cot_model.npz}" \
  --tokenizer  "${TOKENIZER:-checkpoints/tokenizer}" \
  --host       "${HOST:-0.0.0.0}" \
  --port       "$PORT" \
  --dtype      "${DTYPE:-bf16}" \
  "$@"
