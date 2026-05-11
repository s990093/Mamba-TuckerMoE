#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8000}"

cd "$ROOT_DIR"
OLD_PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
if [[ -n "$OLD_PIDS" ]]; then
  echo "[hot] stopping existing process on :$PORT -> $OLD_PIDS"
  kill $OLD_PIDS || true
  sleep 0.3
fi

echo "[hot] starting dev server on :$PORT"
python3 ../src/dev_server.py "$PORT"
