#!/usr/bin/env bash
# run_tests.sh — run the full MLX inference sanity-check suite
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"

cd "${REPO_ROOT}"
exec "${PYTHON}" mamba3_mlx/tests/test_mlx.py
