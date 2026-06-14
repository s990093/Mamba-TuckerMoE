#!/usr/bin/env bash
# scripts/draft/run_overfit.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launch draft model self-awareness overfit on GPU 4 (torch310 conda env).
# Does NOT touch GPUs 1,2 (ongoing main model training).
#
# Usage:
#   bash scripts/draft/run_overfit.sh
#   DRAFT_STEPS=2000 bash scripts/draft/run_overfit.sh   # longer run
# ─────────────────────────────────────────────────────────────────────────────
set -eo pipefail
export TZ=Asia/Taipei

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── conda env ────────────────────────────────────────────────────────────────
if [[ "${CONDA_DEFAULT_ENV:-}" != "torch310" ]]; then
    echo "Activating conda env torch310..."
    eval "$(conda shell.bash hook)"
    conda activate torch310
fi

# ── GPU 4 only — never touch GPUs 1,2 (main model training) ─────────────────
export CUDA_VISIBLE_DEVICES=4
# Within the Python process, GPU 4 appears as cuda:0

export PYTHONPATH="${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export DRAFT_STEPS="${DRAFT_STEPS:-1000}"

echo "═══════════════════════════════════════════════════════════"
echo " Draft Model Overfit — Self-Awareness"
echo " GPU slot 4  |  Steps=${DRAFT_STEPS}  |  arch: d=256, L=3, Dense"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi --query-gpu=index,name,memory.used,memory.free \
           --format=csv,noheader -i 4 2>/dev/null || true
echo ""

python3 scripts/draft/overfit_self_awareness.py "$@"
