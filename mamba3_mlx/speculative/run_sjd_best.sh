#!/usr/bin/env bash
# Launcher for the empirically best SJD (Speculative Jacobi Decoding)
# configuration found on this checkpoint + Makefile-style prompt.
#
# Best config (M2 Pro 16GB, bf16, max_tokens=2048, "Who are you?" self_awareness):
#   K=16, temp=0.15, top_p=0.85, top_k=20, min_p=0.08
#   → 3.32× speedup vs AR-sampling, ARL=5.92, full-accept=25%
#
# K=24 gives ARL=6.70 but slightly lower wall-clock (3.25×). K=16 wins
# the speed race; bump to K=24 if you want higher accepted-run-length
# regardless of throughput (e.g., for diagnostics).
#
# Quality verification done with _quality_check.py — SJD output is
# coherent and matches AR-sampling output distribution
# (spec-sampling rejection-resample preserves the target distribution).
#
# Usage:
#   bash run_sjd_best.sh                          # quad-prompt run
#   bash run_sjd_best.sh --prompt "..."           # single override
#   K=24 bash run_sjd_best.sh                     # override K
#
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python3}"

# ── Defaults (override via env var) ────────────────────────────────────────
K="${K:-16}"
MAX_TOK="${MAX_TOK:-512}"
TEMP="${TEMP:-0.15}"
TOP_K="${TOP_K:-20}"
TOP_P="${TOP_P:-0.85}"
MIN_P="${MIN_P:-0.08}"
SEED="${SEED:-0}"
MODE="${MODE:-self_awareness}"
NGRAM_N="${NGRAM_N:-4}"
DTYPE="${DTYPE:-bf16}"

# Single-prompt override (positional / --prompt …).
if [ "$#" -gt 0 ]; then
    PROMPTS=("$@")
else
    # Quad-launch: 4 representative prompts for a quick warm/full check.
    PROMPTS=(
        "Who are you?"
    )
fi

for prompt in "${PROMPTS[@]}"; do
    echo
    echo "================================================================"
    echo "[sjd-best] prompt: ${prompt}"
    echo "[sjd-best] K=${K}  T=${TEMP}  top_p=${TOP_P}  top_k=${TOP_K}  min_p=${MIN_P}"
    echo "[sjd-best] max_tokens=${MAX_TOK}  seed=${SEED}  ngram_n=${NGRAM_N}"
    echo "================================================================"
    "$PYTHON" -m mamba3_mlx.speculative.run_jacobi_sampling \
        --prompt "${prompt}" \
        --mode "${MODE}" \
        --K "${K}" \
        --max_tokens "${MAX_TOK}" \
        --temp "${TEMP}" \
        --top_k "${TOP_K}" \
        --top_p "${TOP_P}" \
        --min_p "${MIN_P}" \
        --seed "${SEED}" \
        --ngram_n "${NGRAM_N}" \
        --dtype "${DTYPE}" \
        --ar-baseline
done
