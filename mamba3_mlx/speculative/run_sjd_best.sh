#!/usr/bin/env bash
# Best-config Jacobi speculative decoding launcher.
#
# Uses greedy Jacobi (byte-equal to AR-greedy) with:
#   - adaptive_K     K=12 → auto-tunes to 4-24 via GammaTune EWMA
#   - cot_caches     v2 pkl (per-bucket ngram + retriever, pre-warmed)
#   - cot_bucket     derived from --mode
#   - runtime_cache  demo_cache_v2.pkl (AR-sampling pre-warmed NGramCache +
#                    SuffixRetriever from bake_cache.py)
#   - compile_verify mx.compile per-Mamba-layer verify step (fp32 byte-equal,
#                    fuses ~50 dispatches/layer/round into one batch)
#
# Bench reference (M2 Pro, bf16, greedy "Who are you?", max=256):
#   baseline (no caches, no compile)     29 tps   ARL 1.84
#   + compile_verify                     34 tps   ARL 1.84   (+17%)
#   + runtime_cache                      39 tps   ARL 2.44   (+34%)
#   + compile + runtime                  41 tps   ARL 2.44   (+41%)
#
# Usage:
#   bash run_sjd_best.sh                         # default prompt
#   bash run_sjd_best.sh --prompt "Explain SSMs"
#   bash run_sjd_best.sh --stream                # stream tokens
#   bash run_sjd_best.sh --mode math_drill --prompt "Solve x^2=4"
#   K=8 bash run_sjd_best.sh                     # override K
#   COMPILE=0 bash run_sjd_best.sh               # disable mx.compile
#   RUNTIME=none bash run_sjd_best.sh            # disable runtime cache
#
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python3}"

# ── Defaults (override via env var) ────────────────────────────────────────
K="${K:-12}"
K_MIN="${K_MIN:-4}"
K_MAX="${K_MAX:-24}"
MAX_TOK="${MAX_TOK:-512}"
MODE="${MODE:-self_awareness}"
NGRAM_N="${NGRAM_N:-4}"
DTYPE="${DTYPE:-bf16}"
COT_CACHES="${COT_CACHES:-$SCRIPT_DIR/cot_caches_v2.pkl}"
RUNTIME="${RUNTIME:-$SCRIPT_DIR/demo_cache_v2.pkl}"   # 'none' to disable
COMPILE="${COMPILE:-1}"                                # 1 = compile_verify on
STREAM="${STREAM:-0}"

# ── Collect extra flags from CLI ───────────────────────────────────────────
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stream)     STREAM=1; shift ;;
        --no-compile) COMPILE=0; shift ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

STREAM_FLAG=""
[[ "$STREAM" == "1" ]] && STREAM_FLAG="--stream"
COMPILE_FLAG=""
[[ "$COMPILE" == "1" ]] && COMPILE_FLAG="--compile_verify"

echo
echo "================================================================"
echo "[sjd-best] mode=${MODE}  K=${K} (adaptive ${K_MIN}–${K_MAX})"
echo "[sjd-best] cot_caches=$(basename "${COT_CACHES}")  runtime=$(basename "${RUNTIME}")"
echo "[sjd-best] compile_verify=${COMPILE}  max_tokens=${MAX_TOK}  stream=${STREAM}"
echo "================================================================"

cd "$REPO"
"$PYTHON" -m mamba3_mlx.speculative.run_jacobi \
    --mode          "${MODE}" \
    --K             "${K}" \
    --K_min         "${K_MIN}" \
    --K_max         "${K_MAX}" \
    --max_tokens    "${MAX_TOK}" \
    --ngram_n       "${NGRAM_N}" \
    --dtype         "${DTYPE}" \
    --use_ngram \
    --use_retrieval \
    --adaptive_K \
    --cot_caches    "${COT_CACHES}" \
    --runtime_cache "${RUNTIME}" \
    ${COMPILE_FLAG} \
    ${STREAM_FLAG} \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
