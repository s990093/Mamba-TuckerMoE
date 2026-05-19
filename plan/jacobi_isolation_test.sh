#!/usr/bin/env bash
# =============================================================================
# Jacobi 輸出品質隔離測試腳本  (v2 — English prompt + system prompt)
# 目標：找出 ngram / dynamic-K / intra 哪個導致重複輸出
# 輸出：每步結果存入 plan/isolation_results/
# =============================================================================
set -uo pipefail

PYTHON=".venv/bin/python3"
OUTDIR="plan/isolation_results"
mkdir -p "$OUTDIR"

PROMPT="who are you"
SYS_PROMPT="You are Mamba in Self-Awareness mode. Answer identity and capability questions with strict architectural consistency: Hybrid Mamba-TuckerMoE, edge-deployed on iPhone, offline by default, no subjective consciousness, and no fabricated capabilities."

MAX_TOK=200
AR_WARMUP=5
K=32
JAC="inference/jacobi_stream.py"

# Common flags for ALL Jacobi runs
COMMON_ARGS=(
  --prompt "$PROMPT"
  --sys-prompt "$SYS_PROMPT"
  --max-new-tokens "$MAX_TOK"
  --temp 0.0
  --quantize 8
  --no-progress
  --stop-on-eos
  --ar-warmup "$AR_WARMUP"
  --K "$K"
  --seq-len 512
)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUTDIR/run.log"; }

run_step() {
    local step="$1"; local label="$2"; shift 2
    local outfile="$OUTDIR/step${step}.txt"
    log "▶ Step $step: $label"
    "$PYTHON" "$JAC" "${COMMON_ARGS[@]}" "$@" > "$outfile" 2>&1
    local rc=$?
    echo "EXIT_CODE=$rc" >> "$outfile"
    if [ $rc -ne 0 ]; then
        log "  ✗ FAILED (exit $rc)"
    else
        log "  ✓ OK → $outfile"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
log "===== Jacobi 隔離測試開始 ====="
log "Prompt: \"$PROMPT\" | sys-prompt: (self_awareness) | max-tok: $MAX_TOK | K: $K | ar-warmup: $AR_WARMUP"

# ── Step 0: AR baseline ──────────────────────────────────────────────────────
# Use jacobi_stream.py with ar-warmup > max-tokens so Jacobi never fires.
# Same model/quant/sys-prompt as all other steps.
log "▶ Step 0: AR baseline (ar-warmup=201, no Jacobi)"
"$PYTHON" "$JAC" "${COMMON_ARGS[@]}" --ar-warmup 201 --no-ngram --no-dynamic-K --no-intra \
  > "$OUTDIR/step0_ar.txt" 2>&1
echo "EXIT_CODE=$?" >> "$OUTDIR/step0_ar.txt"
log "  ✓ AR baseline → $OUTDIR/step0_ar.txt"

# ── Step 0b: AR token ids via compare_ar_jacobi.py ──────────────────────────
log "▶ Step 0b: compare_ar_jacobi (token-level AR+Jacobi match check)"
"$PYTHON" inference/compare_ar_jacobi.py \
  --prompt "$PROMPT" \
  --max-tokens "$MAX_TOK" \
  --K "$K" \
  --warmup "$AR_WARMUP" \
  > "$OUTDIR/step0b_compare.txt" 2>&1
log "  ✓ saved → $OUTDIR/step0b_compare.txt"

# ── Step 1-6: Isolation runs ─────────────────────────────────────────────────

# Step 1: 乾淨基線 — 關閉所有加速
run_step 1 "純 Jacobi (no-ngram, no-dynamic-K, no-intra)" \
  --no-ngram --no-dynamic-K --no-intra

# Step 2: ngram only
run_step 2 "ngram only (no-dynamic-K, no-intra)" \
  --no-dynamic-K --no-intra

# Step 3: dynamic-K only
run_step 3 "dynamic-K only (no-ngram, no-intra)" \
  --no-ngram --no-intra

# Step 4: intra only
run_step 4 "intra only (no-ngram, no-dynamic-K)" \
  --no-ngram --no-dynamic-K

# Step 5a: ngram + dynamic-K
run_step 5a "ngram + dynamic-K (no-intra)" \
  --no-intra

# Step 5b: ngram + intra
run_step 5b "ngram + intra (no-dynamic-K)" \
  --no-dynamic-K

# Step 5c: dynamic-K + intra
run_step 5c "dynamic-K + intra (no-ngram)" \
  --no-ngram

# Step 6: 全功能 (預設)
run_step 6 "全功能 default (ngram + dynamic-K + intra)"

# ──────────────────────────────────────────────────────────────────────────────
log "===== 所有步驟完成，開始分析 ====="
"$PYTHON" plan/jacobi_analyze.py "$OUTDIR" 2>&1 | tee "$OUTDIR/analysis.txt"
log "===== 分析完成 → $OUTDIR/analysis.txt ====="
