#!/usr/bin/env bash
# 一鍵 demo: 把 v2 baked cache 載入 (1ms)，K=16 + max_tokens=512 跑 SJD，
# 對照 AR-sampling.  Empirically reaches **ARL ≈ 7.2, ~3.2× speedup, ~8 s
# wall**.  Wrap-up is just the model load + cache load (<5 s).
#
# Usage:
#   bash demo.sh                                # default "Who are you?"
#   bash demo.sh "Tell me about your day"       # custom prompt
#
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python3}"

PROMPT="${1:-Who are you?}"
CACHE="${CACHE:-$ROOT/speculative/demo_cache_v2.pkl}"
K="${K:-16}"
MAX_TOK="${MAX_TOK:-512}"

if [ ! -f "$CACHE" ]; then
    echo "[demo] !! cache file not found: $CACHE"
    echo "[demo] !! Run once to bake (one-time, ~5-10 min):"
    echo "[demo] !!   $PYTHON -m mamba3_mlx.speculative.bake_cache --warmup_tokens 6144 --retrieval_max_suffix 16 --out $CACHE"
    exit 1
fi

"$PYTHON" -m mamba3_mlx.speculative.run_sjd_demo \
    --cache "$CACHE" \
    --prompt "$PROMPT" \
    --K "$K" \
    --max_tokens "$MAX_TOK"
