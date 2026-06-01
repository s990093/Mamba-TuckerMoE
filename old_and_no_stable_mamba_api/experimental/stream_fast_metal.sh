#!/usr/bin/env bash
# ==============================================================================
# 「極速」串流（實驗）：與本目錄 bench_pure_metal.sh 對齊的旗標組合。
#
# - bf16 + kv auto、4-bit 量化、Tucker einsum fuse、全圖 decode compile
# - greedy (--fast-sample)、無 penalties、不重複複製 kv/mamba caches
# - fused Metal v2 取樣（greedy 時與 argmax 對齊度高，仍屬實驗路徑）
#
# 穩定優先請使用：inference/run_stable_stream.sh
# ==============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

exec .venv/bin/python inference/stream_mlx.py \
    --max-new-tokens 10000 \
    --dtype bf16 \
    --kv-dtype auto \
    --quantize 4 \
    --tucker-einsum-fuse \
    --full-decode-compile \
    --fused-sample-metal-v2 \
    --fast-sample \
    "$@"
