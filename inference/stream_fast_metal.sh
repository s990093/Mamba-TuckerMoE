#!/bin/bash
# ==============================================================================
# 與 inference/bench_pure_metal.sh 對齊的「最快」串流生成腳本
#
# - bf16 + kv auto、4-bit 量化、Tucker einsum fuse、全圖 decode compile
# - greedy (--fast-sample)、無 penalties、不重複複製 kv/mamba caches
# - stream_mlx 預設：最多 2048 新 token；遇 EOS 仍繼續（要停請加 --stop-on-eos）
#
# 需要較保守行為請改用手動參數，例如：
#   --materialize-caches --no-fast-sample --enable-penalties
# ==============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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