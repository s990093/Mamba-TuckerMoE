#!/usr/bin/env bash
# ==============================================================================
# Pure Metal 極限吞吐量基準（實驗）
#
# 穩定優先請使用：inference/run_stable_benchmark.sh
# 說明請閱讀：inference/INFERENCE_STACK.md
# ==============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "Starting experimental peak-throughput benchmark..."

exec .venv/bin/python inference/benchmark_mlx.py \
    --prompt "Mamba3-XR is a revolutionary" \
    --decode-tokens 2048 \
    --dtype bf16 \
    --quantize 4 \
    --tucker-einsum-fuse \
    --full-decode-compile \
    --fast-sample \
    --no-penalties \
    --no-materialize-caches \
    "$@"
