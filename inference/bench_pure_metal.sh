#!/bin/bash
# ==============================================================================
# Pure Metal 終極優化推論腳本 (Mamba3-XR)
# 
# 啟用機制:
# 1. --quantize 4: 4-bit 權重量化 (極致減少記憶體頻寬負擔)
# 2. --tucker-einsum-fuse: Fused TuckerMoE Custom Metal Kernel
# 3. --tucker-full-fuse: Dense 權重時啟用 U_in/RMS/G/U_out 單 kernel 實驗
# 4. --full-decode-compile: 全圖即時編譯 (消除所有 Python Overhead)
# 5. --fast-sample: 關閉所有無效的 softmax / top_k 計算，直出 argmax
# 6. --no-penalties / --no-materialize-caches: 移除 benchmark 熱路徑外的同步與 logits 修正
# 7. --dtype bf16: 使用 Apple Silicon 硬體原生優化的 BFloat16 格式
# 8. 可另加 --speculative-decode --spec-draft-tokens 5 測投機驗證路徑
# ==============================================================================

echo "🚀 啟動 Pure Metal 最強優化推論..."

.venv/bin/python inference/benchmark_mlx.py \
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
