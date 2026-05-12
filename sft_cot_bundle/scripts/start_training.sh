#!/bin/bash
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="${ENV_NAME:-py310}"
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "Loading conda env ${ENV_NAME} ..."
    CONDA_HOME="${CONDA_HOME:-/home/hungwei/miniconda3}"
    # Use explicit conda path so nohup/setsid background shell can still activate env.
    source "${CONDA_HOME}/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi

# 使用第 2,3,4 張卡（共 3 張）進行 DDP
export CUDA_VISIBLE_DEVICES=0
# Ensure Triton uses system CUDA 11.8 ptxas in both fg/bg runs.
if [[ -z "${TRITON_PTXAS_PATH:-}" ]]; then
    PTXAS_BIN="$(command -v ptxas || true)"
    if [[ -n "$PTXAS_BIN" ]]; then
        export TRITON_PTXAS_PATH="$PTXAS_BIN"
    fi
fi

echo "=========================================================="
echo " TuckerMoE ~670M  |  3x RTX 3090 24GB  |  Multi-GPU DDP"
echo "=========================================================="
if [[ -n "${TRITON_PTXAS_PATH:-}" ]]; then
    echo "TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH}"
fi

export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export PYTHONWARNINGS="ignore::UserWarning"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="${HOME}/.triton/cache"
mkdir -p "$TRITON_CACHE_DIR"


# export NCCL_DEBUG=WARN
# export NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_IB_DISABLE=1       # 若是單機多卡 PCIe，通常關掉 IB 更穩

# 透過 Accelerate 啟動多卡訓練
# 注意：移除了原本針對單卡的限制，並加入 --multi_gpu
accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --gradient_accumulation_steps=1 \
    scripts/train.py