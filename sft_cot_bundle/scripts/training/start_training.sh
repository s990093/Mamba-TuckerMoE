#!/bin/bash
# ==============================================================================
#  start_training.sh — 啟動訓練（支援 GPU Scheduler 或手動模式）
#
#  預設模式：啟動 gpu_scheduler.sh（自動 GPU 排程 + 優雅切換）
#
#  手動模式（當環境變數 GPU_DEVICES / NUM_GPUS / GRAD_ACCUM 有值時跳過 scheduler）：
#     GPU_DEVICES=1,2 NUM_GPUS=2 GRAD_ACCUM=3 ./scripts/training/start_training.sh
#     GPU_DEVICES=1,2,3,4,5 NUM_GPUS=5 GRAD_ACCUM=2 ./scripts/training/start_training.sh
#
#  Scheduler 用法：
#     ./scripts/training/start_training.sh              # 啟動 scheduler（背景）
#     ./scripts/training/start_training.sh status        # 查看 scheduler 狀態
#     ./scripts/training/start_training.sh stop          # 優雅停止 scheduler + training
# ==============================================================================

set -e
export TZ=Asia/Taipei

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCHEDULER_SCRIPT="$PROJECT_ROOT/scripts/training/gpu_scheduler.sh"
SCHEDULER_DIR="$PROJECT_ROOT/output/logs/.scheduler"
PID_FILE="$PROJECT_ROOT/output/logs/train.pid"
SCHEDULER_PID_FILE="$SCHEDULER_DIR/scheduler.pid"
LOG_DIR="$PROJECT_ROOT/output/logs"

mkdir -p "$LOG_DIR"
mkdir -p "$SCHEDULER_DIR"

# ── conda env ──────────────────────────────────────────────────────────
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "torch310" ]]; then
    echo "Loading conda env torch310 ..."
    eval "$(conda shell.bash hook)"
    conda activate torch310
fi

# ── status / stop 子命令 ───────────────────────────────────────────────
case "${1:-}" in
    status)
        if [[ -f "$SCHEDULER_PID_FILE" ]]; then
            spid=$(cat "$SCHEDULER_PID_FILE")
            if kill -0 "$spid" 2>/dev/null; then
                echo "✅ Scheduler running (PID=$spid)"
            else
                echo "⚠️  Stale scheduler PID file (PID=$spid is dead)"
            fi
        else
            echo "ℹ️  No scheduler PID file."
        fi
        if [[ -f "$PID_FILE" ]]; then
            tpid=$(cat "$PID_FILE")
            if kill -0 "$tpid" 2>/dev/null; then
                local tstep
                tstep=$(ls -t "$PROJECT_ROOT"/output/checkpoint_sft_cot_s*.pt 2>/dev/null | head -1 | grep -oP '_s\K\d+' | head -1 || echo "?")
                echo "✅ Training running (PID=$tpid, last ckpt step=$tstep)"
            else
                echo "ℹ️  Stale training PID file."
            fi
        else
            echo "ℹ️  No training process running."
        fi
        echo "   Scheduler log: tail -f $SCHEDULER_DIR/scheduler.log"
        echo "   Sessions:      cat $SCHEDULER_DIR/sessions.csv"
        exit 0
        ;;
    stop)
        if [[ -f "$SCHEDULER_PID_FILE" ]]; then
            bash "$SCHEDULER_SCRIPT" stop
        elif [[ -f "$PID_FILE" ]]; then
            # 沒有 scheduler，直接對 training 發 graceful stop
            touch "$SCHEDULER_DIR/request_graceful_stop"
            echo "🔔 Graceful stop signal sent. Training will stop at next checkpoint."
            echo "   tail -f \$(ls -t $LOG_DIR/train_sft_cot_*.log | head -1)"
        else
            echo "ℹ️  Nothing to stop."
        fi
        exit 0
        ;;
esac

# ── 直接模式（預設 3 卡，不跑 scheduler、不做時間切換）──────────────
# 可透過環境變數覆寫：GPU_DEVICES / NUM_GPUS / GRAD_ACCUM
gpus="${GPU_DEVICES:-1,2}"
ngpu="${NUM_GPUS:-2}"
grad_accum="${GRAD_ACCUM:-2}"
eff_batch=$((4 * ngpu * grad_accum))

echo "⚡ Direct mode"
echo ""

echo "════════════════════════════════════════════"
echo "🚀 Launching training"
echo "   GPU devices : $gpus"
echo "   Num GPUs    : $ngpu"
echo "   Batch size  : 4"
echo "   Grad accum  : $grad_accum"
echo "   Eff batch   : $eff_batch"
echo "════════════════════════════════════════════"

if [[ -f "$PID_FILE" ]]; then
    oldpid=$(cat "$PID_FILE")
    if kill -0 "$oldpid" 2>/dev/null; then
        echo "❌ Training already running (PID=$oldpid). Stop it first: $0 stop"
        exit 1
    fi
fi

export CUDA_VISIBLE_DEVICES="$gpus"

if [[ -z "${TRITON_PTXAS_PATH:-}" ]]; then
    PTXAS_BIN="$(command -v ptxas || true)"
    if [[ -n "$PTXAS_BIN" ]]; then
        export TRITON_PTXAS_PATH="$PTXAS_BIN"
    fi
fi

export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export PYTHONWARNINGS="ignore::UserWarning"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="${HOME}/.triton/cache"
mkdir -p "$TRITON_CACHE_DIR"

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1

export SFT_COT_AUTO_RESUME=1

LOG_FILE="$LOG_DIR/train_sft_cot_$(date +%Y%m%d_%H%M%S).log"

export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}/scripts/data:${PYTHONPATH:-}"

nohup accelerate launch \
    --num_processes="$ngpu" \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --gradient_accumulation_steps="$grad_accum" \
    "${PROJECT_ROOT}/scripts/train_sft_cot.py" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "   PID  : $TRAIN_PID"
echo "   Log  : $LOG_FILE"
echo "   tail -f $LOG_FILE"
echo "   Stop : $0 stop"
exit 0
