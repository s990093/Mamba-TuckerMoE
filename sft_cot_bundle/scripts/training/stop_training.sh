#!/bin/bash
# ==============================================================================
#  stop_training.sh — 優雅停止訓練（先發 graceful stop 信號，等 checkpoint 再結束）
#
#  優先順序：
#   1. scheduler 還活著 → 呼叫 scheduler stop（由 scheduler 負責優雅停止）
#   2. 只有 training → 發 graceful stop flag，等 training 自己在下個 checkpoint 退出
#   3. 逾時 → SIGTERM → SIGKILL
# ==============================================================================

set -e
export TZ=Asia/Taipei

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCHEDULER_DIR="$PROJECT_ROOT/output/logs/.scheduler"
FLAG_GRACEFUL_STOP="$SCHEDULER_DIR/request_graceful_stop"
PID_FILE="$PROJECT_ROOT/output/logs/train.pid"
SCHEDULER_PID_FILE="$SCHEDULER_DIR/scheduler.pid"
GRACEFUL_TIMEOUT="${STOP_TIMEOUT:-1200}"   # 預設等 20 分鐘
KILL_TIMEOUT=30

mkdir -p "$SCHEDULER_DIR"

# ── 有 scheduler → 請 scheduler 處理 ──────────────────────────────────
if [[ -f "$SCHEDULER_PID_FILE" ]]; then
    SCHEDULER_PID=$(cat "$SCHEDULER_PID_FILE")
    if kill -0 "$SCHEDULER_PID" 2>/dev/null; then
        echo "🛑 Asking scheduler to stop (PID=$SCHEDULER_PID) ..."
        touch "$SCHEDULER_DIR/stop_scheduler"
        echo "   Scheduler will handle graceful stop. This may take a few minutes."
        echo "   Monitor: tail -f $SCHEDULER_DIR/scheduler.log"
        exit 0
    fi
fi

# ── 沒有 scheduler，直接對 training 發 graceful stop ─────────────────
if [[ ! -f "$PID_FILE" ]]; then
    echo "❌ 找不到 PID 檔：$PID_FILE"
    echo "   嘗試搜尋 train_sft_cot 相關 process ..."
    pgrep -af "train_sft_cot" || echo "   沒有找到正在跑的訓練 process。"
    exit 1
fi

TRAIN_PID=$(cat "$PID_FILE")

if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "ℹ️  PID=$TRAIN_PID 已不存在（訓練可能已結束）。"
    rm -f "$PID_FILE"
    rm -f "$FLAG_GRACEFUL_STOP"
    exit 0
fi

echo "🔔 Sending graceful stop signal (training will stop at next checkpoint) ..."
echo "   PID: $TRAIN_PID"
touch "$FLAG_GRACEFUL_STOP"
echo "$(date -Iseconds)  →  manual stop requested" >> "$FLAG_GRACEFUL_STOP"

# ── 等待優雅停止 ─────────────────────────────────────────────────────
WAITED=0
INTERVAL=10
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "$INTERVAL"
    WAITED=$((WAITED + INTERVAL))
    if [[ $WAITED -ge $GRACEFUL_TIMEOUT ]]; then
        echo "⚠️  Graceful stop timeout after ${WAITED}s — sending SIGTERM ..."
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        sleep "$KILL_TIMEOUT"
        if kill -0 "$TRAIN_PID" 2>/dev/null; then
            echo "⚠️  SIGTERM failed — sending SIGKILL ..."
            pkill -KILL -P "$TRAIN_PID" 2>/dev/null || true
            kill -KILL "$TRAIN_PID" 2>/dev/null || true
        fi
        break
    fi
    if [[ $((WAITED % 60)) -eq 0 ]]; then
        echo "   ... waiting for next checkpoint (${WAITED}s elapsed)"
    fi
done

rm -f "$PID_FILE"
rm -f "$FLAG_GRACEFUL_STOP"
echo "✅ Training stopped."
