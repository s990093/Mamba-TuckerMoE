#!/bin/bash
# ==============================================================================
#  GPU Training Scheduler — 動態 GPU 排程 + 自動梯度累積調整 + 優雅停止
#
#  運作模式：
#    日用 (非 03:00–07:00):  GPU 1,2,3  (3 卡)
#    半夜 (03:00–07:00):    嘗試加到 5 卡 → 不足 5 則照順序遞減
#                            (優先加 3,4,5,6,0，先偵測是否空閒)
#
#  優雅停止 (graceful stop)：
#    切換 GPU 前先發信號 → 訓練跑完下一個 checkpoint 後自動退出
#    → scheduler 再以新 GPU 配置重啟 → 全程不硬殺
#
#  Session 記錄：
#    output/logs/.scheduler/sessions.csv  每次啟動/停止/切換都記錄
#
#  時區：固定使用台灣時間 (UTC+8)
#    NIGHT_START=3 → 台灣凌晨 03:00
#    NIGHT_END=7   → 台灣早上 07:00
#
#  用法：
#    ./scripts/training/gpu_scheduler.sh
#    ./scripts/training/gpu_scheduler.sh --foreground   (不背景化，前台跑)
#    ./scripts/training/gpu_scheduler.sh stop            (通知 scheduler 退出)
# ==============================================================================

set -euo pipefail
export TZ=Asia/Taipei

# ── 設定 ──────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCHEDULER_DIR="$PROJECT_ROOT/output/logs/.scheduler"
FLAG_GRACEFUL_STOP="$SCHEDULER_DIR/request_graceful_stop"
SESSION_LOG="$SCHEDULER_DIR/sessions.csv"
PID_FILE="$PROJECT_ROOT/output/logs/train.pid"
SCHEDULER_PID_FILE="$SCHEDULER_DIR/scheduler.pid"
LOG_DIR="$PROJECT_ROOT/output/logs"

BASE_GPUS="1,2,3"                  # 永遠使用的 GPU（日夜都用 3 卡）
NIGHT_EXTRA_ORDER=(4 5 6 0)        # 半夜額外 GPU 的嘗試順序
NIGHT_TARGET_TOTAL=5               # 半夜目標 GPU 總數
MAX_TOTAL_GPUS=6                   # 上限：留至少 1 張卡給別人 (7 張總數 → 最多用 6)
NIGHT_START=3                      # 3 AM
NIGHT_END=7                        # 7 AM
BATCH_SIZE=4
TARGET_EFF_BATCH=24                # 目標有效 batch size
CHECKPOINT_STEP_INTERVAL=100       # 對齊 train_sft_cot.py SAVE_EVERY_STEPS
GRACEFUL_TIMEOUT=1800              # 最多等 30 分鐘
GPU_FREE_THRESHOLD=5               # GPU 使用率 < 此值視為空閒
POLL_INTERVAL=60                   # scheduler 檢查間隔 (秒)

mkdir -p "$SCHEDULER_DIR"
mkdir -p "$LOG_DIR"


# ── Helper Functions ───────────────────────────────────────────────────

current_hour() {
    if [[ -n "${SIMULATE_HOUR:-}" ]]; then
        echo "$SIMULATE_HOUR"
    else
        date +%_H
    fi
}

is_night_window() {
    # 半夜 03:00–06:59 (>= NIGHT_START, < NIGHT_END)
    # 注意 07:00 已不在窗口內
    # 測試用：export SIMULATE_HOUR=3 強制模擬凌晨 3 點
    local h
    h=$(current_hour)
    [[ "$h" -ge "$NIGHT_START" && "$h" -lt "$NIGHT_END" ]]
}

gpu_is_free() {
    local gpu_id="$1"
    local util
    util=$(nvidia-smi --query-gpu=index,utilization.gpu \
        --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null \
        | cut -d',' -f2 | tr -d ' ')
    if [[ -z "$util" ]]; then
        return 1  # 查不到，當成忙碌
    fi
    [[ "$util" -lt "$GPU_FREE_THRESHOLD" ]]
}

detect_available_gpus() {
    # 回傳所有空閒的額外 GPU（依照 NIGHT_EXTRA_ORDER 順序）
    local available=()
    for gpu_id in "${NIGHT_EXTRA_ORDER[@]}"; do
        if gpu_is_free "$gpu_id"; then
            available+=("$gpu_id")
        fi
    done
    echo "${available[@]}"
}

calc_grad_accum() {
    # grad_accum = floor(TARGET_EFF_BATCH / (BATCH_SIZE * num_gpus))，最小 1
    # 多出來的 GPU 轉成速度，用整數除法（floor）優先降低 grad_accum
    local num_gpus="$1"
    local val=$(( TARGET_EFF_BATCH / (BATCH_SIZE * num_gpus) ))
    if [[ $val -lt 1 ]]; then
        echo 1
    else
        echo "$val"
    fi
}

build_gpu_list() {
    # 組出最終 GPU list：
    #   1. 不超過 MAX_TOTAL_GPUS（硬上限）
    #   2. 永遠留至少 1 張額外 GPU 空著（如果有多於 1 張可用的話）
    #   3. 不超過 NIGHT_TARGET_TOTAL
    local target_total="$1"
    shift
    local available=("$@")

    local base_count
    base_count=$(echo "$BASE_GPUS" | tr ',' '\n' | wc -l)

    # 硬上限
    if [[ $target_total -gt $MAX_TOTAL_GPUS ]]; then
        target_total=$MAX_TOTAL_GPUS
    fi

    local extra_wanted=$(( target_total - base_count ))
    local extra_available=${#available[@]}

    # 永遠留 1 張額外卡空著（如果有的話）
    local extra_usable=$(( extra_available - 1 ))
    if [[ $extra_usable -lt 0 ]]; then
        extra_usable=0
    fi

    # 由 MAX_TOTAL_GPUS 定的上限
    local max_extra=$(( MAX_TOTAL_GPUS - base_count ))

    # 實際取用 = min(想要, 可用-1, 硬上限)
    local take=$extra_wanted
    if [[ $take -gt $extra_usable ]]; then
        take=$extra_usable
    fi
    if [[ $take -gt $max_extra ]]; then
        take=$max_extra
    fi
    if [[ $take -lt 0 ]]; then
        take=0
    fi

    if [[ $take -le 0 ]]; then
        echo "$BASE_GPUS"
        return
    fi

    local extra_selected=()
    local count=0
    for gpu in "${available[@]}"; do
        if [[ $count -ge $take ]]; then
            break
        fi
        extra_selected+=("$gpu")
        count=$((count + 1))
    done

    if [[ ${#extra_selected[@]} -gt 0 ]]; then
        local extra_list
        extra_list=$(IFS=,; echo "${extra_selected[*]}")
        echo "${BASE_GPUS},${extra_list}"
    else
        echo "$BASE_GPUS"
    fi
}

get_train_pid() {
    if [[ -f "$PID_FILE" ]]; then
        cat "$PID_FILE"
    fi
}

is_training_running() {
    local pid
    pid=$(get_train_pid)
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

signal_graceful_stop() {
    touch "$FLAG_GRACEFUL_STOP"
    echo "$(date -Iseconds)  →  scheduler signaled graceful stop" >> "$FLAG_GRACEFUL_STOP"
}

clear_graceful_stop() {
    rm -f "$FLAG_GRACEFUL_STOP"
}

wait_for_stop() {
    local timeout="$1"
    local waited=0
    local interval=5

    while is_training_running; do
        sleep "$interval"
        waited=$((waited + interval))
        if [[ $waited -ge $timeout ]]; then
            echo "  ⚠️  Timeout after ${timeout}s, force killing..."
            local pid
            pid=$(get_train_pid)
            kill -9 "$pid" 2>/dev/null || true
            rm -f "$PID_FILE"
            return 1
        fi
        if [[ $((waited % 30)) -eq 0 ]]; then
            echo "  ... waiting (${waited}s elapsed)"
        fi
    done
    echo "  ✅ Training stopped after ${waited}s"
    return 0
}

get_current_step() {
    local latest
    latest=$(ls -t "$PROJECT_ROOT"/output/checkpoint_sft_cot_s*.pt 2>/dev/null | head -1)
    if [[ -n "$latest" ]]; then
        basename "$latest" | grep -oP '_s\K\d+' | head -1
    else
        echo "0"
    fi
}

write_session_event() {
    # 寫入 session CSV：event=start 或 end
    local session_id="$1"
    local event="$2"     # start / end
    local step="$3"
    local gpus="$4"
    local num_gpus="$5"
    local grad_accum="$6"
    local eff_batch="$7"
    local trigger="$8"

    local timestamp
    timestamp=$(date -Iseconds)

    if [[ ! -f "$SESSION_LOG" ]]; then
        echo "session_id,event,timestamp,gpu_devices,num_gpus,batch_size,grad_accum,eff_batch,step,trigger" \
            > "$SESSION_LOG"
    fi

    echo "$session_id,$event,$timestamp,$gpus,$num_gpus,$BATCH_SIZE,$grad_accum,$eff_batch,$step,$trigger" \
        >> "$SESSION_LOG"
}

launch_training() {
    local gpu_devices="$1"
    local num_gpus="$2"
    local grad_accum="$3"
    local eff_batch
    eff_batch=$(( BATCH_SIZE * grad_accum * num_gpus ))

    echo ""
    echo "════════════════════════════════════════════"
    echo "🚀 $(date '+%H:%M:%S')  Launching training"
    echo "   GPU devices : $gpu_devices"
    echo "   Num GPUs    : $num_gpus"
    echo "   Batch size  : $BATCH_SIZE"
    echo "   Grad accum  : $grad_accum"
    echo "   Eff batch   : $eff_batch"
    echo "════════════════════════════════════════════"

    if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "torch310" ]]; then
        echo "Loading conda env torch310 ..."
        eval "$(conda shell.bash hook)"
        conda activate torch310
    fi

    export CUDA_VISIBLE_DEVICES="$gpu_devices"

    if [[ -z "${TRITON_PTXAS_PATH:-}" ]]; then
        local ptxas_bin
        ptxas_bin="$(command -v ptxas || true)"
        if [[ -n "$ptxas_bin" ]]; then
            export TRITON_PTXAS_PATH="$ptxas_bin"
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

    local log_file
    log_file="$LOG_DIR/train_sft_cot_$(date +%Y%m%d_%H%M%S).log"

    export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}/scripts/data:${PYTHONPATH:-}"

    nohup accelerate launch \
        --num_processes="$num_gpus" \
        --mixed_precision=bf16 \
        --dynamo_backend=no \
        --gradient_accumulation_steps="$grad_accum" \
        "${PROJECT_ROOT}/scripts/train_sft_cot.py" \
        > "$log_file" 2>&1 &

    local train_pid=$!
    echo "$train_pid" > "$PID_FILE"
    echo "   PID  : $train_pid"
    echo "   Log  : $log_file"
    echo "   tail -f $log_file"
}


# ── Main Scheduler Loop ────────────────────────────────────────────────

main() {
    local foreground_mode=false
    if [[ "${1:-}" == "--foreground" ]]; then
        foreground_mode=true
    fi

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  🎯  GPU Training Scheduler"
    echo "══════════════════════════════════════════════════════"
    echo "  Day mode    : $BASE_GPUS (3 GPUs)"
    echo "  Night mode  : target $NIGHT_TARGET_TOTAL GPUs"
    echo "  Night extra : ${NIGHT_EXTRA_ORDER[*]} (priority order)"
    echo "  Hours       : ${NIGHT_START}:00 – ${NIGHT_END}:00"
    echo "  Eff batch   : target=$TARGET_EFF_BATCH  (auto grad_accum)"
    echo "  Init step   : $(get_current_step)"
    echo "  Project     : $PROJECT_ROOT"
    echo "  Log dir     : $SCHEDULER_DIR"
    echo "══════════════════════════════════════════════════════"
    echo ""

    # 避免重複啟動 scheduler
    if [[ -f "$SCHEDULER_PID_FILE" ]]; then
        local old_pid
        old_pid=$(cat "$SCHEDULER_PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "❌ Scheduler already running (PID=$old_pid)."
            echo "   Stop it first: $0 stop"
            exit 1
        fi
    fi
    echo $$ > "$SCHEDULER_PID_FILE"

    # 清除殘留的 graceful stop flag（上次 crash 等）
    clear_graceful_stop

    # 判斷初始模式
    local current_gpus current_num_gpus current_grad_accum current_mode
    if is_night_window; then
        echo "🌙 $(date '+%H:%M:%S') Starting in NIGHT mode — checking GPU availability..."
        local -a available
        read -ra available <<< "$(detect_available_gpus)"
        echo "   Available extra GPUs: ${available[*]:-(none)}"

        current_gpus=$(build_gpu_list "$NIGHT_TARGET_TOTAL" "${available[@]}")
        current_num_gpus=$(echo "$current_gpus" | tr ',' '\n' | wc -l)
        current_grad_accum=$(calc_grad_accum "$current_num_gpus")
        current_mode="night"
    else
        echo "☀️  $(date '+%H:%M:%S') Starting in DAY mode"
        current_gpus="$BASE_GPUS"
        current_num_gpus=2
        current_grad_accum=$(calc_grad_accum 2)
        current_mode="day"
    fi

    # 產生 session ID 並記錄開始
    local session_id
    session_id="s_$(date +%Y%m%d_%H%M%S)"
    local eff_batch
    eff_batch=$(( BATCH_SIZE * current_num_gpus * current_grad_accum ))

    write_session_event "$session_id" start "$(get_current_step)" \
        "$current_gpus" "$current_num_gpus" "$current_grad_accum" "$eff_batch" \
        "$current_mode"

    # 啟動訓練
    launch_training "$current_gpus" "$current_num_gpus" "$current_grad_accum"
    local start_ts
    start_ts=$(date +%s)

    # ── 監控迴圈 ──────────────────────────────────────────────────
    while true; do
        sleep "$POLL_INTERVAL"

        # 檢查 scheduler 是否被要求停止
        if [[ -f "$SCHEDULER_DIR/stop_scheduler" ]]; then
            echo "🛑 $(date '+%H:%M:%S') Scheduler stop requested."
            signal_graceful_stop
            wait_for_stop "$GRACEFUL_TIMEOUT"
            session_end_step=$(get_current_step)
            local session_dur
            session_dur=$(( ($(date +%s) - start_ts) / 60 ))
            write_session_event "$session_id" end "$session_end_step" \
                "$current_gpus" "$current_num_gpus" "$current_grad_accum" \
                "$eff_batch" "scheduler_stopped"
            rm -f "$SCHEDULER_PID_FILE"
            echo "👋 Scheduler exiting."
            exit 0
        fi

        # 檢查訓練是否 crash
        if ! is_training_running; then
            echo "⚠️  $(date '+%H:%M:%S') Training process died unexpectedly."
            session_end_step=$(get_current_step)
            session_dur=$(( ($(date +%s) - start_ts) / 60 ))
            write_session_event "$session_id" end "$session_end_step" \
                "$current_gpus" "$current_num_gpus" "$current_grad_accum" \
                "$eff_batch" "crashed"

            # 嘗試重啟
            session_id="s_$(date +%Y%m%d_%H%M%S)"
            clear_graceful_stop

            if is_night_window; then
                local -a available_2
                read -ra available_2 <<< "$(detect_available_gpus)"
                echo "   (auto-restart) Available extra GPUs: ${available_2[*]:-(none)}"
                current_gpus=$(build_gpu_list "$NIGHT_TARGET_TOTAL" "${available_2[@]}")
                current_num_gpus=$(echo "$current_gpus" | tr ',' '\n' | wc -l)
                current_grad_accum=$(calc_grad_accum "$current_num_gpus")
                current_mode="night"
            else
                current_gpus="$BASE_GPUS"
                current_num_gpus=2
                current_grad_accum=$(calc_grad_accum 2)
                current_mode="day"
            fi
            eff_batch=$(( BATCH_SIZE * current_num_gpus * current_grad_accum ))
            write_session_event "$session_id" start "$(get_current_step)" \
                "$current_gpus" "$current_num_gpus" "$current_grad_accum" "$eff_batch" \
                "auto_restart"
            launch_training "$current_gpus" "$current_num_gpus" "$current_grad_accum"
            start_ts=$(date +%s)
            continue
        fi

        # ── 判斷是否需要切換模式 ──
        local in_night_now
        in_night_now=false
        is_night_window && in_night_now=true

        if [[ "$current_mode" == "night" && "$in_night_now" == "false" ]]; then
            # 半夜結束 → 切回白天模式
            echo ""
            echo "☀️  $(date '+%H:%M:%S') Night window ended → switching to DAY (2 GPUs) ..."

            signal_graceful_stop
            wait_for_stop "$GRACEFUL_TIMEOUT"

            session_end_step=$(get_current_step)
            session_dur=$(( ($(date +%s) - start_ts) / 60 ))
            write_session_event "$session_id" end "$session_end_step" \
                "$current_gpus" "$current_num_gpus" "$current_grad_accum" \
                "$eff_batch" "night_end"

            session_id="s_$(date +%Y%m%d_%H%M%S)"
            clear_graceful_stop
            current_gpus="$BASE_GPUS"
            current_num_gpus=2
            current_grad_accum=$(calc_grad_accum 2)
            current_mode="day"
            eff_batch=$(( BATCH_SIZE * current_num_gpus * current_grad_accum ))
            write_session_event "$session_id" start "$(get_current_step)" \
                "$current_gpus" "$current_num_gpus" "$current_grad_accum" "$eff_batch" \
                "night_end"
            launch_training "$current_gpus" "$current_num_gpus" "$current_grad_accum"
            start_ts=$(date +%s)

        elif [[ "$current_mode" == "day" && "$in_night_now" == "true" ]]; then
            # 半夜開始 → 偵測 GPU 然後切換
            echo ""
            echo "🌙 $(date '+%H:%M:%S') Night window started → checking GPU availability..."

            local -a available_night
            read -ra available_night <<< "$(detect_available_gpus)"
            echo "   Available extra GPUs: ${available_night[*]:-(none)}"

            local candidate_gpus
            candidate_gpus=$(build_gpu_list "$NIGHT_TARGET_TOTAL" "${available_night[@]}")
            local candidate_ngpu
            candidate_ngpu=$(echo "$candidate_gpus" | tr ',' '\n' | wc -l)

            if [[ $candidate_ngpu -gt 2 ]]; then
                echo "   Switching to $candidate_ngpu GPUs: $candidate_gpus"

                signal_graceful_stop
                wait_for_stop "$GRACEFUL_TIMEOUT"

                session_end_step=$(get_current_step)
                session_dur=$(( ($(date +%s) - start_ts) / 60 ))
                write_session_event "$session_id" end "$session_end_step" \
                    "$current_gpus" "$current_num_gpus" "$current_grad_accum" \
                    "$eff_batch" "night_start"

                session_id="s_$(date +%Y%m%d_%H%M%S)"
                clear_graceful_stop
                current_gpus="$candidate_gpus"
                current_num_gpus=$candidate_ngpu
                current_grad_accum=$(calc_grad_accum "$candidate_ngpu")
                current_mode="night"
                eff_batch=$(( BATCH_SIZE * current_num_gpus * current_grad_accum ))
                write_session_event "$session_id" start "$(get_current_step)" \
                    "$current_gpus" "$current_num_gpus" "$current_grad_accum" "$eff_batch" \
                    "night_start"
                launch_training "$current_gpus" "$current_num_gpus" "$current_grad_accum"
                start_ts=$(date +%s)
            else
                echo "   No extra GPUs available — staying in 2-GPU day mode"
            fi
        fi
    done
}


# ── entry ──────────────────────────────────────────────────────────────

case "${1:-}" in
    stop)
        if [[ -f "$SCHEDULER_PID_FILE" ]]; then
            touch "$SCHEDULER_DIR/stop_scheduler"
            echo "🛑 Stop signal sent to scheduler."
        else
            echo "ℹ️  No scheduler running (no PID file)."
            # 還是嘗試 graceful stop 當前的訓練
            signal_graceful_stop
            echo "🔔 Graceful stop signal sent to training (if running)."
        fi
        ;;
    status)
        if [[ -f "$SCHEDULER_PID_FILE" ]]; then
            local spid
            spid=$(cat "$SCHEDULER_PID_FILE")
            if kill -0 "$spid" 2>/dev/null; then
                echo "✅ Scheduler running (PID=$spid)"
            else
                echo "⚠️  Stale scheduler PID file (PID=$spid is dead)"
            fi
        else
            echo "ℹ️  No scheduler PID file."
        fi
        if is_training_running; then
            local tpid
            tpid=$(get_train_pid)
            local tstep
            tstep=$(get_current_step)
            echo "✅ Training running (PID=$tpid, last ckpt step=$tstep)"
        else
            echo "ℹ️  No training process running."
        fi
        echo "   Night window now: $(is_night_window && echo 'YES' || echo 'NO')"
        ;;
    --foreground)
        main --foreground
        ;;
    *)
        # 預設：背景化啟動
        echo "Starting GPU scheduler in background..."
        nohup bash "$0" --foreground > "$SCHEDULER_DIR/scheduler.log" 2>&1 &
        local mypid=$!
        echo "   Scheduler PID: $mypid"
        echo "   Log: tail -f $SCHEDULER_DIR/scheduler.log"
        echo "   Stop: $0 stop"
        echo "   Status: $0 status"
        ;;
esac
