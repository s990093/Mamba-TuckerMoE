#!/bin/bash
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "torch310" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate torch310
fi

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"

# 權重：手動指定 CKPT_PATH，否則用 output/ 最新 checkpoint_sft_cot_s*.pt
CKPT_LOAD=""
if [[ -n "${CKPT_PATH:-}" ]]; then
    echo "📦 使用指定 checkpoint: $CKPT_PATH"
    CKPT_LOAD="$CKPT_PATH"
else
    LATEST_CKPT=$(ls -t output/checkpoint_sft_cot_s*.pt 2>/dev/null | head -1)
    if [[ -n "$LATEST_CKPT" ]]; then
        echo "📦 使用最新 checkpoint: $LATEST_CKPT"
        CKPT_LOAD="$LATEST_CKPT"
    else
        echo "ℹ️  output/ 無 checkpoint，使用 Kaggle base model（見 sft_cli）"
    fi
fi

MODE="${1:-interactive}"
SYSTEM="${SYSTEM:-}"
REASONING="${REASONING:-}"

# 解碼預設對齊 train_sft 的 SFT 週期解碼預設（見 train_sft.py：SFT_TEST_TEMPERATURE=0 為 greedy，
# SFT_TEST_REPETITION_PENALTY=1.0、TOP_K=None 時內部用 50）。訓練本體為 teacher forcing，
# 推論最接近「每步 argmax」= 低溫 / greedy。
# 可覆寫：TEMPERATURE TOP_P TOP_K REPETITION_PENALTY MAX_NEW_TOKENS
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-30}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"

echo "🎛 decode: T=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K rep_pen=$REPETITION_PENALTY${MAX_NEW_TOKENS:+ max_new=$MAX_NEW_TOKENS}"

MAX_NEW_ARG=()
if [[ -n "$MAX_NEW_TOKENS" ]]; then
    MAX_NEW_ARG=(--max_new_tokens "$MAX_NEW_TOKENS")
fi

COMMON_ARGS=()
if [[ -n "$CKPT_LOAD" ]]; then
    COMMON_ARGS+=(--checkpoint "$CKPT_LOAD")
fi
COMMON_ARGS+=(
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --top_k "$TOP_K"
    --repetition_penalty "$REPETITION_PENALTY"
    "${MAX_NEW_ARG[@]}"
)
if [[ -n "$REASONING" ]] || [[ "$MODE" == *"reasoning"* ]]; then
    COMMON_ARGS+=(--reasoning)
fi
if [[ -n "$SYSTEM" ]]; then
    COMMON_ARGS+=(--system "$SYSTEM")
fi

case "$MODE" in
    interactive|reasoning)
        python3 "${PROJECT_ROOT}/scripts/sft_cli.py" \
            "${COMMON_ARGS[@]}" \
            --interactive
        ;;
    test)
        python3 "${PROJECT_ROOT}/scripts/sft_cli.py" \
            "${COMMON_ARGS[@]}" \
            --prompt "Who are you?" \
            --system self_awareness
        ;;
    *)
        echo "Usage:"
        echo "  ./scripts/chat.sh              # 互動對話（與 train_sft 預設 greedy 對齊）"
        echo "  ./scripts/chat.sh interactive   # 互動對話"
        echo "  ./scripts/chat.sh reasoning     # 互動 + CoT reasoning"
        echo "  ./scripts/chat.sh test          # 單次測試 (Who are you? + self_awareness)"
        echo ""
        echo "  REASONING=1 ./scripts/chat.sh"
        echo "  SYSTEM=emotion ./scripts/chat.sh"
        echo "  CKPT_PATH=output/checkpoint_sft_cot_s560.pt ./scripts/chat.sh"
        echo ""
        echo "  解碼覆寫（預設 T=0 greedy, rep_pen=1.0, top_k=50, top_p=1.0）："
        echo "  TEMPERATURE=0.8 TOP_P=0.85 TOP_K=40 REPETITION_PENALTY=1.2 ./scripts/chat.sh"
        echo "  MAX_NEW_TOKENS=64 ./scripts/chat.sh   # 與 train_sft SFT_TEST_MAX_NEW 相同"
        echo ""
        echo "Available system buckets:"
        echo "  emotion, self_awareness, summarize_email, movie_intro,"
        echo "  daily_conversation, system_call, deep_dive"
        exit 1
        ;;
esac
