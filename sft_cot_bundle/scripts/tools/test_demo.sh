#!/bin/bash
# ==============================================================================
#  test_demo.sh — 在 GPU 0 上測試 sft_cli.py decode 品質（不影響訓練）
#  Usage:  bash scripts/tools/test_demo.sh
# ==============================================================================
set -e

PYTHON=/home/hungwei/miniforge3/envs/torch310/bin/python
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT_DIR="$PROJECT_ROOT/output"
LOG="$CKPT_DIR/logs/demo_test_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT_ROOT/scripts:$PROJECT_ROOT/scripts/data"

# ---- 找最新 checkpoint ----
CKPT=$(ls -t "$CKPT_DIR"/checkpoint_sft_cot_s*.pt 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
    echo "❌ 找不到 checkpoint" | tee "$LOG"
    exit 1
fi

PROMPTS=(
    "What is 2+2?"
    "Mamba, what exactly are you? Are you some kind of chatbot?"
    "Explain what is quantum computing in simple terms."
    "Write a short poem about artificial intelligence."
    "What is the capital of France and tell me something about it."
    "How do I make a good cup of coffee?"
)

echo "=== Demo Test - $(date) ===" | tee "$LOG"
echo "   CKPT: $(basename "$CKPT")" | tee -a "$LOG"
echo "   GPU:  $CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
echo | tee -a "$LOG"

for prompt in "${PROMPTS[@]}"; do
    echo "===== PROMPT =====" | tee -a "$LOG"
    echo "$prompt" | tee -a "$LOG"
    echo "===== OUTPUT =====" | tee -a "$LOG"
    $PYTHON "$PROJECT_ROOT/scripts/sft_cli.py" \
        --checkpoint "$CKPT" \
        --reasoning \
        --max_new_tokens 192 \
        --temperature 0.7 \
        --prompt "$prompt" \
        2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
    echo "-------------------" | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "=== Done - $(date) ===" | tee -a "$LOG"
echo "Log: $LOG"
