#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  CoT Middleware 修复验证  - 对比测试                             ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

PROMPT="${1:-who are you?}"
RUNS="${2:-3}"

echo ""
echo "测试提示: $PROMPT"
echo "运行次数: $RUNS"
echo ""

# 创建结果文件
RESULT_FILE="/tmp/cot_fix_results_$(date +%s).txt"

for ((i=1; i<=RUNS; i++)); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧪 Run $i/$RUNS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    ./chat_precise.sh "$PROMPT" 2>&1 | tee -a "$RESULT_FILE"
    
    echo ""
    sleep 1
done

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  结果分析                                                         ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

echo ""
echo "✅ 检查点："
echo "1. 输出结构是否完整（有 <think>/<final>）"
echo "2. 推理步骤是否清晰（Step 1/2/3/...）"
echo "3. Token 是否正常（无 ##egressive 等损坏 token）"
echo "4. 多轮是否一致（结构相同）"

echo ""
echo "📊 输出统计："
grep -c "<think>" "$RESULT_FILE" | xargs echo "  <think> 块数:"
grep -c "<final>" "$RESULT_FILE" | xargs echo "  <final> 块数:"
grep -c "\[benchmark\]" "$RESULT_FILE" | xargs echo "  benchmark 行数:"

echo ""
echo "💾 完整结果: $RESULT_FILE"

echo ""
echo "┌─ 对比说明"
echo "│  如果输出都正常且一致 → 修复成功 ✅"
echo "│  如果仍有 token 损坏 → 可能是其他原因（H1/H3/H4）"
echo "└─"

