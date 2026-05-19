#!/bin/bash
# 在 Mac 上运行完整的 infer_cot.py 测试
# 将结果保存到 infer_cot_test_results.txt

OUTPUT_FILE="infer_cot_test_results.txt"

{
    echo "╔════════════════════════════════════════════════════════════════════════════════╗"
    echo "║                  infer_cot.py COMPREHENSIVE TEST SUITE                        ║"
    echo "║                   (After middleware.step() Integration Fix)                   ║"
    echo "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Test Date: $(date)"
    echo "System: $(uname -s)"
    echo "Working Directory: $(pwd)"
    echo ""

    echo "════════════════════════════════════════════════════════════════════════════════════"
    echo "TEST 1: Self-Awareness"
    echo "════════════════════════════════════════════════════════════════════════════════════"
    echo "Command: python -m mamba3_mlx.infer_cot --prompt 'Who are you?' --category self_awareness"
    echo ""
    python -m mamba3_mlx.infer_cot --prompt "Who are you?" --category self_awareness 2>&1
    TEST1_RESULT=$?
    echo ""

    echo "════════════════════════════════════════════════════════════════════════════════════"
    echo "TEST 2: Emotion Mode"
    echo "════════════════════════════════════════════════════════════════════════════════════"
    echo "Command: python -m mamba3_mlx.infer_cot --prompt 'I'm feeling overwhelmed' --category emotion"
    echo ""
    python -m mamba3_mlx.infer_cot --prompt "I'm feeling overwhelmed" --category emotion 2>&1
    TEST2_RESULT=$?
    echo ""

    echo "════════════════════════════════════════════════════════════════════════════════════"
    echo "TEST 3: Deep Dive"
    echo "════════════════════════════════════════════════════════════════════════════════════"
    echo "Command: python -m mamba3_mlx.infer_cot --prompt 'Explain quantum computing' --category deep_dive --max-tokens 400"
    echo ""
    python -m mamba3_mlx.infer_cot --prompt "Explain quantum computing" --category deep_dive --max-tokens 400 2>&1
    TEST3_RESULT=$?
    echo ""

    echo "════════════════════════════════════════════════════════════════════════════════════"
    echo "TEST 4: Email Summary"
    echo "════════════════════════════════════════════════════════════════════════════════════"
    echo "Command: python -m mamba3_mlx.infer_cot --prompt 'Summarize this email...' --category email_summary"
    echo ""
    python -m mamba3_mlx.infer_cot --prompt "Summarize this email about the project timeline and deliverables" --category email_summary 2>&1
    TEST4_RESULT=$?
    echo ""

    echo "════════════════════════════════════════════════════════════════════════════════════"
    echo "SUMMARY"
    echo "════════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Expected Results (After Fix):"
    echo "  ✓ Tokens generated: 100+ (not 10-21)"
    echo "  ✓ Has reasoning: True"
    echo "  ✓ Has final answer: True"
    echo "  ✓ Reached final mode: True"
    echo "  ✓ think_tokens: > 0 (properly counted)"
    echo "  ✓ FSM mode: progresses to 'final' or 'done'"
    echo "  ✓ Stop reason: middleware_stop or stop_token"
    echo ""
    echo "Key Differences from Before:"
    echo "  Before: Tokens=10, think_tokens=0, has_reasoning=False, output='EncEncName...'"
    echo "  After:  Tokens=100+, think_tokens=50+, has_reasoning=True, output='coherent text'"
    echo ""

} | tee "$OUTPUT_FILE"

echo ""
echo "✅ Test results saved to: $(pwd)/$OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "  1. Review the QUALITY METRICS in each test section"
echo "  2. Verify all tests show ✅ PASS"
echo "  3. Confirm tokens generated are 100+ (not 10-21)"
echo "  4. Check that reasoning and final answer blocks are present"
