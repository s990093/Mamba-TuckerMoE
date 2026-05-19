#!/bin/bash
# 测试脚本：对比启用/禁用 CoT MW 的日志差异

set -e

cd "$(dirname "$0")"

MODEL_PATH="${1:-.}"
TOKENIZER_PATH="${2:-./checkpoints/tokenizer}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 CoT Middleware Debug Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test 1: 禁用 CoT MW（通过修改 server.py）
echo ""
echo "🧪 Test 1: 禁用 CoT Middleware (enable_cot_mw=False)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 创建临时修改版本
TEMP_SERVER="/tmp/server_disabled_mw.py"
cp mamba3_mlx/server.py "$TEMP_SERVER"

# 临时禁用 CoT MW
sed -i '' 's/enable_cot_mw = msg.get("enable_cot_middleware", True)/enable_cot_mw = False  # TEMP DISABLED FOR TEST/' "$TEMP_SERVER"

echo "✏️  Running with enable_cot_mw = False..."
python3 -c "
import sys
sys.path.insert(0, '.')
from mamba3_mlx.server_config import resolve_system_prompt

# 模拟 WS 消息
msg = {
    'action': 'chat',
    'prompt': 'who are you?',
    'max_tokens': 200,
    'reasoning': True,
    'sampling': {},
}

print('Message:', msg)
print('enable_cot_middleware=False')
" 2>&1 | head -20

# Test 2: 启用 CoT MW
echo ""
echo "🧪 Test 2: 启用 CoT Middleware (enable_cot_mw=True)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✏️  Running with enable_cot_mw = True (default)..."
python3 -c "
import sys
sys.path.insert(0, '.')

msg = {
    'action': 'chat',
    'prompt': 'who are you?',
    'max_tokens': 200,
    'reasoning': True,
    'sampling': {},
}

print('Message:', msg)
print('enable_cot_middleware=True')
" 2>&1 | head -20

# 清理
rm -f "$TEMP_SERVER"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 要看完整日志，请运行："
echo "   python -m mamba3_mlx.server 2>&1 | grep -E '\[logits\]|\[mw\]|\[inj\]'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
