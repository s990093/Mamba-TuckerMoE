#!/bin/bash
# 快速场景测试脚本
# 用法: ./test_scenarios.sh [scenario]
# scenario: baseline, no-injection, no-transform, no-mw, all

set -e
cd "$(dirname "$0")"

SCENARIO="${1:-baseline}"
PROMPT="${2:-who are you?}"
RUNS="${3:-1}"

# 备份原始文件
BACKUP_FILE="/tmp/server_backup_$(date +%s).py"
cp mamba3_mlx/server.py "$BACKUP_FILE"

echo "✅ Backed up to: $BACKUP_FILE"
trap "echo '⏮️  Restoring original...'; cp '$BACKUP_FILE' mamba3_mlx/server.py; rm -f '$BACKUP_FILE'" EXIT

run_test() {
    local name="$1"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧪 Scenario: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Prompt: $PROMPT"
    echo "Runs: $RUNS"
    echo ""

    for ((i=1; i<=RUNS; i++)); do
        echo "  [Run $i/$RUNS]"
        ./chat_precise.sh "$PROMPT" 2>&1 | grep -E "Assistant:|Step|<think>|<final>|\\[benchmark\\]" | head -15
        echo ""
    done
}

apply_patch() {
    local patch_name="$1"

    if [ "$patch_name" = "no-injection" ]; then
        sed -i '' 's/if did_inject:/if False and did_inject:  # DIAGNOSTIC/' mamba3_mlx/server.py
        echo "✏️  Disabled </final> injection"

    elif [ "$patch_name" = "no-transform" ]; then
        # 禁用注入后的二次 transform
        cat > /tmp/patch_transform.py << 'PYTHON_EOF'
import sys

with open(sys.argv[1], 'r') as f:
    content = f.read()

# 找到并修改采样部分
old = """        if use_inj:
            logits_1d = inj_logits
            use_inj   = False
            inj_logits = None
            tok = _sample(logits_1d, debug_label="injected_logits(no_transform)")
            log.info(f"[inj] injected tok={tok}")
        else:
            decode_step_n += 1
            logits, mamba_states, kv_caches = decode_step(
                model, prev_tok, mamba_states, kv_caches, step=decode_step_n)
            _eval_states(mamba_states, kv_caches)
            logits_1d = logits[0]

            # Apply format guard (ban mask + dynamic close-bias) before sampling
            logits_1d = mw.transform_logits(logits_1d)
            tok = _sample(logits_1d, debug_label="decode_logits(after_transform)")"""

new = """        if use_inj:
            logits_1d = inj_logits
            use_inj   = False
            inj_logits = None
            # DIAGNOSTIC: Skip transform for injected logits
            tok = _sample(logits_1d, debug_label="injected_logits(no_transform)")
            log.info(f"[inj] injected tok={tok}")
        else:
            decode_step_n += 1
            logits, mamba_states, kv_caches = decode_step(
                model, prev_tok, mamba_states, kv_caches, step=decode_step_n)
            _eval_states(mamba_states, kv_caches)
            logits_1d = logits[0]

            # Apply format guard (ban mask + dynamic close-bias) before sampling
            # DIAGNOSTIC: Only transform non-injected logits
            logits_1d = mw.transform_logits(logits_1d)
            tok = _sample(logits_1d, debug_label="decode_logits(after_transform)")"""

if old in content:
    content = content.replace(old, new)
    print("Applied no-transform patch", file=sys.stderr)

with open(sys.argv[1], 'w') as f:
    f.write(content)
PYTHON_EOF
        python3 /tmp/patch_transform.py mamba3_mlx/server.py
        echo "✏️  Disabled transform on injected logits"

    elif [ "$patch_name" = "no-mw" ]; then
        sed -i '' 's/enable_cot_mw = msg.get("enable_cot_middleware", True)/enable_cot_mw = False  # DIAGNOSTIC/' mamba3_mlx/server.py
        echo "✏️  Disabled entire CoT Middleware"
    fi
}

# 运行测试
case "$SCENARIO" in
    baseline)
        run_test "Baseline (CoT MW enabled)"
        ;;
    no-injection)
        apply_patch "no-injection"
        run_test "Disabled </final> Injection"
        ;;
    no-transform)
        apply_patch "no-transform"
        run_test "Disabled Transform on Injected Logits"
        ;;
    no-mw)
        apply_patch "no-mw"
        run_test "Disabled entire CoT Middleware"
        ;;
    all)
        run_test "Baseline (CoT MW enabled)"

        cp "$BACKUP_FILE" mamba3_mlx/server.py
        apply_patch "no-injection"
        run_test "Disabled </final> Injection"

        cp "$BACKUP_FILE" mamba3_mlx/server.py
        apply_patch "no-transform"
        run_test "Disabled Transform on Injected Logits"

        cp "$BACKUP_FILE" mamba3_mlx/server.py
        apply_patch "no-mw"
        run_test "Disabled entire CoT Middleware"
        ;;
    *)
        echo "❌ Unknown scenario: $SCENARIO"
        echo "Usage: $0 [baseline|no-injection|no-transform|no-mw|all] [prompt] [runs]"
        exit 1
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Test complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
