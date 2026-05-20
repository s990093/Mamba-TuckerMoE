#!/bin/bash

# MLX Mamba3 + TuckerMoE: Complete Demo Script
# Executes end-to-end MLX inference stack

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  MLX Mamba3 + TuckerMoE: Apple Silicon Inference Stack        ║"
echo "║  Complete Verification & Demo                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

VENV=.venv
PYTHON=$VENV/bin/python3
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Section 1: Environment Check
# ============================================================
echo "[STEP 1] Environment Verification"
echo "─────────────────────────────────────────────────────────────────"

if [ ! -d "$VENV" ]; then
    echo "❌ Virtual environment not found at $VENV"
    echo "   Create with: python3 -m venv .venv"
    exit 1
fi

echo "✓ Virtual environment: $VENV"
$PYTHON --version

# Check MLX
$PYTHON -c "import mlx; print(f'✓ MLX installed: {mlx.__version__}')" 2>/dev/null || {
    echo "❌ MLX not installed. Install with: pip install mlx"
    exit 1
}

echo ""

# ============================================================
# Section 2: Module Import Tests
# ============================================================
echo "[STEP 2] Module Import Verification"
echo "─────────────────────────────────────────────────────────────────"

$PYTHON -c "
import sys
sys.path.insert(0, 'mamba3_mlx')

# Test ops
from mlx_model.ops import tanh_approx, scaled_tanh, silu, RMSNorm
print('✓ mlx_model.ops')

# Test TuckerMoE
from mlx_model.tucker_moe import TuckerMoE
print('✓ mlx_model.tucker_moe')

# Test sampler
from inference.sampler import TextSampler, greedy_sample
print('✓ inference.sampler')

# Test generator
from inference.generator import AutoregressiveGenerator
print('✓ inference.generator')

# Test config
from utils.config import GenerationConfig
print('✓ utils.config')

print('')
print('All core modules loaded successfully!')
" || exit 1

echo ""

# ============================================================
# Section 3: Quick Functionality Test
# ============================================================
echo "[STEP 3] Quick Functionality Tests"
echo "─────────────────────────────────────────────────────────────────"

$PYTHON << 'PYTHON_TESTS'
import sys
sys.path.insert(0, 'mamba3_mlx')

import mlx.core as mx
from mlx_model.ops import silu, tanh_approx, RMSNorm
from inference.sampler import greedy_sample, TextSampler
from mlx_model.tucker_moe import TuckerMoE

# Test 1: Activation functions
x = mx.array([0.0, 1.0, 2.0])
y = silu(x)
print(f"✓ silu activation works: silu([0,1,2]) = {[float(v) for v in y][:2]}...")

# Test 2: RMSNorm
norm = RMSNorm(dim=64)
x = mx.random.normal((2, 64))
y = norm(x)
print(f"✓ RMSNorm works: input {x.shape} → output {y.shape}")

# Test 3: TextSampler
sampler = TextSampler(temperature=0.8, top_k=40)
logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
token = sampler(logits, [])
print(f"✓ TextSampler works: sampled token = {token} (0-4 expected)")

# Test 4: TuckerMoE
moe = TuckerMoE(dim_in=64, dim_out=128, num_experts=4, top_k=2)
x = mx.random.normal((2, 64))
out, lb_loss, z_loss = moe(x, training=False)
print(f"✓ TuckerMoE works: input {x.shape} → output {out.shape}")

print('')
print("All functionality tests passed!")
PYTHON_TESTS

echo ""

# ============================================================
# Section 4: CLI Interface Test
# ============================================================
echo "[STEP 4] CLI Interface Check"
echo "─────────────────────────────────────────────────────────────────"

echo "Available commands:"
$PYTHON mamba3_mlx/run.py 2>&1 | grep -E "(generate|benchmark)" || true

echo ""

# ============================================================
# Section 5: Summary
# ============================================================
echo "[STEP 5] Summary"
echo "─────────────────────────────────────────────────────────────────"

cat << 'EOF'
✓ ALL VERIFICATION TESTS PASSED

MLX Mamba3 + TuckerMoE inference stack is ready to use!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEXT STEPS:

1️⃣ Convert your PyTorch checkpoint to MLX format:
   python mamba3_mlx/mlx_model/convert_weights.py \
     --pt_path checkpoints/latest_sft_cot_model.pt \
     --output mamba3_mlx/model.npz

2️⃣ Run text generation:
   python mamba3_mlx/run.py generate \
     --model_path mamba3_mlx/model.npz \
     --prompt "Explain quantum computing" \
     --max_tokens 256 \
     --temp 0.8 \
     --top_k 40 \
     --verbose

3️⃣ Run performance benchmark:
   python mamba3_mlx/run.py benchmark \
     --model_path mamba3_mlx/model.npz \
     --prompt "The future of AI" \
     --num_generate 256 \
     --verbose

4️⃣ Run unit tests:
   cd mamba3_mlx
   python -m pytest tests/ -v

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FEATURES IMPLEMENTED:

✓ Mamba3 blocks with parallel scan
✓ Tucker-decomposed Mixture-of-Experts (TuckerMoE)
✓ Transformer blocks with Grouped-Query Attention
✓ Prefill/Decode separation with state management
✓ Advanced sampling (7 strategies)
✓ Speculative decoding (draft + verify)
✓ Temperature, Top-k, Top-p, Min-p filtering
✓ Repetition & Frequency/Presence penalties
✓ Weight conversion (PyTorch → MLX)
✓ CLI with 20+ parameters
✓ Full documentation & guides

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DOCUMENTATION:

  README.md              - Quick start & architecture overview
  IMPLEMENTATION_GUIDE.md - Design decisions & debugging tips
  COMPLETION_SUMMARY.md  - Full implementation checklist

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

echo ""
