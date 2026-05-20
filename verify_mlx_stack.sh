#!/bin/bash

# MLX Mamba3 + TuckerMoE: Verification Script
# Run comprehensive tests on the MLX inference stack

echo "=========================================="
echo "MLX Mamba3 Stack Verification"
echo "=========================================="
echo ""

cd "$(dirname "$0")"
VENV=.venv

# Check Python
echo "[1/5] Checking Python environment..."
if [ ! -d "$VENV" ]; then
    echo "  ✗ Virtual environment not found at $VENV"
    exit 1
fi
echo "  ✓ Python environment ready"

# Run unit tests
echo ""
echo "[2/5] Running unit tests..."
$VENV/bin/python3 mamba3_mlx/tests/test_ops.py 2>&1 | tail -5

# Run sampler tests
echo ""
echo "[3/5] Running sampler tests..."
$VENV/bin/python3 mamba3_mlx/tests/test_sampler.py 2>&1 | tail -5

# Check imports
echo ""
echo "[4/5] Verifying core imports..."
$VENV/bin/python3 -c "
import sys
sys.path.insert(0, 'mamba3_mlx')
from mlx_model.ops import *
from mlx_model.tucker_moe import *
from inference.sampler import *
print('  ✓ All core modules imported successfully')
" 2>&1

# Test CLI
echo ""
echo "[5/5] Testing CLI interface..."
$VENV/bin/python3 mamba3_mlx/run.py --help 2>&1 | head -3

echo ""
echo "=========================================="
echo "✓ Verification Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Load checkpoint:"
echo "     python mamba3_mlx/mlx_model/convert_weights.py \\"
echo "       --pt_path checkpoints/model.pt \\"
echo "       --output mamba3_mlx/model.npz"
echo ""
echo "  2. Generate text:"
echo "     python mamba3_mlx/run.py generate \\"
echo "       --model_path mamba3_mlx/model.npz \\"
echo "       --prompt 'Hello world' \\"
echo "       --max_tokens 256"
echo ""
echo "  3. Run benchmark:"
echo "     python mamba3_mlx/run.py benchmark \\"
echo "       --model_path mamba3_mlx/model.npz \\"
echo "       --prompt 'Test' \\"
echo "       --num_generate 256"
echo "=========================================="
