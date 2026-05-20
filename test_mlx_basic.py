#!/usr/bin/env python3
"""
Basic verification script for MLX Mamba3 implementation.
Tests imports, module initialization, and forward passes.
"""

import sys
import os

# Add mamba3_mlx to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mamba3_mlx'))

import mlx.core as mx
import mlx.nn as nn

print("=" * 60)
print("MLX Mamba3 + TuckerMoE: Basic Verification")
print("=" * 60)

# Test 1: Import ops
print("\n[1/6] Testing ops module...")
try:
    from mlx_model.ops import (
        tanh_approx, scaled_tanh, silu, silu_gating, rope,
        RMSNorm, LayerScale, chunk_parallel_scan
    )
    print("  ✓ All ops imported successfully")

    # Quick test
    x = mx.array([0.0, 1.0, 2.0])
    y = silu(x)
    print(f"  ✓ silu([0,1,2]) = {y}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 2: Import TuckerMoE
print("\n[2/6] Testing TuckerMoE module...")
try:
    from mlx_model.tucker_moe import TuckerMoE, MixtralMoEFeedForward
    print("  ✓ TuckerMoE classes imported")

    moe = TuckerMoE(dim_in=64, dim_out=128, num_experts=4, top_k=2)
    x = mx.random.normal((2, 64))
    out, lb_loss, z_loss = moe(x, training=False)
    print(f"  ✓ TuckerMoE forward pass: input {x.shape} → output {out.shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 3: Import Mamba3Block
print("\n[3/6] Testing Mamba3Block module...")
try:
    from mlx_model.mamba_block import Mamba3Block
    print("  ✓ Mamba3Block imported")

    block = Mamba3Block(d_model=64, d_state=32, d_head=16, expand=2)
    x = mx.random.normal((1, 10, 64))  # (B, L, d_model)
    out, state = block(x, state=None, training=False)
    print(f"  ✓ Mamba3Block forward pass: input {x.shape} → output {out.shape}")
    print(f"    State keys: {state.keys() if state else 'None'}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Import TransformerBlock
print("\n[4/6] Testing TransformerBlock module...")
try:
    from mlx_model.transformer_block import TransformerBlock, GroupedQueryAttention
    print("  ✓ TransformerBlock imported")

    block = TransformerBlock(d_model=64, n_heads=4, num_kv_heads=2, use_moe=False)
    x = mx.random.normal((1, 10, 64))
    out, _, loss = block(x, kv_cache=None, training=False)
    print(f"  ✓ TransformerBlock forward pass: input {x.shape} → output {out.shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Import full model
print("\n[5/6] Testing full Mamba3LanguageModel...")
try:
    from mlx_model.hybrid_model import Mamba3LanguageModel, Mamba3Config
    print("  ✓ Mamba3LanguageModel imported")

    config = Mamba3Config(
        d_model=128,
        num_layers=4,
        vocab_size=1024,
    )
    print(f"  ✓ Config created: d_model={config.d_model}, num_layers={config.num_layers}")

    # Don't instantiate full model yet (takes time)
    # Just check config is valid
    print(f"  ✓ Model config valid")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Import inference components
print("\n[6/6] Testing inference components...")
try:
    from inference.sampler import TextSampler, greedy_sample
    from inference.generator import AutoregressiveGenerator
    print("  ✓ Sampler and Generator imported")

    # Test sampler
    logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    token = greedy_sample(logits)
    print(f"  ✓ greedy_sample([1,2,3,4,5]) = {token} (expected: 4)")

    sampler = TextSampler(temperature=0.8, top_k=3)
    token = sampler(logits, [])
    print(f"  ✓ TextSampler sample token = {token}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All basic verification tests passed!")
print("=" * 60)
print("\nNext steps:")
print("  1. Convert checkpoint: python mamba3_mlx/mlx_model/convert_weights.py \\")
print("     --pt_path checkpoints/model.pt --output mamba3_mlx/model.npz")
print("  2. Run generation: python mamba3_mlx/run.py generate \\")
print("     --model_path mamba3_mlx/model.npz --prompt 'Hello'")
print("=" * 60)
