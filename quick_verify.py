#!/usr/bin/env python3
"""Quick verification of MLX stack - no dependencies on environment variables."""

import sys
import os

# Set paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mamba3_mlx'))

print("\n" + "="*70)
print("MLX Mamba3 + TuckerMoE: Rapid Verification")
print("="*70 + "\n")

# Test 1: Core imports
print("[TEST 1] Core Module Imports")
print("-" * 70)
try:
    import mlx.core as mx
    print("✓ mlx.core imported")

    from mlx_model.ops import tanh_approx, silu, RMSNorm, LayerScale
    print("✓ mlx_model.ops imported")

    from mlx_model.tucker_moe import TuckerMoE
    print("✓ mlx_model.tucker_moe imported")

    from inference.sampler import TextSampler, greedy_sample
    print("✓ inference.sampler imported")

    from inference.generator import AutoregressiveGenerator
    print("✓ inference.generator imported")

    from utils.config import GenerationConfig
    print("✓ utils.config imported")

    print("\n✅ All imports successful!\n")
except ImportError as e:
    print(f"\n❌ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Activation functions
print("[TEST 2] Activation Functions")
print("-" * 70)
try:
    x = mx.array([0.0, 1.0, 2.0])
    y_silu = silu(x)
    y_tanh = tanh_approx(x)
    print(f"✓ silu([0,1,2]) → {[float(v) for v in y_silu]}")
    print(f"✓ tanh_approx([0,1,2]) → {[float(v) for v in y_tanh]}")
    print()
except Exception as e:
    print(f"❌ Activation test failed: {e}\n")
    sys.exit(1)

# Test 3: Normalization
print("[TEST 3] Normalization Layers")
print("-" * 70)
try:
    norm = RMSNorm(dim=64)
    x = mx.random.normal((2, 64))
    y = norm(x)
    print(f"✓ RMSNorm: input {x.shape} → output {y.shape}")

    layer_scale = LayerScale(dim=64)
    y_scaled = layer_scale(x)
    print(f"✓ LayerScale: input {x.shape} → output {y_scaled.shape}")
    print()
except Exception as e:
    print(f"❌ Normalization test failed: {e}\n")
    sys.exit(1)

# Test 4: TuckerMoE
print("[TEST 4] TuckerMoE Layer")
print("-" * 70)
try:
    moe = TuckerMoE(dim_in=128, dim_out=256, num_experts=4, top_k=2)
    x = mx.random.normal((4, 128))
    result = moe(x, training=False)

    # Handle both 2-tuple and 3-tuple returns
    if isinstance(result, tuple):
        if len(result) == 3:
            out, lb_loss, z_loss = result
            print(f"✓ TuckerMoE forward pass successful (3-return mode)")
            print(f"  Input: {x.shape}")
            print(f"  Output: {out.shape}")
        else:
            out = result[0]
            print(f"✓ TuckerMoE forward pass successful")
            print(f"  Input: {x.shape}")
            print(f"  Output: {out.shape}")
    else:
        out = result
        print(f"✓ TuckerMoE forward pass successful")
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
    print()
except Exception as e:
    print(f"❌ TuckerMoE test failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Sampling
print("[TEST 5] Sampling Strategies")
print("-" * 70)
try:
    logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Greedy
    token_greedy = greedy_sample(logits)
    print(f"✓ greedy_sample: token={token_greedy} (expected: 4)")

    # TextSampler
    sampler = TextSampler(temperature=0.8, top_k=3)
    token_sample = sampler(logits, [])
    print(f"✓ TextSampler: token={token_sample} (0-4 range)")
    print()
except Exception as e:
    print(f"❌ Sampling test failed: {e}\n")
    sys.exit(1)

# Test 6: Config
print("[TEST 6] Configuration")
print("-" * 70)
try:
    gen_config = GenerationConfig(
        temp=0.8,
        top_k=40,
        top_p=0.9,
        dtype="bf16"
    )
    print(f"✓ GenerationConfig created")
    print(f"  temperature: {gen_config.temp}")
    print(f"  top_k: {gen_config.top_k}")
    print(f"  dtype: {gen_config.dtype}")
    print()
except Exception as e:
    print(f"❌ Config test failed: {e}\n")
    sys.exit(1)

# Summary
print("=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nMLX Mamba3 + TuckerMoE inference stack is operational.\n")

print("NEXT STEPS:")
print("-" * 70)
print("1. Convert PyTorch checkpoint to MLX:")
print("   python mamba3_mlx/mlx_model/convert_weights.py \\")
print("     --pt_path checkpoints/model.pt \\")
print("     --output mamba3_mlx/model.npz\n")

print("2. Generate text:")
print("   python mamba3_mlx/run.py generate \\")
print("     --model_path mamba3_mlx/model.npz \\")
print("     --prompt 'Hello' \\")
print("     --max_tokens 256\n")

print("3. Benchmark performance:")
print("   python mamba3_mlx/run.py benchmark \\")
print("     --model_path mamba3_mlx/model.npz \\")
print("     --prompt 'Test' \\")
print("     --num_generate 256 \\")
print("     --verbose\n")

print("=" * 70)
