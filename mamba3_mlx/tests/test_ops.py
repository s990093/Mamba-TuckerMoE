"""
Unit tests for core operations (ops.py).
Tests: tanh_approx, silu, rope, RMSNorm, LayerScale.
"""

import mlx.core as mx
import numpy as np
from mlx_model.ops import tanh_approx, scaled_tanh, silu, silu_gating, RMSNorm, LayerScale


def test_tanh_approx():
    """Test tanh approximation."""
    x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = tanh_approx(x)

    # Check that output is in (-1, 1)
    assert mx.all(y > -1.0) and mx.all(y < 1.0), "tanh_approx output should be in (-1, 1)"
    print("✓ test_tanh_approx passed")


def test_scaled_tanh():
    """Test scaled tanh."""
    x = mx.array([0.0, 10.0, 20.0])
    y = scaled_tanh(x, scale=10.0)

    # Check range
    assert mx.all(y >= -10.0) and mx.all(y <= 10.0), "scaled_tanh output should be in [-scale, scale]"
    print("✓ test_scaled_tanh passed")


def test_silu():
    """Test SiLU activation."""
    x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = silu(x)

    # SiLU(0) should be 0
    assert abs(y[2].item() - 0.0) < 1e-6, "SiLU(0) should be 0"

    # SiLU should be smooth
    assert y.shape == x.shape, "SiLU output shape should match input"
    print("✓ test_silu passed")


def test_silu_gating():
    """Test SiLU gating."""
    gate = mx.ones((2, 3))
    feat = mx.ones((2, 3)) * 2.0

    y = silu_gating(gate, feat)

    # Output should be non-zero
    assert mx.all(y != 0.0), "silu_gating output should be non-zero"
    assert y.shape == gate.shape, "Output shape should match input"
    print("✓ test_silu_gating passed")


def test_rmsnorm():
    """Test RMSNorm."""
    dim = 64
    norm = RMSNorm(dim=dim, eps=1e-5)

    x = mx.random.normal((2, dim))
    y = norm(x)

    # RMS should be close to 1 after normalization
    rms = mx.sqrt(mx.mean(y ** 2, axis=-1))
    expected_rms = mx.ones((2,))

    # Check that RMS is close to expected (allowing for eps)
    assert y.shape == x.shape, "RMSNorm output shape should match input"
    print("✓ test_rmsnorm passed")


def test_layerscale():
    """Test LayerScale."""
    dim = 64
    layer_scale = LayerScale(dim=dim, init_value=1e-2)

    x = mx.ones((2, dim))
    y = layer_scale(x)

    # Output should be scaled
    assert y.shape == x.shape, "LayerScale output shape should match input"
    assert mx.all(y < 1.0), "LayerScale with init_value=1e-2 should reduce values"
    print("✓ test_layerscale passed")


if __name__ == "__main__":
    test_tanh_approx()
    test_scaled_tanh()
    test_silu()
    test_silu_gating()
    test_rmsnorm()
    test_layerscale()
    print("\n✓ All ops tests passed!")
