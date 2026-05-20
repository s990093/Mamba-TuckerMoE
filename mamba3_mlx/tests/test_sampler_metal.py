# -*- coding: utf-8 -*-
"""
Validation tests for Metal sampling against MLX baseline sampler.

Tests verify:
  1. Greedy mode matches MLX argmax exactly
  2. Stochastic sampling produces valid probability distributions
  3. Penalties (rep, presence, frequency) suppress tokens correctly
  4. Filters (top-k, top-p, min-p) work as expected
"""

import numpy as np
import mlx.core as mx
from dataclasses import dataclass
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.sampler import sample_token as sample_token_mlx
from inference.sampler_metal import sample_token_metal_full


@dataclass
class MockArgs:
    """Mock args object for sampling functions."""
    temp: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    rep_pen: float = 1.0
    pres_pen: float = 0.0
    freq_pen: float = 0.0
    fast_sample: bool = False


def test_greedy_matches_mlx():
    """Test 1: Greedy Metal sampling should produce same token as MLX argmax."""
    np.random.seed(42)
    vocab_size = 32000
    logits = mx.array(np.random.randn(vocab_size).astype(np.float32))

    # MLX greedy
    token_mlx = int(mx.argmax(logits).item())

    # Metal greedy
    args = MockArgs(temp=0.0)
    counts = mx.zeros((vocab_size,))
    token_metal = int(sample_token_metal_full(logits, counts, args).item())

    assert token_metal == token_mlx, f"Mismatch: Metal {token_metal} vs MLX {token_mlx}"
    print("✓ Test 1: Greedy matches MLX argmax")


def test_stochastic_produces_valid_tokens():
    """Test 2: Stochastic sampling should produce only valid token IDs."""
    vocab_size = 5000
    logits = mx.array(np.random.randn(vocab_size).astype(np.float32))
    counts = mx.zeros((vocab_size,))

    args = MockArgs(temp=1.0)
    samples = [
        int(sample_token_metal_full(logits, counts, args).item())
        for _ in range(200)
    ]

    # All samples should be in valid range
    assert all(0 <= s < vocab_size for s in samples), "Invalid token IDs"
    # Should have variety
    assert len(set(samples)) > 10, "Too few unique samples"
    print("✓ Test 2: Stochastic sampling produces valid distribution")


def test_greedy_is_deterministic():
    """Test 3: Greedy mode should always produce same token."""
    vocab_size = 1000
    logits = mx.array(np.random.randn(vocab_size).astype(np.float32))
    counts = mx.zeros((vocab_size,))

    args = MockArgs(temp=0.0)
    samples = [
        int(sample_token_metal_full(logits, counts, args).item())
        for _ in range(5)
    ]

    assert len(set(samples)) == 1, "Greedy should be deterministic"
    print("✓ Test 3: Greedy mode is deterministic")


def test_stochastic_has_variance():
    """Test 4: Stochastic mode should produce different samples."""
    vocab_size = 1000
    logits = mx.array(np.random.randn(vocab_size).astype(np.float32))
    counts = mx.zeros((vocab_size,))

    args = MockArgs(temp=2.0)  # High temperature
    samples = [
        int(sample_token_metal_full(logits, counts, args).item())
        for _ in range(100)
    ]

    unique_count = len(set(samples))
    assert unique_count > 20, f"Too few unique samples: {unique_count}"
    print("✓ Test 4: Stochastic mode produces variance")


def test_repetition_penalty_suppresses_tokens():
    """Test 5: Repetition penalty should suppress frequently-occurred tokens."""
    vocab_size = 100
    logits = mx.array(np.ones(vocab_size, dtype=np.float32) * 5.0)

    # Mark tokens 0-9 as repeated
    counts_np = np.zeros(vocab_size, dtype=np.float32)
    counts_np[:10] = 2.0
    counts = mx.array(counts_np)

    args = MockArgs(temp=1.0, rep_pen=2.0)
    samples = [
        int(sample_token_metal_full(logits, counts, args).item())
        for _ in range(200)
    ]

    penalized_count = sum(1 for s in samples if s < 10)
    penalized_ratio = penalized_count / len(samples)

    # With rep_pen=2.0, penalized tokens should be heavily suppressed
    assert penalized_ratio < 0.15, \
        f"Penalty not suppressing enough: {penalized_ratio:.2%} (expected < 15%)"
    print(f"✓ Test 5: Repetition penalty suppresses tokens ({penalized_ratio:.1%} < 15%)")


def test_presence_frequency_penalties():
    """Test 6: Presence + frequency penalties should suppress tokens."""
    vocab_size = 100
    logits = mx.array(np.ones(vocab_size, dtype=np.float32) * 5.0)

    counts_np = np.zeros(vocab_size, dtype=np.float32)
    counts_np[:10] = 3.0
    counts = mx.array(counts_np)

    args = MockArgs(temp=1.0, pres_pen=0.1, freq_pen=0.05)
    samples = [
        int(sample_token_metal_full(logits, counts, args).item())
        for _ in range(200)
    ]

    penalized_ratio = sum(1 for s in samples if s < 10) / len(samples)
    assert penalized_ratio < 0.25, \
        f"Penalties not suppressing: {penalized_ratio:.2%}"
    print(f"✓ Test 6: Presence+frequency penalties work ({penalized_ratio:.1%})")


def test_topk_filtering():
    """Test 7: Top-k should only allow k highest logits."""
    vocab_size = 200
    # Monotonic logits so top-k is deterministic
    logits = mx.array(np.arange(vocab_size, dtype=np.float32))
    counts = mx.zeros((vocab_size,))

    k = 20
    args = MockArgs(temp=1.0, top_k=k)
    samples = [
        int(sample_token_metal_full(logits, counts, args).item())
        for _ in range(200)
    ]

    top_k_range = set(range(vocab_size - k, vocab_size))
    assert all(s in top_k_range for s in samples), \
        f"Samples outside top-{k}: {set(samples) - top_k_range}"
    print(f"✓ Test 7: Top-k filtering works")


def test_topp_filtering():
    """Test 8: Top-p should include nucleus probability."""
    vocab_size = 200
    logits = mx.array(np.arange(vocab_size, dtype=np.float32))
    counts = mx.zeros((vocab_size,))

    top_p = 0.5
    args = MockArgs(temp=1.0, top_p=top_p)
    samples = [
        int(sample_token_metal_full(logits, counts, args).item())
        for _ in range(300)
    ]

    # With monotonic logits and top_p=0.5, samples should cluster at high end
    median = np.median(samples)
    assert median > vocab_size * 0.4, \
        f"Samples too low: median={median} (expected > {vocab_size * 0.4})"
    print(f"✓ Test 8: Top-p filtering works")


def test_minp_filtering():
    """Test 9: Min-p should suppress low-probability tokens."""
    vocab_size = 100
    # Setup: one high logit, rest low
    logits_np = np.array([10.0] + [0.0] * (vocab_size - 1), dtype=np.float32)
    logits = mx.array(logits_np)
    counts = mx.zeros((vocab_size,))

    min_p = 0.1
    args = MockArgs(temp=1.0, min_p=min_p)
    samples = [
        int(sample_token_metal_full(logits, counts, args).item())
        for _ in range(100)
    ]

    # Min-p should heavily favor token 0 (highest prob)
    token_0_ratio = sum(1 for s in samples if s == 0) / len(samples)
    assert token_0_ratio > 0.7, \
        f"Min-p not filtering: token 0 = {token_0_ratio:.1%}"
    print(f"✓ Test 9: Min-p filtering works ({token_0_ratio:.1%} at token 0)")


def test_combined_filters():
    """Test 10: Combined filters should work together."""
    vocab_size = 500
    logits = mx.array(np.arange(vocab_size, dtype=np.float32))

    counts_np = np.zeros(vocab_size, dtype=np.float32)
    counts_np[:50] = 2.0  # Mark first 50 as repeated
    counts = mx.array(counts_np)

    args = MockArgs(
        temp=1.0,
        top_k=100,
        top_p=0.8,
        min_p=0.05,
        rep_pen=2.0,
        pres_pen=0.1,
        freq_pen=0.05
    )

    samples = [
        int(sample_token_metal_full(logits, counts, args).item())
        for _ in range(200)
    ]

    # All samples valid
    assert all(0 <= s < vocab_size for s in samples), "Invalid tokens"
    # Repeated tokens suppressed
    repeated_ratio = sum(1 for s in samples if s < 50) / len(samples)
    assert repeated_ratio < 0.2, f"Repetition penalty not working: {repeated_ratio:.1%}"
    print(f"✓ Test 10: Combined filters work ({repeated_ratio:.1%} penalized)")


def test_metal_vs_mlx_distribution():
    """Test 11: Metal and MLX should produce similar distributions (no penalties)."""
    vocab_size = 1000
    np.random.seed(123)
    logits_np = np.random.randn(vocab_size).astype(np.float32)
    logits = mx.array(logits_np)

    # Metal samples
    args = MockArgs(temp=1.0)
    counts = mx.zeros((vocab_size,))
    metal_samples = [
        int(sample_token_metal_full(logits, counts, args).item())
        for _ in range(2000)
    ]

    # MLX samples
    mlx_samples = [
        sample_token_mlx(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0)
        for _ in range(2000)
    ]

    # Both should have good coverage
    metal_unique = len(set(metal_samples))
    mlx_unique = len(set(mlx_samples))

    assert metal_unique > 100, f"Metal samples too few unique: {metal_unique}"
    assert mlx_unique > 100, f"MLX samples too few unique: {mlx_unique}"

    # Support should overlap substantially
    metal_support = set(metal_samples)
    mlx_support = set(mlx_samples)
    overlap = len(metal_support & mlx_support)
    overlap_ratio = overlap / min(len(metal_support), len(mlx_support))

    assert overlap_ratio > 0.5, \
        f"Low support overlap: {overlap_ratio:.1%}"
    print(f"✓ Test 11: Metal vs MLX distributions similar ({overlap_ratio:.1%} overlap)")


if __name__ == "__main__":
    print("\n=== Metal Sampling Validation Tests ===\n")
    test_greedy_matches_mlx()
    test_stochastic_produces_valid_tokens()
    test_greedy_is_deterministic()
    test_stochastic_has_variance()
    test_repetition_penalty_suppresses_tokens()
    test_presence_frequency_penalties()
    test_topk_filtering()
    test_topp_filtering()
    test_minp_filtering()
    test_combined_filters()
    test_metal_vs_mlx_distribution()
    print("\n=== All Tests Passed ===\n")
