"""
Unit tests for sampling strategies.
Tests: penalties, filtering, and sampler classes.
"""

import mlx.core as mx
import numpy as np
from inference.sampler import (
    apply_repetition_penalty,
    apply_frequency_presence_penalty,
    apply_top_k_filtering,
    apply_nucleus_filtering,
    apply_min_p_filtering,
    TextSampler,
    greedy_sample,
)


def test_repetition_penalty():
    """Test repetition penalty application."""
    logits = mx.array([1.0, 2.0, 3.0, 4.0])
    generated_ids = [0, 1, 0]  # Tokens 0 and 1 are repeated
    penalty = 2.0

    penalized = apply_repetition_penalty(logits, generated_ids, penalty=penalty)

    # Penalized tokens should have lower logits
    assert penalized[0].item() < logits[0].item(), "Repeated token should be penalized"
    assert penalized[1].item() < logits[1].item(), "Repeated token should be penalized"
    print("✓ test_repetition_penalty passed")


def test_frequency_presence_penalty():
    """Test frequency and presence penalties."""
    logits = mx.array([1.0, 2.0, 3.0, 4.0])
    generated_ids = [0, 0, 1, 2]  # Token 0 appears 2x

    penalized = apply_frequency_presence_penalty(
        logits, generated_ids, pres_pen=0.5, freq_pen=0.1
    )

    # Token 0 should be penalized more than token 3
    assert penalized[0].item() < logits[0].item()
    assert penalized[3].item() == logits[3].item()  # Unseen token unchanged
    print("✓ test_frequency_presence_penalty passed")


def test_top_k_filtering():
    """Test top-k filtering."""
    logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    top_k = 2

    filtered = apply_top_k_filtering(logits, top_k=top_k)

    # Only top 2 should remain (others → -inf)
    inf_count = mx.sum(mx.isinf(filtered)).item()
    assert inf_count == len(logits) - top_k, f"Expected {len(logits) - top_k} -inf values"
    print("✓ test_top_k_filtering passed")


def test_nucleus_filtering():
    """Test nucleus (top-p) filtering."""
    logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    top_p = 0.9

    filtered = apply_nucleus_filtering(logits, top_p=top_p)

    # Some low-probability tokens should be filtered
    inf_count = mx.sum(mx.isinf(filtered)).item()
    assert inf_count > 0, "Nucleus filtering should remove low-probability tokens"
    print("✓ test_nucleus_filtering passed")


def test_min_p_filtering():
    """Test min-p filtering."""
    logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    min_p = 0.1  # Keep tokens with prob >= 0.1 * max_prob

    filtered = apply_min_p_filtering(logits, min_p=min_p)

    # Some low-probability tokens should be filtered
    inf_count = mx.sum(mx.isinf(filtered)).item()
    assert inf_count > 0, "Min-p filtering should remove low-probability tokens"
    print("✓ test_min_p_filtering passed")


def test_text_sampler():
    """Test TextSampler class."""
    sampler = TextSampler(temperature=0.8, top_k=40, top_p=0.9, min_p=0.05)

    logits = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
    generated_ids = [0, 1, 2]

    token = sampler(logits, generated_ids)

    # Should return a valid token ID
    assert 0 <= token < len(logits), f"Sampled token {token} out of range"
    print("✓ test_text_sampler passed")


def test_greedy_sample():
    """Test greedy sampling."""
    logits = mx.array([1.0, 2.0, 5.0, 3.0, 4.0])

    token = greedy_sample(logits)

    # Should return token with highest logit (index 2)
    assert token == 2, f"Greedy sample should return index 2, got {token}"
    print("✓ test_greedy_sample passed")


if __name__ == "__main__":
    test_repetition_penalty()
    test_frequency_presence_penalty()
    test_top_k_filtering()
    test_nucleus_filtering()
    test_min_p_filtering()
    test_text_sampler()
    test_greedy_sample()
    print("\n✓ All sampler tests passed!")
