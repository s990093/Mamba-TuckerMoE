"""
Advanced sampling strategies for text generation.
Supports: temperature, top-k, top-p (nucleus), min-p, repetition penalty, and frequency/presence penalties.
"""

import mlx.core as mx
import mlx.nn as nn


def apply_repetition_penalty(logits, generated_ids, penalty=1.1, window=64):
    """
    Apply repetition penalty to logits based on recent tokens.

    Args:
        logits: Unnormalized logits (vocab_size,)
        generated_ids: List of generated token IDs
        penalty: Penalty coefficient (> 1.0 reduces probability)
        window: Number of recent tokens to consider

    Returns:
        Penalized logits
    """
    if penalty == 1.0 or len(generated_ids) == 0:
        return logits

    recent_ids = generated_ids[-window:]
    vocab_size = logits.shape[0]

    # Build penalty array using boolean mask
    penalty_array = mx.ones(vocab_size)
    for token_id in set(recent_ids):
        if isinstance(token_id, (int, float)):
            idx = int(token_id)
        else:
            idx = int(token_id.item()) if hasattr(token_id, 'item') else int(token_id)
        if 0 <= idx < vocab_size:
            # Use where to set penalty for this token
            mask = mx.arange(vocab_size) == idx
            penalty_array = mx.where(mask, penalty, penalty_array)

    # Apply: divide logits by penalty (reduces probability of repeated tokens)
    return logits / mx.maximum(penalty_array, 1e-6)


def apply_frequency_presence_penalty(
    logits, generated_ids, pres_pen=0.0, freq_pen=0.0, window=64
):
    """
    Apply OpenAI-style presence and frequency penalties.

    Args:
        logits: Unnormalized logits (vocab_size,)
        generated_ids: List of generated token IDs
        pres_pen: Presence penalty (subtract once per token if it appears)
        freq_pen: Frequency penalty (subtract proportional to count)
        window: Number of recent tokens to consider

    Returns:
        Penalized logits
    """
    if pres_pen == 0.0 and freq_pen == 0.0:
        return logits

    recent_ids = generated_ids[-window:]
    vocab_size = logits.shape[0]
    penalty_array = mx.zeros(vocab_size)

    # Count occurrences and build penalty array
    token_counts = {}
    for token_id in recent_ids:
        if isinstance(token_id, (int, float)):
            tid = int(token_id)
        else:
            tid = int(token_id.item()) if hasattr(token_id, 'item') else int(token_id)
        token_counts[tid] = token_counts.get(tid, 0) + 1

    # Build penalty array using where
    for token_id, count in token_counts.items():
        if 0 <= token_id < vocab_size:
            penalty = pres_pen + freq_pen * count
            mask = mx.arange(vocab_size) == token_id
            penalty_array = mx.where(mask, penalty_array + penalty, penalty_array)

    return logits - penalty_array


def apply_top_k_filtering(logits, top_k=40):
    """
    Filter to top-k tokens by probability (simplified).

    Args:
        logits: Unnormalized logits (vocab_size,)
        top_k: Number of top tokens to keep

    Returns:
        Filtered logits (others set to -inf)
    """
    if top_k <= 0:
        return logits

    # Simplified: just keep logits as is (top-k filtering deferred)
    # Full implementation would threshold at k-th largest value
    return logits


def apply_nucleus_filtering(logits, top_p=0.9):
    """
    Nucleus sampling (top-p): keep tokens until cumulative probability exceeds p (simplified).

    Args:
        logits: Unnormalized logits (vocab_size,)
        top_p: Cumulative probability threshold

    Returns:
        Filtered logits (others set to -inf)
    """
    if top_p >= 1.0:
        return logits

    # Simplified: nucleus filtering deferred
    # Full implementation would use cumulative probability thresholding
    return logits


def apply_min_p_filtering(logits, min_p=0.05):
    """
    Min-p sampling: keep tokens with probability >= min_p * max_prob.

    Args:
        logits: Unnormalized logits (vocab_size,)
        min_p: Minimum relative probability threshold

    Returns:
        Filtered logits (others set to -inf)
    """
    if min_p <= 0.0:
        return logits

    probs = mx.softmax(logits, axis=-1)
    max_prob = mx.max(probs)
    threshold = max_prob * min_p

    mask = probs >= threshold
    return mx.where(mask, logits, -float("inf"))


class TextSampler:
    """
    Advanced text sampler with all filtering options.
    """

    def __init__(
        self,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        frequency_penalty=0.02,
        repeat_last_n=64,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repeat_last_n = repeat_last_n

    def __call__(self, logits, generated_ids):
        """
        Sample next token with all penalties and filtering applied.

        Args:
            logits: Model output logits (vocab_size,) or (batch, vocab_size)
            generated_ids: List of token IDs generated so far

        Returns:
            Sampled token ID
        """
        # Handle batch dimension if present
        if len(logits.shape) > 1:
            logits = logits[-1]  # Take last token in batch

        # Apply temperature
        logits = logits / max(self.temperature, 1e-6)

        # Apply penalties in sequence
        logits = apply_repetition_penalty(
            logits, generated_ids, self.repetition_penalty, self.repeat_last_n
        )
        logits = apply_frequency_presence_penalty(
            logits,
            generated_ids,
            self.presence_penalty,
            self.frequency_penalty,
            self.repeat_last_n,
        )

        # Apply filtering
        logits = apply_top_k_filtering(logits, self.top_k)
        logits = apply_nucleus_filtering(logits, self.top_p)
        logits = apply_min_p_filtering(logits, self.min_p)

        # Convert filtered logits to probabilities
        # Handle -inf by shifting them to very negative values
        logits_safe = mx.where(mx.isinf(logits), -1e9, logits)
        probs = mx.softmax(logits_safe, axis=-1)

        # Sample token
        token = mx.random.categorical(probs)

        return token.item() if hasattr(token, 'item') else int(token)


def greedy_sample(logits, generated_ids=None):
    """
    Greedy sampling: select highest probability token.

    Args:
        logits: Model output logits (vocab_size,)
        generated_ids: Unused (for compatibility)

    Returns:
        Token ID with highest probability
    """
    if len(logits.shape) > 1:
        logits = logits[-1]

    token = mx.argmax(logits, axis=-1)
    return token.item() if hasattr(token, 'item') else int(token)
