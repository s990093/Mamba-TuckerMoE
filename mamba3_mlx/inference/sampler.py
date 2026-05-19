# -*- coding: utf-8 -*-
"""
Sampling utilities — MLX-native, lazy-eval friendly.

Design principle (official MLX pattern):
  All ops return lazy arrays.  The caller decides when to eval.
  Only scalar conversions (.item()) force a sync — and only for
  values needed for Python control flow.

Performance notes:
  top_k: uses a lazy MLX scalar threshold — avoids the intermediate
  .item() sync that the naive approach requires, so the full decode
  graph (forward + penalties + sampling) evaluates in a single flush.
"""
import mlx.core as mx
import numpy as np
from typing import List


# ── Penalty helpers (return lazy mx.array) ────────────────────────────────────

def apply_repetition_penalty(
    logits: mx.array,
    token_ids: List[int],
    penalty: float,
    window: int,
) -> mx.array:
    """
    Repetition penalty (llama.cpp / Ollama signed convention):
      positive logits are divided by `penalty`, negative logits are multiplied.
    This prevents suppressed tokens from being artificially promoted.
    Lazy — no mx.eval inside.
    """
    if penalty == 1.0 or not token_ids:
        return logits
    recent = list(set(token_ids[-window:]))
    if not recent:
        return logits
    V = logits.shape[-1]
    pen_np = np.ones(V, dtype=np.float32)
    for tid in recent:
        if 0 <= tid < V:
            pen_np[tid] = penalty
    pen_vec = mx.array(pen_np, dtype=mx.float32)
    logits_f = logits.astype(mx.float32)
    # Positive logits → divide (suppress); negative → multiply (suppress further)
    penalized = mx.where(logits_f > 0, logits_f / pen_vec, logits_f * pen_vec)
    return penalized


def apply_presence_frequency_penalty(
    logits: mx.array,
    token_ids: List[int],
    presence_penalty: float,
    frequency_penalty: float,
    window: int,
) -> mx.array:
    """OpenAI-style presence + frequency penalty (subtract). Lazy."""
    if presence_penalty == 0.0 and frequency_penalty == 0.0:
        return logits
    recent = token_ids[-window:]
    if not recent:
        return logits
    V = logits.shape[-1]
    counts: dict[int, int] = {}
    for tid in recent:
        counts[tid] = counts.get(tid, 0) + 1
    delta_np = np.zeros(V, dtype=np.float32)
    for tid, cnt in counts.items():
        if 0 <= tid < V:
            delta_np[tid] = presence_penalty + frequency_penalty * cnt
    delta = mx.array(delta_np, dtype=mx.float32)
    return logits.astype(mx.float32) - delta    # lazy


# ── Core sampler (MLX-native, single GPU sync) ────────────────────────────────

def sample_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
) -> int:
    """
    Sample the next token.

    Follows the official MLX LLM example pattern:
      - All filtering is done as lazy graph ops
      - mx.random.categorical draws from the distribution in-graph
      - A single .item() call at the end forces the GPU sync

    top_k uses a lazy scalar threshold (no intermediate .item() sync):
      kth = mx.sort(logits)[V - top_k]  — stays in the MLX graph.

    Returns a Python int.
    """
    logits = logits.astype(mx.float32)

    # Greedy shortcut — one sync
    if temperature == 0.0:
        return int(mx.argmax(logits).item())

    logits = logits / temperature

    # ── Top-k ─────────────────────────────────────────────────────────────────
    if top_k > 0 and top_k < logits.shape[-1]:
        # Lazy scalar: no intermediate .item() — stays in the compute graph.
        # mx.sort returns ascending order; V-top_k is the index of the k-th largest.
        kth = mx.sort(logits)[logits.shape[-1] - top_k]
        logits = mx.where(logits >= kth, logits,
                          mx.full(logits.shape, float("-inf")))

    # ── Top-p (nucleus) ───────────────────────────────────────────────────────
    if top_p < 1.0:
        sorted_idx    = mx.argsort(-logits)              # descending
        sorted_logits = logits[sorted_idx]
        cum_probs     = mx.cumsum(mx.softmax(sorted_logits), axis=0)
        keep_sorted   = mx.concatenate([mx.array([True]), cum_probs[:-1] < top_p])
        keep          = keep_sorted[mx.argsort(sorted_idx)]
        logits        = mx.where(keep, logits,
                                 mx.full(logits.shape, float("-inf")))

    # ── Min-p ─────────────────────────────────────────────────────────────────
    if min_p > 0.0:
        probs     = mx.softmax(logits)
        max_prob  = mx.max(probs)
        threshold = max_prob * min_p
        logits    = mx.where(probs >= threshold, logits,
                             mx.full(logits.shape, float("-inf")))

    # ── Sample with mx.random.categorical (stays in MLX graph) ───────────────
    # mx.random.categorical expects log-probabilities
    token = mx.random.categorical(logits)       # lazy draw

    # Single .item() triggers the full graph evaluation
    return int(token.item())
