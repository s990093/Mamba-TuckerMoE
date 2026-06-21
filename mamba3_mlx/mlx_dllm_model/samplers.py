"""dLLM samplers / schedules — MLX ports of the DiffusionGemma optimizations,
adapted to **absorbing ([MASK]) diffusion** (canvas starts all-[MASK] and
positions are filled in; non-committed positions stay [MASK] rather than being
re-noised to random tokens as DiffusionGemma does).

Ported pieces (transformers/models/diffusion_gemma/generation_diffusion_gemma.py):
  * EntropyBoundSampler.accept_canvas  → ``entropy_bound_select``
  * LinearTemperatureScheduleLogitsProcessor → ``linear_temperature``
  * StableAndConfidentStoppingCriteria → ``StableConfidentStop``

The headline idea is the entropy bound: instead of committing a fixed cosine
count per step, commit the lowest-entropy (most-confident) still-masked
positions while their joint mutual-information upper bound stays ≤
``entropy_bound`` — i.e. only tokens that are ~independent given the context,
so the number committed adapts to the model's confidence each step.
(https://arxiv.org/pdf/2505.24857)
"""

from __future__ import annotations

import mlx.core as mx

_BIG = 1e30


def token_entropy(logits):
    """Shannon entropy per row of a (..., V) logits tensor (float32, nats).

    nan-safe: banned classes carry ``-inf`` logits → p == 0, and 0·log0 is
    taken as 0 (instead of the 0·(-inf) = nan a naive product would give).
    """
    z = logits.astype(mx.float32)
    z = z - mx.max(z, axis=-1, keepdims=True)
    e = mx.exp(z)
    s = mx.sum(e, axis=-1, keepdims=True)
    logp = z - mx.log(s)
    p = e / s
    plogp = mx.where(p > 0, p * logp, mx.array(0.0, dtype=p.dtype))
    return -mx.sum(plogp, axis=-1)


def linear_temperature(steps_remaining: int, max_steps: int,
                       t_min: float, t_max: float) -> float:
    """DiffusionGemma's linear schedule.

    ``steps_remaining`` counts DOWN (N..1): early steps (many remaining) get
    ``t_max`` (exploration), the final step gets ≈``t_min`` (sharp commit).
    """
    if max_steps <= 0:
        return max(t_min, 1e-6)
    return t_min + (t_max - t_min) * (steps_remaining / max_steps)


def entropy_bound_select(entropy, filled, entropy_bound: float):
    """Which still-masked positions to commit this step (absorbing variant).

    entropy: (G,) per-position entropy of the denoiser logits.
    filled:  (G,) bool — already-committed positions (excluded, stay fixed).
    Returns a (G,) bool mask of positions to commit now.

    Accept the ascending-entropy prefix while the sum of strictly-lower
    entropies stays ≤ bound (= the joint-MI upper bound of EntropyBoundSampler).
    The single most-confident masked position always passes (prefix sum 0), so
    at least one token is committed per step — no stalls.
    """
    e = mx.where(filled, mx.array(_BIG, dtype=entropy.dtype), entropy)   # (G,)
    order = mx.argsort(e)                                                # ascending
    se = mx.take(e, order)
    cum = mx.cumsum(se)
    prev = cum - se                                                      # Σ strictly-lower
    accept_sorted = (prev <= entropy_bound) & (se < _BIG)
    # scatter the per-sorted decision back to original positions
    G = entropy.shape[0]
    sel = mx.zeros((G,), dtype=mx.int32)
    sel = sel.at[order].add(accept_sorted.astype(mx.int32))
    return sel > 0


class StableConfidentStop:
    """Adaptive early stop (StableAndConfidentStoppingCriteria, batch=1).

    Stops denoising when the argmax canvas has been identical for
    ``stability`` consecutive steps AND the mean per-position entropy is below
    ``confidence`` (the model is both stable and confident).
    """

    def __init__(self, stability: int = 1, confidence: float = 0.005):
        self.stability = int(stability)
        self.confidence = float(confidence)
        self._history: list[list[int]] = []

    def reset(self) -> None:
        self._history = []

    def update(self, argmax_canvas: list[int], mean_entropy: float) -> bool:
        if self.stability <= 0:
            stable = True
        else:
            self._history.append(list(argmax_canvas))
            self._history = self._history[-self.stability:]
            stable = (len(self._history) == self.stability
                      and all(h == self._history[0] for h in self._history))
        confident = mean_entropy < self.confidence
        return stable and confident
