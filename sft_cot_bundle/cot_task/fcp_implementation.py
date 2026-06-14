#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 — FCP (Format / EOS Penalty) and SCALe utilities.

All routines are pure tensor-ops (no `.item()`, no Python loops) so they can
sit inside a hot training loop or be `torch.compile`-d without breaking the
graph.  They are also fully differentiable: gradients flow from
`compute_fcp_penalty(...)` back to `logits`.

Conventions (matches the tokenizer in `sft_cot_bundle/dataset/tokenizer`):

    <|im_start|>  = 32000
    <|im_end|>    = 32001
    <think>       = 32002      (single token, FCP think_start_id)
    </think>      = 32003      (single token, FCP think_end_id)
    <final>       = 32004
    </final>      = 32005
    [PAD]         = 32006
    </s> (EOS)    = 2          (FCP eos_id)

The half-open think mask `[<think>, </think>)` produced by
`build_region_mask(input_ids, 32002, 32003)` includes the `<think>` token
itself but excludes `</think>`.  This is the right semantics for an
EOS-prediction penalty since logits[i] predicts token i+1.

NOTE
----
*   FCP penalty is *averaged* over the in-region tokens of the batch, then
    multiplied by `lambda_eos`.  That keeps the magnitude stable regardless of
    sequence/batch size — important so the same λ works for B=1 and B=8.
*   `compute_scale_weight_tensor` produces a per-position weight that can be
    multiplied into a per-token CE before the masked mean.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Region mask
# ---------------------------------------------------------------------------
def build_region_mask(
    input_ids: torch.Tensor,
    start_id: int,
    end_id: int,
) -> torch.Tensor:
    """Half-open mask `[start_id, end_id)` per sample via cumsum trick.

    Args:
        input_ids: (B, T) long tensor
        start_id:  single token id that opens the region (e.g. <think>)
        end_id:    single token id that closes the region (e.g. </think>)

    Returns:
        (B, T) float tensor with 1.0 inside the region, 0.0 outside.

    Edge cases (all handled implicitly):
      * No `<think>` found → all zeros.
      * `<think>` without matching `</think>` (truncation) → mask continues to
        the end of the sequence.  This is desirable: the EOS penalty should
        keep applying until the end if the model never properly closes think.
      * Multiple `<think>...</think>` regions → mask covers each one.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B, T); got {tuple(input_ids.shape)}")

    is_start = (input_ids == int(start_id)).to(torch.int32)
    is_end = (input_ids == int(end_id)).to(torch.int32)
    diff = is_start.cumsum(dim=1) - is_end.cumsum(dim=1)
    return (diff > 0).to(input_ids.device).float()


# ---------------------------------------------------------------------------
# FCP penalty
# ---------------------------------------------------------------------------
def compute_fcp_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    eos_id: int,
    t_start_id: int,
    t_end_id: int,
    delta: float = 0.01,
    lambda_eos: float = 0.2,
    region_mask: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average EOS-overconfidence penalty inside the think region.

    Penalty formula
    ---------------
        excess[b, t] = relu( p_eos[b, t] - delta )
        penalty      = ( Σ excess^2  over (think ∩ valid) )
                     / max(1, |think ∩ valid|)
                     * lambda_eos

    Args:
        logits:      (B, T, V) — float (any dtype, internally cast to float32)
        input_ids:   (B, T)    — long
        eos_id:      EOS token id (e.g. 2 for `</s>`)
        t_start_id:  `<think>` token id (e.g. 32002)
        t_end_id:    `</think>` token id (e.g. 32003)
        delta:       probability threshold below which EOS is "free".
        lambda_eos:  outer multiplier on the averaged penalty.
        region_mask: optional (B, T) float mask of the think region.  If None
                     it is computed from input_ids on the fly.
        valid_mask:  optional (B, T) float mask of *trainable* positions
                     (e.g. labels != -100).  When provided, masked positions
                     contribute neither to the numerator nor the denominator.

    Returns:
        penalty:     scalar — differentiable, ready to add to total loss.
        avg_eos_prob: scalar — mean p(EOS) in (think ∩ valid), for logging.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be (B, T, V); got {tuple(logits.shape)}")
    if input_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"input_ids shape {tuple(input_ids.shape)} mismatches "
            f"logits[:2] {tuple(logits.shape[:2])}"
        )

    # 1) EOS probability (float32 for stable softmax)
    eos_probs = F.softmax(logits.float(), dim=-1)[..., int(eos_id)]  # (B, T)

    # 2) Region mask
    if region_mask is None:
        region_mask = build_region_mask(input_ids, t_start_id, t_end_id)
    else:
        region_mask = region_mask.float()

    if valid_mask is not None:
        region_mask = region_mask * valid_mask.float()

    # 3) Excess probability and squared penalty
    excess = F.relu(eos_probs - float(delta))
    penalty_sq = (excess ** 2) * region_mask

    # 4) Aggregate (clamp denominator to avoid /0 → matches fix.md cumsum form)
    denom = region_mask.sum().clamp(min=1.0)
    penalty = penalty_sq.sum() / denom * float(lambda_eos)

    # 5) Mean EOS prob in region — same denominator semantics (for logging)
    avg_eos_prob = (eos_probs * region_mask).sum() / denom

    return penalty, avg_eos_prob


# ---------------------------------------------------------------------------
# SCALe — cosine annealing of think-region weight
# ---------------------------------------------------------------------------
def apply_scale_weight(
    global_step: int,
    total_steps: int,
    eta_max: float = 1.0,
    eta_min: float = 0.3,
) -> float:
    """Scalar cosine anneal eta_max → eta_min over `total_steps`."""
    if total_steps <= 0:
        return float(eta_max)
    progress = min(1.0, max(0.0, global_step / max(1, total_steps)))
    return float(eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * progress)))


def compute_scale_weight_tensor(
    input_ids: torch.Tensor,
    think_start_id: int,
    think_end_id: int,
    final_start_id: Optional[int],
    final_end_id: Optional[int],
    w_think: float,
    w_final: float = 1.0,
) -> torch.Tensor:
    """Per-position SCALe weight tensor.

    *   1.0 outside `<think>` and `<final>` regions
    *   `w_think` inside `<think>...</think>`
    *   `w_final` inside `<final>...</final>` (only if final ids are provided)

    Returns: (B, T) float tensor on `input_ids.device`.
    """
    out = torch.ones_like(input_ids, dtype=torch.float32)
    think_mask = build_region_mask(input_ids, think_start_id, think_end_id)
    # Anywhere think_mask is 1: replace with w_think
    if w_think != 1.0:
        out = torch.where(think_mask > 0, out.new_full(out.shape, float(w_think)), out)
    if final_start_id is not None and final_end_id is not None and w_final != 1.0:
        final_mask = build_region_mask(input_ids, final_start_id, final_end_id)
        out = torch.where(final_mask > 0, out.new_full(out.shape, float(w_final)), out)
    return out


# ---------------------------------------------------------------------------
# Tiny self-test (run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("compute_fcp_penalty / build_region_mask smoke tests")
    B, T, V = 2, 12, 32007
    eos_id, ts, te = 2, 32002, 32003

    # Construct a batch where one sample has a clear think region [3,8)
    ids = torch.zeros(B, T, dtype=torch.long)
    ids[0, 3] = ts; ids[0, 8] = te
    ids[1, 1] = ts; ids[1, 11] = te  # no </think> until end-1

    mask = build_region_mask(ids, ts, te)
    print("region_mask row0:", mask[0].tolist())
    print("region_mask row1:", mask[1].tolist())
    assert mask[0, 3] == 1 and mask[0, 7] == 1 and mask[0, 8] == 0
    assert mask[1, 1] == 1 and mask[1, 10] == 1 and mask[1, 11] == 0

    # Forcing high EOS probability inside think
    logits = torch.full((B, T, V), -10.0)
    logits[..., eos_id] = 5.0  # very high EOS prob everywhere
    logits.requires_grad_(True)

    pen, avg = compute_fcp_penalty(logits, ids, eos_id, ts, te,
                                    delta=0.01, lambda_eos=0.2)
    print(f"penalty={pen.item():.6f}  avg_eos_prob={avg.item():.6f}")
    pen.backward()
    print(f"grad finite? {torch.isfinite(logits.grad).all().item()}  "
          f"grad_norm={logits.grad.norm().item():.4f}")

    print("OK")
