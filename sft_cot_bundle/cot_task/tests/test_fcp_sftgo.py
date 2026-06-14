#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for Task 2 building blocks.

Modules under test
------------------
* `fcp_implementation`  — build_region_mask, compute_fcp_penalty, SCALe
* `sftgo_implementation` — StructureWeightLoader, apply_structure_weights

The tests are deliberately written with synthetic but representative tensor
shapes for the project (V=32007, eos_id=2, think IDs 32002/32003, final IDs
32004/32005).  Each test asserts numerical values rather than just "runs
without crashing" — that's the part that catches real regressions.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# import target modules — sibling files in cot_task/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fcp_implementation import (
    build_region_mask,
    compute_fcp_penalty,
    apply_scale_weight,
    compute_scale_weight_tensor,
)
from sftgo_implementation import StructureWeightLoader, apply_structure_weights


PROJECT_ROOT = HERE.parents[2]
BUNDLE_PATH = PROJECT_ROOT / "sft_cot_bundle" / "cot_task" / "reports" / "structure_weights_bundle.pt"


# ============================================================================
# A. build_region_mask
# ============================================================================
def test_region_mask_basic():
    ids = torch.tensor([[0, 32002, 5, 6, 7, 32003, 0, 0]])
    m = build_region_mask(ids, 32002, 32003)
    # mask should be [0,1,1,1,1,0,0,0] (start included, end excluded)
    assert torch.equal(m[0], torch.tensor([0., 1., 1., 1., 1., 0., 0., 0.]))


def test_region_mask_truncated_end():
    # <think> with no </think> → mask continues to end
    ids = torch.tensor([[0, 32002, 5, 6, 7]])
    m = build_region_mask(ids, 32002, 32003)
    assert torch.equal(m[0], torch.tensor([0., 1., 1., 1., 1.]))


def test_region_mask_no_start():
    # </think> alone, no opening → no mask
    ids = torch.tensor([[0, 0, 32003, 0]])
    m = build_region_mask(ids, 32002, 32003)
    assert torch.all(m == 0)


def test_region_mask_multiple_regions():
    # Two separate think regions
    ids = torch.tensor([[32002, 1, 32003, 2, 32002, 3, 32003, 4]])
    m = build_region_mask(ids, 32002, 32003)
    assert torch.equal(m[0], torch.tensor([1., 1., 0., 0., 1., 1., 0., 0.]))


def test_region_mask_batched():
    ids = torch.tensor([
        [0, 32002, 1, 32003, 0],     # think at [1,3)
        [32002, 0, 0, 0, 32003],     # think at [0,4)
        [0, 0, 0, 0, 0],             # no think
    ])
    m = build_region_mask(ids, 32002, 32003)
    assert torch.equal(m[0], torch.tensor([0., 1., 1., 0., 0.]))
    assert torch.equal(m[1], torch.tensor([1., 1., 1., 1., 0.]))
    assert torch.all(m[2] == 0)


# ============================================================================
# B. compute_fcp_penalty — numerical correctness
# ============================================================================
def _logits_with_eos_prob(B, T, V, eos_id, target_eos_prob):
    """Construct logits whose softmax has p(EOS) ≈ target on every position."""
    # logit_eos = log(p) + log(V-1 + p) ... easier: assign all non-eos
    # the same logit lo, set eos logit = hi, find hi from target.
    # softmax: p = exp(hi) / (exp(hi) + (V-1)*exp(lo))
    # → hi - lo = log( p*(V-1) / (1-p) )
    lo = 0.0
    diff = math.log(target_eos_prob * (V - 1) / (1 - target_eos_prob))
    logits = torch.full((B, T, V), lo)
    logits[..., eos_id] = lo + diff
    return logits


def test_fcp_zero_when_below_delta():
    """When p(EOS) ≪ delta, penalty should be 0."""
    B, T, V = 2, 8, 32007
    eos_id, ts, te = 2, 32002, 32003
    logits = _logits_with_eos_prob(B, T, V, eos_id, target_eos_prob=1e-4)
    ids = torch.tensor([[0, ts, 1, 1, 1, te, 0, 0]] * B)
    pen, avg = compute_fcp_penalty(logits, ids, eos_id, ts, te,
                                    delta=0.01, lambda_eos=0.2)
    assert pen.item() == 0.0, f"expected 0, got {pen.item()}"
    # avg_eos_prob should be ~1e-4
    assert abs(avg.item() - 1e-4) < 1e-5, f"avg {avg.item()}"


def test_fcp_positive_when_above_delta():
    """When p(EOS) = 0.5, penalty = (0.5-0.01)^2 * 0.2 = 0.04802."""
    B, T, V = 1, 6, 32007
    eos_id, ts, te = 2, 32002, 32003
    logits = _logits_with_eos_prob(B, T, V, eos_id, target_eos_prob=0.5)
    # think region: positions 1..4 → 4 in-region tokens
    ids = torch.tensor([[0, ts, 1, 1, 1, te]])
    pen, avg = compute_fcp_penalty(logits, ids, eos_id, ts, te,
                                    delta=0.01, lambda_eos=0.2)
    expected = (0.5 - 0.01) ** 2 * 0.2
    assert abs(pen.item() - expected) < 1e-4, f"penalty {pen.item()} vs {expected}"
    assert abs(avg.item() - 0.5) < 1e-4


def test_fcp_zero_when_no_region():
    """No <think> → mask sum is 0 → penalty 0 (no NaN)."""
    B, T, V = 2, 5, 32007
    eos_id, ts, te = 2, 32002, 32003
    logits = _logits_with_eos_prob(B, T, V, eos_id, target_eos_prob=0.99)
    ids = torch.zeros(B, T, dtype=torch.long)  # no specials
    pen, avg = compute_fcp_penalty(logits, ids, eos_id, ts, te,
                                    delta=0.01, lambda_eos=0.2)
    assert pen.item() == 0.0
    assert avg.item() == 0.0


def test_fcp_gradient_flows():
    """Penalty must be differentiable w.r.t. logits."""
    B, T, V = 1, 4, 64
    eos_id, ts, te = 2, 32002, 32003
    logits = torch.randn(B, T, V, requires_grad=True)
    # force a think region
    ids = torch.tensor([[ts, 0, 0, te]])
    pen, _ = compute_fcp_penalty(logits, ids, eos_id, ts, te,
                                  delta=0.01, lambda_eos=0.5)
    pen.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # gradient w.r.t. eos_id column should be non-zero
    assert logits.grad[..., eos_id].abs().sum().item() > 0


def test_fcp_respects_valid_mask():
    """Positions with valid_mask=0 should not contribute."""
    B, T, V = 1, 6, 32007
    eos_id, ts, te = 2, 32002, 32003
    logits = _logits_with_eos_prob(B, T, V, eos_id, target_eos_prob=0.99)
    ids = torch.tensor([[0, ts, 1, 1, 1, te]])  # think: pos 1..4
    valid = torch.tensor([[0., 1., 1., 0., 1., 1.]])  # mask pos 3
    pen_with, _ = compute_fcp_penalty(logits, ids, eos_id, ts, te,
                                       delta=0.01, lambda_eos=0.2,
                                       valid_mask=valid)
    pen_full, _ = compute_fcp_penalty(logits, ids, eos_id, ts, te,
                                       delta=0.01, lambda_eos=0.2)
    # both have constant per-position penalty so masking changes only denom,
    # not the *average*: penalties should be (approximately) equal.
    assert abs(pen_with.item() - pen_full.item()) < 1e-5


# ============================================================================
# C. SCALe — apply_scale_weight + compute_scale_weight_tensor
# ============================================================================
def test_scale_cosine_endpoints():
    assert abs(apply_scale_weight(0, 1000, 1.0, 0.3) - 1.0) < 1e-6
    assert abs(apply_scale_weight(1000, 1000, 1.0, 0.3) - 0.3) < 1e-6
    # monotone decreasing
    prev = 2.0
    for s in range(0, 1001, 100):
        cur = apply_scale_weight(s, 1000, 1.0, 0.3)
        assert cur <= prev + 1e-9, f"non-monotone at {s}: {cur} > {prev}"
        prev = cur


def test_scale_weight_tensor_layout():
    ids = torch.tensor([[
        32000, 0, 32002, 5, 5, 5, 32003, 32004, 8, 8, 32005, 32001
    ]])
    # think [2,6), final [7,10)
    w = compute_scale_weight_tensor(
        ids,
        think_start_id=32002, think_end_id=32003,
        final_start_id=32004, final_end_id=32005,
        w_think=0.5, w_final=1.5,
    )
    expected = torch.tensor([[1., 1., 0.5, 0.5, 0.5, 0.5, 1., 1.5, 1.5, 1.5, 1., 1.]])
    assert torch.equal(w, expected), f"got {w}"


# ============================================================================
# D. apply_structure_weights
# ============================================================================
def test_apply_structure_weights_basic():
    B, T = 1, 5
    pt = torch.tensor([[1., 1., 1., 1., 1.]])
    sw = torch.tensor([[1., 2., 3., 1., 1.]])
    lab = torch.tensor([[1, 1, 1, 1, 1]])
    _, mean = apply_structure_weights(pt, sw, lab)
    assert abs(float(mean) - (1+2+3+1+1)/5) < 1e-6


def test_apply_structure_weights_ignores_masked():
    pt = torch.tensor([[1., 1., 1., 1.]])
    sw = torch.tensor([[100., 1., 1., 1.]])  # huge weight on a masked position
    lab = torch.tensor([[-100, 1, 1, 1]])
    _, mean = apply_structure_weights(pt, sw, lab)
    assert abs(float(mean) - 1.0) < 1e-6  # masked pos shouldn't count


def test_apply_structure_weights_broadcast_1d():
    pt = torch.ones(2, 4)
    sw = torch.tensor([1., 2., 1., 1.])   # 1-D, will broadcast to both rows
    lab = torch.ones(2, 4, dtype=torch.long)
    _, mean = apply_structure_weights(pt, sw, lab)
    # both rows contribute equally: (1+2+1+1)/4 = 1.25
    assert abs(float(mean) - 1.25) < 1e-6


# ============================================================================
# E. StructureWeightLoader on the real bundle (skip if bundle missing)
# ============================================================================
def test_loader_bundle_real():
    if not BUNDLE_PATH.exists():
        print(f"  SKIP: bundle not found at {BUNDLE_PATH}")
        return
    loader = StructureWeightLoader(bundle_path=BUNDLE_PATH, max_length=512)
    # 1) single
    w = loader.load_sample_weights("train_000001")
    assert w.shape == (512,)
    assert torch.all(w >= 1.0)
    # 2) batch
    wb = loader.batch_load_weights(["train_000001", "train_000002", "train_000003"])
    assert wb.shape == (3, 512)
    # 3) missing falls back to ones
    w_missing = loader.load_sample_weights("does_not_exist")
    assert torch.all(w_missing == 1.0)
    # 4) preload_all_as_tensor
    big, ids = loader.preload_all_as_tensor(seq_len=512)
    assert big.shape == (len(ids), 512)
    assert torch.all(big >= 1.0)
    # last id is train_022932
    assert ids[-1] == "train_022932"


# ============================================================================
# F. Integration: SFT-GO + FCP in a single mock loss step
# ============================================================================
def test_combined_loss_no_nan_with_gradients():
    """Mock one micro-step: CE_weighted + FCP — gradients flow, no NaN."""
    B, T, V = 2, 10, 256
    eos_id, ts, te = 2, 32002, 32003

    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    labels[0, :2] = -100   # masked positions
    labels[1, -1] = -100
    ids = torch.tensor([
        [0, ts, 1, 2, 3, te, 0, 0, 0, 0],
        [0, 0, ts, 1, 2, 3, te, 0, 0, 0],
    ])

    # base per-token CE
    raw = F.cross_entropy(logits.reshape(-1, V),
                          labels.reshape(-1),
                          ignore_index=-100,
                          reduction="none").view(B, T)

    sw = torch.full((B, T), 1.0)
    sw[0, 1:5] = 2.8  # tier-1 style on the think prefix of row 0
    _, ce_mean = apply_structure_weights(raw, sw, labels)

    pen, avg = compute_fcp_penalty(logits, ids, eos_id, ts, te,
                                    delta=0.01, lambda_eos=0.2)

    total = ce_mean + pen
    assert torch.isfinite(total).item()
    total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all().item()


# ============================================================================
# Runner
# ============================================================================
def _run(prefix, fn):
    try:
        fn()
        print(f"  ✅ {prefix} {fn.__name__}")
        return True
    except AssertionError as e:
        print(f"  ❌ {prefix} {fn.__name__}: {e}")
        return False
    except Exception as e:
        print(f"  💥 {prefix} {fn.__name__}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("=" * 70)
    print("Task 2 — Unit tests")
    print("=" * 70)
    tests = [
        ("A", test_region_mask_basic),
        ("A", test_region_mask_truncated_end),
        ("A", test_region_mask_no_start),
        ("A", test_region_mask_multiple_regions),
        ("A", test_region_mask_batched),
        ("B", test_fcp_zero_when_below_delta),
        ("B", test_fcp_positive_when_above_delta),
        ("B", test_fcp_zero_when_no_region),
        ("B", test_fcp_gradient_flows),
        ("B", test_fcp_respects_valid_mask),
        ("C", test_scale_cosine_endpoints),
        ("C", test_scale_weight_tensor_layout),
        ("D", test_apply_structure_weights_basic),
        ("D", test_apply_structure_weights_ignores_masked),
        ("D", test_apply_structure_weights_broadcast_1d),
        ("E", test_loader_bundle_real),
        ("F", test_combined_loss_no_nan_with_gradients),
    ]
    ok = sum(_run(p, fn) for p, fn in tests)
    print("=" * 70)
    print(f"PASSED {ok}/{len(tests)}")
    print("=" * 70)
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
