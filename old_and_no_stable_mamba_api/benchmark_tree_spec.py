#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0+P1 Tree Attention & Entropy-Adaptive K — correctness verification + A/B benchmark.

This script is a thin wrapper around benchmark_jacobi.py that:
  1. Runs a correctness check: tree mode must produce identical token sequences
     as linear +ngram for a deterministic (argmax) mock model.
  2. Runs the full A/B benchmark with 5 configs:
       baseline / +ngram (K=32) / +tree (B=3,D=10) / +adaptive_tree (B+entropy) / +adaptive

Usage:
  python inference/benchmark_tree_spec.py --checkpoint weights.npz
  python inference/benchmark_tree_spec.py --checkpoint weights.npz --num-prompts 5 \\
      --max-new-tokens 200 --tree-branches 3 --entropy-low 0.5 --entropy-high 1.5

Experiment table (see Section 4.3 of JACOBI_EXPERIMENT_REPORT.md for baseline):
  ┌──────────────────┬──────────┬────────┬──────┬────────────────┬─────────┐
  │ Config           │  tok/s   │speedup │ ARL  │  Branch-wins   │ MeanEnt │
  ├──────────────────┼──────────┼────────┼──────┼────────────────┼─────────┤
  │ baseline         │   67.2   │  1.00× │  —   │       —        │    —    │
  │ +ngram K32       │  185.7   │  2.82× │ 20.3 │       —        │    —    │
  │ +tree B3 D10     │   ???    │   ???  │  ??? │  b0:?/b1:?/b2:?│    —    │
  │ +adaptive_tree   │   ???    │   ???  │  ??? │  b0:?/b1:?/b2:?│   ???   │
  └──────────────────┴──────────┴────────┴──────┴────────────────┴─────────┘

Expected outcomes (see design doc):
  +tree B3 D10:    ARL 22→26 (+18%), verify cost 3×D=10 ≈ K=30, net ~3.0–3.2×
  +adaptive_tree:  ARL stable across diverse prompts, graceful fallback at high entropy
"""
from __future__ import annotations

import argparse
import os
import sys

_INF_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_INF_DIR, "lib")
for _p in (_LIB_DIR, _INF_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Section 1: Correctness check (no model needed, uses mock)
# ---------------------------------------------------------------------------

def run_correctness_check() -> None:
    """Verify tree mode produces the same token stream as linear mode on a deterministic model."""
    print("=" * 70)
    print("SECTION 1: Correctness check (deterministic mock model)")
    print("=" * 70)

    try:
        import mlx.core as mx
    except ImportError:
        print("[SKIP] MLX not available\n")
        return

    from jacobi_enhanced import (
        enhanced_jacobi_stream, build_compiled_verify_fns,
        NGramCache, build_guess_tree, _entropy, _find_best_tree_branch
    )

    VOCAB = 32
    vocab_range = mx.arange(VOCAB)
    prompt_ids = list[int]－(range(1, 8))  # [1,2,3,4,5,6,7]

    class DeterministicModel:
        """Always predicts (token + 1) % VOCAB — fully deterministic."""
        def __call__(self, x, *, caches, seq_pos, router_temp):
            targets = (x + 1) % VOCAB
            flat = targets.reshape(-1)
            one_hot = (flat[:, None] == vocab_range).astype(mx.float32) * 100.0
            logits = one_hot.reshape(x.shape[0], -1, VOCAB)
            return logits, caches

    model = DeterministicModel()
    router_temp = mx.array(0.5)
    MAX_TOKENS = 20
    K = 8

    last_row_vals = [0.0] * VOCAB
    last_row_vals[prompt_ids[-1] + 1] = 100.0
    dummy_logits = mx.array(
        [[[0.0] * VOCAB] * (len(prompt_ids) - 1) + [last_row_vals]]
    )

    class MockArgs:
        no_materialize_caches = True
        fast_sample = True
        temp = 0.0; top_k = 0; top_p = 1.0; min_p = 0.0
        rep_pen = 1.0; pres_pen = 0.0; freq_pen = 0.0; no_penalties = True

    # Build compiled verify fns for K_values = (2, 4, 8) + branch depths
    K_values = (2, 4, K)
    vfns: dict = {}
    for _k in sorted(set(range(1, K + 3)) | {K}):
        def _mk(kk):
            def _fn(xk, c, pos):
                return model(xk, caches=c, seq_pos=pos, router_temp=router_temp)
            return mx.compile(_fn)
        vfns[_k] = _mk(_k)
    for _k, _fn in vfns.items():
        _lw, _cw = _fn(mx.array([[0]*_k], dtype=mx.int32), None, mx.array(0, dtype=mx.int32))
        mx.eval(_lw, _cw)

    expected = [(prompt_ids[-1] + 1 + i) % VOCAB for i in range(MAX_TOKENS)]

    configs_to_test = [
        ("linear +ngram", dict(use_tree=False, use_entropy_k=False, num_tree_branches=1)),
        ("tree B=3",      dict(use_tree=True,  use_entropy_k=False, num_tree_branches=3)),
        ("entropy-K",     dict(use_tree=False, use_entropy_k=True,  num_tree_branches=1,
                               entropy_low=0.5, entropy_high=1.5)),
        ("tree+entropy",  dict(use_tree=True,  use_entropy_k=True,  num_tree_branches=3,
                               entropy_low=0.5, entropy_high=1.5)),
    ]

    all_ok = True
    for label, extra_kwargs in configs_to_test:
        collected = []
        for tok in enhanced_jacobi_stream(
            model=model,
            run_prefill=None, x_prefill=None,
            prompt_ids=prompt_ids,
            max_cache_len=512,
            router_temp=router_temp,
            max_new_tokens=MAX_TOKENS,
            sample_args=MockArgs(),
            prefill_outputs=(dummy_logits, None),
            K=K, K_min=2, K_max=K, K_values=K_values,
            use_ngram=True,
            ngram_n=2, min_ngram_n=1, ngram_max_size=500,
            use_lookahead=False,
            warm_prompt=True,
            compiled_verify_fns=vfns,
            materialize_cache_tree_fn=lambda x: x,
            pad_transformer_caches_fn=lambda c, _: c,
            **extra_kwargs,
        ):
            mx.eval(tok)
            collected.append(int(tok.item()))

        ok = (collected == expected)
        all_ok = all_ok and ok
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  [{status}] {label:20s}  generated={collected[:8]}...  expected={expected[:8]}...")

    print()
    if all_ok:
        print("Correctness check: ALL PASSED ✓\n")
    else:
        print("Correctness check: SOME FAILED ✗\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Section 2: Verify cost model (estimate expected throughput improvement)
# ---------------------------------------------------------------------------

def print_cost_model(K: int, tree_branches: int, tree_depth: int | None) -> None:
    """
    Analyze sequential tree attention vs. linear Jacobi using the empirical verify
    cost table from JACOBI_EXPERIMENT_REPORT.md.

    KEY CONSTRAINT: Sequential tree attention with B branches of depth D:
      - ARL upper bound = D (each branch can accept at most D tokens per round)
      - Verify cost = B × C_D (B separate forward passes of D tokens each)
      - For B=3, D=K/3: ARL ≤ K/3, but linear K achieves ARL ≈ 0.63×K > K/3

    This means sequential tree with D = K/B CANNOT beat linear Jacobi with the same K.

    Path to true tree attention benefit: batch the B branches in ONE model call [B, D],
    reducing verify cost from B×C_D to approximately C_D (tree attention masking).
    This requires model-level changes not yet implemented.

    Entropy-based K (P1) IS effective because it reduces wasted compute in uncertain
    segments without changing the fundamental linear Jacobi structure.
    """
    print("=" * 70)
    print("SECTION 2: Cost model analysis")
    print("=" * 70)
    print(f"  Parameters: K={K}, tree_branches={tree_branches}, tree_depth={tree_depth or 'K//B'}")

    # Empirical: C_K ≈ 10.4 + (K-1) × 1.64 ms (from JACOBI_EXPERIMENT_REPORT.md)
    # Fixed overhead (launch + Python) ≈ 10.4ms; per-token marginal cost ≈ 1.64ms
    def C(k: int) -> float:
        return 10.4 + max(0, k - 1) * 1.64

    D = tree_depth if tree_depth is not None else K // tree_branches
    ARL_linear = 22.1    # empirical, +ngram K=32, math CoT
    ARL_D_frac = 0.68    # ARL/D ratio at small D (slightly higher than 0.63 for K=32)

    # Max ARL for tree mode is D (branch depth ceiling)
    ARL_tree_max = min(D, D * ARL_D_frac)
    # Best case: all B branches survive D steps each, pick best → still ≤ D
    ARL_tree_best_case = D  # theoretical maximum

    c_linear = C(K)
    c_replay_linear = C(int(ARL_linear) + 1)
    total_linear = c_linear + c_replay_linear

    # Sequential tree: B calls of D tokens each
    c_tree_seq = tree_branches * C(D)
    c_replay_tree = C(int(ARL_tree_best_case) + 1)
    total_tree_seq = c_tree_seq + c_replay_tree

    # Hypothetical true parallel tree: ONE call with B×D tokens (tree attention mask)
    # Verify cost ≈ C(B*D) — same as linear K=B*D but accepts at most D tokens
    c_tree_parallel = C(tree_branches * D)
    total_tree_parallel = c_tree_parallel + c_replay_tree

    baseline_tps = 67.2

    def speedup(arl, total_ms):
        tps = arl / total_ms * 1000
        return tps / baseline_tps

    print(f"\n  [A] Linear K={K} (+ngram, empirical):")
    print(f"    verify cost:              {c_linear:.1f} ms")
    print(f"    replay cost:              {c_replay_linear:.1f} ms  (ARL={ARL_linear:.1f} + 1 tokens)")
    print(f"    total per round:          {total_linear:.1f} ms")
    print(f"    speedup:                  {speedup(ARL_linear, total_linear):.2f}×  (target: 2.82× empirical)")

    print(f"\n  [B] Sequential tree B={tree_branches}, D={D} (THIS IMPLEMENTATION):")
    print(f"    ARL upper bound:          {D}  (each branch max depth = D)")
    print(f"    verify cost:              {tree_branches} × {C(D):.1f} ms = {c_tree_seq:.1f} ms")
    print(f"    replay cost:              {c_replay_tree:.1f} ms  (best-case ARL={ARL_tree_best_case:.0f})")
    print(f"    total per round:          {total_tree_seq:.1f} ms")
    print(f"    speedup (best case):      {speedup(ARL_tree_best_case, total_tree_seq):.2f}×  "
          f"{'✓ better' if speedup(ARL_tree_best_case, total_tree_seq) > speedup(ARL_linear, total_linear) else '✗ WORSE than [A]'}")

    break_even_arl = ARL_linear * (total_tree_seq / total_linear)
    print(f"\n  VERDICT (sequential tree):")
    print(f"    Break-even ARL needed:    {break_even_arl:.1f}")
    print(f"    Maximum achievable ARL:   {D}  (= branch depth D)")
    if D < break_even_arl:
        print(f"    → Sequential tree CANNOT break even because D={D} < break_even={break_even_arl:.1f}")
        print(f"    → Would need D ≥ {break_even_arl:.0f} (but then B×D = {tree_branches * int(break_even_arl)} >> K={K})")
    else:
        print(f"    → Sequential tree CAN break even if ARL_tree ≥ {break_even_arl:.1f}")

    print(f"\n  [C] Hypothetical parallel tree (tree attention mask, one [B,D] call):")
    print(f"    ARL upper bound:          {D}  (same — branch depth still D)")
    print(f"    verify cost:              C({tree_branches}×{D}={tree_branches*D}) = {c_tree_parallel:.1f} ms  (one batched call)")
    print(f"    total per round:          {total_tree_parallel:.1f} ms")
    print(f"    speedup (best case D=ARL):{speedup(ARL_tree_best_case, total_tree_parallel):.2f}×")
    break_even_parallel = ARL_linear * (total_tree_parallel / total_linear)
    print(f"    Break-even ARL needed:    {break_even_parallel:.1f}  (D={D} {'≥' if D >= break_even_parallel else '<'} {break_even_parallel:.1f})")
    if D >= break_even_parallel:
        print(f"    → Parallel tree CAN break even ✓ (requires model-level tree attention mask)")
    else:
        print(f"    → Parallel tree STILL cannot break even at D={D}; need D ≥ {break_even_parallel:.0f}")

    print(f"\n  [P1] Entropy-adaptive K (THIS IMPLEMENTATION, no tree):")
    print(f"    Benefit: avoids wasted compute in uncertain/diverse segments")
    print(f"    Low-entropy (math CoT): K={K}, same as linear → ARL={ARL_linear:.1f}")
    print(f"    High-entropy (open Q&A): K=K_min, prevents ARL collapse → more tokens/round")
    print(f"    Net effect: makes Jacobi RELIABLE across prompt types, not just structured CoT")
    print(f"    Expected improvement: +10-20% overall speedup on mixed prompt sets\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="P0+P1 Tree Attention correctness + A/B benchmark"
    )
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--tree-branches", type=int, default=3)
    p.add_argument("--tree-depth", type=int, default=None)
    p.add_argument("--entropy-low", type=float, default=0.5)
    p.add_argument("--entropy-high", type=float, default=1.5)
    p.add_argument("--no-correctness-check", action="store_true",
                   help="Skip Section 1 (correctness check, no model needed)")
    p.add_argument("--no-benchmark", action="store_true",
                   help="Skip Section 3 (full A/B benchmark, requires model)")
    p.add_argument("--configs", type=str, default="baseline,+ngram,+tree,+adaptive_tree")
    args = p.parse_args()

    # Section 1: Correctness check
    if not args.no_correctness_check:
        run_correctness_check()

    # Section 2: Cost model analysis
    print_cost_model(
        K=args.K,
        tree_branches=args.tree_branches,
        tree_depth=args.tree_depth,
    )

    # Section 3: Full A/B benchmark
    if not args.no_benchmark:
        print("=" * 70)
        print("SECTION 3: Full A/B benchmark")
        print("=" * 70)
        print("Delegating to benchmark_jacobi.py …\n")

        # Build args list for benchmark_jacobi.main()
        bench_argv = [
            "--checkpoint", args.checkpoint,
            "--num-prompts", str(args.num_prompts),
            "--max-new-tokens", str(args.max_new_tokens),
            "--K", str(args.K),
            "--tree-branches", str(args.tree_branches),
            "--entropy-low", str(args.entropy_low),
            "--entropy-high", str(args.entropy_high),
            "--configs", args.configs,
        ]
        if args.tree_depth:
            bench_argv += ["--tree-depth", str(args.tree_depth)]

        import benchmark_jacobi
        sys.argv = ["benchmark_jacobi"] + bench_argv
        benchmark_jacobi.main()


if __name__ == "__main__":
    main()
