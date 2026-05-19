#!/usr/bin/env python3
"""
End-to-end CoT inference test.

Runs actual inference with the CoT middleware to verify generation quality
and identify where the degradation is happening.

Tests with specific system prompts to validate format consistency.

Usage:
  python test_cot_inference.py --checkpoint <path> --tokenizer <path>
"""
import argparse
import sys
import os
import time
import json
from typing import Iterator
from dataclasses import dataclass

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mlx.core as mx

from mamba3_mlx.cot_middleware import (
    CotMiddleware,
    CotMiddlewareConfig,
    CotMiddlewareDeps,
)
from mamba3_mlx.inference.generator import (
    prefill,
    decode_step,
    sample_token,
)


# ────────────────────────────────────────────────────────────────────────────────
# Test cases
# ────────────────────────────────────────────────────────────────────────────────

TEST_CASES = {
    "self_awareness": {
        "system": (
            "You are Mamba in Self-Awareness mode. Answer identity and capability "
            "questions with strict architectural consistency: Hybrid Mamba-TuckerMoE, "
            "edge-deployed on iPhone, offline by default, no subjective consciousness, "
            "and no fabricated capabilities."
        ),
        "prompts": [
            "Who are you?",
            "What are your capabilities?",
            "Are you conscious?",
        ],
    },
    "reasoning": {
        "system": (
            "You are in Reasoning mode. For complex questions, think through the problem "
            "step-by-step before providing your answer. Be precise and explicit about "
            "your reasoning."
        ),
        "prompts": [
            "Why is the sky blue?",
            "Explain photosynthesis.",
        ],
    },
}


# ────────────────────────────────────────────────────────────────────────────────
# Minimal inference loop
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Result from a single generation."""
    prompt: str
    reasoning: str  # Accumulated reasoning text
    final: str      # Accumulated final answer
    think_tokens: int
    final_tokens: int
    total_tokens: int
    time_ms: float
    budget_hit: bool
    final_injected: bool


def run_generation(
    model,
    tokenizer,
    middleware_deps: CotMiddlewareDeps,
    middleware_cfg: CotMiddlewareConfig,
    prompt_text: str,
    *,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_k: int = 40,
) -> GenerationResult:
    """
    Minimal generation loop using the middleware.
    """
    t0 = time.perf_counter()

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt_text)
    x = mx.array([prompt_ids])

    # Prefill
    logits, mamba_states, kv_caches = prefill(model, x)
    mx.eval(logits, *mamba_states, *kv_caches)
    seq_pos = len(prompt_ids)

    # Middleware (reasoning mode for these tests)
    mw = CotMiddleware(
        deps=middleware_deps,
        cfg=middleware_cfg,
        reasoning=True,  # Force thinking mode
        model_apply=None,  # No final injection for now
    )

    # Decode loop
    generated_ids = []
    for step_n in range(max_tokens):
        # Get logits
        logits_row = logits[0, -1, :]

        # Apply middleware transforms
        logits_row = mw.transform_logits(logits_row)
        mx.eval(logits_row)

        # Sample token
        tid = sample_token(logits_row, temperature=temperature, top_k=top_k)
        generated_ids.append(tid)

        # Check stop condition
        if mw.should_break(tid):
            break

        # Step (process token through middleware for state tracking)
        elapsed_s_fn = lambda: (time.perf_counter() - t0)
        for evt in mw.step(tid, n_out=step_n + 1, elapsed_s_fn=elapsed_s_fn):
            if evt.get("__stop__"):
                break

        # Single decode step
        logits, mamba_states, kv_caches = decode_step(
            model, tid, mamba_states, kv_caches, step=seq_pos
        )
        mx.eval(logits, *mamba_states, *kv_caches)
        seq_pos += 1

        if mw.should_break(tid):
            break

    # Decode and split output
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # Flush remaining text from middleware
    elapsed_s_fn = lambda: (time.perf_counter() - t0)
    for evt in mw.flush(n_out=len(generated_ids), elapsed_s_fn=elapsed_s_fn):
        pass

    t_end = time.perf_counter()

    # Health report
    health = mw.health_report()

    return GenerationResult(
        prompt=prompt_text,
        reasoning=mw.reasoning_markdown,
        final=full_output,
        think_tokens=mw.think_tokens,
        final_tokens=health["final_tokens"],
        total_tokens=len(generated_ids),
        time_ms=(t_end - t0) * 1000,
        budget_hit=health["budget_hit"],
        final_injected=health["final_injected"],
    )


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="End-to-end CoT inference test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/latest_sft_cot_model.npz",
        help="Model checkpoint path",
    )
    p.add_argument(
        "--tokenizer", type=str,
        default="checkpoints/tokenizer",
        help="Tokenizer path",
    )
    p.add_argument(
        "--test-case", type=str, choices=list(TEST_CASES.keys()),
        default="self_awareness",
        help="Which test case to run",
    )
    p.add_argument(
        "--max-tokens", type=int, default=300,
        help="Max tokens to generate per prompt",
    )
    p.add_argument(
        "--temp", type=float, default=0.7,
        help="Sampling temperature",
    )
    p.add_argument(
        "--reasoning-budget", type=int, default=2000,
        help="Hard cap on reasoning tokens",
    )
    return p.parse_args()


def main():
    args = get_args()

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "END-TO-END CoT INFERENCE TEST".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝\n")

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...", end=" ", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print("✓")
    except Exception as e:
        print(f"✗\n{e}", file=sys.stderr)
        sys.exit(1)

    # Load model
    print(f"Loading model from {args.checkpoint}...", end=" ", flush=True)
    try:
        from mamba3_mlx.mlx_model.hybrid_model import build_model
        model = build_model(args.checkpoint, dtype=mx.bfloat16)
        print("✓")
    except Exception as e:
        print(f"✗\n{e}", file=sys.stderr)
        sys.exit(1)

    # Setup middleware
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    middleware_cfg = CotMiddlewareConfig(
        enabled=True,
        ban_im_start=True,
        close_bias_value=4.0,
        close_bias_max=16.0,
        reasoning_budget=args.reasoning_budget,
        force_final_inject=False,  # Disabled for this test
        final_min_tokens=16,
    )

    middleware_deps = CotMiddlewareDeps.build(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        existing_stop_ids=None,
        cfg=middleware_cfg,
    )

    print(f"\n{middleware_deps.describe()}\n")

    # Get test case
    test_case = TEST_CASES[args.test_case]
    print(f"Test case: {args.test_case}")
    print(f"System prompt: {test_case['system'][:60]}...\n")

    results = []

    # Run prompts
    for prompt in test_case["prompts"]:
        print(f"\n{'-' * 80}")
        print(f"Prompt: {prompt}")
        print(f"{'-' * 80}")

        # Build full prompt
        prompt_text = (
            f"<|im_start|>system\n{test_case['system']}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>\n"  # Force thinking mode
        )

        try:
            result = run_generation(
                model,
                tokenizer,
                middleware_deps,
                middleware_cfg,
                prompt_text,
                max_tokens=args.max_tokens,
                temperature=args.temp,
            )

            # Display results
            print(f"\n[REASONING] ({result.think_tokens} tokens)")
            print(result.reasoning[:500] if result.reasoning else "(none)")

            print(f"\n[FINAL ANSWER] ({result.final_tokens} tokens)")
            print(result.final[:500] if result.final else "(none)")

            print(f"\n[METRICS]")
            print(f"  Total tokens: {result.total_tokens}")
            print(f"  Time: {result.time_ms:.1f}ms")
            print(f"  Budget hit: {result.budget_hit}")
            print(f"  Final injected: {result.final_injected}")

            results.append({
                "prompt": prompt,
                "reasoning_len": len(result.reasoning),
                "final_len": len(result.final),
                "think_tokens": result.think_tokens,
                "final_tokens": result.final_tokens,
                "total_tokens": result.total_tokens,
                "time_ms": result.time_ms,
                "budget_hit": result.budget_hit,
            })

        except Exception as e:
            print(f"\n✗ ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
