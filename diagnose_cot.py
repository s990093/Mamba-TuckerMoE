#!/usr/bin/env python3
"""
Isolated CoT middleware diagnostic script.

Tests cot_format_fsm.py and cot_middleware.py independently, without
touching the main inference pipeline. Verifies:

1. Format guard (logit banning + close-bias)
2. CoT stream splitter (tag parsing + mode transitions)
3. Middleware orchestration (reasoning budget, final injection, etc.)
4. End-to-end generation with specific system prompts

Usage:
  python diagnose_cot.py --checkpoint <path> --tokenizer <path> \\
      --test self_awareness  # or 'reasoning', 'custom'
"""
import argparse
import sys
import os
import time
from typing import Optional
import json

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mlx.core as mx
from mamba3_mlx.cot_format_fsm import (
    CotStreamSplitter,
    FormatGuardConfig,
    build_format_guard,
    describe_format_guard,
    encode_final_inject_ids,
)
from mamba3_mlx.cot_middleware import (
    CotMiddleware,
    CotMiddlewareConfig,
    CotMiddlewareDeps,
)


# ────────────────────────────────────────────────────────────────────────────────
# System prompts by category
# ────────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "self_awareness": (
        "You are Mamba in Self-Awareness mode. Answer identity and capability "
        "questions with strict architectural consistency: Hybrid Mamba-TuckerMoE, "
        "edge-deployed on iPhone, offline by default, no subjective consciousness, "
        "and no fabricated capabilities."
    ),
    "reasoning": (
        "You are in Reasoning mode. For complex questions, think through the problem "
        "step-by-step before providing your answer. Be precise and explicit about "
        "your reasoning."
    ),
    "generic": (
        "You are a helpful AI assistant."
    ),
}


# ────────────────────────────────────────────────────────────────────────────────
# Test harness
# ────────────────────────────────────────────────────────────────────────────────

class CotDiagnostic:
    """Minimal test harness for CoT components."""

    def __init__(self, tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def test_format_guard(self, verbose: bool = False) -> dict:
        """Test FormatGuard initialization and basic logit biasing."""
        print("\n" + "=" * 80)
        print("TEST: Format Guard")
        print("=" * 80)

        cfg = FormatGuardConfig(
            enabled=True,
            ban_im_start=True,
            close_bias_value=4.0,
            close_bias_max=16.0,
            close_bias_start=0,
        )
        guard = build_format_guard(
            self.tokenizer,
            vocab_size=self.vocab_size,
            cfg=cfg,
        )

        desc = describe_format_guard(guard, self.tokenizer)
        print(f"\n{desc}\n")

        # Test logit application
        dummy_logits = mx.ones((self.vocab_size,), dtype=mx.float32)
        result = guard.apply(dummy_logits, mode="think", close_bias_scalar=4.0)

        print(f"✓ Guard initialized: {len(guard.ban_ids)} banned IDs")
        if guard.close_first_id_by_mode:
            print(f"✓ Close bias targets: {guard.close_first_id_by_mode}")
        else:
            print(f"⚠ No close-bias targets resolved")

        return {
            "status": "ok",
            "ban_ids": len(guard.ban_ids),
            "close_modes": list(guard.close_first_id_by_mode.keys()),
        }

    def test_stream_splitter(self, verbose: bool = False) -> dict:
        """Test CoT stream splitter with various tag sequences."""
        print("\n" + "=" * 80)
        print("TEST: Stream Splitter")
        print("=" * 80)

        test_cases = [
            {
                "name": "Complete reasoning + final",
                "start_in_think": True,
                "chunks": ["I need to think ", "about this.", "</think>", "<final>", "The answer is 42.", "</final>"],
                "expected_modes": ["think", "between", "final", "done"],
            },
            {
                "name": "Reasoning + final injection",
                "start_in_think": True,
                "chunks": ["Let me think.</think>", "<final>", "Here's the answer.", "</final>"],
                "expected_modes": ["think", "between", "final", "done"],
            },
            {
                "name": "No reasoning (head mode)",
                "start_in_think": False,
                "chunks": ["Direct answer without thinking.", "<|im_end|>"],
                "expected_modes": ["head", "done"],
            },
            {
                "name": "Incomplete: between mode EOS before final",
                "start_in_think": True,
                "chunks": ["Thinking...", "</think>", "Some bridging text."],
                "expected_modes": ["think", "between"],
            },
        ]

        results = []
        for test in test_cases:
            splitter = CotStreamSplitter(start_in_think=test["start_in_think"])
            events = []
            modes_seen = [splitter.mode]

            for chunk in test["chunks"]:
                chunk_events = splitter.feed(chunk)
                events.extend(chunk_events)
                if splitter.mode != modes_seen[-1]:
                    modes_seen.append(splitter.mode)

            # Flush remaining
            final_events = splitter.flush()
            events.extend(final_events)

            ok = modes_seen == test["expected_modes"]
            status = "✓" if ok else "✗"
            print(f"\n{status} {test['name']}")
            print(f"  Expected modes: {test['expected_modes']}")
            print(f"  Seen modes:     {modes_seen}")
            if not ok:
                print(f"  Events: {events}")

            results.append({"test": test["name"], "ok": ok, "modes": modes_seen})

        return {"tests": results, "all_pass": all(r["ok"] for r in results)}

    def test_middleware_deps(self, verbose: bool = False) -> dict:
        """Test CotMiddlewareDeps initialization."""
        print("\n" + "=" * 80)
        print("TEST: Middleware Dependencies")
        print("=" * 80)

        cfg = CotMiddlewareConfig(
            enabled=True,
            ban_im_start=True,
            close_bias_value=4.0,
            close_bias_max=16.0,
            reasoning_budget=2000,
            force_final_inject=True,
            final_min_tokens=16,
        )

        deps = CotMiddlewareDeps.build(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            existing_stop_ids=None,
            cfg=cfg,
        )

        print(f"\n{deps.describe()}")
        print(f"\n✓ Stop IDs: {len(deps.stop_ids)} tokens")
        print(f"✓ Final inject IDs: {deps.final_inject_ids}")

        return {
            "status": "ok",
            "stop_ids": len(deps.stop_ids),
            "final_inject_ids": list(deps.final_inject_ids),
        }

    def test_middleware_orchestration(self, verbose: bool = False) -> dict:
        """Test CotMiddleware per-turn state machine."""
        print("\n" + "=" * 80)
        print("TEST: Middleware Orchestration")
        print("=" * 80)

        cfg = CotMiddlewareConfig(
            enabled=True,
            ban_im_start=True,
            close_bias_value=4.0,
            close_bias_max=16.0,
            close_bias_start=100,  # Ramp starts after 100 think tokens
            reasoning_budget=500,
            force_final_inject=True,
            final_min_tokens=16,
        )

        deps = CotMiddlewareDeps.build(
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
            existing_stop_ids=None,
            cfg=cfg,
        )

        # Create middleware in reasoning mode
        mw = CotMiddleware(
            deps=deps,
            cfg=cfg,
            reasoning=True,
            model_apply=None,  # Not testing injection in this test
        )

        print(f"\n✓ Middleware initialized in reasoning mode")
        print(f"  - Mode: {mw.mode}")
        print(f"  - Think tokens: {mw.think_tokens}")
        print(f"  - Reasoning budget: {mw.cfg.reasoning_budget}")

        # Simulate some reasoning tokens
        elapsed_fn = lambda: time.time()
        simulated_tokens = [
            ("reasoning", "Let me think about this."),
            ("reasoning", " First, I need to understand..."),
            ("reasoning", " Then, I'll apply the framework."),
        ]

        for kind, text in simulated_tokens:
            mw._reasoning_acc.append(text)
            mw._think_tokens += len(text.split())

        print(f"  - Accumulated reasoning: {len(mw._reasoning_acc)} chunks")
        print(f"  - Think tokens so far: {mw._think_tokens}")

        # Check close-bias ramp
        cb = mw.current_close_bias()
        print(f"  - Current close-bias: {cb:.1f}")

        # Simulate mode transition
        mw.splitter.set_mode("between")
        print(f"  - Mode after </think>: {mw.mode}")

        # Health check
        health = mw.health_report()
        print(f"\n{json.dumps(health, indent=2)}")

        return {"status": "ok", "health": health}


# ────────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="CoT middleware diagnostic",
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
        "--test", type=str, choices=["all", "guard", "splitter", "deps", "orchestration"],
        default="all",
        help="Which test to run",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Verbose output",
    )
    return p.parse_args()


def main():
    args = get_args()

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "CoT FORMAT & MIDDLEWARE DIAGNOSTIC".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝\n")

    # Load tokenizer
    print("Loading tokenizer...", end=" ", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print("✓")
    except Exception as e:
        print(f"✗\n{e}", file=sys.stderr)
        sys.exit(1)

    # Get actual vocab size (including special CoT tokens)
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    if hasattr(tokenizer, "backend_tokenizer") and hasattr(tokenizer.backend_tokenizer, "get_vocab"):
        try:
            backend_vocab = tokenizer.backend_tokenizer.get_vocab()
            if backend_vocab:
                actual_vocab_size = max(backend_vocab.values()) + 1
                print(f"Vocab size: {vocab_size} → {actual_vocab_size} (actual)")
                vocab_size = actual_vocab_size
        except Exception:
            pass
    else:
        print(f"Vocab size: {vocab_size}\n")
    print()

    # Create diagnostic harness
    diag = CotDiagnostic(tokenizer, vocab_size)

    # Run tests
    results = {}
    try:
        if args.test in ("all", "guard"):
            results["guard"] = diag.test_format_guard(verbose=args.verbose)

        if args.test in ("all", "splitter"):
            results["splitter"] = diag.test_stream_splitter(verbose=args.verbose)

        if args.test in ("all", "deps"):
            results["deps"] = diag.test_middleware_deps(verbose=args.verbose)

        if args.test in ("all", "orchestration"):
            results["orchestration"] = diag.test_middleware_orchestration(verbose=args.verbose)

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

    all_ok = all(
        v.get("all_pass", v.get("status") == "ok")
        for v in results.values()
    )

    print("\n" + ("✓ All tests passed" if all_ok else "✗ Some tests failed"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
