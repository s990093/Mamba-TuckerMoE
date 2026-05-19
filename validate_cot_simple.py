#!/usr/bin/env python3
"""
Simple CoT validation: Run a single "who are you?" prompt and show:
  1. Raw generation with CoT stream splitter
  2. Separated reasoning vs final answer
  3. Middleware state transitions
"""
import argparse
import sys
import os
import time
import json

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mlx.core as mx

from mamba3_mlx.cot_format_fsm import CotStreamSplitter
from mamba3_mlx.cot_middleware import (
    CotMiddleware,
    CotMiddlewareConfig,
    CotMiddlewareDeps,
)
from mamba3_mlx.inference.generator import prefill, decode_step, sample_token


SYSTEM_PROMPTS = {
    "self_awareness": (
        "You are Mamba in Self-Awareness mode. Answer identity and capability "
        "questions with strict architectural consistency: Hybrid Mamba-TuckerMoE, "
        "edge-deployed on iPhone, offline by default, no subjective consciousness, "
        "and no fabricated capabilities."
    ),
}


class SimpleInferenceTest:
    """Minimal inference loop for validation."""

    def __init__(self, model, tokenizer, middleware_deps, middleware_cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.middleware_deps = middleware_deps
        self.middleware_cfg = middleware_cfg

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> dict:
        """Run inference and track CoT stream parsing."""

        # Build full prompt
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

        # Tokenize
        prompt_ids = self.tokenizer.encode(full_prompt)
        print(f"[Input] {len(prompt_ids)} prompt tokens")

        # Prefill
        x = mx.array([prompt_ids])
        logits, mamba_states, kv_caches = prefill(self.model, x)
        mx.eval(logits)
        seq_pos = len(prompt_ids)

        # Initialize splitter and middleware
        splitter = CotStreamSplitter(start_in_think=True)
        middleware = CotMiddleware(
            deps=self.middleware_deps,
            cfg=self.middleware_cfg,
            reasoning=True,
            model_apply=None,
        )

        # Decode loop
        generated_ids = []
        events_log = []  # Track all events
        raw_text = ""
        t0 = time.perf_counter()

        for step_idx in range(max_tokens):
            # Transform logits through middleware
            # Handle both 2D (batch, vocab) and 3D (batch, seq, vocab) logits
            if logits.ndim == 3:
                logits_row = logits[0, -1, :]
            else:
                logits_row = logits[0, :] if logits.ndim == 2 else logits
            logits_row = middleware.transform_logits(logits_row)
            mx.eval(logits_row)

            # Sample
            tid = sample_token(logits_row, temperature=temperature, top_k=40)
            generated_ids.append(tid)

            # Decode chunk
            chunk = self.tokenizer.decode([tid], skip_special_tokens=False)
            raw_text += chunk

            # Feed through splitter for tracking
            for kind, payload in splitter.feed(chunk):
                events_log.append({"kind": kind, "text": payload[:50], "mode": splitter.mode})

            # Check stop
            if middleware.should_break(tid) or splitter.done:
                break

            # Decode step
            logits, mamba_states, kv_caches = decode_step(
                self.model, tid, mamba_states, kv_caches, step=seq_pos
            )
            mx.eval(logits)
            seq_pos += 1

        # Flush
        for kind, payload in splitter.flush():
            events_log.append({"kind": kind, "text": payload[:50], "mode": splitter.mode})

        t_end = time.perf_counter()

        # Extract reasoning and final from split events
        reasoning_text = ""
        final_text = ""
        for evt in events_log:
            if evt["kind"] == "reasoning":
                reasoning_text += evt["text"]
            elif evt["kind"] == "final":
                final_text += evt["text"]

        return {
            "raw_text": raw_text,
            "reasoning": reasoning_text,
            "final": final_text,
            "generated_tokens": len(generated_ids),
            "time_ms": (t_end - t0) * 1000,
            "events": events_log[:20],  # First 20 events
            "middleware_state": middleware.health_report(),
            "splitter_mode": splitter.mode,
        }


def get_args():
    p = argparse.ArgumentParser(description="Simple CoT validation test")
    p.add_argument("--checkpoint", type=str, default="checkpoints/latest_sft_cot_model.npz")
    p.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer")
    p.add_argument("--max-tokens", type=int, default=150)
    p.add_argument("--temp", type=float, default=0.7)
    return p.parse_args()


def main():
    args = get_args()

    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "SIMPLE CoT VALIDATION (who are you?)".center(78) + "║")
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

    # Load model
    print("Loading model...", end=" ", flush=True)
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
        reasoning_budget=1000,
        force_final_inject=False,
    )
    middleware_deps = CotMiddlewareDeps.build(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        existing_stop_ids=None,
        cfg=middleware_cfg,
    )

    # Run test
    tester = SimpleInferenceTest(model, tokenizer, middleware_deps, middleware_cfg)

    system_prompt = SYSTEM_PROMPTS["self_awareness"]
    user_prompt = "Who are you?"

    print(f"\nSystem prompt: {system_prompt[:70]}...\n")
    print(f"User prompt: {user_prompt}\n")
    print("Running inference...\n")

    result = tester.run(
        system_prompt,
        user_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temp,
    )

    # Display results
    print("=" * 80)
    print("RAW OUTPUT (with tags visible)")
    print("=" * 80)
    # Show first 500 chars with markers
    raw_preview = result["raw_text"][:500]
    print(repr(raw_preview))

    print("\n" + "=" * 80)
    print("SEPARATED BY CoT STREAM SPLITTER")
    print("=" * 80)

    print(f"\n[REASONING BLOCK] ({len(result['reasoning'])} chars)")
    print("-" * 40)
    print(result["reasoning"][:300] if result["reasoning"] else "(empty)")

    print(f"\n[FINAL ANSWER] ({len(result['final'])} chars)")
    print("-" * 40)
    print(result["final"][:300] if result["final"] else "(empty)")

    print("\n" + "=" * 80)
    print("MIDDLEWARE STATE")
    print("=" * 80)
    print(json.dumps(result["middleware_state"], indent=2))

    print("\n" + "=" * 80)
    print("EVENT LOG (first 20)")
    print("=" * 80)
    for i, evt in enumerate(result["events"], 1):
        print(f"{i:2}. {evt['kind']:10} | mode={evt['mode']:7} | {repr(evt['text'])}")

    print("\n" + "=" * 80)
    print("METRICS")
    print("=" * 80)
    print(f"Generated tokens: {result['generated_tokens']}")
    print(f"Time: {result['time_ms']:.1f}ms")
    print(f"Final splitter mode: {result['splitter_mode']}")

    # Quality check
    print("\n" + "=" * 80)
    print("QUALITY CHECK")
    print("=" * 80)

    has_reasoning = len(result["reasoning"]) > 0
    has_final = len(result["final"]) > 0
    in_final_mode = result["splitter_mode"] in ("final", "done")

    print(f"✓ Has reasoning: {has_reasoning}")
    print(f"✓ Has final answer: {has_final}")
    print(f"✓ Reached final mode: {in_final_mode}")
    print(f"✓ Thinking blocks properly formatted: {('</think>' in result['raw_text']) or '<final>' in result['raw_text']}")

    if has_reasoning and has_final:
        print("\n✓ CoT separation working: both reasoning and final answer present")
    elif not has_reasoning and has_final:
        print("\n⚠ No reasoning block: model jumped straight to final answer")
    elif has_reasoning and not has_final:
        print("\n⚠ No final answer: model generated only reasoning, missing </think> and <final>")
    else:
        print("\n✗ Neither reasoning nor final answer found - format parsing may be broken")


if __name__ == "__main__":
    main()
