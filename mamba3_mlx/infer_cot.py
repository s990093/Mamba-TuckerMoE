#!/usr/bin/env python3
"""
Interactive CoT Inference with System Prompt Selection

Run inference with selectable system prompts (emotion, self_awareness, etc.)
and display reasoning/final answer separation powered by CotStreamSplitter.

Usage:
    python -m mamba3_mlx.infer_cot --prompt "Who are you?" --category self_awareness
    python -m mamba3_mlx.infer_cot --interactive  (choose prompt from menu)
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mlx.core as mx
from transformers import AutoTokenizer

from mamba3_mlx.cot_format_fsm import CotStreamSplitter
from mamba3_mlx.cot_middleware import CotMiddleware, CotMiddlewareConfig, CotMiddlewareDeps
from mamba3_mlx.inference.generator import prefill, decode_step, sample_token
from mamba3_mlx.mlx_model.hybrid_model import build_model
from mamba3_mlx.server_config import EXPORT_SYSTEM_PROMPTS, CATEGORY_TITLES


class CotInference:
    """Interactive inference with CoT stream splitting."""

    def __init__(self, checkpoint: str, tokenizer_path: str, dtype: str = "bf16"):
        self.checkpoint = checkpoint
        self.tokenizer_path = tokenizer_path
        self.dtype = {"fp32": mx.float32, "fp16": mx.float16}.get(dtype, mx.bfloat16)

        print("Loading tokenizer...", end=" ", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("✓")

        print("Loading model...", end=" ", flush=True)
        self.model = build_model(checkpoint, dtype=self.dtype)
        print("✓")

        # Initialize middleware
        vocab_size = (
            getattr(getattr(self.model, "config", None), "vocab_size", None)
            or len(self.tokenizer)
        )
        # Detect actual backend vocabulary size
        if hasattr(self.tokenizer, "backend_tokenizer"):
            try:
                backend_vocab = self.tokenizer.backend_tokenizer.get_vocab()
                if backend_vocab:
                    vocab_size = max(vocab_size, max(backend_vocab.values()) + 1)
            except Exception:
                pass

        print("Initializing middleware...", end=" ", flush=True)
        self.mw_cfg = CotMiddlewareConfig(enabled=True, reasoning_budget=500)
        self.mw_deps = CotMiddlewareDeps.build(
            tokenizer=self.tokenizer,
            vocab_size=int(vocab_size),
            existing_stop_ids=None,
            cfg=self.mw_cfg,
        )
        print("✓")
        print(f"Format guard: {self.mw_deps.describe()}\n")

    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> dict:
        """Run inference and split output into reasoning + final."""

        # Build full prompt
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

        prompt_ids = self.tokenizer.encode(full_prompt)
        print(f"[Prefill] {len(prompt_ids)} tokens\n")

        # Prefill
        x = mx.array([prompt_ids])
        logits, mamba_states, kv_caches = prefill(self.model, x)
        mx.eval(logits)
        seq_pos = len(prompt_ids)

        # Initialize splitter and middleware
        splitter = CotStreamSplitter(start_in_think=True)
        middleware = CotMiddleware(
            deps=self.mw_deps,
            cfg=self.mw_cfg,
            reasoning=True,
            model_apply=None,
        )

        # Decode
        generated_ids = []
        reasoning_text = ""
        final_text = ""
        raw_text = ""
        t0 = time.perf_counter()

        for step_idx in range(max_tokens):
            # Transform logits
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

            # Feed through splitter
            for kind, payload in splitter.feed(chunk):
                if kind == "reasoning":
                    reasoning_text += payload
                elif kind == "final":
                    final_text += payload

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
            if kind == "reasoning":
                reasoning_text += payload
            elif kind == "final":
                final_text += payload

        t_end = time.perf_counter()

        return {
            "raw_text": raw_text,
            "reasoning": reasoning_text,
            "final": final_text,
            "tokens": len(generated_ids),
            "time_ms": (t_end - t0) * 1000,
            "splitter_mode": splitter.mode,
            "middleware": middleware.health_report(),
        }


def list_categories() -> None:
    """Show available system prompts."""
    print("\n" + "═" * 80)
    print("AVAILABLE SYSTEM PROMPTS")
    print("═" * 80)
    for i, (key, title) in enumerate(CATEGORY_TITLES.items(), 1):
        prompt = EXPORT_SYSTEM_PROMPTS[key]
        print(f"\n{i}. {title.upper()} ({key})")
        print(f"   {prompt[:100]}...")


def interactive_choose() -> tuple[str, str, str]:
    """Interactive menu to choose system prompt and user input."""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "INTERACTIVE COT INFERENCE".center(78) + "║")
    print("╚" + "═" * 78 + "╝\n")

    # List categories
    categories = list(CATEGORY_TITLES.keys())
    list_categories()

    # Choose category
    print("\n" + "─" * 80)
    while True:
        try:
            choice = int(input("\nChoose category (1-7): "))
            if 1 <= choice <= len(categories):
                chosen_key = categories[choice - 1]
                break
            print("Invalid choice. Try again.")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input.")

    # Get system prompt
    system_prompt = EXPORT_SYSTEM_PROMPTS[chosen_key]
    category_name = CATEGORY_TITLES[chosen_key]

    print(f"\n✓ Selected: {category_name}")
    print(f"  Prompt: {system_prompt[:80]}...\n")

    # Get user input
    user_prompt = input("Enter your question: ").strip()
    if not user_prompt:
        user_prompt = "Who are you?"

    return chosen_key, system_prompt, user_prompt


def format_output(result: dict, category: str) -> None:
    """Pretty-print inference results."""
    print("\n" + "═" * 80)
    print("INFERENCE RESULTS")
    print("═" * 80)

    print(f"\nCategory: {CATEGORY_TITLES.get(category, category)}")
    print(f"Tokens: {result['tokens']} | Time: {result['time_ms']:.1f}ms | Mode: {result['splitter_mode']}")

    print("\n" + "─" * 80)
    print("REASONING BLOCK")
    print("─" * 80)
    if result["reasoning"]:
        text = result["reasoning"].strip()
        preview = text[:300]
        print(f"({len(text)} chars)\n{preview}")
        if len(text) > 300:
            print("...")
    else:
        print("(empty)")

    print("\n" + "─" * 80)
    print("FINAL ANSWER")
    print("─" * 80)
    if result["final"]:
        text = result["final"].strip()
        preview = text[:300]
        print(f"({len(text)} chars)\n{preview}")
        if len(text) > 300:
            print("...")
    else:
        print("(empty)")

    print("\n" + "─" * 80)
    print("QUALITY METRICS")
    print("─" * 80)
    has_reasoning = len(result["reasoning"]) > 10
    has_final = len(result["final"]) > 0
    in_final_mode = result["splitter_mode"] in ("final", "done")

    print(f"✓ Has reasoning: {has_reasoning}")
    print(f"✓ Has final answer: {has_final}")
    print(f"✓ Reached final mode: {in_final_mode}")

    if has_reasoning and has_final:
        print("\n✅ PASS: CoT separation working")
    elif not has_reasoning:
        print("\n⚠️ INCOMPLETE: No reasoning block detected")
    elif not has_final:
        print("\n⚠️ INCOMPLETE: No final answer generated")
    else:
        print("\n❌ FAIL: Format parsing broken")

    print("\n" + "─" * 80)
    print("MIDDLEWARE STATE")
    print("─" * 80)
    mw = result["middleware"]
    print(json.dumps(mw, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CoT inference with system prompt selection"
    )
    parser.add_argument(
        "--checkpoint",
        default=str(_REPO_ROOT / "checkpoints" / "latest_sft_cot_model.npz"),
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--tokenizer",
        default=str(_REPO_ROOT / "checkpoints" / "tokenizer"),
        help="Tokenizer directory",
    )
    parser.add_argument("--prompt", type=str, help="User prompt (for non-interactive mode)")
    parser.add_argument(
        "--category",
        type=str,
        default="self_awareness",
        choices=list(EXPORT_SYSTEM_PROMPTS.keys()),
        help="System prompt category",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (choose category and prompt)",
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--list-categories", action="store_true", help="Show available categories")

    args = parser.parse_args()

    # Just list categories and exit
    if args.list_categories:
        list_categories()
        return

    # Interactive mode
    if args.interactive:
        category, system_prompt, user_prompt = interactive_choose()
    else:
        category = args.category
        system_prompt = EXPORT_SYSTEM_PROMPTS[category]
        user_prompt = args.prompt or "Who are you?"

    # Run inference
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + f"COT INFERENCE: {CATEGORY_TITLES.get(category, category)}".center(78) + "║")
    print("╚" + "═" * 78 + "╝\n")

    try:
        engine = CotInference(args.checkpoint, args.tokenizer)
        result = engine.infer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temp,
        )
        format_output(result, category)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
