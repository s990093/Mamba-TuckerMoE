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
from typing import Any, Optional

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
        # Middleware config is set per-inference to allow CLI override
        self.mw_deps = None  # Will be created in infer() with proper config
        print("✓\n")

    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.5,
        reasoning_budget: int = 500,
        final_min_tokens: int = 16,
    ) -> dict:
        """Run inference and split output into reasoning + final."""

        # Build full prompt
        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

        prompt_ids = self.tokenizer.encode(full_prompt)
        print(f"[Prefill] {len(prompt_ids)} tokens")

        # Detect vocab size (same logic as __init__)
        vocab_size = (
            getattr(getattr(self.model, "config", None), "vocab_size", None)
            or len(self.tokenizer)
        )
        if hasattr(self.tokenizer, "backend_tokenizer"):
            try:
                backend_vocab = self.tokenizer.backend_tokenizer.get_vocab()
                if backend_vocab:
                    vocab_size = max(vocab_size, max(backend_vocab.values()) + 1)
            except Exception:
                pass

        # Initialize middleware with CLI-provided config
        mw_cfg = CotMiddlewareConfig(
            enabled=True,
            reasoning_budget=reasoning_budget,
            final_min_tokens=final_min_tokens,
        )
        if self.mw_deps is None:
            self.mw_deps = CotMiddlewareDeps.build(
                tokenizer=self.tokenizer,
                vocab_size=int(vocab_size),
                existing_stop_ids=None,
                cfg=mw_cfg,
            )
        print(f"Format guard: {self.mw_deps.describe()}\n")

        # Prefill
        x = mx.array([prompt_ids])
        logits, mamba_states, kv_caches = prefill(self.model, x)
        mx.eval(logits)
        seq_pos = len(prompt_ids)

        # Create model_apply function for final injection
        def model_apply(x_ids: mx.array, caches: Any, pos: mx.array) -> tuple[mx.array, Any]:
            """Forward pass for <final> injection during decode.

            x_ids shape: (1, N) with N token IDs to process sequentially
            caches: tuple of (mamba_states, kv_caches)
            pos: starting position as mx.array(scalar)

            Returns: (logits[1, N, vocab_size], updated_caches)
            """
            mamba_states, kv_caches = caches
            n_ids = x_ids.shape[1] if x_ids.ndim > 1 else 1
            token_ids = [int(x_ids[0, i]) if x_ids.ndim > 1 else int(x_ids[i]) for i in range(n_ids)]
            logits_list = []
            current_pos = int(pos)

            for tid in token_ids:
                _logits, mamba_states, kv_caches = decode_step(
                    self.model, tid, mamba_states, kv_caches, step=current_pos
                )
                # _logits shape: (1, vocab_size)
                logits_list.append(_logits)
                current_pos += 1

            # Stack logits: [(1, vocab) x N] -> (1, N, vocab_size)
            stacked = mx.stack(logits_list, axis=1)  # Stack on new axis
            return stacked, (mamba_states, kv_caches)

        # Initialize middleware with model_apply so final injection works
        middleware = CotMiddleware(
            deps=self.mw_deps,
            cfg=mw_cfg,
            reasoning=True,
            model_apply=model_apply,
        )

        # Decode
        # Note: max_tokens is a safety limit; actual stopping is controlled by
        # middleware (reasoning_budget, final_min_tokens, </final>, <|im_end|>)
        generated_ids = []
        reasoning_text = ""
        final_text = ""
        raw_text = ""
        t0 = time.perf_counter()
        stop_reason = None
        stopped = False

        step_idx = 0
        while step_idx < max_tokens:
            # Transform logits through middleware format guard
            if logits.ndim == 3:
                logits_row = logits[0, -1, :]
            else:
                logits_row = logits[0, :] if logits.ndim == 2 else logits
            logits_row = middleware.transform_logits(logits_row)
            mx.eval(logits_row)

            # Sample
            tid = sample_token(logits_row, temperature=temperature, top_k=40)
            generated_ids.append(tid)

            # Check if we sampled a hard stop token (before middleware processing)
            should_break = middleware.should_break(tid)

            # Let middleware process the token through the splitter
            # middleware._decode_chunk() handles special token logic and incremental decoding
            elapsed_ms = (time.perf_counter() - t0) * 1000
            prev_mode = middleware.mode
            for event in middleware.step(tid, n_out=len(generated_ids), elapsed_s_fn=lambda: elapsed_ms / 1000):
                if event.get("__stop__"):
                    stopped = True
                    stop_reason = "middleware_stop"
                elif event.get("type") == "reasoning":
                    text = event.get("markdown", "")
                    reasoning_text += text
                    if step_idx < 3:  # Debug first 3 steps
                        print(f"  [Step {step_idx}] reasoning: {repr(text[:50])}")
                elif event.get("type") == "token":
                    text = event.get("text", "")
                    final_text += text
                    if step_idx < 3:
                        print(f"  [Step {step_idx}] token: {repr(text[:50])}")

            # Try to inject <final> if we just transitioned to "between" mode
            if prev_mode != "between" and middleware.mode == "between":
                caches_tuple = (mamba_states, kv_caches)
                new_caches, new_seq_pos, new_logits_row, did_inject, ms = middleware.maybe_inject_final(
                    caches=caches_tuple, pos=seq_pos
                )
                if did_inject:
                    mamba_states, kv_caches = new_caches
                    seq_pos = new_seq_pos
                    if new_logits_row is not None:
                        logits = new_logits_row.reshape((1, 1, -1))
                        mx.eval(logits)
                        # Continue to next sampling without decode_step since we already have logits

            if should_break:
                stop_reason = "stop_token"
                stopped = True

            if stopped:
                break

            # Decode step for next iteration (unless we just did final injection)
            try:
                logits, mamba_states, kv_caches = decode_step(
                    self.model, tid, mamba_states, kv_caches, step=seq_pos
                )
                mx.eval(logits)
            except Exception as e:
                stop_reason = f"decode_error: {e}"
                break
            seq_pos += 1
            step_idx += 1

            # Emit progress every 50 tokens
            if step_idx % 50 == 0:
                print(f"  Generated {step_idx} tokens...", flush=True)

        # If we hit max_tokens hard limit (safety fallback)
        if step_idx >= max_tokens and not stopped:
            stop_reason = "max_tokens_limit"

        # Flush any remaining buffered text from the splitter
        for event in middleware.flush(n_out=len(generated_ids), elapsed_s_fn=lambda: (time.perf_counter() - t0)):
            if event.get("type") == "reasoning":
                reasoning_text += event.get("markdown", "")
            elif event.get("type") == "token":
                final_text += event.get("text", "")

        # Build raw_text at the end (once, not per-token)
        try:
            raw_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
        except Exception:
            raw_text = ""

        t_end = time.perf_counter()

        return {
            "raw_text": raw_text,
            "reasoning": reasoning_text,
            "final": final_text,
            "tokens": len(generated_ids),
            "time_ms": (t_end - t0) * 1000,
            "splitter_mode": middleware.mode,
            "stop_reason": stop_reason,
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
    if result.get("stop_reason"):
        print(f"Stop reason: {result['stop_reason']}")

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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Safety limit (actual stop is controlled by middleware: "
             "reasoning_budget, final_min_tokens, </final>, <|im_end|>). Default 4096."
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=500,
        help="Max tokens inside <think> block before forced stop. Default 500."
    )
    parser.add_argument(
        "--final-min-tokens",
        type=int,
        default=16,
        help="Min tokens inside <final> block before </final> allowed. Default 16."
    )
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
            reasoning_budget=args.reasoning_budget,
            final_min_tokens=args.final_min_tokens,
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
