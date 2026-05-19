#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point for MLX Mamba3 inference.

Usage:
  python mamba3_mlx/run.py --checkpoint checkpoints/latest_sft_cot_model.npz \
      --tokenizer checkpoints/tokenizer \
      --prompt "What is the capital of France?" \
      --max_tokens 200 --temp 0.8 --top_k 40
"""
import argparse
import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mlx.core as mx


# ── CoT stream formatter ───────────────────────────────────────────────────────

class _Formatter:
    """
    State-machine ANSI formatter for CoT streaming output.

    Handles partial tags at delta boundaries so a tag like <think>
    split across two deltas is still caught correctly.

    States:  pre → think → post → final → done
    """
    _DIM  = "\033[2;90m"       # dim gray  — <think> content
    _BOLD = "\033[1m"           # bold      — <final> content
    _TAG  = "\033[38;5;240m"   # dark gray — structural tags
    _RST  = "\033[0m"

    _TRANSITIONS: dict[str, str] = {
        "<think>":   "think",
        "</think>":  "post",
        "<final>":   "final",
        "</final>":  "done",
    }
    _HIDDEN: set[str] = {"<|im_end|>", "<|im_start|>"}
    _ALL_TAGS: set[str] = set(_TRANSITIONS) | _HIDDEN

    def __init__(self) -> None:
        self.state = "pre"
        self._buf  = ""

    def feed(self, chunk: str) -> str:
        """Consume a decoded-text delta and return styled text to print."""
        self._buf += chunk
        out: list[str] = []

        while True:
            best_pos, best_tag = len(self._buf), None
            for tag in self._ALL_TAGS:
                p = self._buf.find(tag)
                if 0 <= p < best_pos:
                    best_pos, best_tag = p, tag

            if best_tag is None:
                # No complete tag — hold back any partial at the tail
                hold = self._partial_hold()
                if hold:
                    safe, self._buf = self._buf[:-hold], self._buf[-hold:]
                else:
                    safe, self._buf = self._buf, ""
                out.append(self._styled(safe))
                break

            # Emit styled text before the tag
            out.append(self._styled(self._buf[:best_pos]))
            tag_str = self._buf[best_pos : best_pos + len(best_tag)]
            self._buf = self._buf[best_pos + len(best_tag):]

            if best_tag in self._HIDDEN:
                pass  # suppress
            else:
                out.append(f"{self._TAG}{tag_str}{self._RST}")
                self.state = self._TRANSITIONS[best_tag]

        return "".join(out)

    def flush(self) -> str:
        """Emit any remaining buffered text (call after the loop ends)."""
        text, self._buf = self._buf, ""
        return self._styled(text)

    def _styled(self, text: str) -> str:
        if not text:
            return ""
        color = {"think": self._DIM, "final": self._BOLD}.get(self.state, "")
        return f"{color}{text}{self._RST}" if color else text

    def _partial_hold(self) -> int:
        """How many trailing chars might be the start of a known tag."""
        best = 0
        for tag in self._ALL_TAGS:
            for n in range(1, min(len(tag), len(self._buf)) + 1):
                if self._buf.endswith(tag[:n]):
                    best = max(best, n)
        return best


# ── CLI ────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Mamba3-XR MLX inference")

    # Paths
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/latest_sft_cot_model.npz")
    p.add_argument("--tokenizer", type=str,
                   default="checkpoints/tokenizer")

    # Prompt
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--raw-prompt", action="store_true",
                   help="Do not wrap prompt in ChatML tags")
    p.add_argument("--system-prompt", type=str, default=None)
    p.add_argument("--prime-think", action="store_true",
                   help="Append <think>\\n to the assistant prefix so the model "
                        "enters the thinking block reliably (default: off)")

    # Generation
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--min_tokens", type=int, default=0,
                   help="Minimum new tokens before stop tokens are honoured (prevents early EOS)")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_p", type=float, default=0.05)
    p.add_argument("--rep_pen", type=float, default=1.1,
                   help="Repetition penalty (divide logits of seen tokens)")
    p.add_argument("--pres_pen", type=float, default=0.0,
                   help="Presence penalty")
    p.add_argument("--freq_pen", type=float, default=0.02,
                   help="Frequency penalty")
    p.add_argument("--repeat_last_n", type=int, default=64,
                   help="Token window for penalties")

    # Data types
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--kv_dtype", type=str, default="auto",
                   choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--quantize", type=int, default=0,
                   choices=[0, 4, 8],
                   help="Post-load linear weight quantization: 0=off, 4=4-bit, 8=8-bit. "
                        "Reduces memory bandwidth (faster decode) at slight quality cost.")

    # Performance
    p.add_argument("--full-decode-compile", action="store_true",
                   help="Compile decode step with mx.compile")
    p.add_argument("--speculative", action="store_true",
                   help="Enable speculative decoding (draft = same model, for demo)")
    p.add_argument("--speculative-k", type=int, default=5)

    # Output
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--benchmark", action="store_true",
                   help="Print throughput summary after generation")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colour formatting")

    return p.parse_args()


def dtype_from_str(s: str):
    return {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}[s]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # ── Load tokenizer ────────────────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:
        print(f"[error] Could not load tokenizer from {args.tokenizer}: {e}",
              file=sys.stderr)
        sys.exit(1)

    # ── Build prompt ──────────────────────────────────────────────────────────
    if args.raw_prompt:
        prompt_text = args.prompt
    else:
        from mamba3_mlx.inference.generator import wrap_prompt_chatml
        if args.system_prompt:
            prompt_text = (
                f"<|im_start|>system\n{args.system_prompt}<|im_end|>\n"
                + wrap_prompt_chatml(args.prompt)
            )
        else:
            prompt_text = wrap_prompt_chatml(args.prompt)
        if args.prime_think:
            prompt_text += "<think>\n"

    prompt_ids = tokenizer.encode(prompt_text)
    if args.verbose:
        print(f"[prompt] {len(prompt_ids)} tokens")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...", end=" ", flush=True)
    dtype = dtype_from_str(args.dtype)
    from mamba3_mlx.mlx_model.hybrid_model import build_model
    model = build_model(args.checkpoint, dtype=dtype)

    if args.quantize > 0:
        import mlx.nn as nn
        bits = args.quantize
        group_size = 64

        def _can_quantize(path: str, module) -> bool:
            if not isinstance(module, nn.Linear):
                return False
            # Skip head (unused; logits use embed.weight.T) and any shape that
            # isn't divisible by the quantisation group size.
            if "head" in path:
                return False
            if module.weight.shape[-1] % group_size != 0:
                return False
            return True

        nn.quantize(model, group_size=group_size, bits=bits,
                    class_predicate=_can_quantize)
        mx.eval(model.parameters())
        print(f"[quantized {bits}-bit]", end=" ", flush=True)

    print("done.")

    # ── Generation config ─────────────────────────────────────────────────────
    from mamba3_mlx.utils.config import GenerationConfig
    stop_ids: list[int] = []
    for tok_str in ["<|im_end|>", "</s>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if isinstance(tid, int) and tid >= 0:
                stop_ids.append(tid)
        except Exception:
            pass

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        min_new_tokens=args.min_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.rep_pen,
        presence_penalty=args.pres_pen,
        frequency_penalty=args.freq_pen,
        repetition_window=args.repeat_last_n,
        stop_token_ids=stop_ids,
        full_decode_compile=args.full_decode_compile,
        num_speculative_tokens=args.speculative_k,
    )

    # ── Pretty header ─────────────────────────────────────────────────────────
    use_color = not args.no_color and sys.stdout.isatty()
    B = "\033[1m" if use_color else ""
    R = "\033[0m" if use_color else ""

    if args.raw_prompt:
        print(prompt_text, end="", flush=True)
    else:
        print(f"\n{B}User:{R} {args.prompt}")
        print(f"{B}Assistant:{R} ", end="", flush=True)

    # ── Stream generation (cumulative decode + formatter) ─────────────────────
    from mamba3_mlx.inference.generator import stream_generate
    import time

    generated_ids: list[int] = []
    decoded_so_far = ""
    fmt = _Formatter() if use_color else None

    t0 = time.perf_counter()
    t_first: float | None = None
    n = 0

    for step in stream_generate(model, prompt_ids, gen_cfg):
        if t_first is None:
            t_first = time.perf_counter()

        generated_ids.append(step.token)
        n += 1

        # Cumulative decode — fixes BPE space handling for single-token streams
        new_decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
        delta = new_decoded[len(decoded_so_far):]
        decoded_so_far = new_decoded

        if delta:
            text = fmt.feed(delta) if fmt else delta
            sys.stdout.write(text)
            sys.stdout.flush()

    # Flush any partial tag held in the formatter buffer
    if fmt:
        remainder = fmt.flush()
        if remainder:
            sys.stdout.write(remainder)
            sys.stdout.flush()

    t_end = time.perf_counter()
    print()  # newline after output

    if args.benchmark or args.verbose:
        prefill_ms    = (t_first - t0) * 1000 if t_first else 0.0
        decode_elapsed = t_end - t_first if (t_first and n > 1) else 0.0
        decode_tps    = (n - 1) / decode_elapsed if decode_elapsed > 0 else 0.0
        print(f"\n{B}[benchmark]{R} {len(prompt_ids)} prompt tokens | "
              f"prefill {prefill_ms:.0f} ms | "
              f"{n} new tokens | "
              f"decode {decode_tps:.1f} tok/s | "
              f"total {t_end - t0:.2f}s")


if __name__ == "__main__":
    main()
