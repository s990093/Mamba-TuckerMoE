#!/usr/bin/env python3
"""
Minimal "pure" MLX inference — no compile, no Metal kernels, no CoT filtering.
Loads weights → prefill → greedy decode → print raw tokens + IDs.

Use this to verify checkpoint quality independent of stream_mlx.py complexity.

Usage:
    .venv/bin/python inference/simple_infer.py --prompt "who are you?"
    .venv/bin/python inference/simple_infer.py --prompt "what is 2+2?" --show-ids
    .venv/bin/python inference/simple_infer.py --prompt "..." --sys-prompt "..." --temp 0.0
"""
import sys, os, math, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizerFast

from inference.lib.mlx_hybrid_infer import (
    Mamba3Config, Mamba3LanguageModel, strict_load_and_convert,
)


def wrap_chatml(user_text: str, sys_text: str = "") -> str:
    if sys_text.strip():
        return (
            f"<|im_start|>system\n{sys_text.strip()}<|im_end|>\n"
            f"<|im_start|>user\n{user_text.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return f"<|im_start|>user\n{user_text.strip()}<|im_end|>\n<|im_start|>assistant\n"


def sample_token(logits_row: mx.array, temp: float, top_k: int) -> int:
    if temp == 0.0:
        return int(mx.argmax(logits_row).item())
    scaled = logits_row / temp
    if top_k > 0:
        kth = int(mx.sort(scaled)[-min(top_k, scaled.shape[0])].item())
        scaled = mx.where(scaled >= kth, scaled, mx.full(scaled.shape, -1e9))
    probs = mx.softmax(scaled, axis=-1)
    return int(mx.random.categorical(mx.log(probs + 1e-10)).item())


def main():
    ap = argparse.ArgumentParser(description="Minimal pure-MLX inference (no compile)")
    ap.add_argument("--checkpoint", default="checkpoints/latest_sft_cot_model.npz")
    ap.add_argument("--tokenizer",  default="inference/tokenizer")
    ap.add_argument("--prompt",     required=True)
    ap.add_argument("--sys-prompt", default="")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--quantize",   type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--temp",       type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k",      type=int, default=0)
    ap.add_argument("--show-ids",   action="store_true", help="Print [tok_id] after each token")
    ap.add_argument("--raw-prompt", action="store_true", help="Skip ChatML wrapping")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── 1. Tokenizer ────────────────────────────────────────────────────
    tok_path = os.path.join(root, args.tokenizer)
    print(f"[1/4] Loading tokenizer from {tok_path} ...", flush=True)
    tok = PreTrainedTokenizerFast.from_pretrained(tok_path)

    # ── 2. Model (no Tucker fuse, no Metal, no compile) ──────────────────
    print("[2/4] Building model ...", flush=True)
    config = Mamba3Config(
        d_model=768, d_state=64, d_head=64, expand=2, num_layers=6,
        mimo_rank=4, num_kv_heads=4, use_kmoe=True,
        kmoe_num_experts=8, kmoe_top_k=2, kmoe_r1=32, kmoe_r2=512, kmoe_r3=256,
        ffn_expand=6,
    )
    config.tucker_einsum_fuse = False
    model = Mamba3LanguageModel(config, vocab_size=32007)

    ckpt_path = os.path.join(root, args.checkpoint)
    strict_load_and_convert(model, ckpt_path)

    if args.quantize > 0:
        print(f"[3/4] Quantizing {args.quantize}-bit ...", flush=True)
        nn.quantize(model, bits=args.quantize)
    else:
        print("[3/4] No quantization.", flush=True)

    mx.eval(model.parameters())
    model.eval()

    # ── 3. Tokenize ──────────────────────────────────────────────────────
    if args.raw_prompt:
        prompt_str = args.prompt
    else:
        prompt_str = wrap_chatml(args.prompt, args.sys_prompt)

    input_ids = tok.encode(prompt_str)
    print(f"[4/4] Prompt → {len(input_ids)} tokens", flush=True)
    print(f"      First 8 IDs: {input_ids[:8]}", flush=True)

    # ── 4. Prefill ───────────────────────────────────────────────────────
    print("\n── Prefill ──────────────────────────────────────────────────", flush=True)
    x_prefill = mx.array(input_ids)[None]   # (1, T)
    t0 = time.perf_counter()
    logits, caches = model(x_prefill, caches=None, seq_pos=None)
    mx.eval(logits, caches)
    prefill_ms = (time.perf_counter() - t0) * 1000
    print(f"   {len(input_ids)} tokens in {prefill_ms:.1f} ms", flush=True)

    # ── 5. Decode ────────────────────────────────────────────────────────
    print("\n── Raw output (all tokens shown, no filtering) ─────────────\n", flush=True)

    STOP_IDS = {tok.eos_token_id, 32001}   # </s>  <|im_end|>
    COT_NAMES = {32002: "<think>", 32003: "</think>", 32004: "<final>", 32005: "</final>"}
    THINK_CLOSE = 32003    # </think>
    FINAL_OPEN  = 32004    # <final>

    pos = len(input_ids)
    row = logits[0, -1, :]
    n_generated = 0
    t_decode_start = time.perf_counter()
    _saw_think_close = False    # set when first </think> fires
    _injected_final  = False    # set once we force-feed <final>

    def _step(feed_id: int):
        nonlocal caches, pos, row, logits
        x_one = mx.array([[feed_id]])
        sp = mx.array(pos, dtype=mx.int32)
        logits, caches = model(x_one, caches=caches, seq_pos=sp)
        mx.eval(logits, caches)
        pos += 1
        row = logits[0, -1, :]

    while n_generated < args.max_tokens:
        tid = sample_token(row, args.temp, args.top_k)
        n_generated += 1

        # ── force-inject <final> once, immediately after first </think> ──
        if tid == THINK_CLOSE and not _saw_think_close:
            _saw_think_close = True

        if _saw_think_close and not _injected_final and tid != THINK_CLOSE and tid != FINAL_OPEN:
            # model failed to generate <final> — inject it silently
            sys.stdout.write("\033[90m[→<final>injected]\033[0m")
            sys.stdout.flush()
            _injected_final = True
            _step(FINAL_OPEN)  # feed <final>, advance caches
            # now let the loop sample normally for the actual answer
            continue

        if tid in COT_NAMES:
            marker = COT_NAMES[tid]
            if args.show_ids:
                sys.stdout.write(f"\033[33m{marker}[{tid}]\033[0m")
            else:
                sys.stdout.write(f"\033[33m{marker}\033[0m")
            sys.stdout.flush()
        else:
            piece = tok.decode([tid])
            if args.show_ids:
                sys.stdout.write(f"{piece}[{tid}]")
            else:
                sys.stdout.write(piece)
            sys.stdout.flush()

        if tid in STOP_IDS:
            sys.stdout.write(f"\n[STOP id={tid}]\n")
            sys.stdout.flush()
            break

        _step(tid)

    elapsed = time.perf_counter() - t_decode_start
    tps = n_generated / elapsed if elapsed > 0 else 0
    print(f"\n\n── {n_generated} tokens  {tps:.1f} tok/s ────────────────────────", flush=True)


if __name__ == "__main__":
    main()
