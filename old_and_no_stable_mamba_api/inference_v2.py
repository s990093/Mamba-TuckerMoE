#!/usr/bin/env python3
"""
inference_v2.py — Pure autoregressive reference inference, mirroring train.py.

Design mirrors TrueHybridMamba.forward_inference() from pre-train/train.py:

  Phase 1  prefill  : full-sequence forward, populate per-layer caches
  Phase 2  decode   : one-token-per-step AR with cached Mamba states + KV buffers

All experimental Metal/fusion paths disabled — only standard MLX ops.
Router temperature fixed at 0.5 (train.py t_end / inference default).
No CoT injection, no speculative decoding, no mx.compile.

Usage:
    python inference/inference_v2.py --prompt "who are you?"
    python inference/inference_v2.py --prompt "2+2?" --temp 0.7 --top-k 50 --show-ids
    python inference/inference_v2.py --prompt "..." --sys-prompt "You are a helpful assistant."
    python inference/inference_v2.py --raw-prompt "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizerFast

from inference.lib.mlx_hybrid_infer import (
    Mamba3Config,
    Mamba3LanguageModel,
    strict_load_and_convert,
)

# ── Special token IDs (must match tokenizer vocab) ────────────────────────────
_IM_END     = 32001   # <|im_end|>  — primary stop token
_THINK_OPEN = 32002
_THINK_CLOSE = 32003
_FINAL_OPEN = 32004
_FINAL_CLOSE = 32005
_SPECIAL_DISPLAY = {
    32000: "<|im_start|>",
    _IM_END:     "<|im_end|>",
    _THINK_OPEN: "<think>",
    _THINK_CLOSE: "</think>",
    _FINAL_OPEN: "<final>",
    _FINAL_CLOSE: "</final>",
}

# Router temperature: matches get_router_temperature(step=None) → t_end=0.5 in train.py
_ROUTER_TEMP = 0.5


# ── Model config ──────────────────────────────────────────────────────────────

def build_config() -> Mamba3Config:
    """
    Exact hyperparameters from the SFT-CoT training run.
    All experimental optimisation flags are explicitly disabled so this matches
    the plain mathematical path in train.py.
    """
    cfg = Mamba3Config(
        d_model=768,
        d_state=64,
        d_head=64,
        expand=2,
        num_layers=6,
        mimo_rank=4,
        num_kv_heads=4,
        use_kmoe=True,
        kmoe_num_experts=8,
        kmoe_top_k=2,
        kmoe_r1=32,
        kmoe_r2=512,
        kmoe_r3=256,
        ffn_expand=6,
        chunk_size=64,
        rms_norm_eps=1e-5,
        layer_scale_init=1e-2,
    )
    # Disable every experimental Metal / fusion path.
    cfg.tucker_einsum_fuse   = False
    cfg.tucker_full_fuse     = False
    cfg.tucker_amx_fuse      = False
    cfg.tucker_scalar_fuse   = False
    cfg.fused_mamba_mixer    = False
    cfg.fused_mamba_mixer_v3 = False
    cfg.lookahead_router     = False
    return cfg


# ── Prompt formatting ─────────────────────────────────────────────────────────

def wrap_chatml(user_text: str, sys_text: str = "") -> str:
    if sys_text.strip():
        return (
            f"<|im_start|>system\n{sys_text.strip()}<|im_end|>\n"
            f"<|im_start|>user\n{user_text.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return (
        f"<|im_start|>user\n{user_text.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_next(logits_1d: mx.array, temp: float, top_k: int) -> int:
    """Greedy (temp=0.0) or temperature-scaled top-k sampling."""
    if temp == 0.0:
        return int(mx.argmax(logits_1d).item())
    scaled = logits_1d / temp
    if top_k > 0:
        k = min(top_k, int(scaled.shape[0]))
        threshold = float(mx.sort(scaled)[-k].item())
        scaled = mx.where(scaled >= threshold, scaled, mx.full(scaled.shape, -1e9))
    probs = mx.softmax(scaled, axis=-1)
    return int(mx.random.categorical(mx.log(probs + 1e-10)).item())


# ── Inference phases (aligned with train.py forward_inference) ────────────────

def run_prefill(
    model: Mamba3LanguageModel,
    input_ids: list,
    router_temp: mx.array,
):
    """
    Phase 1 — parallel prefill of the full prompt.

    Equivalent to train.py TrueHybridMamba.forward_inference(prefill=True):
      - All Mamba blocks run chunk-parallel scan (no per-step loop).
      - Transformer blocks run causal self-attention over the full prefix.
      - Returns logits (B, T, V) and populated per-layer cache list.
    """
    x = mx.array(input_ids, dtype=mx.int32)[None]   # (1, T)
    logits, caches = model(x, caches=None, seq_pos=None, router_temp=router_temp)
    mx.eval(logits, caches)
    return logits, caches


def run_decode_step(
    model: Mamba3LanguageModel,
    token_id: int,
    caches: list,
    seq_pos: int,
    router_temp: mx.array,
):
    """
    Phase 2 — single-token autoregressive step.

    Equivalent to train.py TrueHybridMamba.forward_inference(prefill=False):
      - Each Mamba block: h_new = h_prev * alpha + u  (scalar update, float32)
      - Each TransformerBlock: writes K/V at seq_pos, attends to full prefix.
      - Returns logits (1, 1, V) and updated cache list.
    """
    x = mx.array([[token_id]], dtype=mx.int32)
    sp = mx.array(seq_pos, dtype=mx.int32)
    logits, new_caches = model(x, caches=caches, seq_pos=sp, router_temp=router_temp)
    mx.eval(logits, new_caches)
    return logits, new_caches


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="inferenceV2 — pure autoregressive, train.py-aligned"
    )
    ap.add_argument("--checkpoint", default="checkpoints/latest_sft_cot_model.npz",
                    help="Path to .npz or .pt checkpoint (relative to repo root)")
    ap.add_argument("--tokenizer",  default="inference/tokenizer",
                    help="HuggingFace-compatible tokenizer directory")
    ap.add_argument("--prompt",     required=True, help="User message (auto-wrapped in ChatML)")
    ap.add_argument("--sys-prompt", default="",    help="Optional system prompt")
    ap.add_argument("--max-tokens", type=int,   default=300,  help="Maximum tokens to generate")
    ap.add_argument("--quantize",   type=int,   default=0,    choices=[0, 4, 8],
                    help="Post-training quantization bits (0 = bf16, most accurate)")
    ap.add_argument("--temp",       type=float, default=0.0,  help="Sampling temperature (0 = greedy)")
    ap.add_argument("--top-k",      type=int,   default=0,    help="Top-k filtering (0 = disabled)")
    ap.add_argument("--raw-prompt", action="store_true",
                    help="Disable ChatML wrapping — pass prompt string literally")
    ap.add_argument("--show-ids",   action="store_true",
                    help="Annotate each token with its vocab ID")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── 1. Tokenizer ─────────────────────────────────────────────────────────
    tok_path = os.path.join(root, args.tokenizer)
    print(f"[1/4] Loading tokenizer from {tok_path}", flush=True)
    tok = PreTrainedTokenizerFast.from_pretrained(tok_path)
    eos_id = tok.eos_token_id or 1
    stop_ids = {eos_id, _IM_END}

    # ── 2. Build model ────────────────────────────────────────────────────────
    print("[2/4] Building model (all experimental paths disabled)...", flush=True)
    config = build_config()
    model = Mamba3LanguageModel(config, vocab_size=32007)

    ckpt_path = os.path.join(root, args.checkpoint)
    strict_load_and_convert(model, ckpt_path)

    # ── 3. Optional quantization ──────────────────────────────────────────────
    if args.quantize > 0:
        print(f"[3/4] Quantizing to {args.quantize}-bit ...", flush=True)
        nn.quantize(model, bits=args.quantize)
    else:
        print("[3/4] No quantization (bf16).", flush=True)

    mx.eval(model.parameters())
    model.eval()

    dtype = model.embed.weight.dtype
    router_temp = mx.array(_ROUTER_TEMP, dtype=dtype)

    # ── 4. Tokenize prompt ────────────────────────────────────────────────────
    prompt_str = args.prompt if args.raw_prompt else wrap_chatml(args.prompt, args.sys_prompt)
    input_ids = tok.encode(prompt_str)
    print(f"[4/4] Prompt: {len(input_ids)} tokens  |  first 8 IDs: {input_ids[:8]}", flush=True)

    # ── Phase 1: Prefill ──────────────────────────────────────────────────────
    print("\n── Prefill ──────────────────────────────────────────────────", flush=True)
    t_pre = time.perf_counter()
    logits, caches = run_prefill(model, input_ids, router_temp)
    pre_ms = (time.perf_counter() - t_pre) * 1000
    print(f"   {len(input_ids)} tokens  {pre_ms:.0f} ms  "
          f"({len(input_ids) / (pre_ms / 1000):.0f} tok/s)", flush=True)

    # ── Phase 2: Autoregressive Decode ────────────────────────────────────────
    print("\n── Output ───────────────────────────────────────────────────\n", flush=True)

    seq_pos = len(input_ids)   # position of the next token to generate
    row = logits[0, -1, :]     # logits from the last prefill position
    n_gen = 0
    t_dec = time.perf_counter()

    while n_gen < args.max_tokens:
        tid = sample_next(row, args.temp, args.top_k)
        n_gen += 1

        # Print token
        if tid in _SPECIAL_DISPLAY:
            sys.stdout.write(f"\033[33m{_SPECIAL_DISPLAY[tid]}\033[0m")
        else:
            sys.stdout.write(tok.decode([tid]))
        if args.show_ids:
            sys.stdout.write(f"\033[90m[{tid}]\033[0m")
        sys.stdout.flush()

        # Stop on EOS / <|im_end|>
        if tid in stop_ids:
            sys.stdout.write("\n")
            break

        # Single-token decode step
        logits, caches = run_decode_step(model, tid, caches, seq_pos, router_temp)
        row = logits[0, -1, :]
        seq_pos += 1

    elapsed = time.perf_counter() - t_dec
    tps = n_gen / elapsed if elapsed > 0 else 0.0
    print(f"\n── {n_gen} tokens  {tps:.1f} tok/s ─────────────────────────", flush=True)


if __name__ == "__main__":
    main()
