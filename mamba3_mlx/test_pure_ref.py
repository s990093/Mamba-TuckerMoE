#!/usr/bin/env python3
"""Test pure-mapping mlx_model (commit 7c31343) with v3 checkpoint.

Loads mlx_model_pure_ref as a proper package to check if "I am Mamba"
appears — confirming whether the issue is inference code or training data.

Usage:
    cd /Users/hungwei/Desktop/Proj/Mamba3-XR
    .venv/bin/python3 mamba3_mlx/test_pure_ref.py
    .venv/bin/python3 mamba3_mlx/test_pure_ref.py --seeds 0 1 2 3 4 5 42
"""
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAMBA_ROOT = REPO_ROOT / "mamba3_mlx"

# Add mamba3_mlx to sys.path so mlx_model_pure_ref resolves as a package
sys.path.insert(0, str(MAMBA_ROOT))
sys.path.insert(1, str(REPO_ROOT))

import mlx.core as mx

# Import pure-ref package (uses relative imports internally — works as package)
from mlx_model_pure_ref.hybrid_model import Mamba3LanguageModel
from mlx_model_pure_ref.weights import load_checkpoint as load_checkpoint_pure

from mamba3_mlx.utils.config import Mamba3Config, GenerationConfig
from mamba3_mlx.utils.system_prompts import resolve_system_prompt
from mamba3_mlx.inference.sampler import (
    sample_logits, apply_repetition_penalty, apply_freq_presence_penalty,
)
from tokenizers import Tokenizer

CKPT    = str(REPO_ROOT / "checkpoints" / "v3" / "latest_sft_cot_model.npz")
TOKPATH = str(REPO_ROOT / "cot_dataset" / "tokenizer.json")
EXACT_KW = ["i am mamba", "i'm mamba", "i am a mamba"]


def run_single(model, tok, prompt_ids, gen_cfg, stop_ids):
    ids = mx.array(prompt_ids, dtype=mx.int32)[None, :]
    logits, states = model(ids, states=None)
    mx.eval(logits)
    last_logits = logits[0, -1]

    key = mx.random.key(gen_cfg.seed)
    window = max(1, gen_cfg.repeat_last_n)
    generated = []
    stop_set = set(stop_ids)

    for _ in range(gen_cfg.max_tokens):
        z = last_logits.astype(mx.float32)
        recent = generated[-window:]
        z = apply_repetition_penalty(z, recent, gen_cfg.rep_pen)
        z = apply_freq_presence_penalty(z, recent, gen_cfg.pres_pen, gen_cfg.freq_pen)
        tok_arr, key = sample_logits(
            z, gen_cfg.temperature, gen_cfg.top_k, gen_cfg.top_p, gen_cfg.min_p, key)
        mx.eval(tok_arr)
        tid = int(tok_arr.item())
        generated.append(tid)
        if tid in stop_set:
            break
        step_ids = mx.array([[tid]], dtype=mx.int32)
        logits_out, states = model(step_ids, states=states)
        last_logits = logits_out[0, -1]
        mx.eval(last_logits)

    return tok.decode(generated, skip_special_tokens=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int,
                   default=[0, 1, 2, 3, 4, 5, 7, 11, 13, 17, 42, 99])
    p.add_argument("--temp",     type=float, default=0.259)
    p.add_argument("--top_k",    type=int,   default=60)
    p.add_argument("--top_p",    type=float, default=0.856)
    p.add_argument("--min_p",    type=float, default=0.122)
    p.add_argument("--rep_pen",  type=float, default=1.243)
    p.add_argument("--pres_pen", type=float, default=0.306)
    p.add_argument("--freq_pen", type=float, default=0.031)
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--prompt", default="Who are you?")
    args = p.parse_args()

    print(f"[pure-ref] tokenizer: {TOKPATH}", flush=True)
    tokenizer = Tokenizer.from_file(TOKPATH)
    print(f"[pure-ref] checkpoint: {CKPT}", flush=True)
    cfg = Mamba3Config(vocab_size=tokenizer.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    t0 = time.time()
    load_checkpoint_pure(model, CKPT, dtype=mx.bfloat16)
    mx.eval(model.parameters())
    print(f"[pure-ref] loaded in {time.time()-t0:.2f}s  (bf16 inter-chunk carry, no compile)", flush=True)

    sys_p = resolve_system_prompt("self_awareness", "")
    text = (f"<|im_start|>system\n{sys_p}<|im_end|>\n"
            f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n")
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    bos = tokenizer.token_to_id("<s>")
    if bos:
        ids = [bos] + ids
    stop_ids = [tokenizer.token_to_id(t) for t in ["<|im_end|>", "</s>"]
                if tokenizer.token_to_id(t)]

    print(f"\n[test] {len(args.seeds)} seeds  TEMP={args.temp}  TOP_K={args.top_k}\n")
    hits = []
    for seed in args.seeds:
        gen_cfg = GenerationConfig(
            max_tokens=args.max_tokens, temperature=args.temp,
            top_k=args.top_k, top_p=args.top_p, min_p=args.min_p,
            rep_pen=args.rep_pen, pres_pen=args.pres_pen, freq_pen=args.freq_pen,
            repeat_last_n=128, seed=seed,
        )
        t0 = time.time()
        out = run_single(model, tokenizer, ids, gen_cfg, stop_ids)
        elapsed = time.time() - t0
        found = any(kw in out.lower() for kw in EXACT_KW)
        marker = "✅ I AM MAMBA" if found else "✗"
        print(f"  seed={seed:4d}  {marker}  ({elapsed:.1f}s)")
        if found:
            hits.append(seed)
            lo = out.lower()
            for kw in EXACT_KW:
                idx = lo.find(kw)
                if idx >= 0:
                    print(f"    → ...{out[max(0,idx-20):idx+80].strip()}...")
                    break

    print(f"\n{'='*60}")
    print(f"[result]  {len(hits)}/{len(args.seeds)} seeds → 'I am Mamba'")
    if hits:
        print(f"[result]  winning seeds: {hits}")
        print("[conclusion] pure-ref code CAN produce 'I am Mamba'")
        print("             → optimized code broke it numerically")
    else:
        print("[result]  no exact phrase found in pure-ref either")
        print("[conclusion] issue is in training data / model weights")
        print("             → not an inference code problem")


if __name__ == "__main__":
    main()
