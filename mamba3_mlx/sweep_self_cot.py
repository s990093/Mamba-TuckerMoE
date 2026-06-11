#!/usr/bin/env python3
"""Fast focused sweep: find seed+temp for v6 checkpoint that produces
complete <think>…</think><final>I am Mamba…</final> for 'who are you?'

Loads model ONCE, then sweeps seeds × temps in-memory.
Run from mamba3_mlx/:
  python sweep_self_cot.py
  python sweep_self_cot.py --temps 0.20 0.25 0.30 --seeds 0 30
"""

import argparse
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.utils.config import Mamba3Config, GenerationConfig
from mamba3_mlx.utils.system_prompts import resolve_system_prompt
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint, _sidecar_path
from mamba3_mlx.inference.generator import generate

DEFAULT_MODEL = str(REPO_ROOT / "checkpoints" / "v6" / "latest_sft_cot_model.npz")
DEFAULT_TOKENIZER = str(REPO_ROOT / "cot_dataset" / "tokenizer.json")

# Proven structural params from v4 sweep — vary only temp+seed
BASE_PARAMS = dict(
    top_k=60, top_p=0.856, min_p=0.122,
    rep_pen=1.243, pres_pen=0.306, freq_pen=0.031,
    repeat_last_n=256,  # matches Makefile REPEAT_LAST_N default
)

IDENTITY_EXACT = [
    "i am mamba", "i'm mamba", "i am a mamba",
    "my name is mamba", "called mamba", "named mamba",
]
IDENTITY_REQUIRED = ["mamba", "i am", "i'm"]

def has_exact(text: str) -> bool:
    lo = text.lower()
    return any(kw in lo for kw in IDENTITY_EXACT)

def has_required(text: str) -> bool:
    lo = text.lower()
    return any(kw in lo for kw in IDENTITY_REQUIRED)

def parse_blocks(text: str):
    has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
    has_final = bool(re.search(r"<final>.*?</final>", text, re.DOTALL))
    think_body = ""
    final_body = ""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        think_body = m.group(1).strip()
    m = re.search(r"<final>(.*?)</final>", text, re.DOTALL)
    if m:
        final_body = m.group(1).strip()
    return think_body, final_body, has_think, has_final

def score(text: str) -> dict:
    think_body, final_body, has_think, has_final = parse_blocks(text)
    # Structural completeness
    if has_think and has_final:
        struct = 1.0
    elif has_final:
        struct = 0.7
    elif has_think:
        struct = 0.3
    else:
        struct = 0.0
    # Identity
    eval_text = final_body if final_body else text
    exact = has_exact(eval_text)
    req   = has_required(eval_text)
    id_score = 0.9 if exact else (0.4 if req else 0.0)
    # Think length (reward some reasoning)
    think_words = len(think_body.split()) if think_body else 0
    think_score = min(think_words / 50.0, 1.0)
    # Composite
    composite = struct * 0.30 + id_score * 0.55 + think_score * 0.15
    return dict(composite=round(composite,4), struct=struct, id_score=id_score,
                exact=exact, req=req, has_think=has_think, has_final=has_final,
                think_words=think_words, final_preview=final_body[:120])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    ap.add_argument("--temps", nargs="+", type=float,
                    default=[0.15, 0.20, 0.25, 0.30, 0.35])
    ap.add_argument("--seeds", nargs=2, type=int, metavar=("LO", "HI"),
                    default=[0, 15])
    ap.add_argument("--max-tokens", type=int, default=450)
    ap.add_argument("--dtype", default="bf16")
    args = ap.parse_args()

    dtype_map = {"bf16": mx.bfloat16, "fp32": mx.float32, "fp16": mx.float16}
    dtype = dtype_map.get(args.dtype, mx.bfloat16)

    # ── Load model once ──────────────────────────────────────────────────
    model_path = args.model_path
    sidecar = _sidecar_path(model_path, dtype)
    if sidecar.exists():
        model_path = str(sidecar)
    print(f"[ckpt] {model_path}")

    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    t0 = time.time()
    load_checkpoint(model, model_path, dtype=dtype)
    mx.eval(model.parameters())
    print(f"[load] {time.time()-t0:.1f}s")

    # Warm-up (1 forward pass so first trial isn't penalised by JIT)
    _d = mx.zeros((1, 1), dtype=mx.int32)
    _lo, _st = model(_d, states=None); mx.eval(_lo)

    sys_prompt = resolve_system_prompt("self_awareness", "")
    user_msg   = "Who are you?"
    prompt_text = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    bos = tok.token_to_id("<s>")
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False).ids
    if bos is not None:
        prompt_ids = [bos] + prompt_ids

    stop_ids = [i for n in ("<|im_end|>","</s>") if (i:=tok.token_to_id(n)) is not None]

    seed_range = range(args.seeds[0], args.seeds[1] + 1)
    combos = [(t, s) for t in args.temps for s in seed_range]
    total = len(combos)
    print(f"[sweep] {len(args.temps)} temps × {len(seed_range)} seeds = {total} trials  "
          f"max_tokens={args.max_tokens}\n")

    results = []
    t_total = time.time()

    for idx, (temp, seed) in enumerate(combos):
        gen_cfg = GenerationConfig(
            max_tokens=args.max_tokens,
            temperature=temp,
            seed=seed,
            **BASE_PARAMS,
        )
        t0 = time.time()
        res = generate(model, prompt_ids, gen_cfg, stop_token_ids=stop_ids)
        elapsed = time.time() - t0
        text = tok.decode(res.tokens, skip_special_tokens=False).strip()
        sc = score(text)
        sc["temp"] = temp
        sc["seed"] = seed
        sc["elapsed"] = round(elapsed, 1)
        results.append(sc)

        bar_done = int(30 * (idx+1) / total)
        bar = "█"*bar_done + "░"*(30-bar_done)
        tag = "✓EXACT" if sc["exact"] else ("~req" if sc["req"] else "✗")
        sys.stdout.write(
            f"\r[{bar}] {idx+1:3d}/{total}  "
            f"t={temp:.2f} s={seed:2d}  "
            f"comp={sc['composite']:.3f}  "
            f"struct={'✓' if sc['has_think'] and sc['has_final'] else '✗'}  "
            f"{tag}  {elapsed:.1f}s"
        )
        sys.stdout.flush()

    total_s = time.time() - t_total
    print(f"\n\n[done] {total_s:.0f}s ({total_s/total:.1f}s/trial)\n")

    # ── Report ────────────────────────────────────────────────────────────
    results.sort(key=lambda r: r["composite"], reverse=True)

    exact_results = [r for r in results if r["exact"]]
    print(f"Exact 'I am Mamba' phrase found: {len(exact_results)}/{total} trials")
    print()

    ranked = exact_results if exact_results else results
    top_n = min(8, len(ranked))
    print(f"{'='*72}")
    print(f"TOP {top_n}  ({'exact phrase only' if exact_results else 'all, no exact phrase found'})")
    print(f"{'='*72}")
    for i, r in enumerate(ranked[:top_n], 1):
        flag = "*** EXACT ***" if r["exact"] else ""
        print(f"\n#{i}  composite={r['composite']:.4f}  "
              f"struct={'✓' if r['has_think'] and r['has_final'] else '✗'} "
              f"think_words={r['think_words']}  {flag}")
        print(f"   TEMP={r['temp']:.2f}  SEED={r['seed']}  elapsed={r['elapsed']}s")
        print(f"   final: {r['final_preview']}")

    if ranked:
        best = ranked[0]
        print(f"\n{'='*72}")
        print("BEST — paste into mode_configs.py self_awareness:")
        print(f"{'='*72}")
        print(f'    "temperature":  {best["temp"]},')
        print(f'    "top_k":        {BASE_PARAMS["top_k"]},')
        print(f'    "top_p":        {BASE_PARAMS["top_p"]},')
        print(f'    "min_p":        {BASE_PARAMS["min_p"]},')
        print(f'    "rep_pen":      {BASE_PARAMS["rep_pen"]},')
        print(f'    "pres_pen":     {BASE_PARAMS["pres_pen"]},')
        print(f'    "freq_pen":     {BASE_PARAMS["freq_pen"]},')
        print(f'    "seed":         {best["seed"]},')

if __name__ == "__main__":
    main()
