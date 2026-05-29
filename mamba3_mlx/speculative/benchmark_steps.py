"""SJD step-by-step optimisation ladder benchmark.

Measures incremental TPS gain from layering on:
  - cot_caches      (pre-warmed per-bucket retriever + ngram from training corpus)
  - runtime_cache   (pre-warmed AR-sampled NGramCache + SuffixRetriever)
  - compile_verify  (mx.compile per-Mamba-layer verify step)

Run with::

    .venv/bin/python3 -m mamba3_mlx.speculative.benchmark_steps \\
        --K 16 --max_tokens 256

The numbers map to "Steps 1+2" of the optimisation plan in §15 of
LOOKAHEAD_COT_RESULTS.md.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint
from mamba3_mlx.speculative.drafts import SuffixRetriever
from mamba3_mlx.speculative.jacobi_sampling import jacobi_decode_sampling
from mamba3_mlx.speculative.ngram_cache import NGramCache
from mamba3_mlx.utils.config import GenerationConfig, Mamba3Config
from mamba3_mlx.utils.system_prompts import resolve_system_prompt


def _chatml(tok, sys_p, user, seed_think=False):
    text = (
        f"<|im_start|>system\n{sys_p}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n" + ("<think>\n" if seed_think else "")
    )
    ids = tok.encode(text, add_special_tokens=False).ids
    bos = tok.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids
    return ids


def _fresh(rt_state):
    return (
        NGramCache.from_state(rt_state["ngram"]),
        SuffixRetriever.from_state(rt_state["retriever"]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=str(
        REPO_ROOT / "checkpoints" / "v2" / "latest_sft_cot_model.mlx_bf16.npz"))
    p.add_argument("--tokenizer_path", default=str(
        REPO_ROOT / "cot_dataset" / "tokenizer.json"))
    p.add_argument("--cot_caches", default=str(
        REPO_ROOT / "mamba3_mlx" / "speculative" / "cot_caches_v2.pkl"))
    p.add_argument("--runtime_cache", default=str(
        REPO_ROOT / "mamba3_mlx" / "speculative" / "demo_cache_v2.pkl"))
    p.add_argument("--prompt", default="Who are you?")
    p.add_argument("--mode", default="self_awareness")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    load_checkpoint(model, args.model_path, dtype=mx.bfloat16)
    mx.eval(model.parameters())

    with open(args.runtime_cache, "rb") as f:
        rt = pickle.load(f)

    # K-graph warmup so the uncompiled baseline isn't charged JIT time.
    dummy = mx.zeros((1, args.K), dtype=mx.int32)
    _l, _s = model(dummy, states=None); mx.eval(_l); del _l, _s, dummy

    sys_p = resolve_system_prompt(args.mode, None)
    ids = _chatml(tok, sys_p, args.prompt)
    gc = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=0.15, top_k=20, top_p=0.85, min_p=0.08,
        seed=args.seed,
    )
    eos = [tok.token_to_id(n) for n in ("<|im_end|>", "</s>")
           if tok.token_to_id(n) is not None]

    def run(label, kw, warmup_compile=False):
        if warmup_compile:
            # Pay compile cost outside the timed run.
            jacobi_decode_sampling(
                model, ids, gc, K=args.K,
                use_ngram=True, ngram_n=4, use_retrieval=True,
                seed=args.seed, stop_token_ids=eos, no_eos_stop=False, **kw)
        t0 = time.perf_counter()
        r = jacobi_decode_sampling(
            model, ids, gc, K=args.K,
            use_ngram=True, ngram_n=4, use_retrieval=True,
            seed=args.seed, stop_token_ids=eos, no_eos_stop=False, **kw)
        dt = time.perf_counter() - t0
        print(f"{label:34s}  ARL={r.arl:5.2f}  rounds={r.n_rounds:4d}  "
              f"decode_tps={r.decode_tps:6.1f}  emit={len(r.tokens):4d}  "
              f"wall={dt:5.2f}s")
        return r

    print("=" * 100)
    print(f"SJD K={args.K}  mode={args.mode}  prompt='{args.prompt}'  "
          f"max={args.max_tokens}  seed={args.seed}")
    print("=" * 100)

    run("baseline (cold)", {})
    run("+ cot_caches", dict(cot_caches=args.cot_caches, cot_bucket=args.mode))

    ng, rtr = _fresh(rt)
    run("+ cot + runtime", dict(
        cot_caches=args.cot_caches, cot_bucket=args.mode,
        preloaded_ngram=ng, preloaded_retriever=rtr))

    ng, rtr = _fresh(rt)
    run("+ cot + runtime + compile  ★", dict(
        cot_caches=args.cot_caches, cot_bucket=args.mode,
        preloaded_ngram=ng, preloaded_retriever=rtr,
        compile_verify=True),
        warmup_compile=True)


if __name__ == "__main__":
    main()
