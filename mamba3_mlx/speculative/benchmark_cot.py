"""Comprehensive benchmark of the COT-cache speculative configurations.

The matrix runs every (prompt × K × config) combo against the AR-sampling
baseline so we can read per-row speedup directly off the table.  Each cell
reports::

    arl     average accepted run length per round
    full%   % of rounds where the entire K-token draft was accepted
    tps     decoded tokens per second (excludes prefill and first token)
    wall    full decode wall-clock seconds
    spd     decode_tps / AR-baseline decode_tps for the same prompt

Run::

    .venv/bin/python3 -m mamba3_mlx.speculative.benchmark_cot \\
        --cot_caches mamba3_mlx/speculative/cot_caches_n4.pkl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.inference.generator import generate
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint
from mamba3_mlx.speculative.jacobi import jacobi_decode
from mamba3_mlx.speculative.jacobi_sampling import jacobi_decode_sampling
from mamba3_mlx.utils.config import GenerationConfig, Mamba3Config
from mamba3_mlx.utils.system_prompts import resolve_system_prompt


DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}


def _chatml_ids(tok, sys_prompt, user_msg):
    text = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    ids = tok.encode(text, add_special_tokens=False).ids
    bos = tok.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids
    return ids


def _warmup_graphs(model, Ks: list[int]) -> None:
    """Touch every K-sized verify graph + the 1-token graph so first-round
    compilation doesn't get charged to the timed decode of any config."""
    for L in {1, *Ks}:
        dummy = mx.zeros((1, L), dtype=mx.int32)
        l, _ = model(dummy, states=None)
        mx.eval(l)
        del l, _, dummy


def _run_ar(model, ids, gc, seed):
    t0 = time.perf_counter()
    r = generate(model, ids, gc, stop_token_ids=[], no_eos_stop=True)
    dt = time.perf_counter() - t0
    n = len(r.tokens)
    timed = max(n - 1, 0)
    tps = timed / max(dt, 1e-6)
    return {"arl": 1.0, "full_pct": 0.0,
            "decode_tps": tps, "wall": dt, "n_tokens": n}


def _run_jacobi(model, ids, gc, *, K, **kw):
    t0 = time.perf_counter()
    r = jacobi_decode(model, ids, gc, K=K, no_eos_stop=True, **kw)
    dt = time.perf_counter() - t0
    return {"arl": r.arl, "full_pct": 0.0,
            "decode_tps": r.decode_tps, "wall": dt, "n_tokens": len(r.tokens)}


def _run_sjd(model, ids, gc, *, K, **kw):
    t0 = time.perf_counter()
    r = jacobi_decode_sampling(model, ids, gc, K=K, no_eos_stop=True, **kw)
    dt = time.perf_counter() - t0
    return {"arl": r.arl,
            "full_pct": 100.0 * r.n_full_accepts / max(r.n_rounds, 1),
            "decode_tps": r.decode_tps, "wall": dt, "n_tokens": len(r.tokens)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        default=str(REPO_ROOT / "checkpoints" / "v2" / "latest_sft_cot_model.mlx_bf16.npz"),
    )
    p.add_argument(
        "--tokenizer_path",
        default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"),
    )
    p.add_argument(
        "--cot_caches",
        default=str(REPO_ROOT / "mamba3_mlx" / "speculative" / "cot_caches_n4.pkl"),
    )
    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP))
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--Ks", type=int, nargs="+", default=[8, 12])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompts", type=str, nargs="*", default=None,
                   help="'mode:user' pairs; defaults to a 3-prompt suite.")
    p.add_argument("--greedy", action="store_true",
                   help="Also profile the greedy path (jacobi_decode).")
    args = p.parse_args()

    if not Path(args.cot_caches).exists():
        raise SystemExit(
            f"COT cache not found: {args.cot_caches}\n"
            "Bake it first:\n"
            "  python -m mamba3_mlx.speculative.bake_cot_caches \\\n"
            "    --ngram_n 4 --out mamba3_mlx/speculative/cot_caches_n4.pkl"
        )

    print(f"[bench] loading {args.model_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    load_checkpoint(model, args.model_path, dtype=DTYPE_MAP[args.dtype])
    mx.eval(model.parameters())

    prompts: list[tuple[str, str]] = []
    if args.prompts:
        for spec in args.prompts:
            mode, _, user = spec.partition(":")
            prompts.append((mode, user))
    else:
        prompts = [
            ("self_awareness", "Who are you?"),
            ("math_drill", "Solve (15+3)*4/2 step by step"),
            ("daily_conversation", "What are some quick tips for staying focused while working from home?"),
        ]

    _warmup_graphs(model, args.Ks)

    cot_path = args.cot_caches
    ngram_n = 4
    sjd_filters = dict(temperature=0.15, top_p=0.85, top_k=20, min_p=0.08)

    HEADER = (
        f"{'config':28s} {'K':>3s} {'arl':>6s} {'full%':>6s} "
        f"{'tps':>7s} {'wall':>7s} {'spd':>6s}"
    )
    SEP = "-" * len(HEADER)

    for mode, user in prompts:
        sys_prompt = resolve_system_prompt(mode, "You are a helpful assistant.")
        ids = _chatml_ids(tok, sys_prompt, user)
        print()
        print(f"=== mode={mode}  prompt={user!r}  max={args.max_tokens}  dtype={args.dtype}  prompt_tokens={len(ids)} ===")
        print(HEADER)
        print(SEP)

        # Baseline AR-sampling (denominator for "spd")
        gc_sjd = GenerationConfig(max_tokens=args.max_tokens, seed=args.seed, **sjd_filters)
        ar = _run_ar(model, ids, gc_sjd, seed=args.seed)
        ar_tps = ar["decode_tps"]
        print(f"{'AR-sampling':28s} {'-':>3s} {ar['arl']:6.2f} {ar['full_pct']:6.1f} "
              f"{ar['decode_tps']:7.1f} {ar['wall']:7.2f} {1.0:6.2f}x")

        ar_greedy = None
        if args.greedy:
            gc_g = GenerationConfig(max_tokens=args.max_tokens)
            ar_greedy = _run_ar(model, ids, gc_g, seed=args.seed)
            print(f"{'AR-greedy':28s} {'-':>3s} {ar_greedy['arl']:6.2f} {ar_greedy['full_pct']:6.1f} "
                  f"{ar_greedy['decode_tps']:7.1f} {ar_greedy['wall']:7.2f} "
                  f"{ar_greedy['decode_tps']/max(ar_tps,1e-6):6.2f}x")

        for K in args.Ks:
            sjd_variants: list[tuple[str, dict[str, Any]]] = [
                ("SJD ng+rt",         dict(use_ngram=True, use_retrieval=True, ngram_n=ngram_n)),
                ("SJD ng+rt+cot",     dict(use_ngram=True, use_retrieval=True, ngram_n=ngram_n,
                                           cot_caches=cot_path)),
                ("SJD ng+rt+cot+r",   dict(use_ngram=True, use_retrieval=True, ngram_n=ngram_n,
                                           cot_caches=cot_path, cot_bucket=mode)),
                ("SJD ng+rt+cot+r+aK",dict(use_ngram=True, use_retrieval=True, ngram_n=ngram_n,
                                           cot_caches=cot_path, cot_bucket=mode,
                                           adaptive_K=True, K_min=4, K_max=24)),
            ]
            for name, kw in sjd_variants:
                r = _run_sjd(model, ids, gc_sjd, K=K, seed=args.seed, **kw)
                spd = r["decode_tps"] / max(ar_tps, 1e-6)
                print(f"{name:30s} {K:3d} {r['arl']:6.2f} {r['full_pct']:6.1f} "
                      f"{r['decode_tps']:7.1f} {r['wall']:7.2f} {spd:6.2f}x")
            if args.greedy:
                greedy_variants: list[tuple[str, dict[str, Any]]] = [
                    ("GREEDY ng+rt",       dict(use_ngram=True, use_retrieval=True, ngram_n=ngram_n)),
                    ("GREEDY ng+rt+cot",   dict(use_ngram=True, use_retrieval=True, ngram_n=ngram_n,
                                                cot_caches=cot_path)),
                    ("GREEDY ng+rt+cot+r", dict(use_ngram=True, use_retrieval=True, ngram_n=ngram_n,
                                                cot_caches=cot_path, cot_bucket=mode)),
                ]
                gc_g = GenerationConfig(max_tokens=args.max_tokens)
                for name, kw in greedy_variants:
                    r = _run_jacobi(model, ids, gc_g, K=K, **kw)
                    spd = r["decode_tps"] / max(ar_greedy["decode_tps"], 1e-6)
                    print(f"{name:30s} {K:3d} {r['arl']:6.2f} {r['full_pct']:6.1f} "
                          f"{r['decode_tps']:7.1f} {r['wall']:7.2f} {spd:6.2f}x")
            print(SEP)


if __name__ == "__main__":
    main()
