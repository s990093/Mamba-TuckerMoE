"""CLI runner + sanity check for jacobi_decode_sampling (SJD).

Measures ARL across K values and verifies that the AR-sampling baseline and
the SJD output stay statistically equivalent (token-overlap stays sane
given the same RNG seed and filters).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.inference.generator import _iter_state_arrays, generate, prefill
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import _sidecar_path, load_checkpoint
from mamba3_mlx.speculative.jacobi_sampling import jacobi_decode_sampling
from mamba3_mlx.utils.config import GenerationConfig, Mamba3Config
from mamba3_mlx.utils.system_prompts import MODE_ALIASES, resolve_system_prompt


DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Think step by step before answering."
)


def _build_chatml_prompt(tokenizer, system_prompt, user_msg, seed_think):
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        + ("<think>\n" if seed_think else "")
    )
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    bos = tokenizer.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids
    return ids, text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",
                   default=str(REPO_ROOT / "checkpoints" / "latest_sft_cot_model.npz"))
    p.add_argument("--tokenizer_path",
                   default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"))
    p.add_argument("--prompt", default="Who are you?")
    p.add_argument("--mode", default="self_awareness",
                   choices=sorted(set(MODE_ALIASES.keys())))
    p.add_argument("--system", default=_DEFAULT_SYSTEM)
    p.add_argument("--no-seed-think", action="store_true")
    p.add_argument("--raw-prompt", action="store_true")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--K", action="append", type=int, default=None,
                   help="K values to test. Default: [4, 8, 12, 16].")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_p", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP))
    p.add_argument("--no_ngram", action="store_true")
    p.add_argument("--no_retrieval", action="store_true")
    p.add_argument("--ngram_n", type=int, default=4)
    p.add_argument("--no-eos-stop", action="store_true")
    p.add_argument("--ar-baseline", action="store_true",
                   help="Also run AR-sampling baseline for wall-clock comparison.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    print(f"[load] {args.model_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    sidecar = _sidecar_path(args.model_path, dtype)
    if sidecar.exists():
        print(f"[load] sidecar: {sidecar.name}", file=sys.stderr)
    load_checkpoint(model, args.model_path, dtype=dtype)
    mx.eval(model.parameters())

    sys_prompt = resolve_system_prompt(args.mode, args.system)
    if args.raw_prompt:
        ids = tok.encode(args.prompt, add_special_tokens=False).ids
        bos = tok.token_to_id("<s>")
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
    else:
        ids, _ = _build_chatml_prompt(tok, sys_prompt, args.prompt,
                                      seed_think=not args.no_seed_think)
    print(f"[verify] prompt_tokens={len(ids)} max_tokens={args.max_tokens} "
          f"temp={args.temp} top_p={args.top_p} top_k={args.top_k}",
          file=sys.stderr)

    stop_set: set[int] = set()
    if not args.no_eos_stop:
        for name in ("<|im_end|>", "</s>"):
            tid = tok.token_to_id(name)
            if tid is not None:
                stop_set.add(tid)

    gen_cfg = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temp, top_k=args.top_k,
        top_p=args.top_p, min_p=args.min_p,
        seed=args.seed,
    )

    # ── AR-sampling baseline (optional) ──
    if args.ar_baseline:
        t0 = time.perf_counter()
        ar_res = generate(
            model, ids, gen_cfg,
            stop_token_ids=sorted(stop_set),
            no_eos_stop=args.no_eos_stop,
            full_decode_compile=False,
        )
        t_ar = time.perf_counter() - t0
        ar_tps = len(ar_res.tokens) / max(t_ar, 1e-6)
        print(
            f"[verify] AR-sampling: {len(ar_res.tokens):4d} tok "
            f"in {t_ar*1000:.0f} ms = {ar_tps:6.1f} tok/s",
            file=sys.stderr,
        )
    else:
        ar_tps = None

    Ks = args.K or [4, 8, 12, 16]
    for K in Ks:
        # Warmup K-token graph.
        dummy = mx.zeros((1, K), dtype=mx.int32)
        _l, _s = model(dummy, states=None); mx.eval(_l); del _l, _s, dummy

        t0 = time.perf_counter()
        r = jacobi_decode_sampling(
            model, ids, gen_cfg,
            K=K,
            use_ngram=(not args.no_ngram),
            ngram_n=args.ngram_n,
            use_retrieval=(not args.no_retrieval),
            seed=args.seed,
            stop_token_ids=sorted(stop_set),
            no_eos_stop=args.no_eos_stop,
            verbose=args.verbose,
        )
        dt = time.perf_counter() - t0
        speed_tag = ""
        if ar_tps is not None:
            speed_tag = f"speedup={r.decode_tps / ar_tps:5.2f}x"
        full_pct = 100.0 * r.n_full_accepts / max(r.n_rounds, 1)
        print(
            f"[sjd] K={K:3d} dtype={args.dtype}: "
            f"emitted={len(r.tokens):4d} rounds={r.n_rounds:4d} "
            f"ARL={r.arl:5.2f} full={full_pct:5.1f}% "
            f"decode={r.decode_tps:6.1f} tok/s {speed_tag} "
            f"stop={r.stop_reason}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
