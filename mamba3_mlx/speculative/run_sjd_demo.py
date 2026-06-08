"""Demo runner: load pre-baked draft caches (n-gram + suffix retrieval) from
``bake_cache.py`` output, then time SJD at small ``max_tokens`` against an
AR-sampling baseline.  No untimed wrap-up — caches are loaded in ms from
the .pkl file.
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

from mamba3_mlx.inference.generator import generate
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import _sidecar_path, load_checkpoint
from mamba3_mlx.speculative.drafts import SuffixRetriever
from mamba3_mlx.speculative.jacobi_sampling import jacobi_decode_sampling
from mamba3_mlx.speculative.ngram_cache import NGramCache
from mamba3_mlx.utils.config import GenerationConfig, Mamba3Config
from mamba3_mlx.utils.system_prompts import MODE_ALIASES, resolve_system_prompt


def _chatml_ids(tok, sys_prompt, user_msg, seed_think=False):
    text = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        + ("<think>\n" if seed_think else "")
    )
    ids = tok.encode(text, add_special_tokens=False).ids
    bos = tok.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",
                   default=str(REPO_ROOT / "checkpoints" / "v2" / "latest_sft_cot_model.mlx_bf16.npz"))
    p.add_argument("--tokenizer_path",
                   default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"))
    p.add_argument("--cache",
                   default=str(REPO_ROOT / "mamba3_mlx" / "speculative" /
                               "demo_cache_v2.pkl"))
    p.add_argument("--prompt", default="Who are you?")
    p.add_argument("--mode", default="self_awareness",
                   choices=sorted(set(MODE_ALIASES.keys())))
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--K", action="append", type=int, default=None,
                   help="K values to test. Default: [12, 16, 20, 24].")
    p.add_argument("--temp", type=float, default=0.15)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_p", type=float, default=0.85)
    p.add_argument("--min_p", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--ngram_n", type=int, default=4)
    p.add_argument("--no-eos-stop", action="store_true", default=False)
    p.add_argument("--no_cache", action="store_true",
                   help="Skip loading the baked cache (cold-start baseline).")
    p.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8],
                   help="Quantize nn.Linear layers to N bits (0=disabled, 4=4-bit, 8=8-bit).")
    p.add_argument("--compile_verify", action="store_true",
                   help="Compile Mamba+Transformer verify steps via mx.compile.")
    p.add_argument("--cot_caches",
                   default=str(REPO_ROOT / "mamba3_mlx" / "speculative" /
                               "cot_caches_v2.pkl"),
                   help="Path to cot_caches pkl (think/final n-gram bundles). "
                        "Pass 'none' to disable.")
    p.add_argument("--cot_bucket", default=None,
                   help="Override cot cache bucket (think/final/other). "
                        "Default: auto-detect from mode.")
    args = p.parse_args()

    DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    dtype = DTYPE_MAP[args.dtype]

    print(f"[demo] loading {args.model_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    sidecar = _sidecar_path(args.model_path, dtype)
    if sidecar.exists():
        print(f"[demo] sidecar: {sidecar.name}", file=sys.stderr)
    load_checkpoint(model, args.model_path, dtype=dtype)
    mx.eval(model.parameters())

    if args.quantize > 0:
        import mlx.nn as mlx_nn
        print(f"[demo] quantizing to {args.quantize}-bit…", file=sys.stderr)
        mlx_nn.quantize(model, group_size=64, bits=args.quantize)
        mx.eval(model.parameters())
        print(f"[demo] quantization done", file=sys.stderr)

    model.precompute()
    print(f"[demo] precompute done (compiled decode + verify paths active)", file=sys.stderr)

    # ── Load baked caches ──────────────────────────────────────────────────
    preload_ngram = None
    preload_retriever = None
    if not args.no_cache:
        cache_path = Path(args.cache)
        if not cache_path.exists():
            print(f"[demo] !! cache file not found: {cache_path}",
                  file=sys.stderr)
            print(f"[demo] !! run bake_cache.py first or pass --no_cache",
                  file=sys.stderr)
            sys.exit(1)
        t0 = time.perf_counter()
        with open(cache_path, "rb") as f:
            state = pickle.load(f)
        preload_ngram = NGramCache.from_state(state["ngram"])
        preload_retriever = SuffixRetriever.from_state(state["retriever"])
        print(
            f"[demo] cache loaded in {(time.perf_counter() - t0)*1000:.1f} ms "
            f"({len(preload_ngram)} ngrams, "
            f"{len(preload_retriever.buf)} retriever tokens)",
            file=sys.stderr,
        )

    # ── Resolve cot_caches path ───────────────────────────────────────────────
    cot_caches_arg = None
    if args.cot_caches.lower() != "none":
        cot_path = Path(args.cot_caches)
        if cot_path.exists():
            cot_caches_arg = str(cot_path)
            print(f"[demo] cot_caches: {cot_path.name}", file=sys.stderr)
        else:
            print(f"[demo] !! cot_caches not found: {cot_path} — disabling",
                  file=sys.stderr)
    cot_bucket_arg = args.cot_bucket or args.mode

    sys_prompt = resolve_system_prompt(args.mode, None)
    ids = _chatml_ids(tok, sys_prompt, args.prompt)
    print(f"[demo] prompt='{args.prompt}'  max_tokens={args.max_tokens}",
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

    # ── AR-sampling baseline (cold) ────────────────────────────────────────
    t0 = time.perf_counter()
    ar = generate(model, ids, gen_cfg,
                  stop_token_ids=sorted(stop_set),
                  no_eos_stop=args.no_eos_stop)
    t_ar = time.perf_counter() - t0
    ar_tps = (len(ar.tokens) - ar.n_warmup) / max(ar.elapsed_decode, 1e-6)
    print(
        f"[demo] AR-sampling: {len(ar.tokens):4d} tok in "
        f"{t_ar*1000:.0f} ms = {ar_tps:6.1f} tok/s",
        file=sys.stderr,
    )

    # ── SJD across K values ────────────────────────────────────────────────
    Ks = args.K or [12, 16, 20, 24]
    best = None
    for K in Ks:
        # JIT warmup: a short run with seed+999 to pay compile costs only.
        # Does NOT share the timed seed so the ngram cache stays cold — this
        # gives honest throughput numbers without oracle pre-warming.
        dummy = mx.zeros((1, K), dtype=mx.int32)
        _l, _s = model(dummy, states=None); mx.eval(_l); del _l, _s, dummy
        if preload_ngram is not None:
            warm_ng = NGramCache.from_state(preload_ngram.to_state())
            warm_rt = SuffixRetriever.from_state(preload_retriever.to_state())
        else:
            warm_ng, warm_rt = None, None
        warm_cfg = GenerationConfig(
            max_tokens=32, temperature=args.temp,
            top_k=args.top_k, top_p=args.top_p, min_p=args.min_p,
            seed=args.seed + 999,
        )
        _ = jacobi_decode_sampling(
            model, ids, warm_cfg, K=K,
            use_ngram=True, ngram_n=args.ngram_n, use_retrieval=True,
            preloaded_ngram=warm_ng, preloaded_retriever=warm_rt,
            cot_caches=cot_caches_arg, cot_bucket=cot_bucket_arg,
            seed=args.seed + 999,
            stop_token_ids=[], no_eos_stop=True,
            compile_verify=args.compile_verify,
        )

        # Clone fresh caches for the timed run (independent of warmup path).
        if preload_ngram is not None:
            ng_for_K = NGramCache.from_state(preload_ngram.to_state())
        else:
            ng_for_K = None
        if preload_retriever is not None:
            rt_for_K = SuffixRetriever.from_state(preload_retriever.to_state())
        else:
            rt_for_K = None

        t0 = time.perf_counter()
        r = jacobi_decode_sampling(
            model, ids, gen_cfg,
            K=K,
            use_ngram=True, ngram_n=args.ngram_n,
            use_retrieval=True,
            preloaded_ngram=ng_for_K,
            preloaded_retriever=rt_for_K,
            cot_caches=cot_caches_arg, cot_bucket=cot_bucket_arg,
            seed=args.seed,
            stop_token_ids=sorted(stop_set),
            no_eos_stop=args.no_eos_stop,
            compile_verify=args.compile_verify,
        )
        dt = time.perf_counter() - t0
        speedup = r.decode_tps / max(ar_tps, 1e-6)
        full_pct = 100.0 * r.n_full_accepts / max(r.n_rounds, 1)
        tag = "★" if (best is None or speedup > best[1]) else " "
        print(
            f"[demo] {tag} K={K:3d}: emitted={len(r.tokens):4d} "
            f"rounds={r.n_rounds:4d} ARL={r.arl:5.2f} "
            f"full={full_pct:5.1f}% decode={r.decode_tps:6.1f} tok/s "
            f"speedup={speedup:5.2f}x  ({dt:.1f}s wall)",
            file=sys.stderr,
        )
        if best is None or speedup > best[1]:
            best = (K, speedup, r.arl, r.decode_tps)

    print(file=sys.stderr)
    print(
        f"[demo] BEST: K={best[0]}  speedup={best[1]:.2f}x  "
        f"ARL={best[2]:.2f}  decode={best[3]:.1f} tok/s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
