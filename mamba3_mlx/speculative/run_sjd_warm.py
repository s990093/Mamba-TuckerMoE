"""SJD with warm-cache benchmark: pre-warm n-gram + retrieval caches with
a longer dry-run, then time only the next ``--max_tokens`` of generation
under the warm caches.  This isolates the "steady-state" SJD speed
that a user sees in a multi-turn / long-running session, instead of
the cold-cache start-up that dominates a one-shot max=256 run.

The warm-up does **not** count toward the speedup measurement.
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

from mamba3_mlx.inference.generator import (
    _iter_state_arrays, generate, prefill,
)
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import _sidecar_path, load_checkpoint
from mamba3_mlx.speculative.drafts import SuffixRetriever
from mamba3_mlx.speculative.jacobi_sampling import jacobi_decode_sampling
from mamba3_mlx.speculative.ngram_cache import NGramCache
from mamba3_mlx.utils.config import GenerationConfig, Mamba3Config
from mamba3_mlx.utils.system_prompts import MODE_ALIASES, resolve_system_prompt


DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}


def _chatml_ids(tok: Tokenizer, sys_prompt: str, user_msg: str,
                seed_think: bool):
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
    p.add_argument("--system", default=None)
    p.add_argument("--no-seed-think", action="store_true")
    p.add_argument("--warmup_tokens", type=int, default=1024,
                   help="Generate this many tokens UNTIMED to warm the "
                        "n-gram and suffix-retrieval caches.")
    p.add_argument("--max_tokens", type=int, default=256,
                   help="Tokens to TIME after warmup.")
    p.add_argument("--K", action="append", type=int, default=None,
                   help="K values to test. Default: [16, 20, 24, 32].")
    p.add_argument("--temp", type=float, default=0.15)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_p", type=float, default=0.85)
    p.add_argument("--min_p", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP))
    p.add_argument("--ngram_n", type=int, default=4)
    p.add_argument("--retrieval_max_suffix", type=int, default=8)
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
    ids, _ = _chatml_ids(tok, sys_prompt, args.prompt,
                         seed_think=not args.no_seed_think)
    print(f"[bench] prompt_tokens={len(ids)} warmup={args.warmup_tokens} "
          f"timed_max_tokens={args.max_tokens}", file=sys.stderr)

    stop_set: set[int] = set()      # ignore EOS during bench
    gen_cfg_warm = GenerationConfig(
        max_tokens=args.warmup_tokens,
        temperature=args.temp, top_k=args.top_k,
        top_p=args.top_p, min_p=args.min_p,
        seed=args.seed,
    )
    gen_cfg_timed = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temp, top_k=args.top_k,
        top_p=args.top_p, min_p=args.min_p,
        seed=args.seed + 17,    # different RNG for the timed phase
    )

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: untimed warmup — run AR-sampling for warmup_tokens, build the
    # caches from the produced text.  This emulates the steady-state where
    # the user has already had a long conversation.
    # ──────────────────────────────────────────────────────────────────────
    print(f"[warm] running AR-sampling for {args.warmup_tokens} tokens...",
          file=sys.stderr)
    t0 = time.perf_counter()
    warm = generate(model, ids, gen_cfg_warm,
                    stop_token_ids=[], no_eos_stop=True)
    print(f"[warm] done in {time.perf_counter() - t0:.1f}s "
          f"({len(warm.tokens)} tok)", file=sys.stderr)
    warm_seq = list(ids) + warm.tokens

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: time AR-sampling baseline on the COLD prompt (no KV growth).
    # This is the fair comparison: AR doesn't get cache help; SJD only gets
    # its (free, data-only) n-gram + retrieval cache pre-populated.
    # ──────────────────────────────────────────────────────────────────────
    print("[bench] AR-sampling baseline (cold, timed)...", file=sys.stderr)
    t0 = time.perf_counter()
    ar = generate(model, ids, gen_cfg_timed,
                  stop_token_ids=[], no_eos_stop=True)
    t_ar = time.perf_counter() - t0
    ar_tps = (len(ar.tokens) - ar.n_warmup) / max(ar.elapsed_decode, 1e-6)
    print(
        f"[bench] AR-sampling: {len(ar.tokens):4d} tok in {t_ar*1000:.0f} ms "
        f"= {ar_tps:6.1f} tok/s (prefill={ar.elapsed_prefill*1000:.0f}ms, "
        f"decode={ar.elapsed_decode*1000:.0f}ms)",
        file=sys.stderr,
    )

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: time SJD across K values starting from the warmed context.
    # ──────────────────────────────────────────────────────────────────────
    Ks = args.K or [16, 20, 24, 32]
    best = None
    for K in Ks:
        # Warmup the K-token graph.
        dummy = mx.zeros((1, K), dtype=mx.int32)
        _l, _s = model(dummy, states=None); mx.eval(_l); del _l, _s, dummy

        # SJD runs on the SAME cold prompt as the AR baseline, but with
        # n-gram + retrieval caches pre-populated from the warmup output.
        # The model's KV cache and Mamba state start from scratch — only
        # the (data-only) draft caches are warm.
        t0 = time.perf_counter()
        r = jacobi_decode_sampling(
            model, ids, gen_cfg_timed,            # cold prompt, no KV warmup
            K=K,
            use_ngram=True, ngram_n=args.ngram_n,
            use_retrieval=True,
            retrieval_max_suffix=args.retrieval_max_suffix,
            cache_warmup_tokens=warm.tokens,      # pre-populate draft caches
            seed=args.seed + 17,
            stop_token_ids=[],
            no_eos_stop=True,
        )
        dt = time.perf_counter() - t0
        speedup = r.decode_tps / max(ar_tps, 1e-6)
        full_pct = 100.0 * r.n_full_accepts / max(r.n_rounds, 1)
        tag = "★" if (best is None or speedup > best[1]) else " "
        print(
            f"[bench] {tag} K={K:3d} dtype={args.dtype}: "
            f"emitted={len(r.tokens):4d} rounds={r.n_rounds:4d} "
            f"ARL={r.arl:5.2f} full={full_pct:5.1f}% "
            f"decode={r.decode_tps:6.1f} tok/s "
            f"speedup={speedup:5.2f}x",
            file=sys.stderr,
        )
        if best is None or speedup > best[1]:
            best = (K, speedup, r.arl, r.decode_tps)

    print(file=sys.stderr)
    print(f"[bench] BEST: K={best[0]}  speedup={best[1]:.2f}x  "
          f"ARL={best[2]:.2f}  decode={best[3]:.1f} tok/s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
