"""CLI entry: Jacobi-decoded generation (mirrors run.py interface).

Usage::

    python -m mamba3_mlx.speculative.run_jacobi \\
        --prompt "Solve 2x+3=11 step by step" \\
        --K 16 --use_ngram --max_tokens 256 --stream
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

from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import _sidecar_path, load_checkpoint
from mamba3_mlx.speculative.drafts import SuffixRetriever
from mamba3_mlx.speculative.jacobi import jacobi_decode
from mamba3_mlx.speculative.ngram_cache import NGramCache
from mamba3_mlx.utils.config import GenerationConfig, Mamba3Config
from mamba3_mlx.utils.system_prompts import MODE_ALIASES, resolve_system_prompt


DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Think step by step before answering."
)


def _build_chatml_prompt(tokenizer: Tokenizer, system_prompt: str,
                         user_msg: str, seed_think: bool = True):
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


def _parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Jacobi (speculative) greedy decoder for Mamba3-Hybrid.",
    )
    p.add_argument(
        "--model_path",
        default=str(REPO_ROOT / "checkpoints" / "v2" / "latest_sft_cot_model.mlx_bf16.npz"),
    )
    p.add_argument(
        "--tokenizer_path",
        default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"),
    )
    p.add_argument("--prompt", default="Who are you?")
    p.add_argument(
        "--mode", default="self_awareness",
        choices=sorted(set(MODE_ALIASES.keys())),
        help="System-prompt category. Overrides --system. "
             "(Matches Makefile default: self_awareness.)",
    )
    p.add_argument("--system", default=_DEFAULT_SYSTEM)

    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--K", type=int, default=8, help="Jacobi window (>= 2)")
    p.add_argument("--use_ngram", action="store_true",
                   help="Enable n-gram seeding of guess positions.")
    p.add_argument("--ngram_n", type=int, default=4,
                   help="Total n-gram length (key = n-1 tokens).")
    p.add_argument("--tree_B", type=int, default=1,
                   help="Tree-of-guesses branch count (>=1). Each round runs "
                        "batch=tree_B verify with n-gram top-N candidates for "
                        "position 1 — pays off only when top-1 frequently "
                        "misses but lower-rank candidates hit.")
    p.add_argument("--use_retrieval", action="store_true",
                   help="Enable suffix-retrieval drafter (PLD / N-Grammys "
                        "context source).")
    p.add_argument("--adaptive_K", action="store_true",
                   help="Enable GammaTune-style adaptive K.")
    p.add_argument("--K_min", type=int, default=4)
    p.add_argument("--K_max", type=int, default=24)

    p.add_argument("--compile_verify", action="store_true",
                   help="mx.compile the per-Mamba-layer verify step. "
                        "Fuses ~50 ops/layer into one Metal dispatch per "
                        "layer.  fp32 byte-equal to the uncompiled path.")

    p.add_argument(
        "--cot_caches",
        default=str(Path(__file__).parent / "cot_caches_v2.pkl"),
        help="Path to baked COT-cache pkl (v1 or v2 layout). "
             "Pass 'none' to disable.",
    )
    p.add_argument(
        "--cot_bucket", default=None,
        help="COT bucket name for per-bucket retriever + ngram. "
             "Defaults to --mode when not set.",
    )

    p.add_argument(
        "--runtime_cache",
        default=str(Path(__file__).parent / "demo_cache_v2.pkl"),
        help="Path to baked runtime cache (NGramCache + SuffixRetriever) "
             "from bake_cache.py.  Pre-warms the retriever so the first "
             "verify round already has long-suffix matches.  Pass 'none' "
             "to disable.",
    )

    p.add_argument("--no-eos-stop", action="store_true")
    p.add_argument(
        "--stop-string", action="append", dest="stop_strings",
        metavar="TEXT", default=[],
        help="Decoded-text stop trigger. Repeatable.",
    )

    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP))
    p.add_argument("--stream", action="store_true",
                   help="Stream tokens to stdout as they are emitted.")
    p.add_argument("--show-special", action="store_true")
    p.add_argument("--no-seed-think", action="store_true")
    p.add_argument("--raw-prompt", action="store_true",
                   help="Use --prompt verbatim (no ChatML wrapping).")
    p.add_argument("--verbose", action="store_true",
                   help="Per-round acceptance diagnostics.")
    return p.parse_args()


def main():
    args = _parse_args()
    dtype = DTYPE_MAP[args.dtype]

    print(f"[load] tokenizer: {args.tokenizer_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)

    print(f"[load] model:     {args.model_path}", file=sys.stderr)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)

    t0 = time.time()
    sidecar = _sidecar_path(args.model_path, dtype)
    if sidecar.exists():
        print(f"[load] sidecar:   {sidecar.name}  (mmap instant)",
              file=sys.stderr)
    else:
        print(f"[load] converting → {sidecar.name}", file=sys.stderr)
    load_checkpoint(model, args.model_path, dtype=dtype)
    mx.eval(model.parameters())
    print(
        f"[load] weights done in {time.time() - t0:.2f}s  (dtype={args.dtype})",
        file=sys.stderr,
    )

    # ── Warmup verify graphs: K_min, K, and K_max under adaptive_K ──────────
    t_w = time.time()
    warmup_Ls = {args.K, 1}
    if args.adaptive_K:
        warmup_Ls |= {args.K_min, args.K_max}
    for L in sorted(warmup_Ls):
        dummy = mx.zeros((1, L), dtype=mx.int32)
        _l, _s = model(dummy, states=None)
        mx.eval(_l)
        del _l, _s, dummy
    print(
        f"[warmup] Ls={sorted(warmup_Ls)} JIT done in {time.time() - t_w:.2f}s",
        file=sys.stderr,
    )

    sys_prompt = resolve_system_prompt(args.mode, args.system)
    if args.mode:
        print(f"[mode] {args.mode} → {sys_prompt[:70]}…", file=sys.stderr)

    # COT cache: resolve path + bucket (bucket defaults to --mode).
    cot_caches_arg = None
    cot_bucket_arg = args.cot_bucket or args.mode or None
    if args.cot_caches and args.cot_caches.lower() != "none":
        p_cot = Path(args.cot_caches)
        if p_cot.exists():
            cot_caches_arg = str(p_cot)
            print(f"[cot] {p_cot.name}  bucket={cot_bucket_arg}", file=sys.stderr)
        else:
            print(f"[cot] WARNING: {p_cot} not found — cot cache disabled",
                  file=sys.stderr)

    # Runtime cache: optional pre-warmed NGramCache + SuffixRetriever from
    # bake_cache.py.  Skips the cold-start period where the retriever has no
    # history yet — biggest single ARL bump on structured outputs.
    preloaded_ngram_obj = None
    preloaded_retriever_obj = None
    if args.runtime_cache and args.runtime_cache.lower() != "none":
        p_rt = Path(args.runtime_cache)
        if p_rt.exists():
            import pickle
            with open(p_rt, "rb") as f:
                _rt_state = pickle.load(f)
            preloaded_ngram_obj = NGramCache.from_state(_rt_state["ngram"])
            preloaded_retriever_obj = SuffixRetriever.from_state(_rt_state["retriever"])
            print(
                f"[runtime] {p_rt.name}  "
                f"ngrams={len(preloaded_ngram_obj)} "
                f"retriever_buf={len(preloaded_retriever_obj.buf)}",
                file=sys.stderr,
            )
        else:
            print(f"[runtime] WARNING: {p_rt} not found — cold cache",
                  file=sys.stderr)

    if args.raw_prompt:
        prompt_text = args.prompt
        ids = tok.encode(prompt_text, add_special_tokens=False).ids
        bos = tok.token_to_id("<s>")
        if bos is not None:
            ids = [bos] + ids
    else:
        ids, prompt_text = _build_chatml_prompt(
            tok, sys_prompt, args.prompt,
            seed_think=not args.no_seed_think,
        )

    print("===== PROMPT =====", file=sys.stderr)
    print(prompt_text, file=sys.stderr)
    print(f"===== PROMPT TOKENS: {len(ids)} =====", file=sys.stderr)
    print(
        f"[jacobi] K={args.K}  use_ngram={args.use_ngram}  "
        f"ngram_n={args.ngram_n}",
        file=sys.stderr,
    )

    gen_cfg = GenerationConfig(max_tokens=args.max_tokens)

    stop_ids: list[int] = []
    if not args.no_eos_stop:
        for name in ("<|im_end|>", "</s>"):
            tid = tok.token_to_id(name)
            if tid is not None:
                stop_ids.append(tid)

    # ── Streaming callback ──
    on_token = None
    if args.stream:
        seen: list[int] = []
        prev_len = [0]
        skip_special = not args.show_special

        def _on_token(tid: int) -> None:
            seen.append(tid)
            text_full = tok.decode(seen, skip_special_tokens=skip_special)
            new = text_full[prev_len[0]:]
            prev_len[0] = len(text_full)
            if new:
                print(new, end="", flush=True)

        on_token = _on_token

    result = jacobi_decode(
        model, ids, gen_cfg,
        K=args.K,
        use_ngram=args.use_ngram,
        ngram_n=args.ngram_n,
        tree_B=args.tree_B,
        use_retrieval=args.use_retrieval,
        adaptive_K=args.adaptive_K,
        K_min=args.K_min, K_max=args.K_max,
        compile_verify=args.compile_verify,
        cot_caches=cot_caches_arg,
        cot_bucket=cot_bucket_arg,
        preloaded_ngram=preloaded_ngram_obj,
        preloaded_retriever=preloaded_retriever_obj,
        stop_token_ids=stop_ids,
        stop_strings=args.stop_strings or [],
        no_eos_stop=args.no_eos_stop,
        tokenizer=tok if args.stop_strings else None,
        on_token=on_token,
        verbose=args.verbose,
    )

    if args.stream:
        print()  # newline after streamed content

    branch_tag = (f" branch_wins={result.branch_wins}"
                  if result.tree_B > 1 else "")
    print(
        "===== "
        f"prefill {result.prefill_tps:,.0f} tok/s "
        f"({result.n_prompt} tok, {result.elapsed_prefill*1000:.0f} ms)  |  "
        f"decode  {result.decode_tps:,.0f} tok/s "
        f"({len(result.tokens)} tok, {result.elapsed_decode*1000:.0f} ms)  |  "
        f"rounds={result.n_rounds} ARL={result.arl:.2f} "
        f"ngram_hits={result.n_ngram_hits} K={result.K} "
        f"tree_B={result.tree_B}{branch_tag}  |  "
        f"stop={result.stop_reason}"
        " =====",
        file=sys.stderr,
    )

    if not args.stream:
        full_text = tok.decode(result.tokens,
                               skip_special_tokens=not args.show_special)
        print("===== ASSISTANT OUTPUT =====")
        if not args.no_seed_think:
            print("<think>")
        print(full_text)


if __name__ == "__main__":
    main()
