"""Offline cache baker: run AR-sampling for N tokens on a set of prompts and
write the resulting n-gram + suffix-retrieval caches to a single ``.pkl``
file.  The demo launcher loads this file at startup (instant) and runs SJD
with pre-warm caches — zero wrap-up time at run-time.

Run once after a checkpoint change:

    python -m mamba3_mlx.speculative.bake_cache \
        --warmup_tokens 4096 --out caches.pkl

Then point the demo at it:

    python -m mamba3_mlx.speculative.run_sjd_demo \
        --cache caches.pkl --K 16 --max_tokens 512 --ar-baseline
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
from mamba3_mlx.speculative.ngram_cache import NGramCache
from mamba3_mlx.utils.config import GenerationConfig, Mamba3Config
from mamba3_mlx.utils.system_prompts import resolve_system_prompt


def _chatml_ids(tok, sys_prompt, user_msg, seed_think=True):
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
    p.add_argument(
        "--model_path",
        default=str(REPO_ROOT / "checkpoints" / "latest_sft_cot_model.npz"))
    p.add_argument(
        "--tokenizer_path",
        default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"))
    p.add_argument("--out",
                   default=str(REPO_ROOT / "mamba3_mlx" / "speculative" /
                               "demo_cache.pkl"))
    p.add_argument("--warmup_tokens", type=int, default=4096,
                   help="Total tokens of AR-sampling output to ingest "
                        "across all baking prompts.")
    p.add_argument("--temp", type=float, default=0.15)
    p.add_argument("--top_p", type=float, default=0.85)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--min_p", type=float, default=0.08)
    p.add_argument("--ngram_n", type=int, default=4)
    p.add_argument("--retrieval_max_suffix", type=int, default=8)
    p.add_argument("--retrieval_max_window", type=int, default=8192)
    p.add_argument("--dtype", default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    # Baking prompts: diverse lifestyle topics so the cache covers the
    # common ChatML / self-awareness / chat structure tokens.
    p.add_argument(
        "--prompts", nargs="*",
        default=[
            ("Who are you?", "self_awareness"),
            ("Tell me about yourself", "self_awareness"),
            ("What is your name?", "self_awareness"),
            ("Hello, how can you help me today?", "daily_conversation"),
        ],
        help="Pairs of 'prompt mode'.  If not given, the four defaults are used.",
    )
    args = p.parse_args()

    DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    dtype = DTYPE_MAP[args.dtype]

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"[bake] loading {args.model_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    sidecar = _sidecar_path(args.model_path, dtype)
    if sidecar.exists():
        print(f"[bake] sidecar: {sidecar.name}", file=sys.stderr)
    load_checkpoint(model, args.model_path, dtype=dtype)
    mx.eval(model.parameters())

    # ── Allocate cache objects ─────────────────────────────────────────────
    ngram = NGramCache(n=args.ngram_n)
    retriever = SuffixRetriever(
        max_window=args.retrieval_max_window,
        max_suffix=args.retrieval_max_suffix,
    )

    # ── Normalise prompts arg ──────────────────────────────────────────────
    # ``--prompts`` comes in as a flat list of strings → pair adjacent ones.
    raw = args.prompts if args.prompts else []
    if isinstance(raw, list) and raw and isinstance(raw[0], tuple):
        pairs = raw
    else:
        pairs = []
        it = iter(raw)
        for prompt in it:
            mode = next(it, "self_awareness")
            pairs.append((prompt, mode))

    tokens_per_prompt = max(args.warmup_tokens // max(len(pairs), 1), 64)
    print(f"[bake] {len(pairs)} prompt(s), ~{tokens_per_prompt} tok each "
          f"= {tokens_per_prompt * len(pairs)} total",
          file=sys.stderr)

    # ── Generate + ingest ──────────────────────────────────────────────────
    t_total0 = time.perf_counter()
    for i, (prompt, mode) in enumerate(pairs):
        sys_prompt = resolve_system_prompt(mode, None)
        ids = _chatml_ids(tok, sys_prompt, prompt)
        gen_cfg = GenerationConfig(
            max_tokens=tokens_per_prompt,
            temperature=args.temp, top_k=args.top_k,
            top_p=args.top_p, min_p=args.min_p,
            seed=i,
        )
        t0 = time.perf_counter()
        out = generate(model, ids, gen_cfg, stop_token_ids=[], no_eos_stop=True)
        # Ingest the WHOLE sequence (prompt + generated) into both caches.
        seq = list(ids) + list(out.tokens)
        ngram.update_sequence(seq)
        retriever.extend(seq)
        print(
            f"[bake] {i+1}/{len(pairs)} '{prompt[:40]}…' → "
            f"{len(out.tokens)} tok in {time.perf_counter() - t0:.1f}s",
            file=sys.stderr,
        )
    print(f"[bake] total time: {time.perf_counter() - t_total0:.1f}s "
          f"(this is offline — once per checkpoint)", file=sys.stderr)

    # ── Save ───────────────────────────────────────────────────────────────
    state = {
        "ngram": ngram.to_state(),
        "retriever": retriever.to_state(),
        "args": {
            "model_path": str(args.model_path),
            "warmup_tokens": args.warmup_tokens,
            "ngram_n": args.ngram_n,
            "temp": args.temp,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(state, f)
    print(f"[bake] wrote {out_path}  "
          f"({out_path.stat().st_size / 1024:.1f} KB, "
          f"{len(ngram)} ngrams, {len(retriever.buf)} tokens in retriever)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
