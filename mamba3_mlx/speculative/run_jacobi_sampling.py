"""CLI runner for jacobi_decode_sampling (SJD).

Two modes:
  • --stream : single generation, tokens printed to stdout as emitted
               (rich color-coded: green=draft hit, yellow=bonus/resample)
  • default  : K-sweep benchmark (multi-K, ARL/tps diagnostics)

Usage:
  # Streaming single generation
  python -m mamba3_mlx.speculative.run_jacobi_sampling \\
      --stream --K 8 --prompt "Who are you?" --max_tokens 512

  # With COT cache + bucket
  python -m mamba3_mlx.speculative.run_jacobi_sampling \\
      --stream --K 8 --mode math_drill --prompt "Solve 2x+3=11"
      --cot_caches mamba3_mlx/speculative/cot_caches_n4.pkl
      --cot_bucket math_drill
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

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

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.table import Table
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


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
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="SJD (Speculative Jacobi Decoding) — sampling mode.",
    )
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
    p.add_argument("--K", type=int, default=8,
                   help="Jacobi window size (≥ 2).")
    p.add_argument("--Ks", action="append", type=int, default=None,
                   help="K values to sweep (benchmark mode, ignores --K).")
    p.add_argument("--temp", type=float, default=0.15)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_p", type=float, default=0.85)
    p.add_argument("--min_p", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP))
    p.add_argument("--no_ngram", action="store_true")
    p.add_argument("--no_retrieval", action="store_true")
    p.add_argument("--ngram_n", type=int, default=4)
    p.add_argument("--cot_caches", type=str, default=None,
                   help="Path to COT cache pkl (cot_caches_v2.pkl).")
    p.add_argument("--cot_bucket", type=str, default=None,
                   help="Bucket name for per-category retrieval (e.g. math_drill).")
    p.add_argument(
        "--runtime_cache",
        default=str(Path(__file__).parent / "demo_cache_v2.pkl"),
        help="Path to baked runtime cache (NGramCache + SuffixRetriever). "
             "Pre-warms the retriever from AR-sampling runs on representative "
             "prompts.  Pass 'none' to disable.",
    )
    p.add_argument("--compile_verify", action="store_true",
                   help="mx.compile per-Mamba-layer verify step. "
                        "fp32 byte-equal; cuts ~50ms/round dispatch overhead.")
    p.add_argument("--no-eos-stop", action="store_true")
    p.add_argument("--ar-baseline", action="store_true",
                   help="Also run AR-sampling baseline for wall-clock comparison.")
    p.add_argument("--stream", action="store_true",
                   help="Stream tokens to stdout as they are emitted.")
    p.add_argument("--show-special", action="store_true",
                   help="Include special tokens in streamed output.")
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
        prompt_text = args.prompt
        ids = tok.encode(prompt_text, add_special_tokens=False).ids
        bos = tok.token_to_id("<s>")
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
    else:
        ids, prompt_text = _build_chatml_prompt(tok, sys_prompt, args.prompt,
                                                seed_think=not args.no_seed_think)
    print(f"[sjd] prompt_tokens={len(ids)} max_tokens={args.max_tokens} "
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

    # Resolve COT caches once
    cot_caches_path: Optional[str] = None
    if args.cot_caches:
        p = Path(args.cot_caches)
        cot_caches_path = str(p.resolve())
        if not p.exists():
            print(f"[sjd] WARNING: COT cache not found: {cot_caches_path}",
                  file=sys.stderr)
            cot_caches_path = None

    # Resolve runtime cache once (NGramCache + SuffixRetriever).
    preloaded_ngram_obj = None
    preloaded_retriever_obj = None
    if args.runtime_cache and args.runtime_cache.lower() != "none":
        p_rt = Path(args.runtime_cache)
        if p_rt.exists():
            import pickle
            from mamba3_mlx.speculative.drafts import SuffixRetriever
            from mamba3_mlx.speculative.ngram_cache import NGramCache
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

    # ── AR-sampling baseline (optional) ──
    ar_tps: Optional[float] = None
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
            f"[sjd] AR-sampling: {len(ar_res.tokens):4d} tok "
            f"in {t_ar*1000:.0f} ms = {ar_tps:6.1f} tok/s",
            file=sys.stderr,
        )

    # ── Streaming callback ──
    on_token = None
    on_emit = None
    console = Console(file=sys.stderr) if _HAS_RICH else None

    if args.stream:
        seen: list[int] = []
        prev_len = [0]
        skip_special = not args.show_special

        if _HAS_RICH:
            draft_style = "bold green"
            other_style = "yellow"
            out_console = Console(file=sys.stdout, force_terminal=True)

            def _on_emit(tid: int, is_draft: bool) -> None:
                seen.append(tid)
                text_full = tok.decode(seen, skip_special_tokens=skip_special)
                new = text_full[prev_len[0]:]
                prev_len[0] = len(text_full)
                if new:
                    style = draft_style if is_draft else other_style
                    out_console.print(new, end="", style=style)
        else:
            def _on_token(tid: int) -> None:
                seen.append(tid)
                text_full = tok.decode(seen, skip_special_tokens=skip_special)
                new = text_full[prev_len[0]:]
                prev_len[0] = len(text_full)
                if new:
                    print(new, end="", flush=True)

            on_token = _on_token
        on_emit = _on_emit if _HAS_RICH else None

    if args.stream:
        if not args.no_seed_think:
            print("<think>", flush=True)

    # ── Run SJD ──
    if args.Ks:
        Ks = sorted(set(args.Ks))
    else:
        Ks = [args.K]

    for i, K in enumerate(Ks):
        warmup = mx.zeros((1, K), dtype=mx.int32)
        _l, _s = model(warmup, states=None)
        mx.eval(_l)
        del _l, _s, warmup

        t0 = time.perf_counter()
        r = jacobi_decode_sampling(
            model, ids, gen_cfg,
            K=K,
            use_ngram=(not args.no_ngram),
            ngram_n=args.ngram_n,
            use_retrieval=(not args.no_retrieval),
            seed=args.seed + (i * 999 if len(Ks) > 1 else 0),
            stop_token_ids=sorted(stop_set),
            no_eos_stop=args.no_eos_stop,
            cot_caches=cot_caches_path,
            cot_bucket=args.cot_bucket,
            preloaded_ngram=preloaded_ngram_obj,
            preloaded_retriever=preloaded_retriever_obj,
            compile_verify=args.compile_verify,
            on_token=on_token,
            on_emit=on_emit,
            verbose=args.verbose,
        )
        dt = time.perf_counter() - t0

        if args.stream:
            print(flush=True)

        # ── Stats display ──
        speed_tag = ""
        if ar_tps is not None:
            speed_tag = f"speedup={r.decode_tps / ar_tps:5.2f}x"
        full_pct = 100.0 * r.n_full_accepts / max(r.n_rounds, 1)

        if _HAS_RICH and args.stream:
            table = Table(title="SJD Decode Stats", title_style="bold")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Tokens", str(len(r.tokens)))
            table.add_row("Rounds", str(r.n_rounds))
            table.add_row("ARL", f"{r.arl:.2f}")
            table.add_row("Full accepts", f"{full_pct:.1f}%")
            table.add_row("Decode tps", f"{r.decode_tps:.1f} tok/s")
            if ar_tps is not None:
                table.add_row("Speedup", f"{r.decode_tps / ar_tps:.2f}×")
            table.add_row("Wall time", f"{dt:.1f}s")
            table.add_row("Stop reason", r.stop_reason)
            console.print(table)
        else:
            print(
                f"[sjd] K={K:3d} dtype={args.dtype}: "
                f"emitted={len(r.tokens):4d} rounds={r.n_rounds:4d} "
                f"ARL={r.arl:5.2f} full={full_pct:5.1f}% "
                f"decode={r.decode_tps:6.1f} tok/s {speed_tag} "
                f"stop={r.stop_reason}",
                file=sys.stderr,
            )
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

        # Print full output in non-stream mode (last K only if sweep)
        if not args.stream and (len(Ks) == 1 or K == Ks[-1]):
            full_text = tok.decode(r.tokens,
                                   skip_special_tokens=not args.show_special)
            if not args.no_seed_think:
                print("<think>")
            print(full_text)


if __name__ == "__main__":
    main()
