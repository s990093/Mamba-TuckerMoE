"""Benchmark StaticJacobiDecoder vs AR baselines and the existing jacobi_decode.

All greedy (Jacobi requires a deterministic verifier).  Example:

.venv/bin/python3 mamba3_mlx/bench_static_jacobi.py --mode self_awareness \\
    --prompt "Who are you?" --max_tokens 256 --K 8,12,16
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.utils.config import Mamba3Config, GenerationConfig
from mamba3_mlx.utils.system_prompts import MODE_ALIASES, resolve_system_prompt
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint
from mamba3_mlx.mlx_model.static_decode import StaticDecoder
from mamba3_mlx.inference.generator import generate as ref_generate
from mamba3_mlx.speculative.jacobi import jacobi_decode
from mamba3_mlx.speculative.static_jacobi import StaticJacobiDecoder


def chatml_ids(tok, sys_prompt, user_msg, seed_think=True):
    text = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n" + ("<think>\n" if seed_think else "")
    )
    ids = tok.encode(text, add_special_tokens=False).ids
    bos = tok.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",
                   default=str(REPO_ROOT / "checkpoints" / "v6" / "latest_sft_cot_model.npz"))
    p.add_argument("--tokenizer_path",
                   default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"))
    p.add_argument("--prompt", default="Who are you?")
    p.add_argument("--mode", default="self_awareness",
                   choices=sorted(set(MODE_ALIASES.keys())))
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--K", default="8,12,16")
    p.add_argument("--ngram_n", type=int, default=4)
    p.add_argument("--cot_caches",
                   default=str(REPO_ROOT / "mamba3_mlx" / "speculative" / "cot_caches_v2.pkl"))
    p.add_argument("--no-eos-stop", action="store_true")
    p.add_argument("--skip-old-jacobi", action="store_true")
    p.add_argument("--show-text", action="store_true")
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    print(f"[load] {args.model_path}", file=sys.stderr)
    load_checkpoint(model, args.model_path, dtype=mx.bfloat16)
    mx.eval(model.parameters())

    sys_prompt = resolve_system_prompt(args.mode, None)
    prompt_ids = chatml_ids(tok, sys_prompt, args.prompt)
    print(f"[prompt] mode={args.mode} {len(prompt_ids)} tokens", file=sys.stderr)

    stop_ids: list[int] = []
    if not args.no_eos_stop:
        for name in ("<|im_end|>", "</s>"):
            tid = tok.token_to_id(name)
            if tid is not None:
                stop_ids.append(tid)

    cot_path = Path(args.cot_caches)
    cot_arg = str(cot_path) if cot_path.exists() else None
    if cot_arg:
        print(f"[cot] {cot_path.name} bucket={args.mode}", file=sys.stderr)

    greedy = GenerationConfig(max_tokens=args.max_tokens, temperature=0.0,
                              rep_pen=1.0, pres_pen=0.0, freq_pen=0.0, seed=0)
    rows = []

    # 1) AR reference (per-block compiled path)
    print("[bench] AR reference …", file=sys.stderr)
    r = ref_generate(model, prompt_ids, greedy, stop_token_ids=stop_ids,
                     no_eos_stop=args.no_eos_stop)
    ar_tokens = r.tokens
    ar_tps = r.decode_tps
    rows.append(("AR reference", len(r.tokens), r.decode_tps, "-", "-"))
    print(f"[bench] AR reference: {r.decode_tps:6.1f} tok/s ({len(r.tokens)} tok)",
          file=sys.stderr)

    # 2) StaticDecoder greedy AR
    sd = StaticDecoder(model)
    r = sd.generate(prompt_ids, greedy, stop_token_ids=tuple(stop_ids), unroll=1)
    rows.append(("static AR", len(r.tokens), r.decode_tps, "-", "-"))
    print(f"[bench] static AR:    {r.decode_tps:6.1f} tok/s ({len(r.tokens)} tok)",
          file=sys.stderr)

    Ks = [int(x) for x in args.K.split(",") if x.strip()]

    # 3) existing jacobi_decode (eager verify), strongest prior path
    if not args.skip_old_jacobi:
        for K in Ks:
            r = jacobi_decode(model, prompt_ids, greedy, K=K,
                              use_ngram=True, ngram_n=args.ngram_n,
                              cot_caches=cot_arg, cot_bucket=args.mode if cot_arg else None,
                              stop_token_ids=stop_ids, no_eos_stop=args.no_eos_stop)
            rows.append((f"jacobi K={K}", len(r.tokens), r.decode_tps,
                         f"{r.arl:.2f}", f"{r.n_rounds}"))
            print(f"[bench] jacobi K={K:2d}:  {r.decode_tps:6.1f} tok/s "
                  f"ARL={r.arl:.2f} rounds={r.n_rounds} ({len(r.tokens)} tok)",
                  file=sys.stderr)

    # 4) static jacobi
    sj = StaticJacobiDecoder(model)
    best = None
    for K in Ks:
        r = sj.generate(prompt_ids, args.max_tokens, K=K,
                        stop_token_ids=stop_ids if not args.no_eos_stop else (),
                        use_ngram=True, ngram_n=args.ngram_n,
                        cot_caches=cot_arg, cot_bucket=args.mode if cot_arg else None)
        speedup = r.decode_tps / max(ar_tps, 1e-9)
        rows.append((f"static jacobi K={K}", len(r.tokens), r.decode_tps,
                     f"{r.arl:.2f}", f"{r.n_rounds}"))
        n_match = next((i for i, (a, b) in enumerate(zip(ar_tokens, r.tokens)) if a != b),
                       min(len(ar_tokens), len(r.tokens)))
        print(f"[bench] static jacobi K={K:2d}: {r.decode_tps:6.1f} tok/s "
              f"ARL={r.arl:.2f} rounds={r.n_rounds} compile={r.compile_s:.1f}s "
              f"speedup={speedup:.2f}x  match-vs-AR={n_match}/{min(len(ar_tokens), len(r.tokens))}",
              file=sys.stderr)
        if best is None or r.decode_tps > best[1]:
            best = (K, r.decode_tps, r)

    print("\n  path                  tokens   tok/s     ARL   rounds")
    for name, n, tps, arl, rounds in rows:
        print(f"  {name:<20} {n:>7}  {tps:>7.1f}   {arl:>5}   {rounds:>6}")
    if best:
        print(f"\n  BEST: static jacobi K={best[0]} → {best[1]:.1f} tok/s "
              f"({best[1]/max(ar_tps,1e-9):.2f}× vs AR reference)")
    if args.show_text and best:
        print("\n===== STATIC JACOBI OUTPUT =====")
        print(tok.decode(best[2].tokens, skip_special_tokens=False))


if __name__ == "__main__":
    main()
