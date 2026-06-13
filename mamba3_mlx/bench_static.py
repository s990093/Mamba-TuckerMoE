"""Benchmark + correctness check: StaticDecoder vs the reference decode path.

Examples
--------
# Correctness (greedy, token-exact compare) then throughput sweep:
.venv/bin/python3 mamba3_mlx/bench_static.py --verify

# Throughput only, 512 tokens, specific unrolls:
.venv/bin/python3 mamba3_mlx/bench_static.py --max_tokens 512 --unroll 1,4,8
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
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint
from mamba3_mlx.mlx_model.static_decode import StaticDecoder
from mamba3_mlx.inference.generator import generate as ref_generate

_DEFAULT_SYSTEM = "You are a helpful assistant. Think step by step before answering."


def build_prompt_ids(tok: Tokenizer, user_msg: str) -> list[int]:
    text = (
        f"<|im_start|>system\n{_DEFAULT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
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
    p.add_argument("--prompt", default="Explain why the sky is blue.")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--unroll", default="1,4,8",
                   help="Comma-separated unroll factors to benchmark.")
    p.add_argument("--verify", action="store_true",
                   help="Greedy token-exact comparison against the reference path first.")
    p.add_argument("--verify_tokens", type=int, default=64)
    p.add_argument("--skip-ref", action="store_true",
                   help="Skip the reference-path baseline benchmark.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temp", type=float, default=0.426)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_p", type=float, default=0.981)
    p.add_argument("--min_p", type=float, default=0.067)
    p.add_argument("--rep_pen", type=float, default=1.146)
    p.add_argument("--pres_pen", type=float, default=0.143)
    p.add_argument("--freq_pen", type=float, default=0.133)
    p.add_argument("--show-text", action="store_true",
                   help="Print decoded text of the last static run for eyeballing.")
    p.add_argument("--metal-fuse", action="store_true",
                   help="Fused Metal kernel for the Mamba SSM inner chain "
                        "(bit-exact, ~+25%% over the compiled-graph path).")
    p.add_argument("--quant-moe", type=int, default=0, metavar="BITS",
                   help="Selective quantization of the TuckerMoE U_in/U_out "
                        "factors only (e.g. 8). Requires --metal-fuse. "
                        "NOT bit-exact; --verify divergence is expected.")
    p.add_argument("--quant-proj", type=int, default=0, metavar="BITS",
                   help="Also quantize in_proj (dt/A/λ tail stays bf16), "
                        "mamba_dense_proj and attention q/k/v/o.")
    p.add_argument("--quant-head", type=int, default=0, metavar="BITS",
                   help="Also quantize the 49 MB head projection.")
    args = p.parse_args()

    print(f"[load] tokenizer: {args.tokenizer_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)

    print(f"[load] model:     {args.model_path}", file=sys.stderr)
    t0 = time.time()
    load_checkpoint(model, args.model_path, dtype=mx.bfloat16)
    mx.eval(model.parameters())
    print(f"[load] done in {time.time() - t0:.2f}s", file=sys.stderr)

    prompt_ids = build_prompt_ids(tok, args.prompt)
    print(f"[prompt] {len(prompt_ids)} tokens", file=sys.stderr)

    decoder = StaticDecoder(model, metal_fuse=args.metal_fuse,
                            quant_moe_bits=args.quant_moe,
                            quant_proj_bits=args.quant_proj,
                            quant_head_bits=args.quant_head)

    # ── correctness: greedy, no penalties → both paths must emit identical ids ──
    if args.verify:
        n = args.verify_tokens
        gcfg = GenerationConfig(max_tokens=n, temperature=0.0, rep_pen=1.0,
                                pres_pen=0.0, freq_pen=0.0, seed=args.seed)
        print(f"\n[verify] greedy {n} tokens, no penalties", file=sys.stderr)
        ref = ref_generate(model, prompt_ids, gcfg, stop_token_ids=[], no_eos_stop=True)
        st = decoder.generate(prompt_ids, gcfg, stop_token_ids=(), unroll=1)
        a, b = ref.tokens[:n], st.tokens[:n]
        if a == b:
            print(f"[verify] PASS — {n}/{n} tokens identical", file=sys.stderr)
        else:
            div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"[verify] DIVERGED at token {div}/{n}", file=sys.stderr)
            print(f"  ref:    ...{a[max(0, div-3):div+3]}", file=sys.stderr)
            print(f"  static: ...{b[max(0, div-3):div+3]}", file=sys.stderr)

        # Same-seed sampled comparison (informational: fp noise may diverge late).
        gcfg_s = GenerationConfig(
            max_tokens=n, temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
            min_p=args.min_p, rep_pen=args.rep_pen, pres_pen=args.pres_pen,
            freq_pen=args.freq_pen, seed=args.seed)
        ref_s = ref_generate(model, prompt_ids, gcfg_s, stop_token_ids=[], no_eos_stop=True)
        st_s = decoder.generate(prompt_ids, gcfg_s, stop_token_ids=(), unroll=1)
        a, b = ref_s.tokens[:n], st_s.tokens[:n]
        match = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"[verify] sampled same-seed: first {match}/{n} tokens identical "
              f"(informational)", file=sys.stderr)

    gcfg = GenerationConfig(
        max_tokens=args.max_tokens, temperature=args.temp, top_k=args.top_k,
        top_p=args.top_p, min_p=args.min_p, rep_pen=args.rep_pen,
        pres_pen=args.pres_pen, freq_pen=args.freq_pen, seed=args.seed)

    rows = []

    # ── baseline: reference decode loop (per-block compiled path) ──────────────
    if not args.skip_ref:
        print("\n[bench] reference path …", file=sys.stderr)
        r = ref_generate(model, prompt_ids, gcfg, stop_token_ids=[], no_eos_stop=True)
        rows.append(("reference", len(r.tokens), r.decode_tps, "-"))
        print(f"[bench] reference: {r.decode_tps:7.1f} tok/s", file=sys.stderr)

    # ── static decoder at each unroll ──────────────────────────────────────────
    last = None
    for u in [int(x) for x in args.unroll.split(",") if x.strip()]:
        print(f"[bench] static unroll={u} …", file=sys.stderr)
        r = decoder.generate(prompt_ids, gcfg, stop_token_ids=(), unroll=u)
        rows.append((f"static u={u}", len(r.tokens), r.decode_tps, f"{r.compile_s:.1f}s"))
        print(f"[bench] static u={u}: {r.decode_tps:7.1f} tok/s "
              f"(compile {r.compile_s:.1f}s, prefill {r.prefill_tps:,.0f} tok/s)",
              file=sys.stderr)
        last = r

    print("\n  path           tokens   decode tok/s   compile")
    for name, n, tps, c in rows:
        print(f"  {name:<14} {n:>6}   {tps:>12.1f}   {c:>7}")

    if args.show_text and last is not None:
        print("\n===== STATIC OUTPUT (last run) =====")
        print(tok.decode(last.tokens, skip_special_tokens=False))


if __name__ == "__main__":
    main()
