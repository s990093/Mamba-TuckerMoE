"""dLLM (diffusion-LLM) CLI — inference, benchmark, and validation.

Builds an UNTRAINED ``DLLMModel`` (no checkpoint — the real dLLM weights are
still training) and exercises the four ported pieces from DLLM_MLX_PORT.md:
① [MASK] token, ② bidirectional attention, ④ iterative unmasking, plus the
§驗證 validation suite.  Output text is meaningless until trained weights are
loaded; the point is that the inference + high-performance paths run, are
shape-correct, numerically faithful (eager == compiled), and fast.

Examples
--------
# Iterative-unmasking generation (eager reference path):
.venv/bin/python3 mamba3_mlx/dllm_infer.py --mode generate \
    --prompt "Explain why the sky is blue." --gen-len 64 --steps 16

# Same, high-performance single-compiled-graph forward:
.venv/bin/python3 mamba3_mlx/dllm_infer.py --mode generate --static

# Throughput: eager vs StaticDLLM:
.venv/bin/python3 mamba3_mlx/dllm_infer.py --mode bench --gen-len 64 --steps 16

# Validation suite (parity + reconstruction):
.venv/bin/python3 mamba3_mlx/dllm_infer.py --mode validate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.mlx_dllm_model import (BASE_VOCAB, MASK_ID, DiffusionGenConfig,
                                       DLLMConfig, StaticDLLM, build_random_dllm,
                                       diffusion_generate, trim_to_stop)
from mamba3_mlx.mlx_dllm_model.validate import (check_bidirectional,
                                                check_static_parity,
                                                fixed_ratio_reconstruction,
                                                iterative_reconstruction)

_DEFAULT_SYSTEM = "You are a helpful assistant. Think step by step before answering."


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def build_prompt_ids(tok, user_msg: str) -> list[int]:
    """ChatML prompt up to the assistant header (response region is masked)."""
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


def _decode(tok, ids: list[int]) -> str:
    clean = [t for t in trim_to_stop(ids) if 0 <= t < BASE_VOCAB]
    return tok.decode(clean, skip_special_tokens=False)


def _load_tokenizer(path: str | None):
    if not path:
        return None
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def _make_prompt(tok, args) -> list[int]:
    if tok is not None:
        return build_prompt_ids(tok, args.prompt)
    # No tokenizer: synthetic prompt of random base-vocab ids.
    mx.random.seed(args.seed)
    return [int(t) for t in
            mx.random.randint(0, BASE_VOCAB, (args.synthetic_prompt,)).tolist()]


def _dgcfg(args) -> DiffusionGenConfig:
    """Build the diffusion controls; --temp >= 0 pins a constant temperature."""
    t_min, t_max = args.t_min, args.t_max
    if args.temp >= 0.0:
        t_min = t_max = args.temp
    return DiffusionGenConfig(
        gen_len=args.gen_len, steps=args.steps, sampler=args.sampler,
        entropy_bound=args.entropy_bound, t_min=t_min, t_max=t_max,
        adaptive_stop=args.adaptive_stop, stability_threshold=args.stability_threshold,
        confidence_threshold=args.confidence_threshold, seed=args.seed)


def _canvas_printer(mask_id: int):
    """on_step callback: render the [MASK] canvas filling in (█=filled ·=mask)."""
    def _p(step: int, resp_ids: list[int]) -> None:
        pat = "".join("·" if t == mask_id else "█" for t in resp_ids)
        n = sum(1 for t in resp_ids if t != mask_id)
        _log(f"  step {step:>2}  filled {n:>3}/{len(resp_ids):<3}  [{pat}]")
    return _p


def cmd_generate(model, tok, args) -> None:
    prompt_ids = _make_prompt(tok, args)
    dg = _dgcfg(args)
    path = "prefix-cache" if args.prefix_cache else ("static-full" if args.static else "eager-full")
    _log(f"[prompt] {len(prompt_ids)} tok | canvas G={dg.gen_len} starts all-[MASK] | "
         f"sampler={dg.sampler} steps={dg.steps} t=[{dg.t_min},{dg.t_max}] "
         f"{'adaptive-stop ' if dg.adaptive_stop else ''}{'self-cond ' if model.self_conditioning is not None else ''}"
         f"{path}")
    on_step = _canvas_printer(MASK_ID) if args.show_canvas else None

    if args.prefix_cache:
        res = StaticDLLM(model).diffusion_cached(prompt_ids, dg, on_step=on_step)
    elif args.static:
        res = StaticDLLM(model).diffusion(prompt_ids, dg, on_step=on_step)
    else:
        res = diffusion_generate(model, prompt_ids, dg, on_step=on_step)

    _log(f"[done] {res.tokens_per_s:6.1f} tok/s | {res.forwards_per_s:5.1f} fwd/s "
         f"| {res.steps_used}/{dg.steps} steps | {res.elapsed*1e3:6.1f} ms | encode+compile {res.compile_s:.2f}s")
    print("\n===== RESPONSE TOKEN IDS =====")
    print(res.response_ids)
    if tok is not None:
        print("\n===== DECODED (untrained → gibberish) =====")
        print(_decode(tok, res.response_ids))


def cmd_bench(model, tok, args) -> None:
    prompt_ids = _make_prompt(tok, args)
    dg = _dgcfg(args)
    _log(f"[bench] prompt {len(prompt_ids)} tok, G={dg.gen_len}, steps {dg.steps}, sampler={dg.sampler}")

    rows = []
    _log("[bench] eager full forward …")
    r = diffusion_generate(model, prompt_ids, dg)
    rows.append(("eager-full", r.tokens_per_s, r.forwards_per_s, r.steps_used, r.compile_s))

    _log("[bench] StaticDLLM full forward (compiled P+G) …")
    r = StaticDLLM(model).diffusion(prompt_ids, dg)
    rows.append(("static-full", r.tokens_per_s, r.forwards_per_s, r.steps_used, r.compile_s))

    _log("[bench] StaticDLLM prefix-cache (encode once, denoise G) …")
    r = StaticDLLM(model).diffusion_cached(prompt_ids, dg)
    rows.append(("prefix-cache", r.tokens_per_s, r.forwards_per_s, r.steps_used, r.compile_s))

    print("\n  path           tok/s    fwd/s   steps   enc+comp")
    for name, tps, fps, su, cs in rows:
        print(f"  {name:<13} {tps:6.1f}   {fps:6.1f}   {su:>5}   {cs:6.2f}s")


def cmd_validate(model, tok, args) -> None:
    print("\n=== (A) forward parity ===")
    a1 = check_bidirectional(model, seq_len=args.seq_len, seed=args.seed)
    print(f"  bidirectional vs causal : max|Δ|={a1['max_abs_diff_bi_vs_causal']:.4f} "
          f"differ={a1['outputs_differ']} finite={a1['all_finite']}")
    a2 = check_static_parity(model, seq_len=args.seq_len, seed=args.seed)
    print(f"  eager vs StaticDLLM     : max|Δ|={a2['max_abs_diff_eager_vs_static']:.4e} "
          f"within_tol(<= {a2['tol']})={a2['within_tol']}")

    print("\n=== (B) fixed-ratio reconstruction (top-1 acc; ~chance until trained) ===")
    b = fixed_ratio_reconstruction(model, seq_len=args.seq_len, seed=args.seed)
    for ratio, acc in b.items():
        print(f"  ratio {ratio:>4}: acc {acc:.4f}")

    print("\n=== (C) iterative reconstruction vs gold (~chance until trained) ===")
    for static in (False, True):
        c = iterative_reconstruction(model, gen_len=args.gen_len, steps=args.steps,
                                     temperature=args.temp, seed=args.seed, static=static)
        tag = "static" if static else "eager "
        print(f"  [{tag}] token-acc {c['token_accuracy']:.4f} exact={c['exact_match']} "
              f"| {c['tokens_per_s']:.1f} tok/s compile {c['compile_s']:.2f}s")


def main() -> None:
    p = argparse.ArgumentParser(description="dLLM inference / bench / validate (untrained)")
    p.add_argument("--mode", choices=("generate", "bench", "validate"), default="generate")
    p.add_argument("--prompt", default="Explain why the sky is blue.")
    p.add_argument("--tokenizer_path", default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"),
                   help="HF tokenizer.json; pass '' to use a synthetic random prompt.")
    p.add_argument("--gen-len", type=int, default=64, help="masked response length G (canvas)")
    p.add_argument("--steps", type=int, default=16, help="max denoising iterations T")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--static", action="store_true", help="compiled full (P+G) forward")
    p.add_argument("--prefix-cache", action="store_true",
                   help="fastest: encode prompt once, denoise canvas-only each step "
                        "(prefix-LM attention — DiffusionGemma encoder/decoder split)")
    # ── DiffusionGemma-inspired controls ──────────────────────────────────────
    p.add_argument("--sampler", choices=("entropy", "cosine"), default="entropy",
                   help="entropy=adaptive entropy-bound acceptance; cosine=fixed schedule")
    p.add_argument("--entropy-bound", type=float, default=0.1,
                   help="joint-MI bound for the entropy sampler (higher = commit more/step)")
    p.add_argument("--t-min", type=float, default=0.4, help="final-step temperature")
    p.add_argument("--t-max", type=float, default=0.8, help="first-step temperature")
    p.add_argument("--temp", type=float, default=-1.0,
                   help=">=0 pins a constant temperature (overrides t-min/t-max; 0=greedy)")
    p.add_argument("--adaptive-stop", action="store_true",
                   help="stable+confident early stop (StableAndConfidentStoppingCriteria)")
    p.add_argument("--stability-threshold", type=int, default=1)
    p.add_argument("--confidence-threshold", type=float, default=0.005)
    p.add_argument("--self-cond", action="store_true",
                   help="enable cross-step self-conditioning (adds params; untrained here)")
    p.add_argument("--show-canvas", action="store_true",
                   help="print the [MASK] canvas filling in, step by step")
    p.add_argument("--seq-len", type=int, default=64, help="validate: sequence length")
    p.add_argument("--synthetic-prompt", type=int, default=24,
                   help="prompt length when no tokenizer is used")
    p.add_argument("--no-seed-experts", action="store_true",
                   help="leave TuckerMoE factors at zero-init (MoE emits bias only)")
    p.add_argument("--d-model", type=int, default=None, help="override d_model (smaller = quicker smoke)")
    args = p.parse_args()

    cfg_kw = {"self_conditioning": args.self_cond}
    if args.d_model is not None:
        cfg_kw["d_model"] = args.d_model
    cfg = DLLMConfig(**cfg_kw)

    _log(f"[build] untrained DLLMModel  vocab={cfg.vocab_size} d_model={cfg.d_model} "
         f"layers={cfg.num_layers}×({cfg.mamba_ratio}M+1TF)  MASK_ID={MASK_ID}")
    t0 = time.time()
    model = build_random_dllm(cfg, seed=args.seed, seed_experts=not args.no_seed_experts)
    _log(f"[build] done in {time.time() - t0:.2f}s (no checkpoint loaded)")

    tok = _load_tokenizer(args.tokenizer_path or None)
    if tok is not None:
        _log(f"[tok]   {args.tokenizer_path} (vocab {tok.get_vocab_size()})")

    {"generate": cmd_generate, "bench": cmd_bench, "validate": cmd_validate}[args.mode](
        model, tok, args)


if __name__ == "__main__":
    main()
