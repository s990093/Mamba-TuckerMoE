#!/usr/bin/env python3
"""
Token-level comparison: AR (greedy) vs Jacobi K=32.

Usage:
    python inference/compare_ar_jacobi.py [--prompt TEXT] [--max-tokens N] [--K 32]
"""
import sys, os, argparse, time, types
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.join(_THIS, "lib"))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_REPO_ROOT = os.path.dirname(_THIS)
_TOK_DIR   = os.path.join(_THIS, "tokenizer")

from lib.mlx_hybrid_infer import (
    Mamba3Config, Mamba3LanguageModel, resolve_mlx_checkpoint,
    strict_load_and_convert,
)
from benchmark_mlx import _materialize_cache_tree, _pad_transformer_caches
from jacobi_enhanced import (
    enhanced_jacobi_stream, build_compiled_verify_fns,
    build_compiled_verify_with_intra_fns, _deep_copy_caches,
)


def _make_config():
    cfg = Mamba3Config(
        d_model=768, d_state=64, d_head=64, expand=2, num_layers=6,
        mimo_rank=4, num_kv_heads=4,
        use_parallel_scan=True, chunk_size=64,
        use_kmoe=True, kmoe_num_experts=8, kmoe_top_k=2,
        kmoe_r1=32, kmoe_r2=512, kmoe_r3=256, ffn_expand=6,
    )
    cfg.lookahead_router   = False
    cfg.tucker_einsum_fuse = False
    cfg.tucker_full_fuse   = False
    cfg.fused_mamba_mixer  = False
    return cfg


def load_model(checkpoint: str, quantize: int):
    cfg   = _make_config()
    model = Mamba3LanguageModel(cfg, vocab_size=32003)
    resolved, kind = resolve_mlx_checkpoint(checkpoint, repo_root=_REPO_ROOT)
    if resolved is None:
        print("[warn] No checkpoint – random weights.")
    else:
        print(f"Loading {os.path.basename(resolved)} ({kind}) …")
        strict_load_and_convert(model, resolved)
        mx.eval(model.parameters())
    if quantize in (4, 8):
        nn.quantize(model, group_size=64, bits=quantize)
        mx.eval(model.parameters())
        print(f"  → {quantize}-bit quantized")
    return model


def _deep_copy(c):
    if isinstance(c, mx.array):
        arr = np.array(c).astype(np.float32 if c.dtype == mx.bfloat16 else None)
        return mx.array(arr).astype(c.dtype)
    if isinstance(c, (list, tuple)):
        return type(c)(_deep_copy(x) for x in c)
    return c


def wrap_chatml(text: str, sys_prompt: str = "") -> str:
    if sys_prompt.strip():
        return (
            f"<|im_start|>system\n{sys_prompt.strip()}<|im_end|>\n"
            f"<|im_start|>user\n{text.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    return f"<|im_start|>user\n{text.strip()}<|im_end|>\n<|im_start|>assistant\n"


STOP_IDS = frozenset({2, 32001})   # </s> EOS + <|im_end|> turn-end


def run_ar_compiled(fn1, caches, logits_pf, prompt_len: int,
                    max_tokens: int, save_logits: bool = False):
    """AR using the compiled K=1 verify fn — same compiled graph as Jacobi uses.

    Returns (tokens, elapsed_sec, logit_rows) where logit_rows is a list of (vocab,)
    np.float32 arrays (one per generated token) when save_logits=True, else [].
    """
    tokens = []
    logit_rows = []
    logits = logits_pf
    pos    = prompt_len
    t0     = time.perf_counter()
    for _ in range(max_tokens):
        row = logits[0, -1, :]
        tid = int(mx.argmax(row).item())
        tokens.append(tid)
        if save_logits:
            logit_rows.append(np.array(row.astype(mx.float32), copy=True))
        if tid in STOP_IDS:
            break
        x      = mx.array([[tid]], dtype=mx.int32)
        logits, caches = fn1(x, caches, mx.array(pos, dtype=mx.int32))
        mx.eval(logits, caches)
        pos += 1
    return tokens, time.perf_counter() - t0, logit_rows


def _bench_component(label: str, fn, n_reps: int = 5):
    """Run fn() n_reps times and report min/mean timing."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mn, avg = min(times), sum(times) / len(times)
    print(f"  [{label}] min={mn*1000:.1f}ms  avg={avg*1000:.1f}ms")
    return mn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/latest_sft_cot_model.npz")
    p.add_argument("--prompt",     default="who are you")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--K",          type=int, default=32)
    p.add_argument("--quantize",   type=int, default=8, choices=[0, 4, 8])
    p.add_argument("--ngram-n",    type=int, default=4)
    p.add_argument("--max-cache",  type=int, default=512)
    p.add_argument("--warmup",     type=int, default=0,
                   help="AR warmup tokens before Jacobi (default 0)")
    p.add_argument("--sys-prompt", default="",
                   help="Optional system prompt prepended before user turn")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_TOK_DIR, trust_remote_code=True)

    model       = load_model(args.checkpoint, args.quantize)
    router_temp = mx.array(0.5, dtype=mx.bfloat16)

    prompt_text = wrap_chatml(args.prompt, args.sys_prompt)
    prompt_ids  = tok.encode(prompt_text, add_special_tokens=False)
    prompt_len  = len(prompt_ids)
    if args.sys_prompt:
        print(f"Sys-prompt: {args.sys_prompt[:80]!r}…")
    print(f"Prompt: {prompt_len} tokens")

    # ── Prefill ──────────────────────────────────────────────────────────────
    x_pf = mx.array([prompt_ids], dtype=mx.int32)
    t0   = time.perf_counter()
    logits_pf, caches_pf = model(x_pf, caches=None, router_temp=router_temp)
    mx.eval(logits_pf, caches_pf)
    caches_pf = _materialize_cache_tree(caches_pf)
    mx.eval(caches_pf)
    caches_pf = _pad_transformer_caches(caches_pf, args.max_cache)
    mx.eval(caches_pf)
    print(f"Prefill: {(time.perf_counter()-t0)*1000:.0f} ms")

    caches_ar  = _deep_copy(caches_pf);  mx.eval(caches_ar)
    caches_jac = _deep_copy(caches_pf);  mx.eval(caches_jac)

    # ── Compile warmup ────────────────────────────────────────────────────────
    print(f"\nBuilding compiled fns for K={args.K} …")
    warmup_caches = _deep_copy(caches_pf);  mx.eval(warmup_caches)
    compiled_fns  = build_compiled_verify_fns(
        model, router_temp,
        K_values=(args.K,),
        caches_for_warmup=warmup_caches,
        prompt_len=prompt_len,
    )
    print("  compiled regular fns.")
    warmup_caches2 = _deep_copy(caches_pf);  mx.eval(warmup_caches2)
    compiled_intra_fns = build_compiled_verify_with_intra_fns(
        model, router_temp,
        K_values=(args.K,),
        caches_for_warmup=warmup_caches2,
        prompt_len=prompt_len,
    )
    del warmup_caches, warmup_caches2
    print("  compiled intra fns.")

    # ── Micro-benchmarks ─────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Micro-benchmarks (5 reps each, caches NOT consumed):")

    bench_caches = _deep_copy(caches_pf);  mx.eval(bench_caches)
    bench_x1  = mx.array([[1]], dtype=mx.int32)
    bench_xK  = mx.array([[1] * args.K], dtype=mx.int32)
    bench_x8  = mx.array([[1] * 8], dtype=mx.int32)
    bench_pos = mx.array(prompt_len, dtype=mx.int32)

    def _run_k1():
        lo, cc = compiled_fns[1](bench_x1, bench_caches, bench_pos)
        mx.eval(lo, cc)

    def _run_kmax():
        lo, cc = compiled_fns[args.K](bench_xK, bench_caches, bench_pos)
        mx.eval(lo, cc)

    def _run_kmax_intra():
        lo, cc, mi = compiled_intra_fns[args.K](bench_xK, bench_caches, bench_pos)
        mx.eval(lo, cc)
        for entry in mi:
            if entry is not None:
                mx.eval(*entry)

    def _run_k8():
        lo, cc = compiled_fns[8](bench_x8, bench_caches, bench_pos)
        mx.eval(lo, cc)

    def _run_deepcopy():
        cc = _deep_copy_caches(bench_caches)
        mx.eval(cc)

    t_k1        = _bench_component(f"K=1  (AR step)",            _run_k1)
    t_kmax      = _bench_component(f"K={args.K} (verify)",       _run_kmax)
    t_kmax_intr = _bench_component(f"K={args.K} (verify+intra)", _run_kmax_intra)
    t_k8        = _bench_component("K=8  (batch_replay)",        _run_k8)
    t_copy      = _bench_component("deep_copy_caches",            _run_deepcopy)

    # Theoretical speedup analysis
    arl_guess = 8.0  # assume ARL~8 as rough guess
    # Old path: copy + K-verify + batch_replay(accepted+1)
    round_cost_old = t_copy + t_kmax + t_k8
    theoretical_tps_old = (arl_guess + 1) / round_cost_old
    # New path: copy + K-verify-with-intra + K=1 correction
    round_cost_new = t_copy + t_kmax_intr + t_k1
    theoretical_tps_new = (arl_guess + 1) / round_cost_new
    ar_tps_est = 1.0 / t_k1
    print(f"\n  Theoretical (ARL≈{arl_guess:.0f}):")
    print(f"    Old (batch_replay) : {theoretical_tps_old:.0f} tok/s  {theoretical_tps_old/ar_tps_est:.2f}×")
    print(f"    New (intra+K=1)    : {theoretical_tps_new:.0f} tok/s  {theoretical_tps_new/ar_tps_est:.2f}×")

    # Full-accept case (no partial-accept overhead)
    full_tps = args.K / (t_copy + t_kmax)
    full_tps_intr = args.K / (t_copy + t_kmax_intr)
    print(f"  Full-accept (ARL={args.K}): old={full_tps:.0f} tok/s {full_tps/ar_tps_est:.2f}×  "
          f"new={full_tps_intr:.0f} tok/s {full_tps_intr/ar_tps_est:.2f}×")

    # ── AR decode (compiled fn1) ──────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("AR  (greedy, compiled K=1 fn)")
    ar_toks, ar_sec, ar_logits = run_ar_compiled(
        compiled_fns[1], caches_ar, logits_pf, prompt_len,
        args.max_tokens, save_logits=True)
    ar_tps  = len(ar_toks) / ar_sec
    ar_text = tok.decode(ar_toks, skip_special_tokens=False)
    print(f"  {len(ar_toks)} tokens  {ar_sec*1000:.0f} ms  {ar_tps:.1f} tok/s")
    print(f"  text: {ar_text!r}")
    print(f"  ids:  {ar_toks[:50]}")

    # ── Jacobi decode ─────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"Jacobi K={args.K}  ngram-n={args.ngram_n}  warmup={args.warmup}")

    sa    = types.SimpleNamespace(no_materialize_caches=True)
    stats = {}

    def _dummy_prefill(x, rt):
        return logits_pf, caches_jac

    jac_toks = []
    t0 = time.perf_counter()
    for tok_id in enhanced_jacobi_stream(
        model=model,
        run_prefill=_dummy_prefill,
        x_prefill=x_pf,
        prompt_ids=prompt_ids,
        max_cache_len=args.max_cache,
        router_temp=router_temp,
        max_new_tokens=args.max_tokens,
        sample_args=sa,
        stop_token_ids=STOP_IDS,
        prefill_outputs=(logits_pf, caches_jac),
        K=args.K,
        ngram_n=args.ngram_n,
        use_ngram=True,
        compiled_verify_fns=compiled_fns,
        compiled_verify_with_intra_fns=compiled_intra_fns,
        ar_warmup_tokens=args.warmup,
        stats=stats,
    ):
        tid = int(tok_id) if isinstance(tok_id, mx.array) else tok_id
        jac_toks.append(tid)
        if tid in STOP_IDS:
            break

    jac_sec  = time.perf_counter() - t0
    jac_tps  = len(jac_toks) / jac_sec
    jac_text = tok.decode(jac_toks, skip_special_tokens=False)
    print(f"  {len(jac_toks)} tokens  {jac_sec*1000:.0f} ms  {jac_tps:.1f} tok/s")
    print(f"  text: {jac_text!r}")
    print(f"  ids:  {jac_toks[:50]}")

    # ── Correctness report ────────────────────────────────────────────────────
    W = 60
    n          = min(len(ar_toks), len(jac_toks))
    mismatches = [i for i in range(n) if ar_toks[i] != jac_toks[i]]
    n_match    = n - len(mismatches)
    match_rate = n_match / n if n > 0 else 0.0
    speedup    = ar_sec / jac_sec if jac_sec > 0 else 0.0

    print(f"\n{'═'*W}")
    print(f"  CORRECTNESS CHECK: AR vs Jacobi  (target ≥99%)")
    print(f"{'═'*W}")
    print(f"  Tokens compared  : {n}  (AR={len(ar_toks)}, Jac={len(jac_toks)})")
    print(f"  Matching tokens  : {n_match}")
    print(f"  Mismatched tokens: {len(mismatches)}")
    print(f"  Match rate       : {match_rate*100:.2f}%")
    print()

    verdict = ("✓ PASS  (≥99% match)" if match_rate >= 0.99 else
               "~ CLOSE (≥95% match)" if match_rate >= 0.95 else
               "✗ FAIL  (<95% match)")
    print(f"  RESULT: {verdict}")
    print(f"{'═'*W}")

    # ── Side-by-side token table ──────────────────────────────────────────────
    table_rows = min(n, 40)
    print(f"\nToken comparison (first {table_rows} positions):")
    hdr = f"  {'pos':>4} | {'AR id':>6} {'AR text':<14} | {'Jac id':>6} {'Jac text':<14} | same"
    print(hdr)
    print(f"  {'─'*4}-+-{'─'*21}-+-{'─'*21}-+------")
    for i in range(table_rows):
        a    = ar_toks[i]
        b    = jac_toks[i] if i < len(jac_toks) else -1
        at   = repr(tok.decode([a], skip_special_tokens=False))[:12]
        bt   = repr(tok.decode([b], skip_special_tokens=False))[:12] if b >= 0 else "—"
        same = "✓" if a == b else "✗"
        print(f"  {i:>4} | {a:>6} {at:<14} | {b:>6} {bt:<14} | {same}")
    if n > table_rows:
        remaining_mm = [i for i in mismatches if i >= table_rows]
        print(f"  ... ({n - table_rows} more tokens; {len(remaining_mm)} mismatches beyond row {table_rows})")

    # ── Mismatch detail ───────────────────────────────────────────────────────
    if mismatches:
        print(f"\nMismatch detail (all {len(mismatches)}):")
        for idx in mismatches:
            a  = ar_toks[idx]
            b  = jac_toks[idx] if idx < len(jac_toks) else -1
            at = tok.decode([a], skip_special_tokens=False)
            bt = tok.decode([b], skip_special_tokens=False) if b >= 0 else "—"
            # logit rank of the jacobi token according to AR logits
            rank_str = ""
            if idx < len(ar_logits) and b >= 0:
                logit_row = ar_logits[idx]
                rank = int(np.sum(logit_row >= logit_row[b]))
                rank_str = f"  [Jac tok rank in AR logits: #{rank}]"
            print(f"    pos {idx:3d}: AR={a}({at!r})  Jac={b}({bt!r}){rank_str}")

    # ── Jacobi internal stats ─────────────────────────────────────────────────
    print(f"\nJacobi internal stats:")
    if stats:
        rounds_j  = stats.get("rounds", 0)
        accepted  = stats.get("accepted", 0)
        proposed  = stats.get("proposed", 0)
        arl_real  = accepted / max(rounds_j, 1)
        acc_rate  = 100.0 * accepted / max(proposed, 1)
        print(f"  Rounds           : {rounds_j}")
        print(f"  Accepted/Proposed: {accepted}/{proposed}  ({acc_rate:.1f}%)")
        print(f"  ARL (mean)       : {arl_real:.2f}")
        per_rd = stats.get("per_round_arl", [])
        if per_rd:
            print(f"  ARL per-round    : min={min(per_rd)}  μ={sum(per_rd)/len(per_rd):.2f}"
                  f"  max={max(per_rd)}  dist={per_rd}")
        ng_hit   = stats.get("ngram_hit_rounds", 0)
        ng_rate  = stats.get("ngram_hit_rate", 0.0)
        print(f"  N-gram hit rate  : {ng_rate*100:.1f}%  ({ng_hit}/{rounds_j} rounds)")
        intra_r  = stats.get("intra_rounds", 0)
        replay_r = stats.get("replay_rounds", 0)
        if intra_r + replay_r > 0:
            print(f"  Partial-accept   : intra={intra_r}  replay={replay_r}"
                  f"  (intra rate {100*intra_r/max(intra_r+replay_r,1):.0f}%)")
        la_hits = stats.get("lookahead_hits", 0)
        la_rate = stats.get("la_hit_rate", 0.0)
        print(f"  Lookahead assists: {la_hits}  ({la_rate*100:.1f}%/round)")

    # ── Speed comparison ──────────────────────────────────────────────────────
    print(f"\nSpeed (informational — correctness is primary goal):")
    print(f"  AR  : {ar_tps:.1f} tok/s  ({ar_sec*1000:.0f} ms)")
    print(f"  Jac : {jac_tps:.1f} tok/s  ({jac_sec*1000:.0f} ms)  speedup={speedup:.2f}×")


if __name__ == "__main__":
    main()
