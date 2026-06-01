#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B benchmark for Jacobi enhanced decoding strategies (v4).

Measures configurations on the same prompts and prints a comparison table:
  A) baseline       — pure autoregressive (generate_token_stream)
  B) jacobi         — parallel K-token verify, carry-over init only
  C) +ngram         — B + N-gram seeded guess initialization
  D) +adaptive      — C + Lookahead + dynamic K (EMA-based)
  E) +tree          — C + Tree Attention (P0): B branches of depth D each
  F) +adaptive_tree — C + Tree Attention + Entropy-based K (P0+P1)

Metrics reported per configuration:
  tok/s     — decode throughput (tokens per second)
  speedup   — vs. baseline autoregressive
  ARL       — average tokens accepted per Jacobi round
  branch%   — branch win distribution for tree configs (b0%/b1%/b2%)
  ent       — mean entropy per round (for adaptive_tree config)

Usage (from repo root):
  python inference/benchmark_jacobi.py --checkpoint weights.npz \\
      --num-prompts 10 --max-new-tokens 150 --K 32

  # With tree attention:
  python inference/benchmark_jacobi.py --K 32 --tree-branches 3 \\
      --configs baseline,+ngram,+tree,+adaptive_tree

  # Entropy-adaptive only (no tree):
  python inference/benchmark_jacobi.py --K 32 \\
      --configs baseline,+ngram,+adaptive_tree \\
      --entropy-low 0.5 --entropy-high 1.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Path setup (same pattern as stream_mlx.py)
# ---------------------------------------------------------------------------
_INF_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_INF_DIR, "lib")
_REPO_ROOT = os.path.abspath(os.path.join(_INF_DIR, ".."))
for _p in (_LIB_DIR, _INF_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx
import mlx.nn as nn

from benchmark_mlx import (
    _apply_inference_type,
    _build_prompt_ids,
    _init_token_counts,
    _invalidate_tucker_caches,
    _materialize_cache_tree,
    _pad_transformer_caches,
    sample_decode_token,
)
from mlx_hybrid_infer import (
    Mamba3Config,
    Mamba3LanguageModel,
    attach_decode_compilation,
    attach_verify_compilation,
    resolve_mlx_checkpoint,
    strict_load_and_convert,
)
from jacobi_enhanced import enhanced_jacobi_stream, build_compiled_verify_fns

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box as rich_box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ---------------------------------------------------------------------------
# Built-in CoT / math prompts
# ---------------------------------------------------------------------------
BUILTIN_PROMPTS: list[str] = [
    # --- 自我介紹 / 身份 ---
    "Who are you?",
    "What is your name?",
    "Tell me a little bit about yourself.",
    "Are you an AI?",
    "Who created you?",
    # --- 打招呼 / 問候 ---
    "Hello!",
    "Hi, how are you doing?",
    "Good morning!",
    "What's up?",
    "Hey there, how's your day going?",
    # --- 喜好 / 觀點 ---
    "What's your favorite color?",
    "Do you like music?",
    "What kind of books do you recommend?",
    "Do you have any hobbies?",
    "What's your favorite food?",
    # --- 日常對話 ---
    "Tell me a joke.",
    "What's the weather like today?",
    "Can you give me some motivation?",
    "I'm feeling bored, what should I do?",
    "Tell me something interesting.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompts(path: str) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(item.get("input", item.get("prompt", str(item))))
        return out
    raise ValueError(f"prompts file must be a JSON array, got {type(data)}")


def _make_chatml(user_text: str) -> str:
    return f"<|im_start|>user\n{user_text.strip()}<|im_end|>\n<|im_start|>assistant\n"


def _print_table(rows: list[dict], configs: list[str]) -> None:
    """Print comparison table (Rich if available, else plain text)."""
    headers = ["Config", "tok/s", "Speedup", "ARL", "Branch-wins", "MeanEnt", "Tokens", "Time(s)"]

    if HAS_RICH:
        console = Console(highlight=False)
        tbl = Table(
            title="Jacobi Decode A/B Benchmark (v4)",
            box=rich_box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        col_justifies = {"Config": "left"}
        for h in headers:
            just = col_justifies.get(h, "right")
            tbl.add_column(h, justify=just)

        baseline_tps = None
        for r in rows:
            if r["config"] == "baseline":
                baseline_tps = r["tps"]
            speedup = f"{r['tps'] / baseline_tps:.2f}x" if baseline_tps else "—"
            arl = f"{r['arl']:.2f}" if r["arl"] is not None else "—"
            bwins = _fmt_branch_wins(r.get("tree_branch_wins"))
            ent = f"{r.get('mean_entropy', 0.0):.2f}" if r.get("mean_entropy") is not None else "—"
            is_best = r.get("tps", 0) == max(x["tps"] for x in rows)
            style = "bold green" if is_best else ("green" if r["config"] in ("+tree", "+adaptive_tree") else "")
            tbl.add_row(
                r["config"],
                f"{r['tps']:.1f}",
                speedup,
                arl,
                bwins,
                ent,
                str(r["n_tokens"]),
                f"{r['decode_s']:.2f}",
                style=style,
            )
        console.print(tbl)
    else:
        col_w = [16, 8, 9, 6, 14, 8, 7, 8]
        sep = "  "
        header = sep.join(h.ljust(w) for h, w in zip(headers, col_w))
        print("\n" + "─" * len(header))
        print("  Jacobi Decode A/B Benchmark (v4)")
        print("─" * len(header))
        print(header)
        print("─" * len(header))
        baseline_tps = None
        for r in rows:
            if r["config"] == "baseline":
                baseline_tps = r["tps"]
            speedup = f"{r['tps'] / baseline_tps:.2f}x" if baseline_tps else "—"
            arl = f"{r['arl']:.2f}" if r["arl"] is not None else "—"
            bwins = _fmt_branch_wins(r.get("tree_branch_wins"))
            ent = f"{r.get('mean_entropy', 0.0):.2f}" if r.get("mean_entropy") is not None else "—"
            vals = [
                r["config"], f"{r['tps']:.1f}", speedup, arl, bwins, ent,
                str(r["n_tokens"]), f"{r['decode_s']:.2f}",
            ]
            print(sep.join(v.ljust(w) for v, w in zip(vals, col_w)))
        print("─" * len(header) + "\n")


def _fmt_branch_wins(wins: list[int] | None) -> str:
    """Format branch win counts as 'b0:45/b1:32/b2:23'."""
    if not wins or sum(wins) == 0:
        return "—"
    total = sum(wins)
    return "/".join(f"b{i}:{100*w//total}%" for i, w in enumerate(wins))


# ---------------------------------------------------------------------------
# Single-run helpers
# ---------------------------------------------------------------------------

def _run_baseline(
    model: Any,
    run_prefill: Any,
    prompt_ids: list[int],
    max_cache_len: int,
    router_temp: Any,
    max_new_tokens: int,
    sample_args: Any,
    compiled_single: Any,
) -> dict:
    """Pure autoregressive decode — one compiled single-step per token."""
    x_prefill = mx.array([prompt_ids], dtype=mx.int32)
    logits, caches = run_prefill(x_prefill, router_temp)
    mx.eval(logits, caches)
    caches = _materialize_cache_tree(caches)
    mx.eval(caches)
    caches = _pad_transformer_caches(caches, max_cache_len)
    mx.eval(caches)

    pos = len(prompt_ids)
    generated: list[int] = []
    token_counts = _init_token_counts(prompt_ids, int(logits.shape[-1]))

    row = logits[0, -1, :]
    last = sample_decode_token(row, token_counts, sample_args)
    mx.eval(last)
    generated.append(int(last.item()))

    x_one = last.reshape(1, 1)
    t0 = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        seq_pos = mx.array(pos, dtype=mx.int32)
        logits_d, caches = compiled_single(x_one, caches, seq_pos)
        row = logits_d[0, -1, :]
        last = sample_decode_token(row, token_counts, sample_args)
        mx.eval(last, caches)
        generated.append(int(last.item()))
        x_one = last.reshape(1, 1)
        pos += 1

    decode_s = time.perf_counter() - t0
    n = len(generated)
    return {
        "config": "baseline",
        "tps": n / max(decode_s, 1e-9),
        "decode_s": decode_s,
        "n_tokens": n,
        "arl": None,
        "la_hit_rate": None,
        "tree_branch_wins": None,
        "mean_entropy": None,
    }


def _run_jacobi_config(
    config_name: str,
    *,
    model: Any,
    run_prefill: Any,
    prompt_ids: list[int],
    max_cache_len: int,
    router_temp: Any,
    max_new_tokens: int,
    sample_args: Any,
    compiled_verify_fns: dict,
    K: int,
    K_values: tuple,
    use_ngram: bool,
    use_lookahead: bool,
    use_dynamic_K: bool = False,
    use_tree: bool = False,
    num_tree_branches: int = 3,
    tree_branch_depth: int | None = None,
    use_entropy_k: bool = False,
    entropy_low: float = 0.5,
    entropy_high: float = 1.5,
    ngram_n: int = 4,
    min_ngram_n: int = 1,
    warm_prompt: bool = True,
    lookahead_max_len: int = 8,
    min_lookahead_hit: int = 4,
) -> dict:
    """Run enhanced_jacobi_stream with specified feature flags; return metrics."""
    x_prefill = mx.array([prompt_ids], dtype=mx.int32)
    logits, caches = run_prefill(x_prefill, router_temp)
    mx.eval(logits, caches)
    caches = _materialize_cache_tree(caches)
    mx.eval(caches)
    caches = _pad_transformer_caches(caches, max_cache_len)
    mx.eval(caches)

    run_stats: dict = {}
    generated: list[int] = []
    t0 = time.perf_counter()
    for tok in enhanced_jacobi_stream(
        model=model,
        run_prefill=run_prefill,
        x_prefill=x_prefill,
        prompt_ids=prompt_ids,
        max_cache_len=max_cache_len,
        router_temp=router_temp,
        max_new_tokens=max_new_tokens,
        sample_args=sample_args,
        prefill_outputs=(logits, caches),
        K=K,
        K_min=min(K_values),
        K_max=max(K_values),
        K_values=K_values,
        use_ngram=use_ngram,
        use_lookahead=use_lookahead,
        use_dynamic_K=use_dynamic_K,
        use_tree=use_tree,
        num_tree_branches=num_tree_branches,
        tree_branch_depth=tree_branch_depth,
        use_entropy_k=use_entropy_k,
        entropy_low=entropy_low,
        entropy_high=entropy_high,
        ngram_n=ngram_n,
        min_ngram_n=min_ngram_n,
        warm_prompt=warm_prompt,
        lookahead_max_len=lookahead_max_len,
        min_lookahead_hit=min_lookahead_hit,
        stats=run_stats,
        compiled_verify_fns=compiled_verify_fns,
        compiled_single_fn=compiled_verify_fns.get(1),
        materialize_cache_tree_fn=_materialize_cache_tree,
        pad_transformer_caches_fn=_pad_transformer_caches,
    ):
        mx.eval(tok)
        generated.append(int(tok.item()))

    decode_s = time.perf_counter() - t0
    n = len(generated)
    return {
        "config": config_name,
        "tps": n / max(decode_s, 1e-9),
        "decode_s": decode_s,
        "n_tokens": n,
        "arl": run_stats.get("arl"),
        "la_hit_rate": run_stats.get("la_hit_rate"),
        "tree_branch_wins": run_stats.get("tree_branch_wins"),
        "mean_entropy": run_stats.get("mean_entropy"),
    }


# ---------------------------------------------------------------------------
# Config definitions (extend here to add new configs)
# ---------------------------------------------------------------------------

# Each entry: (use_ngram, use_lookahead, use_dynamic_K, use_tree, use_entropy_k)
_CONFIG_FLAGS: dict[str, tuple[bool, bool, bool, bool, bool]] = {
    "jacobi":         (False, False, False, False, False),
    "+ngram":         (True,  False, False, False, False),
    "+lookahead":     (True,  True,  False, False, False),
    "+adaptive":      (True,  True,  True,  False, False),
    "+tree":          (True,  False, False, True,  False),
    "+adaptive_tree": (True,  False, False, True,  True),
}

ALL_CONFIGS_ORDERED = ["baseline", "jacobi", "+ngram", "+lookahead", "+adaptive",
                       "+tree", "+adaptive_tree"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="A/B benchmark: baseline vs. Jacobi decode configurations (v4)"
    )
    p.add_argument("--checkpoint", type=str, default="",
                   help="Model checkpoint (.pt or .npz). Empty = auto-resolve.")
    p.add_argument("--tokenizer", type=str,
                   default=os.path.join(_INF_DIR, "tokenizer"))
    p.add_argument("--num-prompts", type=int, default=10,
                   help="Number of prompts to benchmark.")
    p.add_argument("--max-new-tokens", type=int, default=150,
                   help="Tokens to generate per prompt per config.")
    p.add_argument("--K", type=int, default=32,
                   help="Jacobi window size. Default 32.")
    p.add_argument("--ngram-n", type=int, default=4)
    p.add_argument("--min-ngram-n", type=int, default=1)
    p.add_argument("--no-warm-prompt", dest="warm_prompt", action="store_false", default=True)
    p.add_argument("--lookahead-len", type=int, default=8)
    p.add_argument("--min-lookahead-hit", type=int, default=4)
    # P0 tree attention args
    p.add_argument("--tree-branches", type=int, default=3,
                   help="Number of branches in tree attention (P0). Default 3.")
    p.add_argument("--tree-depth", type=int, default=None,
                   help="Depth per tree branch. None = K // tree-branches.")
    # P1 entropy-adaptive K args
    p.add_argument("--entropy-low", type=float, default=0.5,
                   help="Entropy threshold below which K=K_max is used (high confidence).")
    p.add_argument("--entropy-high", type=float, default=1.5,
                   help="Entropy threshold above which K=K_min is used (low confidence).")
    # Other args
    p.add_argument("--prompts-file", type=str, default="")
    p.add_argument("--configs", type=str,
                   default="baseline,+ngram,+tree,+adaptive_tree",
                   help="Comma-separated list of configs. Available: " +
                        ", ".join(ALL_CONFIGS_ORDERED))
    p.add_argument("--K-values", type=str, default="4,8,16,24,32")
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--quantize", type=int, default=4, choices=[0, 4, 8])
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--router-temp", type=float, default=0.5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--vocab-size", type=int, default=32007)
    p.add_argument("--fast-sample", dest="fast_sample", action="store_true", default=True)
    p.add_argument("--no-fast-sample", dest="fast_sample", action="store_false")
    p.add_argument("--no-penalties", dest="no_penalties", action="store_true", default=True)
    p.add_argument("--raw-prompt", action="store_true")
    args = p.parse_args()

    # Fake attrs used by benchmark helpers
    args.temp = 0.8
    args.top_k = 40
    args.top_p = 0.9
    args.min_p = 0.05
    args.rep_pen = 1.0
    args.pres_pen = 0.0
    args.freq_pen = 0.0
    args.use_parallel_scan = True
    args.inference_type = "throughput"
    args.no_compile_prefill = False
    args.eager_decode = False
    args.full_decode_compile = False
    args.no_materialize_caches = False
    args.no_outer_compile = True
    args.lookahead_router = False
    args.tucker_einsum_fuse = True
    args.tucker_amx_fuse = False
    args.tucker_scalar_fuse = False
    args.fused_mamba_mixer = False
    args.fused_mamba_mixer_v3 = False
    args.fused_sample_metal = False
    args.fused_sample_metal_v2 = False
    if args.no_penalties:
        args.rep_pen = 1.0
        args.pres_pen = 0.0
        args.freq_pen = 0.0

    # ── Tokenizer ───────────────────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("pip install transformers") from e

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = args.vocab_size if args.vocab_size > 0 else len(tokenizer)

    # ── Model ────────────────────────────────────────────────────────────────
    compute_dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    target_dtype = compute_dtype_map[args.dtype]

    config = Mamba3Config(
        d_model=768, d_state=64, d_head=64, expand=2, num_layers=6,
        mimo_rank=4, num_kv_heads=4, use_parallel_scan=True, chunk_size=64,
        use_kmoe=True, kmoe_num_experts=8, kmoe_top_k=2,
        kmoe_r1=32, kmoe_r2=512, kmoe_r3=256, ffn_expand=6,
    )
    config.lookahead_router = False
    config.tucker_einsum_fuse = True
    config.tucker_amx_fuse = False
    config.tucker_scalar_fuse = False
    config.tucker_full_fuse = False
    config.fused_mamba_mixer = False
    config.fused_mamba_mixer_v3 = False

    model = Mamba3LanguageModel(config, vocab_size)
    resolved, kind = resolve_mlx_checkpoint(args.checkpoint, repo_root=_REPO_ROOT)
    if resolved is None or kind == "none":
        print("[warn] No checkpoint found — using random weights (results meaningless).")
    else:
        print(f"Loading: {resolved} ({kind})")
        strict_load_and_convert(model, resolved)

    model.apply(lambda x: x.astype(target_dtype))
    if args.quantize > 0:
        print(f"Quantizing to {args.quantize}-bit …")
        nn.quantize(model, group_size=64, bits=args.quantize)
    mx.eval(model.parameters())
    _invalidate_tucker_caches(model)

    router_temp = mx.array(args.router_temp, dtype=target_dtype)
    max_cache_len = args.seq_len + args.max_new_tokens + 8

    # Parse K_values
    K_values: tuple[int, ...] = tuple(int(k) for k in args.K_values.split(",") if k.strip())
    K = args.K
    if K not in K_values:
        K_values = tuple(sorted(set(K_values) | {K}))

    # Tree depth
    tree_branch_depth = args.tree_depth  # None → K // tree_branches in each round

    # ── Per-layer decode compilation ─────────────────────────────────────────
    attach_decode_compilation(
        model, max_cache_len=max_cache_len,
        kv_dtype=target_dtype, compile_decode=True,
    )

    # ── Per-layer verify compilation ──────────────────────────────────────────
    print(f"Compiling per-layer verify kernels for K ∈ {K_values} …", flush=True)
    attach_verify_compilation(
        model, K_values=K_values,
        max_cache_len=max_cache_len, kv_dtype=target_dtype,
    )
    print("Verify kernels compiled.\n", flush=True)

    # ── Prefill function ──────────────────────────────────────────────────────
    def prefill_forward(x: Any, rt: Any):
        return model(x, caches=None, seq_pos=None, router_temp=rt)

    run_prefill = mx.compile(prefill_forward)

    # ── Warmup prefill ────────────────────────────────────────────────────────
    print(f"Warmup ({args.warmup} pass(es)) …", flush=True)
    warmup_ids = _build_prompt_ids(
        tokenizer, "Warmup pass.", args.seq_len, chatml_user=not args.raw_prompt
    )
    x_warm = mx.array([warmup_ids], dtype=mx.int32)
    _lw, _cw = run_prefill(x_warm, router_temp)
    mx.eval(_lw, _cw)
    _cw_pad = _pad_transformer_caches(_cw, max_cache_len)
    mx.eval(_cw_pad)

    # ── Build compiled verify functions ───────────────────────────────────────
    print("Building compiled verify fns (including tree branch depth) …", flush=True)
    compiled_verify_fns = build_compiled_verify_fns(
        model, router_temp, K_values,
        caches_for_warmup=_cw_pad,
        prompt_len=len(warmup_ids),
        tree_branch_depth=tree_branch_depth,
        num_tree_branches=args.tree_branches,
    )
    compiled_single = compiled_verify_fns[1]

    for _ in range(max(0, args.warmup - 1)):
        _lw2, _cw2 = run_prefill(x_warm, router_temp)
        mx.eval(_lw2, _cw2)
    del _lw, _cw, _cw_pad
    print("Warmup done.\n", flush=True)

    # ── Prompts ───────────────────────────────────────────────────────────────
    if args.prompts_file:
        raw_prompts = _load_prompts(args.prompts_file)
    else:
        raw_prompts = BUILTIN_PROMPTS

    prompts = raw_prompts[: args.num_prompts]
    if not prompts:
        raise SystemExit("No prompts available.")

    print(f"Running {len(prompts)} prompt(s) × each config, {args.max_new_tokens} tokens/prompt")
    print(f"Tree: branches={args.tree_branches}, depth={tree_branch_depth or 'K//B'}")
    print(f"Entropy thresholds: low={args.entropy_low}, high={args.entropy_high}\n")

    # ── Config selection ──────────────────────────────────────────────────────
    wanted = {c.strip() for c in args.configs.split(",")}
    run_configs = [c for c in ALL_CONFIGS_ORDERED if c in wanted]
    if not run_configs:
        raise SystemExit(f"No valid configs in: {args.configs}. "
                         f"Available: {', '.join(ALL_CONFIGS_ORDERED)}")

    # ── Accumulate results across prompts ─────────────────────────────────────
    agg: dict[str, dict] = {
        c: {
            "tps_list": [], "n_tokens": 0, "decode_s": 0.0,
            "arl_list": [], "la_rate_list": [],
            "tree_branch_wins_agg": None,
            "mean_entropy_list": [],
        }
        for c in run_configs
    }

    for p_idx, raw in enumerate(prompts, 1):
        user_text = raw.strip()
        prompt_ids = _build_prompt_ids(
            tokenizer, user_text, args.seq_len, chatml_user=not args.raw_prompt
        )
        short = user_text[:60] + ("…" if len(user_text) > 60 else "")
        print(f"[{p_idx:2d}/{len(prompts)}] {short}", flush=True)

        for cfg in run_configs:
            if cfg == "baseline":
                r = _run_baseline(
                    model, run_prefill, prompt_ids, max_cache_len,
                    router_temp, args.max_new_tokens, args, compiled_single,
                )
            else:
                flags = _CONFIG_FLAGS[cfg]
                use_ng, use_la, use_dyn, use_tr, use_ent = flags
                r = _run_jacobi_config(
                    cfg,
                    model=model,
                    run_prefill=run_prefill,
                    prompt_ids=prompt_ids,
                    max_cache_len=max_cache_len,
                    router_temp=router_temp,
                    max_new_tokens=args.max_new_tokens,
                    sample_args=args,
                    compiled_verify_fns=compiled_verify_fns,
                    K=K,
                    K_values=K_values,
                    use_ngram=use_ng,
                    use_lookahead=use_la,
                    use_dynamic_K=use_dyn,
                    use_tree=use_tr,
                    num_tree_branches=args.tree_branches,
                    tree_branch_depth=tree_branch_depth,
                    use_entropy_k=use_ent,
                    entropy_low=args.entropy_low,
                    entropy_high=args.entropy_high,
                    ngram_n=args.ngram_n,
                    min_ngram_n=args.min_ngram_n,
                    warm_prompt=args.warm_prompt,
                    lookahead_max_len=args.lookahead_len,
                    min_lookahead_hit=args.min_lookahead_hit,
                )

            a = agg[cfg]
            a["tps_list"].append(r["tps"])
            a["n_tokens"] += r["n_tokens"]
            a["decode_s"] += r["decode_s"]
            if r["arl"] is not None:
                a["arl_list"].append(r["arl"])
            if r["la_hit_rate"] is not None:
                a["la_rate_list"].append(r["la_hit_rate"])
            if r.get("mean_entropy") is not None:
                a["mean_entropy_list"].append(r["mean_entropy"])
            # Accumulate tree branch wins
            bwins = r.get("tree_branch_wins")
            if bwins:
                if a["tree_branch_wins_agg"] is None:
                    a["tree_branch_wins_agg"] = [0] * len(bwins)
                for i, w in enumerate(bwins):
                    if i < len(a["tree_branch_wins_agg"]):
                        a["tree_branch_wins_agg"][i] += w

            bwins_str = _fmt_branch_wins(r.get("tree_branch_wins"))
            ent_str = f"  ent={r.get('mean_entropy', 0.0):.2f}" if r.get("mean_entropy") is not None else ""
            print(
                f"  {cfg:16s}  {r['tps']:6.1f} tok/s"
                + (f"  ARL={r['arl']:.2f}" if r["arl"] is not None else "")
                + (f"  branches={bwins_str}" if bwins_str != "—" else "")
                + ent_str,
                flush=True,
            )
        print()

    # ── Aggregate and display final table ─────────────────────────────────────
    rows = []
    for cfg in run_configs:
        a = agg[cfg]
        n = len(a["tps_list"]) or 1
        mean_tps = sum(a["tps_list"]) / n
        mean_arl = (sum(a["arl_list"]) / len(a["arl_list"])) if a["arl_list"] else None
        mean_la = (sum(a["la_rate_list"]) / len(a["la_rate_list"])) if a["la_rate_list"] else None
        mean_ent = (sum(a["mean_entropy_list"]) / len(a["mean_entropy_list"])) if a["mean_entropy_list"] else None
        rows.append({
            "config": cfg,
            "tps": mean_tps,
            "decode_s": a["decode_s"],
            "n_tokens": a["n_tokens"],
            "arl": mean_arl,
            "la_hit_rate": mean_la,
            "tree_branch_wins": a["tree_branch_wins_agg"],
            "mean_entropy": mean_ent,
        })

    _print_table(rows, run_configs)

    # Plain-text summary for reports
    print("Raw averages (copy-paste friendly):")
    baseline_tps = next((r["tps"] for r in rows if r["config"] == "baseline"), None)
    print(f"{'Config':<18} {'tok/s':>7} {'speedup':>8} {'ARL':>6} {'Branch-wins':>20} {'MeanEnt':>8}")
    print("─" * 70)
    for r in rows:
        sp = f"{r['tps'] / baseline_tps:.2f}x" if baseline_tps else "—"
        arl = f"{r['arl']:.2f}" if r["arl"] is not None else "—"
        bw = _fmt_branch_wins(r.get("tree_branch_wins"))
        ent = f"{r.get('mean_entropy', 0.0):.2f}" if r.get("mean_entropy") is not None else "—"
        print(f"{r['config']:<18} {r['tps']:>7.1f} {sp:>8} {arl:>6} {bw:>20} {ent:>8}")

    # Experimental verdict
    print("\n─── Experiment verdict ───")
    if baseline_tps:
        for r in rows:
            if r["config"] not in ("baseline", "jacobi"):
                sp = r["tps"] / baseline_tps
                verdict = "✓ BREAKTHROUGH" if sp >= 3.0 else ("✓ TARGET MET" if sp >= 2.5 else "─ below target")
                print(f"  {r['config']:<18}  {sp:.2f}×  {verdict}")


if __name__ == "__main__":
    main()
