#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External Draft Model Speculative Decode Benchmark
==================================================
Loads the 13M TinyDraft model (checkpoints/darf-model) as draft and the
417M Mamba3 hybrid as target. Sweeps K (draft tokens per round) over
[1, 2, 4, 6, 8] and prints tok/s, accept rate, and speedup vs baseline.

Usage (from repo root):
  .venv/bin/python inference/benchmark_spec_external.py
  .venv/bin/python inference/benchmark_spec_external.py --k-values 1 2 4 6 8 --n-tokens 200
  .venv/bin/python inference/benchmark_spec_external.py --prompt "Your prompt" --no-baseline
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

_INF_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_INF_DIR, "lib")
_REPO_ROOT = os.path.abspath(os.path.join(_INF_DIR, ".."))
for _p in (_LIB_DIR, _INF_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer

from mlx_hybrid_infer import (
    Mamba3Config,
    Mamba3LanguageModel,
    maybe_export_npz_sidecar_after_pt_load,
    resolve_mlx_checkpoint,
    strict_load_and_convert,
)
from benchmark_mlx import (
    _apply_inference_type,
    _build_prompt_ids,
    _materialize_cache_tree,
    _pad_transformer_caches,
    _invalidate_tucker_caches,
)
from stream_mlx import make_compiled_decode_step
from draft_model import TinyDraftModel, load_draft_model

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = os.path.join(_REPO_ROOT, "checkpoints", "latest_sft_cot_model.npz")
DEFAULT_DRAFT_CKPT = os.path.join(_REPO_ROOT, "checkpoints", "darf-model")
DEFAULT_TOKENIZER = os.path.join(_INF_DIR, "tokenizer")
DEFAULT_K_VALUES = [1, 2, 4, 6, 8]
DEFAULT_N_TOKENS = 200
DEFAULT_WARMUP = 1
DEFAULT_PROMPT = (
    "who are you?"
)
MAX_CACHE_LEN = 512


# ──────────────────────────────────────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────────────────────────────────────

def _build_main_model(args: Any) -> Mamba3LanguageModel:
    import argparse as _ap
    fake = _ap.Namespace(
        inference_type="throughput",
        no_compile_prefill=False,
        eager_decode=False,
        no_materialize_caches=False,
        use_parallel_scan=True,
    )
    vocab_size = 32007
    config = Mamba3Config(
        d_model=768, d_state=64, d_head=64, expand=2,
        num_layers=6, mimo_rank=4, num_kv_heads=4,
        use_parallel_scan=True, chunk_size=64,
        use_kmoe=True, kmoe_num_experts=8, kmoe_top_k=2,
        kmoe_r1=32, kmoe_r2=512, kmoe_r3=256, ffn_expand=6,
    )
    config.tucker_einsum_fuse = False
    config.tucker_full_fuse = False
    config.tucker_scalar_fuse = False
    config.lookahead_router = False
    model = Mamba3LanguageModel(config, vocab_size)

    resolved, kind = resolve_mlx_checkpoint(args.checkpoint, repo_root=_REPO_ROOT)
    if resolved is None:
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    print(f"[main model] loading {resolved} ({kind})")
    strict_load_and_convert(model, resolved)
    if kind == "pt":
        maybe_export_npz_sidecar_after_pt_load(model, resolved)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def _prefill(
    model: Mamba3LanguageModel,
    prompt_ids: list[int],
    router_temp: mx.array,
) -> tuple[mx.array, Any]:
    """Run compiled prefill; return (logits, materialized_caches)."""
    x = mx.array(prompt_ids, dtype=mx.int32).reshape(1, -1)

    def _fwd(xp, rt):
        return model(xp, router_temp=rt)

    compiled_prefill = mx.compile(_fwd)
    logits, caches = compiled_prefill(x, router_temp)
    mx.eval(logits, caches)
    caches = _materialize_cache_tree(caches)
    mx.eval(caches)
    caches = _pad_transformer_caches(caches, MAX_CACHE_LEN)
    mx.eval(caches)
    return logits, caches


def _draft_prefill(
    draft: TinyDraftModel,
    prompt_ids: list[int],
) -> tuple[mx.array, tuple]:
    """Run draft model over full prompt; return (logits, kv_cache)."""
    x = mx.array(prompt_ids, dtype=mx.int32).reshape(1, -1)
    logits, kv = draft(x, kv_cache=None)
    mx.eval(logits, kv)
    return logits, kv


# ──────────────────────────────────────────────────────────────────────────────
# Baseline (no spec)
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline(
    model: Mamba3LanguageModel,
    prompt_ids: list[int],
    n_tokens: int,
    router_temp: mx.array,
    warmup: int,
) -> dict:
    compiled_step = make_compiled_decode_step(model, router_temp)

    def _one_run():
        _invalidate_tucker_caches(model)
        logits, caches = _prefill(model, prompt_ids, router_temp)
        tok = int(mx.argmax(logits[0, -1, :]).item())
        x = mx.array([[tok]], dtype=mx.int32)
        t0 = time.perf_counter()
        for step in range(n_tokens - 1):
            pos = mx.array(len(prompt_ids) + step, dtype=mx.int32)
            logits_d, caches = compiled_step(x, caches, pos)
            tok = int(mx.argmax(logits_d[0, -1, :]).item())
            mx.eval(tok, caches)
            x = mx.array([[tok]], dtype=mx.int32)
        mx.eval(caches)
        return time.perf_counter() - t0

    for _ in range(warmup):
        _one_run()

    elapsed = _one_run()
    tps = n_tokens / elapsed
    return {"mode": "baseline", "k": 0, "tps": tps, "elapsed": elapsed,
            "n_tokens": n_tokens, "accept_rate": None, "speedup": 1.0}


# ──────────────────────────────────────────────────────────────────────────────
# External spec decode
# ──────────────────────────────────────────────────────────────────────────────

def _draft_k_tokens(
    draft: TinyDraftModel,
    seed_tok: int,
    commit_kv: tuple,
    k: int,
) -> tuple[list[int], tuple]:
    """
    Autoregressively draft k tokens from seed_tok, starting from commit_kv cache.
    Returns (draft_tokens, final_kv_at_end_of_draft).
    """
    draft_tokens: list[int] = []
    x = mx.array([[seed_tok]], dtype=mx.int32)
    kv = commit_kv
    for _ in range(k):
        logits, kv = draft(x, kv_cache=kv)
        mx.eval(logits, kv)
        tok = int(mx.argmax(logits[0, -1, :]).item())
        draft_tokens.append(tok)
        x = mx.array([[tok]], dtype=mx.int32)
    return draft_tokens, kv


def _replay_draft_cache(
    draft: TinyDraftModel,
    tokens: list[int],
    base_kv: tuple,
) -> tuple:
    """Replay a list of tokens through draft model from base_kv; return final kv."""
    kv = base_kv
    for tok in tokens:
        x = mx.array([[tok]], dtype=mx.int32)
        _, kv = draft(x, kv_cache=kv)
        mx.eval(kv)
    return kv


def run_spec_external(
    model: Mamba3LanguageModel,
    draft: TinyDraftModel,
    prompt_ids: list[int],
    n_tokens: int,
    k: int,
    router_temp: mx.array,
    warmup: int,
    baseline_tps: float,
) -> dict:
    K = k

    # Compile the verify step (fixed shape: (1, K))
    def _verify(x_k: mx.array, caches: Any, pos: mx.array):
        return model(x_k, caches=caches, seq_pos=pos, router_temp=router_temp)

    compiled_verify = mx.compile(_verify)
    compiled_single = make_compiled_decode_step(model, router_temp)

    def _one_run():
        _invalidate_tucker_caches(model)

        # --- Prefill both models ---
        logits_p, caches = _prefill(model, prompt_ids, router_temp)
        _, draft_kv_commit = _draft_prefill(draft, prompt_ids)

        # --- First token from prefill ---
        tok0 = int(mx.argmax(logits_p[0, -1, :]).item())
        pos = len(prompt_ids)

        # Advance main model one step
        logits_1, caches = compiled_single(
            mx.array([[tok0]], dtype=mx.int32), caches, mx.array(pos, dtype=mx.int32)
        )
        mx.eval(logits_1, caches)
        current_logit_row = logits_1[0, -1, :]
        pos += 1

        # Advance draft model past tok0
        _, draft_kv_commit = draft(mx.array([[tok0]], dtype=mx.int32), kv_cache=draft_kv_commit)
        mx.eval(draft_kv_commit)

        # Warmup verify graph for this K
        _dummy_x = mx.array([[tok0] * K], dtype=mx.int32)
        _lv, _cv = compiled_verify(_dummy_x, caches, mx.array(pos, dtype=mx.int32))
        mx.eval(_lv, _cv)

        generated = [tok0]
        spec_rounds = spec_accepted = spec_proposed = 0

        t0 = time.perf_counter()
        last_tok = tok0

        while len(generated) < n_tokens:
            remaining = n_tokens - len(generated)
            K_this = min(K, remaining)

            # ── Draft ─────────────────────────────────────────────────────────
            draft_tokens, draft_kv_end = _draft_k_tokens(draft, last_tok, draft_kv_commit, K_this)
            pad_tok = draft_tokens[-1] if draft_tokens else last_tok
            draft_padded = draft_tokens + [pad_tok] * (K - K_this)

            spec_rounds += 1
            spec_proposed += K_this

            # ── Verify ────────────────────────────────────────────────────────
            x_v = mx.array([draft_padded], dtype=mx.int32)
            logits_v, verify_caches = compiled_verify(x_v, caches, mx.array(pos, dtype=mx.int32))
            mx.eval(logits_v, verify_caches)

            # ── Accept/reject (greedy) ─────────────────────────────────────────
            accepted = 0
            correction: Optional[int] = None
            for i in range(K_this):
                ref_row = current_logit_row if i == 0 else logits_v[0, i - 1, :]
                pred = int(mx.argmax(ref_row).item())
                if pred == draft_tokens[i]:
                    accepted += 1
                else:
                    correction = pred
                    break

            spec_accepted += accepted

            if accepted == K_this:
                # All accepted
                consume = draft_tokens[:K_this]
                caches = verify_caches
                current_logit_row = logits_v[0, K_this - 1, :]
                last_tok = draft_tokens[-1]
                pos += K_this
                # Draft cache advanced to end of accepted tokens
                draft_kv_commit = draft_kv_end
            else:
                # Partial accept + correction
                assert correction is not None
                consume = draft_tokens[:accepted] + [correction]
                # Replay main model for accepted tokens + correction
                cur_c = caches
                cur_pos = pos
                for t in consume[:-1]:
                    _, cur_c = compiled_single(
                        mx.array([[t]], dtype=mx.int32), cur_c, mx.array(cur_pos, dtype=mx.int32)
                    )
                    cur_pos += 1
                logits_cor, cur_c = compiled_single(
                    mx.array([[correction]], dtype=mx.int32), cur_c, mx.array(cur_pos, dtype=mx.int32)
                )
                mx.eval(logits_cor, cur_c)
                caches = cur_c
                current_logit_row = logits_cor[0, -1, :]
                last_tok = correction
                pos += len(consume)
                # Replay draft model to match accepted state
                draft_kv_commit = _replay_draft_cache(draft, consume, draft_kv_commit)

            for t in consume:
                generated.append(t)
                if len(generated) >= n_tokens:
                    break

        elapsed = time.perf_counter() - t0
        return elapsed, spec_rounds, spec_accepted, spec_proposed

    # Warmup
    for _ in range(warmup):
        _one_run()

    elapsed, rounds, accepted, proposed = _one_run()
    actual_tokens = n_tokens - 1  # subtract first prefill token
    tps = actual_tokens / elapsed
    accept_rate = 100.0 * accepted / max(proposed, 1)
    speedup = tps / baseline_tps if baseline_tps > 0 else float("nan")

    return {
        "mode": "external_spec",
        "k": K,
        "tps": tps,
        "elapsed": elapsed,
        "n_tokens": actual_tokens,
        "accept_rate": accept_rate,
        "speedup": speedup,
        "spec_rounds": rounds,
        "accepted": accepted,
        "proposed": proposed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="External draft speculative decode benchmark")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--draft-checkpoint", default=DEFAULT_DRAFT_CKPT)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--raw-prompt", action="store_true",
                   help="Skip ChatML wrapping; use prompt verbatim")
    p.add_argument("--n-tokens", type=int, default=DEFAULT_N_TOKENS,
                   help="Total tokens to generate per run (default 200)")
    p.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
                   help="Draft token counts to sweep (default: 1 2 4 6 8)")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                   help="Warmup rounds before timing (default 1)")
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip baseline measurement (use 0 as reference)")
    p.add_argument("--router-temp", type=float, default=0.5)
    p.add_argument("--output-json", default="",
                   help="Write results JSON to this path")
    args = p.parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    if args.raw_prompt:
        prompt_ids = list(tokenizer.encode(args.prompt, add_special_tokens=False))
    else:
        prompt_ids = list(tokenizer.encode(
            f"<|im_start|>user\n{args.prompt.strip()}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        ))
    print(f"Prompt: {len(prompt_ids)} tokens")

    # Load models
    print("\nLoading main model...")
    main_model = _build_main_model(args)

    print("\nLoading draft model...")
    draft_model = load_draft_model(args.draft_checkpoint)
    from mlx.utils import tree_flatten
    draft_params = sum(v.size for _, v in tree_flatten(draft_model.parameters()))
    print(f"  Draft model: {draft_params/1e6:.1f}M params")

    router_temp = mx.array(args.router_temp, dtype=mx.bfloat16)
    results: list[dict] = []

    # Baseline
    baseline_tps = 0.0
    if not args.no_baseline:
        print(f"\n{'='*60}")
        print("BASELINE (no speculative decode)")
        print(f"{'='*60}")
        res = run_baseline(main_model, prompt_ids, args.n_tokens, router_temp, args.warmup)
        baseline_tps = res["tps"]
        results.append(res)
        print(f"  {res['n_tokens']} tokens in {res['elapsed']:.2f}s → {res['tps']:.1f} tok/s")

    # Sweep K
    k_values = sorted(set(args.k_values))
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"EXTERNAL SPEC  K={k}  (draft: darf-model 13M)")
        print(f"{'='*60}")
        res = run_spec_external(
            main_model, draft_model, prompt_ids,
            args.n_tokens, k, router_temp, args.warmup, baseline_tps,
        )
        results.append(res)
        print(
            f"  {res['n_tokens']} tokens in {res['elapsed']:.2f}s → {res['tps']:.1f} tok/s"
            f"  |  accept={res['accept_rate']:.1f}%"
            f"  |  speedup={res['speedup']:.2f}×"
            f"  |  rounds={res['spec_rounds']}  accepted={res['accepted']}/{res['proposed']}"
        )

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Mode':<22} {'K':>3} {'tok/s':>8} {'accept%':>9} {'speedup':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        mode = r["mode"]
        k_str = str(r["k"]) if r["k"] > 0 else "-"
        ar = f"{r['accept_rate']:.1f}%" if r["accept_rate"] is not None else "   -"
        sp = f"{r['speedup']:.2f}×" if r["speedup"] is not None else "   -"
        print(f"{mode:<22} {k_str:>3} {r['tps']:>8.1f} {ar:>9} {sp:>9}")

    # Theory: break-even accept rate
    if baseline_tps > 0:
        print(f"\nTheory (greedy): break-even α = t_draft/t_verify ≈ 0–5%")
        print("Any accept rate above that should yield speedup with a fast-enough draft model.")

    # JSON output
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written → {args.output_json}")


if __name__ == "__main__":
    main()
