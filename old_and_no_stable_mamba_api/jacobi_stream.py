#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jacobi +ngram streaming inference — real-time token display + speed/ARL report.

Best config locked-in: +ngram K=32, 2.32-2.36× speedup across all prompt types.
K, ngram-n and sampling params are overridable via CLI.

Usage:
  # single prompt
  python inference/ --prompt "What is 12 × 15?"

  # interactive REPL (model loaded once, ChatML per turn)
  python inference/ -i

  # K=48 experiment
  python inference/ --K 48 --prompt "..."

  # quality mode (stochastic, 8-bit)
  python inference/ --temp 0.3 --quantize 8 --prompt "..."
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from typing import Any

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
from jacobi_enhanced import (
    enhanced_jacobi_stream,
    build_compiled_verify_fns,
    build_compiled_verify_with_intra_fns,
    _deep_copy_caches,
)

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn, Progress, SpinnerColumn,
        TaskProgressColumn, TextColumn, TimeElapsedColumn,
    )
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── EOS tokens that stop generation ─────────────────────────────────────────
# Only stop on <|im_end|> (ChatML turn-end). <\s> is the SentencePiece EOS and
# should NOT be a stop signal here: if the model emits it mid-response (common
# with quant=0 / full-precision), we want to see the broken output rather than
# silently truncating to 1 token.  Add token 2 back explicitly if you need it.
_STOP_STRINGS = ["<|im_end|>"]

# CoT structure tags we want to display even when skip_special_tokens=True
_COT_TAGS = ["<think>", "</think>", "<final>", "</final>"]


def _make_chatml(user_text: str) -> str:
    return f"<|im_start|>user\n{user_text.strip()}<|im_end|>\n<|im_start|>assistant\n"


def _build_stop_ids(tokenizer: Any, *, enabled: bool) -> set[int]:
    """Stop-token ids for ChatML turns (eos + <|im_end|>). Empty when disabled."""
    if not enabled:
        return set()
    ids: set[int] = set()
    eos_raw = getattr(tokenizer, "eos_token_id", None)
    if eos_raw is not None:
        if isinstance(eos_raw, (list, tuple)) and eos_raw:
            ids.add(int(eos_raw[0]))
        else:
            ids.add(int(eos_raw))
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end = None
    if isinstance(im_end, int) and im_end >= 0:
        unk = getattr(tokenizer, "unk_token_id", None)
        if unk is None or im_end != unk:
            ids.add(im_end)
    for s in _STOP_STRINGS:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def _build_cot_markers(tokenizer: Any) -> dict[int, str]:
    """
    Build {token_id: display_string} for CoT structure tags.

    When skip_special_tokens=True, <think>/<final> etc. are silently dropped.
    We intercept their token IDs and inject the tag text manually so the
    streaming output shows the full CoT structure without <unk> spam.
    """
    markers: dict[int, str] = {}
    unk_id = getattr(tokenizer, "unk_token_id", -1)
    for tag in _COT_TAGS:
        # Try as a plain string (may be a regular vocab token)
        enc = tokenizer.encode(tag, add_special_tokens=False)
        if len(enc) == 1 and enc[0] != unk_id:
            markers[enc[0]] = tag
            continue
        # Try via convert_tokens_to_ids (registered special tokens)
        tid = tokenizer.convert_tokens_to_ids(tag)
        if isinstance(tid, int) and tid != unk_id:
            markers[tid] = tag
    return markers


# ── Core stream function ─────────────────────────────────────────────────────

def run_jacobi_stream(
    *,
    model: Any,
    run_prefill: Any,
    tokenizer: Any,
    prompt_text: str,
    router_temp: Any,
    max_cache_len: int,
    sample_args: Any,
    compiled_verify_fns: dict,
    compiled_verify_with_intra_fns: dict | None = None,
    K: int,
    K_values: tuple,
    use_dynamic_K: bool = True,
    use_ngram: bool = True,
    sys_prompt: str = "",
    max_new_tokens: int,
    ngram_n: int,
    min_ngram_n: int,
    seq_len: int,
    stop_ids: set[int],
    cot_markers: dict[int, str],
    ar_warmup_tokens: int = 20,
    show_progress: bool = True,
    debug: bool = False,
) -> tuple[str, float, float, float | None, int]:
    """
    Run one prompt through enhanced_jacobi_stream.
    Returns (generated_text, tok_per_sec, decode_seconds, arl, n_tokens, stats_dict).

    Decode strategy:
      - skip_special_tokens=True  →  no <unk> spam from byte-fallback tokens
      - cot_markers intercept     →  <think>/<final> tags injected manually
    """
    if sys_prompt.strip():
        full_chatml = (
            f"<|im_start|>system\n{sys_prompt.strip()}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            prompt_ids = list(tokenizer.encode(full_chatml, add_special_tokens=False))
        except TypeError:
            prompt_ids = list(tokenizer.encode(full_chatml))
        if seq_len > 0 and len(prompt_ids) > seq_len:
            prompt_ids = prompt_ids[:seq_len]
    else:
        prompt_ids = _build_prompt_ids(
            tokenizer, prompt_text, seq_len, chatml_user=True,
        )

    x_prefill = mx.array([prompt_ids], dtype=mx.int32)
    logits, caches = run_prefill(x_prefill, router_temp)
    mx.eval(logits, caches)
    caches = _materialize_cache_tree(caches)
    mx.eval(caches)
    caches = _pad_transformer_caches(caches, max_cache_len)
    mx.eval(caches)

    stats: dict = {}
    generated_ids: list[int] = []   # all generated token IDs (for count)
    visible_ids: list[int] = []     # non-special IDs decoded incrementally
    prev_text = ""
    ttft: float | None = None
    t0 = time.perf_counter()
    # All special token IDs (includes byte-fallback, <s>, </s>, <unk>, im_start/end, etc.)
    # CoT markers are handled separately; everything else is skipped without adding to visible_ids.
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    token_iter = enhanced_jacobi_stream(
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
        use_lookahead=False,
        use_dynamic_K=use_dynamic_K,
        use_tree=False,
        use_entropy_k=False,
        ngram_n=ngram_n,
        min_ngram_n=min_ngram_n,
        warm_prompt=True,
        ar_warmup_tokens=ar_warmup_tokens,
        stop_token_ids=frozenset(stop_ids) if stop_ids else None,
        stats=stats,
        compiled_verify_fns=compiled_verify_fns,
        compiled_verify_with_intra_fns=compiled_verify_with_intra_fns or {},
        compiled_single_fn=compiled_verify_fns.get(1),
        materialize_cache_tree_fn=_materialize_cache_tree,
        pad_transformer_caches_fn=_pad_transformer_caches,
    )

    # ── CoT display state machine ─────────────────────────────────────────────
    # Suppresses <think>...</think> block content from the display (the model
    # often generates blank lines or garbage there).  Duplicate </think> tags
    # are ignored safely.  <final>/<final> delimiters are hidden.
    _in_think = False

    def _decode_regular(tid: int) -> str:
        """Incremental SentencePiece decode, appending tid to visible_ids."""
        nonlocal prev_text
        visible_ids.append(tid)
        try:
            full = tokenizer.decode(
                visible_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if full.startswith(prev_text):
                chunk = full[len(prev_text):]
            else:
                chunk = tokenizer.decode(
                    [tid],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            prev_text = full
            return chunk
        except Exception:
            return ""

    _think_tok_count: int = 0   # tokens generated inside the current <think> block
    _THINK_LIMIT = 120          # show at most this many raw thinking tokens per block

    def _tok_chunk(tid: int) -> str:
        """Return display chunk for one token.

        CoT structure:
          <think>   → invisible tag; thinking content shown up to _THINK_LIMIT tokens
          </think>  → invisible tag; emits a visual separator (─── line) to mark end of thinking
          <final>   → invisible tag; content between <final>…</final> is the answer (shown)
          </final>  → invisible tag
          All other special tokens → suppressed
        """
        nonlocal _in_think, _think_tok_count
        if tid in cot_markers:
            tag = cot_markers[tid]
            if tag == "<think>":
                _in_think = True
                _think_tok_count = 0
                return ""
            elif tag == "</think>":
                was_thinking = _in_think
                _in_think = False
                _think_tok_count = 0
                # visual break between thinking and answer
                return "\n─────────────────────────────────────────\n" if was_thinking else ""
            else:
                return ""  # drop <final> / </final> tag tokens; content between them is shown
        elif tid in special_ids:
            return ""
        else:
            if _in_think:
                _think_tok_count += 1
                if _think_tok_count == _THINK_LIMIT + 1:
                    return "...\n"  # truncation marker
                if _think_tok_count > _THINK_LIMIT:
                    return ""       # suppress the rest of the thinking block
            return _decode_regular(tid)

    if HAS_RICH and show_progress:
        print("\nAssistant:", flush=True)

        con = Console(highlight=False)
        with Progress(
            TextColumn("[bold cyan]Jacobi[/bold cyan]"),
            SpinnerColumn(style="cyan"),
            TextColumn("{task.fields[tok_s]:>7.1f} tok/s"),
            BarColumn(bar_width=20),
            TimeElapsedColumn(),
            console=con, transient=False,
            redirect_stdout=False, redirect_stderr=False,
            refresh_per_second=12,
        ) as prog:
            task = prog.add_task("", total=max_new_tokens, tok_s=0.0)

            buf = ""
            wrap = max(60, 100)

            def _flush(force: bool = False) -> None:
                nonlocal buf
                while True:
                    nl = buf.find("\n")
                    if nl >= 0:
                        prog.console.print(buf[:nl], highlight=False, soft_wrap=True)
                        buf = buf[nl + 1:]
                        continue
                    if len(buf) >= wrap:
                        cut = buf.rfind(" ", 0, wrap)
                        cut = cut if cut > 0 else wrap
                        prog.console.print(buf[:cut].rstrip(), highlight=False, soft_wrap=True)
                        buf = buf[cut:].lstrip()
                        continue
                    break
                if force and buf:
                    prog.console.print(buf, highlight=False, soft_wrap=True)
                    buf = ""

            for tok in token_iter:
                mx.eval(tok)
                tid = int(tok.item())
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if tid in stop_ids:
                    break
                generated_ids.append(tid)
                chunk = _tok_chunk(tid)
                if chunk:
                    buf += chunk
                    _flush()
                elapsed = time.perf_counter() - t0
                prog.update(task, completed=len(generated_ids),
                            tok_s=len(generated_ids) / max(elapsed, 1e-9))

            _flush(force=True)
        print()
    else:
        print("\nAssistant: ", end="", flush=True)
        _dbg_count = 0
        for tok in token_iter:
            mx.eval(tok)
            tid = int(tok.item())
            if ttft is None:
                ttft = time.perf_counter() - t0
            if debug and _dbg_count < 30:
                raw_str = tokenizer.decode([tid], skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
                sst_str = tokenizer.decode([tid], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
                print(f"\n  [DBG {_dbg_count:02d}] tid={tid}"
                      f"  raw={repr(raw_str)}"
                      f"  sst={repr(sst_str)}"
                      f"  cot={cot_markers.get(tid)}"
                      f"  special={tid in special_ids}"
                      f"  stop={tid in stop_ids}", flush=True)
                _dbg_count += 1
            if tid in stop_ids:
                break
            generated_ids.append(tid)
            chunk = _tok_chunk(tid)
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
        print()

    decode_s = time.perf_counter() - t0
    n = len(generated_ids)
    tps = n / max(decode_s, 1e-9)
    arl = stats.get("arl")
    return prev_text, tps, decode_s, arl, n, stats


# ── Pretty stats footer ───────────────────────────────────────────────────────

def _print_stats(tps: float, decode_s: float, n_tok: int,
                 arl: float | None, baseline_tps: float, K: int,
                 stats: dict | None = None) -> None:
    speedup = tps / max(baseline_tps, 1.0)
    arl_str = f"{arl:.2f}" if arl is not None else "—"

    # Extra metrics from stats dict
    extra = ""
    if stats:
        ng_rate  = stats.get("ngram_hit_rate", 0.0)
        intra_r  = stats.get("intra_rounds", 0)
        replay_r = stats.get("replay_rounds", 0)
        per_rd   = stats.get("per_round_arl", [])
        if per_rd:
            arl_min = min(per_rd); arl_max = max(per_rd)
            extra = f"  ngram={ng_rate*100:.0f}%"
            if intra_r + replay_r > 0:
                extra += f"  intra={intra_r}  replay={replay_r}"
            extra += f"  ARL=[{arl_min}–{arl_max}]"

    if HAS_RICH:
        con = Console(highlight=False)
        con.print(
            f"[dim]──────────────────────────────────────────────────────────[/dim]"
        )
        con.print(
            f"[bold]{tps:>7.1f} tok/s[/bold]  "
            f"[cyan]{speedup:.2f}×[/cyan] speedup  "
            f"ARL [yellow]{arl_str}[/yellow]  "
            f"K=[green]{K}[/green]  "
            f"[dim]{n_tok} tokens  {decode_s:.2f}s{extra}[/dim]"
        )
    else:
        print(f"\n── {tps:.1f} tok/s  {speedup:.2f}× speedup  ARL={arl_str}  "
              f"K={K}  {n_tok} tok  {decode_s:.2f}s{extra} ──\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Jacobi +ngram streaming inference (best config: K=32, 2.32-2.36×)"
    )
    p.add_argument("--checkpoint", type=str, default="",
                   help="Weights .pt or .npz — empty = auto-resolve")
    p.add_argument("--tokenizer", type=str,
                   default=os.path.join(_INF_DIR, "tokenizer"))
    p.add_argument("--prompt", "-p", type=str, default="",
                   help="User prompt text (ChatML-wrapped automatically)")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="REPL mode: type prompts one at a time")
    # Jacobi hyperparams
    p.add_argument("--K", type=int, default=32,
                   help="Jacobi window K (default 32; try 48/64 to test ARL scaling)")
    p.add_argument("--no-dynamic-K", action="store_true",
                   help="Disable adaptive K (dynamic K is ON by default)")
    p.add_argument("--no-intra", action="store_true",
                   help="Disable verify+intra path (skip build of intra compiled fns)")
    p.add_argument("--no-ngram", action="store_true",
                   help="Disable n-gram draft seeding (pure carry-last-token Jacobi)")
    p.add_argument("--sys-prompt", type=str, default="",
                   help="Optional system prompt prepended before the user turn in ChatML")
    p.add_argument("--ngram-n", type=int, default=4,
                   help="Max n-gram order for seeding (default 4)")
    p.add_argument("--min-ngram-n", type=int, default=1)
    # Sampling
    p.add_argument("--temp", type=float, default=0.0,
                   help="Sampling temperature (0 = greedy)")
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_p", type=float, default=0.05)
    p.add_argument("--rep_pen", type=float, default=1.0)
    p.add_argument("--router-temp", type=float, default=0.5)
    # Model
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["fp32", "bf16", "fp16"])
    p.add_argument("--quantize", type=int, default=8, choices=[0, 4, 8],
                   help="Weight quantization bits (default 8; 4-bit degrades TuckerMoE quality)")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--ar-warmup", type=int, default=20,
                   help="Autoregressive tokens before Jacobi (seeds N-gram cache; default 20)")
    p.add_argument("--vocab-size", type=int, default=32007)
    # Display
    p.add_argument("--no-progress", action="store_true",
                   help="Plain stdout (no Rich progress bar)")
    p.add_argument("--debug", action="store_true",
                   help="Print raw token IDs for first 30 tokens (diagnosis)")
    p.add_argument("--baseline-tps", type=float, default=82.1,
                   help="Baseline tok/s for speedup calculation")
    p.add_argument(
        "--stop-on-eos",
        action="store_true",
        default=False,
        help="Stop when eos or <|im_end|> is sampled (ChatML turn end).",
    )
    p.add_argument(
        "--no-eos-stop",
        action="store_true",
        default=False,
        help="Ignore stop tokens; run until --max-new-tokens.",
    )
    args = p.parse_args()

    if not args.prompt and not args.interactive:
        p.error("Provide --prompt TEXT  or  --interactive / -i")

    # Interactive REPL: stop at turn-end unless explicitly disabled.
    if args.interactive and not args.no_eos_stop:
        args.stop_on_eos = True
    if args.no_eos_stop:
        args.stop_on_eos = False

    # ── Fake sample_args namespace ──────────────────────────────────────────
    sample_args = types.SimpleNamespace(
        temp=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        rep_pen=args.rep_pen,
        pres_pen=0.0,
        freq_pen=0.0,
        fast_sample=(args.temp == 0.0),
        no_penalties=(args.rep_pen == 1.0),
        no_materialize_caches=False,
        use_parallel_scan=True,
        inference_type="throughput",
        no_compile_prefill=False,
        eager_decode=False,
        full_decode_compile=False,
        no_outer_compile=True,
        lookahead_router=False,
        tucker_einsum_fuse=False,
        tucker_amx_fuse=False,
        tucker_scalar_fuse=False,
        tucker_full_fuse=False,
        fused_mamba_mixer=False,
        fused_mamba_mixer_v3=False,
        fused_sample_metal=False,
        fused_sample_metal_v2=False,
    )

    # ── Tokenizer ──────────────────────────────────────────────────────────
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("pip install transformers") from e

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = args.vocab_size if args.vocab_size > 0 else len(tokenizer)
    stop_ids = _build_stop_ids(tokenizer, enabled=bool(args.stop_on_eos))

    # ── Model ──────────────────────────────────────────────────────────────
    dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    target_dtype = dtype_map[args.dtype]

    config = Mamba3Config(
        d_model=768, d_state=64, d_head=64, expand=2, num_layers=6,
        mimo_rank=4, num_kv_heads=4, use_parallel_scan=True, chunk_size=64,
        use_kmoe=True, kmoe_num_experts=8, kmoe_top_k=2,
        kmoe_r1=32, kmoe_r2=512, kmoe_r3=256, ffn_expand=6,
    )
    config.lookahead_router = False
    config.tucker_einsum_fuse = False   # must match stream_mlx.py — Metal f32 vs bf16 shifts </think> logit
    config.tucker_amx_fuse = False
    config.tucker_scalar_fuse = False
    config.tucker_full_fuse = False
    config.fused_mamba_mixer = False
    config.fused_mamba_mixer_v3 = False

    model = Mamba3LanguageModel(config, vocab_size)

    resolved, kind = resolve_mlx_checkpoint(args.checkpoint, repo_root=_REPO_ROOT)
    if resolved is None or kind == "none":
        print("[warn] No checkpoint — random weights, output will be garbage.", file=sys.stderr)
    else:
        print(f"Loading {resolved} ({kind}) …", flush=True)
        strict_load_and_convert(model, resolved)

    model.apply(lambda x: x.astype(target_dtype))
    if args.quantize > 0:
        print(f"Quantizing {args.quantize}-bit …", flush=True)
        nn.quantize(model, group_size=64, bits=args.quantize)
    mx.eval(model.parameters())
    _invalidate_tucker_caches(model)

    router_temp = mx.array(args.router_temp, dtype=target_dtype)
    max_cache_len = args.seq_len + args.max_new_tokens + 8

    # ── K_values: always include 1, some smaller steps, and K ─────────────
    K = args.K
    K_base = [1, 4, 8, 16, 24, 32]
    K_values: tuple = tuple(sorted(set(K_base) | {K}))

    # NOTE: attach_verify_compilation is intentionally skipped here.
    # attach_decode_compilation is also deferred until AFTER build_compiled_verify_fns
    # (see below).  Reason: both create per-layer compiled fns for K=1 decode.
    # When they exist during the outer mx.compile trace in build_compiled_verify_fns,
    # the K=1 trace nests these inner compiled calls, causing MLX graph aliasing
    # across different K-shaped specializations → NaN / wrong logits on every
    # single-step call (AR warmup, stall fallback, batch replay).
    print(f"Building compiled verify fns K ∈ {K_values} …", flush=True)

    def _prefill_fn(x: mx.array, rt: mx.array):
        return model(x, caches=None, seq_pos=None, router_temp=rt)

    run_prefill = mx.compile(_prefill_fn)

    # ── Warmup ─────────────────────────────────────────────────────────────
    print(f"Warmup ({args.warmup} pass) …", flush=True)
    warm_ids = _build_prompt_ids(tokenizer, "warmup", args.seq_len, chatml_user=True)
    x_w = mx.array([warm_ids], dtype=mx.int32)
    _lw, _cw = run_prefill(x_w, router_temp)
    mx.eval(_lw, _cw)
    _cw_pad = _pad_transformer_caches(_cw, max_cache_len)
    mx.eval(_cw_pad)

    compiled_verify_fns = build_compiled_verify_fns(
        model, router_temp, K_values,
        caches_for_warmup=_cw_pad,
        prompt_len=len(warm_ids),
        tree_branch_depth=None,
        num_tree_branches=1,
    )

    # ── verify+intra fns (eliminates batch_replay for partial-accept) ──────
    # Built for K only (not all K_values) to keep startup time reasonable.
    # When dynamic-K uses a smaller K that has no intra fn, it falls back
    # to regular verify automatically inside enhanced_jacobi_stream.
    compiled_verify_with_intra_fns: dict = {}
    use_dynamic_K = not args.no_dynamic_K
    if not args.no_intra:
        _cw_pad_intra = _deep_copy_caches(_cw_pad)
        mx.eval(_cw_pad_intra)
        print(f"Building compiled verify+intra fns K={K} …", flush=True)
        compiled_verify_with_intra_fns = build_compiled_verify_with_intra_fns(
            model, router_temp, K_values=(K,),
            caches_for_warmup=_cw_pad_intra,
            prompt_len=len(warm_ids),
        )
        del _cw_pad_intra
        print(f"  compiled verify+intra fns.", flush=True)

    # ── Compile per-layer kernels (AFTER verify fns are traced) ────────────
    # Attaching _compiled_decode here ensures the outer mx.compile in
    # build_compiled_verify_fns traced the model WITHOUT nested per-layer
    # compiled functions, eliminating the aliasing / NaN bug for K=1.
    attach_decode_compilation(
        model, max_cache_len=max_cache_len,
        kv_dtype=target_dtype, compile_decode=True,
    )

    for _ in range(max(0, args.warmup - 1)):
        _lw2, _cw2 = run_prefill(x_w, router_temp)
        mx.eval(_lw2, _cw2)

    cot_markers = _build_cot_markers(tokenizer)
    if cot_markers:
        print(f"CoT markers detected: { {v: k for k, v in cot_markers.items()} }", flush=True)
    else:
        print("CoT markers: none found (CoT tags will be hidden in output)", flush=True)

    eos_mode   = "on" if args.stop_on_eos else "off"
    intra_mode = "off" if args.no_intra else "on"
    dk_mode    = "off" if args.no_dynamic_K else "on"
    print(f"Ready.  K={K}  ngram-n={args.ngram_n}  quant={args.quantize}-bit  "
          f"temp={args.temp}  eos-stop={eos_mode}  "
          f"dynamic-K={dk_mode}  intra={intra_mode}\n", flush=True)

    # ── Run ────────────────────────────────────────────────────────────────
    def _run_one(prompt_text: str) -> None:
        text, tps, decode_s, arl, n_tok, _stats = run_jacobi_stream(
            model=model,
            run_prefill=run_prefill,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            router_temp=router_temp,
            max_cache_len=max_cache_len,
            sample_args=sample_args,
            compiled_verify_fns=compiled_verify_fns,
            compiled_verify_with_intra_fns=compiled_verify_with_intra_fns,
            K=K,
            K_values=K_values,
            use_dynamic_K=use_dynamic_K,
            use_ngram=not args.no_ngram,
            sys_prompt=args.sys_prompt,
            max_new_tokens=args.max_new_tokens,
            ngram_n=args.ngram_n,
            min_ngram_n=args.min_ngram_n,
            seq_len=args.seq_len,
            stop_ids=stop_ids,
            cot_markers=cot_markers,
            ar_warmup_tokens=args.ar_warmup,
            show_progress=not args.no_progress,
            debug=args.debug,
        )
        _print_stats(tps, decode_s, n_tok, arl, args.baseline_tps, K, stats=_stats)

    if args.interactive:
        print("Interactive mode (Ctrl-D or 'quit' to exit)\n")
        while True:
            try:
                user = input("User> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user.lower() in ("quit", "exit", "q"):
                break
            if not user:
                continue
            _run_one(user)
    else:
        _run_one(args.prompt)


if __name__ == "__main__":
    main()
