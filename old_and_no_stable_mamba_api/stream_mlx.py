#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream new tokens from the hybrid MLX model: one compiled decode step + Python loop + yield.

MLX ``mx.compile`` is process-local JIT; this script compiles a **single-token** forward
(``(1,1)`` + cache update) once, then ``yield``s each token id after ``mx.eval`` so you can
print or push to an API without waiting for the full sequence. Core model code is unchanged.

Usage (repo root):
  python inference/stream_mlx.py --prompt "Hello"

Unless ``--raw-prompt`` is set, ``--prompt`` is plain user text wrapped to ChatML (suffix
``<|im_start|>assistant`` + newline) before tokenization — aligned with SFT inference.

Defaults target maximum throughput on Apple Silicon (same spirit as inference/experimental/bench_pure_metal.sh).
For quality-first runs use ``inference/run_stable_stream.sh`` (see ``INFERENCE_STACK.md``).

Throughput-oriented defaults:
bf16 + ``--kv-dtype auto``, 4-bit weight quant, Tucker einsum fuse, full single-step decode compile,
greedy sampling, no penalties, skip cache materialization by default.
By default **up to 2048 new tokens** are streamed and **EOS does not end generation**; use
``--stop-on-eos`` to stop when a stop token id is sampled: tokenizer EOS (e.g. ``</s>``) plus, for ChatML-aligned
runs, ``<|im_end|>`` — or ``--max-new-tokens N`` to cap length.
Use ``--materialize-caches`` / ``--no-fast-sample`` / ``--enable-penalties`` when you need safer or richer sampling.
Use ``-i`` / ``--interactive`` for a stdin REPL (model loaded once); **interactive mode enables EOS stop on every turn**
unless you pass ``--no-eos-stop``. Use ``--plain-output`` for simple terminal rendering (see ``run_stable_stream.sh --interactive``).

Programmatic:
  from stream_mlx import generate_token_stream, make_compiled_decode_step
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import sys
import time
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as nn

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    HAS_RICH = True
except Exception:
    HAS_RICH = False

# Allow `python inference/stream_mlx.py` without installing the package.
_INF_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_INF_DIR, "lib")
_REPO_ROOT = os.path.abspath(os.path.join(_INF_DIR, ".."))
for _p in (_LIB_DIR, _INF_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    export_npz_cache,
    maybe_export_npz_sidecar_after_pt_load,
    resolve_mlx_checkpoint,
    strict_load_and_convert,
)


def _strict_load_with_progress(model: Any, resolved_path: str) -> None:
    """
    Wrap strict checkpoint loading with a compact progress UI.

    Some lower-level loaders print multiple raw lines; capture them for cleaner UX and only surface
    details when failures occur.
    """
    stages = [
        ("resolve", "Resolve checkpoint path"),
        ("load", "Load and convert tensors"),
        ("verify", "Verify parameter mapping"),
        ("finalize", "Finalize model weights"),
    ]
    total = len(stages)

    def _run_load() -> str:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                strict_load_and_convert(model, resolved_path)
        except Exception:
            detail = buf.getvalue().strip()
            if detail:
                print(detail, file=sys.stderr)
            raise
        return buf.getvalue().strip()

    if HAS_RICH:
        try:
            from rich.console import Console
            from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

            con = Console(highlight=False, force_terminal=True)
            con.print(f"[bold cyan]Checkpoint[/bold cyan]: [dim]{resolved_path}[/dim]")
            with Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=28),
                TextColumn("{task.completed}/{task.total}"),
                console=con,
                transient=True,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as prog:
                task = prog.add_task("Loading weights", total=total)
                for _, label in stages[:-1]:
                    prog.update(task, description=label)
                    time.sleep(0.01)
                    prog.advance(task, 1)
                prog.update(task, description=stages[-1][1])
                captured = _run_load()
                prog.advance(task, 1)
            con.print("[green]✓ Weights loaded and verified[/green]")
            if captured:
                # Keep one compact hint instead of dumping internal verbose logs.
                con.print("[dim]loader details captured (hidden for clean output)[/dim]")
            return
        except Exception:
            pass

    # Plain fallback without Rich.
    print(f"[load] checkpoint: {resolved_path}")
    print("[load] resolve → load/convert → verify ...", flush=True)
    captured = _run_load()
    print("[load] done: weights loaded and verified", flush=True)
    if captured:
        print("[load] loader details captured (hidden)", flush=True)


def _print_run_settings_banner(
    *,
    args: Any,
    repo_root: str,
    tokenizer_path: str,
    vocab_size: int,
    checkpoint_requested: str,
    resolved_checkpoint: str | None,
    resolved_kind: str,
    config: Any,
    max_cache_len: int,
    kv_dtype_label: str,
    prefill_compile_mode: str,
    dec_mode: str,
    use_outer_compile: bool,
    stop_tokens_line: str | None = None,
) -> None:
    """Pretty-print effective run configuration (Rich or UTF-8 ASCII fallback)."""
    if getattr(args, "no_run_banner", False):
        return

    if args.fast_sample:
        sampling = f"greedy (fast-sample), router_temp={args.router_temp}"
    else:
        sampling = (
            f"temp={args.temp}  top_k={args.top_k}  top_p={args.top_p}  "
            f"min_p={args.min_p}  router_temp={args.router_temp}"
        )

    ck_req = checkpoint_requested.strip() or "(unset → auto-resolve)"
    if resolved_checkpoint:
        rk = resolved_kind
        ck_res = os.path.abspath(resolved_checkpoint)
        ck_line = os.path.basename(ck_res) + f"  [dim italic]({rk})[/dim italic]"
        ck_full = ck_res
    else:
        ck_line = "[yellow]random init (smoke)[/yellow]"
        ck_full = ""

    prompt_mode = (
        "[green]plain user[/green] → ChatML + assistant header"
        if not args.raw_prompt
        else "[yellow]raw string[/yellow] encode"
    )

    extras = []
    if args.interactive:
        extras.append("[bold]interactive REPL[/bold]")
    if getattr(args, "metal_best_preset", False):
        extras.append("[green]metal-best preset[/green]")
    if args.lookahead_router:
        extras.append("lookahead-router")
    if args.materialize_caches:
        extras.append("[green]materialize caches[/green]")
    if getattr(args, "fused_sample_metal", False):
        extras.append("fused-sample-metal")
    if getattr(args, "fused_sample_metal_v2", False):
        extras.append("fused-sample-metal-v2")
    mode_extra = " · ".join(extras) if extras else ""
    eos_eff = bool(getattr(args, "stop_on_eos", False)) and not bool(getattr(args, "no_eos_stop", False))
    rk = resolved_kind if resolved_checkpoint else ""
    stop_decode_val = stop_tokens_line if eos_eff and stop_tokens_line else ("off" if not eos_eff else "on")

    ascii_rows: list[tuple[str, str]] = [
        ("Repo root", os.path.abspath(repo_root)),
        ("Tokenizer", tokenizer_path),
        ("Vocab size", str(vocab_size)),
        ("Checkpoint argv", ck_req),
        (
            "Resolved weights",
            f"{os.path.basename(ck_full)} ({rk})" if resolved_checkpoint else "random init (smoke)",
        ),
        ("Full path", ck_full if resolved_checkpoint else "—"),
        ("Weights dtype", args.dtype),
        ("KV dtype", kv_dtype_label),
        ("Quantize", f"{args.quantize}-bit" if args.quantize else "off"),
        ("Inference preset", str(getattr(args, "inference_type", ""))),
        ("Seq len cap", str(args.seq_len)),
        ("Max new tokens", str(args.max_new_tokens)),
        ("Warmup passes", str(args.warmup)),
        ("Prefill compile", prefill_compile_mode),
        ("Decode graph", f"{dec_mode}, outer_mx.compile={use_outer_compile}"),
        ("Max KV cache slots", str(max_cache_len)),
        ("Materialize caches", str(not args.no_materialize_caches).lower()),
        ("Tucker einsum fuse", str(bool(getattr(config, "tucker_einsum_fuse", False))).lower()),
        ("Tucker AMX fuse", str(bool(getattr(config, "tucker_amx_fuse", False))).lower()),
        ("Prompt mode", ("plain user → ChatML" if not args.raw_prompt else "raw encode")),
        ("Sampling", sampling),
        ("Stop decoding", stop_decode_val),
    ]

    wants_rich = HAS_RICH and not args.plain_output
    if wants_rich:
        try:
            from rich import box as rich_box_mod
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            console = Console(highlight=False, force_terminal=True)
            tbl = Table(
                show_header=True,
                header_style="bold cyan",
                border_style="dim",
                box=rich_box_mod.ROUNDED,
                expand=True,
                width=min(max(console.size.width - 4, 40), 100),
            )
            tbl.add_column("Setting", style="bold white", ratio=1)
            tbl.add_column("Value", style="cyan", ratio=2)
            tbl.add_row("Repo root", os.path.abspath(repo_root))
            tbl.add_row("Tokenizer", tokenizer_path)
            tbl.add_row("Vocab size", str(vocab_size))
            tbl.add_row("Checkpoint argv", ck_req)
            if resolved_checkpoint:
                tbl.add_row("Resolved weights", ck_line)
                tbl.add_row("Full path", f"[dim]{ck_full}[/dim]")
            else:
                tbl.add_row("Resolved weights", ck_line)

            tbl.add_row("Weights dtype", args.dtype)
            tbl.add_row("KV dtype", kv_dtype_label)
            tbl.add_row("Quantize bits", str(args.quantize) if args.quantize else "0 (off)")
            tbl.add_row("Inference preset", str(getattr(args, "inference_type", "")))
            tbl.add_row("Seq len cap", str(args.seq_len))
            tbl.add_row("Max new tokens", str(args.max_new_tokens))
            tbl.add_row("Warmup passes", str(args.warmup))
            tbl.add_row("Prefill compile", prefill_compile_mode)
            tbl.add_row("Decode strategy", f"{dec_mode}, outer_mx.compile={use_outer_compile}")
            tbl.add_row("Max KV cache slots", str(max_cache_len))
            tbl.add_row("Materialize caches", str(not args.no_materialize_caches).lower())
            tbl.add_row("Tucker einsum fuse", str(bool(config.tucker_einsum_fuse)).lower())
            tbl.add_row("Tucker AMX fuse", str(bool(getattr(config, "tucker_amx_fuse", False))).lower())
            tbl.add_row("Prompt mode", prompt_mode)
            tbl.add_row("Sampling", sampling)
            eos_eff = bool(getattr(args, "stop_on_eos", False)) and not bool(getattr(args, "no_eos_stop", False))
            tbl.add_row(
                "Stop decoding",
                stop_tokens_line if eos_eff and stop_tokens_line else ("off" if not eos_eff else "on"),
            )
            subtitle = "[dim]stable stream · MLX hybrid[/dim]"
            if mode_extra:
                subtitle = mode_extra + " — " + subtitle
            panel = Panel(
                tbl,
                title="[bold magenta]Mamba3-XR · MLX stream[/]",
                subtitle=subtitle,
                border_style="bright_blue",
                box=rich_box_mod.HEAVY,
            )
            console.print(panel)
            return
        except Exception:
            pass

    tw = max(56, shutil.get_terminal_size((96, 20)).columns)
    inner_w = min(tw - 4, 88)
    bar = "─" * inner_w
    sub_plain = ""
    if mode_extra:
        sub_plain = (
            mode_extra.replace("[bold]", "")
            .replace("[/bold]", "")
            .replace("[green]", "")
            .replace("[/green]", "")
            .strip()
        )

    def _box_row(content: str) -> None:
        c = content
        max_c = inner_w - 2
        if len(c) > max_c:
            c = c[: max_c - 1] + "…"
        print("│ " + c.ljust(inner_w - 2)[: inner_w - 2] + " │")

    print(f"╭{bar}╮")
    _box_row("Mamba3-XR · MLX stream  —  run configuration")
    if sub_plain:
        _box_row(sub_plain[:120])
    print(f"├{bar}┤")
    for label, val in ascii_rows:
        if label == "Full path" and val == "—":
            continue
        _box_row(f"{label:21} {val}")
    print(f"╰{bar}╯\n")


def _print_step_metrics(
    *,
    args: Any,
    prefill_tokens: int,
    prefill_ms: float,
    prefill_compile_mode: str,
    dec_mode: str,
    use_outer_compile: bool,
) -> None:
    if HAS_RICH and not args.plain_output:
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box as rich_box_mod

            con = Console(highlight=False, force_terminal=True)
            t = Table(show_header=False, box=rich_box_mod.SIMPLE_HEAVY, pad_edge=False, expand=False)
            t.add_column(style="cyan", justify="left")
            t.add_column(style="white", justify="left")
            t.add_row("[bold]Prefill[/bold]", f"{prefill_tokens} tok in {prefill_ms:.2f} ms  [{prefill_compile_mode}]")
            t.add_row(
                "[bold]Decode[/bold]",
                f"graph={dec_mode}  outer_compile={use_outer_compile}  "
                f"materialize_kv={not args.no_materialize_caches}",
            )
            con.print(t)
        except Exception:
            print(
                f"Prefill: {prefill_tokens} tok · {prefill_ms:.2f} ms ({prefill_compile_mode}) │ "
                f"decode={dec_mode} outer={use_outer_compile}"
            )
    else:
        print(
            f"Prefill · {prefill_tokens} tokens in {prefill_ms:.2f} ms [{prefill_compile_mode}]"
        )
        print(
            f"Decode · graph={dec_mode} outer_mx_compile={use_outer_compile} "
            f"materialize={not args.no_materialize_caches}"
        )


def _special_chunk(tokenizer: Any, tid: int, special_ids: set[int]) -> str | None:
    if tid not in special_ids:
        return None
    try:
        token = str(tokenizer.convert_ids_to_tokens(tid)).strip()
    except Exception:
        token = str(tid)
    normalized = token.lower().replace(" ", "")
    if normalized in {"<s>", "</s>", "<eos>", "<|eot_id|>", "<|endoftext|>", "<|im_end|>"}:
        return "\n"
    return ""


def _build_stream_stop_token_ids(tokenizer: Any, *, enabled: bool) -> frozenset[int] | None:
    """
    IDs that end streamed assistant output. ChatML/SFT wrappers use <|im_end|>; plain tokenizer EOS may be </s>.

    Models often finish assistant turns with ``<|im_end|>`` rather than EOS, so both are included when ``enabled``.
    """

    if not enabled:
        return None
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
    return frozenset(ids) if ids else None


def _format_stop_tokens_banner_value(tokenizer: Any, ids: frozenset[int] | None) -> str:
    if not ids:
        return "(no ids resolved)"
    parts: list[str] = []
    for tid in sorted(ids):
        try:
            piece = tokenizer.convert_ids_to_tokens(tid)
        except Exception:
            piece = ""
        if isinstance(piece, list):
            tok = "".join(str(x) for x in piece)
        else:
            tok = str(piece).strip() or "(token)"
        parts.append(f"{tid} `{tok}`")
    return ", ".join(parts)


def make_compiled_decode_step(
    model: Mamba3LanguageModel,
    router_temp: mx.array,
):
    """Compile a single decode forward: ``(1,1)`` + caches + ``seq_pos`` → ``logits, caches``."""

    def _one_step(x_one: mx.array, caches: Any, seq_pos: mx.array):
        return model(x_one, caches=caches, seq_pos=seq_pos, router_temp=router_temp)

    return mx.compile(_one_step)


def generate_token_stream(
    *,
    model: Mamba3LanguageModel,
    run_prefill: Any,
    x_prefill: mx.array,
    prompt_ids: list[int],
    max_cache_len: int,
    router_temp: mx.array,
    max_new_tokens: int,
    sample_args: Any,
    stop_token_ids: frozenset[int] | None = None,
    use_compiled_decode_step: bool = True,
    prefill_outputs: tuple[mx.array, Any] | None = None,
) -> Iterator[mx.array]:
    """
    Official-style generator:
    1) process prompt once (prefill + cache),
    2) autoregressively generate one token at a time and yield ``mx.array`` token.

    Mirrors benchmark decode semantics (penalties + sampling), while exposing a true generator API.

    Sampled token ids in ``stop_token_ids`` end the stream immediately after that yield (typically EOS ``</s>`` and/or
    ``<|im_end|>`` for ChatML).
    """
    if max_new_tokens <= 0:
        return

    if prefill_outputs is None:
        logits, caches = run_prefill(x_prefill, router_temp)
        mx.eval(logits, caches)
    else:
        logits, caches = prefill_outputs
    if not sample_args.no_materialize_caches:
        caches = _materialize_cache_tree(caches)
        mx.eval(caches)
    caches = _pad_transformer_caches(caches, max_cache_len)
    mx.eval(caches)

    decode_fn: Any
    if use_compiled_decode_step:
        decode_fn = make_compiled_decode_step(model, router_temp)
    else:

        def decode_fn(x_one, c, sp):  # type: ignore[misc,no-redef]
            return model(x_one, caches=c, seq_pos=sp, router_temp=router_temp)

    pos = len(prompt_ids)
    generated_ids: list[int] = []

    row = logits[0, -1, :]
    token_counts = _init_token_counts(prompt_ids, int(row.shape[0]))
    last = sample_decode_token(row, token_counts, sample_args)
    if not getattr(sample_args, "no_penalties", False):
        token_counts[last] = token_counts[last] + 1
        mx.eval(last)
    else:
        mx.eval(last)
    yield last
    tid = int(last.item())
    generated_ids.append(tid)
    if stop_token_ids is not None and tid in stop_token_ids:
        return

    x_one = last.reshape(1, 1)

    for _ in range(max_new_tokens - 1):
        seq_pos = mx.array(pos, dtype=mx.int32)
        logits_d, caches = decode_fn(x_one, caches, seq_pos)
        row = logits_d[0, -1, :]
        last = sample_decode_token(row, token_counts, sample_args)
        if not getattr(sample_args, "no_penalties", False):
            token_counts[last] = token_counts[last] + 1
            mx.eval(last, caches, token_counts)
        else:
            mx.eval(last, caches)
        yield last
        tid = int(last.item())
        generated_ids.append(tid)
        if stop_token_ids is not None and tid in stop_token_ids:
            break
        x_one = last.reshape(1, 1)
        pos += 1


def _spec_early_exit_draft(
    model: Any,
    seed_x: mx.array,
    base_caches: list,
    seq_pos: int,
    n_draft_layers: int,
    k_draft: int,
    router_temp: mx.array,
) -> tuple[list[int], list]:
    """
    Draft k_draft tokens using only the first n_draft_layers of the backbone.

    The draft runs through layers [0, n_draft_layers) starting from the same
    SSM/KV state as the full model (base_caches[:n_draft_layers]).  The tied
    LM head is applied after the final draft layer so no extra weights are needed.

    Returns (draft_token_ids, updated_draft_caches_for_first_n_layers).
    """
    import math

    layers = model.backbone.layers
    scale = math.sqrt(model.config.d_model)

    cur_caches = list(base_caches[:n_draft_layers])
    x_cur = seed_x
    draft_tokens: list[int] = []

    for step in range(k_draft):
        h = model.embed(x_cur)
        pos_arr = mx.array(seq_pos + step, dtype=mx.int32)
        new_layer_caches: list = []

        for i in range(n_draft_layers):
            layer = layers[i]
            cache = cur_caches[i] if i < len(cur_caches) else None
            lt = getattr(layer, "l_type", None)

            if lt == "transformer":
                if cache is not None and getattr(layer, "_compiled_decode", None) is not None:
                    k_c, v_c = cache
                    h, nk, nv = layer._compiled_decode(h, k_c, v_c, pos_arr)
                    new_layer_caches.append((nk, nv))
                else:
                    h, nc = layer(h, cache=cache, seq_pos=pos_arr, router_temp=router_temp)
                    new_layer_caches.append(nc)
            else:
                if cache is not None and getattr(layer, "_compiled_decode", None) is not None:
                    h_s, prev_in, prev_ang = cache
                    h, nh, npi, nas = layer._compiled_decode(h, h_s, prev_in, prev_ang)
                    new_layer_caches.append((nh, npi, nas))
                else:
                    h, nc = layer(h, cache=cache, router_temp=router_temp)
                    new_layer_caches.append(nc)

        mx.eval(h, *new_layer_caches)
        cur_caches = new_layer_caches

        h_normed = mx.fast.rms_norm(h, model.norm.weight, float(getattr(model.norm, "eps", 1e-5)))
        logits_draft = model.head(h_normed / scale)
        tok_arr = mx.argmax(logits_draft[0, 0, :], axis=-1)
        mx.eval(tok_arr)
        tok = int(tok_arr.item())
        draft_tokens.append(tok)
        x_cur = mx.array([[tok]], dtype=mx.int32)

    return draft_tokens, cur_caches


def generate_spec_token_stream(
    *,
    model: Any,
    run_prefill: Any,
    x_prefill: mx.array,
    prompt_ids: list[int],
    max_cache_len: int,
    router_temp: mx.array,
    max_new_tokens: int,
    sample_args: Any,
    stop_token_ids: frozenset[int] | None = None,
    n_draft_layers: int = 8,
    k_draft: int = 4,
    prefill_outputs: tuple[mx.array, Any] | None = None,
    jacobi: bool = False,
) -> Iterator[mx.array]:
    """
    Speculative decode stream — two modes selectable via ``jacobi``:

    Early-exit mode (jacobi=False, default):
      Draft phase   : first n_draft_layers propose k_draft tokens (early exit).
      Verify phase  : full model runs ONE compiled parallel forward.
      Accept/reject : greedy match; corrections use compiled single-step decode.

    Jacobi mode (jacobi=True):
      No separate draft model — uses the full model's own previous-round predictions
      as the draft for the next round (carry-over n-gram reuse).
      Only the compiled K-token verify and compiled single-step correction are needed.
      Draft cost = 0; accept rate depends on text predictability.
    """
    if max_new_tokens <= 0:
        return

    if prefill_outputs is None:
        logits, caches = run_prefill(x_prefill, router_temp)
        mx.eval(logits, caches)
    else:
        logits, caches = prefill_outputs

    if not sample_args.no_materialize_caches:
        caches = _materialize_cache_tree(caches)
        mx.eval(caches)
    caches = _pad_transformer_caches(caches, max_cache_len)
    mx.eval(caches)

    # Pre-compile verify (fixed shape (1, k_draft)) and single-step correction.
    K = k_draft

    def _verify_step(x_k: mx.array, c: Any, pos: mx.array) -> tuple:
        return model(x_k, caches=c, seq_pos=pos, router_temp=router_temp)

    compiled_verify = mx.compile(_verify_step)
    compiled_single = make_compiled_decode_step(model, router_temp)

    pos = len(prompt_ids)
    generated_ids: list[int] = []

    # First token from prefill logits (greedy)
    row = logits[0, -1, :]
    tok0_arr = mx.argmax(row, axis=-1)
    mx.eval(tok0_arr)
    yield tok0_arr
    tok0 = int(tok0_arr.item())
    generated_ids.append(tok0)
    if stop_token_ids and tok0 in stop_token_ids:
        return

    # Advance full model one step to seed the decode loop properly
    x_last = mx.array([[tok0]], dtype=mx.int32)
    logits_1, caches = compiled_single(x_last, caches, mx.array(pos, dtype=mx.int32))
    mx.eval(logits_1, caches)
    current_logit_row = logits_1[0, -1, :]
    pos += 1

    # Warmup verify (first call compiles the K-token verify graph)
    _dummy = mx.array([[tok0] * K], dtype=mx.int32)
    _lv, _cv = compiled_verify(_dummy, caches, mx.array(pos, dtype=mx.int32))
    mx.eval(_lv, _cv)
    del _dummy, _lv, _cv

    spec_rounds = spec_accepted = spec_proposed = 0

    # Jacobi: carry-over buffer holds full-model predictions from the previous round.
    # Initialized to repeat-last; updated after every verify pass.
    carry_draft: list[int] = [tok0] * K  # only used in Jacobi mode

    while len(generated_ids) < max_new_tokens:
        remaining = max_new_tokens - len(generated_ids)
        K_this = min(K, remaining)

        # ── 1. Draft ────────────────────────────────────────────────────────
        if jacobi:
            # Jacobi: reuse predictions from previous verify round (zero draft cost).
            draft_tokens = carry_draft[:K_this]
            draft_padded = carry_draft  # already length K
        else:
            # Early-exit: run first n_draft_layers and apply head.
            draft_tokens, _ = _spec_early_exit_draft(
                model, x_last, caches, pos, n_draft_layers, K_this, router_temp
            )
            pad = draft_tokens[-1] if draft_tokens else tok0
            draft_padded = draft_tokens + [pad] * (K - K_this)

        spec_rounds += 1
        spec_proposed += K_this

        # ── 2. Verify: ONE compiled parallel forward over K tokens ───────────
        x_verify = mx.array([draft_padded], dtype=mx.int32)
        logits_v, verify_caches = compiled_verify(
            x_verify, caches, mx.array(pos, dtype=mx.int32)
        )
        mx.eval(logits_v, verify_caches)

        # Extract full-K carry predictions for next Jacobi round.
        if jacobi:
            carry_draft = [int(mx.argmax(logits_v[0, i, :], axis=-1).item()) for i in range(K)]

        # ── 3. Greedy accept/reject ─────────────────────────────────────────
        accepted = 0
        correction: int | None = None
        for i in range(K_this):
            row_i = current_logit_row if i == 0 else logits_v[0, i - 1, :]
            pred = int(mx.argmax(row_i, axis=-1).item())
            if pred == draft_tokens[i]:
                accepted += 1
            else:
                correction = pred
                break

        if accepted == K_this:
            # All K_this tokens accepted; commit verify state directly.
            consume = draft_tokens
            caches = verify_caches
            current_logit_row = logits_v[0, K_this - 1, :]
            x_last = mx.array([[draft_tokens[-1]]], dtype=mx.int32)
            pos += K_this
            if jacobi:
                # Carry is already set from logits_v; shift it to reflect consumed K_this tokens.
                carry_draft = (carry_draft[K_this:] + carry_draft[-1:] * K_this)[:K]
        else:
            # Correction: replay accepted tokens + correction via compiled single-step.
            assert correction is not None
            consume = draft_tokens[:accepted] + [correction]
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
            x_last = mx.array([[correction]], dtype=mx.int32)
            pos += len(consume)
            if jacobi:
                # Shift carry: skip accepted+1 positions, reuse the tail from this verify.
                skip = accepted + 1
                carry_draft = (carry_draft[skip:] + carry_draft[-1:] * skip)[:K]

        spec_accepted += accepted

        for t in consume:
            yield mx.array(t, dtype=mx.int32)
            generated_ids.append(t)
            if stop_token_ids and t in stop_token_ids:
                _spec_log(spec_rounds, spec_accepted, spec_proposed, jacobi=jacobi)
                return
            if len(generated_ids) >= max_new_tokens:
                _spec_log(spec_rounds, spec_accepted, spec_proposed, jacobi=jacobi)
                return

    _spec_log(spec_rounds, spec_accepted, spec_proposed, jacobi=jacobi)


def _spec_log(rounds: int, accepted: int, proposed: int, *, jacobi: bool = False) -> None:
    rate = 100 * accepted / max(proposed, 1)
    mode = "jacobi" if jacobi else "spec"
    print(
        f"\n[{mode}] rounds={rounds} accepted={accepted}/{proposed} ({rate:.1f}%)",
        flush=True,
    )


def _stream_tokens_to_stdio(
    *,
    token_iter: Iterator[mx.array],
    tokenizer: Any,
    plain_output: bool,
    max_new_tokens: int,
) -> tuple[int, float, float | None]:
    """Consume ``generate_token_stream`` and print decoding; Rich progress optional."""
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    generated_for_display: list[int] = []
    prev_decoded_text = ""

    use_rich = HAS_RICH and not plain_output
    t_dec0 = time.perf_counter()
    n_out = 0
    ttft_s: float | None = None

    if use_rich:
        console = Console(highlight=False, force_terminal=True)
        progress = Progress(
            TextColumn("[bold cyan]Generating[/bold cyan]"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn(" [yellow]{task.completed}[/yellow]/[yellow]{task.total}[/yellow] tok"),
            TextColumn(" [green]{task.fields[tok_s]:5.1f} tok/s[/green]"),
            console=console,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
            refresh_per_second=12,
        )
        task_id = progress.add_task("decode", total=max_new_tokens, tok_s=0.0)
        out_buf = ""
        wrap_width = max(48, shutil.get_terminal_size((100, 20)).columns - 4)

        def _flush_ready_text(force: bool = False) -> None:
            nonlocal out_buf
            while True:
                nl = out_buf.find("\n")
                if nl >= 0:
                    line = out_buf[:nl]
                    progress.console.print(line, highlight=False, soft_wrap=True)
                    out_buf = out_buf[nl + 1 :]
                    continue
                if len(out_buf) >= wrap_width:
                    cut = out_buf.rfind(" ", 0, wrap_width)
                    if cut <= 0:
                        cut = wrap_width
                    line = out_buf[:cut].rstrip()
                    progress.console.print(line, highlight=False, soft_wrap=True)
                    out_buf = out_buf[cut:].lstrip()
                    continue
                break
            if force and out_buf:
                progress.console.print(out_buf, highlight=False, soft_wrap=True)
                out_buf = ""

        with progress:
            progress.console.print("[bold green]Assistant:[/bold green]", highlight=False)
            for tok in token_iter:
                mx.eval(tok)
                tid = int(tok.item())
                n_out += 1
                if ttft_s is None:
                    ttft_s = time.perf_counter() - t_dec0
                chunk = _special_chunk(tokenizer, tid, special_ids)
                if chunk is None:
                    generated_for_display.append(tid)
                    try:
                        full_text = tokenizer.decode(
                            generated_for_display,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        if full_text.startswith(prev_decoded_text):
                            chunk = full_text[len(prev_decoded_text) :]
                        else:
                            chunk = tokenizer.decode(
                                [tid],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                        prev_decoded_text = full_text
                    except Exception:
                        chunk = f"<{tid}>"
                if chunk:
                    out_buf += chunk
                    _flush_ready_text(force=False)
                elapsed = time.perf_counter() - t_dec0
                tok_s = n_out / max(elapsed, 1e-9)
                progress.update(task_id, completed=n_out, tok_s=tok_s)
            _flush_ready_text(force=True)
        print()
    else:
        term_width = max(40, shutil.get_terminal_size((100, 20)).columns - 8)
        line_len = 0
        print("Assistant:", end=" ", flush=True)
        for tok in token_iter:
            mx.eval(tok)
            tid = int(tok.item())
            n_out += 1
            if ttft_s is None:
                ttft_s = time.perf_counter() - t_dec0
            chunk = _special_chunk(tokenizer, tid, special_ids)
            if chunk is None:
                generated_for_display.append(tid)
                try:
                    full_text = tokenizer.decode(
                        generated_for_display,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if full_text.startswith(prev_decoded_text):
                        chunk = full_text[len(prev_decoded_text) :]
                    else:
                        chunk = tokenizer.decode(
                            [tid],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    prev_decoded_text = full_text
                except Exception:
                    chunk = f"<{tid}>"
            for ch in chunk:
                if ch == "\n":
                    print()
                    line_len = 0
                    continue
                if line_len >= term_width and ch == " ":
                    print()
                    line_len = 0
                    continue
                print(ch, end="", flush=True)
                line_len += 1
    decode_s = time.perf_counter() - t_dec0
    return n_out, decode_s, ttft_s


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stream MLX hybrid generation (compiled single-step decode + yield per token)"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Weights: .pt/.pth/.npz, or empty for repo model.npz / checkpoint.pt",
    )
    p.add_argument("--npz-cache", type=str, default="")
    p.add_argument("--force-pt", action="store_true")
    p.add_argument(
        "--save-npz",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
    )
    p.add_argument("--tokenizer", type=str, default=os.path.join(_INF_DIR, "tokenizer"))
    p.add_argument(
        "--inference-type",
        type=str,
        default="throughput",
        choices=("throughput", "safe", "eager", "sequential-ssm", "custom"),
    )
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to stream after the prompt (default 2048).",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=32007)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument(
        "--kv-dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="KV cache dtype; 'auto' matches --dtype (recommended with bf16 weights).",
    )
    p.add_argument("--quantize", type=int, choices=[0, 4, 8], default=4)
    p.add_argument("--router-temp", type=float, default=0.5)
    p.add_argument("--no-compile-prefill", action="store_true")
    p.add_argument("--eager-decode", action="store_true")
    p.add_argument(
        "--full-decode-compile",
        dest="full_decode_compile",
        action="store_true",
        default=True,
        help="Single mx.compile for whole decode step; disables per-layer decode compile (default on for speed).",
    )
    p.add_argument(
        "--no-full-decode-compile",
        dest="full_decode_compile",
        action="store_false",
        help="Use per-layer compiled decode instead of one outer mx.compile step.",
    )
    p.add_argument(
        "--materialize-caches",
        action="store_true",
        default=False,
        help=(
            "Copy caches out of compiled prefill outputs (safer streaming; slower). "
            "Default skips this copy for throughput."
        ),
    )
    p.add_argument(
        "--no-materialize-caches",
        dest="materialize_caches",
        action="store_false",
        help="Do not materialize caches (fastest).",
    )
    p.add_argument(
        "--no-outer-compile",
        action="store_true",
        help="Do not wrap the single-token forward in mx.compile (per-layer compile only when enabled)",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Plain user text; wrapped as ChatML user turn + <|im_start|>assistant + newline before generation. "
        "Use --raw-prompt to tokenize this string verbatim.",
    )
    p.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Encode --prompt as-is (no ChatML wrap; misaligned with default SFT if you paste full ChatML).",
    )
    p.add_argument("--synthetic-prompt", action="store_true")
    p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "REPL: load the model once, then repeatedly read a line from stdin as the user utterance "
            "(ChatML-wrapped unless --raw-prompt). Type quit / exit / q or EOF to leave. "
            "Uses eager prefill so prompt length can vary each turn. "
            "Each assistant reply stops at eos / <|im_end|> (when enabled) unless --no-eos-stop."
        ),
    )
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_p", type=float, default=0.05)
    p.add_argument("--rep_pen", type=float, default=1.1)
    p.add_argument("--pres_pen", type=float, default=0.0)
    p.add_argument("--freq_pen", type=float, default=0.02)
    p.add_argument(
        "--fast-sample",
        dest="fast_sample",
        action="store_true",
        default=True,
        help="Greedy argmax (default; fastest).",
    )
    p.add_argument(
        "--no-fast-sample",
        dest="fast_sample",
        action="store_false",
        help="Use temp / top-k / top-p / min-p per --temp/--top_k/...",
    )
    p.add_argument(
        "--fused-sample-metal",
        action="store_true",
        help=(
            "Penalties, temperature, min-p, top-k, top-p, softmax, and sampling run on Metal (inverse CDF uses "
            "MLX cumsum). Greedy uses Metal argmax. Requires Apple MLX Metal; falls back to MLX on error."
        ),
    )
    p.add_argument(
        "--fused-sample-metal-v2",
        action="store_true",
        help="Single-dispatch fully-fused Metal sampling (v2). Faster than v1 and pure MLX in most cases.",
    )
    p.add_argument(
        "--plain-output",
        action="store_true",
        help="Disable rich chat-style renderer and use plain terminal output.",
    )
    p.add_argument(
        "--no-run-banner",
        action="store_true",
        help="Skip the startup configuration panel / ASCII box.",
    )
    p.add_argument(
        "--metal-best-preset",
        action="store_true",
        help=(
            "Apply the current best latency/throughput knobs from metal benchmarks "
            "(throughput preset + no cache materialize + 4-bit quant + fused Tucker einsum + fast greedy)."
        ),
    )
    p.add_argument(
        "--no-penalties",
        dest="no_penalties",
        action="store_true",
        default=True,
        help="Skip repetition / presence / frequency penalties (default; faster).",
    )
    p.add_argument(
        "--enable-penalties",
        dest="no_penalties",
        action="store_false",
        help="Apply --rep_pen / --pres_pen / --freq_pen on logits (slower).",
    )
    p.add_argument(
        "--stop-on-eos",
        action="store_true",
        default=False,
        help=(
            "Stop when a stop-token id is sampled: tokenizer eos (e.g. </s>) and, when present in vocab, "
            "<|im_end|> for ChatML/SFT-aligned turns."
        ),
    )
    p.add_argument(
        "--no-eos-stop",
        action="store_true",
        default=False,
        help="Same as default behavior (ignore EOS); kept for scripts/Make compatibility.",
    )
    p.add_argument(
        "--kmoe-no-gather",
        action="store_true",
        help="A/B mode: avoid dynamic g_all[top_k] gather in TuckerMoE (computes all experts then sparse-weighted sum)",
    )
    p.add_argument(
        "--lookahead-router",
        action="store_true",
        help=(
            "Experimental MoE routing from previous sublayer hidden (see mlx_hybrid_infer). "
            "Disables single-step outer mx.compile for decode; use with per-layer compile only."
        ),
    )
    p.add_argument(
        "--tucker-einsum-fuse",
        dest="tucker_einsum_fuse",
        action="store_true",
        default=True,
        help="Fused Tucker MoE latent einsum Metal path (default on for speed).",
    )
    p.add_argument(
        "--no-tucker-einsum-fuse",
        dest="tucker_einsum_fuse",
        action="store_false",
        help="Disable fused Tucker einsum (reference path).",
    )
    p.add_argument(
        "--tucker-amx-fuse",
        dest="tucker_amx_fuse",
        action="store_true",
        default=False,
        help=(
            "Use AMX simdgroup_matrix partial-output Metal kernel for TuckerMoE decode. "
            "Fastest benchmarked path for single-token b=1 bfloat16 decode; "
            "auto-falls-back to einsum_fuse for prefill or batch>1."
        ),
    )
    p.add_argument(
        "--no-tucker-amx-fuse",
        dest="tucker_amx_fuse",
        action="store_false",
        help="Disable Tucker AMX kernel (default).",
    )
    p.add_argument(
        "--tucker-scalar-fuse",
        dest="tucker_scalar_fuse",
        action="store_true",
        default=False,
        help="Scalar optimized kernel for decode b=1.",
    )
    p.add_argument(
        "--fused-mamba-mixer",
        dest="fused_mamba_mixer",
        action="store_true",
        default=False,
        help="Fused Norm+RoPE+Einsum+SSM Metal kernel for decode b=1 (2× faster Mamba layers).",
    )
    p.add_argument(
        "--fused-mamba-mixer-v3",
        dest="fused_mamba_mixer_v3",
        action="store_true",
        default=False,
        help="V3 extension: also fuse y_down_proj + D_skip into the same dispatch (saves 2 kernels/layer).",
    )
    p.add_argument(
        "--speculative",
        dest="speculative",
        action="store_true",
        default=False,
        help=(
            "Enable early-exit self-speculative decoding. "
            "First --spec-draft-layers layers propose --spec-max-draft tokens; "
            "full model verifies in ONE parallel forward. "
            "No retraining or extra head required."
        ),
    )
    p.add_argument(
        "--spec-draft-layers",
        dest="spec_draft_layers",
        type=int,
        default=8,
        metavar="N",
        help="Number of backbone layers used for the early-exit draft (default 8 of 30).",
    )
    p.add_argument(
        "--spec-max-draft",
        dest="spec_max_draft",
        type=int,
        default=4,
        metavar="K",
        help="Draft tokens proposed per speculative round (default 4).",
    )
    p.add_argument(
        "--jacobi",
        dest="jacobi",
        action="store_true",
        default=False,
        help=(
            "Jacobi decoding: no draft model — reuse full-model predictions from the previous "
            "verify round as the next round's draft (zero draft cost, carry-over n-gram reuse). "
            "Enables --speculative automatically. "
            "Best on repetitive / structured text."
        ),
    )
    args = p.parse_args()
    if args.interactive and args.synthetic_prompt:
        p.error("--interactive cannot be combined with --synthetic-prompt")
    # Interactive: stop each turn at EOS (tokenizer id); opt out explicitly with --no-eos-stop.
    if getattr(args, "interactive", False) and not getattr(args, "no_eos_stop", False):
        args.stop_on_eos = True
    if args.lookahead_router and args.full_decode_compile:
        print(
            "Note: --lookahead-router disables --full-decode-compile (per-layer decode graphs only).",
            file=sys.stderr,
        )
        args.full_decode_compile = False
    if getattr(args, "metal_best_preset", False):
        # Benchmark-guided streaming profile from current metal runs.
        args.inference_type = "throughput"
        args.materialize_caches = False
        args.quantize = 4
        args.kv_dtype = "auto"
        args.fast_sample = True
        args.no_penalties = True
        args.tucker_einsum_fuse = True
        args.full_decode_compile = True
        args.no_outer_compile = False
        if getattr(args, "lookahead_router", False):
            print(
                "Note: --metal-best-preset disables --lookahead-router to keep outer decode compile on.",
                file=sys.stderr,
            )
            args.lookahead_router = False
    if getattr(args, "interactive", False) and bool(getattr(args, "tucker_amx_fuse", False)):
        # AMX path in REPL mode is more stable when KV caches are explicitly materialized.
        args.materialize_caches = True
    _apply_inference_type(args)
    # Benchmark ``throughput`` preset requests cache materialization; streaming defaults omit the copy unless
    # ``--materialize-caches``. (Overrides preset; matches experimental peak-throughput scripts intent.)
    args.no_materialize_caches = not bool(getattr(args, "materialize_caches", False))
    if getattr(args, "no_penalties", False):
        args.rep_pen = 1.0
        args.pres_pen = 0.0
        args.freq_pen = 0.0

    compute_dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    target_dtype = compute_dtype_map[args.dtype]
    kv_map = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
    kv_dtype = target_dtype if args.kv_dtype == "auto" else kv_map[args.kv_dtype]

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("Please `pip install transformers` to load the tokenizer.") from e

    tok_path = args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    vocab_size = len(tokenizer) if args.vocab_size <= 0 else args.vocab_size

    config = Mamba3Config(
        d_model=768,
        d_state=64,
        d_head=64,
        expand=2,
        num_layers=6,
        mimo_rank=4,
        num_kv_heads=4,
        use_parallel_scan=args.use_parallel_scan,
        chunk_size=64,
        use_kmoe=True,
        kmoe_num_experts=8,
        kmoe_top_k=2,
        kmoe_r1=32,
        kmoe_r2=512,
        kmoe_r3=256,
        ffn_expand=6,
    )
    config.lookahead_router = bool(args.lookahead_router)
    config.tucker_einsum_fuse = bool(args.tucker_einsum_fuse)
    config.tucker_amx_fuse = bool(args.tucker_amx_fuse)
    config.tucker_scalar_fuse = bool(getattr(args, "tucker_scalar_fuse", False))
    config.tucker_full_fuse = False
    config.fused_mamba_mixer = bool(getattr(args, "fused_mamba_mixer", False))
    config.fused_mamba_mixer_v3 = bool(getattr(args, "fused_mamba_mixer_v3", False))

    model = Mamba3LanguageModel(config, vocab_size)

    resolved, kind = resolve_mlx_checkpoint(
        args.checkpoint,
        repo_root=_REPO_ROOT,
        npz_cache=args.npz_cache,
        force_pt=args.force_pt,
    )
    if resolved is None or kind == "none":
        print("No checkpoint found — random weights (smoke test only).")
    else:
        _strict_load_with_progress(model, resolved)
        if kind == "pt":
            sidecar_written = maybe_export_npz_sidecar_after_pt_load(
                model, resolved, force_refresh=args.force_pt
            )
            if sidecar_written is not None:
                print(f"💾 npz cache: {sidecar_written}")
        if kind == "pt" and args.save_npz is not None:
            dest = (
                os.path.splitext(resolved)[0] + ".npz"
                if args.save_npz == "__default__"
                else args.save_npz
            )
            dest_abs = os.path.abspath(dest)
            stem_npz = os.path.abspath(os.path.splitext(resolved)[0] + ".npz")
            if dest_abs != stem_npz:
                export_npz_cache(model, dest)
                print(f"Wrote npz cache → {dest}")

    model.apply(lambda x: x.astype(target_dtype))
    if args.quantize > 0:
        print(f"Quantizing Linear/Embedding to {args.quantize}-bit (group_size=64, mode=affine)...")
        nn.quantize(model, group_size=64, bits=args.quantize)
    mx.eval(model.parameters())
    _invalidate_tucker_caches(model)
    router_temp = mx.array(args.router_temp, dtype=target_dtype)

    pre_budget = int(args.seq_len) if args.seq_len > 0 else 4096
    max_cache_len = pre_budget + args.max_new_tokens + 8

    per_layer_decode = not args.eager_decode and not args.full_decode_compile
    attach_decode_compilation(
        model,
        max_cache_len=max_cache_len,
        kv_dtype=kv_dtype,
        compile_decode=per_layer_decode,
    )

    def prefill_forward(x: mx.array, rt: mx.array):
        return model(x, caches=None, seq_pos=None, router_temp=rt)

    eager_prefill = bool(args.no_compile_prefill or args.interactive)
    if args.interactive and not args.no_compile_prefill:
        print(
            "Note: --interactive uses eager prefill (variable user length each turn).",
            file=sys.stderr,
        )

    prefill_compile_mode = "eager"
    run_prefill: Any = prefill_forward
    if not eager_prefill:
        run_prefill = mx.compile(prefill_forward)
        prefill_compile_mode = "static"

    stop_at_eos_cfg = bool(getattr(args, "stop_on_eos", False))
    if getattr(args, "no_eos_stop", False):
        stop_at_eos_cfg = False
    stream_stop_ids = _build_stream_stop_token_ids(tokenizer, enabled=stop_at_eos_cfg)
    stop_banner_line = (
        _format_stop_tokens_banner_value(tokenizer, stream_stop_ids) if stream_stop_ids else ""
    )

    base_outer_compile = bool(args.full_decode_compile or not args.no_outer_compile)
    use_outer_compile = base_outer_compile and not args.lookahead_router
    if args.lookahead_router and base_outer_compile:
        print(
            "Note: lookahead-router → per-layer compiled decode only (outer single-step mx.compile off).",
            file=sys.stderr,
        )
    if args.full_decode_compile:
        dec_mode = "full"
    elif not args.eager_decode:
        dec_mode = "per-layer"
    else:
        dec_mode = "eager"

    kv_dtype_banner = args.kv_dtype if args.kv_dtype != "auto" else f"auto→{args.dtype}"
    chk_resolved = resolved if kind != "none" else None
    _print_run_settings_banner(
        args=args,
        repo_root=_REPO_ROOT,
        tokenizer_path=tok_path,
        vocab_size=vocab_size,
        checkpoint_requested=args.checkpoint,
        resolved_checkpoint=chk_resolved,
        resolved_kind=str(kind),
        config=config,
        max_cache_len=max_cache_len,
        kv_dtype_label=kv_dtype_banner,
        prefill_compile_mode=prefill_compile_mode,
        dec_mode=dec_mode,
        use_outer_compile=use_outer_compile,
        stop_tokens_line=stop_banner_line if stop_at_eos_cfg else None,
    )

    if args.synthetic_prompt:
        filler = list(tokenizer.encode("stream " * max(1, args.seq_len // 4)))[: args.seq_len]
        if len(filler) < args.seq_len:
            filler = (filler * (args.seq_len // max(len(filler), 1) + 1))[: args.seq_len]
        warmup_ids = filler
        prompt_ids = filler
    elif args.interactive:
        warmup_ids = _build_prompt_ids(
            tokenizer, ".", args.seq_len, chatml_user=not args.raw_prompt
        )
        prompt_ids = warmup_ids
    else:
        warmup_ids = _build_prompt_ids(
            tokenizer, args.prompt, args.seq_len, chatml_user=not args.raw_prompt
        )
        prompt_ids = warmup_ids

    x_warm = mx.array([warmup_ids], dtype=mx.int32)
    for _ in range(args.warmup):
        logits, caches = run_prefill(x_warm, router_temp)
        mx.eval(logits, caches)

    def exec_one_generation(*, pid: list[int], print_stats_line: bool) -> None:
        x_prefill = mx.array([pid], dtype=mx.int32)
        t_prefill0 = time.perf_counter()
        logits, caches = run_prefill(x_prefill, router_temp)
        mx.eval(logits, caches)
        prefill_s = time.perf_counter() - t_prefill0

        if print_stats_line:
            _print_step_metrics(
                args=args,
                prefill_tokens=len(pid),
                prefill_ms=prefill_s * 1000.0,
                prefill_compile_mode=prefill_compile_mode,
                dec_mode=dec_mode,
                use_outer_compile=use_outer_compile,
            )

        use_jacobi = bool(getattr(args, "jacobi", False))
        use_spec = bool(getattr(args, "speculative", False)) or use_jacobi
        if use_spec:
            token_iter = generate_spec_token_stream(
                model=model,
                run_prefill=run_prefill,
                x_prefill=x_prefill,
                prompt_ids=pid,
                max_cache_len=max_cache_len,
                router_temp=router_temp,
                max_new_tokens=args.max_new_tokens,
                sample_args=args,
                stop_token_ids=stream_stop_ids,
                n_draft_layers=int(getattr(args, "spec_draft_layers", 8)),
                k_draft=int(getattr(args, "spec_max_draft", 4)),
                prefill_outputs=(logits, caches),
                jacobi=use_jacobi,
            )
        else:
            token_iter = generate_token_stream(
                model=model,
                run_prefill=run_prefill,
                x_prefill=x_prefill,
                prompt_ids=pid,
                max_cache_len=max_cache_len,
                router_temp=router_temp,
                max_new_tokens=args.max_new_tokens,
                sample_args=args,
                stop_token_ids=stream_stop_ids,
                use_compiled_decode_step=use_outer_compile,
                prefill_outputs=(logits, caches),
            )
        if not HAS_RICH and not args.plain_output:
            print("\n[hint] Install rich for progress UI: pip install rich", flush=True)

        n_out, decode_s, ttft_s = _stream_tokens_to_stdio(
            token_iter=token_iter,
            tokenizer=tokenizer,
            plain_output=args.plain_output,
            max_new_tokens=args.max_new_tokens,
        )
        print()
        print("─" * 60)
        if n_out > 0:
            ttft_ms = (ttft_s or 0.0) * 1000.0
            e2e_ttft_ms = prefill_s * 1000.0 + ttft_ms
            print(
                f"Streamed {n_out} tokens in {decode_s * 1000:.2f} ms  "
                f"({n_out / max(decode_s, 1e-9):.1f} tok/s)"
            )
            print(
                f"TTFT: {ttft_ms:.2f} ms (decode-only)  |  {e2e_ttft_ms:.2f} ms (incl. prefill)"
            )

    if args.interactive:
        print()
        line_w = min(72, max(52, shutil.get_terminal_size((100, 20)).columns))
        ruler = "═" * line_w
        print(ruler)
        print(f" Interactive MLX  │  Ctrl-D · quit · exit · q → end session")
        if not args.raw_prompt:
            print(" Each line → plain user text + ChatML (assistant header appended)")
        else:
            print(" --raw-prompt: encode each line verbatim (no ChatML)")
        print(ruler + "\n")
        while True:
            try:
                line = input("User: ")
            except EOFError:
                print()
                break
            utterance = line.strip()
            if utterance.lower() in {"quit", "exit", "q"}:
                break
            if not utterance:
                continue
            try:
                pid = _build_prompt_ids(
                    tokenizer, utterance, args.seq_len, chatml_user=not args.raw_prompt
                )
            except ValueError as e:
                print(f"[skip] {e}", file=sys.stderr)
                continue
            prev = utterance if len(utterance) <= 400 else utterance[:400] + " …"
            print(f"(turn) User: {prev}")
            exec_one_generation(pid=pid, print_stats_line=False)
        return

    if not args.raw_prompt and not args.synthetic_prompt:
        try:
            u = args.prompt.strip() or "(empty)"
            prev = u if len(u) <= 400 else u[:400] + " …"
            print(f"User: {prev}")
        except Exception:
            pass
    else:
        try:
            prompt_preview = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            if prompt_preview.strip():
                prev = prompt_preview if len(prompt_preview) <= 400 else prompt_preview[:400] + " …"
                print(f"Prompt: {prev}")
        except Exception:
            pass

    exec_one_generation(pid=prompt_ids, print_stats_line=True)


if __name__ == "__main__":
    main()
