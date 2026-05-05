#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream new tokens from the hybrid MLX model: one compiled decode step + Python loop + yield.

MLX ``mx.compile`` is process-local JIT; this script compiles a **single-token** forward
(``(1,1)`` + cache update) once, then ``yield``s each token id after ``mx.eval`` so you can
print or push to an API without waiting for the full sequence. Core model code is unchanged.

Usage (repo root):
  python inference/stream_mlx.py --prompt "Hello"

Defaults target maximum throughput on Apple Silicon (same spirit as inference/bench_pure_metal.sh):
bf16 + ``--kv-dtype auto``, 4-bit weight quant, Tucker einsum fuse, full single-step decode compile,
greedy sampling, no penalties, skip cache materialization by default.
By default **up to 2048 new tokens** are streamed and **EOS does not end generation**; use
``--stop-on-eos`` to stop when the tokenizer emits EOS, or ``--max-new-tokens N`` to cap length.
Use ``--materialize-caches`` / ``--no-fast-sample`` / ``--enable-penalties`` when you need safer or richer sampling.

Programmatic:
  from stream_mlx import generate_token_stream, make_compiled_decode_step
"""
from __future__ import annotations

import argparse
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
_REPO_ROOT = os.path.abspath(os.path.join(_INF_DIR, ".."))
if _INF_DIR not in sys.path:
    sys.path.insert(0, _INF_DIR)

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


def _special_chunk(tokenizer: Any, tid: int, special_ids: set[int]) -> str | None:
    if tid not in special_ids:
        return None
    try:
        token = str(tokenizer.convert_ids_to_tokens(tid)).strip()
    except Exception:
        token = str(tid)
    normalized = token.lower().replace(" ", "")
    if normalized in {"<s>", "</s>", "<eos>", "<|eot_id|>", "<|endoftext|>"}:
        return "\n"
    return ""


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
    eos_token_id: int | None = None,
    use_compiled_decode_step: bool = True,
    prefill_outputs: tuple[mx.array, Any] | None = None,
) -> Iterator[mx.array]:
    """
    Official-style generator:
    1) process prompt once (prefill + cache),
    2) autoregressively generate one token at a time and yield ``mx.array`` token.

    Mirrors benchmark decode semantics (penalties + sampling), while exposing a true generator API.
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
    token_counts[last] = token_counts[last] + 1
    mx.eval(last)
    yield last
    tid = int(last.item())
    generated_ids.append(tid)
    if eos_token_id is not None and tid == eos_token_id:
        return

    x_one = last.reshape(1, 1)

    for _ in range(max_new_tokens - 1):
        seq_pos = mx.array(pos, dtype=mx.int32)
        logits_d, caches = decode_fn(x_one, caches, seq_pos)
        row = logits_d[0, -1, :]
        last = sample_decode_token(row, token_counts, sample_args)
        token_counts[last] = token_counts[last] + 1
        # Evaluate sampled token and updated cache together to keep one larger graph.
        mx.eval(last, caches, token_counts)
        yield last
        tid = int(last.item())
        generated_ids.append(tid)
        if eos_token_id is not None and tid == eos_token_id:
            break
        x_one = last.reshape(1, 1)
        pos += 1


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
        "--no-outer-compile",
        action="store_true",
        help="Do not wrap the single-token forward in mx.compile (per-layer compile only when enabled)",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
    )
    p.add_argument("--synthetic-prompt", action="store_true")
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
        help="Stop streaming when the tokenizer EOS id is emitted (default: keep generating until --max-new-tokens).",
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
    args = p.parse_args()
    if args.lookahead_router and args.full_decode_compile:
        print(
            "Note: --lookahead-router disables --full-decode-compile (per-layer decode graphs only).",
            file=sys.stderr,
        )
        args.full_decode_compile = False
    _apply_inference_type(args)
    # Benchmark ``throughput`` preset requests cache materialization; streaming defaults omit the copy unless
    # ``--materialize-caches``. (Overrides preset; matches inference/bench_pure_metal.sh intent.)
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
        strict_load_and_convert(model, resolved)
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
        # print("Model structure after quantization:")
        # print(model)
    mx.eval(model.parameters())
    _invalidate_tucker_caches(model)
    router_temp = mx.array(args.router_temp, dtype=target_dtype)

    if args.synthetic_prompt:
        filler = list(tokenizer.encode("stream " * max(1, args.seq_len // 4)))[: args.seq_len]
        if len(filler) < args.seq_len:
            filler = (filler * (args.seq_len // max(len(filler), 1) + 1))[: args.seq_len]
        prompt_ids = filler
    else:
        prompt_ids = _build_prompt_ids(tokenizer, args.prompt, args.seq_len)

    max_cache_len = len(prompt_ids) + args.max_new_tokens + 8
    per_layer_decode = not args.eager_decode and not args.full_decode_compile
    attach_decode_compilation(
        model,
        max_cache_len=max_cache_len,
        kv_dtype=kv_dtype,
        compile_decode=per_layer_decode,
    )

    x_prefill = mx.array([prompt_ids], dtype=mx.int32)

    def prefill_forward(x: mx.array, rt: mx.array):
        return model(x, caches=None, seq_pos=None, router_temp=rt)

    prefill_compile_mode = "eager"
    if args.no_compile_prefill:
        run_prefill = prefill_forward
    else:
        run_prefill = mx.compile(prefill_forward)
        prefill_compile_mode = "static"

    for _ in range(args.warmup):
        logits, caches = run_prefill(x_prefill, router_temp)
        mx.eval(logits, caches)

    t_prefill0 = time.perf_counter()
    logits, caches = run_prefill(x_prefill, router_temp)
    mx.eval(logits, caches)
    prefill_s = time.perf_counter() - t_prefill0

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    stop_at_eos = bool(getattr(args, "stop_on_eos", False))
    if getattr(args, "no_eos_stop", False):
        stop_at_eos = False
    if not stop_at_eos:
        eos_id = None

    # Lookahead needs per-layer backbone compiles; wrapping the whole step in mx.compile breaks it.
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
    print(
        f"Prefill: {len(prompt_ids)} tokens in {prefill_s*1000:.2f} ms  compile={prefill_compile_mode}  "
        f"decode_step={dec_mode}  outer_mx_compile={use_outer_compile}  "
        f"dtype={args.dtype}  kv_dtype={args.kv_dtype}  quantize={args.quantize}  "
        f"materialize_caches={not args.no_materialize_caches}  fast_sample={args.fast_sample}  "
        f"fused_sample_metal={bool(getattr(args, 'fused_sample_metal', False))}  "
        f"fused_sample_metal_v2={bool(getattr(args, 'fused_sample_metal_v2', False))}  "
        f"penalties={'off' if args.no_penalties else 'on'}  "
        f"kmoe_no_gather={args.kmoe_no_gather}  lookahead_router={args.lookahead_router}  "
        f"tucker_einsum_fuse={args.tucker_einsum_fuse}  "
        f"max_new_tokens={args.max_new_tokens}  stop_on_eos={stop_at_eos}"
    )
    print("─" * 60)
    try:
        prompt_preview = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        if prompt_preview.strip():
            prev = prompt_preview if len(prompt_preview) <= 400 else prompt_preview[:400] + " …"
            print(f"Prompt: {prev}")
    except Exception:
        pass
    use_rich = HAS_RICH and not args.plain_output
    if not use_rich:
        print("Output: ", end="", flush=True)
        if not HAS_RICH and not args.plain_output:
            print("\n[hint] Install rich for chat UI: pip install rich")

    t_dec0 = time.perf_counter()
    n_out = 0
    generated_for_display: list[int] = []
    prev_decoded_text = ""
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    token_iter = generate_token_stream(
        model=model,
        run_prefill=run_prefill,
        x_prefill=x_prefill,
        prompt_ids=prompt_ids,
        max_cache_len=max_cache_len,
        router_temp=router_temp,
        max_new_tokens=args.max_new_tokens,
        sample_args=args,
        eos_token_id=eos_id,
        use_compiled_decode_step=use_outer_compile,
        prefill_outputs=(logits, caches),
    )

    if use_rich:
        # Force interactive terminal behavior so Progress updates in-place
        # instead of printing a new "Generating ..." line every refresh.
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
        task_id = progress.add_task("decode", total=args.max_new_tokens, tok_s=0.0)
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
            progress.console.print("Output:", highlight=False)
            for tok in token_iter:
                mx.eval(tok)
                tid = int(tok.item())
                n_out += 1
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
                            # Fallback when tokenizer normalization changes prior characters.
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
        for tok in token_iter:
            mx.eval(tok)
            tid = int(tok.item())
            n_out += 1
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
                        # Fallback when tokenizer normalization changes prior characters.
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
    print()
    print("─" * 60)
    if n_out > 0:
        print(f"Streamed {n_out} tokens in {decode_s*1000:.2f} ms  ({n_out / max(decode_s, 1e-9):.1f} tok/s)")


if __name__ == "__main__":
    main()
