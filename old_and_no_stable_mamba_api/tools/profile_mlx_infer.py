#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone MLX / Metal bottleneck profiler for mlx_hybrid_infer Mamba3LanguageModel.

Does NOT modify ``lib/mlx_hybrid_infer.py`` — only imports model + checkpoint helpers + benchmark_mlx utilities.

Usage:
  python inference/tools/profile_mlx_infer.py --seq-len 256 --profile-decode-steps 32

Long prefill uses chunk-parallel Mamba scan unless ``--inference-type sequential-ssm``.
Profiler releases lazy graphs between stages (``gc`` + ``mx.clear_cache()``) to limit peak RAM.

Extra tiers (CLI):
  ``--fine-op-table`` — per-op-key wall ms + call counts + shape examples + heuristic kernel hints
    (many ``mx.eval`` fences → totals are **not** comparable to a single fused decode step).
  ``--decode-timeline`` — CPU-side phase stamps for the first few decode iterations (launch/sync proxy).
  ``--metal-gputrace PATH`` — ``mx.metal.start_capture`` around a short prefill + decode micro-run; open the
    ``.gputrace`` in Xcode → Metal System Trace / GPU counters. Requires ``MTL_CAPTURE_ENABLED=1`` and a
    MLX build that exposes capture (see MLX Metal Debugger docs).
  ``--json-out PATH.json`` — full report (prefill/decode/summary + embedded ``mlx_profiler_trace``).
  ``--mlx-profiler-json PATH.json`` — trace file compatible with ``mlx-profiler view`` (mlx_profiler 0.2 schema).

See module docstring in file header for interpretation notes.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.utils as mx_utils

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_INF_ROOT = os.path.dirname(_TOOLS_DIR)
_LIB_DIR = os.path.join(_INF_ROOT, "lib")
_REPO_ROOT = os.path.abspath(os.path.join(_INF_ROOT, ".."))
for _p in (_LIB_DIR, _INF_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx_hybrid_infer as mhi

from mlx_hybrid_infer import (
    Mamba3Config,
    Mamba3LanguageModel,
    attach_decode_compilation,
    maybe_export_npz_sidecar_after_pt_load,
    resolve_mlx_checkpoint,
    strict_load_and_convert,
)

from benchmark_mlx import (
    _apply_inference_type,
    _build_prompt_ids,
    _invalidate_tucker_caches,
    _materialize_cache_tree,
    _pad_transformer_caches,
)

from mlx_fine_decode_profile import (
    FineDecodeTable,
    aggregate_fine_decode_profile,
    build_mlx_profiler_trace_dict,
    print_fine_table,
    profiler_dtype_string,
)
from mlx_profile_components import aggregate_decode_component_profile


@dataclass
class Timed:
    wall_ms: float
    cpu_ms: float

    @property
    def wait_proxy_ms(self) -> float:
        return max(0.0, self.wall_ms - self.cpu_ms)


def _sync() -> None:
    try:
        mx.synchronize()
    except Exception:
        pass


def _eval_any(out: Any) -> None:
    """mx.eval all mlx arrays in a nested tuple/list."""
    flat: List[mx.array] = []

    def walk(x: Any) -> None:
        if isinstance(x, mx.array):
            flat.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                walk(y)

    walk(out)
    if flat:
        mx.eval(*flat)


def _time_block(fn) -> Tuple[Any, Timed]:
    _sync()
    w0 = time.perf_counter()
    c0 = time.thread_time()
    out = fn()
    if out is not None:
        _eval_any(out)
    _sync()
    w1 = time.perf_counter()
    c1 = time.thread_time()
    return out, Timed(wall_ms=(w1 - w0) * 1000.0, cpu_ms=(c1 - c0) * 1000.0)


def _release_lazy_graph(*refs: Any) -> None:
    """Drop MLX array / cache references so lazy graphs can be freed; return pooled Metal memory when possible."""
    for r in refs:
        del r
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def _metal_capture_api() -> bool:
    m = getattr(mx, "metal", None)
    return m is not None and hasattr(m, "start_capture") and hasattr(m, "stop_capture")


def _print_metal_capture_instructions(trace_path: str) -> None:
    print("[Metal / Xcode GPU trace]")
    print("-" * 72)
    print("  1) 終端機先匯出:  export MTL_CAPTURE_ENABLED=1")
    print("  2) 再執行本 script 並帶 --metal-gputrace（PATH 請用 .gputrace 副檔名）。")
    print("  3) Xcode → Open Developer Tool → Instruments → Metal System Trace（或雙擊 .gputrace）。")
    print("  4) 看 kernel 時間線、command buffer 排隊、GPU idle；GPU counters 需在 Instruments 內開。")
    print(f"  目標檔案: {os.path.abspath(trace_path)}")
    print("-" * 72)
    print()


def _write_short_metal_gputrace(
    model: Mamba3LanguageModel,
    x_prefill: mx.array,
    prompt_ids: List[int],
    router_temp: mx.array,
    max_cache_len: int,
    trace_path: str,
) -> bool:
    """One prefill + one greedy decode step inside mx.metal capture. Returns True if capture ran."""
    if not trace_path:
        return False
    if not _metal_capture_api():
        print("⚠️  目前 MLX 沒有 mx.metal.start_capture（需支援 Metal capture 的 build）。")
        return False
    if os.environ.get("MTL_CAPTURE_ENABLED", "").strip() != "1":
        print("⚠️  未設定 MTL_CAPTURE_ENABLED=1；Metal 可能拒絕寫入 trace。仍嘗試 start_capture…")
    ap = os.path.abspath(trace_path)
    d = os.path.dirname(ap)
    if d:
        os.makedirs(d, exist_ok=True)
    if os.path.exists(ap):
        try:
            os.remove(ap)
        except OSError as e:
            print(f"⚠️  無法刪除已存在的 trace 檔（MLX 要求路徑不存在）: {e}")
            return False
    _print_metal_capture_instructions(trace_path)
    logits: Any
    caches: Any
    try:
        mx.metal.start_capture(ap)
        logits, caches = model(x_prefill, caches=None, seq_pos=None, router_temp=router_temp)
        mx.eval(logits, caches)
        caches = _pad_transformer_caches(_materialize_cache_tree(caches), max_cache_len)
        mx.eval(caches)
        last = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(last)
        x_one = last.reshape(1, 1)
        h = model.embed(x_one)
        mx.eval(h)
        ch = list(caches)
        seqp = mx.array(len(prompt_ids), dtype=mx.int32)
        h, ch = backbone_forward_incremental(model.backbone, h, ch, seqp, router_temp)
        _eval_any((h, ch))
        hh = model.norm(h)
        out = mhi.fast_scaled_tanh(model.head(hh / math.sqrt(model.config.d_model)), 30.0)
        mx.eval(out)
    finally:
        try:
            mx.metal.stop_capture()
        except Exception:
            pass
    if os.path.isfile(ap):
        print(f"✅ Metal GPU trace 已寫入（請用 Xcode 開啟）: {ap}")
        return True
    print("⚠️  未偵測到輸出檔；請確認 MLX/驅動支援 capture，且環境變數正確。")
    return False


def _one_backbone_layer(
    layer: Any,
    x: mx.array,
    cache: Any,
    seq_pos: Optional[mx.array],
    router_temp: mx.array,
) -> Tuple[mx.array, Any]:
    lt = getattr(layer, "l_type", None)
    if lt == "transformer":
        return layer(x, cache=cache, seq_pos=seq_pos, router_temp=router_temp)
    return layer(x, cache=cache, router_temp=router_temp)


def backbone_forward_incremental(
    backbone: Any,
    x: mx.array,
    caches: Optional[Sequence[Any]],
    seq_pos: Optional[mx.array],
    router_temp: mx.array,
) -> Tuple[mx.array, List[Any]]:
    if caches is None:
        caches = [None] * len(backbone.layers)
    new_caches: List[Any] = []
    for layer, cache in zip(backbone.layers, caches):
        lt = getattr(layer, "l_type", None)
        if lt == "transformer":
            x, nc = layer(x, cache=cache, seq_pos=seq_pos, router_temp=router_temp)
        else:
            x, nc = layer(x, cache=cache, router_temp=router_temp)
        new_caches.append(nc)
    return x, new_caches


def profile_prefill_layers(
    model: Mamba3LanguageModel,
    x: mx.array,
    router_temp: mx.array,
) -> Tuple[List[Tuple[str, Timed, str]], Timed, Timed, float]:
    bb = model.backbone
    rows: List[Tuple[str, Timed, str]] = []

    h, embed_t = _time_block(lambda: model.embed(x))

    caches: List[Any] = [None] * len(bb.layers)
    backbone_sum = 0.0
    for idx, layer in enumerate(bb.layers):
        lt = getattr(layer, "l_type", "unknown")
        # Default-arg bind avoids late-binding closure bugs if this were ever deferred.
        (h, nc), t = _time_block(
            lambda li=layer, ii=idx, hh=h, cc=caches: _one_backbone_layer(
                li, hh, cc[ii], None, router_temp
            )
        )
        caches[idx] = nc
        backbone_sum += t.wall_ms
        rows.append((str(idx), t, str(lt)))

    def _head():
        hh = model.norm(h)
        return mhi.fast_scaled_tanh(model.head(hh / math.sqrt(model.config.d_model)), 30.0)

    _, head_t = _time_block(_head)
    return rows, embed_t, head_t, backbone_sum


def _rollup_by_type(rows: List[Tuple[str, Timed, str]]) -> List[Tuple[str, float, float, int]]:
    agg: dict[str, List[float]] = defaultdict(list)
    for _, t, lt in rows:
        agg[lt].append(t.wall_ms)
    out: List[Tuple[str, float, float, int]] = []
    total = sum(sum(v) for v in agg.values()) or 1e-9
    for lt in sorted(agg.keys()):
        s = sum(agg[lt])
        out.append((lt, s, 100.0 * s / total, len(agg[lt])))
    return out


def _print_component_rollup(mamba_acc: dict[str, float], xf_acc: dict[str, float]) -> None:
    labels = {
        "m01_in_norm_proj_angles": "Mamba: input RMSNorm + in_proj + angles",
        "m02_norm_bc_rope": "Mamba: B/C RMSNorm + RoPE",
        "m03_x_up_proj": "Mamba: x_up (Linear or Tucker MoE)",
        "m04_ssm_core": "Mamba: SSM core (recurrence + y_stack)",
        "m05_dense_branch": "Mamba: y_down + D skip + mamba_dense + out RMSNorm",
        "m06_out_proj": "Mamba: out_proj (Tucker MoE) + residual",
        "xf_norm_qkv": "Transformer: attn RMSNorm + QKV",
        "xf_attn_kv_sdpa_o": "Transformer: KV write + SDPA + O proj + residual",
        "xf_ffn_norm": "Transformer: FFN RMSNorm",
        "ffn_gate_proj": "FFN Tucker gate_proj",
        "ffn_up_proj": "FFN Tucker up_proj",
        "ffn_down_proj": "FFN Tucker down_proj (incl. silu×)",
        "xf_post_ffn_ls": "Transformer: final LayerScale + residual",
    }

    def _block(title: str, acc: dict[str, float]) -> None:
        if not acc:
            return
        total = sum(acc.values()) or 1e-9
        items = sorted(acc.items(), key=lambda kv: -kv[1])
        print(title)
        print("-" * 100)
        print(f"{'Key':<28} {'Component':<48} {'ms (sum)':<12} {'%':<8}")
        print("-" * 100)
        for k, ms in items:
            lab = labels.get(k, k)
            print(f"{k:<28} {lab:<48} {ms:<12.3f} {100.0 * ms / total:<8.1f}")
        print("-" * 100)
        print(f"{'SUBTOTAL':<28} {'':<48} {total:<12.3f} {'100.0':<8}")
        print()

    _block("[Mamba — sum over all Mamba layers, one decode step]", mamba_acc)
    _block("[Transformer — sum over all XF layers, one decode step]", xf_acc)
    approx_bb = sum(mamba_acc.values()) + sum(xf_acc.values())
    print(f"  Approx. backbone component sum (one decode pass, excl. embed & lm_head): {approx_bb:.3f} ms")


def rough_param_bytes(model: Mamba3LanguageModel) -> int:
    n = 0
    for _, v in mx_utils.tree_flatten(model.parameters()):
        n += int(v.size) * (4 if v.dtype == mx.float32 else 2)
    return n


def _timed_to_json(t: Timed) -> dict[str, float]:
    return {"wall_ms": t.wall_ms, "cpu_ms": t.cpu_ms, "wait_proxy_ms": t.wait_proxy_ms}


def main() -> None:
    p = argparse.ArgumentParser(description="MLX hybrid model — layer & host/GPU-proxy profiler (standalone)")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--npz-cache", type=str, default="")
    p.add_argument("--force-pt", action="store_true")
    p.add_argument("--tokenizer", type=str, default=os.path.join(_INF_ROOT, "tokenizer"))
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--profile-decode-steps", type=int, default=32)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--vocab-size", type=int, default=32007)
    p.add_argument(
        "--inference-type",
        type=str,
        default="eager",
        choices=("throughput", "safe", "eager", "sequential-ssm", "custom"),
    )
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--kv-dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--router-temp", type=float, default=0.5)
    p.add_argument("--prompt", type=str, default="Hello! Write one short sentence about MLX on Apple Silicon.")
    p.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Do not wrap --prompt in ChatML (see benchmark_mlx._build_prompt_ids).",
    )
    p.add_argument("--synthetic-prompt", action="store_true")
    p.add_argument("--no-compile-prefill", action="store_true")
    p.add_argument(
        "--fine-op-table",
        action="store_true",
        help="Print decode fine op table (op_key, Σms, calls, shapes, kernel hints); extra sync fences",
    )
    p.add_argument(
        "--metal-gputrace",
        type=str,
        default="",
        metavar="PATH.gputrace",
        help="Capture a short prefill+decode workload to a Metal GPU trace for Xcode Instruments",
    )
    p.add_argument(
        "--decode-timeline",
        action="store_true",
        help="Print CPU-side timestamps for embed / backbone / head within the first decode steps",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default="",
        metavar="PATH.json",
        help="Write full profiling report JSON (includes mlx_profiler-compatible trace)",
    )
    p.add_argument(
        "--mlx-profiler-json",
        type=str,
        default="",
        metavar="PATH.json",
        help="Write mlx-profiler Trace JSON only (mlx-profiler view PATH)",
    )
    args = p.parse_args()
    if not args.no_compile_prefill and args.inference_type == "eager":
        args.no_compile_prefill = True

    _apply_inference_type(args)

    compute_dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    target_dtype = compute_dtype_map[args.dtype]
    kv_map = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
    kv_dtype = target_dtype if args.kv_dtype == "auto" else kv_map[args.kv_dtype]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = len(tokenizer) if args.vocab_size <= 0 else args.vocab_size

    # Long prefill: sequential Mamba (use_parallel_scan=False) unrolls L steps in the graph and
    # can spike unified memory; keep chunk scan on unless user explicitly chose sequential-ssm.
    use_parallel_mamba = args.inference_type != "sequential-ssm"

    config = Mamba3Config(
        d_model=768,
        d_state=64,
        d_head=64,
        expand=2,
        num_layers=6,
        mimo_rank=4,
        num_kv_heads=4,
        use_parallel_scan=use_parallel_mamba,
        chunk_size=64,
        use_kmoe=True,
        kmoe_num_experts=8,
        kmoe_top_k=2,
        kmoe_r1=32,
        kmoe_r2=512,
        kmoe_r3=256,
        ffn_expand=6,
    )
    model = Mamba3LanguageModel(config, vocab_size)

    resolved, kind = resolve_mlx_checkpoint(
        args.checkpoint, repo_root=_REPO_ROOT, npz_cache=args.npz_cache, force_pt=args.force_pt
    )
    if resolved and kind != "none":
        strict_load_and_convert(model, resolved)
        if kind == "pt":
            sidecar = maybe_export_npz_sidecar_after_pt_load(
                model, resolved, force_refresh=args.force_pt
            )
            if sidecar is not None:
                print(f"💾 已建立 npz 快取，下次將優先讀取: {sidecar}")

    model.apply(lambda x: x.astype(target_dtype))
    mx.eval(model.parameters())
    _invalidate_tucker_caches(model)
    router_temp = mx.array(args.router_temp, dtype=target_dtype)

    max_cache_len = args.seq_len + args.profile_decode_steps + 8
    attach_decode_compilation(model, max_cache_len=max_cache_len, kv_dtype=kv_dtype, compile_decode=False)
    model.set_lm_head_compile(False)

    if args.synthetic_prompt:
        filler = list(tokenizer.encode("benchmark " * max(1, args.seq_len // 4)))[: args.seq_len]
        if len(filler) < args.seq_len:
            filler = (filler * (args.seq_len // max(len(filler), 1) + 1))[: args.seq_len]
        prompt_ids = filler
    else:
        prompt_ids = _build_prompt_ids(
            tokenizer, args.prompt, args.seq_len, chatml_user=not args.raw_prompt
        )
    x_prefill = mx.array([prompt_ids], dtype=mx.int32)

    mx.reset_peak_memory()
    for _ in range(args.warmup):
        logits, caches = model(x_prefill, caches=None, seq_pos=None, router_temp=router_temp)
        mx.eval(logits, caches)

    print("🔍 MLX / Metal Bottleneck Profile (short run)")
    print()
    print("ℹ️  MLX 不暴露 ATen 級算子名稱；此為各段在 mx.eval + mx.synchronize() 後的 wall time。")
    print("   CPU = time.thread_time()（多數情況不含阻塞在 GPU 上的等待）。")
    print("   wait/GPU proxy ≈ wall − CPU。逐層 eval 會比單次 fused eval 多同步開銷。")
    print(f"   Decode 取樣步數: {args.profile_decode_steps}（greedy argmax）")
    print()

    rows, embed_t, head_t, bb_sum = profile_prefill_layers(model, x_prefill, router_temp)
    _release_lazy_graph()

    rollup = _rollup_by_type(rows)
    type_wait: dict[str, float] = defaultdict(float)
    for _, t, lt in rows:
        type_wait[lt] += t.wait_proxy_ms

    print("[Prefill — backbone by layer type]")
    print("-" * 92)
    print(f"{'Name':<22} {'Self wall (ms)':<16} {'% of backbone':<16} {'Layer visits':<12} {'Σ wait~ (ms)':<16}")
    print("-" * 92)
    for name, ms, pct, visits in rollup:
        tw = type_wait.get(name, 0.0)
        print(f"{name:<22} {ms:<16.3f} {pct:<16.2f} {visits:<12} {tw:<16.3f}")
    print("-" * 92)
    print(f"{'Backbone sum (wall ms)':<40} {bb_sum:.3f}")
    print()

    top = sorted(rows, key=lambda r: r[1].wall_ms, reverse=True)[:30]
    print("[Top 30 layers by wall time (single prefill visit)]")
    print("-" * 76)
    print(f"{'Layer':<8} {'Type':<18} {'Wall ms':<12} {'CPU ms':<12} {'wait~ ms':<12}")
    print("-" * 76)
    for idx, t, lt in top:
        print(f"{idx:<8} {lt:<18} {t.wall_ms:<12.3f} {t.cpu_ms:<12.3f} {t.wait_proxy_ms:<12.3f}")
    print("-" * 76)
    print()

    print(
        "[Prefill segments]  "
        f"embed {embed_t.wall_ms:.3f} ms (CPU {embed_t.cpu_ms:.3f}, wait~ {embed_t.wait_proxy_ms:.3f}) | "
        f"norm+head {head_t.wall_ms:.3f} ms | backbone Σ(layer) {bb_sum:.3f} ms"
    )
    print()

    _release_lazy_graph()
    mx.reset_peak_memory()

    # --- Decode: rolling caches from one prefill (single graph per step: no duplicate full model()) ---
    logits, caches = model(x_prefill, caches=None, seq_pos=None, router_temp=router_temp)
    mx.eval(logits, caches)
    caches = _materialize_cache_tree(caches)
    mx.eval(caches)
    caches = _pad_transformer_caches(caches, max_cache_len)
    mx.eval(caches)

    pos = len(prompt_ids)
    last = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
    mx.eval(last)
    _release_lazy_graph(logits)

    dec_embed: List[float] = []
    dec_bb: List[float] = []
    dec_head: List[float] = []
    dec_cpu_e: List[float] = []
    dec_wait_e: List[float] = []

    logits_d: Optional[mx.array] = None
    timeline_rows: List[Tuple[int, List[Tuple[str, float]]]] = []
    for step_i in range(args.profile_decode_steps):
        x_one = last.reshape(1, 1)
        tl: List[Tuple[str, float]] = []
        t_lane0 = time.perf_counter()

        h, e_t = _time_block(lambda: model.embed(x_one))
        dec_embed.append(e_t.wall_ms)
        dec_cpu_e.append(e_t.cpu_ms)
        dec_wait_e.append(e_t.wait_proxy_ms)
        if args.decode_timeline and step_i < 4:
            tl.append(("embed_end_ms", (time.perf_counter() - t_lane0) * 1000.0))

        _sync()
        ch = list(caches)
        seqp = mx.array(pos, dtype=mx.int32)
        w0 = time.perf_counter()
        h, ch = backbone_forward_incremental(model.backbone, h, ch, seqp, router_temp)
        _eval_any((h, ch))
        _sync()
        w1 = time.perf_counter()
        dec_bb.append((w1 - w0) * 1000.0)
        if args.decode_timeline and step_i < 4:
            tl.append(("backbone_end_ms", (time.perf_counter() - t_lane0) * 1000.0))

        def _hd():
            hh = model.norm(h)
            return mhi.fast_scaled_tanh(model.head(hh / math.sqrt(model.config.d_model)), 30.0)

        logits_d, hd_t = _time_block(_hd)
        dec_head.append(hd_t.wall_ms)
        if args.decode_timeline and step_i < 4:
            tl.append(("head_end_ms", (time.perf_counter() - t_lane0) * 1000.0))
            timeline_rows.append((step_i, tl))

        last = mx.argmax(logits_d[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(last)
        caches = ch
        mx.eval(caches)
        pos += 1

    if args.profile_decode_steps > 0:
        _release_lazy_graph(logits_d, h, last, x_one)

    n = max(1, args.profile_decode_steps)
    avg_e = sum(dec_embed) / n
    avg_bb = sum(dec_bb) / n
    avg_h = sum(dec_head) / n
    step_wall = avg_e + avg_bb + avg_h

    print(
        f"[Decode {args.profile_decode_steps} steps — avg wall]  "
        f"embed/step {avg_e:.3f} ms | backbone/step {avg_bb:.3f} ms | "
        f"norm+head/step {avg_h:.3f} ms | ~step {step_wall:.3f} ms"
    )
    print()

    if args.decode_timeline and timeline_rows:
        print("[Decode timeline — CPU stamps from step start to phase end (first ≤4 steps)]")
        print("-" * 72)
        print("  僅主線程 wall；不含 GPU queue 真實重疊。對照請用 --metal-gputrace + Xcode。")
        for si, events in timeline_rows:
            parts = " | ".join(f"{n} {v:.3f}" for n, v in events)
            print(f"  step {si}: {parts}")
        print("-" * 72)
        print()

    print("🧩 Decode component rollup (single step, eager, no layer compile)")
    print("   將每個子階段在所有同名層上累加；分段與 mlx_profile_components 對齊。")
    logits, caches = model(x_prefill, caches=None, seq_pos=None, router_temp=router_temp)
    mx.eval(logits, caches)
    caches = _pad_transformer_caches(_materialize_cache_tree(caches), max_cache_len)
    mx.eval(caches)
    last_tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
    mx.eval(last_tok)
    x_one = last_tok.reshape(1, 1)
    h = model.embed(x_one)
    mx.eval(h)
    _sync()
    ch = list(caches)
    seq_pos_c = mx.array(len(prompt_ids), dtype=mx.int32)
    _, _, mamba_acc, xf_acc = aggregate_decode_component_profile(
        model.backbone, h, ch, seq_pos_c, router_temp
    )
    _print_component_rollup(mamba_acc, xf_acc)
    _release_lazy_graph(logits, caches, last_tok, x_one, h, ch)
    print()

    fine_tab: Optional[FineDecodeTable] = None
    need_fine = args.fine_op_table or bool(args.json_out.strip()) or bool(args.mlx_profiler_json.strip())
    if need_fine:
        if args.fine_op_table:
            print("[Fine op table — decode, one token; Σms ≫ single fused step due to many eval fences]")
            print("-" * 72)
        logits_f, caches_f = model(x_prefill, caches=None, seq_pos=None, router_temp=router_temp)
        mx.eval(logits_f, caches_f)
        caches_f = _pad_transformer_caches(_materialize_cache_tree(caches_f), max_cache_len)
        mx.eval(caches_f)
        last_f = mx.argmax(logits_f[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(last_f)
        x_one_f = last_f.reshape(1, 1)
        h_f = model.embed(x_one_f)
        mx.eval(h_f)
        ch_f = list(caches_f)
        seq_pos_f = mx.array(len(prompt_ids), dtype=mx.int32)
        fine_tab, _, _ = aggregate_fine_decode_profile(
            model.backbone, h_f, ch_f, seq_pos_f, router_temp
        )
        if args.fine_op_table:
            print_fine_table(
                "  Keys are logical MLX ops / subgraphs (not literal Metal kernel names).", fine_tab
            )
        _release_lazy_graph(logits_f, caches_f, last_f, x_one_f, h_f, ch_f)

    # Decode single-step per-layer (reuse state after fresh prefill)
    print("[Decode — single-step layer split (first token, eager)]")
    logits, caches = model(x_prefill, caches=None, seq_pos=None, router_temp=router_temp)
    mx.eval(logits, caches)
    caches = _pad_transformer_caches(_materialize_cache_tree(caches), max_cache_len)
    mx.eval(caches)
    last_tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
    mx.eval(last_tok)
    x_one = last_tok.reshape(1, 1)
    h = model.embed(x_one)
    mx.eval(h)
    _sync()
    ch = list(caches)
    seq_pos_f = mx.array(len(prompt_ids), dtype=mx.int32)
    bb_rows: List[Tuple[str, Timed, str]] = []
    for idx, layer in enumerate(model.backbone.layers):
        lt = getattr(layer, "l_type", "unknown")
        (h, nc), tt = _time_block(
            lambda li=layer, ii=idx, hh=h, cc=ch: _one_backbone_layer(
                li, hh, cc[ii], seq_pos_f, router_temp
            )
        )
        ch[idx] = nc
        bb_rows.append((str(idx), tt, str(lt)))

    agg2: dict[str, float] = defaultdict(float)
    cnt2: dict[str, int] = defaultdict(int)
    for _, tt, lt in bb_rows:
        agg2[lt] += tt.wall_ms
        cnt2[lt] += 1
    tot2 = sum(agg2.values()) or 1e-9
    print("-" * 88)
    print(f"{'Name':<22} {'Self wall (ms)':<16} {'% of backbone':<16} {'Visits':<8}")
    print("-" * 88)
    for lt in sorted(agg2.keys()):
        print(f"{lt:<22} {agg2[lt]:<16.3f} {100 * agg2[lt] / tot2:<16.2f} {cnt2[lt]:<8}")
    print("-" * 88)
    print()
    peak_mb = mx.get_peak_memory() / (1024**2)
    _release_lazy_graph(logits, caches, last_tok, x_one, h, ch)

    print("[Host vs device (proxy, this thread)]")
    print("-" * 72)
    print(f"  Prefill embed:  wall {embed_t.wall_ms:.3f} | CPU {embed_t.cpu_ms:.3f} | wait~ {embed_t.wait_proxy_ms:.3f} ms")
    print(f"  Prefill head:   wall {head_t.wall_ms:.3f} | CPU {head_t.cpu_ms:.3f} | wait~ {head_t.wait_proxy_ms:.3f} ms")
    print(
        f"  Decode embed avg: wall {sum(dec_embed)/n:.3f} | CPU {sum(dec_cpu_e)/n:.3f} | wait~ {sum(dec_wait_e)/n:.3f} ms"
    )
    print("  若 wall ≫ CPU：主線程多半在等 GPU/驅動；若兩者接近：同步或 Python 開銷較顯著。")
    print()

    print("[Memory (MLX Metal)]")
    print("-" * 72)
    print(
        f"  Peak (Metal, since reset before rolling decode): {peak_mb:.2f} MB"
    )
    print(f"  Active now: {mx.get_active_memory() / (1024**2):.2f} MB")
    print()

    pbytes = rough_param_bytes(model)
    print("[Rough roofline / bound hint]")
    print("-" * 72)
    print(f"  Param storage (heuristic MB): {pbytes / 1e6:.2f}")
    print("  精確區分 compute-bound vs memory-bound 請用 Xcode Instruments → Metal GPU Counters。")
    print()

    mlx_trace: Optional[dict[str, Any]] = None
    if args.json_out.strip() or args.mlx_profiler_json.strip():
        dtype_ml = profiler_dtype_string(args.dtype)
        trace_extra = {
            "seq_len": args.seq_len,
            "profile_decode_steps": args.profile_decode_steps,
            "dtype_cli": args.dtype,
            "inference_type": args.inference_type,
        }
        mlx_trace = build_mlx_profiler_trace_dict(
            name="mamba3_hybrid_decode_profile",
            dtype=dtype_ml,
            fine_table=fine_tab,
            mamba_acc=mamba_acc,
            xf_acc=xf_acc,
            extra_metadata=trace_extra,
            include_component_ops=True,
            include_fine_ops=fine_tab is not None,
        )

    if args.json_out.strip() and mlx_trace is not None:
        ap = os.path.abspath(args.json_out.strip())
        d = os.path.dirname(ap)
        if d:
            os.makedirs(d, exist_ok=True)
        envelope: dict[str, Any] = {
            "schema_version": 1,
            "profiler": "profile_mlx_infer",
            "generated_unix_s": time.time(),
            "argv": sys.argv,
            "dtype_mlx_profiler": profiler_dtype_string(args.dtype),
            "prefill": {
                "embed_ms": embed_t.wall_ms,
                "head_ms": head_t.wall_ms,
                "backbone_sum_layer_ms": bb_sum,
                "embed": _timed_to_json(embed_t),
                "head": _timed_to_json(head_t),
                "rollup_by_layer_type": [
                    {"name": name, "wall_ms": ms, "pct_of_backbone": pct, "visits": visits, "sum_wait_proxy_ms": type_wait.get(name, 0.0)}
                    for name, ms, pct, visits in rollup
                ],
                "layers": [
                    {"index": idx, "type": lt, **_timed_to_json(t)} for idx, t, lt in rows
                ],
                "top_30_layers": [
                    {"index": idx, "type": lt, **_timed_to_json(t)} for idx, t, lt in top
                ],
            },
            "decode_rolling": {
                "steps": args.profile_decode_steps,
                "avg_embed_ms": avg_e,
                "avg_backbone_ms": avg_bb,
                "avg_norm_head_ms": avg_h,
                "approx_step_ms": step_wall,
                "per_step_embed_ms": dec_embed,
                "per_step_backbone_ms": dec_bb,
                "per_step_head_ms": dec_head,
            },
            "decode_component_rollup_one_step": {"mamba": mamba_acc, "transformer": xf_acc},
            "decode_single_step_by_layer_type": {
                lt: {"wall_ms": agg2[lt], "visits": cnt2[lt], "pct_of_backbone": 100.0 * agg2[lt] / tot2}
                for lt in sorted(agg2.keys())
            },
            "memory_metal_mb": {"peak": peak_mb, "active": mx.get_active_memory() / (1024**2)},
            "param_bytes_heuristic": pbytes,
            "decode_timeline_first_steps": [
                {
                    "step": si,
                    "phases_ms_from_step_start": [{"name": n, "ms": v} for n, v in events],
                }
                for si, events in timeline_rows
            ],
            "mlx_profiler_trace": mlx_trace,
        }
        with open(ap, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2, ensure_ascii=False)
        print(f"📄 Full JSON report: {ap}")

    if args.mlx_profiler_json.strip() and mlx_trace is not None:
        mp = os.path.abspath(args.mlx_profiler_json.strip())
        d2 = os.path.dirname(mp)
        if d2:
            os.makedirs(d2, exist_ok=True)
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(mlx_trace, f, indent=2, ensure_ascii=False)
        print(f"📄 mlx-profiler trace JSON: {mp}")
        print("   Try:  mlx-profiler view " + mp)
    if args.json_out.strip() or args.mlx_profiler_json.strip():
        print()

    if args.metal_gputrace.strip():
        _write_short_metal_gputrace(
            model,
            x_prefill,
            prompt_ids,
            router_temp,
            max_cache_len,
            args.metal_gputrace.strip(),
        )
        print()


if __name__ == "__main__":
    main()
