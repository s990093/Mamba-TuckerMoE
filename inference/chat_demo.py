#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mamba3-XR Chat Demo — WebSocket real-time streaming chat.

Launches a local FastAPI server with:
  - True WebSocket bidirectional connection (persistent, no reconnect per turn)
  - Multi-turn ChatML conversation history
  - Token-by-token streaming with live metrics
  - Beautiful chat interface for demo

Usage:
  python inference/chat_demo.py                           # defaults
  python inference/chat_demo.py --port 8080             # custom port
  python inference/chat_demo.py --checkpoint path/to.pt # custom weights
  python inference/chat_demo.py --mock                  # UI only; data from ui/mock_config.json

Requires: pip install fastapi uvicorn websockets (MLX stack still imported for non-mock runs)
"""
from __future__ import annotations

import random
import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

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
    maybe_export_npz_sidecar_after_pt_load,
    resolve_mlx_checkpoint,
    strict_load_and_convert,
)

_CATEGORY_PROMPTS_MERGE: Any = None


def _category_system_prompts_map(data: dict[str, Any]) -> dict[str, str]:
    """Per sidebar category key → short SFT/export string (see `cot_dataset/category_system_prompts.py`)."""
    global _CATEGORY_PROMPTS_MERGE
    if _CATEGORY_PROMPTS_MERGE is None:
        try:
            _cp = os.path.join(_REPO_ROOT, "cot_dataset")
            if _cp not in sys.path:
                sys.path.insert(0, _cp)
            from category_system_prompts import merged_category_prompts_for_api as _m

            _CATEGORY_PROMPTS_MERGE = _m
        except Exception:
            _CATEGORY_PROMPTS_MERGE = False
    if callable(_CATEGORY_PROMPTS_MERGE):
        return dict(_CATEGORY_PROMPTS_MERGE(data))
    return {}

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model: Mamba3LanguageModel | None = None
_tokenizer: Any = None
_run_prefill: Any = None
_router_temp: mx.array | None = None
_max_cache_len: int = 0
_args: Any = None
_config: Any = None
_stop_ids: frozenset[int] | None = None
_model_ready = False
_vocab_size: int = 0
_MOCK_MODE: bool = False

# Inference lock — model is single-threaded on Metal GPU
_infer_lock = asyncio.Lock()


def _read_ui_config() -> dict[str, Any]:
    """Load `ui/mock_config.json` for mock status, stream chunks, and welcome examples."""
    try:
        p = Path(__file__).parent / "ui" / "mock_config.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _load_model(args: argparse.Namespace) -> dict[str, Any]:
    global _model, _tokenizer, _run_prefill, _router_temp
    global _max_cache_len, _args, _config, _stop_ids, _model_ready, _vocab_size

    _args = args
    timings: dict[str, Any] = {}
    t0 = time.perf_counter()

    compute_dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    target_dtype = compute_dtype_map[args.dtype]
    kv_map = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
    kv_dtype = target_dtype if args.kv_dtype == "auto" else kv_map[args.kv_dtype]

    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    _vocab_size = len(_tokenizer) if args.vocab_size <= 0 else args.vocab_size
    timings["tokenizer_ms"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    _config = Mamba3Config(
        d_model=768, d_state=64, d_head=64, expand=2, num_layers=6,
        mimo_rank=4, num_kv_heads=4, use_parallel_scan=args.use_parallel_scan,
        chunk_size=64, use_kmoe=True, kmoe_num_experts=8, kmoe_top_k=2,
        kmoe_r1=32, kmoe_r2=512, kmoe_r3=256, ffn_expand=6,
    )
    _config.lookahead_router = False
    _config.tucker_einsum_fuse = bool(args.tucker_einsum_fuse)
    _config.tucker_amx_fuse = False
    _model = Mamba3LanguageModel(_config, _vocab_size)
    timings["model_init_ms"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    resolved, kind = resolve_mlx_checkpoint(
        args.checkpoint, repo_root=_REPO_ROOT,
        npz_cache=args.npz_cache, force_pt=args.force_pt,
    )
    if resolved is None or kind == "none":
        print("[chat_demo] No checkpoint — random weights (smoke test).")
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            strict_load_and_convert(_model, resolved)
        if kind == "pt":
            maybe_export_npz_sidecar_after_pt_load(_model, resolved, force_refresh=args.force_pt)
    timings["weights_ms"] = (time.perf_counter() - t2) * 1000
    timings["checkpoint"] = resolved or "(random init)"
    timings["kind"] = kind

    _model.apply(lambda x: x.astype(target_dtype))
    if args.quantize > 0:
        nn.quantize(_model, group_size=64, bits=args.quantize)
    mx.eval(_model.parameters())
    _invalidate_tucker_caches(_model)
    _router_temp = mx.array(args.router_temp, dtype=target_dtype)

    pre_budget = int(args.seq_len) if args.seq_len > 0 else 4096
    _max_cache_len = pre_budget + args.max_new_tokens + 8

    attach_decode_compilation(
        _model, max_cache_len=_max_cache_len, kv_dtype=kv_dtype,
        compile_decode=False,
    )

    def prefill_forward(x: mx.array, rt: mx.array):
        return _model(x, caches=None, seq_pos=None, router_temp=rt)

    _run_prefill = mx.compile(prefill_forward)

    from stream_mlx import _build_stream_stop_token_ids
    _stop_ids = _build_stream_stop_token_ids(_tokenizer, enabled=True)

    t3 = time.perf_counter()
    warmup_ids = _build_prompt_ids(_tokenizer, "warmup", args.seq_len, chatml_user=True)
    x_warm = mx.array([warmup_ids], dtype=mx.int32)
    for _ in range(args.warmup):
        lo, ca = _run_prefill(x_warm, _router_temp)
        mx.eval(lo, ca)
    timings["warmup_ms"] = (time.perf_counter() - t3) * 1000
    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    _model_ready = True
    return timings


# ---------------------------------------------------------------------------
# Multi-turn ChatML prompt builder
# ---------------------------------------------------------------------------
def _encode_plain(text: str) -> list[int]:
    try:
        return list(_tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(_tokenizer.encode(text))


def _build_multiturn_ids(
    history: list[dict],
    seq_len: int,
    system_prompt: str | None = None,
    reasoning: bool = False,
) -> list[int]:
    """ChatML prompt builder aligned with the production SFT-CoT script.

    - When `system_prompt` is provided, prepends `<|im_start|>system\n...<|im_end|>\n`.
    - When `reasoning=True`, appends `<think>\n` after the assistant marker so the
      small SFT-CoT model continues inside the reasoning distribution
      (`<think>…</think><final>…</final>`).
    """
    parts: list[str] = []
    sys_text = (system_prompt or "").strip()
    if sys_text:
        parts.append(f"<|im_start|>system\n{sys_text}<|im_end|>\n")
    for msg in history:
        role = msg["role"]
        content = msg["content"].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if reasoning:
        parts.append("<think>\n")
    full_prompt = "".join(parts)
    ids = _encode_plain(full_prompt)
    if seq_len > 0 and len(ids) > seq_len:
        ids = ids[-seq_len:]
    if not ids:
        tail = "<|im_start|>assistant\n" + ("<think>\n" if reasoning else "")
        ids = _encode_plain(tail)
    return ids


# === CoT stream splitter -----------------------------------------------------
class _CotStreamSplitter:
    """Stateful splitter that consumes streamed decoded text from the model
    (which may include `<think>…</think>` and `<final>…</final>` blocks) and
    emits ``(kind, text)`` events with `kind` in ``{"reasoning", "final", "stop"}``.

    Buffers partial tag suffixes so a tag spanning two chunks never leaks into
    output. When the prompt already injected ``<think>\n`` (reasoning mode), set
    ``start_in_think=True`` so the splitter starts inside the think block.
    """

    TAGS = ("<think>", "</think>", "<final>", "</final>", "<|im_end|>")
    MAX_TAG_LEN = max(len(t) for t in TAGS)

    def __init__(self, start_in_think: bool = False) -> None:
        self._buf = ""
        # "head"  → before any block (also tolerates a plain answer with no tags)
        # "think" → inside <think>...
        # "between" → after </think>, waiting for <final> or stop
        # "final" → inside <final>...
        # "done"  → terminal
        self._mode = "think" if start_in_think else "head"
        self._loose_final = not start_in_think  # tagless answer? then 'head' acts as final

    @property
    def done(self) -> bool:
        return self._mode == "done"

    def _safe_tail_cut(self) -> str:
        """Pop & return prefix of `_buf` that is safe to flush; keep tail that
        could still be the start of any TAG. Mutates `_buf`."""
        s = self._buf
        n = len(s)
        keep = 0
        for k in range(1, min(self.MAX_TAG_LEN, n) + 1):
            suf = s[n - k:]
            if any(t.startswith(suf) for t in self.TAGS):
                keep = k
        emit = s[: n - keep]
        self._buf = s[n - keep:]
        return emit

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        if not chunk:
            return []
        self._buf += chunk
        out: list[tuple[str, str]] = []
        progressed = True
        while progressed:
            progressed = False
            if self._mode == "done":
                self._buf = ""
                break
            if self._mode in ("head", "between"):
                ti = self._buf.find("<think>")
                fi = self._buf.find("<final>")
                ei = self._buf.find("<|im_end|>")
                fe = self._buf.find("</final>")
                hits = [(i, n) for i, n in [(ti, "think"), (fi, "final"), (ei, "stop"), (fe, "stop")] if i >= 0]
                if hits:
                    i, name = min(hits)
                    if self._mode == "head" and self._loose_final and i > 0:
                        out.append(("final", self._buf[:i]))
                    if name == "think":
                        self._buf = self._buf[i + len("<think>"):]
                        self._mode = "think"
                    elif name == "final":
                        self._buf = self._buf[i + len("<final>"):]
                        self._mode = "final"
                    else:
                        self._mode = "done"
                        out.append(("stop", ""))
                    progressed = True
                else:
                    if self._mode == "head" and self._loose_final:
                        emit = self._safe_tail_cut()
                        if emit:
                            out.append(("final", emit))
                    else:
                        self._buf = self._buf[-(self.MAX_TAG_LEN - 1):] if len(self._buf) > self.MAX_TAG_LEN else self._buf
            elif self._mode == "think":
                idx_close = self._buf.find("</think>")
                idx_end = self._buf.find("<|im_end|>")
                hits = [(i, n) for i, n in [(idx_close, "close"), (idx_end, "stop")] if i >= 0]
                if hits:
                    i, name = min(hits)
                    if i > 0:
                        out.append(("reasoning", self._buf[:i]))
                    if name == "close":
                        self._buf = self._buf[i + len("</think>"):]
                        self._mode = "between"
                    else:
                        self._mode = "done"
                        out.append(("stop", ""))
                    progressed = True
                else:
                    emit = self._safe_tail_cut()
                    if emit:
                        out.append(("reasoning", emit))
            elif self._mode == "final":
                fe = self._buf.find("</final>")
                ie = self._buf.find("<|im_end|>")
                hits = [(i, "stop") for i in (fe, ie) if i >= 0]
                if hits:
                    i, _ = min(hits)
                    if i > 0:
                        out.append(("final", self._buf[:i]))
                    self._mode = "done"
                    out.append(("stop", ""))
                    progressed = True
                else:
                    emit = self._safe_tail_cut()
                    if emit:
                        out.append(("final", emit))
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Emit any remaining safe text at end-of-stream (e.g. when the model
        stops on EOS without a closing `</final>`). Drop a trailing partial
        tag-prefix so fragments like ``</th`` never leak into the UI."""
        s = self._buf
        n = len(s)
        keep_drop = 0
        for k in range(1, min(self.MAX_TAG_LEN, n) + 1):
            suf = s[n - k:]
            if any(t.startswith(suf) for t in self.TAGS):
                keep_drop = k
        emit = s[: n - keep_drop]
        out: list[tuple[str, str]] = []
        if emit:
            if self._mode == "think":
                out.append(("reasoning", emit))
            elif self._mode == "final" or (self._mode == "head" and self._loose_final):
                out.append(("final", emit))
        self._buf = ""
        self._mode = "done"
        return out


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------
_COT_LITERAL_TAGS = ("<think>", "</think>", "<final>", "</final>")


# === Primed system-prefix cache (real-mode KV reuse) =========================
# Maps system_prompt text → {"ids": [...], "caches": <mx tree>, "pos": int}.
# Built by `play_system_prompt` (and optionally auto-primed on WS connect) so
# subsequent chats that share the same system prompt only run a *continuation*
# prefill over the user turn + assistant marker (much cheaper than re-running
# prefill over the whole system block every time).
_primed_prefix: dict[str, dict[str, Any]] = {}


def _deepcopy_caches(c: Any) -> Any:
    """Shallow-recursive copy of the cache tree so reusing the primed entry
    across multiple chats cannot mutate the canonical primed state."""
    if isinstance(c, mx.array):
        return mx.array(c)
    if isinstance(c, list):
        return [_deepcopy_caches(x) for x in c]
    if isinstance(c, tuple):
        return tuple(_deepcopy_caches(x) for x in c)
    if isinstance(c, dict):
        return {k: _deepcopy_caches(v) for k, v in c.items()}
    return c


def _build_system_block_ids(system_prompt: str) -> list[int]:
    text = f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
    return _encode_plain(text)


def _prime_system_prefix_sync(system_prompt: str) -> dict[str, Any]:
    """Run prefill over the system block alone and cache the resulting state.
    Caller must hold ``_infer_lock``. Return ``{"ok": bool, ...}``."""
    if not _model_ready or _MOCK_MODE:
        return {"ok": False, "reason": "model not ready or mock mode", "prefill_ms": 0.0, "prompt_tokens": 0}
    sys_text = (system_prompt or "").strip()
    if not sys_text:
        return {"ok": False, "reason": "empty system prompt", "prefill_ms": 0.0, "prompt_tokens": 0}
    if sys_text in _primed_prefix:
        cached = _primed_prefix[sys_text]
        return {"ok": True, "cached": True, "prompt_tokens": int(cached["pos"]), "prefill_ms": 0.0}

    ids = _build_system_block_ids(sys_text)
    if not ids:
        return {"ok": False, "reason": "tokenizer returned no ids", "prefill_ms": 0.0, "prompt_tokens": 0}
    if _args.seq_len > 0 and len(ids) > _args.seq_len:
        return {
            "ok": False,
            "reason": f"system block too long ({len(ids)} > seq_len {_args.seq_len})",
            "prefill_ms": 0.0,
            "prompt_tokens": len(ids),
        }

    x = mx.array([ids], dtype=mx.int32)
    t0 = time.perf_counter()
    try:
        logits, caches = _run_prefill(x, _router_temp)
        mx.eval(logits, caches)
        if not _args.no_materialize_caches:
            caches = _materialize_cache_tree(caches)
            mx.eval(caches)
        caches = _pad_transformer_caches(caches, _max_cache_len)
        mx.eval(caches)
    except Exception as exc:
        return {"ok": False, "reason": f"prefill failed: {exc}", "prefill_ms": 0.0, "prompt_tokens": len(ids)}

    prefill_ms = (time.perf_counter() - t0) * 1000
    _primed_prefix[sys_text] = {
        "ids": list(ids),
        "caches": caches,
        "pos": len(ids),
    }
    return {
        "ok": True,
        "cached": False,
        "prompt_tokens": len(ids),
        "prefill_ms": round(prefill_ms, 2),
    }


def _stream_generate(
    history: list[dict],
    max_tokens: int = 512,
    system_prompt: str | None = None,
    reasoning: bool = False,
    args_for_call: Any = None,
) -> Iterator[dict]:
    """
    Yields dicts sent as WebSocket JSON frames:
      {"type": "meta",  ...}
      {"type": "reasoning", "markdown": "...", ...}   # accumulated <think> body
      {"type": "token", "text": "...", ...}           # streamed <final> body
      {"type": "done",  ...}
    """
    assert _model_ready

    sys_text = (system_prompt or "").strip()
    prompt_ids: list[int] = []
    caches = None
    logits = None
    prefill_ms = 0.0
    cached_prefix_tokens = 0
    primed = _primed_prefix.get(sys_text) if sys_text else None

    if primed:
        # === Continuation prefill: reuse the primed system-block KV cache. ====
        suffix_parts: list[str] = []
        for msg in history:
            role = msg["role"]
            content = msg["content"].strip()
            suffix_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        suffix_parts.append("<|im_start|>assistant\n")
        if reasoning:
            suffix_parts.append("<think>\n")
        suffix_ids = _encode_plain("".join(suffix_parts))
        full_len = int(primed["pos"]) + len(suffix_ids)
        if (_args.seq_len > 0 and full_len > _args.seq_len) or full_len + 8 > _max_cache_len:
            primed = None  # too long; fall through to full prefill below
        else:
            try:
                starting_caches = _deepcopy_caches(primed["caches"])
                x_suffix = mx.array([suffix_ids], dtype=mx.int32)
                seq_pos = mx.array(int(primed["pos"]), dtype=mx.int32)
                t_pre0 = time.perf_counter()
                logits, caches = _model(
                    x_suffix, caches=starting_caches, seq_pos=seq_pos, router_temp=_router_temp,
                )
                mx.eval(logits, caches)
                if not _args.no_materialize_caches:
                    caches = _materialize_cache_tree(caches)
                    mx.eval(caches)
                caches = _pad_transformer_caches(caches, _max_cache_len)
                mx.eval(caches)
                prefill_ms = (time.perf_counter() - t_pre0) * 1000
                prompt_ids = list(primed["ids"]) + list(suffix_ids)
                cached_prefix_tokens = int(primed["pos"])
            except Exception as exc:
                print(f"[chat_demo] WARN: continuation prefill failed → fallback. ({exc})")
                primed = None

    if not primed:
        prompt_ids = _build_multiturn_ids(
            history, _args.seq_len, system_prompt=system_prompt, reasoning=reasoning,
        )
        x_prefill = mx.array([prompt_ids], dtype=mx.int32)
        t_pre0 = time.perf_counter()
        logits, caches = _run_prefill(x_prefill, _router_temp)
        mx.eval(logits, caches)
        prefill_ms = (time.perf_counter() - t_pre0) * 1000
        if not _args.no_materialize_caches:
            caches = _materialize_cache_tree(caches)
            mx.eval(caches)
        caches = _pad_transformer_caches(caches, _max_cache_len)
        mx.eval(caches)

    meta_ev: dict[str, Any] = {
        "type": "meta",
        "prefill_ms": round(prefill_ms, 2),
        "prompt_tokens": len(prompt_ids),
        "turns": len([m for m in history if m["role"] == "user"]),
    }
    if cached_prefix_tokens:
        meta_ev["cached_prefix_tokens"] = cached_prefix_tokens
    yield meta_ev

    from stream_mlx import make_compiled_decode_step
    decode_fn = make_compiled_decode_step(_model, _router_temp)

    pos = len(prompt_ids)
    generated_ids: list[int] = []
    special_ids = set(getattr(_tokenizer, "all_special_ids", []) or [])
    prev_decoded_text = ""

    sample_args = args_for_call if args_for_call is not None else _args
    row = logits[0, -1, :]
    token_counts = _init_token_counts(prompt_ids, int(row.shape[0]))
    last = sample_decode_token(row, token_counts, sample_args)
    token_counts[last] = token_counts[last] + 1
    mx.eval(last)

    t_dec0 = time.perf_counter()
    ttft_ms: float | None = None
    n_out = 0

    splitter = _CotStreamSplitter(start_in_think=bool(reasoning))
    reasoning_acc: list[str] = []

    def _decode_chunk(tid: int) -> str:
        """Decode a single id into a streaming-safe string. If `tid` decodes to
        one of the CoT literal tags (`<think>` / `</think>` / `<final>` /
        `</final>`), surface it as the exact tag so the splitter can route the
        following content; other special tokens map to `''` (EOS-like → ``\\n``)."""
        nonlocal prev_decoded_text
        if tid in special_ids:
            try:
                tok = str(_tokenizer.convert_ids_to_tokens(tid))
            except Exception:
                tok = ""
            if tok in _COT_LITERAL_TAGS:
                return tok
            tok_n = tok.strip().lower().replace(" ", "")
            if tok_n in {"<s>", "</s>", "<eos>", "<|eot_id|>", "<|endoftext|>", "<|im_end|>"}:
                return "\n"
            return ""
        generated_ids.append(tid)
        try:
            full_text = _tokenizer.decode(
                generated_ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if full_text.startswith(prev_decoded_text):
                chunk = full_text[len(prev_decoded_text):]
            else:
                chunk = _tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            prev_decoded_text = full_text
        except Exception:
            chunk = f"<{tid}>"
        return chunk

    def _route(chunk: str):
        """Yield events for one decoded chunk; returns True if splitter signaled stop."""
        stopped = False
        for kind, payload in splitter.feed(chunk):
            if kind == "reasoning":
                reasoning_acc.append(payload)
                yield {"type": "reasoning", "markdown": "".join(reasoning_acc)}
            elif kind == "final":
                el = time.perf_counter() - t_dec0
                yield {
                    "type": "token",
                    "text": payload,
                    "n": n_out,
                    "tok_s": round(n_out / max(el, 1e-9), 1),
                }
            elif kind == "stop":
                stopped = True
        if stopped:
            yield {"__stop__": True}

    tid = int(last.item())
    n_out += 1
    ttft_ms = (time.perf_counter() - t_dec0) * 1000
    stop_after = False
    for ev in _route(_decode_chunk(tid)):
        if ev.get("__stop__"):
            stop_after = True
        else:
            yield ev

    if stop_after or (_stop_ids and tid in _stop_ids):
        for kind, payload in splitter.flush():
            if kind == "reasoning":
                reasoning_acc.append(payload)
                yield {"type": "reasoning", "markdown": "".join(reasoning_acc)}
            else:
                el = time.perf_counter() - t_dec0
                yield {"type": "token", "text": payload, "n": n_out, "tok_s": round(n_out / max(el, 1e-9), 1)}
        elapsed = time.perf_counter() - t_dec0
        yield _done_event(n_out, elapsed, ttft_ms, prefill_ms)
        return

    x_one = last.reshape(1, 1)
    for _ in range(max_tokens - 1):
        seq_pos = mx.array(pos, dtype=mx.int32)
        logits_d, caches = decode_fn(x_one, caches, seq_pos)
        row = logits_d[0, -1, :]
        last = sample_decode_token(row, token_counts, sample_args)
        token_counts[last] = token_counts[last] + 1
        mx.eval(last, caches, token_counts)

        tid = int(last.item())
        n_out += 1
        for ev in _route(_decode_chunk(tid)):
            if ev.get("__stop__"):
                stop_after = True
            else:
                yield ev

        if stop_after or (_stop_ids and tid in _stop_ids):
            break
        x_one = last.reshape(1, 1)
        pos += 1

    for kind, payload in splitter.flush():
        if kind == "reasoning":
            reasoning_acc.append(payload)
            yield {"type": "reasoning", "markdown": "".join(reasoning_acc)}
        else:
            el = time.perf_counter() - t_dec0
            yield {"type": "token", "text": payload, "n": n_out, "tok_s": round(n_out / max(el, 1e-9), 1)}
    elapsed = time.perf_counter() - t_dec0
    yield _done_event(n_out, elapsed, ttft_ms, prefill_ms)


def _done_event(n_out, elapsed, ttft_ms, prefill_ms):
    return {
        "type": "done",
        "total_tokens": n_out,
        "total_ms": round(elapsed * 1000, 2),
        "tok_s": round(n_out / max(elapsed, 1e-9), 1),
        "ttft_ms": round(ttft_ms or 0, 2),
        "prefill_ms": round(prefill_ms, 2),
    }


# ---------------------------------------------------------------------------
# FastAPI + WebSocket
# ---------------------------------------------------------------------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Mamba3-XR Chat Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_UI_DIR = Path(__file__).parent / "ui"
if _UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")

_load_timings: dict[str, Any] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat UI HTML; rewrite JS/CSS refs to `?v=<mtime>` so any
    edit busts the browser cache. Also stamps `no-store` so the HTML itself
    is never cached during dev iteration."""
    ui_dir = Path(__file__).parent / "ui"
    html = (ui_dir / "chat_demo.html").read_text(encoding="utf-8")
    try:
        js_v = int((ui_dir / "chat_demo.js").stat().st_mtime)
    except OSError:
        js_v = 0
    try:
        css_v = int((ui_dir / "chat_demo.css").stat().st_mtime)
    except OSError:
        css_v = 0
    html = html.replace("/static/chat_demo.js", f"/ui/chat_demo.js?v={js_v}")
    html = html.replace("/static/chat_demo.css", f"/ui/chat_demo.css?v={css_v}")
    resp = HTMLResponse(html)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _serve_no_store(path: Path, media_type: str):
    """Read a UI asset and return it with aggressive no-store headers so
    edits to `chat_demo.js` / `chat_demo.css` always reach the browser."""
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    body = path.read_bytes()
    from fastapi.responses import Response
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/ui/chat_demo.js")
async def ui_js():
    return _serve_no_store(Path(__file__).parent / "ui" / "chat_demo.js", "application/javascript")


@app.get("/ui/chat_demo.css")
async def ui_css():
    return _serve_no_store(Path(__file__).parent / "ui" / "chat_demo.css", "text/css")


@app.get("/api/status")
async def status():
    if _MOCK_MODE:
        data = _read_ui_config().get("status") or {}
        return JSONResponse(data)
    return JSONResponse({
        "ready": _model_ready,
        "load_timings": _load_timings,
        "config": {
            "d_model": _config.d_model if _config else 0,
            "num_layers": _config.num_layers if _config else 0,
            "kmoe_num_experts": _config.kmoe_num_experts if _config else 0,
            "kmoe_top_k": _config.kmoe_top_k if _config else 0,
            "vocab_size": _vocab_size,
            "quantize": _args.quantize if _args else 0,
            "dtype": _args.dtype if _args else "",
            "max_new_tokens": _args.max_new_tokens if _args else 0,
        },
    })


@app.get("/api/demo-config")
async def demo_config():
    data = _read_ui_config()
    flat: list[dict[str, Any]] = []
    for cat in data.get("categories") or []:
        ck = cat.get("key") or ""
        title = cat.get("title") or ""
        for ex in cat.get("examples") or []:
            flat.append({
                "example_id": ex.get("id"),
                "label": f"{title} · {ex.get('subcategory', '')}",
                "prompt": ex.get("user", ""),
                "category_key": ck,
            })
    legacy = data.get("welcome_examples") or []
    return JSONResponse({
        "mock": _MOCK_MODE,
        "system_prompt_markdown": data.get("system_prompt_markdown", ""),
        "category_system_prompts": _category_system_prompts_map(data),
        "style_constraints": data.get("style_constraints", {}),
        "tool_registry": data.get("tool_registry", []),
        "categories": data.get("categories", []),
        "examples": flat if flat else legacy,
        "sampling_defaults": _sampling_defaults_dict(),
    })


def _sampling_defaults_dict() -> dict[str, Any]:
    """CLI defaults exposed to the UI's Sampling drawer tab. Reset button uses these."""
    if _args is None:
        return {
            "temperature": 0.3,
            "top_k": 40,
            "top_p": 0.9,
            "min_p": 0.05,
            "repetition_penalty": 1.0,
        }
    return {
        "temperature": float(getattr(_args, "temp", 0.3)),
        "top_k": int(getattr(_args, "top_k", 40)),
        "top_p": float(getattr(_args, "top_p", 0.9)),
        "min_p": float(getattr(_args, "min_p", 0.05)),
        "repetition_penalty": float(getattr(_args, "rep_pen", 1.0)),
    }


def _apply_sampling_override(base_args: Any, sampling: dict[str, Any] | None) -> Any:
    """Return a shallow copy of `_args` with sampling fields overridden for one
    chat call. Unrecognized keys / out-of-range values are silently clamped so
    a misconfigured UI never produces invalid sampling state."""
    import copy as _copy

    if not isinstance(sampling, dict) or not sampling:
        return base_args
    a = _copy.copy(base_args)
    if "temperature" in sampling:
        try:
            a.temp = max(0.0, min(5.0, float(sampling["temperature"])))
        except (TypeError, ValueError):
            pass
    if "top_k" in sampling:
        try:
            a.top_k = max(0, min(2048, int(sampling["top_k"])))
        except (TypeError, ValueError):
            pass
    if "top_p" in sampling:
        try:
            a.top_p = max(0.0, min(1.0, float(sampling["top_p"])))
        except (TypeError, ValueError):
            pass
    if "min_p" in sampling:
        try:
            a.min_p = max(0.0, min(1.0, float(sampling["min_p"])))
        except (TypeError, ValueError):
            pass
    if "repetition_penalty" in sampling:
        try:
            a.rep_pen = max(1.0, min(2.0, float(sampling["repetition_penalty"])))
        except (TypeError, ValueError):
            pass
    return a


def _flatten_mock_examples(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cat in data.get("categories") or []:
        ck = str(cat.get("key") or "")
        title = str(cat.get("title") or "")
        for ex in cat.get("examples") or []:
            if not isinstance(ex, dict):
                continue
            row = dict(ex)
            row["category_key"] = ck
            row["category_title"] = title
            out.append(row)
    return out


def _find_mock_example(data: dict[str, Any], example_id: str | None, prompt: str) -> dict[str, Any] | None:
    examples = _flatten_mock_examples(data)
    if example_id:
        eid = str(example_id).strip()
        for ex in examples:
            if ex.get("id") == eid:
                return ex
    pn = " ".join(prompt.split())
    for ex in examples:
        u = str(ex.get("user", "")).strip()
        if u and pn == " ".join(u.split()):
            return ex
    return None


_TOKEN_RE = __import__("re").compile(r"\S+|\s+")


def _chunk_stream_text(text: str, max_chunk: int = 40) -> list[str]:
    """Split into small streaming chunks while preserving original whitespace
    (especially `\\n`, which the markdown renderer needs to detect headings,
    tables, lists, etc. at the end of the stream)."""
    if not text:
        return []
    parts = _TOKEN_RE.findall(text)
    chunks: list[str] = []
    buf = ""
    for p in parts:
        if buf and len(buf) + len(p) > max_chunk:
            chunks.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        chunks.append(buf)
    return chunks if chunks else [text]


async def _mock_stream_category_system_prompt(ws: WebSocket, category_key: str) -> None:
    """Mock-only: stream the short per-category string aligned with SFT export."""
    data = _read_ui_config()
    stream_cfg = data.get("mock_stream") or {}
    jitter = float(stream_cfg.get("chunk_delay_jitter", 0.12))
    intro_tok_s = float(stream_cfg.get("intro_tok_s", 60.0))
    pause_intro = float(stream_cfg.get("pause_after_intro_ms", 120)) / 1000.0

    ck = str(category_key or "").strip()
    prompts = _category_system_prompts_map(data)
    text = prompts.get(ck, "").strip()
    if not text:
        await ws.send_json({
            "type": "error",
            "message": f"No training system prompt for category {ck!r} (see cot_dataset/category_system_prompts.py).",
        })
        return

    cat_title = ck
    for c in data.get("categories") or []:
        if str(c.get("key") or "") == ck:
            cat_title = str(c.get("title") or ck)
            break

    await ws.send_json({
        "type": "intro_start",
        "category_key": ck,
        "category_title": cat_title,
    })
    intro_n = 0
    intro_per = 1.0 / max(intro_tok_s, 1.0)
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    for ch in _chunk_stream_text(text, 56):
        j = 1.0 + random.uniform(-jitter, jitter)
        await asyncio.sleep(max(0.003, intro_per * j))
        intro_n += 1
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000.0
        el = time.perf_counter() - t0
        await ws.send_json({
            "type": "token",
            "text": ch,
            "n": intro_n,
            "tok_s": round(intro_n / max(el, 1e-9), 1),
        })
    elapsed = time.perf_counter() - t0
    await asyncio.sleep(max(0.0, pause_intro))
    await ws.send_json({
        "type": "done",
        "total_tokens": intro_n,
        "total_ms": round(elapsed * 1000.0, 2),
        "tok_s": round(intro_n / max(elapsed, 1e-9), 1),
        "ttft_ms": round(ttft_ms or 0.0, 2),
        "prefill_ms": 0.0,
        "play_system_only": True,
    })


async def _mock_chat_stream(
    ws: WebSocket,
    prompt: str,
    max_tokens: int,
    turns: int,
    example_id: str | None = None,
) -> str:
    """Simulate meta / reasoning / tokens / optional tool / done (mock only)."""
    data = _read_ui_config()
    stream_cfg = data.get("mock_stream") or {}
    target_tok_s = float(stream_cfg.get("target_tok_s", 40.0))
    jitter = float(stream_cfg.get("chunk_delay_jitter", 0.12))
    if target_tok_s <= 0:
        target_tok_s = 40.0
    base_per_chunk = 1.0 / target_tok_s
    if "chunk_delay_ms" in stream_cfg and "target_tok_s" not in stream_cfg:
        base_per_chunk = max(0.002, float(stream_cfg.get("chunk_delay_ms", 25)) / 1000.0)

    pause_meta = float(stream_cfg.get("pause_after_meta_ms", 160)) / 1000.0
    pause_reason = float(stream_cfg.get("pause_after_reasoning_ms", 400)) / 1000.0
    pause_tool = float(stream_cfg.get("pause_after_tool_ms", 450)) / 1000.0
    pause_split = float(stream_cfg.get("pause_after_assistant_split_ms", 280)) / 1000.0
    pause_fallback = float(stream_cfg.get("pause_before_fallback_ms", 220)) / 1000.0

    ex = _find_mock_example(data, example_id, prompt)

    async def sleep_chunk_delay() -> None:
        j = 1.0 + random.uniform(-jitter, jitter)
        await asyncio.sleep(max(0.003, base_per_chunk * j))

    prefill_ms = round(6.0 + (len(prompt) % 11) * 0.45, 2)
    prompt_tokens = min(16 + len(prompt) // 2, 600)
    meta: dict[str, Any] = {
        "type": "meta",
        "prefill_ms": prefill_ms,
        "prompt_tokens": prompt_tokens,
        "turns": turns,
    }
    if ex:
        meta["category"] = ex.get("category_key")
        meta["example_id"] = ex.get("id")
        meta["subcategory"] = ex.get("subcategory")
    await ws.send_json(meta)
    await asyncio.sleep(max(0.0, pause_meta))

    n_out = 0
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    assembled: list[str] = []

    async def emit_chunk(ch: str) -> None:
        nonlocal n_out, ttft_ms
        if n_out >= max_tokens:
            return
        await sleep_chunk_delay()
        n_out += 1
        assembled.append(ch)
        elapsed = time.perf_counter() - t0
        if ttft_ms is None:
            ttft_ms = elapsed * 1000.0
        tok_s = n_out / max(elapsed, 1e-9)
        await ws.send_json({
            "type": "token",
            "text": ch,
            "n": n_out,
            "tok_s": round(tok_s, 1),
        })

    if not ex:
        await asyncio.sleep(max(0.0, pause_fallback))
        fallback = list(stream_cfg.get("fallback_chunks") or stream_cfg.get("chunks") or ["[mock] ok."])
        echo = f"(No example match — {len(prompt)} chars.) "
        for ch in _chunk_stream_text(echo + "".join(fallback), 48):
            if n_out >= max_tokens:
                break
            await emit_chunk(ch)
        elapsed = time.perf_counter() - t0
        await ws.send_json({
            "type": "done",
            "total_tokens": n_out,
            "total_ms": round(elapsed * 1000.0, 2),
            "tok_s": round(n_out / max(elapsed, 1e-9), 1),
            "ttft_ms": round(ttft_ms or 0.0, 2),
            "prefill_ms": prefill_ms,
        })
        return "".join(assembled)

    cot = str(ex.get("cot_markdown") or ex.get("cot") or "").strip()
    if cot:
        await ws.send_json({"type": "reasoning", "markdown": cot})
        await asyncio.sleep(max(0.0, pause_reason))

    tf = ex.get("tool_flow") if isinstance(ex.get("tool_flow"), dict) else {}
    trig = (tf.get("trigger_output") or "").strip() if tf else ""
    sysres = (tf.get("system_result") or "").strip() if tf else ""
    final_md = (tf.get("final_markdown") or "").strip() if tf else ""
    tool_name = str(tf.get("tool_name") or "") if tf else ""

    async def emit_md(md: str, chunk_size: int = 44) -> None:
        for ch in _chunk_stream_text(md, chunk_size):
            if n_out >= max_tokens:
                return
            await emit_chunk(ch)

    if trig and not final_md:
        await emit_md(trig, chunk_size=64)
    elif final_md and sysres and not trig:
        await ws.send_json({
            "type": "tool_action",
            "call": f"(host injected {tool_name})" if tool_name else "(host injected)",
            "tool_name": tool_name,
            "system_result": sysres,
            "phase": "response",
        })
        await asyncio.sleep(max(0.0, pause_tool))
        await emit_md(final_md)
    elif trig and final_md and sysres:
        await emit_md(trig, chunk_size=64)
        await ws.send_json({
            "type": "tool_action",
            "call": trig,
            "tool_name": tool_name,
            "system_result": sysres,
            "phase": "trigger",
        })
        await asyncio.sleep(max(0.0, pause_tool))
        await ws.send_json({"type": "assistant_split"})
        await asyncio.sleep(max(0.0, pause_split))
        await emit_md(final_md)
    else:
        body = str(ex.get("assistant_markdown") or ex.get("output") or "").strip()
        await emit_md(body)

    elapsed = time.perf_counter() - t0
    await ws.send_json({
        "type": "done",
        "total_tokens": n_out,
        "total_ms": round(elapsed * 1000.0, 2),
        "tok_s": round(n_out / max(elapsed, 1e-9), 1),
        "ttft_ms": round(ttft_ms or 0.0, 2),
        "prefill_ms": prefill_ms,
    })
    return "".join(assembled)


@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "connected", "ready": _model_ready, "mock": _MOCK_MODE})

    # === Auto-prime the default category system prefix in real mode. =========
    # Runs in the background while the user is reading the welcome screen, so
    # the first chat that uses the default system prompt already has its
    # system-block KV cache ready and only has to prefill the user turn.
    if _model_ready and not _MOCK_MODE:
        try:
            ui_data = _read_ui_config()
            cat_map = _category_system_prompts_map(ui_data)
            default_sys = (cat_map.get("daily_conversation") or "").strip()
            if default_sys and default_sys not in _primed_prefix:
                async def _bg_autoprime(sys_text=default_sys):
                    try:
                        async with _infer_lock:
                            await asyncio.get_event_loop().run_in_executor(
                                None, _prime_system_prefix_sync, sys_text,
                            )
                    except Exception as exc:
                        print(f"[chat_demo] WARN: auto-prime failed: {exc}")

                asyncio.create_task(_bg_autoprime())
        except Exception as exc:
            print(f"[chat_demo] WARN: skip auto-prime: {exc}")

    conversation: list[dict] = []
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            action = msg.get("action")

            if action == "ping":
                await ws.send_json({"type": "pong", "ready": _model_ready})
                continue

            if action == "clear":
                conversation.clear()
                await ws.send_json({"type": "cleared"})
                continue

            if action == "play_system_prompt":
                ck = str(msg.get("category_key", "")).strip()
                if not ck:
                    await ws.send_json({"type": "error", "message": "category_key is required"})
                    continue
                if _MOCK_MODE:
                    await _mock_stream_category_system_prompt(ws, ck)
                    continue
                if not _model_ready:
                    await ws.send_json({"type": "error", "message": "Model not loaded"})
                    continue

                # === Real mode: prime the system-prefix KV cache + stream the
                # short SFT/export prompt text so the UI shows the same intro.
                ui_data = _read_ui_config()
                cat_map = _category_system_prompts_map(ui_data)
                sys_text = (cat_map.get(ck) or "").strip()
                if not sys_text:
                    await ws.send_json({"type": "error", "message": f"No system prompt for category {ck!r}"})
                    continue
                cat_title = ck
                for c in ui_data.get("categories") or []:
                    if str(c.get("key") or "") == ck:
                        cat_title = str(c.get("title") or ck)
                        break

                async def _prime_task():
                    async with _infer_lock:
                        return await loop.run_in_executor(None, _prime_system_prefix_sync, sys_text)

                prime_future = asyncio.create_task(_prime_task())
                await ws.send_json({
                    "type": "intro_start",
                    "category_key": ck,
                    "category_title": cat_title,
                })
                # Stream the system text concurrently; the heavy prefill runs
                # in the executor and usually finishes well before the text.
                stream_cfg = ui_data.get("mock_stream") or {}
                intro_tok_s = float(stream_cfg.get("intro_tok_s", 64.0)) or 64.0
                jitter = float(stream_cfg.get("chunk_delay_jitter", 0.12))
                intro_per = 1.0 / max(intro_tok_s, 1.0)
                t0 = time.perf_counter()
                ttft_ms: float | None = None
                n_out = 0
                for ch in _chunk_stream_text(sys_text, 56):
                    j = 1.0 + random.uniform(-jitter, jitter)
                    await asyncio.sleep(max(0.003, intro_per * j))
                    n_out += 1
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    el = time.perf_counter() - t0
                    await ws.send_json({
                        "type": "token",
                        "text": ch,
                        "n": n_out,
                        "tok_s": round(n_out / max(el, 1e-9), 1),
                    })
                try:
                    prime_result = await prime_future
                except Exception as exc:
                    prime_result = {"ok": False, "reason": str(exc), "prefill_ms": 0.0, "prompt_tokens": 0}
                elapsed = time.perf_counter() - t0
                done_payload: dict[str, Any] = {
                    "type": "done",
                    "total_tokens": n_out,
                    "total_ms": round(elapsed * 1000.0, 2),
                    "tok_s": round(n_out / max(elapsed, 1e-9), 1),
                    "ttft_ms": round(ttft_ms or 0.0, 2),
                    "prefill_ms": float(prime_result.get("prefill_ms", 0.0)),
                    "play_system_only": True,
                    "primed_ok": bool(prime_result.get("ok")),
                    "cached_prefix_tokens": int(prime_result.get("prompt_tokens", 0)),
                }
                if not prime_result.get("ok"):
                    done_payload["primed_reason"] = str(prime_result.get("reason", ""))
                await ws.send_json(done_payload)
                continue

            if action == "chat":
                prompt = msg.get("prompt", "").strip()
                max_tokens = min(msg.get("max_tokens", 512), _args.max_new_tokens if _args else 2048)
                if not prompt:
                    await ws.send_json({"type": "error", "message": "Empty prompt"})
                    continue
                if not _model_ready:
                    await ws.send_json({"type": "error", "message": "Model not loaded"})
                    continue

                if _MOCK_MODE:
                    conversation.append({"role": "user", "content": prompt})
                    turns = len([m for m in conversation if m["role"] == "user"])
                    ex_id = msg.get("example_id")
                    assistant_text = await _mock_chat_stream(
                        ws, prompt, max_tokens, turns, example_id=ex_id,
                    )
                    conversation.append({"role": "assistant", "content": assistant_text})
                    continue

                # === Real inference: align with production SFT-CoT script. ===
                category_key = str(msg.get("category_key") or "").strip()
                ui_data = _read_ui_config()
                cat_map = _category_system_prompts_map(ui_data)
                fallback_key = (
                    "daily_conversation"
                    if "daily_conversation" in cat_map
                    else (next(iter(cat_map.keys()), ""))
                )
                effective_key = category_key if category_key in cat_map else fallback_key
                system_prompt = cat_map.get(effective_key, "") if effective_key else ""
                reasoning_flag = msg.get("reasoning")
                if reasoning_flag is None:
                    reasoning_flag = bool(getattr(_args, "reasoning", True))
                else:
                    reasoning_flag = bool(reasoning_flag)
                sampling_override = msg.get("sampling") if isinstance(msg.get("sampling"), dict) else None
                args_for_call = _apply_sampling_override(_args, sampling_override)

                conversation.append({"role": "user", "content": prompt})
                history_snapshot = list(conversation)

                def _send_now(ev: dict):
                    """Fire ws.send_json from generator thread — blocks until sent."""
                    fut = asyncio.run_coroutine_threadsafe(ws.send_json(ev), loop)
                    fut.result()

                def run_gen() -> str:
                    text_parts: list[str] = []
                    try:
                        for ev in _stream_generate(
                            history_snapshot,
                            max_tokens=max_tokens,
                            system_prompt=system_prompt,
                            reasoning=reasoning_flag,
                            args_for_call=args_for_call,
                        ):
                            _send_now(ev)
                            if ev.get("type") == "token":
                                text_parts.append(ev.get("text", ""))
                    except Exception as exc:
                        try:
                            _send_now({"type": "error", "message": str(exc)})
                        except Exception:
                            pass
                    return "".join(text_parts)

                async with _infer_lock:
                    assistant_text = await loop.run_in_executor(None, run_gen)

                conversation.append({"role": "assistant", "content": assistant_text})
                continue

            await ws.send_json({"type": "error", "message": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mamba3-XR Chat Demo Server")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument(
        "--mock",
        action="store_true",
        help="Skip MLX load; serve mock /api/status and WebSocket stream from ui/mock_config.json",
    )
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--npz-cache", type=str, default="")
    p.add_argument("--force-pt", action="store_true")
    p.add_argument("--tokenizer", type=str, default=os.path.join(_INF_DIR, "tokenizer"))
    p.add_argument("--inference-type", type=str, default="safe",
                    choices=("throughput", "safe", "eager", "sequential-ssm", "custom"))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=32007)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--kv-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--quantize", type=int, choices=[0, 4, 8], default=4)
    p.add_argument("--router-temp", type=float, default=0.5)
    p.add_argument("--temp", type=float, default=0.3)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_p", type=float, default=0.05)
    p.add_argument("--rep_pen", type=float, default=1.0)
    p.add_argument("--pres_pen", type=float, default=0.0)
    p.add_argument("--freq_pen", type=float, default=0.0)
    p.add_argument("--no-fast-sample", dest="fast_sample", action="store_false", default=False)
    p.add_argument("--fast-sample", dest="fast_sample", action="store_true")
    p.add_argument("--tucker-einsum-fuse", dest="tucker_einsum_fuse", action="store_true", default=True)
    p.add_argument("--no-tucker-einsum-fuse", dest="tucker_einsum_fuse", action="store_false")
    p.add_argument("--materialize-caches", action="store_true", default=True)
    p.add_argument("--no-compile-prefill", action="store_true", default=False)
    p.add_argument("--eager-decode", action="store_true", default=False)
    p.add_argument("--full-decode-compile", action="store_true", default=True)
    p.add_argument(
        "--reasoning",
        dest="reasoning",
        action="store_true",
        default=True,
        help="Inject `<think>\\n` after the assistant marker so the SFT-CoT "
        "model continues inside its reasoning distribution (default: on).",
    )
    p.add_argument(
        "--no-reasoning",
        dest="reasoning",
        action="store_false",
        help="Disable the `<think>` prompt injection (model emits direct answer).",
    )
    args = p.parse_args()

    _apply_inference_type(args)
    args.no_materialize_caches = not bool(getattr(args, "materialize_caches", True))
    args.no_penalties = True
    args.fused_sample_metal = False
    args.fused_sample_metal_v2 = False
    args.interactive = False
    args.lookahead_router = False
    args.no_eos_stop = False
    args.stop_on_eos = True
    return args


def main():
    args = parse_args()
    global _load_timings, _model_ready, _config, _vocab_size, _args, _MOCK_MODE

    print(f"\n{'='*60}")
    print(f"  Mamba3-XR Chat Demo (WebSocket)")
    print(f"{'='*60}\n")

    _args = args
    _MOCK_MODE = bool(args.mock)

    if _MOCK_MODE:
        print("  MOCK MODE — no MLX weights (edit inference/ui/mock_config.json)")
        data = _read_ui_config()
        st = data.get("status") or {}
        _load_timings = st.get("load_timings") or {"total_ms": 0.0}
        cfg = dict(st.get("config") or {})
        defaults = {
            "d_model": 768,
            "num_layers": 6,
            "kmoe_num_experts": 8,
            "kmoe_top_k": 2,
            "vocab_size": 32007,
            "quantize": 8,
            "dtype": "bf16",
            "max_new_tokens": 2048,
        }
        defaults.update(cfg)
        _config = SimpleNamespace(**defaults)
        _vocab_size = int(getattr(_config, "vocab_size", 32007))
        _model_ready = True
    else:
        print("  Loading model...")
        _load_timings = _load_model(args)
        print(f"\n  Model loaded in {_load_timings['total_ms']:.0f} ms")
        print(f"  Checkpoint: {_load_timings.get('checkpoint', 'N/A')}")
        print(f"  Quantize: {args.quantize}-bit | dtype: {args.dtype}")

    print(f"\n  Server: http://localhost:{args.port}")
    print(f"  WebSocket: ws://localhost:{args.port}/ws")
    print(f"{'='*60}\n")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
