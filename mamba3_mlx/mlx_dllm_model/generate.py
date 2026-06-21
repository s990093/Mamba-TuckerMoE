"""dLLM change ④ — iterative unmasking generation (batch = 1).

A dLLM does not decode left-to-right.  The response region starts as all
``[MASK]`` and is filled over ``T`` forward passes; each pass re-reads the
whole (prompt + current-response) sequence (bidirectional attention) and the
most-confident still-masked positions are committed, following a cosine
schedule (DLLM_MLX_PORT.md §④).

This module holds the schedule + selection logic as one ``forward``-agnostic
core so the eager reference path and the compiled high-performance path
(``static_dllm.StaticDLLM``) share identical numerics — only the ``forward``
callable differs.

Scheduling note
---------------
The number of positions committed at step ``s`` depends only on (G, T, s) —
NOT on the data — so the whole ``need`` schedule is precomputed in Python.
Each step therefore has fully static shapes and needs no host sync for
control flow; only the final token ids are read back.
"""

from __future__ import annotations

import math
import time
from typing import Callable, NamedTuple

import mlx.core as mx

from .config import MASK_ID, STOP_IDS, DiffusionGenConfig
from .samplers import (StableConfidentStop, entropy_bound_select,
                       linear_temperature, token_entropy)


class DLLMResult(NamedTuple):
    prompt_ids: list[int]
    response_ids: list[int]      # length G, in committed order of the schedule
    steps: int
    n_prompt: int
    gen_len: int
    elapsed: float               # decode seconds (compile/warmup excluded)
    compile_s: float             # one-time trace+compile (0.0 for eager path)
    tokens_per_s: float          # gen_len / elapsed
    forwards_per_s: float        # steps  / elapsed
    steps_used: int = 0          # forwards actually run (≤ steps with adaptive stop)


def cosine_fill_schedule(gen_len: int, steps: int) -> list[int]:
    """Per-step count of newly-committed positions (sums to gen_len).

    keep_s = floor(G·cos(π/2 · s/T)) = MASKs remaining after step s, so the
    cumulative filled count is G − keep_s and the per-step delta is the
    difference of consecutive cumulative counts.
    """
    prev_filled = 0
    out: list[int] = []
    for s in range(1, steps + 1):
        keep = int(math.floor(gen_len * math.cos(math.pi / 2 * s / steps)))
        filled = gen_len - keep
        out.append(max(0, filled - prev_filled))
        prev_filled = filled
    # cosine hits exactly 0 at s=T, so prev_filled == gen_len already; this is
    # a guard for rounding only.
    if prev_filled < gen_len:
        out[-1] += gen_len - prev_filled
    return out


def ban_mask(logits, mask_id: int = MASK_ID):
    """Set the [MASK] logit to -inf — the model must never *emit* a MASK token
    (it is an input-only placeholder; LLaDA-style decode)."""
    V = logits.shape[-1]
    if 0 <= mask_id < V:
        ban = mx.zeros((V,), dtype=logits.dtype)
        ban[mask_id] = -mx.inf
        return logits + ban
    return logits


def _confidence_and_pred(logits, temperature: float, key, mask_id: int = MASK_ID):
    """(conf, pred, key) for a (G, V) logits block.

    Greedy (temperature<=0): pred=argmax, conf=max softmax prob.
    Sampled: pred~categorical(logits/temp), conf=prob assigned to pred.
    """
    logits = ban_mask(logits, mask_id)
    if temperature <= 0.0:
        probs = mx.softmax(logits, axis=-1)
        conf = mx.max(probs, axis=-1)
        pred = mx.argmax(logits, axis=-1).astype(mx.int32)
        return conf, pred, key
    key, sub = mx.random.split(key)
    z = logits / temperature
    probs = mx.softmax(z, axis=-1)
    pred = mx.random.categorical(z, key=sub).astype(mx.int32)
    conf = mx.take_along_axis(probs, pred[:, None], axis=-1)[:, 0]
    return conf, pred, key


def iterative_unmask(
    model,
    prompt_ids: list[int],
    gen_len: int,
    *,
    steps: int = 16,
    temperature: float = 0.0,
    seed: int = 0,
    mask_id: int = MASK_ID,
    forward: Callable | None = None,
    on_commit: Callable[[int, list[int]], None] | None = None,
) -> DLLMResult:
    """Run §④ iterative unmasking for one prompt (batch = 1).

    ``forward(x) -> logits (1, L, V)`` defaults to ``model.__call__``.  The
    high-performance path passes a compiled forward with the same signature.
    """
    fwd = forward if forward is not None else model
    P = len(prompt_ids)
    G = int(gen_len)
    schedule = cosine_fill_schedule(G, steps)

    x = mx.array(list(prompt_ids) + [mask_id] * G, dtype=mx.int32)[None]  # (1, P+G)
    filled = mx.zeros((G,), dtype=mx.bool_)
    key = mx.random.key(seed)
    mx.eval(x, filled)

    t0 = time.perf_counter()
    for s, need in enumerate(schedule, start=1):
        logits = fwd(x)[0, P:]                                   # (G, V) f32
        conf, pred, key = _confidence_and_pred(logits, temperature, key, mask_id)
        if need > 0:
            # restrict to not-yet-committed positions, take the top-`need`
            cand = mx.where(filled, mx.array(-1.0, dtype=conf.dtype), conf)
            order = mx.argsort(-cand)[:need]                     # static shape
            sel = mx.zeros((G,), dtype=mx.int32)
            sel = sel.at[order].add(mx.ones((need,), dtype=mx.int32))
            newly = sel > 0
            resp = x[0, P:]
            new_resp = mx.where(newly, pred.astype(x.dtype), resp)
            x = mx.concatenate([x[:, :P], new_resp[None]], axis=1)
            filled = filled | newly
        mx.eval(x, filled)
        if on_commit is not None:
            on_commit(s, x[0, P:].tolist())
    elapsed = time.perf_counter() - t0

    response_ids = x[0, P:].tolist()
    return DLLMResult(
        prompt_ids=list(prompt_ids),
        response_ids=[int(t) for t in response_ids],
        steps=steps,
        n_prompt=P,
        gen_len=G,
        elapsed=elapsed,
        compile_s=0.0,
        tokens_per_s=(G / elapsed if elapsed > 0 else 0.0),
        forwards_per_s=(steps / elapsed if elapsed > 0 else 0.0),
    )


def diffusion_generate(
    model,
    prompt_ids: list[int],
    dgcfg: DiffusionGenConfig,
    *,
    forward: Callable | None = None,
    mask_id: int = MASK_ID,
    on_step: Callable[[int, list[int]], None] | None = None,
) -> DLLMResult:
    """Optimized absorbing-[MASK] diffusion (batch = 1) with the DiffusionGemma
    inference tricks: entropy-bound adaptive acceptance, linear temperature
    schedule, optional self-conditioning, and optional stable+confident early
    stop.

    The canvas starts as ``prompt_ids + [MASK]*G`` and positions are filled in;
    non-committed positions stay ``[MASK]`` (absorbing — committed tokens are
    never re-noised).  ``forward(x, sc_logits) -> (1, L, V)`` defaults to the
    eager model; the static path passes a compiled forward.
    """
    if forward is None:
        def forward(x, sc):
            return model(x, self_conditioning_logits=sc, eval_boundary=True)
    use_sc = getattr(model, "self_conditioning", None) is not None

    P = len(prompt_ids)
    G = int(dgcfg.gen_len)
    T = int(dgcfg.steps)
    cosine = (dgcfg.sampler == "cosine")
    schedule = cosine_fill_schedule(G, T) if cosine else None
    stopper = (StableConfidentStop(dgcfg.stability_threshold, dgcfg.confidence_threshold)
               if dgcfg.adaptive_stop else None)

    x = mx.array(list(prompt_ids) + [mask_id] * G, dtype=mx.int32)[None]
    filled = mx.zeros((G,), dtype=mx.bool_)
    key = mx.random.key(dgcfg.seed)
    sc_logits = None
    mx.eval(x, filled)

    pred = None
    steps_used = 0
    t0 = time.perf_counter()
    for s in range(1, T + 1):
        temp = linear_temperature(T - s + 1, T, dgcfg.t_min, dgcfg.t_max)
        logits_full = forward(x, sc_logits if use_sc else None)
        steps_used += 1
        banned = ban_mask(logits_full[0, P:], mask_id)               # (G, V)

        if temp <= 0.0:
            probs = mx.softmax(banned, axis=-1)
            conf = mx.max(probs, axis=-1)
            pred = mx.argmax(banned, axis=-1).astype(mx.int32)
        else:
            key, sub = mx.random.split(key)
            z = banned / temp
            probs = mx.softmax(z, axis=-1)
            pred = mx.random.categorical(z, key=sub).astype(mx.int32)
            conf = mx.take_along_axis(probs, pred[:, None], axis=-1)[:, 0]

        # commit set: cosine (fixed count) or entropy-bound (adaptive)
        if cosine:
            need = schedule[s - 1]
            if need > 0:
                cand = mx.where(filled, mx.array(-1.0, dtype=conf.dtype), conf)
                order = mx.argsort(-cand)[:need]
                sel = mx.zeros((G,), dtype=mx.int32).at[order].add(
                    mx.ones((need,), dtype=mx.int32))
                commit = sel > 0
            else:
                commit = mx.zeros((G,), dtype=mx.bool_)
        else:
            commit = entropy_bound_select(token_entropy(banned), filled, dgcfg.entropy_bound)

        resp = x[0, P:]
        x = mx.concatenate([x[:, :P], mx.where(commit, pred.astype(x.dtype), resp)[None]], axis=1)
        filled = filled | commit
        if use_sc:
            sc_logits = logits_full
        mx.eval(x, filled)
        if on_step is not None:
            on_step(s, [int(t) for t in x[0, P:].tolist()])

        if bool(mx.all(filled).item()):
            break
        if stopper is not None:
            pred_canvas = mx.where(filled, x[0, P:], pred.astype(x.dtype))
            n_masked = float((~filled).sum().item())
            mean_ent = float((mx.where(filled, mx.array(0.0), token_entropy(banned)).sum()
                              / max(1.0, n_masked)).item())
            if stopper.update([int(t) for t in pred_canvas.tolist()], mean_ent):
                x = mx.concatenate([x[:, :P], pred_canvas[None]], axis=1)
                filled = mx.ones((G,), dtype=mx.bool_)
                mx.eval(x)
                break

    # safety: commit argmax for any still-masked positions (no [MASK] in output)
    if pred is not None and not bool(mx.all(filled).item()):
        resp = x[0, P:]
        x = mx.concatenate([x[:, :P], mx.where(filled, resp, pred.astype(x.dtype))[None]], axis=1)
        mx.eval(x)
    elapsed = time.perf_counter() - t0

    response_ids = [int(t) for t in x[0, P:].tolist()]
    return DLLMResult(
        prompt_ids=list(prompt_ids), response_ids=response_ids, steps=T,
        n_prompt=P, gen_len=G, elapsed=elapsed, compile_s=0.0,
        tokens_per_s=(G / elapsed if elapsed > 0 else 0.0),
        forwards_per_s=(steps_used / elapsed if elapsed > 0 else 0.0),
        steps_used=steps_used,
    )


def _commit_mask(banned, conf, filled, dgcfg, schedule, step):
    """Which still-masked canvas positions to commit this step (G,) bool."""
    G = banned.shape[0]
    if dgcfg.sampler == "cosine":
        need = schedule[step - 1]
        if need <= 0:
            return mx.zeros((G,), dtype=mx.bool_)
        cand = mx.where(filled, mx.array(-1.0, dtype=conf.dtype), conf)
        order = mx.argsort(-cand)[:need]
        sel = mx.zeros((G,), dtype=mx.int32).at[order].add(mx.ones((need,), dtype=mx.int32))
        return sel > 0
    return entropy_bound_select(token_entropy(banned), filled, dgcfg.entropy_bound)


def diffusion_generate_cached(
    model,
    prompt_ids: list[int],
    dgcfg: DiffusionGenConfig,
    *,
    denoise: Callable | None = None,
    mask_id: int = MASK_ID,
    on_step: Callable[[int, list[int]], None] | None = None,
) -> DLLMResult:
    """Prefix-cache absorbing-[MASK] diffusion (DiffusionGemma encoder/decoder
    split, batch = 1).

    The prompt is encoded ONCE into a per-layer cache (Mamba state + TF KV);
    each denoising step forwards only the G-token canvas (1 chunk when G≤64),
    not the whole P+G sequence — avoiding the per-step prompt re-read and the
    chunk-boundary cliff.  ``denoise(canvas_ids, sc) -> (G, V)`` defaults to the
    eager ``model.denoise``; the static path passes a compiled denoiser.

    Uses prefix-LM attention (the prompt does not attend to the canvas) — this
    differs from the full-bidirectional path by design and should be matched at
    training time for best quality.
    """
    P = len(prompt_ids)
    G = int(dgcfg.gen_len)
    T = int(dgcfg.steps)
    use_sc = getattr(model, "self_conditioning", None) is not None

    # Encode the prompt only when the caller did not supply a denoiser (the
    # static path builds its own cache + compiled denoiser and passes it in).
    encode_s = 0.0
    if denoise is None:
        t_enc = time.perf_counter()
        cache, _ = model.encode_prefix(mx.array(prompt_ids, dtype=mx.int32)[None])
        flat = [v for st in cache if isinstance(st, dict) for v in st.values() if v is not None]
        mx.eval(*flat)
        encode_s = time.perf_counter() - t_enc

        def denoise(canvas, sc):
            return model.denoise(canvas, cache, self_conditioning_logits=sc)[0]

    cosine = (dgcfg.sampler == "cosine")
    schedule = cosine_fill_schedule(G, T) if cosine else None
    stopper = (StableConfidentStop(dgcfg.stability_threshold, dgcfg.confidence_threshold)
               if dgcfg.adaptive_stop else None)

    canvas = mx.array([mask_id] * G, dtype=mx.int32)[None]            # (1, G)
    filled = mx.zeros((G,), dtype=mx.bool_)
    key = mx.random.key(dgcfg.seed)
    sc_logits = None
    mx.eval(canvas, filled)

    pred = None
    steps_used = 0
    t0 = time.perf_counter()
    for s in range(1, T + 1):
        temp = linear_temperature(T - s + 1, T, dgcfg.t_min, dgcfg.t_max)
        logits = denoise(canvas, sc_logits if use_sc else None)       # (G, V)
        steps_used += 1
        banned = ban_mask(logits, mask_id)

        if temp <= 0.0:
            probs = mx.softmax(banned, axis=-1)
            conf = mx.max(probs, axis=-1)
            pred = mx.argmax(banned, axis=-1).astype(mx.int32)
        else:
            key, sub = mx.random.split(key)
            z = banned / temp
            probs = mx.softmax(z, axis=-1)
            pred = mx.random.categorical(z, key=sub).astype(mx.int32)
            conf = mx.take_along_axis(probs, pred[:, None], axis=-1)[:, 0]

        commit = _commit_mask(banned, conf, filled, dgcfg, schedule, s)
        canvas = mx.where(commit, pred.astype(canvas.dtype), canvas[0])[None]
        filled = filled | commit
        if use_sc:
            sc_logits = logits[None]
        mx.eval(canvas, filled)
        if on_step is not None:
            on_step(s, [int(t) for t in canvas[0].tolist()])

        if bool(mx.all(filled).item()):
            break
        if stopper is not None:
            pred_canvas = mx.where(filled, canvas[0], pred.astype(canvas.dtype))
            n_masked = float((~filled).sum().item())
            mean_ent = float((mx.where(filled, mx.array(0.0), token_entropy(banned)).sum()
                              / max(1.0, n_masked)).item())
            if stopper.update([int(t) for t in pred_canvas.tolist()], mean_ent):
                canvas = pred_canvas[None]
                filled = mx.ones((G,), dtype=mx.bool_)
                mx.eval(canvas)
                break

    if pred is not None and not bool(mx.all(filled).item()):
        canvas = mx.where(filled, canvas[0], pred.astype(canvas.dtype))[None]
        mx.eval(canvas)
    elapsed = time.perf_counter() - t0

    response_ids = [int(t) for t in canvas[0].tolist()]
    return DLLMResult(
        prompt_ids=list(prompt_ids), response_ids=response_ids, steps=T,
        n_prompt=P, gen_len=G, elapsed=elapsed, compile_s=encode_s,
        tokens_per_s=(G / elapsed if elapsed > 0 else 0.0),
        forwards_per_s=(steps_used / elapsed if elapsed > 0 else 0.0),
        steps_used=steps_used,
    )


def trim_to_stop(ids: list[int], stop_ids=STOP_IDS) -> list[int]:
    """Cut the generated region at the first stop token (</final> / <|im_end|>)."""
    stop = set(int(s) for s in stop_ids)
    out: list[int] = []
    for t in ids:
        if int(t) in stop:
            out.append(int(t))
            break
        out.append(int(t))
    return out
