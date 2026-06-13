"""Static-graph Jacobi decoder — one compiled program per verify round.

Combines the two proven pieces of this repo:

  * speculative/jacobi.py     — greedy Jacobi rounds with n-gram / CoT drafts
                                (ARL ≈ 2–4 tokens accepted per round)
  * mlx_model/static_decode.py — TPU-style fixed-shape graphs (static KV via
                                ``mx.slice_update``, whole-step ``mx.compile``)

The decode bottleneck on Apple Silicon is GPU kernel-launch latency
(~1500 tiny kernels per forward).  A Jacobi verify forward of K tokens costs
roughly the SAME kernel count as a 1-token step, so accepted-run-length (ARL)
divides the per-token kernel cost.  The existing jacobi.py pays that win back
in Python dispatch (~2000 eager op dispatches per round); here the WHOLE round
is one compiled program:

    embed(K) → 24×Mamba per-position verify → 6×Transformer (static KV)
            → head → argmax → cumprod acceptance → m
            → gather per-position Mamba state at m-1 → write_pos += m

Acceptance and state extraction run on-GPU, so Python only reads back
(preds, m) once per round.  Static KV makes partial-accept rollback free for
Transformer layers: rejected positions are simply overwritten by the next
round and never attended (mask is derived from write_pos).

Math is line-for-line speculative/forward.py::mamba_verify_step, with TuckerMoE
calls routed through ``._forward`` so they inline into this graph instead of
nesting compiled programs.  No existing file is modified.

Greedy only (like jacobi.py): Jacobi acceptance requires a deterministic
verifier.  bf16 chunk-scan vs single-step rounding caveats from jacobi.py
apply here unchanged.

STATUS (measured 2026-06-12, checkpoint v6, M2 Pro)
---------------------------------------------------
The compiled round IS faster than the eager one in isolation (44.5 ms vs
50.4 ms at K=12, S=512).  But ARL on v6 is only 1.4–1.9 — even with an
oracle cache baked from the same prompt's greedy trajectory — because the
bf16 chunk-scan verifier drifts from the AR single-step path, so the
verifier disagrees with AR-derived drafts after a few tokens.  At ARL≈1.7
the ceiling is ~1.7/44.5 ms ≈ 38 tok/s, below plain static AR (60–71).
Jacobi on this checkpoint therefore does NOT pay off until either (a) the
verify/AR numerical drift is closed (fp32 carry in the verify scan), or
(b) a sampling-style lenient acceptance is used (see jacobi_sampling.py).
Keep this module as infrastructure; do not expect a speedup as-is.
"""
from __future__ import annotations

import time
from typing import Callable, NamedTuple, Optional

import mlx.core as mx

from ..mlx_model.mamba_block import Mamba3Block
from ..mlx_model.ops import apply_rope, silu, softplus
from .cot_cache import (
    CoTCachesArg,
    CoTPhaseTracker,
    infer_initial_phase,
    load_cot_caches,
)
from .drafts import build_hybrid_branches
from .forward import _scan_per_pos
from .jacobi import _build_guesses
from .ngram_cache import NGramCache


class StaticJacobiResult(NamedTuple):
    tokens: list[int]
    stop_reason: str            # "max_tokens" | "eos"
    n_prompt: int
    elapsed_prefill: float
    elapsed_decode: float
    prefill_tps: float
    decode_tps: float
    n_rounds: int
    n_accepted: int             # tokens emitted from verify rounds (excl. first token)
    arl: float                  # n_accepted / n_rounds
    n_ngram_hits: int
    K: int
    compile_s: float


def _mamba_verify_traced(blk: Mamba3Block, x, h_prev, prev_input_signal, angles_cum,
                         exact: bool = True):
    """forward.py::mamba_verify_step with state always concrete and TuckerMoE
    inlined via ``._forward`` (so the op stream is captured by the enclosing
    mx.compile instead of dispatching nested compiled programs).

    ``exact=True`` replaces the chunk-parallel scan with a K-step sequential
    recurrence that mirrors Mamba3Block._decode_impl op-for-op:

        h_l = av_l * h_{l-1} + u_ssm_l        (bf16, per step)
        ac_l = delta_l + ac_{l-1}             (float32, per step)
        av   = exp((dt*A).astype(f32)).astype(bf16)   (AR's cast order)

    The chunk scan is mathematically equivalent but rounds differently; that
    difference compounds ACROSS Jacobi rounds through the extracted h state,
    drifting the verifier away from the AR trajectory and collapsing ARL.
    Sequential h costs ~2 extra kernels × K × 24 blocks per round — worth it:
    the verifier then evolves state exactly like the AR decode it must agree
    with (residual disagreement only from shape-dependent matmul ulps).

    Returns (out, h_per_pos, prev_input_signal_per_pos, angles_cum_per_pos).
    """
    B_sz, L, _ = x.shape
    H, G, P, N, R = blk.H, blk.G, blk.P, blk.N, blk.R

    residual_mamba = x
    u = blk.norm_mamba(x)
    raw = blk.in_proj(u)
    z, x_prime, B_param, C_param, dt_p, A_p, lam = blk._split_inproj(raw, B_sz, L)

    x_prime_hp = x_prime.reshape(B_sz, L, H, P)
    dt = softplus(dt_p)
    A = -mx.exp(A_p)
    theta = mx.exp(blk.theta_log.astype(mx.float32))

    dt_b = blk._broadcast_groups(dt, axis=-1)
    A_b = blk._broadcast_groups(A, axis=-1)
    theta_h = blk._broadcast_groups(theta, axis=0)

    delta_angle = (dt_b.astype(mx.float32)[..., None]
                   * theta_h[None, None, :, :])                 # (B, L, H, N//2)
    prev_cum = angles_cum.astype(mx.float32)
    if exact:
        # Sequential accumulate — bit-matches AR's per-step `delta + prev`.
        ac = prev_cum
        ac_steps = []
        for l in range(L):
            ac = delta_angle[:, l] + ac
            ac_steps.append(ac)
        angles_cum_seq = mx.stack(ac_steps, axis=1)             # (B, L, H, N//2)
    else:
        angles_cum_seq = mx.cumsum(delta_angle, axis=1) + prev_cum[:, None, :, :]
    angles_cum_per_pos = angles_cum_seq                         # (B, L, H, N//2)
    angles = angles_cum_seq.astype(x.dtype)

    B_p, C_p = blk._prepare_BC(B_param, C_param, B_sz, L)
    B_rot = apply_rope(B_p, angles)
    C_rot = apply_rope(C_p, angles)

    x_up = blk.x_up_proj._forward(x_prime_hp.reshape(B_sz, L, H * P))
    x_ssm = x_up.reshape(B_sz, L, H, P, R)

    input_signal = mx.einsum("blhnr,blhpr->blhnp", B_rot, x_ssm)
    prev_input_signal_per_pos = input_signal                    # (B, L, H, N, P)

    lv = mx.sigmoid(blk._broadcast_groups(lam, axis=-1)).reshape(B_sz, L, H, 1, 1).astype(x.dtype)
    dv = dt_b.reshape(B_sz, L, H, 1, 1).astype(x.dtype)
    if exact:
        # AR cast order: multiply in bf16, cast f32, exp, cast back.
        la_full = (dt_b * A_b).astype(mx.float32)
        av = mx.exp(la_full).reshape(B_sz, L, H, 1, 1).astype(x.dtype)
    else:
        av = mx.exp(dt_b * A_b).reshape(B_sz, L, H, 1, 1).astype(x.dtype)

    prev_inp = prev_input_signal.astype(input_signal.dtype)
    if L > 1:
        ip = mx.concatenate([prev_inp[:, None], input_signal[:, :-1]], axis=1)
    else:
        ip = prev_inp[:, None]

    u_ssm = lv * dv * input_signal + (1.0 - lv) * dv * av * ip

    if exact:
        # Sequential SSM recurrence — op-for-op the AR single-step path, so
        # the per-position states match what AR decode would have produced.
        h = h_prev.astype(u_ssm.dtype)
        h_steps = []
        for l in range(L):
            h = av[:, l] * h + u_ssm[:, l]                   # (B, H, N, P) bf16
            h_steps.append(h)
        h_prev_per_pos = mx.stack(h_steps, axis=1)           # (B, L, H, N, P)
        y_stack = mx.einsum("blhnp,blhnr->blhpr", h_prev_per_pos, C_rot)
    else:
        # Chunk-parallel scan (shrunk to L) — mathematically equivalent but
        # rounds differently; drift compounds across rounds via the state.
        co = L if L <= blk.chunk_size else None
        y_stack, h_prev_per_pos = _scan_per_pos(
            blk, u_ssm, dt_b, A_b, C_rot, h_init=h_prev, chunk_size_override=co)

    y = blk.y_down_proj(y_stack.reshape(B_sz, L, H, P * R)).reshape(B_sz, L, H * P)
    D_expand = mx.repeat(blk.D, P, axis=0).astype(x.dtype)
    y = y + x_prime.reshape(B_sz, L, H * P) * D_expand
    mamba_out = blk.mamba_dense_proj(blk.pre_gate_norm(y) * silu(z))
    mid = residual_mamba + blk.ls_mamba(mamba_out)

    normed_mid = blk.norm_out_proj(mid)
    proj_out = blk.out_proj._forward(normed_mid)
    out = mid + blk.ls_out_proj(proj_out)

    return out, h_prev_per_pos, prev_input_signal_per_pos, angles_cum_per_pos


class StaticJacobiDecoder:
    """Greedy Jacobi decoding with the entire verify round in one mx.compile
    program over static KV buffers.

    Usage:
        dec = StaticJacobiDecoder(model)           # model loaded via load_checkpoint
        res = dec.generate(prompt_ids, max_tokens=512, K=12,
                           stop_token_ids=stop_ids, cot_caches=..., cot_bucket=...)
    """

    def __init__(self, model, *, kv_round: int = 256, exact: bool = True):
        self._model = model
        self._kv_round = kv_round
        self._exact = exact
        self._rounds: dict[tuple, Callable] = {}

    def _get_round(self, K: int) -> Callable:
        cached = self._rounds.get((K, self._exact))
        if cached is not None:
            return cached

        model = self._model
        layers = model.backbone.layers
        exact = self._exact

        def round_fn(ids, guesses, write_pos, m, kvs):
            # ids (1, K) int32 = [prev_token, *guesses]; guesses (K-1,) int32.
            x = model.embed(ids)                              # (1, K, d)
            S = kvs[0].shape[2]
            Kk = ids.shape[1]
            qi = mx.arange(Kk)[:, None]
            kj = mx.arange(S)[None, :]
            allow = kj <= (write_pos[0] + qi)
            mask = mx.where(allow,
                            mx.array(0.0, dtype=x.dtype),
                            mx.array(-mx.inf, dtype=x.dtype)).reshape(1, 1, Kk, S)

            per_pos: list[tuple] = []
            new_kv = list(kvs)
            mi = ki = 0
            for blk in layers:
                if isinstance(blk, Mamba3Block):
                    x, h_pp, ip_pp, ac_pp = _mamba_verify_traced(
                        blk, x, m[mi], m[mi + 1], m[mi + 2], exact=exact)
                    per_pos.append((h_pp, ip_pp, ac_pp))
                    mi += 3
                else:
                    q, k_new, v_new = blk._decode_pre(x)      # (1, ., K, 64)
                    kc = mx.slice_update(kvs[ki], k_new, write_pos, axes=(2,))
                    vc = mx.slice_update(kvs[ki + 1], v_new, write_pos, axes=(2,))
                    attn = mx.fast.scaled_dot_product_attention(
                        q, kc, vc, scale=blk.scale, mask=mask)
                    x = blk._decode_post(attn, x)
                    new_kv[ki], new_kv[ki + 1] = kc, vc
                    ki += 2

            logits = model._head_forward(x)                   # (1, K, V) float32
            preds = mx.argmax(logits, axis=-1).astype(mx.int32)[0]   # (K,)

            # Greedy Jacobi acceptance, branchless:
            #   accept while pred[i] == guess[i]; m_acc = leading-match run + 1.
            match = (preds[:-1] == guesses).astype(mx.int32)  # (K-1,)
            m_acc = mx.sum(mx.cumprod(match)) + 1             # scalar, 1..K
            midx = (m_acc - 1).reshape(1)

            # State after exactly m_acc tokens, plucked from per-position records.
            new_m: list[mx.array] = []
            for h_pp, ip_pp, ac_pp in per_pos:
                new_m.append(mx.take(h_pp, midx, axis=1)[:, 0])
                new_m.append(mx.take(ip_pp, midx, axis=1)[:, 0])
                new_m.append(mx.take(ac_pp, midx, axis=1)[:, 0])

            # Rejected KV rows stay in the buffer but are overwritten by the
            # next round (it writes K rows at the new write_pos) and are never
            # attended (mask derives from write_pos) — rollback is free.
            return preds, m_acc, write_pos + m_acc, new_m, new_kv

        self._last_uncompiled = round_fn      # exposed for A/B profiling
        compiled = mx.compile(round_fn)
        self._rounds[(K, self._exact)] = compiled
        return compiled

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        *,
        K: int = 12,
        stop_token_ids: list[int] | tuple = (),
        on_token: Optional[Callable[[int], None]] = None,
        use_ngram: bool = True,
        ngram_n: int = 4,
        preloaded_ngram: Optional[NGramCache] = None,
        preloaded_retriever=None,
        cot_caches: CoTCachesArg = None,
        cot_bucket: Optional[str] = None,
        verbose: bool = False,
    ) -> StaticJacobiResult:
        if K < 2:
            raise ValueError(f"K must be >= 2, got {K}")
        model = self._model
        stop_set = set(int(t) for t in (stop_token_ids or ()))
        round_fn = self._get_round(K)

        # ── prefill (untouched reference path) ───────────────────────────────
        ids = mx.array(prompt_ids, dtype=mx.int32)[None, :]
        t0 = time.perf_counter()
        logits, states = model(ids, states=None)
        last_row = logits[0, -1]
        state_arrays = [v for st in states if st is not None
                        for v in st.values() if v is not None]
        mx.eval(last_row, *state_arrays)
        elapsed_prefill = time.perf_counter() - t0
        n_prompt = len(prompt_ids)

        # ── static buffers ────────────────────────────────────────────────────
        S = -(-(n_prompt + max_tokens + K + 1) // self._kv_round) * self._kv_round
        m_flat: list[mx.array] = []
        kvs: list[mx.array] = []
        for blk, st in zip(model.backbone.layers, states):
            if isinstance(blk, Mamba3Block):
                m_flat += [st["h_prev"], st["prev_input_signal"], st["angles_cum"]]
            else:
                pad = S - st["k"].shape[2]
                kvs.append(mx.pad(st["k"], ((0, 0), (0, 0), (0, pad), (0, 0))))
                kvs.append(mx.pad(st["v"], ((0, 0), (0, 0), (0, pad), (0, 0))))
        write_pos = mx.array([n_prompt], dtype=mx.int32)
        mx.eval(*kvs, write_pos)

        # ── one-time compile (functional: inputs unchanged, outputs dropped) ─
        t_c = time.perf_counter()
        warm = round_fn(mx.zeros((1, K), dtype=mx.int32),
                        mx.zeros((K - 1,), dtype=mx.int32),
                        write_pos, m_flat, kvs)
        mx.eval(warm[0])
        compile_s = time.perf_counter() - t_c

        # ── draft sources (same wiring as jacobi.py) ──────────────────────────
        ngram: Optional[NGramCache] = None
        if use_ngram:
            ngram = preloaded_ngram if preloaded_ngram is not None else NGramCache(n=ngram_n)
            ngram.update_sequence(list(prompt_ids))
        retriever = preloaded_retriever
        if retriever is not None:
            retriever.extend(list(prompt_ids))

        # ── first token: greedy from prefill logits ───────────────────────────
        t_dec = time.perf_counter()
        first_token = int(mx.argmax(last_row).item())
        generated: list[int] = [first_token]
        if on_token is not None:
            on_token(first_token)
        stop_reason = "max_tokens"
        if first_token in stop_set:
            stop_reason = "eos"

        cot_tracker: Optional[CoTPhaseTracker] = None
        if cot_caches is not None:
            bundle = load_cot_caches(cot_caches)
            think_cache, final_cache, cot_retr = bundle.get_caches(cot_bucket)
            if cot_bucket is None:
                cot_retr = None
            init_phase = infer_initial_phase(
                list(prompt_ids) + [first_token], bundle.markers)
            cot_tracker = CoTPhaseTracker(
                bundle.markers, think_cache, final_cache, cot_retr,
                initial=init_phase)

        prev_token = first_token
        fallback_seed = first_token
        n_rounds = 0
        n_accepted = 0
        n_ngram_hits = 0

        # ── Jacobi rounds ─────────────────────────────────────────────────────
        while len(generated) < max_tokens and stop_reason == "max_tokens":
            history = list(prompt_ids) + generated[:-1]
            if cot_tracker is not None or retriever is not None:
                branches, hits = build_hybrid_branches(
                    K, prev_token, history, ngram, retriever, fallback_seed, 1,
                    cot_ngram=(cot_tracker.active_cache() if cot_tracker else None),
                    cot_retriever=(cot_tracker.active_retriever() if cot_tracker else None),
                )
                guesses = branches[0]
            else:
                guesses, hits = _build_guesses(
                    K, prev_token, history, ngram, fallback_seed)
            n_ngram_hits += hits

            ids_arr = mx.array([[prev_token] + guesses], dtype=mx.int32)
            g_arr = mx.array(guesses, dtype=mx.int32)
            preds, m_acc, write_pos, m_flat, kvs = round_fn(
                ids_arr, g_arr, write_pos, m_flat, kvs)

            m_int = int(m_acc.item())              # round boundary sync
            accepted = [int(t) for t in preds.tolist()[:m_int]]
            n_rounds += 1
            n_accepted += m_int

            for tok in accepted:
                generated.append(tok)
                if on_token is not None:
                    on_token(tok)
                if tok in stop_set:
                    stop_reason = "eos"
                    break
                if len(generated) >= max_tokens:
                    break

            prev_token = accepted[-1]
            fallback_seed = prev_token
            if retriever is not None:
                retriever.extend(accepted)
            if ngram is not None:
                full = list(prompt_ids) + generated
                start = max(0, len(full) - m_int - ngram.key_len)
                ngram.update_sequence(full, start_idx=start)
            if cot_tracker is not None:
                cot_tracker.observe(accepted)

            if verbose:
                print(f"[static-jacobi] round={n_rounds:4d} m={m_int:2d}/{K} "
                      f"emitted={len(generated):4d} "
                      f"arl={n_accepted / n_rounds:.2f}", flush=True)

        elapsed_decode = time.perf_counter() - t_dec
        timed = max(len(generated) - 1, 0)
        return StaticJacobiResult(
            tokens=generated,
            stop_reason=stop_reason,
            n_prompt=n_prompt,
            elapsed_prefill=elapsed_prefill,
            elapsed_decode=elapsed_decode,
            prefill_tps=n_prompt / max(elapsed_prefill, 1e-9),
            decode_tps=timed / max(elapsed_decode, 1e-9),
            n_rounds=n_rounds,
            n_accepted=n_accepted,
            arl=n_accepted / max(n_rounds, 1),
            n_ngram_hits=n_ngram_hits,
            K=K,
            compile_s=compile_s,
        )
