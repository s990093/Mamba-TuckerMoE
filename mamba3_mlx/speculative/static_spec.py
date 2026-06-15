"""Optimized static speculative decoder — quantized, single-graph (TPU-like)
verify + compiled draft.

Builds on the two existing compiled pieces and adds the *portable half* of the
production decode optimizations (``mlx_model/static_decode.py``) to the
speculative **verify** path:

  * **Selective 8-bit quant** of every ``TuckerMoE`` U_in/U_out factor (the
    dominant bandwidth carriers) and the LM head, via ``mx.quantized_matmul``.
    Unlike the fused SSM Metal kernel — which is L=1-only — quantized_matmul is
    L-agnostic, so it ports straight into the L=K verify.  Router / inner_norm /
    G_w / SSM stay bf16, so routing decisions (and therefore acceptance) are
    unchanged to within 8-bit U-factor noise (cosine≈1.0 vs bf16; the
    production path documents "CoT quality intact").
  * **One compiled program per K** (TPU-like): the whole verify round —
    embed → 24× Mamba per-position verify → 6× Transformer (static KV via
    slice_update) → quantized head → argmax → cumprod acceptance → per-position
    state gather — is a single ``mx.compile`` graph, exactly like
    ``static_jacobi``; only the heavy matmuls switch to the quantized kernels.
  * **Compiled static-KV draft** (``CompiledDraftGuesser``) with free rollback.

The SSM inner chain stays the exact per-position recurrence (bit-matches the AR
single-step path; the L=1 fused Metal kernel does not vectorize over K and the
prior L>1 Metal scan fusion was rejected — see project memory).

Greedy only.  bf16 verify-vs-AR drift caveats from ``jacobi.py`` apply.
"""
from __future__ import annotations

import time
from typing import Callable, NamedTuple, Optional

import mlx.core as mx

from ..mlx_model.mamba_block import Mamba3Block
from ..mlx_model.ops import apply_rope, scaled_tanh, silu, silu_gating, softplus
from .cot_cache import CoTCachesArg
from .draft_transformer import CompiledDraftGuesser, DraftTransformer
from .ngram_cache import NGramCache


# ── Quantized building blocks (L-agnostic) ──────────────────────────────────

def _qmm(x, pack):
    bits, wq, sc, bs = pack
    return mx.quantized_matmul(x, wq, sc, bs, transpose=True,
                               group_size=64, bits=bits)


def _quant_moe_pack(moe, bits: int):
    """Pre-quantize U_in.T / U_out.T for a TuckerMoE (rows along input dim)."""
    wq_i, s_i, b_i = mx.quantize(moe.U_in.T, group_size=64, bits=bits)
    wq_o, s_o, b_o = mx.quantize(moe.U_out.T, group_size=64, bits=bits)
    return (bits, wq_i, s_i, b_i, wq_o, s_o, b_o)


def _tucker_q8(moe, x, q8, temp: float = 0.5):
    """TuckerMoE forward (decode-style gather, any L) with quantized U_in/U_out.

    Mirrors ``TuckerMoE._forward`` decode branch op-for-op; only the two U
    matmuls become ``quantized_matmul``.  Valid for any L (here L=K verify).
    ``q8=None`` → fall back to the dense bf16 forward.
    """
    if q8 is None:
        return moe._forward(x, temp)
    bits, wq_in, s_in, b_in, wq_out, s_out, b_out = q8
    orig = x.shape
    xf = x.reshape(-1, orig[-1])
    dtype = xf.dtype
    raw = moe.router(xf)                                   # (BL, E) bf16
    rl = scaled_tanh(raw.astype(mx.float32), 10.0) / temp
    probs = mx.softmax(rl, axis=-1)
    idx = mx.argpartition(-rl, kth=moe.top_k - 1, axis=-1)[..., :moe.top_k]
    tkr = mx.take_along_axis(probs, idx, axis=-1)
    tkp = (tkr / (mx.sum(tkr, axis=-1, keepdims=True) + 1e-6)).astype(dtype)
    xs = _qmm(xf, (bits, wq_in, s_in, b_in))
    xs = moe.inner_norm(xs)
    G = moe._G_experts_cache_bf16                          # (E, r3, r2)
    Gw = G[idx[:, 0]] * tkp[:, 0:1, None]
    for k in range(1, moe.top_k):
        Gw = Gw + G[idx[:, k]] * tkp[:, k:k + 1, None]
    xc = mx.einsum("br,brs->bs", xs, Gw)
    out = _qmm(xc, (bits, wq_out, s_out, b_out)) + moe.bias.astype(dtype)
    return out.reshape(*orig[:-1], moe.dim_out)


# ── Mamba per-position verify with quantized MoE U-factors ───────────────────

def _mamba_verify_q8(blk, x, h_prev, prev_input_signal, angles_cum,
                     q8_xup, q8_out, exact: bool = True):
    """``static_jacobi._mamba_verify_traced`` with x_up_proj/out_proj quantized.

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

    delta_angle = (dt_b.astype(mx.float32)[..., None] * theta_h[None, None, :, :])
    prev_cum = angles_cum.astype(mx.float32)
    if exact:
        ac = prev_cum
        ac_steps = []
        for l in range(L):
            ac = delta_angle[:, l] + ac
            ac_steps.append(ac)
        angles_cum_seq = mx.stack(ac_steps, axis=1)
    else:
        angles_cum_seq = mx.cumsum(delta_angle, axis=1) + prev_cum[:, None, :, :]
    angles_cum_per_pos = angles_cum_seq
    angles = angles_cum_seq.astype(x.dtype)

    B_p, C_p = blk._prepare_BC(B_param, C_param, B_sz, L)
    B_rot = apply_rope(B_p, angles)
    C_rot = apply_rope(C_p, angles)

    x_up = _tucker_q8(blk.x_up_proj, x_prime_hp.reshape(B_sz, L, H * P), q8_xup)
    x_ssm = x_up.reshape(B_sz, L, H, P, R)

    input_signal = mx.einsum("blhnr,blhpr->blhnp", B_rot, x_ssm)
    prev_input_signal_per_pos = input_signal

    lv = mx.sigmoid(blk._broadcast_groups(lam, axis=-1)).reshape(B_sz, L, H, 1, 1).astype(x.dtype)
    dv = dt_b.reshape(B_sz, L, H, 1, 1).astype(x.dtype)
    if exact:
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

    h = h_prev.astype(u_ssm.dtype)
    h_steps = []
    for l in range(L):
        h = av[:, l] * h + u_ssm[:, l]
        h_steps.append(h)
    h_prev_per_pos = mx.stack(h_steps, axis=1)
    y_stack = mx.einsum("blhnp,blhnr->blhpr", h_prev_per_pos, C_rot)

    y = blk.y_down_proj(y_stack.reshape(B_sz, L, H, P * R)).reshape(B_sz, L, H * P)
    D_expand = mx.repeat(blk.D, P, axis=0).astype(x.dtype)
    y = y + x_prime.reshape(B_sz, L, H * P) * D_expand
    mamba_out = blk.mamba_dense_proj(blk.pre_gate_norm(y) * silu(z))
    mid = residual_mamba + blk.ls_mamba(mamba_out)

    normed_mid = blk.norm_out_proj(mid)
    proj_out = _tucker_q8(blk.out_proj, normed_mid, q8_out)
    out = mid + blk.ls_out_proj(proj_out)

    return out, h_prev_per_pos, prev_input_signal_per_pos, angles_cum_per_pos


def _tf_post_q8(blk, attn, x, q8_g, q8_u, q8_d):
    """TransformerBlock._decode_post with quantized FFN (gate/up/down)."""
    B, L, D = x.shape
    attn_out = attn.transpose(0, 2, 1, 3).reshape(B, L, D)
    x = x + blk.ls_attn(blk.o_proj(attn_out))
    h = blk.norm_ffn(x)
    gate = _tucker_q8(blk.ffn.gate_proj, h, q8_g)
    feat = _tucker_q8(blk.ffn.up_proj, h, q8_u)
    ffn_out = _tucker_q8(blk.ffn.down_proj, silu_gating(gate, feat), q8_d)
    return x + blk.ls_ffn(ffn_out)


# ── Result ──────────────────────────────────────────────────────────────────

class StaticSpecResult(NamedTuple):
    tokens: list[int]
    stop_reason: str
    n_prompt: int
    elapsed_prefill: float
    elapsed_decode: float
    prefill_tps: float
    decode_tps: float
    n_rounds: int
    n_accepted: int
    arl: float
    K: int
    compile_s: float


# ── Decoder ─────────────────────────────────────────────────────────────────

class StaticSpecDecoder:
    """Quantized single-graph verify + compiled draft speculative decoder.

    Usage::

        dec = StaticSpecDecoder(model, draft, quant_moe_bits=8, quant_head_bits=8)
        res = dec.generate(prompt_ids, max_tokens=128, K=4, stop_token_ids=stop)
    """

    def __init__(self, model, draft: DraftTransformer, *, kv_round: int = 256,
                 quant_moe_bits: int = 8, quant_head_bits: int = 8,
                 exact: bool = True, draft_max_seq: int = 1024):
        self._model = model
        self._kv_round = kv_round
        self._exact = exact
        self._qmoe = int(quant_moe_bits)
        self._qhead = int(quant_head_bits)
        self._rounds: dict[tuple, Callable] = {}
        self._guesser = CompiledDraftGuesser(draft, max_seq=draft_max_seq)

        # Precompute TuckerMoE G caches + quant packs.
        self._q8: dict[int, tuple] = {}
        for blk in model.backbone.layers:
            moes = ([blk.x_up_proj, blk.out_proj] if isinstance(blk, Mamba3Block)
                    else [blk.ffn.gate_proj, blk.ffn.up_proj, blk.ffn.down_proj])
            for moe in moes:
                if moe._G_experts_cache_bf16 is None:
                    moe.precompute_G_experts()
                if self._qmoe:
                    self._q8[id(moe)] = _quant_moe_pack(moe, self._qmoe)
        self._q8_head = None
        if self._qhead:
            wq, sc, bs = mx.quantize(model.head.weight, group_size=64,
                                     bits=self._qhead)
            self._q8_head = (self._qhead, wq, sc, bs)
        mx.eval(*[t for pk in self._q8.values() for t in pk[1:]])

    def _head(self, x):
        h = self._model.norm(x)
        if self._q8_head is not None:
            logits = _qmm(h * self._model.inv_sqrt_d, self._q8_head).astype(mx.float32)
        else:
            logits = self._model.head(h * self._model.inv_sqrt_d).astype(mx.float32)
        from ..mlx_model.ops import scaled_tanh as _st
        return _st(logits, 30.0)

    def _get_round(self, K: int) -> Callable:
        key = (K, self._exact, self._qmoe, self._qhead)
        if key in self._rounds:
            return self._rounds[key]
        model = self._model
        layers = model.backbone.layers
        exact = self._exact
        q8 = self._q8

        def round_fn(ids, guesses, write_pos, m, kvs):
            x = model.embed(ids)
            S = kvs[0].shape[2]
            Kk = ids.shape[1]
            qi = mx.arange(Kk)[:, None]
            kj = mx.arange(S)[None, :]
            allow = kj <= (write_pos[0] + qi)
            mask = mx.where(allow, mx.array(0.0, dtype=x.dtype),
                            mx.array(-mx.inf, dtype=x.dtype)).reshape(1, 1, Kk, S)

            per_pos: list[tuple] = []
            new_kv = list(kvs)
            mi = ki = 0
            for blk in layers:
                if isinstance(blk, Mamba3Block):
                    x, h_pp, ip_pp, ac_pp = _mamba_verify_q8(
                        blk, x, m[mi], m[mi + 1], m[mi + 2],
                        q8.get(id(blk.x_up_proj)), q8.get(id(blk.out_proj)),
                        exact=exact)
                    per_pos.append((h_pp, ip_pp, ac_pp))
                    mi += 3
                else:
                    q, k_new, v_new = blk._decode_pre(x)
                    kc = mx.slice_update(kvs[ki], k_new, write_pos, axes=(2,))
                    vc = mx.slice_update(kvs[ki + 1], v_new, write_pos, axes=(2,))
                    attn = mx.fast.scaled_dot_product_attention(
                        q, kc, vc, scale=blk.scale, mask=mask)
                    if self._qmoe:
                        x = _tf_post_q8(blk, attn, x,
                                        q8[id(blk.ffn.gate_proj)],
                                        q8[id(blk.ffn.up_proj)],
                                        q8[id(blk.ffn.down_proj)])
                    else:
                        x = blk._decode_post(attn, x)
                    new_kv[ki], new_kv[ki + 1] = kc, vc
                    ki += 2

            logits = self._head(x)
            preds = mx.argmax(logits, axis=-1).astype(mx.int32)[0]

            match = (preds[:-1] == guesses).astype(mx.int32)
            m_acc = mx.sum(mx.cumprod(match)) + 1
            midx = (m_acc - 1).reshape(1)

            new_m: list[mx.array] = []
            for h_pp, ip_pp, ac_pp in per_pos:
                new_m.append(mx.take(h_pp, midx, axis=1)[:, 0])
                new_m.append(mx.take(ip_pp, midx, axis=1)[:, 0])
                new_m.append(mx.take(ac_pp, midx, axis=1)[:, 0])
            return preds, m_acc, write_pos + m_acc, new_m, new_kv

        compiled = mx.compile(round_fn)
        self._rounds[key] = compiled
        return compiled

    def generate(self, prompt_ids: list[int], max_tokens: int, *, K: int = 4,
                 stop_token_ids: list[int] | tuple = (),
                 on_token: Optional[Callable[[int], None]] = None,
                 verbose: bool = False) -> StaticSpecResult:
        if K < 2:
            raise ValueError(f"K must be >= 2, got {K}")
        model = self._model
        stop_set = set(int(t) for t in (stop_token_ids or ()))
        round_fn = self._get_round(K)
        guesser = self._guesser

        ids = mx.array(prompt_ids, dtype=mx.int32)[None, :]
        t0 = time.perf_counter()
        logits, states = model(ids, states=None)
        last_row = logits[0, -1]
        mx.eval(last_row, *[v for st in states if st is not None
                            for v in st.values() if v is not None])
        elapsed_prefill = time.perf_counter() - t0
        n_prompt = len(prompt_ids)

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

        t_c = time.perf_counter()
        warm = round_fn(mx.zeros((1, K), dtype=mx.int32),
                        mx.zeros((K - 1,), dtype=mx.int32),
                        write_pos, m_flat, kvs)
        mx.eval(warm[0])
        compile_s = time.perf_counter() - t_c

        t_dec = time.perf_counter()
        first_token = int(mx.argmax(last_row).item())
        generated: list[int] = [first_token]
        if on_token:
            on_token(first_token)
        stop_reason = "eos" if first_token in stop_set else "max_tokens"

        guesser.reset()
        guesser.prefill(list(prompt_ids))
        prev_token = first_token
        n_rounds = n_accepted = 0

        while len(generated) < max_tokens and stop_reason == "max_tokens":
            guesses = guesser.draft(prev_token, K - 1)
            ids_arr = mx.array([[prev_token] + guesses], dtype=mx.int32)
            g_arr = mx.array(guesses, dtype=mx.int32)
            preds, m_acc, write_pos, m_flat, kvs = round_fn(
                ids_arr, g_arr, write_pos, m_flat, kvs)
            m_int = int(m_acc.item())
            accepted = [int(t) for t in preds.tolist()[:m_int]]
            guesser.commit(m_int)
            n_rounds += 1
            n_accepted += m_int
            for tok in accepted:
                generated.append(tok)
                if on_token:
                    on_token(tok)
                if tok in stop_set:
                    stop_reason = "eos"
                    break
                if len(generated) >= max_tokens:
                    break
            prev_token = accepted[-1]
            if verbose:
                print(f"[static-spec] round={n_rounds} m={m_int}/{K} "
                      f"emitted={len(generated)} arl={n_accepted/n_rounds:.2f}",
                      flush=True)

        elapsed_decode = time.perf_counter() - t_dec
        timed = max(len(generated) - 1, 0)
        return StaticSpecResult(
            tokens=generated, stop_reason=stop_reason, n_prompt=n_prompt,
            elapsed_prefill=elapsed_prefill, elapsed_decode=elapsed_decode,
            prefill_tps=n_prompt / max(elapsed_prefill, 1e-9),
            decode_tps=timed / max(elapsed_decode, 1e-9),
            n_rounds=n_rounds, n_accepted=n_accepted,
            arl=n_accepted / max(n_rounds, 1), K=K, compile_s=compile_s)
