"""TPU-style fully-static decode for Mamba3LanguageModel — additive wrapper.

This module does NOT modify any existing mlx_model class or function.  It wraps
an already-loaded ``Mamba3LanguageModel`` and builds one single ``mx.compile``
program for the entire decode step:

    embed → 24×Mamba3Block → 6×TransformerBlock (static KV) → head
          → ban-bias → rep/pres/freq penalties → top-k/top-p/min-p sampling
          → on-GPU repeat-window update → next token

Why this is faster than the per-block compiled path:

  * KV caches are pre-allocated fixed-size buffers written with
    ``mx.slice_update`` — every tensor in the step has a fixed shape, so the
    whole model (not just individual blocks) compiles into ONE program.
    Per-token Python work drops from ~50 dispatches to 1 call.
  * Penalties + sampling run inside the graph on a GPU-resident count vector
    and ring buffer, so the next token never has to round-trip through
    ``.item()`` before the next forward starts.
  * The decode loop is pipelined with ``mx.async_eval``: while the GPU runs
    step n, Python detokenizes/streams step n-1.  ``unroll`` > 1 additionally
    chains several tokens inside one compiled program (sampled token feeds the
    next embed in-graph), dividing the remaining dispatch overhead.

The per-block math is byte-for-byte the existing code: ``Mamba3Block
._decode_impl``, ``TransformerBlock._decode_pre/_decode_post``,
``TuckerMoE._forward`` and ``Mamba3LanguageModel._head_forward`` are inlined
into the traced graph, so numerics match the reference decode path.

Caveats (by design, for the fast path):
  * Stop tokens are detected with up to ``unroll`` tokens of lag; the extra
    tokens are discarded from the output, but the internal state has advanced
    past them — so a finished ``generate()`` is not meant for cross-turn cache
    reuse.  Re-prefill for the next turn (the chat path already does this).
  * ``ban_bias`` is constant during one ``generate()`` call.  Dynamic
    FSM-driven masks (CoT middleware) should keep using the reference path,
    or update the bias at call boundaries with ``unroll`` tokens of lag.
"""

from __future__ import annotations

import time
from typing import Callable, NamedTuple

import mlx.core as mx

from .fused_ssm_step import (fused_pregate, fused_ssm_decode_step,
                             fused_ssm_norm_decode_step, fused_tucker_core_v2,
                             fused_tucker_gw, get_fused_pregate_kernel,
                             get_fused_ssm_kernel, get_fused_ssm_norm_kernel,
                             get_fused_tucker_core_v2_kernel,
                             get_fused_tucker_gw_kernel)
from .hybrid_model import Mamba3LanguageModel
from .mamba_block import Mamba3Block
from .ops import apply_rope_with_sc, scaled_tanh, silu, silu_gating, softplus
from .transformer_block import TransformerBlock


class StaticGenerateResult(NamedTuple):
    tokens: list[int]           # emitted token ids (stop token included, overshoot dropped)
    stop_reason: str            # "max_tokens" | "eos"
    n_prompt: int
    elapsed_prefill: float      # seconds
    elapsed_decode: float       # seconds (compile warmup excluded)
    prefill_tps: float
    decode_tps: float
    compile_s: float            # one-time trace+compile cost paid before decode
    overshoot: int              # tokens computed past the stop token and dropped


def _make_sampler(temp: float, top_k: int, top_p: float, min_p: float,
                  rep_pen: float, pres_pen: float, freq_pen: float) -> Callable:
    """Build a (row_f32, counts, key) -> (tok_i32, key) sampler closure.

    Same math and op order as inference/sampler.py: repetition penalty, then
    presence/frequency penalty, then temperature / top-k / top-p / min-p.
    ``counts`` is the per-vocab occurrence count of the repeat window, so
    ``counts > 0`` is exactly the unique-id set the reference path uses.
    """

    def _sample(row, counts, key):
        z = row
        if rep_pen != 1.0:
            present = counts > 0
            z = mx.where(present, mx.where(z > 0, z / rep_pen, z * rep_pen), z)
        if pres_pen != 0.0 or freq_pen != 0.0:
            z = z - ((counts > 0).astype(z.dtype) * pres_pen + counts * freq_pen)

        if temp <= 0.0:
            return mx.argmax(z, axis=-1).astype(mx.int32), key

        z = z / temp
        V = z.shape[-1]
        topk_vals = None
        if 0 < top_k < V:
            topk_vals = mx.topk(z, top_k)        # ascending; [0] = smallest of the top-k
            kth = topk_vals[..., 0:1]
            z = mx.where(z < kth, mx.array(-1e9, dtype=z.dtype), z)
        if top_p < 1.0:
            # After the top-k filter only `top_k` finite values remain; the
            # -1e9 entries softmax to exactly 0.0, so sorting just the top-k
            # values gives a bit-identical cutoff while skipping a full-vocab
            # sort + cumsum.
            if topk_vals is not None:
                sorted_desc = topk_vals[::-1]
            else:
                sorted_desc = mx.sort(z)[::-1]
            probs = mx.softmax(sorted_desc, axis=-1)
            cum = mx.cumsum(probs, axis=-1)
            shifted_cum = mx.concatenate(
                [mx.zeros((1,), dtype=cum.dtype), cum[:-1]], axis=0)
            keep = shifted_cum < top_p
            kept = mx.where(keep, sorted_desc, mx.array(mx.inf, dtype=sorted_desc.dtype))
            cutoff = mx.min(kept)
            z = mx.where(z < cutoff, mx.array(-1e9, dtype=z.dtype), z)
        if min_p > 0.0:
            probs = mx.softmax(z, axis=-1)
            z = mx.where(probs < mx.max(probs) * min_p, mx.array(-1e9, dtype=z.dtype), z)
        key, sub = mx.random.split(key)
        return mx.random.categorical(z, key=sub).astype(mx.int32), key

    return _sample


def _make_sampler_b(temp: float, top_k: int, top_p: float, min_p: float,
                    rep_pen: float, pres_pen: float, freq_pen: float) -> Callable:
    """Batched _make_sampler: z (B, V), counts (B, V) → tok (B,).

    Same math with axis=-1 reductions; one key draws independent samples
    per row."""

    def _sample(z, counts, key):
        if rep_pen != 1.0:
            present = counts > 0
            z = mx.where(present, mx.where(z > 0, z / rep_pen, z * rep_pen), z)
        if pres_pen != 0.0 or freq_pen != 0.0:
            z = z - ((counts > 0).astype(z.dtype) * pres_pen + counts * freq_pen)

        if temp <= 0.0:
            return mx.argmax(z, axis=-1).astype(mx.int32), key

        z = z / temp
        V = z.shape[-1]
        topk_vals = None
        if 0 < top_k < V:
            topk_vals = mx.topk(z, top_k)        # (B, k) ascending
            kth = topk_vals[..., 0:1]
            z = mx.where(z < kth, mx.array(-1e9, dtype=z.dtype), z)
        if top_p < 1.0:
            if topk_vals is not None:
                sorted_desc = topk_vals[..., ::-1]
            else:
                sorted_desc = mx.sort(z, axis=-1)[..., ::-1]
            probs = mx.softmax(sorted_desc, axis=-1)
            cum = mx.cumsum(probs, axis=-1)
            shifted_cum = mx.concatenate(
                [mx.zeros_like(cum[..., :1]), cum[..., :-1]], axis=-1)
            keep = shifted_cum < top_p
            kept = mx.where(keep, sorted_desc,
                            mx.array(mx.inf, dtype=sorted_desc.dtype))
            cutoff = mx.min(kept, axis=-1, keepdims=True)
            z = mx.where(z < cutoff, mx.array(-1e9, dtype=z.dtype), z)
        if min_p > 0.0:
            probs = mx.softmax(z, axis=-1)
            z = mx.where(probs < mx.max(probs, axis=-1, keepdims=True) * min_p,
                         mx.array(-1e9, dtype=z.dtype), z)
        key, sub = mx.random.split(key)
        return mx.random.categorical(z, key=sub).astype(mx.int32), key

    return _sample


class _MambaConsts(NamedTuple):
    """Per-block constants hoisted out of the per-token graph.

    Mamba3Block._decode_impl recomputes these from weights every step
    (exp(theta_log), head broadcasts, dtype casts).  They are pure functions
    of frozen weights, so StaticDecoder evaluates them once at init.
    """
    theta_h: mx.array    # (H, N//2) float32 = broadcast_groups(exp(theta_log))
    theta0: mx.array     # (N//2,)   float32 — single-group row (G == 1)
    bias_B: mx.array     # (G, N, R) compute dtype
    bias_C: mx.array     # (G, N, R) compute dtype
    D_expand: mx.array   # (H*P,)   compute dtype


def _hoist_mamba_consts(blk: Mamba3Block) -> _MambaConsts:
    dtype = blk.in_proj.weight.dtype
    theta = mx.exp(blk.theta_log.astype(mx.float32))      # (G, N//2)
    d_exp = blk._D_expand if blk._D_expand is not None else mx.repeat(blk.D, blk.P, axis=0)
    c = _MambaConsts(
        theta_h=blk._broadcast_groups(theta, axis=0),     # (H, N//2)
        theta0=theta[0],                                  # (N//2,)
        bias_B=blk.bias_B.astype(dtype),
        bias_C=blk.bias_C.astype(dtype),
        D_expand=d_exp.astype(dtype),
    )
    mx.eval(*c)
    return c


def _mamba_decode_fast(blk: Mamba3Block, c: _MambaConsts,
                       x, h_prev, prev_input_signal, angles_cum):
    """L=1 Mamba decode — same op order as Mamba3Block._decode_impl with two
    value-preserving changes (G == 1 only):

      * weight-derived constants come precomputed from ``c`` instead of being
        recomputed in-graph every token;
      * the group→head ``mx.repeat`` of dt / A / lambda is dropped — with one
        group every head shares the same scalar, so downstream elementwise ops
        broadcast to bit-identical results without materialising (B, 1, H)
        copies.

    Verified token-exact against the reference path (see bench_static --verify).
    """
    B_sz = x.shape[0]
    H, G, P, N, R = blk.H, blk.G, blk.P, blk.N, blk.R
    residual_mamba = x                                        # (B, 1, d_model)
    u = blk.norm_mamba(x)
    raw = blk.in_proj(u)
    z, x_prime, B_param, C_param, dt_p, A_p, lam = blk._split_inproj(raw, B_sz, 1)

    x_prime_hp = x_prime.reshape(B_sz, 1, H, P)
    dt = softplus(dt_p)                                       # (B, 1, 1)  [G == 1]
    A = -mx.exp(A_p)                                          # (B, 1, 1)

    la_full = (dt * A).astype(mx.float32)                     # (B, 1, 1)
    delta_angle = (dt.astype(mx.float32)[..., None]
                   * c.theta_h[None, None, :, :])             # (B, 1, H, N//2)
    angles_cum_seq = delta_angle + angles_cum[:, None, :, :].astype(mx.float32)
    new_angles_cum = angles_cum_seq[:, 0]                     # (B, H, N//2)
    angles = angles_cum_seq.astype(x.dtype)

    _sin_a = mx.sin(angles)[..., None].astype(x.dtype)        # (B, 1, H, N//2, 1)
    _cos_a = mx.cos(angles)[..., None].astype(x.dtype)

    B_p = blk.norm_B(B_param.reshape(B_sz, 1, G, N * R)).reshape(B_sz, 1, G, N, R) + c.bias_B
    C_p = blk.norm_C(C_param.reshape(B_sz, 1, G, N * R)).reshape(B_sz, 1, G, N, R) + c.bias_C
    B_p = blk._broadcast_groups(B_p, axis=2)                  # (B, 1, H, N, R)
    C_p = blk._broadcast_groups(C_p, axis=2)
    B_rot = apply_rope_with_sc(B_p, _sin_a, _cos_a)
    C_rot = apply_rope_with_sc(C_p, _sin_a, _cos_a)

    x_up = blk.x_up_proj._forward(x_prime_hp.reshape(B_sz, 1, H * P))
    x_ssm = x_up.reshape(B_sz, 1, H, P, R)
    input_signal = mx.einsum("blhnr,blhpr->blhnp", B_rot, x_ssm)  # (B, 1, H, N, P)

    lv = mx.sigmoid(lam).reshape(B_sz, 1, 1, 1, 1).astype(x.dtype)
    dv = dt.reshape(B_sz, 1, 1, 1, 1).astype(x.dtype)
    av = mx.exp(la_full).reshape(B_sz, 1, 1, 1, 1).astype(x.dtype)

    ip = prev_input_signal[:, None].astype(input_signal.dtype)
    new_prev_input_signal = input_signal[:, 0]                # (B, H, N, P)
    u_ssm = lv * dv * input_signal + (1.0 - lv) * dv * av * ip

    alpha = av[:, 0]                                          # (B, 1, 1, 1)
    h_new = alpha * h_prev.astype(u_ssm.dtype) + u_ssm[:, 0]  # (B, H, N, P)
    y_stack = mx.einsum("bhnp,bhnr->bhpr", h_new, C_rot[:, 0])[:, None]

    y = blk.y_down_proj(y_stack.reshape(B_sz, 1, H, P * R)).reshape(B_sz, 1, H * P)
    y = y + x_prime.reshape(B_sz, 1, H * P) * c.D_expand

    mamba_out = blk.mamba_dense_proj(blk.pre_gate_norm(y) * silu(z))
    mid = residual_mamba + blk.ls_mamba(mamba_out)
    normed_mid = blk.norm_out_proj(mid)
    proj_out = blk.out_proj._forward(normed_mid)
    out = mid + blk.ls_out_proj(proj_out)
    return out, h_new, new_prev_input_signal, new_angles_cum


def _qmm(x, q):
    """quantized_matmul against a pre-quantized (bits, wq, scales, biases) pack."""
    bits, wq, sc, bs = q
    return mx.quantized_matmul(x, wq, sc, bs, transpose=True,
                               group_size=64, bits=bits)


def _tf_pre_q8(blk: TransformerBlock, x, q8_q, q8_k, q8_v):
    """_decode_pre with 8-bit q/k/v projections (qmv: 1 launch vs splitk's 2)."""
    B, L, D = x.shape
    nx = blk.norm_attn(x)
    q = _qmm(nx, q8_q).reshape(B, L, blk.num_heads, 64).transpose(0, 2, 1, 3)
    k = _qmm(nx, q8_k).reshape(B, L, blk.num_kv_heads, 64).transpose(0, 2, 1, 3)
    v = _qmm(nx, q8_v).reshape(B, L, blk.num_kv_heads, 64).transpose(0, 2, 1, 3)
    return q, k, v


def _tf_pre_q8cat(blk: TransformerBlock, x, q8_qkv):
    """q/k/v as ONE concatenated qmv (quant path: changed tiling is fine)."""
    B, L, D = x.shape
    nx = blk.norm_attn(x)
    qkv = _qmm(nx, q8_qkv)                          # (B, L, (H + 2*kvH) * 64)
    hd = blk.num_heads * 64
    kd = blk.num_kv_heads * 64
    q = qkv[..., :hd].reshape(B, L, blk.num_heads, 64).transpose(0, 2, 1, 3)
    k = qkv[..., hd:hd + kd].reshape(B, L, blk.num_kv_heads, 64).transpose(0, 2, 1, 3)
    v = qkv[..., hd + kd:].reshape(B, L, blk.num_kv_heads, 64).transpose(0, 2, 1, 3)
    return q, k, v


def _tucker_forward_metal(moe, x, kern, wcat=None, q8=None, kern2=None, gT=None):
    """TuckerMoE._forward decode path with the routing chain + expert blend
    (scaled_tanh → softmax → top-2 → weights → G_w) in one fused kernel.
    The real matmul work — router/U_in gemvs, the core einsum, addmm — stays
    on MLX's tuned kernels (a handwritten contraction measured slower).

    ``wcat`` (dim_in, E + r3) = concat(router.weight.T, U_in): router and
    U_in share the same input, so one gemv launch replaces two.

    ``q8`` = (bits, wq_in, s_in, b_in, wq_out, s_out, b_out): pre-quantized
    U_in.T / U_out.T (see StaticDecoder quant_moe_bits).  The U factors are
    the model's dominant bandwidth carriers; router/inner_norm/G_w/einsum
    stay bf16, so routing decisions are unaffected.  NOT bit-exact.

    ``kern2``+``gT``: tucker_core_v2 mega-kernel — inner_norm + routing +
    blend + contraction in one launch over pre-transposed G (quant path
    only; replaces 3 launches and the G_w roundtrip)."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    dtype = x_flat.dtype
    if q8 is not None and kern2 is not None and gT is not None:
        bits, wq_in, s_in, b_in, _wq_o, _s_o, _b_o = q8
        gt, obits, wq_op, s_op, b_op = gT
        raw_logits = moe.router(x_flat)                            # (1, E)
        xs_raw = mx.quantized_matmul(x_flat, wq_in, s_in, b_in,
                                     transpose=True, group_size=64, bits=bits)
        x_core = fused_tucker_core_v2(kern2, moe._r3, moe._r2,
                                      raw_logits, xs_raw, gt,
                                      moe.inner_norm.weight)
        # padded x_core: [R2]=1 row picks the baked-in bias inside the qmv
        out = mx.quantized_matmul(x_core, wq_op, s_op, b_op,
                                  transpose=True, group_size=64, bits=obits)
        return out.reshape(*orig_shape[:-1], moe.dim_out)
    if q8 is not None:
        bits, wq_in, s_in, b_in, wq_out, s_out, b_out = q8
        raw_logits = moe.router(x_flat)                            # (1, E)
        x_shared = mx.quantized_matmul(x_flat, wq_in, s_in, b_in,
                                       transpose=True, group_size=64, bits=bits)
        x_shared = moe.inner_norm(x_shared)
    elif wcat is not None:
        fused = mx.matmul(x_flat, wcat)                            # (1, E + r3)
        raw_logits = fused[..., :moe.num_experts]
        x_shared = moe.inner_norm(fused[..., moe.num_experts:])
    else:
        raw_logits = moe.router(x_flat)                            # (1, E)
        x_shared = mx.matmul(x_flat, moe.U_in.astype(dtype))       # (1, r3)
        x_shared = moe.inner_norm(x_shared)
    G_w = fused_tucker_gw(kern, moe._r3, moe._r2, raw_logits,
                          moe._G_experts_cache_bf16)               # (1, r3, r2)
    x_core = mx.einsum("br,brs->bs", x_shared, G_w)
    if q8 is not None:
        out = mx.quantized_matmul(x_core, wq_out, s_out, b_out,
                                  transpose=True, group_size=64, bits=bits)
        out = out + moe.bias.astype(dtype)
    else:
        out = mx.addmm(moe.bias.astype(dtype), x_core, moe.U_out.astype(dtype))
    return out.reshape(*orig_shape[:-1], moe.dim_out)


def _tf_pre_metal(blk: TransformerBlock, x, wqkv):
    """TransformerBlock._decode_pre with q/k/v projected by one concatenated
    gemv (they share the normed input; weights pre-concatenated at init)."""
    B, L, D = x.shape
    nx = blk.norm_attn(x)
    qkv = mx.matmul(nx, wqkv)                       # (B, L, (H + 2*kvH) * 64)
    hd = blk.num_heads * 64
    kd = blk.num_kv_heads * 64
    q = qkv[..., :hd].reshape(B, L, blk.num_heads, 64).transpose(0, 2, 1, 3)
    k = qkv[..., hd:hd + kd].reshape(B, L, blk.num_kv_heads, 64).transpose(0, 2, 1, 3)
    v = qkv[..., hd + kd:].reshape(B, L, blk.num_kv_heads, 64).transpose(0, 2, 1, 3)
    return q, k, v


def _tf_post_metal(blk: TransformerBlock, attn, x, kern, wcats=None, q8s=None,
                   q8_o=None, kern2=None, gTs=None):
    """TransformerBlock._decode_post with fused-MoE FFN (same op order)."""
    B, L, D = x.shape
    attn_out = attn.transpose(0, 2, 1, 3).reshape(B, L, D)
    if q8_o is not None:
        x = x + blk.ls_attn(_qmm(attn_out, q8_o) + blk.o_proj.bias)
    else:
        x = x + blk.ls_attn(blk.o_proj(attn_out))
    h = blk.norm_ffn(x)
    wg = wu = wd = None
    if wcats is not None:
        wg, wu, wd = wcats
    qg = qu = qd = None
    if q8s is not None:
        qg, qu, qd = q8s
    gg = gu = gd = None
    if gTs is not None:
        gg, gu, gd = gTs
    gate = _tucker_forward_metal(blk.ffn.gate_proj, h, kern, wg, qg, kern2, gg)
    feat = _tucker_forward_metal(blk.ffn.up_proj, h, kern, wu, qu, kern2, gu)
    ffn_out = _tucker_forward_metal(blk.ffn.down_proj, silu_gating(gate, feat),
                                    kern, wd, qd, kern2, gd)
    return x + blk.ls_ffn(ffn_out)


def _mamba_decode_metal(blk: Mamba3Block, c: _MambaConsts, kernel,
                        x, h_prev, prev_input_signal, angles_cum,
                        kern_moe=None, wcat_xup=None, wcat_out=None,
                        q8_xup=None, q8_out=None, q8_inproj=None, q8_dense=None,
                        nf=None, kern2=None, gt_xup=None, gt_out=None,
                        kern_pg=None):
    """_mamba_decode_fast with the SSM inner chain in one fused Metal kernel.

    Matmuls and norms stay on MLX kernels (in_proj, TuckerMoE, y_down_proj,
    dense_proj, RMS norms); everything between — scalar activations, RoPE,
    the two einsums, the lv/dv/av blend and the h recurrence — is a single
    launch (see fused_ssm_step.py).  bf16 rounding boundaries match the mx
    op sequence, so outputs are bit-compatible with the unfused path.
    """
    B_sz = x.shape[0]
    H, G, P, N, R = blk.H, blk.G, blk.P, blk.N, blk.R
    residual_mamba = x
    u = blk.norm_mamba(x)
    if q8_inproj is not None:
        # 8-bit main columns [z | x' | B | C]; the SSM-sensitive dt/A/λ tail
        # (3 columns) stays a bf16 gemv — full in_proj quant broke decode
        # quality before precisely through this path.
        q_main, w_tail, b_main, b_tail = q8_inproj
        main = _qmm(u, q_main) + b_main
        tail = mx.matmul(u, w_tail) + b_tail                  # (B, 1, 3)
        dz, dx, db = blk.dim_z, blk.dim_x, blk.dim_B
        z       = main[..., :dz]
        x_prime = main[..., dz:dz + dx]
        B_param = main[..., dz + dx:dz + dx + db]
        C_param = main[..., dz + dx + db:]
        dt_p, A_p, lam = tail[..., 0:1], tail[..., 1:2], tail[..., 2:3]
    else:
        raw = blk.in_proj(u)
        z, x_prime, B_param, C_param, dt_p, A_p, lam = blk._split_inproj(raw, B_sz, 1)

    if kern_moe is not None:
        x_up = _tucker_forward_metal(blk.x_up_proj, x_prime.reshape(B_sz, 1, H * P),
                                     kern_moe, wcat_xup, q8_xup, kern2, gt_xup)
    else:
        x_up = blk.x_up_proj._forward(x_prime.reshape(B_sz, 1, H * P))   # (1, 1, H*P*R)

    if nf is not None:
        # norm_B/norm_C + bias folded into the kernel (bit-exact rms replica)
        wB, wC, bB, bC = nf
        y4, h_new, new_ip, new_ac = fused_ssm_norm_decode_step(
            kernel, H, N, P, R,
            dt_p.reshape(B_sz), A_p.reshape(B_sz), lam.reshape(B_sz),
            c.theta0, angles_cum.astype(mx.float32),
            B_param.reshape(B_sz, N * R), C_param.reshape(B_sz, N * R),
            wB, wC, bB, bC,
            x_up, h_prev, prev_input_signal)
    else:
        B_p = blk.norm_B(B_param.reshape(B_sz, 1, G, N * R)).reshape(B_sz, 1, G, N, R) + c.bias_B
        C_p = blk.norm_C(C_param.reshape(B_sz, 1, G, N * R)).reshape(B_sz, 1, G, N, R) + c.bias_C
        y4, h_new, new_ip, new_ac = fused_ssm_decode_step(
            kernel, H, N, P, R,
            dt_p.reshape(B_sz), A_p.reshape(B_sz), lam.reshape(B_sz),
            c.theta0, angles_cum.astype(mx.float32),
            B_p.reshape(B_sz, N * R), C_p.reshape(B_sz, N * R),
            x_up, h_prev, prev_input_signal)

    y = blk.y_down_proj(y4).reshape(B_sz, 1, H * P)
    y = y + x_prime.reshape(B_sz, 1, H * P) * c.D_expand

    if kern_pg is not None:
        gated = fused_pregate(kern_pg, H * P, y, z, blk.pre_gate_norm.weight)
    else:
        gated = blk.pre_gate_norm(y) * silu(z)
    if q8_dense is not None:
        mamba_out = _qmm(gated, q8_dense)
    else:
        mamba_out = blk.mamba_dense_proj(gated)
    mid = residual_mamba + blk.ls_mamba(mamba_out)
    normed_mid = blk.norm_out_proj(mid)
    if kern_moe is not None:
        proj_out = _tucker_forward_metal(blk.out_proj, normed_mid, kern_moe,
                                         wcat_out, q8_out, kern2, gt_out)
    else:
        proj_out = blk.out_proj._forward(normed_mid)
    out = mid + blk.ls_out_proj(proj_out)
    return out, h_new, new_ip, new_ac


def _push_window(tok_i32, counts, ring, ring_pos):
    """Record a sampled token in the GPU-resident repeat window (branchless).

    ring holds the last W token ids (-1 = empty slot); counts is the (V,)
    occurrence vector the penalties read.  Evicting an empty slot decrements
    index 0 by 0.0, i.e. a no-op — no Python branching needed.
    """
    W = ring.shape[0]
    old = mx.take(ring, ring_pos)                          # (1,)
    dec = (old >= 0).astype(counts.dtype)                  # (1,) 1.0 if slot occupied
    counts = counts.at[mx.maximum(old, 0)].add(-dec)
    tok1 = tok_i32.reshape(1)
    counts = counts.at[tok1].add(mx.ones((1,), dtype=counts.dtype))
    ring = mx.slice_update(ring, tok1, ring_pos, axes=(0,))
    ring_pos = (ring_pos + 1) % W
    return counts, ring, ring_pos


def _push_window_b(tok_i32, counts, ring, ring_pos):
    """Batched _push_window: tok (B,), counts (B, V), ring (B, W).

    Per-row scatter via take/put_along_axis (all rows advance in lockstep,
    so ring_pos stays a shared scalar)."""
    W = ring.shape[1]
    old = mx.take_along_axis(
        ring, mx.broadcast_to(ring_pos.reshape(1, 1), (ring.shape[0], 1)),
        axis=1)                                            # (B, 1)
    dec = (old >= 0).astype(counts.dtype)                  # (B, 1)
    idx_old = mx.maximum(old, 0)
    cur = mx.take_along_axis(counts, idx_old, axis=1)
    counts = mx.put_along_axis(counts, idx_old, cur - dec, axis=1)
    tok1 = tok_i32.reshape(-1, 1)                          # (B, 1)
    cur = mx.take_along_axis(counts, tok1, axis=1)
    counts = mx.put_along_axis(counts, tok1, cur + 1.0, axis=1)
    ring = mx.slice_update(ring, tok1, ring_pos, axes=(1,))
    ring_pos = (ring_pos + 1) % W
    return counts, ring, ring_pos


class StaticDecoder:
    """Single-graph static decode wrapper around an existing Mamba3LanguageModel.

    Usage:
        model = Mamba3LanguageModel(cfg); load_checkpoint(model, path)
        dec = StaticDecoder(model)
        res = dec.generate(prompt_ids, gen_cfg, stop_token_ids=stop_ids, unroll=4)
    """

    def __init__(self, model: Mamba3LanguageModel, *, kv_round: int = 256,
                 fast_math: bool = True, metal_fuse: bool = False,
                 metal_moe: bool = True, concat_gemv: bool = False,
                 quant_moe_bits: int = 0, quant_proj_bits: int = 0,
                 quant_head_bits: int = 0, norm_fold: bool = True,
                 moe_fuse_v2: bool = False):
        self._model = model
        self._kv_round = kv_round
        # Compiled step programs keyed by (unroll, sampling-params tuple).
        # Shape changes (different KV bucket / window) re-trace automatically
        # inside mx.compile, so shapes are not part of the key.
        self._steps: dict[tuple, Callable] = {}
        self._ensure_caches()
        # fast_math: hoisted-constant Mamba step (value-identical, fewer
        # kernels).  Only valid for single-group models; falls back otherwise.
        self._fast = fast_math and model.config.n_groups == 1
        # metal_fuse: single fused Metal kernel for the SSM inner chain.
        self._metal = metal_fuse and model.config.n_groups == 1
        self._consts: dict[int, _MambaConsts] = {}
        if self._fast or self._metal:
            for i, blk in enumerate(model.backbone.layers):
                if isinstance(blk, Mamba3Block):
                    self._consts[i] = _hoist_mamba_consts(blk)
        self._kernel = None
        self._kern_moe = None
        self._kern_pg = None
        # norm-fold: norm_B/norm_C + bias computed inside the SSM kernel via
        # a bit-exact replica of MLX's rms_single_row reduction (saves 2 rms
        # + 2 add launches per Mamba block).  Needs R == 4 (= rms N_READS).
        self._nf: dict[int, tuple] = {}
        if self._metal:
            cfg = model.config
            use_nf = norm_fold and cfg.mimo_rank == 4
            if use_nf:
                self._kernel = get_fused_ssm_norm_kernel(
                    cfg.n_heads, cfg.d_state, cfg.d_head, cfg.mimo_rank,
                    eps=cfg.rms_norm_eps)
                nr = cfg.d_state * cfg.mimo_rank
                for li, blk in enumerate(model.backbone.layers):
                    if isinstance(blk, Mamba3Block):
                        t = (blk.norm_B.weight, blk.norm_C.weight,
                             self._consts[li].bias_B.reshape(nr),
                             self._consts[li].bias_C.reshape(nr))
                        mx.eval(*t)
                        self._nf[id(blk)] = t
            else:
                self._kernel = get_fused_ssm_kernel(
                    cfg.n_heads, cfg.d_state, cfg.d_head, cfg.mimo_rank)
            # pre_gate_norm(y) * silu(z) in one launch (bit-exact replica)
            self._kern_pg = get_fused_pregate_kernel(
                cfg.n_heads * cfg.d_head, cfg.rms_norm_eps)
            # Routing-chain fusion: scaled_tanh→softmax→top-2→weights→G_w in
            # one launch (the contraction einsum stays on MLX — a handwritten
            # contraction kernel measured slower, see git history).
            if metal_moe and cfg.kmoe_top_k == 2 and cfg.kmoe_num_experts == 8:
                self._kern_moe = get_fused_tucker_gw_kernel(
                    E=cfg.kmoe_num_experts, R3=cfg.kmoe_r3, R2=cfg.kmoe_r2)
        # Shared-input gemv concatenation: router+U_in per MoE and q/k/v per
        # transformer block share one input each → one launch instead of 2-3.
        # MEASURED AND REJECTED as default (2026-06-12): the merged gemvs are
        # bandwidth-carriers (same bytes either way) so only the ~5 µs launch
        # is saved (~0.4 ms total, within noise), while the changed gemv
        # tiling breaks bit-exactness (greedy diverges, sampled 36/128).
        # Kept behind concat_gemv=True for experiments only.
        self._wcat: dict[int, mx.array] = {}     # id(moe) → (dim_in, E + r3)
        self._wqkv: dict[int, mx.array] = {}     # layer idx → (d_model, (H+2kvH)*64)
        # Selective MoE quantization: only the U_in/U_out factor matrices of
        # the 66 TuckerMoE calls (~40% of per-token weight traffic, the
        # dominant bandwidth carriers per the gputrace analysis).  Routing
        # (router gemv, G_w blend), in_proj/dt-path, dense_proj, attention
        # and head stay bf16 — full 8-bit quant broke output quality before
        # (SSM dt/A sensitivity); this targets only the robust factors.
        # NOT bit-exact vs the bf16 path by definition.
        self._q8: dict[int, tuple] = {}          # id(moe) → (bits, wq/s/b ×2)
        if quant_moe_bits and self._metal and self._kern_moe is not None:
            bits = int(quant_moe_bits)
            for blk in model.backbone.layers:
                if isinstance(blk, Mamba3Block):
                    moes = (blk.x_up_proj, blk.out_proj)
                elif isinstance(blk, TransformerBlock):
                    moes = (blk.ffn.gate_proj, blk.ffn.up_proj, blk.ffn.down_proj)
                else:
                    moes = ()
                for moe in moes:
                    # quantized_matmul wants (out, in) rows quantized along
                    # the input dim → quantize the transposes.
                    wq_i, s_i, b_i = mx.quantize(moe.U_in.T, group_size=64, bits=bits)
                    wq_o, s_o, b_o = mx.quantize(moe.U_out.T, group_size=64, bits=bits)
                    mx.eval(wq_i, s_i, b_i, wq_o, s_o, b_o)
                    self._q8[id(moe)] = (bits, wq_i, s_i, b_i, wq_o, s_o, b_o)
        # tucker_core_v2 mega-kernel: inner_norm + routing chain + expert
        # blend + core contraction in ONE launch over pre-transposed G.
        # MEASURED AND REJECTED as default (2026-06-13, 6th contraction
        # attempt): barrier variant -18.6%, barrier-free streaming variant
        # -17.2% on a cool machine (113 vs 137 tok/s; an earlier +8% reading
        # was thermal-throttle noise).  512 threads with serial 64-iteration
        # loops are latency-bound vs MLX's gemv occupancy.  Kept behind
        # moe_fuse_v2=True for reference.
        self._kern_moe2 = None
        self._gt: dict[int, mx.array] = {}
        if (moe_fuse_v2 and self._q8 and self._kern_moe is not None):
            cfg = model.config
            self._kern_moe2 = get_fused_tucker_core_v2_kernel(
                E=cfg.kmoe_num_experts, R3=cfg.kmoe_r3, R2=cfg.kmoe_r2,
                eps=cfg.rms_norm_eps)
            for blk in model.backbone.layers:
                if isinstance(blk, Mamba3Block):
                    moes = (blk.x_up_proj, blk.out_proj)
                elif isinstance(blk, TransformerBlock):
                    moes = (blk.ffn.gate_proj, blk.ffn.up_proj, blk.ffn.down_proj)
                else:
                    moes = ()
                for moe in moes:
                    gt = mx.contiguous(
                        moe._G_experts_cache_bf16.transpose(0, 2, 1))
                    # bias baked into the U_out qmv: row R2 of the padded
                    # weight is the bias (v2 kernel emits x_core[R2] = 1).
                    pad = mx.concatenate([
                        moe.U_out,
                        moe.bias[None, :],
                        mx.zeros((63, moe.dim_out), dtype=moe.U_out.dtype),
                    ], axis=0)
                    wq_o, s_o, b_o = mx.quantize(pad.T, group_size=64,
                                                 bits=int(quant_moe_bits))
                    mx.eval(gt, wq_o, s_o, b_o)
                    self._gt[id(moe)] = (gt, int(quant_moe_bits), wq_o, s_o, b_o)
        # Stage 2: the plain Linear bandwidth carriers.  in_proj is split —
        # its dt/A/λ tail (3 output columns feeding the SSM scalar path, the
        # part that broke under full quantization) stays a bf16 gemv.
        self._q8_inproj: dict[int, tuple] = {}   # id(blk) → (qpack, w_tail, b_main, b_tail)
        self._q8_lin: dict[int, tuple] = {}      # id(linear) → (bits, wq, sc, bs)
        self._q8_qkv: dict[int, tuple] = {}      # id(blk) → concat qkv qpack
        self._q8_head: tuple | None = None
        if quant_proj_bits and self._metal:
            bits = int(quant_proj_bits)

            def qpack(w):
                wq, sc, bs = mx.quantize(w, group_size=64, bits=bits)
                mx.eval(wq, sc, bs)
                return (bits, wq, sc, bs)

            for blk in model.backbone.layers:
                if isinstance(blk, Mamba3Block):
                    n_main = blk.dim_z + blk.dim_x + blk.dim_B + blk.dim_C
                    if n_main % 64 == 0 and blk.G == 1:
                        w = blk.in_proj.weight                   # (out, in)
                        w_tail = w[n_main:].T                    # (in, 3)
                        b = blk.in_proj.bias
                        mx.eval(w_tail)
                        self._q8_inproj[id(blk)] = (
                            qpack(w[:n_main]), w_tail, b[:n_main], b[n_main:])
                    self._q8_lin[id(blk.mamba_dense_proj)] = qpack(
                        blk.mamba_dense_proj.weight)
                elif isinstance(blk, TransformerBlock):
                    for lin in (blk.q_proj, blk.k_proj, blk.v_proj, blk.o_proj):
                        self._q8_lin[id(lin)] = qpack(lin.weight)
                    # one concatenated q/k/v qmv (3 launches → 1)
                    self._q8_qkv[id(blk)] = qpack(mx.concatenate(
                        [blk.q_proj.weight, blk.k_proj.weight,
                         blk.v_proj.weight], axis=0))
        if quant_head_bits and self._metal:
            wq, sc, bs = mx.quantize(model.head.weight,
                                     group_size=64, bits=int(quant_head_bits))
            mx.eval(wq, sc, bs)
            self._q8_head = (int(quant_head_bits), wq, sc, bs)
        if concat_gemv and self._metal and self._kern_moe is not None:
            for li, blk in enumerate(model.backbone.layers):
                if isinstance(blk, Mamba3Block):
                    moes = (blk.x_up_proj, blk.out_proj)
                elif isinstance(blk, TransformerBlock):
                    moes = (blk.ffn.gate_proj, blk.ffn.up_proj, blk.ffn.down_proj)
                    wqkv = mx.concatenate(
                        [blk.q_proj.weight.T, blk.k_proj.weight.T, blk.v_proj.weight.T],
                        axis=1)
                    mx.eval(wqkv)
                    self._wqkv[li] = wqkv
                else:
                    moes = ()
                for moe in moes:
                    wc = mx.concatenate([moe.router.weight.T, moe.U_in], axis=1)
                    mx.eval(wc)
                    self._wcat[id(moe)] = wc

    # ── setup ────────────────────────────────────────────────────────────────

    def _ensure_caches(self) -> None:
        """Make sure the per-block constant caches exist (idempotent).

        load_checkpoint() already calls model.precompute(); this guard only
        covers models whose weights were loaded by other means.
        """
        for blk in self._model.backbone.layers:
            if isinstance(blk, Mamba3Block):
                if blk._D_expand is None or blk.x_up_proj._G_experts_cache_bf16 is None:
                    self._model.precompute()
                    return
            elif isinstance(blk, TransformerBlock):
                if blk.ffn.gate_proj._G_experts_cache_bf16 is None:
                    self._model.precompute()
                    return

    # ── compiled step builder ────────────────────────────────────────────────

    def _make_core_fn(self, batched: bool = False,
                      metal: bool | None = None) -> Callable:
        """Build core(x_tok, write_pos, m, kvs) → (row, new_m, new_kv):
        embed → 30 blocks → head logits row (V,) f32, no ban_bias/sampler.
        Shared by the sampling step (_get_step) and the streaming forward
        (_get_fwd_step) so both compiled paths trace the same graph.
        ``batched``: row is (B, V) instead of (V,) — used by generate_batch;
        the B == 1 trace is untouched (bit-exactness preserved).
        ``metal``: override the decoder's metal_fuse for this core — the
        custom SSM kernel scales poorly with B (measured B=8: 160 vs 270
        aggregate), so the batched step forces the plain compiled graph."""
        model = self._model
        layers = model.backbone.layers
        consts = self._consts
        use_fast = self._fast
        use_metal = self._metal if metal is None else (metal and self._metal)
        kernel = self._kernel
        kern_moe = self._kern_moe
        wcat = self._wcat
        wqkv = self._wqkv
        q8 = self._q8
        q8_inproj = self._q8_inproj
        q8_lin = self._q8_lin
        q8_head = self._q8_head
        nf = self._nf
        kern2 = self._kern_moe2
        gt = self._gt
        q8_qkv = self._q8_qkv
        kern_pg = self._kern_pg

        def core(x_tok, write_pos, m, kvs):
            x = model.embed(x_tok)                              # (1, 1, d)
            S = kvs[0].shape[2]
            valid = mx.arange(S) <= write_pos[0]
            mask = mx.where(valid,
                            mx.array(0.0, dtype=x.dtype),
                            mx.array(-mx.inf, dtype=x.dtype)).reshape(1, 1, 1, S)

            new_m = list(m)
            new_kv = list(kvs)
            mi = ki = 0
            for li, blk in enumerate(layers):
                if isinstance(blk, Mamba3Block):
                    if use_metal:
                        x, new_m[mi], new_m[mi + 1], new_m[mi + 2] = _mamba_decode_metal(
                            blk, consts[li], kernel, x, m[mi], m[mi + 1], m[mi + 2],
                            kern_moe=kern_moe,
                            wcat_xup=wcat.get(id(blk.x_up_proj)),
                            wcat_out=wcat.get(id(blk.out_proj)),
                            q8_xup=q8.get(id(blk.x_up_proj)),
                            q8_out=q8.get(id(blk.out_proj)),
                            q8_inproj=q8_inproj.get(id(blk)),
                            q8_dense=q8_lin.get(id(blk.mamba_dense_proj)),
                            nf=nf.get(id(blk)), kern2=kern2,
                            gt_xup=gt.get(id(blk.x_up_proj)),
                            gt_out=gt.get(id(blk.out_proj)),
                            kern_pg=kern_pg)
                    elif use_fast:
                        x, new_m[mi], new_m[mi + 1], new_m[mi + 2] = _mamba_decode_fast(
                            blk, consts[li], x, m[mi], m[mi + 1], m[mi + 2])
                    else:
                        x, new_m[mi], new_m[mi + 1], new_m[mi + 2] = blk._decode_impl(
                            x, m[mi], m[mi + 1], m[mi + 2])
                    mi += 3
                else:
                    if use_metal and id(blk) in q8_qkv:
                        q, k_new, v_new = _tf_pre_q8cat(blk, x, q8_qkv[id(blk)])
                    elif use_metal and id(blk.q_proj) in q8_lin:
                        q, k_new, v_new = _tf_pre_q8(
                            blk, x, q8_lin[id(blk.q_proj)],
                            q8_lin[id(blk.k_proj)], q8_lin[id(blk.v_proj)])
                    elif use_metal and li in wqkv:
                        q, k_new, v_new = _tf_pre_metal(blk, x, wqkv[li])
                    else:
                        q, k_new, v_new = blk._decode_pre(x)
                    kc = mx.slice_update(kvs[ki], k_new, write_pos, axes=(2,))
                    vc = mx.slice_update(kvs[ki + 1], v_new, write_pos, axes=(2,))
                    attn = mx.fast.scaled_dot_product_attention(
                        q, kc, vc, scale=blk.scale, mask=mask)
                    if use_metal and kern_moe is not None:
                        x = _tf_post_metal(blk, attn, x, kern_moe, wcats=(
                            wcat.get(id(blk.ffn.gate_proj)),
                            wcat.get(id(blk.ffn.up_proj)),
                            wcat.get(id(blk.ffn.down_proj))), q8s=(
                            q8.get(id(blk.ffn.gate_proj)),
                            q8.get(id(blk.ffn.up_proj)),
                            q8.get(id(blk.ffn.down_proj))),
                            q8_o=q8_lin.get(id(blk.o_proj)),
                            kern2=kern2, gTs=(
                            gt.get(id(blk.ffn.gate_proj)),
                            gt.get(id(blk.ffn.up_proj)),
                            gt.get(id(blk.ffn.down_proj))))
                    else:
                        x = blk._decode_post(attn, x)
                    new_kv[ki], new_kv[ki + 1] = kc, vc
                    ki += 2

            if q8_head is not None:
                h = model.norm(x)
                lg = _qmm(h * model.inv_sqrt_d, q8_head).astype(mx.float32)
                lg = scaled_tanh(lg, 30.0)
            else:
                lg = model._head_forward(x)
            row = lg[:, -1] if batched else lg[0, -1]            # f32 logits
            return row, new_m, new_kv

        return core

    def _get_fwd_step(self) -> Callable:
        """Compiled forward-only step for streaming callers that own their
        sampling/penalties/injection (see StaticStreamSession): one dispatch
        per token, returns the raw logits row.  Cached like _get_step."""
        key = ("fwd", self._fast, self._metal, self._kern_moe is not None,
               bool(self._q8), bool(self._q8_lin), self._q8_head is not None,
               bool(self._nf), self._kern_moe2 is not None)
        cached = self._steps.get(key)
        if cached is not None:
            return cached
        core = self._make_core_fn()

        def fwd(x_tok, write_pos, m, kvs):
            row, new_m, new_kv = core(x_tok, write_pos, m, kvs)
            return row, write_pos + 1, new_m, new_kv

        compiled = mx.compile(fwd)
        self._steps[key] = compiled
        return compiled

    def _get_step(self, unroll: int, samp_key: tuple) -> Callable:
        cached = self._steps.get(
            (unroll, samp_key, self._fast, self._metal, self._kern_moe is not None,
             bool(self._q8), bool(self._q8_lin), self._q8_head is not None,
             bool(self._nf), self._kern_moe2 is not None))
        if cached is not None:
            return cached

        core = self._make_core_fn()
        sample = _make_sampler(*samp_key)

        def one_token(x_tok, write_pos, m, kvs, counts, ring, ring_pos, ban_bias, key):
            row, new_m, new_kv = core(x_tok, write_pos, m, kvs)
            row = row + ban_bias                                 # (V,) float32
            tok, key = sample(row, counts, key)
            counts, ring, ring_pos = _push_window(tok, counts, ring, ring_pos)
            return tok, write_pos + 1, new_m, new_kv, counts, ring, ring_pos, key

        def step(x_tok, write_pos, m, kvs, counts, ring, ring_pos, ban_bias, key):
            toks = []
            for _ in range(unroll):
                tok, write_pos, m, kvs, counts, ring, ring_pos, key = one_token(
                    x_tok, write_pos, m, kvs, counts, ring, ring_pos, ban_bias, key)
                toks.append(tok)
                x_tok = tok.reshape(1, 1)                        # feeds next embed in-graph
            return mx.stack(toks), write_pos, m, kvs, counts, ring, ring_pos, key

        compiled = mx.compile(step)
        self._steps[(unroll, samp_key, self._fast, self._metal,
                     self._kern_moe is not None, bool(self._q8),
                     bool(self._q8_lin), self._q8_head is not None,
                     bool(self._nf), self._kern_moe2 is not None)] = compiled
        return compiled

    def _get_step_batch(self, unroll: int, samp_key: tuple) -> Callable:
        """Batched compiled step: B independent streams in one graph.

        Decode is launch-bound, so the batch dimension is nearly free
        (measured B=4 ≈ 1.55× the B=1 step cost → ~2.6× aggregate tok/s);
        all matmuls/kernels carry a B leading dim, penalties/sampling are
        per-row.  Same-shape prompts only (replicated prompt or equal-length
        batch)."""
        key = ("batch", unroll, samp_key, self._fast, self._metal,
               self._kern_moe is not None, bool(self._q8), bool(self._q8_lin),
               self._q8_head is not None, bool(self._nf))
        cached = self._steps.get(key)
        if cached is not None:
            return cached

        # no-metal core: MLX's batched primitives beat the B=1-shaped
        # custom kernels at B >= 4 (270 vs 160 aggregate at B=8)
        core = self._make_core_fn(batched=True, metal=False)
        sample = _make_sampler_b(*samp_key)

        def one_token(x_tok, write_pos, m, kvs, counts, ring, ring_pos, ban_bias, key):
            row, new_m, new_kv = core(x_tok, write_pos, m, kvs)   # (B, V)
            row = row + ban_bias
            tok, key = sample(row, counts, key)                   # (B,)
            counts, ring, ring_pos = _push_window_b(tok, counts, ring, ring_pos)
            return tok, write_pos + 1, new_m, new_kv, counts, ring, ring_pos, key

        def step(x_tok, write_pos, m, kvs, counts, ring, ring_pos, ban_bias, key):
            toks = []
            for _ in range(unroll):
                tok, write_pos, m, kvs, counts, ring, ring_pos, key = one_token(
                    x_tok, write_pos, m, kvs, counts, ring, ring_pos, ban_bias, key)
                toks.append(tok)
                x_tok = tok.reshape(-1, 1)
            return mx.stack(toks), write_pos, m, kvs, counts, ring, ring_pos, key

        compiled = mx.compile(step)
        self._steps[key] = compiled
        return compiled

    # ── public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        prompt_ids: list[int],
        gen_config,
        *,
        stop_token_ids: list[int] | tuple = (),
        on_token: Callable[[int], None] | None = None,
        unroll: int = 4,
        ban_bias: mx.array | None = None,
    ) -> StaticGenerateResult:
        """Autoregressive generation through the single compiled step graph.

        gen_config: utils.config.GenerationConfig (or any object with the same
        fields).  ``unroll`` tokens are produced per compiled call; stop tokens
        are honoured with ≤ unroll tokens of (discarded) overshoot.
        """
        model = self._model
        V = model.config.vocab_size
        W = max(1, int(gen_config.repeat_last_n))
        max_tokens = int(gen_config.max_tokens)
        samp_key = (
            float(gen_config.temperature), int(gen_config.top_k),
            float(gen_config.top_p), float(gen_config.min_p),
            float(gen_config.rep_pen), float(gen_config.pres_pen),
            float(gen_config.freq_pen),
        )
        step_fn = self._get_step(unroll, samp_key)
        sample = _make_sampler(*samp_key)
        stop_set = set(int(t) for t in (stop_token_ids or ()))

        # ── prefill through the untouched reference path ─────────────────────
        ids = mx.array(prompt_ids, dtype=mx.int32)[None, :]
        t0 = time.perf_counter()
        logits, states = model(ids, states=None)
        last_row = logits[0, -1].astype(mx.float32)
        state_arrays = [v for st in states if st is not None
                        for v in st.values() if v is not None]
        mx.eval(last_row, *state_arrays)
        elapsed_prefill = time.perf_counter() - t0
        n_prompt = len(prompt_ids)

        # ── convert per-layer dict states → flat static buffers ─────────────
        S = -(-(n_prompt + max_tokens + unroll + 1) // self._kv_round) * self._kv_round
        m_flat: list[mx.array] = []
        kvs: list[mx.array] = []
        for blk, st in zip(model.backbone.layers, states):
            if isinstance(blk, Mamba3Block):
                m_flat += [st["h_prev"], st["prev_input_signal"], st["angles_cum"]]
            else:
                pad = S - st["k"].shape[2]
                kvs.append(mx.pad(st["k"], ((0, 0), (0, 0), (0, pad), (0, 0))))
                kvs.append(mx.pad(st["v"], ((0, 0), (0, 0), (0, pad), (0, 0))))
        counts = mx.zeros((V,), dtype=mx.float32)
        ring = mx.full((W,), -1, dtype=mx.int32)
        ring_pos = mx.zeros((1,), dtype=mx.int32)
        write_pos = mx.array([n_prompt], dtype=mx.int32)
        bb = (ban_bias.astype(mx.float32) if ban_bias is not None
              else mx.zeros((V,), dtype=mx.float32))
        key = mx.random.key(int(gen_config.seed))
        mx.eval(*kvs, counts, ring, ring_pos, write_pos, bb)

        # ── one-time trace+compile (functional: inputs unchanged, outputs dropped)
        t_c = time.perf_counter()
        warm = step_fn(mx.zeros((1, 1), dtype=mx.int32), write_pos, m_flat, kvs,
                       counts, ring, ring_pos, bb, key)
        mx.eval(warm[0])
        compile_s = time.perf_counter() - t_c

        # ── decode ───────────────────────────────────────────────────────────
        emitted: list[int] = []
        overshoot = 0
        stop_reason = "max_tokens"
        stop_hit = False

        def drain(tok_arr) -> None:
            nonlocal overshoot, stop_reason, stop_hit
            for tid in tok_arr.tolist():
                tid = int(tid)
                if stop_hit or len(emitted) >= max_tokens:
                    overshoot += 1
                    continue
                emitted.append(tid)
                if on_token is not None:
                    on_token(tid)
                if tid in stop_set:
                    stop_reason = "eos"
                    stop_hit = True

        t_dec = time.perf_counter()

        # First token: sampled eagerly from the prefill logits with the exact
        # same closure the graph uses (counts is all-zero → penalties no-op).
        tok0, key = sample(last_row + bb, counts, key)
        counts, ring, ring_pos = _push_window(tok0, counts, ring, ring_pos)
        tid0 = int(tok0.item())
        emitted.append(tid0)
        if on_token is not None:
            on_token(tid0)
        if tid0 in stop_set:
            stop_reason = "eos"
            stop_hit = True

        x_tok = tok0.reshape(1, 1)
        planned = 1
        pending = None                       # one call kept in flight (GPU↔Python overlap)
        while planned < max_tokens and not stop_hit:
            toks, write_pos, m_flat, kvs, counts, ring, ring_pos, key = step_fn(
                x_tok, write_pos, m_flat, kvs, counts, ring, ring_pos, bb, key)
            x_tok = toks[-1].reshape(1, 1)
            mx.async_eval(toks)
            if pending is not None:
                drain(pending)               # runs while the GPU executes this call
            pending = toks
            planned += unroll
        if pending is not None:
            drain(pending)

        elapsed_decode = time.perf_counter() - t_dec
        return StaticGenerateResult(
            tokens=emitted,
            stop_reason=stop_reason,
            n_prompt=n_prompt,
            elapsed_prefill=elapsed_prefill,
            elapsed_decode=elapsed_decode,
            prefill_tps=n_prompt / max(elapsed_prefill, 1e-9),
            decode_tps=len(emitted) / max(elapsed_decode, 1e-9),
            compile_s=compile_s,
            overshoot=overshoot,
        )

    def generate_batch(
        self,
        prompt_ids: list[int],
        gen_config,
        batch_size: int,
        *,
        stop_token_ids: list[int] | tuple = (),
        unroll: int = 1,
    ) -> tuple[list[list[int]], float]:
        """B independent sampled streams of the SAME prompt in one compiled
        graph (best-of-N / serving / data generation).  Returns
        (per_row_tokens, aggregate_decode_tps).  Rows stop individually on a
        stop token (their tails are dropped); decode runs until every row is
        done or max_tokens."""
        model = self._model
        B = int(batch_size)
        V = model.config.vocab_size
        W = max(1, int(gen_config.repeat_last_n))
        max_tokens = int(gen_config.max_tokens)
        samp_key = (
            float(gen_config.temperature), int(gen_config.top_k),
            float(gen_config.top_p), float(gen_config.min_p),
            float(gen_config.rep_pen), float(gen_config.pres_pen),
            float(gen_config.freq_pen),
        )
        step_fn = self._get_step_batch(unroll, samp_key)
        stop_set = set(int(t) for t in (stop_token_ids or ()))

        ids = mx.array([prompt_ids] * B, dtype=mx.int32)           # (B, L)
        logits, states = model(ids, states=None)
        state_arrays = [v for st in states if st is not None
                        for v in st.values() if v is not None]
        mx.eval(logits[:, -1], *state_arrays)
        n_prompt = len(prompt_ids)

        S = -(-(n_prompt + max_tokens + unroll + 1) // self._kv_round) * self._kv_round
        m_flat: list[mx.array] = []
        kvs: list[mx.array] = []
        for blk, st in zip(model.backbone.layers, states):
            if isinstance(blk, Mamba3Block):
                m_flat += [st["h_prev"], st["prev_input_signal"], st["angles_cum"]]
            else:
                pad = S - st["k"].shape[2]
                kvs.append(mx.pad(st["k"], ((0, 0), (0, 0), (0, pad), (0, 0))))
                kvs.append(mx.pad(st["v"], ((0, 0), (0, 0), (0, pad), (0, 0))))
        counts = mx.zeros((B, V), dtype=mx.float32)
        ring = mx.full((B, W), -1, dtype=mx.int32)
        ring_pos = mx.zeros((1,), dtype=mx.int32)
        write_pos = mx.array([n_prompt], dtype=mx.int32)
        bb = mx.zeros((V,), dtype=mx.float32)
        key = mx.random.key(int(gen_config.seed))
        mx.eval(*kvs, counts, ring, ring_pos, write_pos, bb)

        # trace+compile outside the timed region
        warm = step_fn(mx.zeros((B, 1), dtype=mx.int32), write_pos, m_flat, kvs,
                       counts, ring, ring_pos, bb, key)
        mx.eval(warm[0])

        rows: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B

        t0 = time.perf_counter()
        pending = None

        def drain(tok_mat) -> None:
            for tok_row in tok_mat.tolist():                       # (unroll, B)
                for b, tid in enumerate(tok_row):
                    if done[b] or len(rows[b]) >= max_tokens:
                        continue
                    rows[b].append(int(tid))
                    if int(tid) in stop_set:
                        done[b] = True

        # first token: sampled eagerly from the prefill logits (same closure)
        sample_b = _make_sampler_b(*samp_key)
        tok0, key = sample_b(logits[:, -1].astype(mx.float32) + bb, counts, key)
        counts, ring, ring_pos = _push_window_b(tok0, counts, ring, ring_pos)
        drain(tok0.reshape(1, B))
        x_tok = tok0.reshape(B, 1)

        steps = -(-(max_tokens - 1) // unroll)
        for _ in range(steps):
            out = step_fn(x_tok, write_pos, m_flat, kvs,
                          counts, ring, ring_pos, bb, key)
            toks, write_pos, m_flat, kvs, counts, ring, ring_pos, key = out
            mx.async_eval(toks)
            if pending is not None:
                drain(pending)
                if all(done):
                    break
            pending = toks
            x_tok = toks[-1].reshape(B, 1)
        if pending is not None and not all(done):
            drain(pending)
        elapsed = time.perf_counter() - t0
        n_total = sum(len(r) for r in rows)
        return rows, n_total / max(elapsed, 1e-9)

    def start_stream(self, states, n_prompt: int, *,
                     max_new: int = 512) -> "StaticStreamSession":
        """Open a per-token streaming session over the compiled forward step.

        ``states``: per-layer state dicts from a normal (chunked) prefill via
        ``model(ids, states=...)``; ``n_prompt``: absolute position of the
        next token to be written.  See StaticStreamSession.
        """
        return StaticStreamSession(self, states, n_prompt, max_new=max_new)


class StaticStreamSession:
    """Per-token streaming server for UIs/middleware that own the sampling.

    chat_demo's CoT middleware needs to (a) inspect/mask the logits row each
    token, (b) run its own penalties/sampler in Python, and (c) inject token
    sequences mid-stream (<final> forcing) — incompatible with the in-graph
    sampler of StaticDecoder.generate().  This session keeps that token loop
    untouched and only replaces the model forward: embed → 30 blocks → head
    runs as ONE compiled dispatch over static KV buffers, so metal_fuse /
    quant / norm_fold all apply.

        sess = decoder.start_stream(states, len(prompt_ids), max_new=512)
        row  = sess.step(tid)        # feed ANY token id → next logits row
    """

    def __init__(self, decoder: StaticDecoder, states, n_prompt: int,
                 max_new: int = 512):
        self._dec = decoder
        self._fwd = decoder._get_fwd_step()
        model = decoder._model
        kvr = decoder._kv_round
        S = -(-(n_prompt + max_new + 1) // kvr) * kvr
        self._m: list[mx.array] = []
        self._kvs: list[mx.array] = []
        for blk, st in zip(model.backbone.layers, states):
            if isinstance(blk, Mamba3Block):
                self._m += [st["h_prev"], st["prev_input_signal"], st["angles_cum"]]
            else:
                pad = S - st["k"].shape[2]
                self._kvs.append(mx.pad(st["k"], ((0, 0), (0, 0), (0, pad), (0, 0))))
                self._kvs.append(mx.pad(st["v"], ((0, 0), (0, 0), (0, pad), (0, 0))))
        self._write_pos = mx.array([n_prompt], dtype=mx.int32)
        self._cap = S
        self._pos = n_prompt
        mx.eval(*self._m, *self._kvs)

    @property
    def pos(self) -> int:
        """Absolute position of the next write (prompt + consumed tokens)."""
        return self._pos

    def step(self, tid: int) -> mx.array:
        """Advance the model by one token; returns the (V,) float32 logits
        row for the NEXT position (lazy — kicked off with async_eval so the
        GPU works while the caller does its Python-side bookkeeping)."""
        if self._pos + 1 >= self._cap:        # grow KV bucket (one retrace)
            kvr = self._dec._kv_round
            self._kvs = [mx.pad(t, ((0, 0), (0, 0), (0, kvr), (0, 0)))
                         for t in self._kvs]
            self._cap += kvr
        x = mx.array([[int(tid)]], dtype=mx.int32)
        row, self._write_pos, self._m, self._kvs = self._fwd(
            x, self._write_pos, self._m, self._kvs)
        self._pos += 1
        mx.async_eval(row)
        return row
