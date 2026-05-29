"""Metal-kernel accelerated verify forward for the speculative decoder.

Twin of :mod:`speculative.forward` whose only difference is that the
intra-block SSM scan is performed by the Metal kernel
:func:`mlx_model_v2.scan_metal.chunk_scan_per_pos` instead of the pure-MLX
``_scan_per_pos`` (which builds an O(Lc²) dense exponential decay matrix
``M`` and einsums it against ``u``).

For the verify path with ``shrink_chunk=True`` and K up to ~16, ``Lc=K``;
``M`` is a few hundred entries and the FLOP count is trivial, but each MLX
op carries non-trivial Python + dispatch overhead.  The Metal kernel folds
the scan into a single GPU dispatch per layer and reuses the existing
intra/inter chunk kernels from v1 verbatim, so numerics match the in-tree
``_scan_per_pos`` within bf16 quantisation noise (see correctness tests in
``mlx_model_v2/scan_metal.py``).

Public entry points:

* :func:`mamba_verify_step_metal` — drop-in replacement for
  :func:`mamba3_mlx.speculative.forward.mamba_verify_step`.
* :func:`model_verify_forward_metal` — drop-in replacement for
  :func:`mamba3_mlx.speculative.forward.model_verify_forward`.

Both share the same return shapes as the pure-MLX siblings so they can be
swapped in via ``jacobi_decode(..., metal_verify=True)``.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx

from ..mlx_model.mamba_block import Mamba3Block
from ..mlx_model.ops import apply_rope, scaled_tanh, silu, softplus
from ..mlx_model_v2.scan_metal import chunk_scan_per_pos
from .forward import _MAMBA_BLOCK_TYPES


def mamba_verify_step_metal(blk: Mamba3Block, x, state=None,
                             shrink_chunk: bool = True):
    """Metal-accelerated counterpart to ``forward.mamba_verify_step``.

    Identical math at every position; the only difference is the SSM scan
    is performed by :func:`chunk_scan_per_pos` instead of ``_scan_per_pos``.
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
                   * theta_h[None, None, :, :])             # (B, L, H, N//2)

    if state is None or state.get("angles_cum") is None:
        angles_cum_seq = mx.cumsum(delta_angle, axis=1)
    else:
        prev_cum = state["angles_cum"].astype(mx.float32)
        angles_cum_seq = mx.cumsum(delta_angle, axis=1) + prev_cum[:, None, :, :]
    angles_cum_per_pos = angles_cum_seq                     # (B, L, H, N//2)
    angles = angles_cum_seq.astype(x.dtype)

    B_p, C_p = blk._prepare_BC(B_param, C_param, B_sz, L)
    B_rot = apply_rope(B_p, angles)
    C_rot = apply_rope(C_p, angles)

    x_up = blk.x_up_proj(x_prime_hp.reshape(B_sz, L, H * P))
    x_ssm = x_up.reshape(B_sz, L, H, P, R)

    input_signal = mx.einsum("blhnr,blhpr->blhnp", B_rot, x_ssm)
    prev_input_signal_per_pos = input_signal                # (B, L, H, N, P)

    lv = mx.sigmoid(blk._broadcast_groups(lam, axis=-1)).reshape(B_sz, L, H, 1, 1).astype(x.dtype)
    dv = dt_b.reshape(B_sz, L, H, 1, 1).astype(x.dtype)
    av = mx.exp(dt_b * A_b).reshape(B_sz, L, H, 1, 1).astype(x.dtype)

    if state is None or state.get("prev_input_signal") is None:
        zero_pad = mx.zeros_like(input_signal[:, :1])
        if L > 1:
            ip = mx.concatenate([zero_pad, input_signal[:, :-1]], axis=1)
        else:
            ip = zero_pad
    else:
        prev_inp = state["prev_input_signal"].astype(input_signal.dtype)
        if L > 1:
            ip = mx.concatenate([prev_inp[:, None], input_signal[:, :-1]], axis=1)
        else:
            ip = prev_inp[:, None]

    u_ssm = lv * dv * input_signal + (1.0 - lv) * dv * av * ip

    h_init = state["h_prev"] if state is not None else None
    if L == 1:
        alpha = av[:, 0]                                    # (B, H, 1, 1)
        h_prev_init = (mx.zeros((B_sz, H, N, P), dtype=u_ssm.dtype)
                       if h_init is None else h_init.astype(u_ssm.dtype))
        h_new = alpha * h_prev_init + u_ssm[:, 0]           # (B, H, N, P)
        y_stack = mx.einsum("bhnp,bhnr->bhpr", h_new, C_rot[:, 0])[:, None]
        h_prev_per_pos = h_new[:, None]                     # (B, 1, H, N, P)
    else:
        # Metal-kernel scan: same shrink-chunk policy as the MLX path —
        # when ``L <= blk.chunk_size`` we run a single chunk of size L so
        # the kernel grid doesn't pad up to 64.
        co = L if (shrink_chunk and L <= blk.chunk_size) else blk.chunk_size
        la = (dt_b * A_b).astype(mx.float32)
        y_stack, h_prev_per_pos = chunk_scan_per_pos(
            u_ssm, la, C_rot, co, h_init=h_init,
        )

    y = blk.y_down_proj(y_stack.reshape(B_sz, L, H, P * R)).reshape(B_sz, L, H * P)
    D_expand = mx.repeat(blk.D, P, axis=0).astype(x.dtype)
    y = y + x_prime.reshape(B_sz, L, H * P) * D_expand
    mamba_out = blk.mamba_dense_proj(blk.pre_gate_norm(y) * silu(z))
    mid = residual_mamba + blk.ls_mamba(mamba_out)

    normed_mid = blk.norm_out_proj(mid)
    proj_out = blk.out_proj(normed_mid)
    out = mid + blk.ls_out_proj(proj_out)

    return out, {
        "h_prev": h_prev_per_pos,                           # (B, L, H, N, P)
        "prev_input_signal": prev_input_signal_per_pos,     # (B, L, H, N, P)
        "angles_cum": angles_cum_per_pos,                   # (B, L, H, N//2)
    }


def model_verify_forward_metal(model, input_ids, states=None):
    """Metal-accelerated counterpart to ``forward.model_verify_forward``.

    Same per-position payload structure as the MLX path so
    :func:`speculative.forward.extract_state_at` works unchanged.
    """
    x = model.embed(input_ids)
    payload: list[dict] = []

    for i, blk in enumerate(model.backbone.layers):
        st = states[i] if states is not None else None
        if isinstance(blk, _MAMBA_BLOCK_TYPES):
            x, sp = mamba_verify_step_metal(blk, x, st)
            payload.append({"kind": "mamba", **sp})
        else:
            S_past = 0
            if st is not None and st.get("k") is not None:
                S_past = int(st["k"].shape[2])
            x, new_st = blk(x, state=st)
            payload.append({
                "kind": "tf",
                "k": new_st["k"],
                "v": new_st["v"],
                "S_past": S_past,
            })

    h = model.norm(x)
    logits = model.head(h * model.inv_sqrt_d).astype(mx.float32)
    logits = scaled_tanh(logits, 30.0)
    return logits, payload
