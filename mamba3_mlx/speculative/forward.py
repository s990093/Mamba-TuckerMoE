"""Per-position state verify forward — no replay needed.

The existing ``Mamba3Block.__call__`` advances state by L positions in one call
and only returns the *final* state (after all L tokens).  For Jacobi
verification, we additionally want the state at position ``m-1`` so that on a
partial accept (m < K) we can take that intermediate state directly instead of
running a separate replay forward.

This module provides:

* :func:`mamba_verify_step` — same math as ``Mamba3Block.__call__`` but
  returns ``h_prev``, ``prev_input_signal``, ``angles_cum`` as per-position
  tensors of shape ``(B, L, ...)``.
* :func:`model_verify_forward` — full-model wrapper.  Mamba layers use
  :func:`mamba_verify_step`; Transformer layers run their normal forward
  (the KV cache it returns is already per-position, just slice it).
* :func:`extract_state_at` — given the per-position payload returned by
  :func:`model_verify_forward`, build a normal per-layer state list
  representing the model *after m tokens*.

Reuses the existing ``Mamba3Block`` weight set and ``ops`` helpers.  No
existing file is modified.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx
 
from ..mlx_model.mamba_block import Mamba3Block

# Optional v2 (parallel-prefix scan) — keep the isinstance check working when
# the caller swaps the model implementation.  We can't import v2 at module
# import time (it may not exist), so build the tuple lazily.
try:
    from ..mlx_model_v2.mamba_block import Mamba3Block as _Mamba3Block_v2  # noqa: E501
    _MAMBA_BLOCK_TYPES: tuple = (Mamba3Block, _Mamba3Block_v2)
except Exception:  # pragma: no cover — v2 not installed
    _MAMBA_BLOCK_TYPES = (Mamba3Block,)
from ..mlx_model.ops import apply_rope, scaled_tanh, silu, softplus


# ── Mamba per-position scan ──────────────────────────────────────────────────

def _scan_per_pos(blk: Mamba3Block, u, dt_b, A_b, C_rot,
                  h_init: Optional[mx.array] = None,
                  chunk_size_override: Optional[int] = None):
    """Chunk-parallel scan that also returns per-position ``h``.

    For position ``l`` in chunk ``c`` the SSM state is

        h[c, l] = h_inter[c] * exp(la_cum[c, l]) + h_intra[c, l]

    where ``h_inter[c]`` is the state at the *start* of chunk ``c`` and
    ``h_intra[c, l] = sum_{j<=l} M[l, j] * u_c[c, j]``.

    ``chunk_size_override`` may be set (e.g. to ``L``) to avoid padding the
    scan up to ``blk.chunk_size`` — for the Jacobi verify path where L is
    small (8–32), this shrinks the ``(Lc × Lc)`` exp/einsum from 64×64 to
    L×L, a major fraction of the per-round cost.

    Returns
    -------
    y_stack    : (B, L_orig, H, P, R) — same as ``_chunk_parallel_scan``
    h_per_pos  : (B, L_orig, H, N, P) — SSM state *after* each position
    """
    B, L, H, N, P = u.shape
    R = C_rot.shape[-1]
    Lc = chunk_size_override if chunk_size_override is not None else blk.chunk_size
    L_orig = L
    pad = (Lc - L % Lc) % Lc
    if pad:
        u = mx.pad(u, [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
        dt_b = mx.pad(dt_b, [(0, 0), (0, pad), (0, 0)])
        A_b = mx.pad(A_b, [(0, 0), (0, pad), (0, 0)])
        C_rot = mx.pad(C_rot, [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
        L = L + pad
    nc = L // Lc

    la = (dt_b * A_b).astype(mx.float32)
    u_c = u.reshape(B, nc, Lc, H, N, P)
    la_c = la.reshape(B, nc, Lc, H)
    C_c = C_rot.reshape(B, nc, Lc, H, N, R)

    la_cum = mx.cumsum(la_c, axis=2)
    log_M = la_cum[:, :, :, None, :] - la_cum[:, :, None, :, :]
    tri_bool = mx.tri(Lc, dtype=mx.bool_)
    neg_inf = mx.array(-1e9, dtype=log_M.dtype)
    log_M = mx.where(tri_bool[None, None, :, :, None], log_M, neg_inf)
    M = mx.exp(log_M).astype(u.dtype)
    h_intra = mx.einsum("bcijh,bcjhnp->bcihnp", M, u_c)

    y_diag = mx.einsum("bclhnp,bclhnr->bclhpr", h_intra, C_c)

    decay = mx.exp(mx.sum(la_c, axis=2))
    if h_init is None:
        h_prev = mx.zeros((B, H, N, P), dtype=u.dtype)
    else:
        h_prev = h_init.astype(u.dtype)
    h_inter_list = []
    for c in range(nc):
        h_inter_list.append(h_prev)
        decay_c = decay[:, c].reshape(B, H, 1, 1).astype(u.dtype)
        h_prev = h_prev * decay_c + h_intra[:, c, -1]
    h_inter = mx.stack(h_inter_list, axis=1)  # (B, nc, H, N, P)

    cdec_scale = mx.exp(la_cum).astype(u.dtype)         # (B, nc, Lc, H)
    c_dec = C_c * cdec_scale[..., None, None]
    y_off = mx.einsum("bchnp,bclhnr->bclhpr", h_inter, c_dec)

    # Per-position SSM state.
    h_per_pos = (
        h_inter[:, :, None, :, :, :] * cdec_scale[:, :, :, :, None, None]
        + h_intra
    )                                                   # (B, nc, Lc, H, N, P)
    h_per_pos = h_per_pos.reshape(B, L, H, N, P)
    if L_orig != L:
        h_per_pos = h_per_pos[:, :L_orig]

    y = (y_diag + y_off).reshape(B, L, H, P, R)
    if L_orig != L:
        y = y[:, :L_orig]
    return y, h_per_pos


def mamba_verify_step(blk: Mamba3Block, x, state=None,
                      shrink_chunk: bool = True):
    """``Mamba3Block.__call__`` rewritten to expose per-position state.

    Same math, same outputs at every position; the only difference vs the
    in-tree ``__call__`` is the return shape of the state — instead of three
    ``(B, ...)`` tensors representing position ``L-1``, we return three
    ``(B, L, ...)`` tensors.

    ``shrink_chunk=True`` (default) forces the SSM chunk size down to ``L``
    instead of the model's full ``blk.chunk_size`` (typically 64).  This is
    safe and numerically identical when ``L <= blk.chunk_size`` (no
    cross-chunk carry needed) and cuts the dominant ``(Lc × Lc)`` einsum
    cost by ``(blk.chunk_size / L)^2``.
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
    # ``prev_input_signal_per_pos[:, l]`` is the value to use as
    # ``state['prev_input_signal']`` *if* we ended the model at position l.
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
        # When ``L <= blk.chunk_size`` we can shrink the scan to a single
        # chunk of size L instead of padding up to 64 — same math, much less
        # compute (the (Lc x Lc) einsum and exp dominate the verify cost).
        co = L if (shrink_chunk and L <= blk.chunk_size) else None
        y_stack, h_prev_per_pos = _scan_per_pos(
            blk, u_ssm, dt_b, A_b, C_rot, h_init=h_init,
            chunk_size_override=co,
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


# ── Whole-model verify wrapper ───────────────────────────────────────────────

def model_verify_forward(model, input_ids, states=None):
    """Run the full Mamba3LanguageModel with per-position state recording.

    Mirrors ``Mamba3LanguageModel.__call__`` exactly (same logits) but the
    second return value is a list of *per-position* payloads usable with
    :func:`extract_state_at`.

    Returns
    -------
    logits           : (B, L, V) float32
    per_pos_payload  : list, one entry per layer:
                        - mamba  → {"kind": "mamba", h_prev, prev_input_signal, angles_cum}
                        - tf     → {"kind": "tf", k, v, S_past}
    """
    x = model.embed(input_ids)
    payload: list[dict] = []

    for i, blk in enumerate(model.backbone.layers):
        st = states[i] if states is not None else None
        if isinstance(blk, _MAMBA_BLOCK_TYPES):
            x, sp = mamba_verify_step(blk, x, st)
            payload.append({"kind": "mamba", **sp})
        else:
            # Transformer KV cache already grows per-position; capture S_past
            # so the extractor can slice [:, :, :S_past + m, :].
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


def extract_state_at(payload: list[dict], m: int,
                     branch: int = 0) -> list[dict]:
    """Build a per-layer state list representing the model *after m tokens*.

    ``m`` is 1-indexed (m=1 → state after the first token).
    ``branch`` selects which row of the batched per-position payload to keep
    (default 0 for single-branch verify; used by tree-of-guesses to extract
    the winning branch and collapse the batch back to 1).
    """
    if m < 1:
        raise ValueError(f"extract_state_at requires m >= 1, got {m}")
    idx = m - 1
    b0, b1 = branch, branch + 1     # slice so result keeps batch dim = 1
    out: list[dict] = []
    for st in payload:
        if st["kind"] == "mamba":
            out.append({
                "h_prev": st["h_prev"][b0:b1, idx],
                "prev_input_signal": st["prev_input_signal"][b0:b1, idx],
                "angles_cum": st["angles_cum"][b0:b1, idx],
            })
        else:
            cutoff = st["S_past"] + m
            out.append({
                "k": st["k"][b0:b1, :, :cutoff, :],
                "v": st["v"][b0:b1, :, :cutoff, :],
            })
    return out


def replicate_state(state, num_branches: int):
    """Broadcast a batch-1 layer-state list across ``num_branches`` branches.

    Returns ``None`` if ``state`` is ``None`` (prefill of fresh prompt with no
    prior decoding).  For ``num_branches == 1`` the input is returned
    unchanged (no copy).
    """
    if state is None or num_branches == 1:
        return state
    out = []
    for st in state:
        if st is None:
            out.append(None)
            continue
        new_st = {}
        for k, v in st.items():
            if v is None:
                new_st[k] = None
            elif hasattr(v, "shape"):
                # ``mx.repeat`` broadcasts and materializes; that's fine since
                # the cost is dwarfed by the per-layer model forward.
                new_st[k] = mx.repeat(v, num_branches, axis=0)
            else:
                new_st[k] = v        # scalar / shape-less stash
        out.append(new_st)
    return out
