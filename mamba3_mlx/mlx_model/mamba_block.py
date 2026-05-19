# -*- coding: utf-8 -*-
import math
import mlx.core as mx
import mlx.nn as nn

from .ops import RMSNorm, LayerScale, apply_rope, silu, silu_gating, fast_scaled_tanh
from .tucker_moe import TuckerMoE


def _chunk_scan_intra(la_chunk, u_chunk):
    """
    Intra-chunk parallel scan using cumulative-product matrix method.

    la_chunk : (B, nc, Lc, H)    log-alpha per step
    u_chunk  : (B, nc, Lc, H, N, P)

    Returns h_intra : (B, nc, Lc, H, N, P)
    Each element h_intra[b,c,t] = Σ_{k≤t} exp(cumsum[t] - cumsum[k]) * u[b,c,k]
    """
    B, nc, Lc, H = la_chunk.shape
    N = u_chunk.shape[-2]
    P = u_chunk.shape[-1]

    log_cum = mx.cumsum(la_chunk, axis=2)           # (B, nc, Lc, H)

    # Lower-triangular transition matrix
    # M[t, k] = exp(log_cum[t] - log_cum[k])  for k ≤ t
    log_cum_t = log_cum[:, :, :, None, :]           # (B, nc, Lc, 1, H)
    log_cum_k = log_cum[:, :, None, :, :]           # (B, nc, 1, Lc, H)
    log_M = log_cum_t - log_cum_k                   # (B, nc, Lc, Lc, H)

    # Apply lower-triangular mask
    tril = mx.tril(mx.ones((Lc, Lc), dtype=mx.bool_))
    neg_inf = mx.full((1,), -1e30, dtype=log_M.dtype)
    log_M = mx.where(tril[None, None, :, :, None], log_M, neg_inf)
    M = mx.exp(log_M)                               # (B, nc, Lc, Lc, H)

    # Transpose to (B, nc, H, Lc, Lc) for batched matmul
    M = M.transpose(0, 1, 4, 2, 3)

    # u_chunk → (B, nc, H, Lc, N*P)
    u_flat = u_chunk.transpose(0, 1, 3, 2, 4, 5).reshape(B, nc, H, Lc, N * P)

    # h_intra_flat : (B, nc, H, Lc, N*P)
    h_intra_flat = M @ u_flat

    # Reshape back
    return h_intra_flat.reshape(B, nc, H, Lc, N, P).transpose(0, 1, 3, 2, 4, 5)


def chunk_parallel_scan(u_ssm, la, C, chunk_size=64):
    """
    Full chunk-wise parallel scan (prefill, arbitrary L).

    u_ssm  : (B, L, H, N, P)   SSM input signal
    la     : (B, L, H)          log-alpha = dt_b * A_b
    C      : (B, L, H, N, R)    output projection (rotated)
    chunk_size: int

    Returns:
        y      : (B, L, H, P, R)
        h_last : (B, H, N, P)   final hidden state for decode continuation
    """
    B, L, H, N, P = u_ssm.shape
    R = C.shape[-1]
    L_orig = L

    # Pad to multiple of chunk_size
    if L % chunk_size != 0:
        pad = chunk_size - (L % chunk_size)
        u_ssm = mx.pad(u_ssm, [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
        la = mx.pad(la,    [(0, 0), (0, pad), (0, 0)])
        C = mx.pad(C,      [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
        L = L + pad

    nc = L // chunk_size
    u_c  = u_ssm.reshape(B, nc, chunk_size, H, N, P)
    la_c = la.reshape(B, nc, chunk_size, H)
    C_c  = C.reshape(B, nc, chunk_size, H, N, R)

    # Intra-chunk scan
    h_intra = _chunk_scan_intra(la_c, u_c)          # (B, nc, Lc, H, N, P)

    # Diagonal contribution: y_diag[b,c,l,h,p,r] = Σ_n h_intra[b,c,l,h,n,p] * C_c[b,c,l,h,n,r]
    y_diag = mx.einsum("bclhnp, bclhnr -> bclhpr", h_intra, C_c)

    # Inter-chunk decay
    decay = mx.exp(la_c.sum(axis=2))                # (B, nc, H)

    # Accumulate h_prev across chunks
    h_prev = mx.zeros((B, H, N, P), dtype=u_ssm.dtype)
    h_inter_chunks = []
    for c in range(nc):
        h_inter_chunks.append(h_prev)
        h_prev = h_prev * decay[:, c, :, None, None] + h_intra[:, c, -1]

    h_inter = mx.stack(h_inter_chunks, axis=1)      # (B, nc, H, N, P)

    # Off-diagonal contribution from previous chunk state
    # c_dec[b,c,l,h,n,r] = C_c[b,c,l,h,n,r] * exp(cumsum(la_c)[b,c,l,h])
    cum_la = mx.cumsum(la_c, axis=2)                # (B, nc, Lc, H)
    c_dec = C_c * mx.exp(cum_la)[:, :, :, :, None, None]  # (B, nc, Lc, H, N, R)

    y_off = mx.einsum("bchnp, bclhnr -> bclhpr", h_inter, c_dec)

    y = (y_diag + y_off).reshape(B, L, H, P, R)

    # Trim padding
    if L_orig < L:
        y = y[:, :L_orig]

    return y, h_prev


def single_step_scan(u_t, la_t, h_prev):
    """
    Single decode step: h_new = exp(la_t) * h_prev + u_t.

    u_t    : (B, H, N, P)
    la_t   : (B, H)           log-alpha at this step
    h_prev : (B, H, N, P)

    Returns h_new : (B, H, N, P)
    """
    alpha = mx.exp(la_t)[:, :, None, None]           # (B, H, 1, 1)
    return alpha * h_prev + u_t


class Mamba3Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_in = config.d_model
        H = config.n_heads          # 24
        G = config.n_groups         # 1
        P = config.d_head           # 64
        N = config.d_state          # 64
        R = config.mimo_rank        # 4

        self.H, self.G, self.P, self.N, self.R = H, G, P, N, R
        self.ratio = H // G         # 24
        self.chunk_size = config.chunk_size

        dim_z = H * P               # 1536
        dim_x = H * P               # 1536
        dim_B = G * N * R           # 256
        dim_C = G * N * R           # 256
        dim_dt = G                  # 1
        dim_A = G                   # 1
        dim_lam = G                 # 1

        self.dim_z = dim_z
        self.dim_splits = [dim_z, dim_x, dim_B, dim_C, dim_dt, dim_A, dim_lam]
        total_proj = sum(self.dim_splits)  # 3587

        self.in_proj = nn.Linear(d_in, total_proj, bias=True)

        kw = dict(
            num_experts=config.kmoe_num_experts,
            top_k=config.kmoe_top_k,
            r1=config.kmoe_r1,
            r2=config.kmoe_r2,
            r3=config.kmoe_r3,
        )
        self.x_up_proj = TuckerMoE(H * P, H * P * R, **kw)      # (1536, 6144)
        self.out_proj   = TuckerMoE(d_in, d_in, **kw)            # (768, 768)

        self.y_down_proj      = nn.Linear(P * R, P, bias=False)  # (256→64)
        self.theta_log        = mx.zeros((G, N // 2))
        self.D                = mx.ones((H,))
        self.norm_B           = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.norm_C           = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.bias_B           = mx.ones((G, N, R))
        self.bias_C           = mx.ones((G, N, R))
        self.mamba_dense_proj = nn.Linear(config.d_inner, d_in, bias=False)  # (1536→768)
        self.pre_gate_norm    = RMSNorm(H * P)
        self.norm_mamba       = RMSNorm(d_in, eps=config.rms_norm_eps)
        self.norm_out_proj    = RMSNorm(d_in, eps=config.rms_norm_eps)
        self.ls_mamba         = LayerScale(d_in, init_value=config.layer_scale_init)
        self.ls_out_proj      = LayerScale(d_in, init_value=config.layer_scale_init)

    def _broadcast_to_heads(self, t):
        """Expand G-dim (axis=-1 level) to H-dim via repeat."""
        return mx.repeat(t, self.ratio, axis=2)

    def __call__(self, x, step=None, mamba_state=None):
        """
        Prefill forward (mamba_state=None) or decode forward (mamba_state provided).

        mamba_state: dict with keys:
          'h'       (B,H,N,P)   – Mamba hidden state
          'angles'  (B,H,N//2)  – cumulative RoPE angles
          'last_ip' (B,H,N,P)   – input_signal from the PREVIOUS step
                                   (needed for trapezoidal discretisation; ip coeff ≈ 0.95)
        Returns: (output, lb, z_loss, new_mamba_state)
        """
        B, L, _ = x.shape
        H, G, P, N, R, ratio = self.H, self.G, self.P, self.N, self.R, self.ratio

        residual_mamba = x
        u = self.norm_mamba(x)  # (B, L, d_model)

        # Project
        proj = self.in_proj(u)
        dim_z, dim_x, dim_B, dim_C, dim_dt, dim_A, dim_lam = self.dim_splits
        z, x_prime, B_param, C_param, dt_raw, A_param, lambda_param = (
            proj[..., :dim_z],
            proj[..., dim_z:dim_z+dim_x],
            proj[..., dim_z+dim_x:dim_z+dim_x+dim_B],
            proj[..., dim_z+dim_x+dim_B:dim_z+dim_x+dim_B+dim_C],
            proj[..., dim_z+dim_x+dim_B+dim_C:dim_z+dim_x+dim_B+dim_C+dim_dt],
            proj[..., dim_z+dim_x+dim_B+dim_C+dim_dt:dim_z+dim_x+dim_B+dim_C+dim_dt+dim_A],
            proj[..., dim_z+dim_x+dim_B+dim_C+dim_dt+dim_A:],
        )

        x_prime = x_prime.reshape(B, L, H, P)

        # dt, A per head
        dt = mx.log1p(mx.exp(dt_raw))           # softplus: (B, L, G)
        A = -mx.exp(A_param)                    # (B, L, G)  negative

        # theta: (G, N//2) → (H, N//2) after repeat
        theta = mx.exp(self.theta_log)                         # (G, N//2)
        theta_h = mx.repeat(theta, ratio, axis=0)              # (H, N//2)

        # Expand G→H for dt, A, lambda
        dt_b  = mx.repeat(dt,  ratio, axis=-1)         # (B, L, H)
        A_b   = mx.repeat(A,   ratio, axis=-1)         # (B, L, H)
        lam_b = mx.repeat(lambda_param, ratio, axis=-1)  # (B, L, H)

        # RoPE angles: cumsum of dt_b * theta_h along sequence
        angle_increments = mx.einsum("blh, hn -> blhn", dt_b, theta_h)  # (B,L,H,N//2)
        if mamba_state is not None:
            # Decode: start from saved cumulative angle
            angles_last = mamba_state["angles"]                # (B, H, N//2)
            cum_base = angles_last[:, None, :, :]              # (B, 1, H, N//2)
            angles = cum_base + mx.cumsum(angle_increments, axis=1)
        else:
            angles = mx.cumsum(angle_increments, axis=1)       # (B, L, H, N//2)

        # B and C parameters: norm → view → add bias → broadcast → RoPE
        B_normed = self.norm_B(B_param.reshape(B, L, G, N * R))          # (B,L,G,N*R)
        B_viewed = B_normed.reshape(B, L, G, N, R) + self.bias_B         # (B,L,G,N,R)
        B_broad  = mx.repeat(B_viewed, ratio, axis=2)                    # (B,L,H,N,R)
        B_rotated = apply_rope(B_broad, angles)                          # (B,L,H,N,R)

        C_normed = self.norm_C(C_param.reshape(B, L, G, N * R))
        C_viewed = C_normed.reshape(B, L, G, N, R) + self.bias_C
        C_broad  = mx.repeat(C_viewed, ratio, axis=2)
        C_rotated = apply_rope(C_broad, angles)

        # x_up_proj: TuckerMoE(H*P → H*P*R)
        x_up, lb_up, z_up = self.x_up_proj(x_prime.reshape(B, L, -1), step=step)
        x_ssm = x_up.reshape(B, L, H, P, R)                              # (B,L,H,P,R)

        # input_signal[b,l,h,n,p] = Σ_r B_rotated[b,l,h,n,r] * x_ssm[b,l,h,p,r]
        input_signal = mx.einsum("blhnr, blhpr -> blhnp", B_rotated, x_ssm)  # (B,L,H,N,P)

        # Gating coefficients
        lv = mx.sigmoid(lam_b).reshape(B, L, H, 1, 1)   # (B,L,H,1,1)
        dv = dt_b.reshape(B, L, H, 1, 1)
        av = mx.exp(dt_b * A_b).reshape(B, L, H, 1, 1)

        # Trapezoidal discretisation: u_ssm[t] = lv*dv*is[t] + (1-lv)*dv*av*is[t-1]
        # ip = input_signal shifted right by 1; coefficient (1-lv) ≈ 0.95 so this matters a lot.
        #
        # WARNING — L==1 branch: this correctly handles single-token decode.
        # If L > 1 with a mamba_state (e.g. chunked speculative verification),
        # the code falls through to the prefill path, which pads ip with zeros
        # instead of using last_ip — breaking state continuity.
        # Rule: speculative verification MUST use stateless prefill (no mamba_state)
        # until this block is refactored to support stateful L>1 decode.
        if L == 1 and mamba_state is not None:
            # Decode: use the stored previous input_signal from mamba_state
            last_ip = mamba_state["last_ip"]                   # (B, H, N, P)
            ip = last_ip[:, None, :, :, :]                     # (B, 1, H, N, P)
        else:
            # Prefill (or stateless multi-token): shift right by 1, pad start with zeros
            ip = mx.pad(input_signal[:, :-1], [(0,0),(1,0),(0,0),(0,0),(0,0)])  # (B,L,H,N,P)

        u_ssm = lv * dv * input_signal + (1.0 - lv) * dv * av * ip       # (B,L,H,N,P)

        # SSM scan
        la = dt_b * A_b                                                    # (B,L,H)
        if L == 1 and mamba_state is not None:
            # Decode: single recurrence step
            h_prev = mamba_state["h"]                                      # (B,H,N,P)
            la_t = la[:, 0, :]                                             # (B,H)
            u_t  = u_ssm[:, 0, :]                                         # (B,H,N,P)
            h_new = single_step_scan(u_t, la_t, h_prev)                   # (B,H,N,P)
            C_t = C_rotated[:, 0, :]                                       # (B,H,N,R)
            y_stack = mx.einsum("bhnp, bhnr -> bhpr", h_new, C_t)         # (B,H,P,R)
            y_stack = y_stack[:, None, :, :, :]                           # (B,1,H,P,R)
            h_final = h_new
        else:
            # Prefill: full chunk scan
            y_stack, h_final = chunk_parallel_scan(u_ssm, la, C_rotated, self.chunk_size)
            # y_stack: (B, L, H, P, R)

        # Down-project: (B, L, H, P*R) → (B, L, H*P)
        y = self.y_down_proj(y_stack.reshape(B, L, H, P * R)).reshape(B, L, H * P)

        # Skip connection with D
        D_expanded = mx.repeat(self.D, P)               # (H*P,)
        y = y + x_prime.reshape(B, L, H * P) * D_expanded

        # Gate and project
        mamba_out = self.mamba_dense_proj(
            self.pre_gate_norm(y) * silu(z)
        )  # (B, L, d_model)

        mid_x = residual_mamba + self.ls_mamba(mamba_out)

        # Output projection
        proj_out, lb_out, z_out = self.out_proj(self.norm_out_proj(mid_x), step=step)
        out = mid_x + self.ls_out_proj(proj_out)

        # Build new state — save last timestep's input_signal for next decode step
        new_state = {
            "h":       h_final,
            "angles":  angles[:, -1, :, :],          # (B, H, N//2)
            "last_ip": input_signal[:, -1, :, :, :], # (B, H, N, P) — previous step's B*x
        }

        return out, lb_up + lb_out, 0.0, new_state
