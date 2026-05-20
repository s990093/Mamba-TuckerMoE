import math

import mlx.core as mx
import mlx.nn as nn

from .ops import RMSNorm, LayerScale, silu, softplus, apply_rope
from .tucker_moe import TuckerMoE


class Mamba3Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_in = config.d_model
        H = config.n_heads
        G = config.n_groups
        P = config.d_head
        N = config.d_state
        R = config.mimo_rank

        self.H, self.G, self.P, self.N, self.R = H, G, P, N, R
        self.ratio = H // G
        self.dim_z = H * P
        self.dim_x = H * P
        self.dim_B = G * N * R
        self.dim_C = G * N * R
        self.dim_dt = G
        self.dim_A = G
        self.dim_lambda = G

        in_out = (self.dim_z + self.dim_x + self.dim_B + self.dim_C
                  + self.dim_dt + self.dim_A + self.dim_lambda)
        self.in_proj = nn.Linear(d_in, in_out, bias=True)

        kw = dict(num_experts=config.kmoe_num_experts, top_k=config.kmoe_top_k,
                  r1=config.kmoe_r1, r2=config.kmoe_r2, r3=config.kmoe_r3,
                  eps=config.rms_norm_eps)
        self.x_up_proj = TuckerMoE(H * P, H * P * R, **kw)
        self.out_proj = TuckerMoE(d_in, d_in, **kw)

        self.y_down_proj = nn.Linear(P * R, P, bias=False)
        self.theta_log = mx.zeros((G, N // 2))
        self.D = mx.ones((H,))
        self.norm_B = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.norm_C = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.bias_B = mx.ones((G, N, R))
        self.bias_C = mx.ones((G, N, R))
        self.mamba_dense_proj = nn.Linear(config.d_inner, d_in, bias=False)
        self.pre_gate_norm = RMSNorm(H * P, eps=config.rms_norm_eps)
        self.norm_mamba = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm_out_proj = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ls_mamba = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_out_proj = LayerScale(config.d_model, init_value=config.layer_scale_init)

        self.chunk_size = config.chunk_size

    def _broadcast_groups(self, t, axis: int = -1):
        """Repeat along ``axis`` by ratio = H/G. Mirrors torch repeat_interleave."""
        if self.ratio == 1:
            return t
        return mx.repeat(t, self.ratio, axis=axis)

    def _split_inproj(self, raw, B_sz, L):
        H, P, G, N, R = self.H, self.P, self.G, self.N, self.R
        offs = 0
        z = raw[..., offs:offs + self.dim_z]; offs += self.dim_z
        x_prime = raw[..., offs:offs + self.dim_x]; offs += self.dim_x
        B_param = raw[..., offs:offs + self.dim_B]; offs += self.dim_B
        C_param = raw[..., offs:offs + self.dim_C]; offs += self.dim_C
        dt_p = raw[..., offs:offs + self.dim_dt]; offs += self.dim_dt
        A_p = raw[..., offs:offs + self.dim_A]; offs += self.dim_A
        lam = raw[..., offs:offs + self.dim_lambda]
        return z, x_prime, B_param, C_param, dt_p, A_p, lam

    def _prepare_BC(self, B_param, C_param, B_sz, L):
        G, N, R = self.G, self.N, self.R
        B_p = self.norm_B(B_param.reshape(B_sz, L, G, N * R)).reshape(B_sz, L, G, N, R)
        B_p = B_p + self.bias_B.astype(B_p.dtype)
        C_p = self.norm_C(C_param.reshape(B_sz, L, G, N * R)).reshape(B_sz, L, G, N, R)
        C_p = C_p + self.bias_C.astype(C_p.dtype)
        # Broadcast groups -> H
        B_p = self._broadcast_groups(B_p, axis=2)   # (B, L, H, N, R)
        C_p = self._broadcast_groups(C_p, axis=2)
        return B_p, C_p

    def _chunk_parallel_scan(self, u, dt_b, A_b, C_rot, h_init=None):
        """Forward chunk scan. Returns (y, h_final) where
        u:     (B, L, H, N, P)
        dt_b:  (B, L, H)
        A_b:   (B, L, H)
        C_rot: (B, L, H, N, R)
        h_init: (B, H, N, P) or None
        """
        B, L, H, N, P = u.shape
        R = C_rot.shape[-1]
        Lc = self.chunk_size
        L_orig = L
        pad = (Lc - L % Lc) % Lc
        if pad:
            u = mx.pad(u, [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
            dt_b = mx.pad(dt_b, [(0, 0), (0, pad), (0, 0)])
            A_b = mx.pad(A_b, [(0, 0), (0, pad), (0, 0)])
            C_rot = mx.pad(C_rot, [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
            L = L + pad
        nc = L // Lc

        la = (dt_b * A_b).astype(mx.float32)        # (B, L, H)
        u_c = u.reshape(B, nc, Lc, H, N, P)
        la_c = la.reshape(B, nc, Lc, H)
        C_c = C_rot.reshape(B, nc, Lc, H, N, R)

        # Cumulative sum within each chunk: la_cum[c, l] = sum_{k<=l} la[c,k]
        la_cum = mx.cumsum(la_c, axis=2)             # (B, nc, Lc, H)
        # log_M[c, i, j] = la_cum[c, i] - la_cum[c, j]; only valid for j <= i.
        log_M = la_cum[:, :, :, None, :] - la_cum[:, :, None, :, :]    # (B, nc, Lc, Lc, H)
        # Mask BEFORE exp to avoid 0 * inf = NaN in the upper triangle.
        tri_bool = mx.tri(Lc, dtype=mx.bool_)         # (Lc, Lc) True on lower tri
        neg_inf = mx.array(-1e9, dtype=log_M.dtype)
        log_M = mx.where(tri_bool[None, None, :, :, None], log_M, neg_inf)
        M = mx.exp(log_M).astype(u.dtype)
        # h_intra: sum_j M[c, l, j] * u_c[c, j]
        h_intra = mx.einsum("bcijh,bcjhnp->bcihnp", M, u_c)

        y_diag = mx.einsum("bclhnp,bclhnr->bclhpr", h_intra, C_c)

        # Cross-chunk carry
        decay = mx.exp(mx.sum(la_c, axis=2))         # (B, nc, H)
        if h_init is None:
            h_prev = mx.zeros((B, H, N, P), dtype=u.dtype)
        else:
            h_prev = h_init.astype(u.dtype)
        h_inter_list = []
        for c in range(nc):
            h_inter_list.append(h_prev)
            decay_c = decay[:, c].reshape(B, H, 1, 1).astype(u.dtype)
            h_prev = h_prev * decay_c + h_intra[:, c, -1]
        h_inter = mx.stack(h_inter_list, axis=1)     # (B, nc, H, N, P)

        # Decay applied to carried h within each chunk
        cdec_scale = mx.exp(la_cum).astype(u.dtype)  # (B, nc, Lc, H)
        # c_dec[c, l] = C_c[c, l] * cdec_scale[c, l]
        c_dec = C_c * cdec_scale[..., None, None]
        y_off = mx.einsum("bchnp,bclhnr->bclhpr", h_inter, c_dec)

        y = (y_diag + y_off).reshape(B, L, H, P, R)
        if L_orig != L:
            y = y[:, :L_orig]
        return y, h_prev

    def __call__(self, x, state=None):
        """Forward.

        state (dict) for decode:
          - 'h_prev': (B, H, N, P)
          - 'prev_input_signal': (B, H, N, P)
          - 'angles_cum': (B, H, N//2)  -- cumulative RoPE phase up to (and including) last token

        Returns (y, new_state).
        """
        B_sz, L, _ = x.shape
        H, G, P, N, R = self.H, self.G, self.P, self.N, self.R
        residual_mamba = x
        u = self.norm_mamba(x)
        raw = self.in_proj(u)
        z, x_prime, B_param, C_param, dt_p, A_p, lam = self._split_inproj(raw, B_sz, L)

        x_prime_hp = x_prime.reshape(B_sz, L, H, P)
        dt = softplus(dt_p)
        A = -mx.exp(A_p)
        theta = mx.exp(self.theta_log.astype(mx.float32))           # (G, N//2)

        dt_b = self._broadcast_groups(dt, axis=-1)                  # (B, L, H)
        A_b = self._broadcast_groups(A, axis=-1)                    # (B, L, H)
        theta_h = self._broadcast_groups(theta, axis=0)             # (H, N//2)

        # RoPE phase
        # delta_angle[b, l, h, n2] = dt_b[b, l, h] * theta_h[h, n2]
        delta_angle = (dt_b.astype(mx.float32)[..., None]
                       * theta_h[None, None, :, :])                  # (B, L, H, N//2)

        if state is None or state.get("angles_cum") is None:
            angles_cum_seq = mx.cumsum(delta_angle, axis=1)           # (B, L, H, N//2)
        else:
            prev_cum = state["angles_cum"].astype(mx.float32)         # (B, H, N//2)
            angles_cum_seq = mx.cumsum(delta_angle, axis=1) + prev_cum[:, None, :, :]
        new_angles_cum = angles_cum_seq[:, -1]                       # (B, H, N//2)
        angles = angles_cum_seq.astype(x.dtype)

        # Prepare B, C rotated
        B_p, C_p = self._prepare_BC(B_param, C_param, B_sz, L)        # (B, L, H, N, R)
        B_rot = apply_rope(B_p, angles)
        C_rot = apply_rope(C_p, angles)

        # x_up via MoE
        x_up = self.x_up_proj(x_prime_hp.reshape(B_sz, L, H * P))    # (B, L, H*P*R)
        x_ssm = x_up.reshape(B_sz, L, H, P, R)

        # input_signal: (B, L, H, N, P) = einsum("blhnr,blhpr->blhnp", B_rot, x_ssm)
        input_signal = mx.einsum("blhnr,blhpr->blhnp", B_rot, x_ssm)

        # lv, dv, av blending
        lv = mx.sigmoid(self._broadcast_groups(lam, axis=-1)).reshape(B_sz, L, H, 1, 1).astype(x.dtype)
        dv = dt_b.reshape(B_sz, L, H, 1, 1).astype(x.dtype)
        av = mx.exp(dt_b * A_b).reshape(B_sz, L, H, 1, 1).astype(x.dtype)

        # ip = input_signal shifted right by 1
        if state is None or state.get("prev_input_signal") is None:
            zero_pad = mx.zeros_like(input_signal[:, :1])
            if L > 1:
                ip = mx.concatenate([zero_pad, input_signal[:, :-1]], axis=1)
            else:
                ip = zero_pad
        else:
            prev_inp = state["prev_input_signal"].astype(input_signal.dtype)  # (B, H, N, P)
            if L > 1:
                ip = mx.concatenate([prev_inp[:, None], input_signal[:, :-1]], axis=1)
            else:
                ip = prev_inp[:, None]
        new_prev_input_signal = input_signal[:, -1]                  # (B, H, N, P)

        u_ssm = lv * dv * input_signal + (1.0 - lv) * dv * av * ip

        # SSM scan
        h_init = state["h_prev"] if state is not None else None
        if L == 1:
            # single-step recurrence
            alpha = av[:, 0]                                          # (B, H, 1, 1)
            h_prev = mx.zeros((B_sz, H, N, P), dtype=u_ssm.dtype) if h_init is None else h_init.astype(u_ssm.dtype)
            h_new = alpha * h_prev + u_ssm[:, 0]                      # (B, H, N, P)
            y_stack = mx.einsum("bhnp,bhnr->bhpr", h_new, C_rot[:, 0])[:, None]   # (B, 1, H, P, R)
            new_h_prev = h_new
        else:
            y_stack, new_h_prev = self._chunk_parallel_scan(
                u_ssm, dt_b, A_b, C_rot, h_init=h_init)

        # y_stack: (B, L, H, P, R) -> (B, L, H, P*R) -> y_down_proj -> (B, L, H, P) -> (B, L, H*P)
        y = self.y_down_proj(y_stack.reshape(B_sz, L, H, P * R)).reshape(B_sz, L, H * P)
        D_expand = mx.repeat(self.D, P, axis=0).astype(x.dtype)
        y = y + x_prime.reshape(B_sz, L, H * P) * D_expand
        mamba_out = self.mamba_dense_proj(self.pre_gate_norm(y) * silu(z))
        mid = residual_mamba + self.ls_mamba(mamba_out)

        normed_mid = self.norm_out_proj(mid)
        proj_out = self.out_proj(normed_mid)
        out = mid + self.ls_out_proj(proj_out)

        new_state = {
            "h_prev": new_h_prev,
            "prev_input_signal": new_prev_input_signal,
            "angles_cum": new_angles_cum,
        }
        return out, new_state
