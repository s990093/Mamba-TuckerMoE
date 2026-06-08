import mlx.core as mx
import mlx.nn as nn

from .ops import RMSNorm, LayerScale, silu, softplus, apply_rope_with_sc
from .tucker_moe import TuckerMoE
from .scan_metal import chunk_scan


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

        # Precomputed constants (chunk_size is fixed at init time).
        # _tri_mask: causal lower-triangular boolean mask for the intra-chunk scan.
        self._tri_mask = mx.tri(config.chunk_size, dtype=mx.bool_)

        # _D_expand and _D_expand_dtype are populated by precompute().
        self._D_expand: mx.array | None = None

        # Compiled L=1 decode — set by precompute(), dispatched from __call__.
        self._compiled_decode: object | None = None

    def precompute(self) -> None:
        """Cache tensors that are constant after weight loading.

        Called by Mamba3LanguageModel.precompute() right after loading weights.
        """
        # D_expand is mx.repeat(self.D, P) — a tiny (H*P,) vector reused every decode step.
        self._D_expand = mx.repeat(self.D, self.P, axis=0)
        mx.eval(self._D_expand)

        # Compile the L=1 decode step. Eliminates ~60 Python MLX dispatch calls per
        # block call, since the entire decode graph is fused into one compiled program.
        # x_up_proj._forward / out_proj._forward are inlined (not re-compiled) so the
        # TuckerMoE ops are captured inside this larger compiled graph.
        self._compiled_decode = mx.compile(self._decode_impl)
        H, N, P = self.H, self.N, self.P
        d = self.config.d_model
        _dx  = mx.zeros((1, 1, d),          dtype=mx.bfloat16)
        _dh  = mx.zeros((1, H, N, P),       dtype=mx.bfloat16)
        _dip = mx.zeros((1, H, N, P),       dtype=mx.bfloat16)
        _dac = mx.zeros((1, H, N // 2),     dtype=mx.float32)
        mx.eval(self._compiled_decode(_dx, _dh, _dip, _dac))

    def _decode_impl(self, x, h_prev, prev_input_signal, angles_cum):
        """Compiled single-step (L=1) Mamba decode — no Python state conditionals.

        All arguments are concrete arrays (state always provided by caller).
        Returns (out, new_h_prev, new_prev_input_signal, new_angles_cum).

        Calls x_up_proj._forward / out_proj._forward directly so the TuckerMoE
        ops are inlined into this compiled graph rather than dispatched as
        separate compiled programs.
        """
        B_sz = x.shape[0]
        H, G, P, N, R = self.H, self.G, self.P, self.N, self.R
        residual_mamba = x                                        # (B, 1, d_model)
        u = self.norm_mamba(x)
        raw = self.in_proj(u)                                     # (B, 1, in_out)
        z, x_prime, B_param, C_param, dt_p, A_p, lam = self._split_inproj(raw, B_sz, 1)

        x_prime_hp = x_prime.reshape(B_sz, 1, H, P)
        dt = softplus(dt_p)                                       # (B, 1, G)
        A = -mx.exp(A_p)                                          # (B, 1, G)
        theta = mx.exp(self.theta_log.astype(mx.float32))         # (G, N//2)

        dt_b = self._broadcast_groups(dt, axis=-1)                # (B, 1, H)
        A_b  = self._broadcast_groups(A,  axis=-1)                # (B, 1, H)
        theta_h = self._broadcast_groups(theta, axis=0)           # (H, N//2)

        la_full = (dt_b * A_b).astype(mx.float32)                 # (B, 1, H)
        # For L=1, cumsum over a single element is identity — skip the cumsum.
        delta_angle = (dt_b.astype(mx.float32)[..., None]
                       * theta_h[None, None, :, :])               # (B, 1, H, N//2)
        angles_cum_seq = delta_angle + angles_cum[:, None, :, :].astype(mx.float32)
        new_angles_cum  = angles_cum_seq[:, 0]                    # (B, H, N//2)
        angles = angles_cum_seq.astype(x.dtype)                   # (B, 1, H, N//2)

        _sin_a = mx.sin(angles)[..., None].astype(x.dtype)        # (B, 1, H, N//2, 1)
        _cos_a = mx.cos(angles)[..., None].astype(x.dtype)

        B_p, C_p = self._prepare_BC(B_param, C_param, B_sz, 1)   # (B, 1, H, N, R)
        B_rot = apply_rope_with_sc(B_p, _sin_a, _cos_a)
        C_rot = apply_rope_with_sc(C_p, _sin_a, _cos_a)

        # Inline TuckerMoE ops into this compiled graph.
        x_up = self.x_up_proj._forward(x_prime_hp.reshape(B_sz, 1, H * P))
        x_ssm = x_up.reshape(B_sz, 1, H, P, R)

        input_signal = mx.einsum("blhnr,blhpr->blhnp", B_rot, x_ssm)  # (B, 1, H, N, P)

        lv = mx.sigmoid(self._broadcast_groups(lam, axis=-1)).reshape(B_sz, 1, H, 1, 1).astype(x.dtype)
        dv = dt_b.reshape(B_sz, 1, H, 1, 1).astype(x.dtype)
        av = mx.exp(la_full).reshape(B_sz, 1, H, 1, 1).astype(x.dtype)

        ip  = prev_input_signal[:, None].astype(input_signal.dtype)  # (B, 1, H, N, P)
        new_prev_input_signal = input_signal[:, 0]                    # (B, H, N, P)

        u_ssm = lv * dv * input_signal + (1.0 - lv) * dv * av * ip   # (B, 1, H, N, P)

        # Single-step SSM recurrence.
        alpha = av[:, 0]                                          # (B, H, 1, 1)
        h_new = alpha * h_prev.astype(u_ssm.dtype) + u_ssm[:, 0] # (B, H, N, P)
        y_stack = mx.einsum("bhnp,bhnr->bhpr", h_new, C_rot[:, 0])[:, None]  # (B, 1, H, P, R)

        y = self.y_down_proj(y_stack.reshape(B_sz, 1, H, P * R)).reshape(B_sz, 1, H * P)
        D_expand = self._D_expand.astype(x.dtype)
        y = y + x_prime.reshape(B_sz, 1, H * P) * D_expand

        mamba_out = self.mamba_dense_proj(self.pre_gate_norm(y) * silu(z))
        mid = residual_mamba + self.ls_mamba(mamba_out)

        normed_mid = self.norm_out_proj(mid)
        proj_out = self.out_proj._forward(normed_mid)
        out = mid + self.ls_out_proj(proj_out)

        return out, h_new, new_prev_input_signal, new_angles_cum

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

    def __call__(self, x, state=None):
        """Forward.

        state (dict) for decode:
          - 'h_prev': (B, H, N, P)
          - 'prev_input_signal': (B, H, N, P)
          - 'angles_cum': (B, H, N//2)  -- cumulative RoPE phase up to (and including) last token

        Returns (y, new_state).
        """
        B_sz, L, _ = x.shape

        # Fast path: compiled single-step decode (no Python overhead per block).
        if (L == 1 and state is not None
                and self._compiled_decode is not None
                and state.get("h_prev") is not None):
            out, new_h, new_ip, new_ac = self._compiled_decode(
                x,
                state["h_prev"],
                state["prev_input_signal"],
                state["angles_cum"],
            )
            return out, {
                "h_prev":             new_h,
                "prev_input_signal":  new_ip,
                "angles_cum":         new_ac,
            }
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

        # Precompute la = dt_b * A_b once — shared by av blending and chunk scan.
        la_full = (dt_b * A_b).astype(mx.float32)                   # (B, L, H)

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

        # Precompute sin/cos once — both B_rot and C_rot use the same angles.
        _sin_a = mx.sin(angles)[..., None].astype(x.dtype)          # (B, L, H, N//2, 1)
        _cos_a = mx.cos(angles)[..., None].astype(x.dtype)

        # Prepare B, C rotated
        B_p, C_p = self._prepare_BC(B_param, C_param, B_sz, L)      # (B, L, H, N, R)
        B_rot = apply_rope_with_sc(B_p, _sin_a, _cos_a)
        C_rot = apply_rope_with_sc(C_p, _sin_a, _cos_a)

        # x_up via MoE
        x_up = self.x_up_proj(x_prime_hp.reshape(B_sz, L, H * P))   # (B, L, H*P*R)
        x_ssm = x_up.reshape(B_sz, L, H, P, R)

        # input_signal: (B, L, H, N, P) = einsum("blhnr,blhpr->blhnp", B_rot, x_ssm)
        input_signal = mx.einsum("blhnr,blhpr->blhnp", B_rot, x_ssm)

        # lv, dv, av blending
        # av reuses la_full (= dt_b * A_b) — avoids a redundant elementwise product.
        lv = mx.sigmoid(self._broadcast_groups(lam, axis=-1)).reshape(B_sz, L, H, 1, 1).astype(x.dtype)
        dv = dt_b.reshape(B_sz, L, H, 1, 1).astype(x.dtype)
        av = mx.exp(la_full).reshape(B_sz, L, H, 1, 1).astype(x.dtype)

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
            y_stack, new_h_prev = chunk_scan(
                u_ssm, la_full, C_rot, self.chunk_size,
                h_init=h_init, _tri_mask=self._tri_mask)

        # y_stack: (B, L, H, P, R) -> (B, L, H, P*R) -> y_down_proj -> (B, L, H, P) -> (B, L, H*P)
        y = self.y_down_proj(y_stack.reshape(B_sz, L, H, P * R)).reshape(B_sz, L, H * P)

        # D_expand: use precomputed cache when available (avoids mx.repeat every step).
        if self._D_expand is not None:
            D_expand = self._D_expand.astype(x.dtype)
        else:
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
