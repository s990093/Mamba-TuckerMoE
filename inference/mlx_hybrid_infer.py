# -*- coding: utf-8 -*-
"""
MLX inference stack aligned with repo root train.py (Mamba3 + TuckerMoE + GQA).
- Prefill: chunk-parallel scan matching train.py chunk_parallel_scan (no Triton; same math).
- Decode: per-layer @mx.compile steps (see attach_decode_compilation) + compiled LM head.
- Norms: :func:`rms_norm_fast` wraps ``mx.fast.rms_norm``; attention uses ``mx.fast.scaled_dot_product_attention``.
  Mamba B/C rotation uses :func:`apply_rope` (custom angles), not ``mx.fast.rope``.
"""
from __future__ import annotations

import math
import os
from typing import Any, List, Literal, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn


class Mamba3Config:
    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 64,
        d_head: int = 64,
        n_groups: int = 1,
        mimo_rank: int = 4,
        expand: int = 4,
        num_layers: int = 6,
        use_parallel_scan: bool = True,
        chunk_size: int = 64,
        use_kmoe: bool = True,
        kmoe_num_experts: int = 8,
        kmoe_top_k: int = 2,
        kmoe_r1: int = 32,
        kmoe_r2: int = 512,
        kmoe_r3: int = 256,
        ffn_expand: int = 6,
        num_kv_heads: int = 4,
        rms_norm_eps: float = 1e-5,
        layer_scale_init: float = 1e-2,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.expand = expand
        self.num_layers = num_layers
        self.d_inner = int(expand * d_model)
        self.n_heads = self.d_inner // d_head
        self.n_groups = n_groups
        self.mimo_rank = mimo_rank
        self.rms_norm_eps = rms_norm_eps
        self.use_parallel_scan = use_parallel_scan
        self.chunk_size = chunk_size
        self.use_kmoe = use_kmoe
        self.kmoe_num_experts = kmoe_num_experts
        self.kmoe_top_k = kmoe_top_k
        self.kmoe_r1 = kmoe_r1
        self.kmoe_r2 = kmoe_r2
        self.kmoe_r3 = kmoe_r3
        self.ffn_expand = ffn_expand
        self.num_kv_heads = num_kv_heads
        self.kv_groups = self.n_heads // num_kv_heads
        self.layer_scale_init = layer_scale_init
        # Experimental decode / MoE optimizations (benchmark-friendly; default off).
        self.lookahead_router: bool = False
        #: Single einsum ``bk,br,bkrs->bs`` latent MoE (vs staged matmul + sum).
        self.tucker_einsum_fuse: bool = False
        #: Fuse dense U_in + RMSNorm + routed Tucker core + U_out into one Metal kernel.
        self.tucker_full_fuse: bool = False


def fast_scaled_tanh(x: mx.array, scale: float = 10.0) -> mx.array:
    return mx.tanh(x * (1.0 / scale)) * scale


def rms_norm_fast(x: mx.array, norm: nn.RMSNorm) -> mx.array:
    """Same math as ``nn.RMSNorm`` forward using :func:`mlx.core.fast.rms_norm`."""
    eps = float(getattr(norm, "eps", 1e-5))
    w = norm.weight
    return mx.fast.rms_norm(x, w, eps)


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def fast_silu_gating(gate: mx.array, feat: mx.array) -> mx.array:
    return silu(gate) * feat


def apply_rope(x: mx.array, angles: mx.array) -> mx.array:
    """Angle-based 2D rotate; not ``mx.fast.rope`` (positional RoPE with base/offset)."""
    n_half = angles.shape[-1]
    x_reshaped = x.reshape(*x.shape[:-2], n_half, 2, x.shape[-1])
    x1, x2 = x_reshaped[..., 0, :], x_reshaped[..., 1, :]
    sin_a = mx.expand_dims(mx.sin(angles), -1)
    cos_a = mx.expand_dims(mx.cos(angles), -1)
    out = mx.stack([x1 * cos_a - x2 * sin_a, x2 * cos_a + x1 * sin_a], axis=-2)
    return out.reshape(x.shape)


def _topk_indices(router_logits: mx.array, k: int) -> mx.array:
    """Largest-k expert indices (order among the k not guaranteed). O(E) vs argsort O(E log E)."""
    return mx.argpartition(-router_logits, k - 1, axis=-1)[:, :k]


def _router_anchor_matching_dim(raw: Optional[mx.array], expected_last: int) -> Optional[mx.array]:
    """
    Previous-layer hidden lookahead only applies when Tucker ``router`` in_features matches
    *expected_last* (e.g. Mamba ``x_up_proj`` routers see ``n_heads * d_head``, not ``d_model``).
    """
    if raw is None:
        return None
    if int(raw.shape[-1]) != int(expected_last):
        return None
    return raw


def _is_dense_linear(m: Any) -> bool:
    """True for plain MLX Linear modules; QuantizedLinear needs a different packed-weight kernel."""
    return m.__class__.__name__ == "Linear" and hasattr(m, "weight")


def chunk_parallel_scan_mlx(
    u: mx.array,
    dt_b: mx.array,
    a_b: mx.array,
    c_rotated: mx.array,
    chunk_size: int,
) -> Tuple[mx.array, mx.array]:
    """
    u: (B,L,H,N,P), dt_b/a_b: (B,L,H), C_rotated: (B,L,G,N,R) with G=n_groups — train uses head-expanded C via bg.
    Matches train.py chunk_parallel_scan (Triton path) layout.
    """
    b, l, h, n, p = u.shape
    r = c_rotated.shape[-1]
    l_orig = l
    pad = 0
    if l % chunk_size != 0:
        pad = chunk_size - (l % chunk_size)
        u = mx.pad(u, [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
        dt_b = mx.pad(dt_b, [(0, 0), (0, pad), (0, 0)])
        a_b = mx.pad(a_b, [(0, 0), (0, pad), (0, 0)])
        c_rotated = mx.pad(c_rotated, [(0, 0), (0, pad), (0, 0), (0, 0), (0, 0)])
        l = u.shape[1]

    nc = l // chunk_size
    log_alpha = dt_b * a_b
    u_c = u.reshape(b, nc, chunk_size, h, n, p)
    la_c = log_alpha.reshape(b, nc, chunk_size, h)
    c_c = c_rotated.reshape(b, nc, chunk_size, c_rotated.shape[2], n, r)

    lc = la_c.shape[2]
    d_inner = n * p
    
    # Intra-chunk scan: Fast Custom Metal Kernel (Sequential Scan O(N))
    # Replace O(N^2) matmul with optimal thread-per-channel sequential scan.
    _SSM_METAL_SRC = """
        uint d = thread_position_in_grid.x;
        uint h = thread_position_in_grid.y;
        uint b_c = thread_position_in_grid.z;
        
        if (d >= D || h >= H || b_c >= B * nc) return;
        
        T h_val = T(0.0);
        for (uint t = 0; t < Lc; ++t) {
            T la = la_c[b_c * Lc * H + t * H + h];
            T u  = u_c[b_c * Lc * H * D + t * H * D + h * D + d];
            
            la = la > T(40.0) ? T(40.0) : la;
            la = la < T(-40.0) ? T(-40.0) : la;
            
            h_val = metal::exp(la) * h_val + u;
            out[b_c * Lc * H * D + t * H * D + h * D + d] = h_val;
        }
    """
    
    u_c_reshaped = u_c.reshape(b, nc, lc, h, d_inner)
    kernel = mx.fast.metal_kernel(
        name="ssm_chunk_scan",
        input_names=["la_c", "u_c"],
        output_names=["out"],
        source=_SSM_METAL_SRC,
        header=f"constant uint B = {b}; constant uint nc = {nc}; constant uint Lc = {lc}; constant uint H = {h}; constant uint D = {d_inner};"
    )
    
    h_intra_flat = kernel(
        inputs=[la_c, u_c_reshaped],
        template=[("T", la_c.dtype)],
        grid=(d_inner, h, b * nc),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, nc, lc, h, d_inner)],
        output_dtypes=[u_c.dtype],
    )[0]
    
    h_intra = h_intra_flat.reshape(b, nc, lc, h, n, p)

    _CLIP = mx.array(40.0, dtype=la_c.dtype)
    y_diag = mx.einsum("bclhnp,bclhnr->bclhpr", h_intra, c_c)

    decay = mx.exp(mx.clip(mx.sum(la_c, axis=2), -_CLIP, _CLIP))
    h_prev = mx.zeros((b, h, n, p), dtype=u.dtype)
    inter_chunks: List[mx.array] = []
    for c in range(nc):
        inter_chunks.append(h_prev)
        dec = mx.expand_dims(mx.expand_dims(decay[:, c], -1), -1)
        h_prev = h_prev * dec + h_intra[:, c, -1]
    h_inter = mx.stack(inter_chunks, axis=1)

    # la_c: (B, nc, Lc, H); broadcast over (N, R) like train unsqueeze(-1).unsqueeze(-1)
    c_dec = c_c * mx.expand_dims(mx.exp(mx.clip(mx.cumsum(la_c, axis=2), -_CLIP, _CLIP)), (-1, -2))
    y_off = mx.einsum("bchnp,bclhnr->bclhpr", h_inter, c_dec)
    y = (y_diag + y_off).reshape(b, -1, h, p, r)
    if l_orig < l:
        y = y[:, :l_orig]
    return y.astype(u.dtype), h_prev.astype(u.dtype)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = mx.ones((dim,)) * init_value

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class TuckerMoE(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_experts: int = 8,
        top_k: int = 2,
        r1: int = 4,
        r2: int = 1024,
        r3: int = 256,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router = nn.Linear(dim_in, num_experts, bias=False)
        self.U_expert = mx.zeros((num_experts, r1))
        # nn.Linear so MLX compile / nn.quantize see standard matmuls; weight (out, in).
        self.U_in = nn.Linear(dim_in, r3, bias=False)
        self.U_out = nn.Linear(r2, dim_out, bias=False)
        self.core = mx.zeros((r1, r3, r2))
        self.bias = mx.zeros((dim_out,))
        self.inner_norm = nn.RMSNorm(r3)
        self._G_cache: Optional[mx.array] = None

    def invalidate_g_cache(self) -> None:
        self._G_cache = None

    def _get_G(self) -> mx.array:
        if self._G_cache is None:
            # (E, r1) @ (r1, r3*r2) -> (E, r3, r2)
            r1, r3, r2 = self.core.shape
            core_2d = self.core.reshape(r1, r3 * r2)
            self._G_cache = mx.matmul(self.U_expert, core_2d).reshape(self.num_experts, r3, r2)
            mx.eval(self._G_cache)
        return self._G_cache

    def _full_fused_dense(
        self,
        x_flat: mx.array,
        top_k_probs: mx.array,
        top_k_indices: mx.array,
        g_all: mx.array,
    ) -> mx.array:
        """
        Experimental dense-only full Tucker path:
        x -> U_in -> RMSNorm -> routed G -> U_out -> bias in one Metal dispatch.

        This intentionally does not handle MLX QuantizedLinear yet; callers guard for dense Linears.
        """
        bsz, dim_in = x_flat.shape
        _, top_k = top_k_probs.shape
        num_experts, r3, r2 = g_all.shape
        dim_out = self.U_out.weight.shape[0]
        tg_size = 128
        bank_shift = 5
        r3_pad = r3 + ((r3 + ((1 << bank_shift) - 1)) >> bank_shift)
        r2_pad = r2 + ((r2 + ((1 << bank_shift) - 1)) >> bank_shift)
        norm_eps = float(getattr(self.inner_norm, "eps", 1e-5))

        source = """
            uint j = thread_position_in_grid.x;
            uint b = thread_position_in_grid.y;
            uint lid = thread_index_in_threadgroup;

            threadgroup float x_sram[R3_PAD];
            threadgroup float core_sram[R2_PAD];

            for (uint r = lid; r < R3; r += TG_SIZE) {
                float acc = 0.0f;
                uint w_base = r * DIM_IN;
                uint x_base = b * DIM_IN;
                for (uint i = 0; i < DIM_IN; ++i) {
                    acc += float(x[x_base + i]) * float(u_in_w[w_base + i]);
                }
                x_sram[r + (r >> BANK_SHIFT)] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float ss = 0.0f;
            for (uint r = 0; r < R3; ++r) {
                float v = x_sram[r + (r >> BANK_SHIFT)];
                ss += v * v;
            }
            float inv_rms = metal::rsqrt(ss / float(R3) + NORM_EPS);
            for (uint r = lid; r < R3; r += TG_SIZE) {
                uint rp = r + (r >> BANK_SHIFT);
                x_sram[rp] = x_sram[rp] * inv_rms * float(norm_w[r]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint s = lid; s < R2; s += TG_SIZE) {
                float acc = 0.0f;
                for (uint k = 0; k < K; ++k) {
                    int expert = expert_idx[b * K + k];
                    if (expert < 0 || expert >= int(NUM_EXPERTS)) {
                        continue;
                    }
                    float prob = float(probs[b * K + k]);
                    uint g_base = uint(expert) * R3 * R2 + s;
                    uint r = 0;
                    for (; r + 3 < R3; r += 4) {
                        float x0 = x_sram[r + (r >> BANK_SHIFT)];
                        float x1 = x_sram[(r + 1) + ((r + 1) >> BANK_SHIFT)];
                        float x2 = x_sram[(r + 2) + ((r + 2) >> BANK_SHIFT)];
                        float x3 = x_sram[(r + 3) + ((r + 3) >> BANK_SHIFT)];
                        float g0 = float(g_all[g_base + r * R2]);
                        float g1 = float(g_all[g_base + (r + 1) * R2]);
                        float g2 = float(g_all[g_base + (r + 2) * R2]);
                        float g3 = float(g_all[g_base + (r + 3) * R2]);
                        acc += prob * (x0 * g0 + x1 * g1 + x2 * g2 + x3 * g3);
                    }
                    for (; r < R3; ++r) {
                        acc += prob * x_sram[r + (r >> BANK_SHIFT)] * float(g_all[g_base + r * R2]);
                    }
                }
                core_sram[s + (s >> BANK_SHIFT)] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (b >= B || j >= DIM_OUT) return;
            float y = float(bias[j]);
            uint w_out_base = j * R2;
            uint s = 0;
            for (; s + 3 < R2; s += 4) {
                float c0 = core_sram[s + (s >> BANK_SHIFT)];
                float c1 = core_sram[(s + 1) + ((s + 1) >> BANK_SHIFT)];
                float c2 = core_sram[(s + 2) + ((s + 2) >> BANK_SHIFT)];
                float c3 = core_sram[(s + 3) + ((s + 3) >> BANK_SHIFT)];
                y += c0 * float(u_out_w[w_out_base + s]);
                y += c1 * float(u_out_w[w_out_base + s + 1]);
                y += c2 * float(u_out_w[w_out_base + s + 2]);
                y += c3 * float(u_out_w[w_out_base + s + 3]);
            }
            for (; s < R2; ++s) {
                y += core_sram[s + (s >> BANK_SHIFT)] * float(u_out_w[w_out_base + s]);
            }
            out[b * DIM_OUT + j] = T(y);
        """

        kernel = mx.fast.metal_kernel(
            name=f"tucker_full_fused_dense_d{dim_in}_o{dim_out}_k{top_k}_r{r3}_{r2}",
            input_names=["x", "probs", "expert_idx", "g_all", "u_in_w", "u_out_w", "norm_w", "bias"],
            output_names=["out"],
            source=source,
            header=(
                f"constant uint B={bsz}; constant uint DIM_IN={dim_in}; constant uint DIM_OUT={dim_out}; "
                f"constant uint NUM_EXPERTS={num_experts}; constant uint K={top_k}; "
                f"constant uint R3={r3}; constant uint R2={r2}; constant uint TG_SIZE={tg_size}; "
                f"constant uint BANK_SHIFT={bank_shift}; constant uint R3_PAD={r3_pad}; "
                f"constant uint R2_PAD={r2_pad}; constant float NORM_EPS={norm_eps};"
            ),
        )
        return kernel(
            inputs=[
                x_flat,
                top_k_probs,
                top_k_indices.astype(mx.int32),
                g_all,
                self.U_in.weight,
                self.U_out.weight,
                self.inner_norm.weight,
                self.bias,
            ],
            template=[("T", x_flat.dtype)],
            grid=(dim_out, bsz, 1),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[(bsz, dim_out)],
            output_dtypes=[x_flat.dtype],
        )[0]

    def __call__(
        self,
        x: mx.array,
        router_temp: mx.array,
        *,
        router_x: Optional[mx.array] = None,
        einsum_fuse: Optional[bool] = None,
        full_fuse: Optional[bool] = None,
    ) -> mx.array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        rx = x_flat if router_x is None else router_x.reshape(-1, router_x.shape[-1])
        raw_logits = self.router(rx)
        rt_dt = router_temp.dtype
        t = mx.maximum(router_temp, mx.array(1e-4, dtype=rt_dt))
        capped = fast_scaled_tanh(raw_logits, 10.0)
        router_logits = capped / t
        router_probs = mx.softmax(router_logits, axis=-1)
        top_k_indices = _topk_indices(router_logits, self.top_k)
        top_k_raw = mx.take_along_axis(router_probs, top_k_indices, axis=-1)
        eps = mx.array(1e-6, dtype=top_k_raw.dtype)
        top_k_probs = top_k_raw / (mx.sum(top_k_raw, axis=-1, keepdims=True) + eps)

        g_all = self._get_G()
        do_full_fuse = bool(full_fuse) if full_fuse is not None else False
        if (
            do_full_fuse
            and _is_dense_linear(self.U_in)
            and _is_dense_linear(self.U_out)
            and self.U_out.weight.shape[0] >= 128
        ):
            out = self._full_fused_dense(x_flat, top_k_probs, top_k_indices, g_all)
            return out.reshape(*orig_shape[:-1], -1)

        x_shared = rms_norm_fast(self.U_in(x_flat), self.inner_norm)
        g_sel = g_all[top_k_indices]
        do_fuse = bool(einsum_fuse) if einsum_fuse is not None else False
        if do_fuse and g_sel.shape[-1] >= 128:
            # Σ_k p_k · (x @ G_k) === Σ_{k,r} p_k · x_r · G_{k,r,s}
            # Custom Metal Kernel for Fused Tucker MoE.
            # Cache x_shared per output-s tile to avoid rereading the same latent vector for every s.
            _TUCKER_METAL_SRC = """
                uint s = thread_position_in_grid.x;
                uint b = thread_position_in_grid.y;
                uint lid = thread_index_in_threadgroup;

                threadgroup float x_sram[R1_PAD];
                for (uint r = lid; r < R1; r += TG_SIZE) {
                    x_sram[r + (r >> BANK_SHIFT)] = float(x_shared[b * R1 + r]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                if (b >= B || s >= R2) return;

                float sum = 0.0f;
                for (uint k = 0; k < K; ++k) {
                    float prob = float(probs[b * K + k]);
                    uint g_base = b * K * R1 * R2 + k * R1 * R2 + s;
                    uint r = 0;
                    for (; r + 3 < R1; r += 4) {
                        float x0 = x_sram[r + (r >> BANK_SHIFT)];
                        float x1 = x_sram[(r + 1) + ((r + 1) >> BANK_SHIFT)];
                        float x2 = x_sram[(r + 2) + ((r + 2) >> BANK_SHIFT)];
                        float x3 = x_sram[(r + 3) + ((r + 3) >> BANK_SHIFT)];
                        float g0 = float(g_sel[g_base + r * R2]);
                        float g1 = float(g_sel[g_base + (r + 1) * R2]);
                        float g2 = float(g_sel[g_base + (r + 2) * R2]);
                        float g3 = float(g_sel[g_base + (r + 3) * R2]);
                        sum += prob * (x0 * g0 + x1 * g1 + x2 * g2 + x3 * g3);
                    }
                    for (; r < R1; ++r) {
                        sum += prob * x_sram[r + (r >> BANK_SHIFT)] * float(g_sel[g_base + r * R2]);
                    }
                }
                out[b * R2 + s] = T(sum);
            """
            _b, _k = top_k_probs.shape
            _, _r1 = x_shared.shape
            _, _, _, _r2 = g_sel.shape
            _tg_size = 128
            _bank_shift = 5
            _r1_pad = _r1 + ((_r1 + ((1 << _bank_shift) - 1)) >> _bank_shift)
            
            kernel = mx.fast.metal_kernel(
                name=f"fused_tucker_einsum_sram_b{_b}_k{_k}_r{_r1}_{_r2}",
                input_names=["probs", "x_shared", "g_sel"],
                output_names=["out"],
                source=_TUCKER_METAL_SRC,
                header=(
                    f"constant uint B={_b}; constant uint K={_k}; constant uint R1={_r1}; "
                    f"constant uint R2={_r2}; constant uint TG_SIZE={_tg_size}; "
                    f"constant uint BANK_SHIFT={_bank_shift}; constant uint R1_PAD={_r1_pad};"
                )
            )
            out_core = kernel(
                inputs=[top_k_probs, x_shared, g_sel],
                template=[("T", top_k_probs.dtype)],
                grid=(_r2, _b, 1),
                threadgroup=(_tg_size, 1, 1),
                output_shapes=[(_b, _r2)],
                output_dtypes=[top_k_probs.dtype],
            )[0]
        else:
            expert_outs = mx.einsum("br,bkrs->bks", x_shared, g_sel)
            out_core = mx.sum(expert_outs * mx.expand_dims(top_k_probs, axis=-1), axis=1)
        out = self.U_out(out_core).reshape(*orig_shape[:-1], -1)
        return out + self.bias


class MixtralMoEFeedForward(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        d_ff = int(math.ceil(config.ffn_expand * config.d_model / 256) * 256)
        kw = dict(
            num_experts=config.kmoe_num_experts,
            top_k=config.kmoe_top_k,
            r1=config.kmoe_r1,
            r2=config.kmoe_r2,
            r3=config.kmoe_r3,
        )
        self.gate_proj = TuckerMoE(config.d_model, d_ff, **kw)
        self.up_proj = TuckerMoE(config.d_model, d_ff, **kw)
        self.down_proj = TuckerMoE(d_ff, config.d_model, **kw)

    def __call__(
        self,
        x: mx.array,
        router_temp: mx.array,
        *,
        router_anchor: Optional[mx.array] = None,
        einsum_fuse: Optional[bool] = None,
        full_fuse: Optional[bool] = None,
    ) -> mx.array:
        d_ff_router_in = self.down_proj.router.weight.shape[1]
        dm = self.gate_proj.router.weight.shape[1]
        anch_up = _router_anchor_matching_dim(router_anchor, int(dm))
        anch_down = _router_anchor_matching_dim(router_anchor, int(d_ff_router_in))
        gate = self.gate_proj(x, router_temp, router_x=anch_up, einsum_fuse=einsum_fuse, full_fuse=full_fuse)
        feat = self.up_proj(x, router_temp, router_x=anch_up, einsum_fuse=einsum_fuse, full_fuse=full_fuse)
        return self.down_proj(
            fast_silu_gating(gate, feat),
            router_temp,
            router_x=anch_down,
            einsum_fuse=einsum_fuse,
            full_fuse=full_fuse,
        )


class Mamba3Block(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        d_in = config.d_model
        h, g, p, n, r = config.n_heads, config.n_groups, config.d_head, config.d_state, config.mimo_rank
        self.ratio = h // g
        self.dim_z = h * p
        self.dim_x = h * p
        self.dim_B = g * n * r
        self.dim_C = g * n * r
        self.dim_dt = g
        self.dim_A = g
        self.dim_lambda = g
        total_in = self.dim_z + self.dim_x + self.dim_B + self.dim_C + self.dim_dt + self.dim_A + self.dim_lambda
        self.in_proj = nn.Linear(d_in, total_in, bias=True)

        if config.use_kmoe:
            kw = dict(
                num_experts=config.kmoe_num_experts,
                top_k=config.kmoe_top_k,
                r1=config.kmoe_r1,
                r2=config.kmoe_r2,
                r3=config.kmoe_r3,
            )
            self.x_up_proj = TuckerMoE(h * p, h * p * r, **kw)
            self.out_proj = TuckerMoE(d_in, d_in, **kw)
        else:
            self.x_up_proj = nn.Linear(p, p * r, bias=False)
            self.out_proj = nn.Linear(d_in, d_in, bias=False)

        self.y_down_proj = nn.Linear(p * r, p, bias=False)
        self.theta_log = mx.zeros((g, n // 2))
        self.D = mx.ones((h,))
        self.norm_B = nn.RMSNorm(n * r, eps=config.rms_norm_eps)
        self.norm_C = nn.RMSNorm(n * r, eps=config.rms_norm_eps)
        self.bias_B = mx.zeros((g, n, r))
        self.bias_C = mx.zeros((g, n, r))
        self.mamba_dense_proj = nn.Linear(config.d_inner, d_in, bias=False)
        self.pre_gate_norm = nn.RMSNorm(h * p)
        self.norm_mamba = nn.RMSNorm(config.d_model)
        self.norm_out_proj = nn.RMSNorm(config.d_model)
        self.ls_mamba = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_out_proj = LayerScale(config.d_model, init_value=config.layer_scale_init)

        import numpy as _np

        splits = [self.dim_z, self.dim_x, self.dim_B, self.dim_C, self.dim_dt, self.dim_A, self.dim_lambda]
        self._split_indices = _np.cumsum(splits)[:-1].tolist()
        self._theta_rep_cache: Optional[mx.array] = None
        self._D_rep_cache: Optional[mx.array] = None

    def _bg(self, t: mx.array) -> mx.array:
        return mx.repeat(t, self.ratio, axis=2)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Tuple[mx.array, mx.array, mx.array]] = None,
        router_temp: Optional[mx.array] = None,
        router_anchor: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array, mx.array]]]:
        b_sz, l, _ = x.shape
        h, g, p, n, r, ratio = (
            self.config.n_heads,
            self.config.n_groups,
            self.config.d_head,
            self.config.d_state,
            self.config.mimo_rank,
            self.ratio,
        )
        if router_temp is None:
            router_temp = mx.array(0.5, dtype=x.dtype)

        residual_mamba = x
        u = rms_norm_fast(x, self.norm_mamba)
        proj_out = self.in_proj(u)
        z, x_prime, b_param, c_param, dt, a_param, lambda_param = mx.split(proj_out, self._split_indices, axis=-1)
        x_prime = x_prime.reshape(b_sz, l, h, p)

        dt = mx.logaddexp(mx.array(0.0, dt.dtype), dt)
        a = -mx.exp(a_param)

        dt_b = self._bg(mx.expand_dims(dt, -1)).squeeze(-1)
        a_b = self._bg(mx.expand_dims(a, -1)).squeeze(-1)

        if self._theta_rep_cache is None:
            theta = mx.exp(self.theta_log)
            self._theta_rep_cache = mx.repeat(theta, ratio, axis=0)
            mx.eval(self._theta_rep_cache)
        theta_rep = self._theta_rep_cache

        current_angle_step = mx.einsum("blh, hn -> blhn", dt_b, theta_rep)
        if cache is not None:
            prev_h, prev_input, prev_angle_sum = cache
            angles = prev_angle_sum + mx.cumsum(current_angle_step, axis=1)
        else:
            angles = mx.cumsum(current_angle_step, axis=1)
        new_angle_sum = angles[:, -1:]

        b_reshaped = rms_norm_fast(
            b_param.reshape(b_sz, l, g, n * r), self.norm_B
        ).reshape(b_sz, l, g, n, r)
        c_reshaped = rms_norm_fast(
            c_param.reshape(b_sz, l, g, n * r), self.norm_C
        ).reshape(b_sz, l, g, n, r)
        b_rotated = apply_rope(self._bg(b_reshaped) + self.bias_B, angles)
        c_rotated = apply_rope(self._bg(c_reshaped) + self.bias_C, angles)

        _ef = self.config.tucker_einsum_fuse
        _ff = self.config.tucker_full_fuse
        xp_dim = int(h * p)
        xm_anchor = _router_anchor_matching_dim(router_anchor, xp_dim)
        out_anchor = _router_anchor_matching_dim(router_anchor, int(self.config.d_model))
        if self.config.use_kmoe:
            x_up = self.x_up_proj(
                x_prime.reshape(b_sz, l, -1),
                router_temp,
                router_x=xm_anchor,
                einsum_fuse=_ef,
                full_fuse=_ff,
            )
            x_ssm = x_up.reshape(b_sz, l, h, p, r)
        else:
            x_ssm = self.x_up_proj(x_prime).reshape(b_sz, l, h, p, r)

        input_signal = mx.einsum("blhnr, blhpr -> blhnp", b_rotated, x_ssm)
        lv = mx.sigmoid(self._bg(mx.expand_dims(lambda_param, -1)).squeeze(-1)).reshape(b_sz, l, h, 1, 1)
        dv = dt_b.reshape(b_sz, l, h, 1, 1)
        av = mx.exp(dt_b * a_b).reshape(b_sz, l, h, 1, 1)

        if cache is not None:
            ip = mx.concatenate([prev_input, input_signal[:, :-1]], axis=1)
        else:
            ip = mx.concatenate([mx.zeros_like(input_signal[:, :1]), input_signal[:, :-1]], axis=1)
        u_ssm = lv * dv * input_signal + (1.0 - lv) * dv * av * ip

        if cache is not None and l == 1:
            h_final = prev_h * av[:, 0] + u_ssm[:, 0]
            y_stack = mx.einsum("bhnp, bhnr -> bhpr", h_final, c_rotated[:, 0])[:, None, ...]
        elif self.config.use_parallel_scan and l > 1:
            if cache is None:
                y_stack, h_final = chunk_parallel_scan_mlx(
                    u_ssm, dt_b, a_b, c_rotated, self.config.chunk_size
                )
            else:
                # Speculative verification feeds a short token block with existing recurrent state.
                h_final = prev_h
                ys = []
                for t in range(l):
                    h_final = h_final * av[:, t] + u_ssm[:, t]
                    ys.append(mx.einsum("bhnp,bhnr->bhpr", h_final, c_rotated[:, t])[:, None, ...])
                y_stack = mx.concatenate(ys, axis=1)
        else:
            h_final = mx.zeros((b_sz, h, n, p), dtype=x.dtype)
            ys = []
            for t in range(l):
                h_final = h_final * av[:, t] + u_ssm[:, t]
                ys.append(mx.einsum("bhnp,bhnr->bhpr", h_final, c_rotated[:, t])[:, None, ...])
            y_stack = mx.concatenate(ys, axis=1)

        new_cache = (h_final, input_signal[:, -1:], new_angle_sum)

        y = self.y_down_proj(y_stack.reshape(b_sz, l, h, p * r)).reshape(b_sz, l, h * p)
        if self._D_rep_cache is None:
            self._D_rep_cache = mx.repeat(self.D, p, axis=0)
            mx.eval(self._D_rep_cache)
        y = y + x_prime.reshape(b_sz, l, h * p) * self._D_rep_cache

        mamba_out = self.mamba_dense_proj(rms_norm_fast(y, self.pre_gate_norm) * silu(z))
        mid_x = residual_mamba + self.ls_mamba(mamba_out)
        normed_mid = rms_norm_fast(mid_x, self.norm_out_proj)
        if self.config.use_kmoe:
            proj_out = self.out_proj(
                normed_mid,
                router_temp,
                router_x=out_anchor,
                einsum_fuse=_ef,
                full_fuse=_ff,
            )
        else:
            proj_out = self.out_proj(normed_mid)
        out = mid_x + self.ls_out_proj(proj_out)
        return out, new_cache


class TransformerBlock(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        self.head_dim = 64
        self.num_heads = config.d_model // 64
        self.num_kv_heads = config.num_kv_heads
        self.kv_groups = self.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.d_model, self.num_heads * 64, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.num_kv_heads * 64, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.num_kv_heads * 64, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=True)

        self.norm_attn = nn.RMSNorm(config.d_model)
        self.use_kmoe = config.use_kmoe
        if config.use_kmoe:
            self.ffn = MixtralMoEFeedForward(config)
        else:
            d_ff = int(math.ceil(8 * config.d_model / 3 / 256) * 256)
            self.ffn_gate = nn.Linear(config.d_model, d_ff, bias=False)
            self.ffn_up = nn.Linear(config.d_model, d_ff, bias=False)
            self.ffn_down = nn.Linear(d_ff, config.d_model, bias=False)

        self.norm_ffn = nn.RMSNorm(config.d_model)
        self.ls_attn = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_ffn = LayerScale(config.d_model, init_value=config.layer_scale_init)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        seq_pos: Optional[mx.array] = None,
        router_temp: Optional[mx.array] = None,
        router_anchor: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        b, l, d = x.shape
        if router_temp is None:
            router_temp = mx.array(0.5, dtype=x.dtype)

        residual = x
        nx = rms_norm_fast(x, self.norm_attn)
        q = self.q_proj(nx).reshape(b, l, self.num_heads, 64).transpose(0, 2, 1, 3)
        k = self.k_proj(nx).reshape(b, l, self.num_kv_heads, 64).transpose(0, 2, 1, 3)
        v = self.v_proj(nx).reshape(b, l, self.num_kv_heads, 64).transpose(0, 2, 1, 3)
        k = mx.repeat(k, self.kv_groups, axis=1)
        v = mx.repeat(v, self.kv_groups, axis=1)

        if cache is not None:
            k_cache, v_cache = cache
            if seq_pos is not None:
                # MLX ArrayAt has no .set(); use slice_update for contiguous KV writes.
                k_cache = mx.slice_update(k_cache, k, start_indices=seq_pos, axes=(2,))
                v_cache = mx.slice_update(v_cache, v, start_indices=seq_pos, axes=(2,))
            else:
                k_cache = mx.concatenate([k_cache, k], axis=2)
                v_cache = mx.concatenate([v_cache, v], axis=2)
            k, v = k_cache, v_cache
            new_cache = (k_cache, v_cache)
        else:
            new_cache = (k, v)

        if q.dtype != k.dtype:
            q = q.astype(k.dtype)

        if seq_pos is not None:
            max_l = k.shape[2]
            q_pos = seq_pos + mx.arange(l)
            mask = mx.arange(max_l).reshape(1, 1, 1, max_l) > q_pos.reshape(1, 1, l, 1)
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(64.0), mask=mask)
        else:
            # Causal LM prefill (matches PyTorch scaled_dot_product_attention is_causal=True).
            causal = mx.triu(mx.full((l, l), -1e9, dtype=q.dtype), k=1)
            causal = mx.reshape(causal, (1, 1, l, l))
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(64.0), mask=causal)

        attn_out = attn.transpose(0, 2, 1, 3).reshape(b, l, d)
        if attn_out.dtype != residual.dtype:
            attn_out = attn_out.astype(residual.dtype)
        x = residual + self.ls_attn(self.o_proj(attn_out))
        residual2 = x
        h = rms_norm_fast(x, self.norm_ffn)
        if self.use_kmoe:
            ffn_out = self.ffn(
                h,
                router_temp,
                router_anchor=router_anchor,
                einsum_fuse=self.config.tucker_einsum_fuse,
                full_fuse=self.config.tucker_full_fuse,
            )
        else:
            ffn_out = self.ffn_down(fast_silu_gating(self.ffn_gate(h), self.ffn_up(h)))
        return residual2 + self.ls_ffn(ffn_out), new_cache


class TrueHybridMamba(nn.Module):
    def __init__(self, config: Mamba3Config, mamba_ratio: int = 4):
        super().__init__()
        self.config = config
        self.layers: List[Any] = []
        for _ in range(config.num_layers):
            for _ in range(mamba_ratio):
                blk = Mamba3Block(config)
                blk.l_type = "mamba"
                self.layers.append(blk)
            blk = TransformerBlock(config)
            blk.l_type = "transformer"
            self.layers.append(blk)

    def __call__(
        self,
        x: mx.array,
        caches: Optional[Sequence[Any]] = None,
        seq_pos: Optional[mx.array] = None,
        router_temp: Optional[mx.array] = None,
    ) -> Tuple[mx.array, List[Any]]:
        if caches is None:
            caches = [None] * len(self.layers)
        new_caches: List[Any] = []
        l = x.shape[1]
        lookahead = self.config.lookahead_router
        layer_in: List[mx.array] = []
        for li, (layer, cache) in enumerate(zip(self.layers, caches)):
            lt = getattr(layer, "l_type", None)
            anchor = layer_in[-1] if lookahead and li >= 1 else None
            x_in = x
            if lt == "transformer":
                if l == 1 and cache is not None and seq_pos is not None and getattr(layer, "_compiled_decode", None) is not None:
                    k_cache, v_cache = cache
                    if lookahead:
                        x, nk, nv = layer._compiled_decode(x, k_cache, v_cache, seq_pos, anchor)
                    else:
                        x, nk, nv = layer._compiled_decode(x, k_cache, v_cache, seq_pos)
                    new_caches.append((nk, nv))
                else:
                    x, nc = layer(
                        x,
                        cache=cache,
                        seq_pos=seq_pos,
                        router_temp=router_temp,
                        router_anchor=anchor,
                    )
                    new_caches.append(nc)
            elif l == 1 and cache is not None and getattr(layer, "_compiled_decode", None) is not None:
                h_s, prev_in, prev_ang = cache
                if lookahead:
                    x, nh, npi, nas = layer._compiled_decode(x, h_s, prev_in, prev_ang, anchor)
                else:
                    x, nh, npi, nas = layer._compiled_decode(x, h_s, prev_in, prev_ang)
                new_caches.append((nh, npi, nas))
            else:
                x, nc = layer(x, cache=cache, router_temp=router_temp, router_anchor=anchor)
                new_caches.append(nc)
            layer_in.append(x_in)
        return x, new_caches


class Mamba3LanguageModel(nn.Module):
    def __init__(self, config: Mamba3Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.backbone = TrueHybridMamba(config)
        self.norm = nn.RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size, bias=False)
        self._compiled_lm_head = None

    def set_lm_head_compile(self, enabled: bool) -> None:
        if not enabled:
            self._compiled_lm_head = None
            return
        d = self.config.d_model
        scale = math.sqrt(d)
        norm = self.norm
        head = self.head

        def _lm(x: mx.array) -> mx.array:
            h = rms_norm_fast(x, norm)
            return fast_scaled_tanh(head(h / scale), 30.0)

        self._compiled_lm_head = mx.compile(_lm)

    def __call__(
        self,
        input_ids: mx.array,
        caches: Optional[Sequence[Any]] = None,
        seq_pos: Optional[mx.array] = None,
        router_temp: Optional[mx.array] = None,
    ) -> Tuple[mx.array, List[Any]]:
        self.head.weight = self.embed.weight
        x = self.embed(input_ids)
        hidden, new_caches = self.backbone(x, caches, seq_pos=seq_pos, router_temp=router_temp)
        if self._compiled_lm_head is not None:
            logits = self._compiled_lm_head(hidden)
        else:
            hidden = rms_norm_fast(hidden, self.norm)
            logits = fast_scaled_tanh(self.head(hidden / math.sqrt(self.config.d_model)), 30.0)
        return logits, new_caches


def attach_decode_compilation(
    model: Mamba3LanguageModel,
    max_cache_len: int,
    kv_dtype: mx.Dtype,
    compile_decode: bool = True,
) -> None:
    """Per-layer @mx.compile for L==1 decode (matches backend local_inf/main.py strategy)."""
    if not compile_decode:
        model.set_lm_head_compile(False)
        for layer in getattr(model.backbone, "layers", []):
            if hasattr(layer, "_compiled_decode"):
                delattr(layer, "_compiled_decode")
        mx.eval(model.parameters())
        return

    config = model.config
    dtype = model.embed.weight.dtype
    rt_const = mx.array(0.5, dtype=dtype)
    h_dim, p_dim, n_dim = config.n_heads, config.d_head, config.d_state
    dummy_x = mx.zeros((1, 1, config.d_model), dtype=dtype)

    la = getattr(config, "lookahead_router", False)
    dummy_anchor = mx.zeros((1, 1, config.d_model), dtype=dtype)

    for layer in model.backbone.layers:
        if getattr(layer, "l_type", None) == "mamba":

            def make_mamba_compiled(blk: Mamba3Block, *, use_la: bool):
                def _decode_step_plain(x, h_s, prev_in, prev_ang):
                    out_step, nc = blk(x, cache=(h_s, prev_in, prev_ang), router_temp=rt_const)
                    return out_step, nc[0], nc[1], nc[2]

                def _decode_step_la(x, h_s, prev_in, prev_ang, router_anchor):
                    out_step, nc = blk(
                        x,
                        cache=(h_s, prev_in, prev_ang),
                        router_temp=rt_const,
                        router_anchor=router_anchor,
                    )
                    return out_step, nc[0], nc[1], nc[2]

                return mx.compile(_decode_step_la if use_la else _decode_step_plain)

            dh = mx.zeros((1, h_dim, n_dim, p_dim), dtype=dtype)
            dpi = mx.zeros((1, 1, h_dim, n_dim, p_dim), dtype=dtype)
            dang = mx.zeros((1, 1, h_dim, n_dim // 2), dtype=dtype)
            kw: dict[str, Any] = dict(cache=(dh, dpi, dang), router_temp=rt_const)
            if la:
                kw["router_anchor"] = dummy_anchor
            _o, _ = layer(dummy_x, **kw)
            mx.eval(_o)

            layer._compiled_decode = make_mamba_compiled(layer, use_la=la)
        elif getattr(layer, "l_type", None) == "transformer":

            def make_transformer_compiled(blk: TransformerBlock, *, use_la: bool):
                def _decode_step_plain(x, k_cache, v_cache, seq_pos_i):
                    out_step, nc = blk(x, cache=(k_cache, v_cache), seq_pos=seq_pos_i, router_temp=rt_const)
                    return out_step, nc[0], nc[1]

                def _decode_step_la(x, k_cache, v_cache, seq_pos_i, router_anchor):
                    out_step, nc = blk(
                        x,
                        cache=(k_cache, v_cache),
                        seq_pos=seq_pos_i,
                        router_temp=rt_const,
                        router_anchor=router_anchor,
                    )
                    return out_step, nc[0], nc[1]

                return mx.compile(_decode_step_la if use_la else _decode_step_plain)

            dk = mx.zeros((1, layer.num_heads, max_cache_len, 64), dtype=kv_dtype)
            dv = mx.zeros((1, layer.num_heads, max_cache_len, 64), dtype=kv_dtype)
            tk: dict[str, Any] = dict(cache=(dk, dv), seq_pos=mx.array(0, dtype=mx.int32), router_temp=rt_const)
            if la:
                tk["router_anchor"] = dummy_anchor
            _o, _ = layer(dummy_x, **tk)
            mx.eval(_o)

            layer._compiled_decode = make_transformer_compiled(layer, use_la=la)

    model.set_lm_head_compile(True)
    mx.eval(model.parameters())


def save_checkpoint_numpy(model: nn.Module, filepath: str) -> None:
    """Same as ``backend/app/local_inf/tool.py`` — export weights to ``.npz``."""
    import numpy as np
    import mlx.utils as mlx_utils

    print("💾 開始將模型權重匯出為 NumPy 格式...")
    flat_params = mlx_utils.tree_flatten(model.parameters())
    np_params = {k: np.array(v) for k, v in flat_params}
    np.savez(filepath, **np_params)
    del np_params
    print(f"✅ 模型已成功儲存至 {filepath}")
    mx.eval(model.parameters())


def export_npz_cache(model: nn.Module, path: str) -> None:
    """Alias for :func:`save_checkpoint_numpy` (benchmark ``--save-npz``)."""
    save_checkpoint_numpy(model, path)


def maybe_export_npz_sidecar_after_pt_load(
    model: nn.Module,
    resolved_checkpoint: str,
    *,
    force_refresh: bool = False,
) -> Optional[str]:
    """
    After loading weights from a PyTorch file, write ``<stem>.npz`` so :func:`resolve_mlx_checkpoint`
    can prefer it on the next run (faster than ``torch.load``).

    If the sidecar already exists, skips unless *force_refresh* (use when ``--force-pt`` re-reads ``.pt``).
    Returns the path written, or ``None`` if skipped.
    """
    p = os.path.abspath(resolved_checkpoint)
    if not p.lower().endswith((".pt", ".pth", ".bin")):
        return None
    sidecar = os.path.splitext(p)[0] + ".npz"
    if os.path.isfile(sidecar) and not force_refresh:
        return None
    export_npz_cache(model, sidecar)
    return sidecar


def resolve_mlx_checkpoint(
    checkpoint: str,
    *,
    repo_root: str,
    npz_cache: str = "",
    force_pt: bool = False,
) -> Tuple[Optional[str], Literal["npz", "pt", "none"]]:
    """
    Choose which file to load: prefer an existing .npz cache when a .pt is requested.

    (Benchmark / profile scripts write ``<stem>.npz`` after the first successful ``.pt`` load so later runs hit this branch.)

    Resolution order when *checkpoint* points to a .pt/.pth:
      1. If *npz_cache* is set and exists → use it (unless *force_pt*).
      2. Else *checkpoint* with suffix replaced by `.npz` if that file exists.
      3. Else ``{repo_root}/model.npz`` if it exists.
      4. Else load the .pt file.

    When *checkpoint* is empty, tries ``{repo_root}/model.npz`` then ``{repo_root}/checkpoint.pt``
    (the latter may still resolve to a sidecar .npz).

    Returns ``(path_or_none, kind)`` where kind is ``npz``, ``pt``, or ``none``.
    """
    repo_root = os.path.abspath(repo_root)

    def _try_npz_for_pt(pt_path: str) -> Optional[str]:
        if force_pt:
            return None
        candidates: List[str] = []
        if npz_cache:
            candidates.append(os.path.abspath(npz_cache))
        stem, _ = os.path.splitext(pt_path)
        candidates.append(stem + ".npz")
        candidates.append(os.path.join(repo_root, "model.npz"))
        seen = set()
        for c in candidates:
            if not c or c in seen:
                continue
            seen.add(c)
            if os.path.isfile(c):
                return c
        return None

    ck = checkpoint.strip()
    if not ck:
        mnpz = os.path.join(repo_root, "model.npz")
        if os.path.isfile(mnpz) and not force_pt:
            return (mnpz, "npz")
        cpt = os.path.join(repo_root, "checkpoint.pt")
        if os.path.isfile(cpt):
            side = _try_npz_for_pt(cpt)
            if side is not None:
                return (side, "npz")
            return (cpt, "pt")
        return (None, "none")

    ck_abs = os.path.abspath(ck)
    ext = os.path.splitext(ck_abs)[1].lower()

    if ext == ".npz":
        if os.path.isfile(ck_abs):
            return (ck_abs, "npz")
        return (None, "none")

    if ext in (".pt", ".pth", ".bin"):
        if not os.path.isfile(ck_abs):
            return (None, "none")
        side = _try_npz_for_pt(ck_abs)
        if side is not None:
            return (side, "npz")
        return (ck_abs, "pt")

    return (None, "none")


def _normalize_tucker_checkpoint_keys(np_data: dict[str, Any]) -> dict[str, Any]:
    """
    Map legacy TuckerMoE ``U_in`` / ``U_out`` matrix checkpoints to ``nn.Linear`` weights.

    Old files store ``(dim_in, r3)`` / ``(r2, dim_out)`` for ``x @ U`` matmuls.
    :class:`nn.Linear` stores ``weight`` as ``(out_features, in_features)``, i.e. the transpose.
    """
    import numpy as np

    out = dict(np_data)
    for k in list(out.keys()):
        if k.endswith(".U_in.weight") or k.endswith(".U_out.weight"):
            continue
        if not (k.endswith(".U_in") or k.endswith(".U_out")):
            continue
        wkey = k + ".weight"
        if wkey in out:
            del out[k]
            continue
        arr = np.asarray(out.pop(k))
        if arr.ndim != 2:
            out[k] = arr
            continue
        out[wkey] = np.ascontiguousarray(arr.T)
    return out


def _load_checkpoint_to_numpy_dict(filepath: str) -> Tuple[dict[str, Any], str]:
    """
    Load a ``.pt`` / ``.pth`` / ``.bin`` or ``.npz`` into plain numpy arrays.

    If the PyTorch file is a **truncated zip** (common with incomplete downloads), tries
    ``<stem>.npz`` then ``<dir>/model.npz`` before failing.
    """
    import numpy as np
    import torch

    path = os.path.abspath(filepath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到 Checkpoint 檔案: {path}")

    def _from_pt(p: str) -> dict[str, Any]:
        pt_state_dict = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(pt_state_dict, dict) and "model" in pt_state_dict:
            pt_state_dict = pt_state_dict["model"]
        elif isinstance(pt_state_dict, dict) and "state_dict" in pt_state_dict:
            pt_state_dict = pt_state_dict["state_dict"]
        return {k: v.float().numpy() for k, v in pt_state_dict.items()}

    def _from_npz(p: str) -> dict[str, Any]:
        npz_file = np.load(p, mmap_mode="r")
        return {k: npz_file[k] for k in npz_file.files}

    if path.endswith((".pt", ".pth", ".bin")):
        try:
            return _from_pt(path), path
        except RuntimeError as e:
            low = str(e).lower()
            if not any(
                s in low
                for s in (
                    "zip",
                    "central directory",
                    "pytorchstreamreader",
                    "invalid header",
                )
            ):
                raise
            stem_npz = os.path.splitext(path)[0] + ".npz"
            dir_npz = os.path.join(os.path.dirname(path), "model.npz")
            for alt in (stem_npz, dir_npz):
                if os.path.isfile(alt):
                    print(
                        f"⚠️  PyTorch 檔無法讀取（多為不完整 zip / 截斷）: {path}\n"
                        f"    改載入: {alt}"
                    )
                    return _from_npz(alt), alt
            raise RuntimeError(
                "PyTorch checkpoint 無法讀取：多數為 .pt 不完整（例如 zip 缺少尾端中央目錄）。"
                "請重新取得完整檔案，或於同目錄放置 checkpoint.npz / model.npz 再試。"
            ) from e

    if path.endswith(".npz"):
        return _from_npz(path), path

    raise ValueError(f"不支援的 checkpoint 副檔名: {path}")


def strict_load_and_convert(model: nn.Module, filepath: str) -> None:
    """
    Load PyTorch (``.pt`` / ``.bin``) or NumPy (``.npz``) weights into *model*,
    matching ``inference/backend/app/local_inf/tool.py`` (key remap ``.block.`` → ``.``).
    """
    import mlx.utils as mlx_utils

    print(f"📥 正在從 {filepath} 讀取權重...")
    np_data, used_path = _load_checkpoint_to_numpy_dict(filepath)
    print(f"✅ 成功提取 Checkpoint，來源 {used_path}，共 {len(np_data)} 個權重張量。")

    np_data = _normalize_tucker_checkpoint_keys(np_data)

    mlx_params = mlx_utils.tree_flatten(model.parameters())
    mlx_keys = {k for k, _ in mlx_params}
    loaded_flat_params: List[Tuple[str, mx.array]] = []
    loaded_keys: set[str] = set()
    for k, v in np_data.items():
        new_k = k.replace(".block.", ".")
        loaded_flat_params.append((new_k, mx.array(v)))
        loaded_keys.add(new_k)

    missing_in_mlx = mlx_keys - loaded_keys
    missing_in_ckpt = loaded_keys - mlx_keys
    match_rate = (len(mlx_keys) - len(missing_in_mlx)) / len(mlx_keys) * 100 if len(mlx_keys) > 0 else 0
    print(f"🔍 模型權重匹配率: {match_rate:.2f}% ({len(mlx_keys) - len(missing_in_mlx)}/{len(mlx_keys)})")

    if len(missing_in_mlx) > 0 or len(missing_in_ckpt) > 0:
        raise ValueError("Model layer 完美轉換檢查失敗，請檢察您的模型架構或權重名稱！")

    loaded_params = mlx_utils.tree_unflatten(loaded_flat_params)
    model.update(loaded_params)
    mx.eval(model.parameters())
    print("✨ 強制嚴格檢查通過！權重已完美轉換並載入至 MLX 模型中！")


def load_npz_checkpoint(model: nn.Module, path: str) -> None:
    """Backward-compatible alias; prefer :func:`strict_load_and_convert`."""
    strict_load_and_convert(model, path)


def load_torch_checkpoint(model: nn.Module, path: str) -> None:
    """Backward-compatible alias; prefer :func:`strict_load_and_convert`."""
    strict_load_and_convert(model, path)


def load_and_compare_vocab(model: nn.Module, filepath: str, expected_vocab_size: int) -> None:
    """Same as ``tool.load_and_compare_vocab``: load weights, then warn if vocab size differs."""
    strict_load_and_convert(model, filepath)
    got = int(model.embed.weight.shape[0])
    if got != expected_vocab_size:
        print(f"⚠️  Vocab size: embed={got}, expected_vocab_size={expected_vocab_size}")
