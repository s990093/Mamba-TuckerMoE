"""
Pure PyTorch nn.Module implementation of Mamba3-TuckerMoE.

Implements exactly the same computation graph as mamba3_mlx/ but using
eager PyTorch (no compilation, no Metal kernels, no quantization).
Compatible with weights loaded via pure_torch_mlx/weights.py.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Mamba3Config


# ── Primitives ─────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


# ── Tucker MoE ─────────────────────────────────────────────────────────────────

class TuckerMoE(nn.Module):
    """
    Low-rank Tucker-decomposed Mixture-of-Experts.

    Weight shapes (matching checkpoint keys):
      router:    (E, d_in)
      U_in:      (d_in, r3)
      inner_norm: (r3,)
      U_expert:  (E, r1)
      core:      (r1, r3, r2)
      U_out:     (r2, d_out)
      bias:      (d_out,)
    """
    def __init__(self, d_in: int, d_out: int, cfg: Mamba3Config):
        super().__init__()
        E   = cfg.kmoe_num_experts
        r1, r2, r3 = cfg.kmoe_r1, cfg.kmoe_r2, cfg.kmoe_r3
        self.top_k = cfg.kmoe_top_k

        self.router     = nn.Linear(d_in, E, bias=False)
        self.U_in       = nn.Parameter(torch.empty(d_in, r3))
        self.inner_norm = RMSNorm(r3, eps=cfg.rms_norm_eps)
        self.U_expert   = nn.Parameter(torch.empty(E, r1))
        self.core       = nn.Parameter(torch.empty(r1, r3, r2))
        self.U_out      = nn.Parameter(torch.empty(r2, d_out))
        self.bias       = nn.Parameter(torch.zeros(d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_in)  →  (..., d_out)"""
        *batch, d_in = x.shape
        x_flat = x.reshape(-1, d_in)          # (B_flat, d_in)
        B_flat = x_flat.shape[0]

        # Router
        scores = self.router(x_flat)           # (B_flat, E)
        topk_w, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        topk_w = F.softmax(topk_w, dim=-1)    # (B_flat, top_k)

        # Shared input projection
        x_in = x_flat @ self.U_in              # (B_flat, r3)
        x_in = self.inner_norm(x_in)           # (B_flat, r3)

        # Tucker contraction per expert — vectorised over batch
        # core: (r1, r3, r2), U_expert: (E, r1)
        # G_e = U_expert[e] @ core.reshape(r1, r3*r2) → (r3, r2)
        # For top-k: accumulate over k
        r2, d_out = self.U_out.shape
        acc = x_in.new_zeros(B_flat, r2)       # (B_flat, r2)

        for k in range(self.top_k):
            e_idx = topk_idx[:, k]             # (B_flat,)
            w_k   = topk_w[:, k].unsqueeze(1)  # (B_flat, 1)
            g_k   = self.U_expert[e_idx]       # (B_flat, r1)
            # G_e per batch: (B_flat, r3, r2)
            G_k = torch.einsum("br,rst->bst", g_k, self.core)
            # x_mid: (B_flat, r3) @ (B_flat, r3, r2) → (B_flat, r2)
            x_mid = torch.bmm(x_in.unsqueeze(1), G_k).squeeze(1)
            acc.add_(w_k * x_mid)

        # Output projection
        out = acc @ self.U_out + self.bias     # (B_flat, d_out)
        return out.reshape(*batch, d_out)


# ── RoPE helper ─────────────────────────────────────────────────────────────────

def apply_rope(
    x: torch.Tensor,       # (..., N, R)
    sin_a: torch.Tensor,   # (..., N//2)
    cos_a: torch.Tensor,   # (..., N//2)
) -> torch.Tensor:
    N = x.shape[-2]
    x1, x2 = x[..., : N // 2, :], x[..., N // 2:, :]
    s = sin_a.unsqueeze(-1)
    c = cos_a.unsqueeze(-1)
    return torch.cat([c * x1 - s * x2, s * x1 + c * x2], dim=-2)


# ── Mamba3Block ────────────────────────────────────────────────────────────────

class Mamba3Block(nn.Module):
    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        H, G, P, N, R = cfg.n_heads, cfg.n_groups, cfg.d_head, cfg.d_state, cfg.mimo_rank
        d = cfg.d_model

        self.H, self.G, self.P, self.N, self.R = H, G, P, N, R
        self.ratio = H // G

        dim_z    = H * P
        dim_x    = H * P
        dim_B    = G * N * R
        dim_C    = G * N * R
        in_out   = dim_z + dim_x + dim_B + dim_C + G * 3
        self.dim_z, self.dim_x, self.dim_B, self.dim_C = dim_z, dim_x, dim_B, dim_C

        self.norm_mamba    = RMSNorm(d, eps=cfg.rms_norm_eps)
        self.in_proj       = nn.Linear(d, in_out, bias=True)
        self.x_up_proj     = TuckerMoE(dim_x, H * P * R, cfg)
        self.out_proj      = TuckerMoE(d, d, cfg)
        self.y_down_proj   = nn.Linear(P * R, P, bias=False)
        self.mamba_dense_proj = nn.Linear(H * P, d, bias=False)
        self.pre_gate_norm = RMSNorm(H * P, eps=cfg.rms_norm_eps)
        self.norm_out_proj = RMSNorm(d, eps=cfg.rms_norm_eps)
        self.norm_B        = RMSNorm(N * R, eps=cfg.rms_norm_eps)
        self.norm_C        = RMSNorm(N * R, eps=cfg.rms_norm_eps)
        self.ls_mamba      = LayerScale(d)
        self.ls_out_proj   = LayerScale(d)

        self.register_parameter("theta_log", nn.Parameter(torch.zeros(G, N // 2)))
        self.register_parameter("D",        nn.Parameter(torch.ones(H)))
        self.register_buffer("bias_B",      torch.ones(G, N, R))
        self.register_buffer("bias_C",      torch.ones(G, N, R))

    # ── State helpers ──────────────────────────────────────────────────────────

    def init_state(self, device, dtype) -> dict:
        H, N, P = self.H, self.N, self.P
        return {
            "h_prev":            torch.zeros(1, H, N, P, device=device, dtype=dtype),
            "prev_input_signal": torch.zeros(1, H, N, P, device=device, dtype=dtype),
            "angles_cum":        torch.zeros(1, H, N // 2, device=device, dtype=torch.float32),
        }

    # ── Decode step (L=1) ─────────────────────────────────────────────────────

    def decode(self, x: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        """
        x:     (1, 1, d_model) — batch=1, seq=1
        state: dict with h_prev, prev_input_signal, angles_cum
        returns (out, new_state)
        """
        H, G, P, N, R = self.H, self.G, self.P, self.N, self.R
        h_prev   = state["h_prev"]           # (1, H, N, P)
        prev_ips = state["prev_input_signal"] # (1, H, N, P)
        ang_cum  = state["angles_cum"]        # (1, H, N//2)  float32

        residual = x                          # (1, 1, d)
        u   = self.norm_mamba(x)              # (1, 1, d)
        raw = self.in_proj(u)                 # (1, 1, in_out)

        z       = raw[..., :self.dim_z]
        x_prime = raw[..., self.dim_z : self.dim_z + self.dim_x]
        B_param = raw[..., self.dim_z + self.dim_x : self.dim_z + self.dim_x + self.dim_B]
        C_param = raw[..., self.dim_z + self.dim_x + self.dim_B : self.dim_z + self.dim_x + self.dim_B + self.dim_C]
        offs    = self.dim_z + self.dim_x + self.dim_B + self.dim_C
        dt_p    = raw[..., offs : offs + G]
        A_p     = raw[..., offs + G : offs + 2 * G]
        lam_p   = raw[..., offs + 2 * G : offs + 3 * G]

        # x_up_proj: (1,1,H*P) → (1,1,H*P*R)
        x_up  = self.x_up_proj(x_prime)       # (1, 1, H*P*R)
        x_ssm = x_up.view(1, 1, H, P, R)      # (1, 1, H, P, R)

        # Discretise
        dt  = F.softplus(dt_p)                 # (1, 1, G)
        A   = -torch.exp(A_p)                  # (1, 1, G)
        lam = torch.sigmoid(lam_p)             # (1, 1, G)

        # Broadcast G → H
        dt_h = dt.repeat_interleave(self.ratio, dim=-1)   # (1, 1, H)
        A_h  = A.repeat_interleave(self.ratio, dim=-1)
        lam_h = lam.repeat_interleave(self.ratio, dim=-1)

        av_h = torch.exp(dt_h * A_h)           # (1, 1, H)

        # RoPE angles
        theta = torch.exp(self.theta_log.float())        # (G, N//2)
        theta_h = theta.repeat_interleave(self.ratio, dim=0)  # (H, N//2)
        delta_angle = dt_h[0, 0, :, None].float() * theta_h  # (H, N//2)
        new_ang = ang_cum[:, :, :] + delta_angle          # (1, H, N//2)
        angles  = new_ang.to(x.dtype)                     # (1, H, N//2)
        sin_a   = torch.sin(angles)                        # (1, H, N//2)
        cos_a   = torch.cos(angles)

        # B, C: norm + bias + RoPE
        B_n = self.norm_B(B_param.view(1, 1, G, N * R)).view(1, 1, G, N, R)
        B_n = B_n + self.bias_B.unsqueeze(0).unsqueeze(0)
        B_h_t = B_n.repeat_interleave(self.ratio, dim=2)          # (1, 1, H, N, R)
        B_rot = apply_rope(B_h_t[0, 0], sin_a[0], cos_a[0])       # (H, N, R)

        C_n = self.norm_C(C_param.view(1, 1, G, N * R)).view(1, 1, G, N, R)
        C_n = C_n + self.bias_C.unsqueeze(0).unsqueeze(0)
        C_h_t = C_n.repeat_interleave(self.ratio, dim=2)
        C_rot = apply_rope(C_h_t[0, 0], sin_a[0], cos_a[0])       # (H, N, R)

        # input_signal: (H, N, P)
        inp_sig = torch.einsum("hnr,hpr->hnp", B_rot, x_ssm[0, 0])

        # SSM recurrence
        dt_v  = dt_h[0, 0]                    # (H,)
        av_v  = av_h[0, 0]                    # (H,)
        lv    = lam_h[0, 0]                   # (H,)

        av4   = av_v[:, None, None]            # (H, 1, 1)
        dt4   = dt_v[:, None, None]
        lv4   = lv[:, None, None]

        u_ssm = lv4 * dt4 * inp_sig + (1 - lv4) * dt4 * av4 * prev_ips[0]
        h_new = av4 * h_prev[0] + u_ssm      # (H, N, P)

        # Output: y = C_rot ⊗ h_new → (H, P, R)
        y_stack = torch.einsum("hnp,hnr->hpr", h_new, C_rot)   # (H, P, R)
        y_flat  = y_stack.reshape(H, P * R)                     # (H, P*R)
        y       = self.y_down_proj(y_flat).reshape(1, 1, H * P) # (1, 1, H*P)

        # D skip
        D_e = self.D.repeat_interleave(P)      # (H*P,)
        y   = y + x_prime * D_e

        # Gate
        y_ng    = self.pre_gate_norm(y)
        gated   = y_ng * F.silu(z)             # (1, 1, H*P)
        m_out   = self.mamba_dense_proj(gated) # (1, 1, d)
        mid     = residual + self.ls_mamba(m_out)

        # out_proj
        mid_n   = self.norm_out_proj(mid)
        proj    = self.out_proj(mid_n)
        out     = mid + self.ls_out_proj(proj)

        new_state = {
            "h_prev":            h_new.unsqueeze(0),
            "prev_input_signal": inp_sig.unsqueeze(0),
            "angles_cum":        new_ang,
        }
        return out, new_state


# ── TransformerBlock ───────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        d   = cfg.d_model
        n_q = d // cfg.d_head           # 12 Q heads
        n_kv = (d // 4) // cfg.d_head  # 4 KV heads (GQA: d//4=192, /64=3 ... hmm)

        # Actual shapes from checkpoint: k_proj (256, 768), q_proj (768, 768)
        # 256 / 64 = 4 KV heads; 768 / 64 = 12 Q heads
        self.n_q  = 768 // cfg.d_head   # 12
        self.n_kv = 256 // cfg.d_head   # 4
        self.d_head = cfg.d_head

        self.norm_attn = RMSNorm(d, eps=cfg.rms_norm_eps)
        self.q_proj    = nn.Linear(d, 768, bias=False)
        self.k_proj    = nn.Linear(d, 256, bias=False)
        self.v_proj    = nn.Linear(d, 256, bias=False)
        self.o_proj    = nn.Linear(768, d, bias=True)
        self.ls_attn   = LayerScale(d)

        self.norm_ffn  = RMSNorm(d, eps=cfg.rms_norm_eps)
        self.ffn_gate  = TuckerMoE(d, cfg.ffn_expand * d, cfg)
        self.ffn_up    = TuckerMoE(d, cfg.ffn_expand * d, cfg)
        self.ffn_down  = TuckerMoE(cfg.ffn_expand * d, d, cfg)
        self.ls_ffn    = LayerScale(d)

    def init_state(self, kv_len: int, device, dtype) -> dict:
        return {
            "k": torch.zeros(1, self.n_kv, kv_len, self.d_head, device=device, dtype=dtype),
            "v": torch.zeros(1, self.n_kv, kv_len, self.d_head, device=device, dtype=dtype),
            "write_pos": 0,
        }

    def decode(self, x: torch.Tensor, state: dict) -> tuple[torch.Tensor, dict]:
        """x: (1, 1, d_model)"""
        K_cache = state["k"]              # (1, n_kv, kv_len, d_head)
        V_cache = state["v"]
        wpos    = state["write_pos"]
        ratio   = self.n_q // self.n_kv  # 3 = 12 // 4

        # Attention
        u = self.norm_attn(x)             # (1, 1, d)
        q = self.q_proj(u)                # (1, 1, 768)
        k = self.k_proj(u)                # (1, 1, 256)
        v = self.v_proj(u)                # (1, 1, 256)

        # Write into KV cache
        K_cache[:, :, wpos, :] = k.view(1, self.n_kv, self.d_head)
        V_cache[:, :, wpos, :] = v.view(1, self.n_kv, self.d_head)

        seq  = wpos + 1
        K_s  = K_cache[:, :, :seq]        # (1, n_kv, seq, d_head)
        V_s  = V_cache[:, :, :seq]
        # GQA: expand KV heads to match Q heads
        K_s  = K_s.repeat_interleave(ratio, dim=1)    # (1, n_q, seq, d_head)
        V_s  = V_s.repeat_interleave(ratio, dim=1)

        # SDPA expects (batch, heads, seq_q, d_head)
        q_s  = q.view(1, self.n_q, 1, self.d_head)    # (1, n_q, 1, d_head)
        y    = F.scaled_dot_product_attention(q_s, K_s, V_s)  # (1, n_q, 1, d_head)

        y_out  = self.o_proj(y.reshape(1, 1, self.n_q * self.d_head))  # (1, 1, d)
        x_res  = x + self.ls_attn(y_out)

        # FFN
        x_f  = self.norm_ffn(x_res)
        gate = self.ffn_gate(x_f)
        up   = self.ffn_up(x_f)
        dn   = self.ffn_down(F.silu(gate) * up)
        x_res = x_res + self.ls_ffn(dn)

        nxt = min(wpos + 1, K_cache.shape[2] - 1)
        new_state = {"k": K_cache, "v": V_cache, "write_pos": nxt}
        return x_res, new_state


# ── Full Language Model ────────────────────────────────────────────────────────

class Mamba3LM(nn.Module):
    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.norm   = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        types = cfg.block_types()
        self.blocks: nn.ModuleList = nn.ModuleList()
        for t in types:
            if t == "mamba":
                self.blocks.append(Mamba3Block(cfg))
            else:
                self.blocks.append(TransformerBlock(cfg))
        self._block_types = types

    def init_states(self, kv_len: int = 512) -> list[dict]:
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        states = []
        for t, blk in zip(self._block_types, self.blocks):
            if t == "mamba":
                states.append(blk.init_state(device, dtype))
            else:
                states.append(blk.init_state(kv_len, device, dtype))
        return states

    def prefill(self, prompt_ids: list[int], kv_len: int = 512):
        """Run full prefill (sequential per token for simplicity) — returns last logits + states."""
        states = self.init_states(kv_len)
        logits = None
        for tid in prompt_ids:
            x = self.embed.weight[tid].view(1, 1, -1)  # (1, 1, d)
            for i, (t, blk) in enumerate(zip(self._block_types, self.blocks)):
                x, states[i] = blk.decode(x, states[i])
            x_norm = self.norm(x)
            logits = self.head(x_norm)[0, 0]           # (V,)
        return logits, states

    def decode_step(self, tok_id: int, states: list[dict]):
        """One autoregressive decode step — returns (logits, new_states)."""
        x = self.embed.weight[tok_id].view(1, 1, -1)   # (1, 1, d)
        for i, (t, blk) in enumerate(zip(self._block_types, self.blocks)):
            x, states[i] = blk.decode(x, states[i])
        x_norm = self.norm(x)
        logits = self.head(x_norm)[0, 0]                # (V,)
        return logits, states
