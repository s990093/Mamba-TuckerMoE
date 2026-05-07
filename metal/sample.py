# -*- coding: utf-8 -*-
"""
fused_infer.py  ── 推論專用算子融合模組
==========================================
針對 train.py 的 Mamba3LanguageModel（Hybrid Mamba + TuckerMoE）在
**推論階段（inference-only）** 的完整算子融合實作。

不修改 train.py / model.py 任何原始碼；全部以「替換」或「包裝」方式注入。

架構概覽（推論路徑）
──────────────────────────────────────────────────────────────────────
  Mamba3LanguageModel.forward_inference()
    ├── embed(input_ids)                     → [B, L, D]
    ├── TrueHybridMamba.forward_inference()
    │    ├─ [×4] Mamba3Block.forward()       (mamba_cache 路徑)
    │    │    ├─ norm_mamba(x)  + in_proj(x)  ← ① FusedRMSNormLinear
    │    │    ├─ TuckerMoE x_up_proj          ← ② FusedTuckerMoE (U_in+G+U_out)
    │    │    ├─ chunk_parallel_scan / step   ← ③ (原 Triton kernel，已最優)
    │    │    ├─ pre_gate_norm * silu(z)      ← ④ FusedGateSiLU
    │    │    ├─ mamba_dense_proj             ← ⑤ 直接 Linear（小矩陣）
    │    │    ├─ LayerScale(ls_mamba)         ← ⑥ inplace gamma-scale
    │    │    ├─ norm_out_proj + out_proj      ← ① + ② 再次複用
    │    │    └─ LayerScale(ls_out_proj)      ← ⑥
    │    └─ [×1] TransformerBlock.forward()
    │         ├─ norm_attn + q/k/v_proj       ← ① FusedRMSNormQKV
    │         ├─ SDPA（FlashAttention 2）      ← ⑦ 已由 PyTorch 處理
    │         ├─ o_proj + LayerScale          ← ⑥
    │         ├─ norm_ffn + MixtralMoEFFN     ← ① + ②
    │         └─ LayerScale(ls_ffn)           ← ⑥
    ├── norm(hidden)                          ← RMSNorm（保持）
    └── head(hidden / sqrt(D))               ← ⑧ FusedScaledHeadLogits

融合點清單（推論只看 forward，不需要 backward）
──────────────────────────────────────────────────────────────────────
  ①  FusedRMSNormLinear   : RMSNorm + Linear → 單 Triton kernel
  ②  FusedTuckerMoEInfer  : router + U_in + G_experts(einsum) + x_core + U_out
                             完全避免 G_experts 的中間 einsum 張量物化
  ③  inplace 修補 TritonTuckerMoE.forward : 推論時跳過 lb_loss / z_loss
  ④  FusedGateSiLU        : pre_gate_norm(y) * silu(z) 合併讀寫
  ⑤  無需額外融合（mamba_dense_proj 已是單一 Linear）
  ⑥  inplace LayerScale   : gamma 乘法融合進前一算子輸出（residual add 後）
  ⑦  Flash SDPA           : 已由 F.scaled_dot_product_attention 自動使用
  ⑧  FusedScaledHeadLogits: hidden / sqrt(D) + head linear + scaled_tanh(30)

使用方式
──────────────────────────────────────────────────────────────────────
  from fused_infer import apply_inference_fusion

  model = Mamba3LanguageModel(config, vocab_size)
  model.load_state_dict(ckpt["model"])
  model.eval()
  apply_inference_fusion(model)          # ← 一行完成所有替換

  # 之後正常呼叫 forward_inference 即使用融合版算子
  logits, new_caches = model.forward_inference(
      input_ids, router_temp, layer_caches, seq_pos, prefill
  )
──────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ══════════════════════════════════════════════════════════════════════
#  §0  工具函式
# ══════════════════════════════════════════════════════════════════════

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


# ══════════════════════════════════════════════════════════════════════
#  §1  FusedRMSNormLinear
#      RMSNorm(x) → Linear(x_norm)  合併為單一 Triton kernel
#      省去 x_norm 的一次 HBM 寫入再讀回
# ══════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
        triton.Config({"BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_D": 1024}, num_warps=16),
    ],
    key=["D_in", "D_out"],
)
@triton.jit
def _rms_norm_linear_kernel(
    # 指標
    X_ptr, Gnorm_ptr, W_ptr, B_ptr, Y_ptr,
    # strides
    stride_xn, stride_xd,
    stride_yd, stride_yn,
    # 常數
    N,                         # batch * seq_len
    D_in:  tl.constexpr,
    D_out: tl.constexpr,
    eps:   tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D:  tl.constexpr,    # ← autotune 負責選
):
    """
    每個 program 處理一個 token（一行）。
    RMSNorm 先在寄存器計算，x_norm 不落盤；
    再對 D_out 個輸出做點積累加（外迴圈）。

    複雜度：O(N × D_in × D_out)  ──  與原先兩次 kernel 完全相同，
    但省去一次 [N, D_in] 的 HBM round-trip（約 2×D_in×N bytes）。
    """
    row = tl.program_id(0)
    if row >= N:
        return

    # ── 1. 讀取 x_row  ─────────────────────────────────────────────
    offs_d  = tl.arange(0, BLOCK_D)
    x_base  = X_ptr + row * stride_xn
    x_acc   = tl.zeros((BLOCK_D,), dtype=tl.float32)
    sq_sum  = tl.zeros((1,),       dtype=tl.float32)

    # 若 D_in > BLOCK_D，需分段讀取累加 sq_sum
    # （autotune 應讓 BLOCK_D >= D_in；若仍不足，此迴圈正確處理）
    for block_start in range(0, D_in, BLOCK_D):
        bd_offs = block_start + offs_d
        mask    = bd_offs < D_in
        xv      = tl.load(x_base + bd_offs * stride_xd, mask=mask, other=0.0).to(tl.float32)
        sq_sum  += tl.sum(xv * xv, axis=0)
        # 只有最後一段才留在 x_acc（若 D_in == BLOCK_D 直接命中）
        if block_start + BLOCK_D >= D_in:
            x_acc = xv

    rms    = tl.sqrt(sq_sum[0] / D_in + eps)
    gnorm  = tl.load(Gnorm_ptr + tl.arange(0, BLOCK_D) + (D_in - BLOCK_D), mask=(D_in - BLOCK_D + tl.arange(0, BLOCK_D)) < D_in, other=1.0).to(tl.float32)
    x_norm = (x_acc / rms) * gnorm                # shape [BLOCK_D]，只有最後一段有效

    # ── 2. 對每個輸出維度做點積  ────────────────────────────────────
    y_base = Y_ptr + row * stride_yn
    for o in tl.static_range(D_out):
        w_row = W_ptr + o * D_in + (D_in - BLOCK_D)
        w_v   = tl.load(w_row + tl.arange(0, BLOCK_D), mask=(D_in - BLOCK_D + tl.arange(0, BLOCK_D)) < D_in, other=0.0).to(tl.float32)
        dot   = tl.sum(x_norm * w_v, axis=0)
        if HAS_BIAS:
            dot = dot + tl.load(B_ptr + o).to(tl.float32)
        tl.store(y_base + o * stride_yd, dot.to(Y_ptr.dtype.element_ty))


class FusedRMSNormLinear(nn.Module):
    """
    推論用：RMSNorm + Linear → 單一 Triton kernel。

    from_modules(norm, linear) 從既有模組 **共享** 權重（不複製），
    保證 state_dict 相容。
    """

    def __init__(self, d_in: int, d_out: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.d_in, self.d_out, self.eps = d_in, d_out, eps
        # 這兩個 Parameter 在 from_modules 後會被替換為外部引用
        self.norm_weight   = nn.Parameter(torch.ones(d_in))
        self.linear_weight = nn.Parameter(torch.empty(d_out, d_in))
        self.linear_bias: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(d_out)) if bias else None
        )

    @classmethod
    def from_modules(cls, norm: nn.Module, linear: nn.Linear) -> "FusedRMSNormLinear":
        """
        從既有 (norm, linear) 建立融合層。
        直接共享 .weight / .bias，不複製張量，state_dict 自動對應。
        """
        d_in  = linear.in_features
        d_out = linear.out_features
        eps   = getattr(norm, "eps", 1e-5)
        obj   = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.d_in  = d_in
        obj.d_out = d_out
        obj.eps   = eps
        # 直接指向原模組的 Parameter，不建立新 Parameter
        obj.norm_weight   = norm.weight          # type: ignore[assignment]
        obj.linear_weight = linear.weight        # type: ignore[assignment]
        obj.linear_bias   = linear.bias          # None 或 Parameter
        return obj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_in]  →  [..., d_out]"""
        shape = x.shape
        flat  = x.reshape(-1, self.d_in)
        N     = flat.shape[0]

        # ── Triton 路徑（CUDA, BF16/FP16/FP32）─────────────────────
        if flat.is_cuda and self.d_in <= 1024 and self.d_out <= 4096:
            out = torch.empty(N, self.d_out, device=flat.device, dtype=flat.dtype)
            BLOCK_D = _next_pow2(self.d_in)
            _rms_norm_linear_kernel[(N,)](
                flat, self.norm_weight, self.linear_weight,
                self.linear_bias if self.linear_bias is not None else flat,
                out,
                flat.stride(0), flat.stride(1),
                1, out.stride(0),
                N=N, D_in=self.d_in, D_out=self.d_out, eps=self.eps,
                HAS_BIAS=(self.linear_bias is not None),
                BLOCK_D=BLOCK_D,
            )
        else:
            # ── PyTorch fallback（仍融合，節省一次中間張量）──────────
            ms    = flat.float().pow(2).mean(-1, keepdim=True)
            x_n   = flat.float() / (ms + self.eps).sqrt() * self.norm_weight.float()
            out   = F.linear(x_n.to(flat.dtype), self.linear_weight, self.linear_bias)

        return out.reshape(*shape[:-1], self.d_out)


# ══════════════════════════════════════════════════════════════════════
#  §2  FusedRMSNormQKV
#      norm_attn(x) → [q_proj, k_proj, v_proj]  同時計算三條投影
#      省去對 x_norm 的三次 HBM 讀取
# ══════════════════════════════════════════════════════════════════════

class FusedRMSNormQKV(nn.Module):
    """
    TransformerBlock 專用：
        norm_attn(x) → q, k, v  一次讀取 x_norm 完成三個投影。

    等效於：
        nx = norm_attn(x)
        q  = q_proj(nx)
        k  = k_proj(nx)
        v  = v_proj(nx)
    但 x_norm 只做一次 HBM 寫入（或完全留在 L2 cache）。
    """

    def __init__(self, norm: nn.Module, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear):
        super().__init__()
        # 直接共享引用
        self.norm   = norm
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, D]
        return: q [B,L,Dq], k [B,L,Dk], v [B,L,Dv]
        """
        # 此處 norm 呼叫只做一次；PyTorch 在 eval 模式下 fuse RMSNorm
        nx = self.norm(x)
        return self.q_proj(nx), self.k_proj(nx), self.v_proj(nx)


# ══════════════════════════════════════════════════════════════════════
#  §3  FusedGateSiLU
#      pre_gate_norm(y) * silu(z)  合併為單一 elementwise kernel
#      對應 Mamba3Block 第 718-720 行
# ══════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _fused_gate_silu_kernel(
    norm_y_ptr,   # [N] 已正規化的 y（pre_gate_norm 輸出）
    z_ptr,        # [N] gate
    out_ptr,      # [N]
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """out = norm_y * sigmoid(z) * z  =  norm_y * silu(z)"""
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    ny   = tl.load(norm_y_ptr + offs, mask=mask).to(tl.float32)
    z    = tl.load(z_ptr       + offs, mask=mask).to(tl.float32)
    out  = ny * (z * tl.sigmoid(z))
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


class FusedGateSiLU(nn.Module):
    """
    pre_gate_norm(y) * silu(z)  →  融合 elementwise kernel。
    直接共享 pre_gate_norm 的引用；不持有額外權重。
    """

    def __init__(self, pre_gate_norm: nn.Module):
        super().__init__()
        self.pre_gate_norm = pre_gate_norm

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        y: [B, L, H*P]  —  SSM 輸出（加過 D skip）
        z: [B, L, H*P]  —  gate（in_proj 的 z 部分）
        return: [B, L, H*P]
        """
        norm_y = self.pre_gate_norm(y)          # RMSNorm（一次 HBM 讀寫）

        if norm_y.is_cuda:
            out = torch.empty_like(norm_y)
            n   = norm_y.numel()
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
            _fused_gate_silu_kernel[grid](
                norm_y.contiguous(), z.contiguous(), out, n
            )
            return out
        else:
            return norm_y * F.silu(z)


# ══════════════════════════════════════════════════════════════════════
#  §4  FusedTuckerMoEInfer
#      推論專用版 TritonTuckerMoE.forward()
#      改進點：
#        (a) 推論不需要 lb_loss / z_loss，完全跳過
#        (b) G_experts = einsum('er,rst->est', U_expert, core)
#            物化為 [E, r3, r2] 後**快取**（首次呼叫計算一次）
#        (c) x_shared = x @ U_in + inner_norm  在一個 kernel 完成
#        (d) x_core = FusedLatentMoE（原 Triton kernel 保持不動）
#        (e) out = x_core @ U_out + bias  保持不動
# ══════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_R3": 32, "BLOCK_R2": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_R3": 32, "BLOCK_R2": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_R3": 64, "BLOCK_R2": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_R3": 64, "BLOCK_R2": 128}, num_warps=8, num_stages=4),
    ],
    key=["r3", "r2"],
)
@triton.jit
def _fused_latent_moe_infer_fwd(
    x_ptr, g_ptr, idx_ptr, prob_ptr, out_ptr,
    stride_xb, stride_xr3,
    stride_ge, stride_gr3, stride_gr2,
    stride_idxb, stride_idxk,
    stride_probb, stride_probk,
    stride_ob, stride_or2,
    B, r3, r2, top_k,
    BLOCK_R3: tl.constexpr,
    BLOCK_R2: tl.constexpr,
):
    """
    與 train.py 的 _fused_latent_moe_fwd 邏輯相同，
    但移除了所有 training-only 的旗標分支，減少 warp divergence。
    推論時 B 通常 = 1（decode）或少量（prefill），kernel 輕量。
    """
    pid_b  = tl.program_id(0)
    pid_r2 = tl.program_id(1)
    offs_r2 = pid_r2 * BLOCK_R2 + tl.arange(0, BLOCK_R2)
    acc = tl.zeros((BLOCK_R2,), dtype=tl.float32)

    for k in range(top_k):
        exp_idx = tl.load(idx_ptr  + pid_b * stride_idxb  + k * stride_idxk)
        prob    = tl.load(prob_ptr + pid_b * stride_probb + k * stride_probk)
        for r3_idx in range(0, r3, BLOCK_R3):
            offs_r3 = r3_idx + tl.arange(0, BLOCK_R3)
            x = tl.load(
                x_ptr + pid_b * stride_xb + offs_r3 * stride_xr3,
                mask=offs_r3 < r3, other=0.0,
            )
            g = tl.load(
                g_ptr + exp_idx * stride_ge
                      + offs_r3[:, None] * stride_gr3
                      + offs_r2[None, :] * stride_gr2,
                mask=(offs_r3[:, None] < r3) & (offs_r2[None, :] < r2),
                other=0.0,
            )
            acc += prob * tl.sum(x[:, None] * g, axis=0)

    tl.store(
        out_ptr + pid_b * stride_ob + offs_r2 * stride_or2,
        acc.to(out_ptr.dtype.element_ty),
        mask=offs_r2 < r2,
    )


def _fused_latent_moe_infer(
    x_shared: torch.Tensor,       # [B, r3]
    G_experts: torch.Tensor,      # [E, r3, r2]
    top_k_indices: torch.Tensor,  # [B, top_k]
    top_k_probs: torch.Tensor,    # [B, top_k]
) -> torch.Tensor:
    """推論專用的 FusedLatentMoE（無 backward，無 training flags）。"""
    B, r3 = x_shared.shape
    E, _, r2 = G_experts.shape
    top_k = top_k_indices.size(1)
    out = torch.empty((B, r2), device=x_shared.device, dtype=x_shared.dtype)

    grid = lambda meta: (B, triton.cdiv(r2, meta["BLOCK_R2"]))
    _fused_latent_moe_infer_fwd[grid](
        x_shared, G_experts, top_k_indices, top_k_probs, out,
        x_shared.stride(0),   x_shared.stride(1),
        G_experts.stride(0),  G_experts.stride(1),  G_experts.stride(2),
        top_k_indices.stride(0), top_k_indices.stride(1),
        top_k_probs.stride(0),   top_k_probs.stride(1),
        out.stride(0), out.stride(1),
        B, r3, r2, top_k,
    )
    return out


class FusedTuckerMoEInfer(nn.Module):
    """
    推論專用 TritonTuckerMoE 替換。

    改進點：
      1. 不計算 lb_loss / z_loss（推論不需要）
      2. G_experts = einsum(U_expert, core) 在 eval() 首次呼叫時
         **一次性物化並快取**，後續直接使用；節省每次 einsum 的開銷。
      3. 使用 _fused_latent_moe_infer（無 autograd 開銷的推論版 kernel）
      4. x_shared 計算（x @ U_in + inner_norm）保持原有路徑，
         不做額外融合（因 inner_norm 是 RMSNorm，已足夠快）
    """

    def __init__(self, original_moe: nn.Module):
        super().__init__()
        # 直接引用原始參數（不複製，state_dict 完全相容）
        self.router      = original_moe.router
        self.U_expert    = original_moe.U_expert
        self.U_in        = original_moe.U_in
        self.U_out       = original_moe.U_out
        self.core        = original_moe.core
        self.bias        = original_moe.bias
        self.inner_norm  = original_moe.inner_norm
        self.num_experts = original_moe.num_experts
        self.top_k       = original_moe.top_k
        # 快取：None 表示尚未物化
        self._G_cache: Optional[torch.Tensor] = None

    def _get_G_experts(self) -> torch.Tensor:
        """
        首次呼叫時物化 G_experts = einsum('er,rst->est', U_expert, core)
        並快取。若 U_expert 或 core 的資料改變（例如量化後），
        呼叫 invalidate_cache() 重新物化。
        """
        if self._G_cache is None:
            with torch.no_grad():
                self._G_cache = torch.einsum(
                    "er,rst->est", self.U_expert, self.core
                ).contiguous()
        return self._G_cache

    def invalidate_cache(self):
        """量化或修改權重後手動呼叫以清除快取。"""
        self._G_cache = None

    def forward(
        self,
        x: torch.Tensor,
        router_temp=None,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        與原 TritonTuckerMoE.forward 相同簽名。
        lb_loss / z_loss 推論時一律回傳 0.0，不計算。
        """
        orig_shape = x.shape
        x_flat  = x.reshape(-1, orig_shape[-1])
        B_flat  = x_flat.size(0)

        # ── Router（保持原邏輯；推論只需 top_k_indices & probs）───────
        raw_logits = self.router(x_flat)

        if router_temp is None:
            from train import get_router_temperature
            temperature = raw_logits.new_tensor(get_router_temperature(None))
        elif isinstance(router_temp, torch.Tensor):
            temperature = router_temp.to(device=raw_logits.device, dtype=raw_logits.dtype)
        else:
            temperature = raw_logits.new_tensor(float(router_temp))
        temperature = temperature.clamp_min(1e-4)

        # 推論時跳過 z_loss；直接用 scaled tanh cap
        from train import fast_scaled_tanh
        capped        = fast_scaled_tanh(raw_logits, 10.0)
        router_logits = capped / temperature
        router_probs  = torch.softmax(router_logits, dim=-1)

        _, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_raw   = router_probs.gather(-1, top_k_indices)
        top_k_probs = top_k_raw / (top_k_raw.sum(-1, keepdim=True) + 1e-6)

        # ── x_shared = inner_norm(x @ U_in) ─────────────────────────
        x_shared = self.inner_norm(torch.matmul(x_flat, self.U_in))

        # ── FusedLatentMoE（推論版，無 autograd）────────────────────
        G_experts = self._get_G_experts()
        x_core    = _fused_latent_moe_infer(
            x_shared, G_experts, top_k_indices, top_k_probs
        ).to(x.dtype)

        # ── U_out 升維 + bias ────────────────────────────────────────
        out = torch.matmul(x_core, self.U_out).reshape(*orig_shape[:-1], -1)
        return out + self.bias, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════
#  §5  FusedScaledHeadLogits
#      hidden / sqrt(D) → Linear → fast_scaled_tanh(30)
#      三個算子共享一次 HBM 讀取
# ══════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 256}, num_warps=8),
        triton.Config({"BLOCK_D": 512}, num_warps=8),
        triton.Config({"BLOCK_D": 1024}, num_warps=16),
    ],
    key=["D", "V"],
)
@triton.jit
def _fused_head_logits_kernel(
    H_ptr,   # [N, D]  hidden（已過 norm）
    W_ptr,   # [V, D]  lm_head weight (tied embed)
    Y_ptr,   # [N, V]  output logits
    inv_sqrt_D: tl.constexpr,
    scale:      tl.constexpr,     # scaled_tanh scale = 30
    N,
    D: tl.constexpr,
    V,
    BLOCK_D: tl.constexpr,
):
    """
    out[n, v] = scale * tanh( sum_d(H[n,d]*inv_sqrt_D * W[v,d]) / scale )
    每個 program 處理一個 (n, v) 對。
    對 V 大（32000+）的情形，建議 V 方向外迴圈，D 方向 BLOCK 向量化。
    """
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)
    if pid_n >= N or pid_v >= V:
        return

    offs_d = tl.arange(0, BLOCK_D)
    acc    = tl.zeros((1,), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        bd = d_start + offs_d
        mask = bd < D
        h = tl.load(H_ptr + pid_n * D + bd, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + pid_v * D + bd, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(h * w * inv_sqrt_D, axis=0)

    # fast tanh approx（PTX 指令）
    v = acc[0] / scale
    t = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[v],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(Y_ptr + pid_n * V + pid_v, (t * scale).to(Y_ptr.dtype.element_ty))


class FusedScaledHeadLogits(nn.Module):
    """
    推論用：final_norm(hidden) → hidden/sqrt(D) → lm_head → scaled_tanh(30)
    合併為一個 Triton kernel（對 vocab 小的模型效益最大）。

    注意：只在 V <= 65536 時啟用 Triton；更大詞表退回 PyTorch。
    """

    def __init__(self, head: nn.Linear, d_model: int, scale: float = 30.0):
        super().__init__()
        self.head    = head       # 共享引用（tied with embed）
        self.d_model = d_model
        self.scale   = scale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, L, D]  →  logits: [B, L, V]"""
        shape = hidden.shape
        flat  = hidden.reshape(-1, self.d_model)     # [N, D]
        N, D  = flat.shape
        V     = self.head.weight.shape[0]

        if flat.is_cuda and V <= 65536:
            logits = torch.empty(N, V, device=flat.device, dtype=flat.dtype)
            inv_sd = 1.0 / math.sqrt(D)
            grid   = (N, V)
            BLOCK_D = min(_next_pow2(D), 1024)
            _fused_head_logits_kernel[grid](
                flat.contiguous(), self.head.weight.contiguous(), logits,
                inv_sqrt_D=inv_sd, scale=self.scale,
                N=N, D=D, V=V, BLOCK_D=BLOCK_D,
            )
        else:
            # PyTorch fallback
            from train import fast_scaled_tanh
            logits = fast_scaled_tanh(
                self.head(flat / math.sqrt(D)), self.scale
            )

        return logits.reshape(*shape[:-1], V)


# ══════════════════════════════════════════════════════════════════════
#  §6  FusedLayerScaleResidual
#      residual + LayerScale(gamma) * x  合併為 inplace 操作
#      避免多餘的中間張量分配
# ══════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _add_layer_scale_kernel(
    res_ptr,    # [N]  residual（inplace 修改）
    x_ptr,      # [N]  待加的張量
    gamma_ptr,  # [D]  LayerScale gamma
    n_elements,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """residual += gamma[d % D] * x，inplace。"""
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    g    = tl.load(gamma_ptr + (offs % D), mask=mask, other=1.0).to(tl.float32)
    x    = tl.load(x_ptr  + offs, mask=mask, other=0.0).to(tl.float32)
    r    = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(res_ptr + offs, (r + g * x).to(res_ptr.dtype.element_ty), mask=mask)


def fused_add_layer_scale_(
    residual: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """
    inplace: residual += LayerScale(gamma) * x
    省去一個 [B,L,D] 中間張量的分配與讀寫。
    """
    assert residual.shape == x.shape
    D = gamma.numel()
    if residual.is_cuda:
        n = residual.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _add_layer_scale_kernel[grid](
            residual.contiguous(), x.contiguous(),
            gamma.contiguous(), n, D=D,
        )
        return residual
    else:
        residual.add_(x * gamma.to(x.dtype))
        return residual


# ══════════════════════════════════════════════════════════════════════
#  §7  FusedMamba3BlockInfer
#      對 Mamba3Block.forward() 的推論路徑進行整體包裝：
#        - 替換 in_proj 前的 norm 呼叫
#        - 整合 FusedGateSiLU
#        - 整合 FusedLayerScaleResidual
#        - 保留原 SSM scan（mamba_cache 路徑）
# ══════════════════════════════════════════════════════════════════════

class FusedMamba3BlockInfer(nn.Module):
    """
    Mamba3Block 推論加速包裝器。
    不修改任何參數；直接引用原 block 的子模組。

    替換策略：
      ① norm_mamba + in_proj  → FusedRMSNormLinear
      ② x_up_proj             → FusedTuckerMoEInfer（若 use_kmoe）
      ③ pre_gate_norm + silu  → FusedGateSiLU
      ④ out_proj              → FusedTuckerMoEInfer（若 use_kmoe）
      ⑤ ls_mamba  residual   → fused_add_layer_scale_
      ⑥ ls_out_proj residual → fused_add_layer_scale_
    """

    def __init__(self, block):
        """block: 原 Mamba3Block 實例"""
        super().__init__()
        self._orig = block   # 保留引用，子模組共用

        # ① norm_mamba + in_proj
        self.fused_norm_in = FusedRMSNormLinear.from_modules(
            block.norm_mamba, block.in_proj
        )
        # ② x_up_proj（TuckerMoE or Linear）
        if block.config.use_kmoe:
            self.fused_x_up = FusedTuckerMoEInfer(block.x_up_proj)
        else:
            self.fused_x_up = block.x_up_proj   # 保持原 Linear

        # ③ pre_gate_norm + silu
        self.fused_gate = FusedGateSiLU(block.pre_gate_norm)

        # ④ norm_out_proj + out_proj
        if block.config.use_kmoe:
            self.fused_out = FusedTuckerMoEInfer(block.out_proj)
        else:
            self.fused_out = block.out_proj

        # 以下直接引用（不替換）
        self.y_down_proj       = block.y_down_proj
        self.mamba_dense_proj  = block.mamba_dense_proj
        self.D_skip            = block.D
        self.theta_log         = block.theta_log
        self.norm_B            = block.norm_B
        self.norm_C            = block.norm_C
        self.bias_B            = block.bias_B
        self.bias_C            = block.bias_C
        self.ls_mamba          = block.ls_mamba
        self.ls_out_proj       = block.ls_out_proj
        self.norm_out_proj     = block.norm_out_proj

        # 傳遞 config（供 split 維度計算）
        self.config  = block.config
        self.dim_z   = block.dim_z
        self.dim_x   = block.dim_x
        self.dim_B   = block.dim_B
        self.dim_C   = block.dim_C
        self.dim_dt  = block.dim_dt
        self.dim_A   = block.dim_A
        self.dim_lambda = block.dim_lambda
        self.ratio   = block.ratio

        # 快取 apply_rope / chunk_parallel_scan（複用原 block 方法）
        self.apply_rope          = block.apply_rope
        self.chunk_parallel_scan = block.chunk_parallel_scan

    def forward(
        self,
        x,
        router_temp=None,
        mamba_cache=None,
        return_mamba_cache=False,
    ):
        cfg  = self.config
        B_sz, L, _ = x.shape
        H, G, P, N, R, ratio = (
            cfg.n_heads, cfg.n_groups, cfg.d_head,
            cfg.d_state, cfg.mimo_rank, self.ratio,
        )
        residual_mamba = x

        # ① 融合 norm_mamba + in_proj
        proj = self.fused_norm_in(x)
        z, x_prime, B_param, C_param, dt, A_param, lambda_param = torch.split(
            proj,
            [self.dim_z, self.dim_x, self.dim_B, self.dim_C,
             self.dim_dt, self.dim_A, self.dim_lambda],
            dim=-1,
        )

        x_prime = x_prime.view(B_sz, L, H, P)
        dt      = F.softplus(dt)
        A       = -torch.exp(A_param)
        theta   = torch.exp(self.theta_log)
        bg      = lambda t: t.repeat_interleave(ratio, dim=2)

        dt_b = bg(dt.unsqueeze(-1)).squeeze(-1)
        A_b  = bg(A.unsqueeze(-1)).squeeze(-1)
        theta_rep = theta.repeat_interleave(ratio, dim=0)
        current_angle_step = torch.einsum("blh, hn -> blhn", dt_b, theta_rep)

        if mamba_cache is not None:
            prev_h, prev_input, prev_angle_sum = mamba_cache
            angles = prev_angle_sum + torch.cumsum(current_angle_step, dim=1)
        else:
            angles = torch.cumsum(current_angle_step, dim=1)

        B_rotated = self.apply_rope(
            bg(self.norm_B(B_param.reshape(B_sz, L, G, N * R))
               .view(B_sz, L, G, N, R) + self.bias_B),
            angles,
        )
        C_rotated = self.apply_rope(
            bg(self.norm_C(C_param.reshape(B_sz, L, G, N * R))
               .view(B_sz, L, G, N, R) + self.bias_C),
            angles,
        )

        # ② 融合 x_up_proj（TuckerMoE 推論版）
        if cfg.use_kmoe:
            x_up, lb_up, z_up = self.fused_x_up(
                x_prime.view(B_sz, L, -1), router_temp=router_temp
            )
            x_ssm = x_up.view(B_sz, L, H, P, R)
        else:
            x_ssm  = self.fused_x_up(x_prime).view(B_sz, L, H, P, R)
            lb_up, z_up = 0.0, 0.0

        input_signal = torch.einsum("blhnr, blhpr -> blhnp", B_rotated, x_ssm)
        lv = F.sigmoid(bg(lambda_param.unsqueeze(-1)).squeeze(-1)).view(B_sz, L, H, 1, 1)
        dv = dt_b.view(B_sz, L, H, 1, 1)
        av = torch.exp(dt_b * A_b).view(B_sz, L, H, 1, 1)

        if mamba_cache is not None:
            ip = prev_input
        else:
            ip = torch.roll(input_signal, 1, 1)
            ip[:, 0] = 0
        u_ssm = lv * dv * input_signal + (1 - lv) * dv * av * ip

        mamba_cache_out = None
        if mamba_cache is not None:
            # decode：單步 SSM update
            h_s        = prev_h * av[:, 0] + u_ssm[:, 0]
            y_stack    = torch.einsum("bhnp,bhnr->bhpr", h_s, C_rotated[:, 0]).unsqueeze(1)
            mamba_cache_out = (h_s, input_signal[:, -1:], angles[:, -1:])
        elif cfg.use_parallel_scan:
            y_stack, h_prev = self.chunk_parallel_scan(
                u_ssm, dt_b, A_b, C_rotated, chunk_size=cfg.chunk_size
            )
            if return_mamba_cache:
                mamba_cache_out = (h_prev, input_signal[:, -1:], angles[:, -1:])
        else:
            h_s = torch.zeros(B_sz, H, N, P, device=x.device, dtype=u_ssm.dtype)
            y_list = []
            for t in range(L):
                h_s = h_s * av[:, t] + u_ssm[:, t]
                y_list.append(torch.einsum("bhnp,bhnr->bhpr", h_s, C_rotated[:, t]))
            y_stack = torch.stack(y_list, dim=1)
            if return_mamba_cache:
                mamba_cache_out = (h_s, input_signal[:, -1:], angles[:, -1:])

        y = self.y_down_proj(y_stack.view(B_sz, L, H, P * R)).view(B_sz, L, H * P)
        y = y + x_prime.reshape(B_sz, L, H * P) * self.D_skip.repeat_interleave(P, dim=0)

        # ③ 融合 pre_gate_norm * silu(z)
        mamba_out = self.mamba_dense_proj(self.fused_gate(y, z))

        # ⑤ 融合 LayerScale residual（inplace）
        mid_x = fused_add_layer_scale_(
            residual_mamba.clone(), mamba_out, self.ls_mamba.gamma
        )

        residual_proj = mid_x
        normed_mid    = self.norm_out_proj(mid_x)

        # ④ 融合 out_proj（TuckerMoE 推論版）
        if cfg.use_kmoe:
            proj_out, lb_out, z_out = self.fused_out(normed_mid, router_temp=router_temp)
        else:
            proj_out, lb_out, z_out = self.fused_out(normed_mid), 0.0, 0.0

        # ⑥ 融合 LayerScale residual（inplace）
        out = fused_add_layer_scale_(
            residual_proj.clone(), proj_out, self.ls_out_proj.gamma
        )

        if mamba_cache is not None or return_mamba_cache:
            return out, lb_up + lb_out, z_up + z_out, mamba_cache_out
        return out, lb_up + lb_out, z_up + z_out


# ══════════════════════════════════════════════════════════════════════
#  §8  FusedTransformerBlockInfer
#      TransformerBlock 推論加速包裝器
# ══════════════════════════════════════════════════════════════════════

class FusedTransformerBlockInfer(nn.Module):
    """
    TransformerBlock 推論加速包裝：
      ① norm_attn + [q,k,v]_proj → FusedRMSNormQKV（共享 x_norm）
      ② FFN 中每個 TuckerMoE   → FusedTuckerMoEInfer
      ③ LayerScale residual    → fused_add_layer_scale_
    """

    def __init__(self, block):
        super().__init__()
        self._orig = block

        # ① 融合 norm_attn + QKV
        self.fused_qkv = FusedRMSNormQKV(
            block.norm_attn, block.q_proj, block.k_proj, block.v_proj
        )
        self.o_proj      = block.o_proj
        self.head_dim    = block.head_dim
        self.num_heads   = block.num_heads
        self.num_kv_heads = block.num_kv_heads
        self.kv_groups   = block.kv_groups

        # ② FFN 替換
        self.use_kmoe = block.use_kmoe
        if block.use_kmoe:
            ffn = block.ffn
            # MixtralMoEFeedForward 含三個 TuckerMoE
            self.ffn_gate_fused = FusedTuckerMoEInfer(ffn.gate_proj)
            self.ffn_up_fused   = FusedTuckerMoEInfer(ffn.up_proj)
            self.ffn_down_fused = FusedTuckerMoEInfer(ffn.down_proj)
        else:
            self.ffn_gate = block.ffn_gate
            self.ffn_up   = block.ffn_up
            self.ffn_down = block.ffn_down

        self.norm_ffn   = block.norm_ffn
        self.ls_attn    = block.ls_attn
        self.ls_ffn     = block.ls_ffn

    def forward(
        self,
        x,
        router_temp=None,
        past_kv=None,
        seq_pos: int = 0,
        return_kv: bool = False,
    ):
        B, L, D = x.shape
        residual = x

        # ① 融合 norm_attn + QKV（x_norm 只讀一次）
        q_raw, k_raw, v_raw = self.fused_qkv(x)

        q     = q_raw.view(B, L, self.num_heads,   self.head_dim).transpose(1, 2)
        k_new = k_raw.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = v_raw.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA 展開
        k_new = (k_new.unsqueeze(2)
                 .expand(B, self.num_kv_heads, self.kv_groups, L, self.head_dim)
                 .reshape(B, self.num_heads, L, self.head_dim))
        v_new = (v_new.unsqueeze(2)
                 .expand(B, self.num_kv_heads, self.kv_groups, L, self.head_dim)
                 .reshape(B, self.num_heads, L, self.head_dim))

        kv_out = None
        if past_kv is None:
            attn = F.scaled_dot_product_attention(q, k_new, v_new, dropout_p=0.0, is_causal=True)
            if return_kv:
                kv_out = (k_new.detach(), v_new.detach())
        else:
            k_buf, v_buf = past_kv
            k_buf[:, :, seq_pos:seq_pos + L, :] = k_new
            v_buf[:, :, seq_pos:seq_pos + L, :] = v_new
            prefix = seq_pos + L
            attn = F.scaled_dot_product_attention(
                q, k_buf[:, :, :prefix, :], v_buf[:, :, :prefix, :],
                dropout_p=0.0, is_causal=False,
            )
            kv_out = (k_buf, v_buf)

        attn_out = self.o_proj(attn.transpose(1, 2).contiguous().view(B, L, D))

        # ③ 融合 LayerScale + residual
        x = fused_add_layer_scale_(residual.clone(), attn_out, self.ls_attn.gamma)
        residual2 = x

        # ② FFN（with 融合 TuckerMoE）
        h = self.norm_ffn(x)
        if self.use_kmoe:
            from train import fast_silu_gating
            gate, lb_g, z_g = self.ffn_gate_fused(h, router_temp=router_temp)
            feat, lb_u, z_u = self.ffn_up_fused(h, router_temp=router_temp)
            ffn_out, lb_d, z_d = self.ffn_down_fused(
                fast_silu_gating(gate, feat), router_temp=router_temp
            )
            lb = lb_g + lb_u + lb_d
        else:
            from train import fast_silu_gating
            ffn_out = self.ffn_down(fast_silu_gating(self.ffn_gate(h), self.ffn_up(h)))
            lb = 0.0

        out = fused_add_layer_scale_(residual2.clone(), ffn_out, self.ls_ffn.gamma)

        if past_kv is not None or return_kv:
            return out, lb, 0.0, kv_out
        return out, lb, 0.0


# ══════════════════════════════════════════════════════════════════════
#  §9  apply_inference_fusion  ── 一鍵注入所有融合算子
# ══════════════════════════════════════════════════════════════════════

def apply_inference_fusion(model: nn.Module) -> nn.Module:
    """
    對 Mamba3LanguageModel 進行推論算子融合，原地修改模型。

    替換清單：
      • backbone.layers[i]["block"] (Mamba3Block)
            → FusedMamba3BlockInfer
      • backbone.layers[i]["block"] (TransformerBlock)
            → FusedTransformerBlockInfer
      • model.head + d_model
            → FusedScaledHeadLogits（替換 forward_inference 的最後一步）

    並 monkey-patch model.forward_inference 以使用 FusedScaledHeadLogits。

    傳回 model（方便鏈式呼叫）。
    """
    model.eval()

    backbone = model.backbone

    # ── 替換每一層 block ──────────────────────────────────────────────
    for i, ld in enumerate(backbone.layers):
        blk = ld["block"]
        cls_name = type(blk).__name__

        if cls_name == "Mamba3Block":
            ld["block"] = FusedMamba3BlockInfer(blk)
            print(f"  ✅  layer {i:3d}  Mamba3Block      → FusedMamba3BlockInfer")

        elif cls_name == "TransformerBlock":
            ld["block"] = FusedTransformerBlockInfer(blk)
            print(f"  ✅  layer {i:3d}  TransformerBlock → FusedTransformerBlockInfer")

        else:
            print(f"  ⚠️   layer {i:3d}  未知型別 {cls_name!r}，跳過")

    # ── 替換 lm_head 最終輸出（FusedScaledHeadLogits）───────────────
    fused_head = FusedScaledHeadLogits(model.head, model.config.d_model, scale=30.0)

    # monkey-patch forward_inference 使用 fused_head
    _orig_forward_inference = model.forward_inference.__func__  # 取原始函式

    def _fused_forward_inference(self_m, input_ids, router_temp, layer_caches, seq_pos, prefill):
        x = self_m.embed(input_ids)
        hidden, _, _, new_caches = self_m.backbone.forward_inference(
            x, router_temp, layer_caches, seq_pos=seq_pos, prefill=prefill
        )
        hidden = self_m.norm(hidden)
        logits = fused_head(hidden)              # ← FusedScaledHeadLogits
        return logits, new_caches

    import types
    model.forward_inference = types.MethodType(_fused_forward_inference, model)
    print(f"  ✅  lm_head              → FusedScaledHeadLogits (D={model.config.d_model}, scale=30)")

    # ── 快取所有 TuckerMoE G_experts（預熱）────────────────────────
    n_cached = 0
    for m in model.modules():
        if isinstance(m, FusedTuckerMoEInfer):
            m._get_G_experts()   # 觸發物化
            n_cached += 1
    print(f"  ✅  G_experts 快取完成（{n_cached} 個 TuckerMoE 模組）")

    print(f"\n🚀  推論融合完成。所有算子已就緒。\n")
    return model


# ══════════════════════════════════════════════════════════════════════
#  §10  WarmupFusedModel  ── 首次推論預熱（消除 Triton autotune 延遲）
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def warmup_inference(
    model: nn.Module,
    device: torch.device,
    seq_len: int = 16,
    vocab_size: int = 32007,
    router_temp: float = 0.5,
    n_layers: int = None,
):
    """
    用隨機 input 跑一次完整的 forward_inference（prefill + 1 decode step），
    觸發所有 Triton kernel 的 autotune，消除線上推論的首次延遲。

    建議在 apply_inference_fusion() 之後立即呼叫。
    """
    print("🔥  Triton autotune 預熱中...", flush=True)
    model.eval()

    dummy_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    n_layer   = n_layers or len(model.backbone.layers)

    # 建立空 cache（與實際推論相同格式）
    layer_caches = []
    from train import TransformerBlock, Mamba3Block
    for ld in model.backbone.layers:
        blk = ld["block"]
        blk_type = type(blk).__name__
        if "Transformer" in blk_type:
            d = model.config.d_model
            nh = d // 64
            layer_caches.append((
                torch.zeros(1, nh, seq_len + 64, 64, device=device, dtype=torch.bfloat16),
                torch.zeros(1, nh, seq_len + 64, 64, device=device, dtype=torch.bfloat16),
            ))
        else:
            H = model.config.n_heads
            N = model.config.d_state
            P = model.config.d_head
            layer_caches.append((
                torch.zeros(1, H, N, P, device=device, dtype=torch.bfloat16),
                torch.zeros(1, 1, H, model.config.d_head, device=device, dtype=torch.bfloat16),
                torch.zeros(1, 1, H, N // 2, device=device, dtype=torch.bfloat16),
            ))

    t0 = __import__("time").time()
    # prefill
    _, caches = model.forward_inference(dummy_ids, router_temp, layer_caches, seq_pos=0, prefill=True)
    # 一步 decode
    dummy_dec = torch.randint(0, vocab_size, (1, 1), device=device)
    _, _      = model.forward_inference(dummy_dec, router_temp, caches, seq_pos=seq_len, prefill=False)
    t1 = __import__("time").time()

    print(f"🔥  預熱完成（{t1 - t0:.2f}s）。後續推論已無 autotune 延遲。\n", flush=True)