# -*- coding: utf-8 -*-
import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.checkpoint import checkpoint


# ── Triton Kernels & Model Classes ──

@triton.jit
def tanh_approx(x):
    # Avoid inline asm so torch.compile can analyze Triton kernels safely.
    # This keeps behavior close to tanh while removing tt.elementwise_inline_asm.
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ]

@triton.autotune(configs=get_cuda_autotune_config(), key=['n_elements'])
@triton.jit
def _fused_scaled_tanh_fwd(x_ptr, y_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    t = tanh_approx(x * (1.0 / scale))
    tl.store(y_ptr + offsets, t * scale, mask=mask)

@triton.autotune(configs=get_cuda_autotune_config(), key=['n_elements'])
@triton.jit
def _fused_scaled_tanh_bwd(dy_ptr, x_ptr, dx_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)
    x  = tl.load(x_ptr  + offsets, mask=mask).to(tl.float32)
    t  = tanh_approx(x * (1.0 / scale))
    tl.store(dx_ptr + offsets, dy * (1.0 - t * t), mask=mask)

class _FastScaledTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=10.0):
        ctx.save_for_backward(x); ctx.scale = scale
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        _fused_scaled_tanh_fwd[grid](x, y, scale, x.numel())
        return y
    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        _fused_scaled_tanh_bwd[grid](dy, x, dx, ctx.scale, x.numel())
        return dx, None

def fast_scaled_tanh(x, scale=10.0):
    return _FastScaledTanh.apply(x, scale)

@triton.autotune(configs=get_cuda_autotune_config(), key=['n_elements'])
@triton.jit
def _fused_silu_mul_fwd(gate_ptr, feat_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    feat = tl.load(feat_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(out_ptr + offsets, silu(gate) * feat, mask=mask)

@triton.autotune(configs=get_cuda_autotune_config(), key=['n_elements'])
@triton.jit
def _fused_silu_mul_bwd(dout_ptr, gate_ptr, feat_ptr, dgate_ptr, dfeat_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dout = tl.load(dout_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    feat = tl.load(feat_ptr + offsets, mask=mask).to(tl.float32)
    sig = tl.sigmoid(gate); s = gate * sig
    tl.store(dfeat_ptr + offsets, dout * s, mask=mask)
    tl.store(dgate_ptr + offsets, dout * feat * sig * (1.0 + gate * (1.0 - sig)), mask=mask)

class _FastSiluGating(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, feat):
        ctx.save_for_backward(gate, feat)
        out = torch.empty_like(gate)
        grid = lambda meta: (triton.cdiv(gate.numel(), meta["BLOCK_SIZE"]),)
        _fused_silu_mul_fwd[grid](gate, feat, out, gate.numel())
        return out
    @staticmethod
    def backward(ctx, dout):
        gate, feat = ctx.saved_tensors
        dgate, dfeat = torch.empty_like(gate), torch.empty_like(feat)
        grid = lambda meta: (triton.cdiv(gate.numel(), meta["BLOCK_SIZE"]),)
        _fused_silu_mul_bwd[grid](dout, gate, feat, dgate, dfeat, gate.numel())
        return dgate, dfeat

def fast_silu_gating(gate, feat):
    return _FastSiluGating.apply(gate, feat)

def get_router_temperature(step, warmup=500, total=10000, t_start=2.0, t_end=0.5):
    if step is None: return t_end
    
    # 轉為 PyTorch 張量運算，避開 Dynamo 將 step 視為 SymInt 時與 Python math 函式庫發生的型別衝突
    step_t = torch.as_tensor(step, dtype=torch.float32)
    progress = torch.clamp((step_t - warmup) / max(1, total - warmup), min=0.0, max=1.0)
    
    return t_end + 0.5 * (t_start - t_end) * (1.0 + torch.cos(torch.pi * progress))

class Mamba3Config:
    def __init__(
        self, d_model=768, d_state=64, d_head=64, n_groups=1, mimo_rank=4, expand=4,
        num_layers=15, use_parallel_scan=True, use_kmoe=True,
        kmoe_num_experts=8, kmoe_top_k=2, kmoe_r1=4, kmoe_r2=1024, kmoe_r3=256, ffn_expand=6, num_kv_heads=4,
        dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, layer_scale_init=1e-2, rms_norm_eps=1e-5, chunk_size=64,
        use_activation_checkpoint=True,
    ):
        self.d_model = d_model; self.d_state = d_state; self.d_head = d_head
        self.expand = expand; self.num_layers = num_layers
        self.d_inner = int(expand * d_model); self.n_heads = self.d_inner // d_head
        self.n_groups = n_groups; self.mimo_rank = mimo_rank
        self.rms_norm_eps = rms_norm_eps; self.chunk_size = chunk_size
        self.use_parallel_scan = use_parallel_scan; self.use_kmoe = use_kmoe
        self.kmoe_num_experts = kmoe_num_experts; self.kmoe_top_k = kmoe_top_k
        self.kmoe_r1 = kmoe_r1; self.kmoe_r2 = kmoe_r2; self.kmoe_r3 = kmoe_r3
        self.ffn_expand = ffn_expand; self.num_kv_heads = num_kv_heads
        self.kv_groups = self.n_heads // num_kv_heads
        self.dt_min, self.dt_max, self.dt_init_floor = dt_min, dt_max, dt_init_floor
        self.layer_scale_init = layer_scale_init
        self.use_activation_checkpoint = use_activation_checkpoint

class RMSNorm(nn.RMSNorm):
    def __init__(self, dim, eps=1e-5):
        super().__init__(normalized_shape=dim, eps=eps)    

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-2):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)
    def forward(self, x):
        return x * self.gamma

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R3': 32, 'BLOCK_R2': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R3': 32, 'BLOCK_R2': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_R3': 64, 'BLOCK_R2': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R3': 16, 'BLOCK_R2': 128}, num_warps=4, num_stages=4),
    ],
    key=['r3', 'r2'],
)
@triton.jit
def _fused_latent_moe_fwd(
    x_ptr, g_ptr, idx_ptr, prob_ptr, out_ptr,
    stride_xb, stride_xr3, stride_ge, stride_gr3, stride_gr2,
    stride_idxb, stride_idxk, stride_probb, stride_probk, stride_ob, stride_or2,
    B, r3, r2, top_k, BLOCK_R3: tl.constexpr, BLOCK_R2: tl.constexpr
):
    pid_b = tl.program_id(0); pid_r2 = tl.program_id(1)
    offs_r2 = pid_r2 * BLOCK_R2 + tl.arange(0, BLOCK_R2)
    acc = tl.zeros((BLOCK_R2,), dtype=tl.float32)
    for k in range(top_k):
        exp_idx = tl.load(idx_ptr  + pid_b * stride_idxb  + k * stride_idxk)
        prob    = tl.load(prob_ptr + pid_b * stride_probb + k * stride_probk)
        for r3_idx in range(0, r3, BLOCK_R3):
            offs_r3 = r3_idx + tl.arange(0, BLOCK_R3)
            x = tl.load(x_ptr + pid_b * stride_xb + offs_r3 * stride_xr3, mask=offs_r3 < r3, other=0.0)
            g = tl.load(g_ptr + exp_idx * stride_ge + offs_r3[:, None] * stride_gr3 + offs_r2[None, :] * stride_gr2,
                        mask=(offs_r3[:, None] < r3) & (offs_r2[None, :] < r2), other=0.0)
            acc += prob * tl.sum(x[:, None] * g, axis=0)
    tl.store(out_ptr + pid_b * stride_ob + offs_r2 * stride_or2, acc.to(out_ptr.dtype.element_ty), mask=offs_r2 < r2)

def get_dG_bwd_autotune_config():
    return [
        triton.Config({'BLOCK_R3': 32,  'BLOCK_R2': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R3': 32,  'BLOCK_R2': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_R3': 64,  'BLOCK_R2': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_R3': 64,  'BLOCK_R2': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R3': 128, 'BLOCK_R2': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R3': 64,  'BLOCK_R2': 256}, num_warps=8, num_stages=4),
    ]

@triton.autotune(configs=get_dG_bwd_autotune_config(), key=['r3', 'r2', 'B'])
@triton.jit
def _fused_latent_moe_bwd_dG_kernel(
    x_ptr, dout_ptr, prob_ptr, idx_ptr, dG_ptr,
    stride_xb, stride_xr3, stride_doutb, stride_doutr2,
    stride_probb, stride_probk, stride_idxb, stride_idxk,
    stride_dGe, stride_dGr3, stride_dGr2,
    B, top_k, r3, r2, BLOCK_R3: tl.constexpr, BLOCK_R2: tl.constexpr
):
    pid_e = tl.program_id(0); pid_r3 = tl.program_id(1); pid_r2 = tl.program_id(2)
    offs_r3 = pid_r3 * BLOCK_R3 + tl.arange(0, BLOCK_R3)
    offs_r2 = pid_r2 * BLOCK_R2 + tl.arange(0, BLOCK_R2)
    mask_r3 = offs_r3 < r3; mask_r2 = offs_r2 < r2
    acc = tl.zeros((BLOCK_R3, BLOCK_R2), dtype=tl.float32)
    for b in range(B):
        for k in range(top_k):
            e = tl.load(idx_ptr + b * stride_idxb + k * stride_idxk)
            if e == pid_e:
                prob = tl.load(prob_ptr + b * stride_probb + k * stride_probk).to(tl.float32)
                x    = tl.load(x_ptr   + b * stride_xb    + offs_r3 * stride_xr3,   mask=mask_r3, other=0.0).to(tl.float32)
                dout = tl.load(dout_ptr + b * stride_doutb + offs_r2 * stride_doutr2, mask=mask_r2, other=0.0).to(tl.float32)
                acc += x[:, None] * (dout * prob)[None, :]
    tl.store(dG_ptr + pid_e * stride_dGe + offs_r3[:, None] * stride_dGr3 + offs_r2[None, :] * stride_dGr2,
             acc.to(dG_ptr.dtype.element_ty), mask=mask_r3[:, None] & mask_r2[None, :])
    



class FusedLatentMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_shared, G_experts, top_k_indices, top_k_probs):
        B, r3 = x_shared.shape; E, _, r2 = G_experts.shape; top_k = top_k_indices.size(1)
        ctx.save_for_backward(x_shared, G_experts, top_k_indices, top_k_probs)
        out = torch.empty((B, r2), device=x_shared.device, dtype=x_shared.dtype)
        _fused_latent_moe_fwd[lambda meta: (B, triton.cdiv(r2, meta['BLOCK_R2']))](
            x_shared, G_experts, top_k_indices, top_k_probs, out,
            r3, 1,
            r3 * r2, r2, 1,
            top_k, 1,
            top_k, 1,
            r2, 1, B, r3, r2, top_k)
        return out

    @staticmethod
    def backward(ctx, dout):
        x_shared, G_experts, top_k_indices, top_k_probs = ctx.saved_tensors
        B, r3 = x_shared.shape; E, _, r2 = G_experts.shape; top_k = top_k_indices.size(1)
        dx_shared = torch.zeros_like(x_shared)
        dprobs    = torch.zeros_like(top_k_probs)

        target_dtype = x_shared.dtype
        if dout.dtype != target_dtype:
            dout = dout.to(target_dtype)

        need_cast = G_experts.dtype != target_dtype

        # 為了避免 aten::nonzero 帶來的 CPU-GPU 同步瓶頸，移除所有 boolean mask 與 .any() 的動態判斷
        # 取而代之，將 mask 轉為 float 進行逐元素乘法，這讓所有的操作維持在純 CUDA Kernel 內獨立運算。
        for k in range(top_k):
            idx  = top_k_indices[:, k]
            prob = top_k_probs[:, k].unsqueeze(1)
            if prob.dtype != target_dtype:
                prob = prob.to(target_dtype)
            
            dout_k = dout * prob # (B_flat, r2)
            
            for e in range(E):
                # 建立 0.0 或 1.0 的 mask，避免觸發 index_select
                mask_e = (idx == e).unsqueeze(1).to(target_dtype) # (B_flat, 1)
                
                G_e = G_experts[e].to(target_dtype) if need_cast else G_experts[e]

                # dout_e 在非此 Expert 的位置全部歸零
                dout_e = dout_k * mask_e # (B_flat, r2)
                
                # 所有 Row 都進行矩陣相乘，但 0 的 Row 加進 dx_shared 等於沒加，完美規避記憶體突波與 CPU 同步
                dx_shared += torch.matmul(dout_e, G_e.transpose(0, 1))
                
                # 計算機率梯度
                dprobs[:, k] += (dout * torch.matmul(x_shared, G_e)).sum(dim=-1) * mask_e.squeeze(-1)

        dG_experts = torch.zeros_like(G_experts)
        # Strides for contiguous tensors
        # x_shared: (B, r3) -> r3, 1
        # dout: (B, r2) -> r2, 1
        # top_k_probs/indices: (B, top_k) -> top_k, 1
        # dG_experts: (E, r3, r2) -> r3 * r2, r2, 1
        _fused_latent_moe_bwd_dG_kernel[lambda meta: (E, triton.cdiv(r3, meta['BLOCK_R3']), triton.cdiv(r2, meta['BLOCK_R2']))](
            x_shared, dout, top_k_probs, top_k_indices, dG_experts,
            r3, 1, r2, 1,
            top_k, 1,
            top_k, 1,
            r3 * r2, r2, 1, B, top_k, r3, r2)
        return dx_shared, dG_experts, None, dprobs

class TritonTuckerMoE(nn.Module):
    def __init__(self, dim_in, dim_out, num_experts=8, top_k=2, r1=4, r2=1024, r3=256):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        self.router = nn.Linear(dim_in, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        
        self.U_expert = nn.Parameter(torch.empty(num_experts, r1))
        self.U_in     = nn.Parameter(torch.empty(dim_in, r3))
        self.U_out    = nn.Parameter(torch.empty(r2, dim_out))
        self.core     = nn.Parameter(torch.empty(r1, r3, r2))
        self.bias     = nn.Parameter(torch.zeros(dim_out))

        self.inner_norm = RMSNorm(r3) 
        
        nn.init.orthogonal_(self.U_in)
        nn.init.orthogonal_(self.U_out)
        nn.init.xavier_uniform_(self.U_expert)
        nn.init.xavier_uniform_(self.core)

    def forward(self, x, step=None):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        B_flat = x_flat.size(0)
        
        temperature = get_router_temperature(step)
        raw_logits  = self.router(x_flat)
        capped      = fast_scaled_tanh(raw_logits, 10.0) 
        
        z_loss = (torch.mean(torch.logsumexp(capped, dim=-1) ** 2) if self.training else 0.0)
        
        router_logits = capped / temperature
        router_probs  = torch.softmax(router_logits, dim=-1)
        
        _, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_raw   = router_probs.gather(-1, top_k_indices)
        top_k_probs = top_k_raw / (top_k_raw.sum(-1, keepdim=True) + 1e-6)
        
        if self.training:
            expert_mask = torch.zeros_like(router_logits).scatter_(1, top_k_indices, 1.0)
            lb_loss = self.num_experts * torch.sum(expert_mask.mean(0) * router_probs.float().mean(0))
        else:
            lb_loss = 0.0
            
        x_shared = torch.matmul(x_flat, self.U_in)
        x_shared = self.inner_norm(x_shared)
        
        G_experts = torch.einsum('er, rst -> est', self.U_expert, self.core)
        x_core = FusedLatentMoE.apply(x_shared, G_experts, top_k_indices, top_k_probs).to(x.dtype)
        
        out = torch.matmul(x_core, self.U_out).reshape(*orig_shape[:-1], -1)
        
        return out + self.bias, lb_loss, z_loss

TuckerMoE = TritonTuckerMoE

class MixtralMoEFeedForward(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        d_ff = int(math.ceil(config.ffn_expand * config.d_model / 256) * 256)
        kw = dict(num_experts=config.kmoe_num_experts, top_k=config.kmoe_top_k,
                  r1=config.kmoe_r1, r2=config.kmoe_r2, r3=config.kmoe_r3)
        self.gate_proj = TuckerMoE(config.d_model, d_ff, **kw)
        self.up_proj   = TuckerMoE(config.d_model, d_ff, **kw)
        self.down_proj = TuckerMoE(d_ff, config.d_model, **kw)

    def forward(self, x, step=None):
        gate, lb_g, z_g = self.gate_proj(x, step=step)
        feat, lb_u, z_u = self.up_proj(x, step=step)
        y,    lb_d, z_d = self.down_proj(fast_silu_gating(gate, feat), step=step)
        return y, lb_g + lb_u + lb_d, z_g + z_u + z_d


# ── 新版 Triton Parallel Scan (包含完整的 Forward 和 Backward) ──

@triton.jit
def first_order_combine_op(alpha_left, beta_left, alpha_right, beta_right):
    return alpha_right * alpha_left, alpha_right * beta_left + beta_right

def get_fwd_autotune_configs():
    return [
        triton.Config({'BLOCK_D': 32}, num_warps=4),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
        triton.Config({'BLOCK_D': 512}, num_warps=16),
    ]

@triton.autotune(configs=get_fwd_autotune_configs(), key=['D', 'L'])
@triton.jit
def _chunk_scan_fwd_kernel(
    log_alpha_ptr, u_ptr, h_out_ptr,
    sa_b, sa_c, sa_l, sa_h,
    su_b, su_c, su_l, su_h,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_bch, pid_d = tl.program_id(0), tl.program_id(1)
    
    h_idx = pid_bch % H
    rem = pid_bch // H
    c_idx = rem % C
    b_idx = rem // C
    
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offset_l = tl.arange(0, L)
    mask_d = offset_d < D

    alpha_base = log_alpha_ptr + b_idx * sa_b + c_idx * sa_c + h_idx * sa_h
    alpha_ptrs = alpha_base + offset_l * sa_l
    alpha = tl.exp(tl.load(alpha_ptrs).to(tl.float32))
    
    u_base = u_ptr + b_idx * su_b + c_idx * su_c + h_idx * su_h
    u_ptrs = u_base + offset_l[:, None] * su_l + offset_d[None, :] 
    u = tl.load(u_ptrs, mask=mask_d[None, :], other=0.0).to(tl.float32)

    _, h = tl.associative_scan((tl.broadcast_to(alpha[:, None], (L, BLOCK_D)), u), axis=0, combine_fn=first_order_combine_op)

    h_out_base = h_out_ptr + b_idx * su_b + c_idx * su_c + h_idx * su_h
    h_out_ptrs = h_out_base + offset_l[:, None] * su_l + offset_d[None, :]
    tl.store(h_out_ptrs, h.to(u_ptr.dtype.element_ty), mask=mask_d[None, :])

def get_bwd_autotune_configs():
    return [
        triton.Config({'BLOCK_D': 32}, num_warps=4),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
    ]

@triton.autotune(configs=get_bwd_autotune_configs(), key=['D', 'L'])
@triton.jit
def _chunk_scan_bwd_kernel(
    log_alpha_ptr, h_ptr, dh_ptr, du_ptr, dlog_alpha_ptr,
    sa_b, sa_c, sa_l, sa_h,
    sh_b, sh_c, sh_l, sh_h,
    su_b, su_c, su_l, su_h,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_bch, pid_d = tl.program_id(0), tl.program_id(1)
    
    h_idx = pid_bch % H
    rem = pid_bch // H
    c_idx = rem % C
    b_idx = rem // C
    
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offset_d < D
    offset_l = tl.arange(0, L)
    rev_offset_l = L - 1 - offset_l  

    alpha_base = log_alpha_ptr + b_idx * sa_b + c_idx * sa_c + h_idx * sa_h
    dalpha_base = dlog_alpha_ptr + b_idx * sa_b + c_idx * sa_c + h_idx * sa_h
    
    u_base = h_ptr + b_idx * sh_b + c_idx * sh_c + h_idx * sh_h
    dh_base = dh_ptr + b_idx * su_b + c_idx * su_c + h_idx * su_h
    du_base = du_ptr + b_idx * su_b + c_idx * su_c + h_idx * su_h

    dh_ptrs = dh_base + rev_offset_l[:, None] * su_l + offset_d[None, :]
    dh = tl.load(dh_ptrs, mask=mask_d[None, :], other=0.0).to(tl.float32)

    alpha_next_idx = L - offset_l
    alpha_next_mask = alpha_next_idx < L
    log_alpha_next = tl.load(alpha_base + alpha_next_idx * sa_l, mask=alpha_next_mask, other=-float('inf')).to(tl.float32)
    alpha_rev = tl.where(alpha_next_mask, tl.exp(log_alpha_next), 0.0)

    _, delta_rev = tl.associative_scan((tl.broadcast_to(alpha_rev[:, None], (L, BLOCK_D)), dh), axis=0, combine_fn=first_order_combine_op)

    du_ptrs = du_base + rev_offset_l[:, None] * su_l + offset_d[None, :]
    tl.store(du_ptrs, delta_rev.to(du_ptr.dtype.element_ty), mask=mask_d[None, :])

    h_prev_idx = L - 2 - offset_l
    h_prev = tl.load(u_base + h_prev_idx[:, None] * sh_l + offset_d[None, :], mask=(h_prev_idx >= 0)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    alpha_curr_idx = L - 1 - offset_l
    alpha_curr = tl.exp(tl.load(alpha_base + alpha_curr_idx * sa_l).to(tl.float32))

    dlog_alpha_sum = tl.sum(delta_rev * alpha_curr[:, None] * h_prev, axis=1)
    tl.atomic_add(dalpha_base + alpha_curr_idx * sa_l, dlog_alpha_sum)

class TritonParallelScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_alpha_chunk, u_chunk):
        B, num_chunks, L, H = log_alpha_chunk.shape
        N_dim, P_dim = u_chunk.shape[-2], u_chunk.shape[-1]
        D = N_dim * P_dim
        
        log_alpha_chunk = log_alpha_chunk.contiguous()
        u_chunk = u_chunk.contiguous()
        
        BCH = B * num_chunks * H
        h_out = torch.empty_like(u_chunk)
        
        # Calculate contiguous strides
        sa_l = H; sa_c = L * sa_l; sa_b = num_chunks * sa_c
        su_h = N_dim * P_dim; su_l = H * su_h; su_c = L * su_l; su_b = num_chunks * su_c
        sa_h = 1
        
        _chunk_scan_fwd_kernel[lambda meta: (BCH, triton.cdiv(D, meta['BLOCK_D']))](
            log_alpha_chunk, u_chunk, h_out,
            sa_b, sa_c, sa_l, sa_h,
            su_b, su_c, su_l, su_h,
            B=B, C=num_chunks, H=H, L=L, D=D
        )
        
        ctx.save_for_backward(log_alpha_chunk, h_out)
        return h_out

    @staticmethod
    def backward(ctx, dh_out):
        log_alpha_chunk, h_out = ctx.saved_tensors
        B, num_chunks, L, H = log_alpha_chunk.shape
        N_dim, P_dim = h_out.shape[-2], h_out.shape[-1]
        D = N_dim * P_dim
        BCH = B * num_chunks * H
        
        dh_out = dh_out.contiguous()
        du_chunk = torch.empty_like(dh_out)
        dlog_alpha_chunk = torch.zeros_like(log_alpha_chunk, dtype=torch.float32)
        
        # Calculate contiguous strides analytically
        sa_l = H; sa_c = L * sa_l; sa_b = num_chunks * sa_c
        sh_h = N_dim * P_dim; sh_l = H * sh_h; sh_c = L * sh_l; sh_b = num_chunks * sh_c
        sa_h = 1
        
        _chunk_scan_bwd_kernel[lambda meta: (BCH, triton.cdiv(D, meta['BLOCK_D']))](
            log_alpha_chunk, h_out, dh_out, du_chunk, dlog_alpha_chunk,
            sa_b, sa_c, sa_l, sa_h,
            sh_b, sh_c, sh_l, sh_h,
            sh_b, sh_c, sh_l, sh_h, # su corresponds directly to dh and h layouts
            B=B, C=num_chunks, H=H, L=L, D=D
        )
        
        dlog_alpha_out = dlog_alpha_chunk.to(log_alpha_chunk.dtype)
        return dlog_alpha_out, du_chunk

def fast_triton_chunk_scan(log_alpha_chunk, u_chunk):
    return TritonParallelScanFn.apply(log_alpha_chunk, u_chunk)


# ── Main Architecture Blocks ──────────────────────────────────────────

class Mamba3Block(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        d_in, H, G, P, N, R = config.d_model, config.n_heads, config.n_groups, config.d_head, config.d_state, config.mimo_rank
        self.ratio, self.dim_z, self.dim_x = H // G, H * P, H * P
        self.dim_B, self.dim_C, self.dim_dt, self.dim_A, self.dim_lambda = G*N*R, G*N*R, G, G, G
        self.in_proj = nn.Linear(d_in, self.dim_z+self.dim_x+self.dim_B+self.dim_C+self.dim_dt+self.dim_A+self.dim_lambda, bias=True)
        if config.use_kmoe:
            kw = dict(num_experts=config.kmoe_num_experts, top_k=config.kmoe_top_k, r1=config.kmoe_r1, r2=config.kmoe_r2, r3=config.kmoe_r3)
            self.x_up_proj = TuckerMoE(H*P, H*P*R, **kw)
            self.out_proj  = TuckerMoE(d_in, d_in, **kw)
        else:
            self.x_up_proj = nn.Linear(P, P*R, bias=False)
            self.out_proj  = nn.Linear(d_in, d_in, bias=False)
            
        self.y_down_proj      = nn.Linear(P*R, P, bias=False)
        self.theta_log        = nn.Parameter(torch.randn(G, N//2))
        self.D                = nn.Parameter(torch.ones(H))
        self.norm_B           = RMSNorm(N*R, eps=config.rms_norm_eps)
        self.norm_C           = RMSNorm(N*R, eps=config.rms_norm_eps)
        self.bias_B           = nn.Parameter(torch.zeros(G, N, R))
        self.bias_C           = nn.Parameter(torch.zeros(G, N, R))
        self.mamba_dense_proj = nn.Linear(config.d_inner, d_in, bias=False)
        self.pre_gate_norm    = RMSNorm(H*P)
        self.act              = nn.SiLU()
        self.norm_mamba       = RMSNorm(config.d_model)
        self.norm_out_proj    = RMSNorm(config.d_model)
        self.ls_mamba         = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_out_proj      = LayerScale(config.d_model, init_value=config.layer_scale_init)

        
        with torch.no_grad():
            self.bias_B.fill_(1.0); self.bias_C.fill_(1.0)
            dt = torch.clamp(torch.exp(torch.rand(G) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)), min=config.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            dt_start = self.dim_z + self.dim_x + self.dim_B + self.dim_C
            dt_end = dt_start + self.dim_dt; A_end = dt_end + self.dim_A
            self.in_proj.bias[dt_start:dt_end].copy_(inv_dt)
            self.in_proj.bias[dt_end:A_end].uniform_(1, 16).log_()
            self.in_proj.bias[A_end:].fill_(-3.0)

    def apply_rope(self, x, angles):
        N_half = angles.shape[-1]
        x_reshaped = x.view(*x.shape[:-2], N_half, 2, x.shape[-1])
        x1, x2 = x_reshaped[..., 0, :], x_reshaped[..., 1, :]
        sin_a, cos_a = torch.sin(angles).unsqueeze(-1), torch.cos(angles).unsqueeze(-1)
        return torch.stack([x1*cos_a - x2*sin_a, x2*cos_a + x1*sin_a], dim=-2).reshape_as(x)

    def segsum(self, x):
        x_cumsum = torch.cumsum(x, dim=-1)
        mask = torch.tril(torch.ones(x.size(-1), x.size(-1), device=x.device, dtype=torch.bool))
        return (x_cumsum[..., :, None] - x_cumsum[..., None, :]).masked_fill(~mask, -float("inf"))

    def chunk_parallel_scan(self, u, dt, A, C, chunk_size=128):
        B, L, H, N, P = u.shape; R = C.shape[-1]; input_dtype = u.dtype; L_orig = L
        if L % chunk_size != 0:
            pad = chunk_size - (L % chunk_size)
            u = F.pad(u, (0,0,0,0,0,0,0,pad)); dt = F.pad(dt,(0,0,0,pad))
            C = F.pad(C, (0,0,0,0,0,0,0,pad)); A = F.pad(A, (0,0,0,pad)); L += pad
        nc = L // chunk_size
        log_alpha = dt * A
        u_c  = u.view(B, nc, chunk_size, H, N, P)
        la_c = log_alpha.view(B, nc, chunk_size, H)
        C_c  = C.view(B, nc, chunk_size, H, N, R)
        del log_alpha  # 節省記憶體
        
        # ── 這裡已經接上了我們最新的 TritonParallelScanFn ──
        h_intra = fast_triton_chunk_scan(la_c, u_c)
        del u_c  # scan 完成後立即釋放
        
        y_diag  = torch.einsum("bclhnp, bclhnr -> bclhpr", h_intra, C_c)
        decay   = torch.exp(torch.sum(la_c, dim=2))
        h_prev  = torch.zeros(B, H, N, P, device=u.device, dtype=input_dtype)
        h_inter = torch.empty(B, nc, H, N, P, device=u.device, dtype=input_dtype)
        for c in range(nc):
            h_inter[:, c] = h_prev
            h_prev = h_prev * decay[:, c].view(B, H, 1, 1) + h_intra[:, c, -1]
        del h_intra, decay  # 釋放已消耗的中間結果
        c_dec = C_c * torch.exp(torch.cumsum(la_c, dim=2)).unsqueeze(-1).unsqueeze(-1)
        del la_c, C_c  # 釋放
        y_off = torch.einsum("bchnp, bclhnr -> bclhpr", h_inter, c_dec)
        del h_inter, c_dec  # 釋放
        y = (y_diag + y_off).view(B, -1, H, P, R)
        del y_diag, y_off
        return (y[:, :L_orig] if L_orig < L else y).to(input_dtype), h_prev.to(input_dtype)

    def forward(self, x, step=None):
        B_sz, L, _ = x.shape
        H, G, P, N, R, ratio = self.config.n_heads, self.config.n_groups, self.config.d_head, self.config.d_state, self.config.mimo_rank, self.ratio
        residual_mamba, u = x, self.norm_mamba(x)
        z, x_prime, B_param, C_param, dt, A_param, lambda_param = torch.split(
            self.in_proj(u), [self.dim_z, self.dim_x, self.dim_B, self.dim_C, self.dim_dt, self.dim_A, self.dim_lambda], dim=-1)
        x_prime = x_prime.view(B_sz, L, H, P)
        dt = F.softplus(dt); A = -torch.exp(A_param); theta = torch.exp(self.theta_log)
        bg = lambda t: t.repeat_interleave(ratio, dim=2)
        dt_b = bg(dt.unsqueeze(-1)).squeeze(-1); A_b = bg(A.unsqueeze(-1)).squeeze(-1)
        angles = torch.cumsum(torch.einsum("blh, hn -> blhn", dt_b, theta.repeat_interleave(ratio, dim=0)), dim=1)
        B_rotated = self.apply_rope(bg(self.norm_B(B_param.reshape(B_sz,L,G,N*R)).view(B_sz,L,G,N,R) + self.bias_B), angles)
        C_rotated = self.apply_rope(bg(self.norm_C(C_param.reshape(B_sz,L,G,N*R)).view(B_sz,L,G,N,R) + self.bias_C), angles)
        if self.config.use_kmoe:
            x_up, lb_up, z_up = self.x_up_proj(x_prime.view(B_sz, L, -1), step=step)
            x_ssm = x_up.view(B_sz, L, H, P, R)
        else:
            x_ssm, lb_up, z_up = self.x_up_proj(x_prime).view(B_sz,L,H,P,R), 0.0, 0.0
        input_signal = torch.einsum("blhnr, blhpr -> blhnp", B_rotated, x_ssm)
        del B_rotated, x_ssm
        
        lv = F.sigmoid(bg(lambda_param.unsqueeze(-1)).squeeze(-1)).view(B_sz,L,H,1,1)
        dv = dt_b.view(B_sz,L,H,1,1); av = torch.exp(dt_b*A_b).view(B_sz,L,H,1,1)
        
        # 使用函數式寫法 (Functional approach) 讓 torch.compile 有效進行底層 Triton 算子融合
        # 這會自動把記憶體讀寫大幅減少，避免我們先前看到的 1.5GB HBM 等級暴衝
        ip = F.pad(input_signal[:, :-1], (0,0,0,0,0,0,1,0))
        
        # 為了避免在後續發生隱含的 precision 丟失，讓計算都在高精度下進行然後轉回 bf16
        u_ssm = (lv * dv * input_signal) + ((1.0 - lv) * dv * av * ip)
        u_ssm = u_ssm.to(input_signal.dtype)
        
        del input_signal, ip, lv, dv, av

        if self.config.use_parallel_scan:
            y_stack, _ = self.chunk_parallel_scan(u_ssm, dt_b, A_b, C_rotated, chunk_size=self.config.chunk_size)
            del u_ssm, C_rotated  # scan 後釋放
        else:
            h_s = torch.zeros(B_sz,H,N,P,device=x.device); y_list=[]
            for t in range(L):
                h_s = h_s * av[:,t] + u_ssm[:,t]
                y_list.append(torch.einsum("bhnp,bhnr->bhpr", h_s, C_rotated[:,t]))
            y_stack = torch.stack(y_list, dim=1)
            del u_ssm, C_rotated
            
        y = self.y_down_proj(y_stack.view(B_sz,L,H,P*R)).view(B_sz,L,H*P)
        del y_stack  # 釋放
        y = y + x_prime.reshape(B_sz,L,H*P) * self.D.repeat_interleave(P,dim=0)
        mamba_out = self.mamba_dense_proj(self.pre_gate_norm(y) * self.act(z))
        del y, z  # 釋放
        mid_x = residual_mamba + self.ls_mamba(mamba_out)
        residual_proj, normed_mid = mid_x, self.norm_out_proj(mid_x)
        
        if self.config.use_kmoe:
            proj_out, lb_out, z_out = self.out_proj(normed_mid, step=step)
        else:
            proj_out, lb_out, z_out = self.out_proj(normed_mid), 0.0, 0.0
        return residual_proj + self.ls_out_proj(proj_out), lb_up + lb_out, z_up + z_out

class TransformerBlock(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.head_dim=64; self.num_heads=config.d_model//64
        self.num_kv_heads=config.num_kv_heads; self.kv_groups=self.num_heads//config.num_kv_heads
        self.q_proj  = nn.Linear(config.d_model, self.num_heads*64, bias=False)
        self.k_proj  = nn.Linear(config.d_model, self.num_kv_heads*64, bias=False)
        self.v_proj  = nn.Linear(config.d_model, self.num_kv_heads*64, bias=False)
        self.o_proj  = nn.Linear(config.d_model, config.d_model, bias=True)
        self.norm_attn = RMSNorm(config.d_model); self.use_kmoe = config.use_kmoe
        if config.use_kmoe:
            self.ffn = MixtralMoEFeedForward(config)
        else:
            d_ff = int(math.ceil(8*config.d_model/3/256)*256)
            self.ffn_gate = nn.Linear(config.d_model, d_ff, bias=False)
            self.ffn_up   = nn.Linear(config.d_model, d_ff, bias=False)
            self.ffn_down = nn.Linear(d_ff, config.d_model, bias=False)
        self.norm_ffn = RMSNorm(config.d_model)
        self.ls_attn = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_ffn  = LayerScale(config.d_model, init_value=config.layer_scale_init)

    def forward(self, x, step=None):
        B, L, D = x.shape; residual, nx = x, self.norm_attn(x)
        q = self.q_proj(nx).view(B,L,self.num_heads,64).transpose(1,2)
        k = self.k_proj(nx).view(B,L,self.num_kv_heads,64).transpose(1,2)
        v = self.v_proj(nx).view(B,L,self.num_kv_heads,64).transpose(1,2)
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)
        attn = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=True)
        x = residual + self.ls_attn(self.o_proj(attn.transpose(1,2).reshape(B,L,D)))
        residual, h = x, self.norm_ffn(x)
        if self.use_kmoe:
            ffn_out, lb, z = self.ffn(h, step=step)
        else:
            ffn_out = self.ffn_down(fast_silu_gating(self.ffn_gate(h), self.ffn_up(h))); lb=0.0; z=0.0
        return residual + self.ls_ffn(ffn_out), lb, z

class TrueHybridMamba(nn.Module):
    def __init__(self, config: Mamba3Config, mamba_ratio=4):
        super().__init__()
        self.use_activation_checkpoint = bool(getattr(config, "use_activation_checkpoint", True))
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            for _ in range(mamba_ratio):
                self.layers.append(nn.ModuleDict({"block": Mamba3Block(config)}))
            self.layers.append(nn.ModuleDict({"block": TransformerBlock(config)}))

    def forward(self, x, step=None):
        total_lb, total_z = 0.0, 0.0
        for ld in self.layers:
            if self.use_activation_checkpoint:
                x, lb, z = checkpoint(ld["block"], x, step, use_reentrant=False)
            else:
                x, lb, z = ld["block"](x, step)
            if isinstance(lb, torch.Tensor): total_lb = total_lb + lb; total_z = total_z + z
        return x, total_lb, total_z

class Mamba3LanguageModel(nn.Module):
    def __init__(self, config: Mamba3Config, vocab_size: int, **kwargs):
        super().__init__()
        self.config = config
        self.embed    = nn.Embedding(vocab_size, config.d_model)
        self.backbone = TrueHybridMamba(config)
        self.norm     = RMSNorm(config.d_model)
        self.head     = nn.Linear(config.d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self._last_loss_terms = None
        self.register_buffer(
            "_structure_ce_weight_ids",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )
        self._structure_ce_weight_mult: float = 1.0
        # ---- Final Enhance: Final-SW + PDL + InfoEntropy (default-off) -------
        # Set on the module before calling forward() to enable. No behaviour
        # change for pretrain / when all three flags stay False.
        self._enable_final_sw: bool = False   # final-region prefix sliding-window weighting
        self._final_sw_eta: float = 0.1       # decay rate over final-region offset
        self._final_sw_lambda: float = 0.8    # prefix boost amplitude
        self._enable_pdl: bool = False        # static per-token frequency weighting
        self._pdl_weights = None              # torch.Tensor [vocab_size] or None
        self._enable_ie: bool = False         # dynamic entropy weighting (final region)
        self._ie_gamma: float = 0.5           # entropy weight amplitude
        self._ie_on_think: bool = False       # also apply IE inside <think> region
        self._final_start_id: int = 32004     # <final>
        self._final_end_id: int = 32005       # </final>
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def enable_structure_token_ce_weighting(
        self, token_ids: Sequence[int], multiplier: float
    ) -> None:
        """Up-weight CE at label positions matching token_ids (SFT only). multiplier<=1 disables."""
        m = float(multiplier)
        ids = sorted({int(x) for x in token_ids})
        if m <= 1.0 or not ids:
            self.disable_structure_token_ce_weighting()
            return
        self._structure_ce_weight_mult = m
        dev = next(self.parameters()).device
        self._structure_ce_weight_ids = torch.tensor(ids, dtype=torch.long, device=dev)

    def disable_structure_token_ce_weighting(self) -> None:
        self._structure_ce_weight_mult = 1.0
        dev = next(self.parameters()).device
        self._structure_ce_weight_ids = torch.empty(0, dtype=torch.long, device=dev)

    def forward(self, input_ids, labels=None, step=None,
                structure_weights=None, scale_weights=None):
        """Forward pass.

        Optional Task 2 hooks (all default-off, no behaviour change for pretrain):
          * `structure_weights` (B, T) float — per-token SFT-GO multiplier on CE.
          * `scale_weights`     (B, T) float — per-token SCALe multiplier on CE.
          * Set on the module before calling to enable FCP (Format/EOS Penalty):
              `_fcp_eos_id`, `_fcp_think_start_id`, `_fcp_think_end_id`,
              `_fcp_lambda`  (set >0 to activate), `_fcp_delta`.
            When active, the FCP penalty is added into `loss` and returned as the
            6th tuple element (detached) along with mean / max p(EOS) in-region as 7th / 8th.
        """
        backbone_out = self.backbone(self.embed(input_ids), step=step)
        hidden = self.norm(backbone_out[0])
        total_lb_loss, total_z_loss = backbone_out[1], backbone_out[2]
        logits = fast_scaled_tanh(self.head(hidden / math.sqrt(self.config.d_model)).float(), 30.0)
        if labels is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            vocab_size = logits.size(-1)

            # ---- Final Enhance setup (flags + shared softmax) ----------------
            _enable_final_sw = bool(getattr(self, "_enable_final_sw", False))
            _pdl_w = getattr(self, "_pdl_weights", None)
            _enable_pdl = bool(getattr(self, "_enable_pdl", False)) and isinstance(_pdl_w, torch.Tensor)
            _enable_ie = bool(getattr(self, "_enable_ie", False))
            _any_enhance = _enable_final_sw or _enable_pdl or _enable_ie
            # logged means (1.0 = disabled / region empty), overwritten below
            pdl_weight_mean = logits.new_ones(())
            ie_weight_mean = logits.new_ones(())

            fcp_eos_id = getattr(self, "_fcp_eos_id", None)
            fcp_lambda = float(getattr(self, "_fcp_lambda", 0.0))
            _fcp_active = fcp_eos_id is not None and fcp_lambda > 0.0

            # Shared softmax: FCP needs one channel, IE needs the full dist.
            # Materialise [B,T,V] at most once; skip when neither is active.
            _probs = None
            if _fcp_active or _enable_ie:
                _probs = F.softmax(logits.float(), dim=-1)

            w_mult = float(getattr(self, "_structure_ce_weight_mult", 1.0))
            w_ids = getattr(self, "_structure_ce_weight_ids", None)
            use_struct_ids = (
                w_ids is not None
                and isinstance(w_ids, torch.Tensor)
                and w_ids.numel() > 0
                and w_mult > 1.0
            )
            use_struct_w  = isinstance(structure_weights, torch.Tensor)
            use_scale_w   = isinstance(scale_weights, torch.Tensor)
            use_w = use_struct_ids or use_struct_w or use_scale_w or _any_enhance
            if use_w:
                loss_none = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                raw = loss_none(logits_flat, labels_flat)
                sup = labels_flat != -100
                n_sup = int(sup.sum().item())
                if n_sup <= 0:
                    ce_weighted = logits_flat.sum() * 0.0
                    ce_plain = ce_weighted
                else:
                    w = torch.ones_like(raw, dtype=raw.dtype)
                    if use_struct_ids:
                        ids_on = w_ids.to(device=labels_flat.device, dtype=torch.long)
                        is_sp = torch.isin(labels_flat, ids_on) & sup
                        w = torch.where(is_sp, torch.full_like(w, w_mult), w)
                        w = w * (float(n_sup) / (w * sup).sum().clamp(min=1.0))
                    if use_struct_w:
                        sw_flat = structure_weights.to(device=raw.device, dtype=raw.dtype).reshape(-1)
                        if sw_flat.numel() != raw.numel():
                            raise ValueError(
                                f"structure_weights numel {sw_flat.numel()} != labels {raw.numel()}"
                            )
                        sw_flat = sw_flat * (float(n_sup) / (sw_flat * sup).sum().clamp(min=1.0))
                        w = w * sw_flat
                    if use_scale_w:
                        sc_flat = scale_weights.to(device=raw.device, dtype=raw.dtype).reshape(-1)
                        if sc_flat.numel() != raw.numel():
                            raise ValueError(
                                f"scale_weights numel {sc_flat.numel()} != labels {raw.numel()}"
                            )
                        sc_flat = sc_flat * (float(n_sup) / (sc_flat * sup).sum().clamp(min=1.0))
                        w = w * sc_flat
                    # ---- Final Enhance: Final-SW × PDL × InfoEntropy ----------
                    if _any_enhance:
                        enhance_w = torch.ones_like(raw)
                        sup_2d = (labels != -100)
                        fs = int(getattr(self, "_final_start_id", 32004))
                        fe = int(getattr(self, "_final_end_id", 32005))
                        # half-open final region [<final> .. </final>) via cumsum
                        f_diff = (input_ids == fs).to(torch.int32).cumsum(1) \
                            - (input_ids == fe).to(torch.int32).cumsum(1)
                        final_region = (f_diff > 0) & sup_2d
                        final_flat = final_region.reshape(-1)

                        # Final-SW: exp prefix boost over offset inside final region
                        if _enable_final_sw:
                            eta = float(getattr(self, "_final_sw_eta", 0.1))
                            lam = float(getattr(self, "_final_sw_lambda", 0.8))
                            offset = (final_region.to(torch.int32).cumsum(1) - 1).clamp(min=0).to(raw.dtype)
                            fw = 1.0 + lam * torch.exp(-eta * offset)
                            fw = torch.where(final_region, fw, torch.ones_like(fw))
                            fw_flat = fw.reshape(-1)
                            fw_flat = fw_flat * (float(n_sup) / (fw_flat * sup).sum().clamp(min=1.0))
                            enhance_w = enhance_w * fw_flat

                        # PDL: static per-token frequency lookup over all labels
                        if _enable_pdl:
                            tbl = _pdl_w.to(device=raw.device, dtype=raw.dtype)
                            pdl_lookup = tbl[labels_flat.clamp(0, vocab_size - 1)]
                            pdl_lookup = torch.where(sup, pdl_lookup, torch.ones_like(pdl_lookup))
                            enhance_w = enhance_w * pdl_lookup
                            _fn = final_flat.to(raw.dtype).sum().clamp(min=1.0)
                            pdl_weight_mean = (pdl_lookup * final_flat.to(raw.dtype)).sum() / _fn

                        # IE: dynamic entropy weight (final region; +think if on)
                        if _enable_ie and _probs is not None:
                            H = -(_probs * torch.log(_probs + 1e-12)).sum(dim=-1)
                            Hn = (H / math.log(max(2, vocab_size))).clamp(0.0, 1.0)
                            gamma = float(getattr(self, "_ie_gamma", 0.5))
                            ie_region = final_region
                            if bool(getattr(self, "_ie_on_think", False)):
                                ts2 = int(getattr(self, "_fcp_think_start_id", 32002))
                                te2 = int(getattr(self, "_fcp_think_end_id", 32003))
                                t_diff = (input_ids == ts2).to(torch.int32).cumsum(1) \
                                    - (input_ids == te2).to(torch.int32).cumsum(1)
                                ie_region = ie_region | ((t_diff > 0) & sup_2d)
                            iew = 1.0 + gamma * Hn.to(raw.dtype)
                            iew = torch.where(ie_region, iew, torch.ones_like(iew))
                            iew_flat = iew.reshape(-1)
                            iew_flat = iew_flat * (float(n_sup) / (iew_flat * sup).sum().clamp(min=1.0))
                            enhance_w = enhance_w * iew_flat
                            _ied = ie_region.reshape(-1).to(raw.dtype)
                            ie_weight_mean = (iew_flat * _ied).sum() / _ied.sum().clamp(min=1.0)

                        w = w * enhance_w
                    ce_weighted = (raw * w).sum() / float(n_sup)
                    ce_plain = raw[sup].mean()
            else:
                ce_weighted = self.ce_loss_fn(logits_flat, labels_flat)
                ce_plain = ce_weighted
            if isinstance(total_lb_loss, torch.Tensor):
                total_lb_loss = total_lb_loss.mean(); total_z_loss = total_z_loss.mean()
            n = self.config.num_layers * (4*2 + 1*3)
            lb_contrib = (0.1 / max(1, n)) * total_lb_loss
            z_contrib  = (5e-3 / max(1, n)) * total_z_loss

            # ---- Task 2: FCP penalty (Format / EOS Penalty) -----------------
            # fcp_eos_id / fcp_lambda / _fcp_active resolved in the enhance setup.
            zero = ce_weighted.new_zeros(())
            fcp_penalty = zero
            avg_eos_prob = zero
            max_eos_prob = zero
            if _fcp_active:
                ts = int(getattr(self, "_fcp_think_start_id", 32002))
                te = int(getattr(self, "_fcp_think_end_id", 32003))
                fcp_delta = float(getattr(self, "_fcp_delta", 0.01))
                # cumsum trick for half-open think region mask (no CPU sync)
                is_start = (input_ids == ts).to(torch.int32)
                is_end   = (input_ids == te).to(torch.int32)
                diff = is_start.cumsum(dim=1) - is_end.cumsum(dim=1)
                region_mask = (diff > 0).to(logits.dtype)
                # restrict to trainable positions (labels != -100)
                valid_mask = (labels != -100).to(logits.dtype)
                region_mask = region_mask * valid_mask
                eos_probs = _probs[..., int(fcp_eos_id)].to(logits.dtype)
                excess = F.relu(eos_probs - fcp_delta)
                denom = region_mask.sum().clamp(min=1.0)
                fcp_penalty = (excess * excess * region_mask).sum() / denom * fcp_lambda
                # 僅供 log/CSV；勿在 eos_probs 上再建可微分支（易觸發 backward CUDA 錯誤）
                with torch.no_grad():
                    avg_eos_prob = (eos_probs * region_mask).sum() / denom
                    if region_mask.sum() > 0:
                        masked = eos_probs.masked_fill(region_mask <= 0, float("-inf"))
                        max_eos_prob = masked.amax()

            # Free the [B,T,V] softmax before backward to cap peak VRAM.
            if _probs is not None:
                del _probs

            loss = ce_weighted + lb_contrib + z_contrib + fcp_penalty.to(ce_weighted.dtype)

            # ---- Structural-token top-1 accuracy (detached, no gradient) --------
            # Compute accuracy for </think> (idx 10), </final> (idx 11),
            # <|im_end|> (idx 12).  Returns NaN when a token is absent in the batch.
            with torch.no_grad():
                preds = logits_flat.argmax(dim=-1)   # [B*T]

                def _struct_acc(tok_id: int) -> torch.Tensor:
                    if tok_id < 0:
                        return loss.new_full((), float("nan"))
                    mask = (labels_flat == tok_id)
                    if not mask.any():
                        return loss.new_full((), float("nan"))
                    return (preds[mask] == tok_id).float().mean()

                _think_end_id  = int(getattr(self, "_fcp_think_end_id",  -1))
                _final_end_id2 = int(getattr(self, "_final_end_id",      -1))
                _im_end_id     = int(getattr(self, "_fcp_eos_id",        -1))

                think_end_acc  = _struct_acc(_think_end_id)
                final_end_acc  = _struct_acc(_final_end_id2)
                im_end_acc     = _struct_acc(_im_end_id)

            def _det_aux(t):
                return t.detach() if isinstance(t, torch.Tensor) else t

            return (
                loss.unsqueeze(0),
                total_lb_loss.detach().unsqueeze(0) if isinstance(total_lb_loss, torch.Tensor) else loss.unsqueeze(0),
                ce_plain.detach(),
                _det_aux(lb_contrib),
                _det_aux(z_contrib),
                fcp_penalty.detach(),
                avg_eos_prob.detach(),
                max_eos_prob.detach(),
                pdl_weight_mean.detach(),
                ie_weight_mean.detach(),
                think_end_acc,    # [10] top-1 acc at </think> label positions
                final_end_acc,    # [11] top-1 acc at </final> label positions
                im_end_acc,       # [12] top-1 acc at <|im_end|> label positions
            )
        return logits



