# -*- coding: utf-8 -*-
import os
import gc
import csv
import math
import time
import warnings
import shutil
from contextlib import nullcontext
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from accelerate import Accelerator

warnings.filterwarnings(
    "ignore",
    message=".*Online softmax is disabled on the fly.*",
    category=UserWarning,
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    MIXED_PRECISION = "bf16"
    torch.backends.cuda.matmul.fp32_precision = 'tf32'   # 新版 API，取代 allow_tf32
    torch.backends.cudnn.conv.fp32_precision  = 'tf32'
    print("🚀 高階 GPU 偵測成功，啟用 bf16 + TF32 最佳化！")
else:
    MIXED_PRECISION = "fp16"
    torch.backends.cuda.matmul.fp32_precision = 'ieee'   # 標準 IEEE 浮點數
    torch.backends.cudnn.conv.fp32_precision  = 'ieee'
    print("🐢 舊版 GPU，自動 Fallback 至 fp16。")


class PretokenizedDataset(IterableDataset):
    """
    【資料集】記憶體映射的預先 Tokenize 資料集
    ─────────────────────────────────────────
    直接從 .bin 檔（uint16 token id）以 mmap 方式串流讀取。
    """
    def __init__(self, data_path: str, seq_len: int, buffer_size: int = 4_000_000):
        self.data_path    = data_path
        self.seq_len      = seq_len
        self.buffer_size  = buffer_size
        self.total_tokens = len(np.memmap(data_path, dtype=np.uint16, mode="r"))

    def __iter__(self):
        worker_info = get_worker_info()
        start_idx, end_idx = 0, self.total_tokens
        if worker_info is not None:
            per       = self.total_tokens // worker_info.num_workers
            start_idx = worker_info.id * per
            end_idx   = (self.total_tokens
                         if worker_info.id == worker_info.num_workers - 1
                         else start_idx + per)

        mmap_data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        curr_idx  = start_idx

        while curr_idx + self.seq_len < end_idx:
            chunk_end = min(curr_idx + self.buffer_size, end_idx)
            buffer   = mmap_data[curr_idx:chunk_end].astype(np.int64)
            num_seqs = (len(buffer) - 1) // self.seq_len
            if num_seqs == 0:
                break

            x_t = torch.from_numpy(
                buffer[: num_seqs * self.seq_len].reshape(num_seqs, self.seq_len)
            ) 
            y_t = torch.from_numpy(
                buffer[1 : num_seqs * self.seq_len + 1].reshape(num_seqs, self.seq_len)
            ) 

            for i in range(num_seqs):
                yield x_t[i], y_t[i]

            curr_idx += num_seqs * self.seq_len

def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):   model = model.module
    if hasattr(model, "_orig_mod"): model = model._orig_mod
    return model

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, resume_step: int = 0, rewarmup_steps: int = 100):
    decay_steps = 12000 
    stable_steps = total_steps - warmup_steps - decay_steps

    def lr_lambda(step):
        # 1. 算出「理想狀態」下，目前 step 應該要有的 LR 比例
        if step < warmup_steps:
            target_mult = step / max(1, warmup_steps)
        elif step < warmup_steps + stable_steps:
            progress = (step - warmup_steps) / max(1, stable_steps)
            target_mult = 1.0 - (0.2 * progress) # 從 1.0 緩降到 0.8
        else:
            decay_progress = (step - (warmup_steps + stable_steps)) / max(1, decay_steps)
            min_lr_ratio = 0.05 
            target_mult = min_lr_ratio + 0.5 * (0.8 - min_lr_ratio) * (1.0 + math.cos(math.pi * decay_progress))

        # 2. 如果是接續訓練，強制加上一段短暫的 Rewarmup 避免震盪
        if resume_step > 0 and step >= resume_step and step < resume_step + rewarmup_steps:
            # 從目標值的 10% 開始，線性拉升回 100% 的目標值
            rewarmup_progress = (step - resume_step) / rewarmup_steps
            return target_mult * (0.1 + 0.9 * rewarmup_progress)

        return target_mult

    return LambdaLR(optimizer, lr_lambda)

# ── Triton Kernels & Model Classes ──

@triton.jit
def tanh_approx(x):
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", constraints="=f,f", args=[x],
        dtype=tl.float32, is_pure=True, pack=1,
    )

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
    if step < warmup: return t_start
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    return t_end + 0.5 * (t_start - t_end) * (1.0 + math.cos(math.pi * progress))

class Mamba3Config:
    def __init__(
        self, d_model=768, d_state=64, d_head=64, n_groups=1, mimo_rank=4, expand=4,
        num_layers=15, use_parallel_scan=True, use_kmoe=True,
        kmoe_num_experts=8, kmoe_top_k=2, kmoe_r1=4, kmoe_r2=1024, kmoe_r3=256, ffn_expand=6, num_kv_heads=4,
        dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, layer_scale_init=1e-2, rms_norm_eps=1e-5, chunk_size=64
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

class RMSNorm(nn.RMSNorm):
    def __init__(self, dim, eps=1e-5):
        super().__init__(normalized_shape=dim, eps=eps)    

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-2):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)
    def forward(self, x):
        return x * self.gamma.to(x.dtype)

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
            x_shared.stride(0), x_shared.stride(1),
            G_experts.stride(0), G_experts.stride(1), G_experts.stride(2),
            top_k_indices.stride(0), top_k_indices.stride(1),
            top_k_probs.stride(0), top_k_probs.stride(1),
            out.stride(0), out.stride(1), B, r3, r2, top_k)
        
        return out

    @staticmethod
    def backward(ctx, dout):
        x_shared, G_experts, top_k_indices, top_k_probs = ctx.saved_tensors
        B, r3 = x_shared.shape; E, _, r2 = G_experts.shape; top_k = top_k_indices.size(1)
        dx_shared = torch.zeros_like(x_shared)
        dprobs    = torch.zeros_like(top_k_probs)

        target_dtype = x_shared.dtype
        dout = dout.to(target_dtype)

        for k in range(top_k):
            idx  = top_k_indices[:, k]
            prob = top_k_probs[:, k].unsqueeze(1).to(target_dtype)
            G_k  = G_experts[idx].to(target_dtype)

            dout_k     = dout * prob
            dx_shared += torch.matmul(dout_k.unsqueeze(1), G_k.transpose(1, 2)).squeeze(1)

            dprobs_k        = (dout * torch.matmul(x_shared.unsqueeze(1), G_k).squeeze(1)).sum(dim=-1)
            dprobs[:, k]    = dprobs_k.to(dprobs.dtype)

        dG_experts = torch.zeros_like(G_experts)
        _fused_latent_moe_bwd_dG_kernel[lambda meta: (E, triton.cdiv(r3, meta['BLOCK_R3']), triton.cdiv(r2, meta['BLOCK_R2']))](
            x_shared, dout, top_k_probs, top_k_indices, dG_experts,
            x_shared.stride(0), x_shared.stride(1), dout.stride(0), dout.stride(1),
            top_k_probs.stride(0), top_k_probs.stride(1),
            top_k_indices.stride(0), top_k_indices.stride(1),
            dG_experts.stride(0), dG_experts.stride(1), dG_experts.stride(2), B, top_k, r3, r2)
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

    def forward(self, x, router_temp=None):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        B_flat = x_flat.size(0)

        raw_logits  = self.router(x_flat)
        if router_temp is None:
            temperature = raw_logits.new_tensor(get_router_temperature(None))
        elif isinstance(router_temp, torch.Tensor):
            temperature = router_temp.to(device=raw_logits.device, dtype=raw_logits.dtype)
        else:
            temperature = raw_logits.new_tensor(float(router_temp))
            
            
        temperature = temperature.clamp_min(1e-4)
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

    def forward(self, x, router_temp=None):
        gate, lb_g, z_g = self.gate_proj(x, router_temp=router_temp)
        feat, lb_u, z_u = self.up_proj(x, router_temp=router_temp)
        y,    lb_d, z_d = self.down_proj(fast_silu_gating(gate, feat), router_temp=router_temp)
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
    stride_a_b, stride_a_l, stride_u_b, stride_u_l, stride_u_d,
    B_flat, L: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_b, pid_d = tl.program_id(0), tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offset_l = tl.arange(0, L)
    mask_d = offset_d < D

    alpha_ptrs = log_alpha_ptr + pid_b * stride_a_b + offset_l * stride_a_l
    alpha = tl.exp(tl.load(alpha_ptrs).to(tl.float32))
    
    u_ptrs = u_ptr + pid_b * stride_u_b + offset_l[:, None] * stride_u_l + offset_d[None, :] * stride_u_d
    u = tl.load(u_ptrs, mask=mask_d[None, :], other=0.0).to(tl.float32)

    _, h = tl.associative_scan((tl.broadcast_to(alpha[:, None], (L, BLOCK_D)), u), axis=0, combine_fn=first_order_combine_op)

    h_out_ptrs = h_out_ptr + pid_b * stride_u_b + offset_l[:, None] * stride_u_l + offset_d[None, :] * stride_u_d
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
    stride_a_b, stride_a_l, stride_u_b, stride_u_l, stride_u_d,
    B_flat, L: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_b, pid_d = tl.program_id(0), tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offset_d < D
    offset_l = tl.arange(0, L)
    rev_offset_l = L - 1 - offset_l  

    dh_ptrs = dh_ptr + pid_b * stride_u_b + rev_offset_l[:, None] * stride_u_l + offset_d[None, :] * stride_u_d
    dh = tl.load(dh_ptrs, mask=mask_d[None, :], other=0.0).to(tl.float32)

    alpha_next_idx = L - offset_l
    alpha_next_mask = alpha_next_idx < L
    log_alpha_next = tl.load(log_alpha_ptr + pid_b * stride_a_b + alpha_next_idx * stride_a_l, mask=alpha_next_mask, other=-float('inf')).to(tl.float32)
    alpha_rev = tl.where(alpha_next_mask, tl.exp(log_alpha_next), 0.0)

    _, delta_rev = tl.associative_scan((tl.broadcast_to(alpha_rev[:, None], (L, BLOCK_D)), dh), axis=0, combine_fn=first_order_combine_op)

    du_ptrs = du_ptr + pid_b * stride_u_b + rev_offset_l[:, None] * stride_u_l + offset_d[None, :] * stride_u_d
    tl.store(du_ptrs, delta_rev.to(du_ptr.dtype.element_ty), mask=mask_d[None, :])

    h_prev_idx = L - 2 - offset_l
    h_prev = tl.load(h_ptr + pid_b * stride_u_b + h_prev_idx[:, None] * stride_u_l + offset_d[None, :] * stride_u_d, mask=(h_prev_idx >= 0)[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    alpha_curr_idx = L - 1 - offset_l
    alpha_curr = tl.exp(tl.load(log_alpha_ptr + pid_b * stride_a_b + alpha_curr_idx * stride_a_l).to(tl.float32))

    dlog_alpha_sum = tl.sum(delta_rev * alpha_curr[:, None] * h_prev, axis=1)
    tl.atomic_add(dlog_alpha_ptr + pid_b * stride_a_b + alpha_curr_idx * stride_a_l, dlog_alpha_sum)

class TritonParallelScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_alpha_chunk, u_chunk):
        B, num_chunks, L, H = log_alpha_chunk.shape
        D = u_chunk.shape[-1] * u_chunk.shape[-2]
        
        log_alpha_flat = log_alpha_chunk.transpose(2, 3).reshape(-1, L).contiguous()
        u_flat = u_chunk.transpose(2, 3).reshape(-1, L, D).contiguous()
        
        h_out_flat = torch.empty_like(u_flat)
        B_flat = log_alpha_flat.shape[0]
        
        _chunk_scan_fwd_kernel[lambda meta: (B_flat, triton.cdiv(D, meta['BLOCK_D']))](
            log_alpha_flat, u_flat, h_out_flat,
            log_alpha_flat.stride(0), log_alpha_flat.stride(1),
            u_flat.stride(0), u_flat.stride(1), u_flat.stride(2),
            B_flat=B_flat, L=L, D=D
        )
        
        ctx.save_for_backward(log_alpha_flat, h_out_flat)
        ctx.dims = (B, num_chunks, L, H, u_chunk.shape[-2], u_chunk.shape[-1])
        return h_out_flat.reshape(B, num_chunks, H, L, u_chunk.shape[-2], u_chunk.shape[-1]).transpose(2, 3)

    @staticmethod
    def backward(ctx, dh_out):
        log_alpha_flat, h_out_flat = ctx.saved_tensors
        B, num_chunks, L, H, N, P = ctx.dims
        D = N * P
        B_flat = log_alpha_flat.shape[0]
        
        dh_flat = dh_out.transpose(2, 3).reshape(-1, L, D).contiguous()
        du_flat = torch.empty_like(dh_flat)
        dlog_alpha_flat = torch.zeros_like(log_alpha_flat)
        
        _chunk_scan_bwd_kernel[lambda meta: (B_flat, triton.cdiv(D, meta['BLOCK_D']))](
            log_alpha_flat, h_out_flat, dh_flat, du_flat, dlog_alpha_flat,
            log_alpha_flat.stride(0), log_alpha_flat.stride(1),
            dh_flat.stride(0), dh_flat.stride(1), dh_flat.stride(2),
            B_flat=B_flat, L=L, D=D
        )
        
        return dlog_alpha_flat.reshape(B, num_chunks, H, L).transpose(2, 3), du_flat.reshape(B, num_chunks, H, L, N, P).transpose(2, 3)

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
        x_reshaped = x.float().view(*x.shape[:-2], N_half, 2, x.shape[-1])
        x1, x2 = x_reshaped[..., 0, :], x_reshaped[..., 1, :]
        sin_a, cos_a = torch.sin(angles).unsqueeze(-1), torch.cos(angles).unsqueeze(-1)
        return torch.stack([x1*cos_a - x2*sin_a, x2*cos_a + x1*sin_a], dim=-2).reshape_as(x).type_as(x)

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
        
        # ── 這裡已經接上了我們最新的 TritonParallelScanFn ──
        h_intra = fast_triton_chunk_scan(la_c, u_c)
        
        y_diag  = torch.einsum("bclhnp, bclhnr -> bclhpr", h_intra, C_c)
        decay   = torch.exp(torch.sum(la_c, dim=2))
        h_prev  = torch.zeros(B, H, N, P, device=u.device, dtype=input_dtype)
        h_inter = torch.empty(B, nc, H, N, P, device=u.device, dtype=input_dtype)
        for c in range(nc):
            h_inter[:, c] = h_prev
            h_prev = h_prev * decay[:, c].view(B, H, 1, 1) + h_intra[:, c, -1]
        c_dec = C_c * torch.exp(torch.cumsum(la_c, dim=2)).unsqueeze(-1).unsqueeze(-1)
        y_off = torch.einsum("bchnp, bclhnr -> bclhpr", h_inter, c_dec)
        y = (y_diag + y_off).view(B, -1, H, P, R)
        return (y[:, :L_orig] if L_orig < L else y).to(input_dtype), h_prev.to(input_dtype)

    def forward(self, x, router_temp=None, mamba_cache=None, return_mamba_cache=False):
        B_sz, L, _ = x.shape
        H, G, P, N, R, ratio = self.config.n_heads, self.config.n_groups, self.config.d_head, self.config.d_state, self.config.mimo_rank, self.ratio
        residual_mamba, u = x, self.norm_mamba(x)
        z, x_prime, B_param, C_param, dt, A_param, lambda_param = torch.split(
            self.in_proj(u), [self.dim_z, self.dim_x, self.dim_B, self.dim_C, self.dim_dt, self.dim_A, self.dim_lambda], dim=-1)
        x_prime = x_prime.view(B_sz, L, H, P)
        dt = F.softplus(dt); A = -torch.exp(A_param); theta = torch.exp(self.theta_log)
        bg = lambda t: t.repeat_interleave(ratio, dim=2)
        dt_b = bg(dt.unsqueeze(-1)).squeeze(-1); A_b = bg(A.unsqueeze(-1)).squeeze(-1)
        theta_rep = theta.repeat_interleave(ratio, dim=0)
        current_angle_step = torch.einsum("blh, hn -> blhn", dt_b, theta_rep)
        if mamba_cache is not None:
            assert L == 1, "mamba_cache is only valid for single-token decode (L == 1)."
            prev_h, prev_input, prev_angle_sum = mamba_cache
            angles = prev_angle_sum + torch.cumsum(current_angle_step, dim=1)
        else:
            angles = torch.cumsum(current_angle_step, dim=1)
        B_rotated = self.apply_rope(bg(self.norm_B(B_param.reshape(B_sz, L, G, N * R)).view(B_sz, L, G, N, R) + self.bias_B), angles)
        C_rotated = self.apply_rope(bg(self.norm_C(C_param.reshape(B_sz, L, G, N * R)).view(B_sz, L, G, N, R) + self.bias_C), angles)
        if self.config.use_kmoe:
            x_up, lb_up, z_up = self.x_up_proj(x_prime.view(B_sz, L, -1), router_temp=router_temp)
            x_ssm = x_up.view(B_sz, L, H, P, R)
        else:
            x_ssm, lb_up, z_up = self.x_up_proj(x_prime).view(B_sz, L, H, P, R), 0.0, 0.0
        input_signal = torch.einsum("blhnr, blhpr -> blhnp", B_rotated, x_ssm)
        lv = F.sigmoid(bg(lambda_param.unsqueeze(-1)).squeeze(-1)).view(B_sz, L, H, 1, 1)
        dv = dt_b.view(B_sz, L, H, 1, 1)
        av = torch.exp(dt_b * A_b).view(B_sz, L, H, 1, 1)
        if mamba_cache is not None:
            ip = prev_input
        else:
            ip = torch.roll(input_signal, 1, 1); ip[:, 0] = 0
        u_ssm = lv * dv * input_signal + (1 - lv) * dv * av * ip

        mamba_cache_out = None
        if mamba_cache is not None:
            h_s = prev_h * av[:, 0] + u_ssm[:, 0]
            y_stack = torch.einsum("bhnp,bhnr->bhpr", h_s, C_rotated[:, 0]).unsqueeze(1)
            mamba_cache_out = (h_s, input_signal[:, -1:], angles[:, -1:])
        elif self.config.use_parallel_scan:
            y_stack, h_prev = self.chunk_parallel_scan(u_ssm, dt_b, A_b, C_rotated, chunk_size=self.config.chunk_size)
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
        y = y + x_prime.reshape(B_sz,L,H*P) * self.D.repeat_interleave(P,dim=0)
        mamba_out = self.mamba_dense_proj(self.pre_gate_norm(y) * self.act(z))
        mid_x = residual_mamba + self.ls_mamba(mamba_out)
        residual_proj, normed_mid = mid_x, self.norm_out_proj(mid_x)
        
        if self.config.use_kmoe:
            proj_out, lb_out, z_out = self.out_proj(normed_mid, router_temp=router_temp)
        else:
            proj_out, lb_out, z_out = self.out_proj(normed_mid), 0.0, 0.0
        out = residual_proj + self.ls_out_proj(proj_out)
        if mamba_cache is not None or return_mamba_cache:
            return out, lb_up + lb_out, z_up + z_out, mamba_cache_out
        return out, lb_up + lb_out, z_up + z_out

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

    def forward(self, x, router_temp=None, past_kv=None, seq_pos=0, return_kv=False):
        B, L, D = x.shape; residual, nx = x, self.norm_attn(x)
        q = self.q_proj(nx).view(B, L, self.num_heads, 64).transpose(1, 2)
        k_new = self.k_proj(nx).view(B, L, self.num_kv_heads, 64).transpose(1, 2)
        v_new = self.v_proj(nx).view(B, L, self.num_kv_heads, 64).transpose(1, 2)
        k_new = k_new.unsqueeze(2).expand(B, self.num_kv_heads, self.kv_groups, L, 64).reshape(B, self.num_heads, L, 64)
        v_new = v_new.unsqueeze(2).expand(B, self.num_kv_heads, self.kv_groups, L, 64).reshape(B, self.num_heads, L, 64)
        kv_out = None
        if past_kv is None:
            attn = F.scaled_dot_product_attention(q, k_new, v_new, dropout_p=0.0, is_causal=True)
            if return_kv:
                kv_out = (k_new.detach(), v_new.detach())
        else:
            k_buf, v_buf = past_kv
            k_buf[:, :, seq_pos : seq_pos + L, :] = k_new
            v_buf[:, :, seq_pos : seq_pos + L, :] = v_new
            prefix = seq_pos + L
            attn = F.scaled_dot_product_attention(
                q, k_buf[:, :, :prefix, :], v_buf[:, :, :prefix, :], dropout_p=0.0, is_causal=False
            )
            kv_out = (k_buf, v_buf)
        x = residual + self.ls_attn(self.o_proj(attn.transpose(1, 2).contiguous().view(B, L, D)))
        residual, h = x, self.norm_ffn(x)
        if self.use_kmoe:
            ffn_out, lb, z = self.ffn(h, router_temp=router_temp)
        else:
            ffn_out = self.ffn_down(fast_silu_gating(self.ffn_gate(h), self.ffn_up(h))); lb=0.0; z=0.0
        out = residual + self.ls_ffn(ffn_out)
        if past_kv is not None or return_kv:
            return out, lb, z, kv_out
        return out, lb, z

class TrueHybridMamba(nn.Module):
    def __init__(self, config: Mamba3Config, mamba_ratio=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            for _ in range(mamba_ratio):
                self.layers.append(nn.ModuleDict({"block": Mamba3Block(config)}))
            self.layers.append(nn.ModuleDict({"block": TransformerBlock(config)}))

    def forward(self, x, router_temp=None):
        total_lb, total_z = 0.0, 0.0
        for ld in self.layers:
            if router_temp is None:
                x, lb, z = ld["block"](x, router_temp=None)
            else:
                x, lb, z = checkpoint(ld["block"], x, router_temp, use_reentrant=False)
            if isinstance(lb, torch.Tensor): total_lb = total_lb + lb; total_z = total_z + z
        return x, total_lb, total_z

    def forward_inference(self, x, router_temp, layer_caches, seq_pos: int, prefill: bool):
        """Batch size 1 inference: prefill (prefill=True) or decode (prefill=False) with per-layer caches."""
        total_lb, total_z = 0.0, 0.0
        new_caches = []
        for i, ld in enumerate(self.layers):
            blk = ld["block"]
            if isinstance(blk, TransformerBlock):
                if prefill:
                    x, lb, z, kv_out = blk(
                        x, router_temp=router_temp, past_kv=None, seq_pos=0, return_kv=True
                    )
                else:
                    past = layer_caches[i]
                    x, lb, z, kv_out = blk(
                        x, router_temp=router_temp, past_kv=past, seq_pos=seq_pos, return_kv=False
                    )
                new_caches.append(kv_out)
            else:
                if prefill:
                    x, lb, z, mc_out = blk(
                        x, router_temp=router_temp, mamba_cache=None, return_mamba_cache=True
                    )
                else:
                    x, lb, z, mc_out = blk(
                        x, router_temp=router_temp, mamba_cache=layer_caches[i], return_mamba_cache=False
                    )
                new_caches.append(mc_out)
            if isinstance(lb, torch.Tensor):
                total_lb = total_lb + lb
                total_z = total_z + z
        return x, total_lb, total_z, new_caches

class Mamba3LanguageModel(nn.Module):
    def __init__(self, config: Mamba3Config, vocab_size: int, **kwargs):
        super().__init__()
        self.config = config
        self.embed    = nn.Embedding(vocab_size, config.d_model)
        self.backbone = TrueHybridMamba(config)
        self.norm     = RMSNorm(config.d_model)
        self.head     = nn.Linear(config.d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.ce_loss_fn  = nn.CrossEntropyLoss()
        self._last_loss_terms = None
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, router_temp=None):
        backbone_out = self.backbone(self.embed(input_ids), router_temp=router_temp)
        hidden = self.norm(backbone_out[0])
        total_lb_loss, total_z_loss = backbone_out[1], backbone_out[2]
        logits = fast_scaled_tanh(self.head(hidden / math.sqrt(self.config.d_model)), 30.0)
        if labels is not None:
            ce_loss = self.ce_loss_fn(logits.float().view(-1, logits.size(-1)), labels.view(-1))
            if isinstance(total_lb_loss, torch.Tensor):
                total_lb_loss = total_lb_loss.mean(); total_z_loss = total_z_loss.mean()
            n = self.config.num_layers * (4*2 + 1*3)
            lb_contrib = (0.1 / max(1, n)) * total_lb_loss
            z_contrib  = (5e-3 / max(1, n)) * total_z_loss
            
            loss = ce_loss + lb_contrib + z_contrib
            return (
                loss.unsqueeze(0),
                total_lb_loss.detach().unsqueeze(0) if isinstance(total_lb_loss, torch.Tensor) else loss.unsqueeze(0),
                ce_loss.detach(), lb_contrib.detach(), z_contrib.detach(),
            )
            
            
        return logits

    def forward_inference(self, input_ids, router_temp, layer_caches, seq_pos: int, prefill: bool):
        x = self.embed(input_ids)
        hidden, _, _, new_caches = self.backbone.forward_inference(
            x, router_temp, layer_caches, seq_pos=seq_pos, prefill=prefill
        )
        hidden = self.norm(hidden)
        logits = fast_scaled_tanh(self.head(hidden / math.sqrt(self.config.d_model)), 30.0)
        return logits, new_caches


# ┌──────────────────────────────────────────────────────────────────┐
# │  §10  train() — 完整訓練迴圈                                       │
# └──────────────────────────────────────────────────────────────────┘

def print_model_analysis(model, config, vocab_size):
    total_params = 0; trainable_params = 0; active_params = 0
    bucket = {"embed_head": 0, "mamba_ssm": 0, "cpmoe_router": 0, "cpmoe_U_expert": 0,
              "cpmoe_bias": 0, "layer_scale": 0, "norm": 0, "attn_proj": 0, "other": 0}

    for name, p in model.named_parameters():
        num_p = p.numel(); total_params += num_p
        if not p.requires_grad: continue
        trainable_params += num_p
        if any(k in name for k in ["U_expert", "U_in", "U_out", "core"]):
            active_params += int(num_p * (config.kmoe_top_k / config.kmoe_num_experts))
            bucket["cpmoe_U_expert"] += num_p
        else:
            active_params += num_p
            if "embed" in name or "head.weight" in name:     bucket["embed_head"] += num_p
            elif "router" in name:                           bucket["cpmoe_router"] += num_p
            elif ".bias" in name and any(k in name for k in ("gate_proj","up_proj","down_proj","out_proj","x_up_proj")): bucket["cpmoe_bias"] += num_p
            elif "ls_" in name or "gamma" in name:          bucket["layer_scale"] += num_p
            elif any(k in name for k in ("norm_","norm.")): bucket["norm"] += num_p
            elif any(k in name for k in ("q_proj","k_proj","v_proj","o_proj")): bucket["attn_proj"] += num_p
            elif any(k in name for k in ("in_proj","y_down_proj","mamba_dense","theta_log",".D","bias_B","bias_C")): bucket["mamba_ssm"] += num_p
            else: bucket["other"] += num_p

    num_cpmoe = len([m for _, m in model.named_modules() if "TuckerMoE" in type(m).__name__])
    active_cpmoe = int(bucket["cpmoe_U_expert"] * config.kmoe_top_k / config.kmoe_num_experts)
    W = 68
    print("═" * W)
    print("🚀  Mamba3 Hybrid TuckerMoE  ·  Model Analysis")
    print("═" * W)
    print("📦  【總參數覽表】")
    print(f"   {'總參數量  (Total)':.<42} {total_params/1e6:>8.2f} M")
    print(f"   {'可訓練    (Trainable)':.<42} {trainable_params/1e6:>8.2f} M")
    print(f"   {'⚡ 實際激活 (Active / step)':.<42} {active_params/1e6:>8.2f} M  ({active_params/max(1,trainable_params)*100:.1f}%)")
    print(f"\n{'─'*W}")
    print("🔬  【參數分類明細】")
    labels = {"embed_head": "Embedding + LM-Head (tied)", "mamba_ssm": "Mamba SSM 核心",
              "cpmoe_router": "TuckerMoE Router", "cpmoe_U_expert": "TuckerMoE 核心張量 & 模式矩陣",
              "cpmoe_bias": "TuckerMoE Bias", "layer_scale": "LayerScale γ",
              "norm": "RMSNorm", "attn_proj": "Attention Q/K/V/O", "other": "其他"}
    for key, label in labels.items():
        val = bucket[key]
        if val == 0: continue
        pct = val / max(1, trainable_params) * 100
        bar = "█" * int(pct/2) + "░" * (25 - int(pct/2))
        print(f"   {label:<40} {val/1e6:>7.2f} M  {pct:>5.1f}%  {bar}")
    print(f"\n{'─'*W}")
    print("🧩  【TuckerMoE 分析】")
    print(f"   {'模組總數':.<42} {num_cpmoe:>4} 個")
    print(f"   {'全 Tucker 參數 (全儲存)':.<42} {bucket['cpmoe_U_expert']/1e6:>7.2f} M")
    print(f"   {'每步實際激活':.<42} {active_cpmoe/1e6:>7.2f} M")
    print(f"   {'Tucker Rank (r1, r2, r3)':.<42} {config.kmoe_r1}, {config.kmoe_r2}, {config.kmoe_r3}")
    print(f"   {'FFN 擴張倍率':.<42} {config.ffn_expand}")
    print(f"   {'Vocab Size':.<42} {vocab_size}")
    print("═" * W)


def train(
    # ── 模型超參數
    D_MODEL=768, D_STATE=64, D_HEAD=64, EXPAND=2, NUM_LAYERS=6,
    MIMO_RANK=4, NUM_KV_HEADS=4, CHUNK_SIZE=64,
    # ── TuckerMoE
    KMOE_NUM_EXPERTS=8, KMOE_TOP_K=2,
    KMOE_R1=4, KMOE_R2=1024, KMOE_R3=256, FFN_EXPAND=6,
    USE_KMOE=True,
    # ── 資料集與路徑
    DATA_PATH="data/train.bin",
    OUTPUT_DIR="output/",
    LOG_FILE="output/train_log.csv",
    CHECKPOINT_SAVE_PATH="output/checkpoint.pt",
    PRETRAINED_EMBED_PATH="",
    VOCAB_SIZE=32000,
    SEQ_LEN=512,
    # ── 訓練超參數
    BATCH_SIZE=4,
    GRADIENT_ACCUMULATION_STEPS=8,
    LR=3e-4, WARMUP=500, STEPS=50000, CHECKPOINT_EVERY=500,
    # ── Router 退火
    ROUTER_T_START=2.0, ROUTER_T_END=0.5,
    # ── 模式
    TRAIN_MODE=True,
    # ── torch.compile
    ENABLE_TORCH_COMPILE=True,      # False = 強制停用 torch.compile
    DISABLE_COMPILE_ON_RESUME=True, # 舊相容參數：Dummy Pass 方案下不再自動停用 compile
    COMPILE_MODE="default",       # "default" | "reduce-overhead" | "max-autotune"
    COMPILE_FULLGRAPH=False,      # True = 單體圖（需模型無 Python mutation）
    # ── 診斷
    GRAD_CHECK_INTERVAL=50,       # 每 N steps 印出梯度診斷
):

    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    config = Mamba3Config(
        d_model=D_MODEL, d_state=D_STATE, d_head=D_HEAD, expand=EXPAND,
        num_layers=NUM_LAYERS, use_parallel_scan=True, chunk_size=CHUNK_SIZE, use_kmoe=USE_KMOE,
        kmoe_num_experts=KMOE_NUM_EXPERTS, kmoe_top_k=KMOE_TOP_K,
        kmoe_r1=KMOE_R1, kmoe_r2=KMOE_R2, kmoe_r3=KMOE_R3,
        ffn_expand=FFN_EXPAND, mimo_rank=MIMO_RANK, num_kv_heads=NUM_KV_HEADS,
        layer_scale_init=1e-2,
    )
    model = Mamba3LanguageModel(config, VOCAB_SIZE)

    if accelerator.is_main_process:
        print_model_analysis(model, config, VOCAB_SIZE)

    decay_params = []
    no_decay_params = []

    # 1. 先讀 checkpoint metadata，讓 scheduler 能依 resume step 建立正確的 lambda。
    start_step = 0
    ckpt_cache = None
    if os.path.exists(CHECKPOINT_SAVE_PATH):
        accelerator.print(f"🔍 檢查 Checkpoint 以校準 Scheduler：{CHECKPOINT_SAVE_PATH}")
        ckpt_cache = torch.load(CHECKPOINT_SAVE_PATH, map_location="cpu")
        start_step = ckpt_cache.get("step", 0)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(k in name for k in ["U_expert", "U_in", "U_out", "core", "bias", "norm", "LayerScale"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0}
        ],
        lr=LR, 
        betas=(0.9, 0.95), 
        fused=True
    )
    
    # 3. 建立 Scheduler 並套用 Rewarmup (假設緩衝 100 步)
    scheduler = get_lr_scheduler(
        optimizer, 
        warmup_steps=WARMUP, 
        total_steps=STEPS, 
        resume_step=start_step, 
        rewarmup_steps=100
    )

    # 2. 先只載入模型權重；optimizer / scheduler state 延後到 compile 預熱後。
    if ckpt_cache is not None:
        accelerator.print(f"📂 發現 checkpoint，先僅載入模型權重：{CHECKPOINT_SAVE_PATH}")
        model.load_state_dict(ckpt_cache["model"])
    else:
        # 處理沒有 Checkpoint 時的預訓練 Embedding 掛載邏輯
        if os.path.isfile(PRETRAINED_EMBED_PATH):
            accelerator.print(f"🌟 嘗試掛載預處理 Embedding：{PRETRAINED_EMBED_PATH}")
            pretrained_embed = torch.load(PRETRAINED_EMBED_PATH, map_location="cpu")
            expected_shape   = model.embed.weight.shape
            if pretrained_embed.shape == expected_shape:
                model.embed.weight.data.copy_(pretrained_embed)
                model.head.weight.data.copy_(pretrained_embed)
                accelerator.print(f"✅ 成功將預訓練 Embedding 載入！維度: {expected_shape}")
            else:
                accelerator.print(f"⚠️ 預訓練 Embedding 維度不符，略過載入。")
        else:
            accelerator.print("🌱 沒有 Checkpoint 也沒有預訓練 Embedding，從頭隨機初始化。")

    should_compile = TRAIN_MODE and ENABLE_TORCH_COMPILE
    compile_skip_reason = None
    did_compile = False
    if not TRAIN_MODE:
        compile_skip_reason = "DEBUG 模式"
    elif not ENABLE_TORCH_COMPILE:
        compile_skip_reason = "ENABLE_TORCH_COMPILE=False"

    if start_step > 0 and DISABLE_COMPILE_ON_RESUME and accelerator.is_main_process:
        print("ℹ️  偵測到續訓 checkpoint；目前改採 Dummy Pass 預熱 compile，因此不再因續訓自動停用 torch.compile。")

    if should_compile:
        if accelerator.is_main_process:
            print(f"🔥 [TRAIN] 啟動 torch.compile (mode='{COMPILE_MODE}', fullgraph={COMPILE_FULLGRAPH})...")
        try:
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = False
            model = torch.compile(model, mode=COMPILE_MODE, fullgraph=COMPILE_FULLGRAPH)
            did_compile = True
            if accelerator.is_main_process:
                fg_label = "單體圖加速" if COMPILE_FULLGRAPH else "分段圖編譯加速"
                print(f"✅ torch.compile 成功，進入{fg_label}模式 (mode='{COMPILE_MODE}')。")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"⚠️ torch.compile 啟動失敗，退回 eager 模式: {e}")
    else:
        if accelerator.is_main_process:
            print("⚠️ 跳過 torch.compile。")
            if compile_skip_reason is not None:
                print(f"   原因: {compile_skip_reason}")

    dataset   = PretokenizedDataset(DATA_PATH, SEQ_LEN)

    _dl_workers = 2 if TRAIN_MODE else 0
    _dl_pin_mem = True if TRAIN_MODE else False
    dataloader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=_dl_workers,
        pin_memory=_dl_pin_mem,
    )
    if accelerator.is_main_process:
        print(f"📦 DataLoader: num_workers={_dl_workers}, pin_memory={_dl_pin_mem}")

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # 3. 在 optimizer state 仍為空時，先用 dummy pass 觸發 compile 預熱，避開 resume 峰值顯存。
    if did_compile:
        accelerator.print("🔥 執行 Dummy Pass 觸發 torch.compile 預熱（Optimizer state 尚未載入）...")
        model.train()
        _amp_dtype = torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float16
        dummy_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=accelerator.device)
        dummy_y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=accelerator.device)
        dummy_router_temp = torch.tensor(
            get_router_temperature(
                start_step,
                warmup=WARMUP,
                total=STEPS,
                t_start=ROUTER_T_START,
                t_end=ROUTER_T_END,
            ),
            dtype=torch.float32,
            device=accelerator.device,
        )
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        # 對齊真實第一個 micro-batch：gradient accumulation 初期通常走 no_sync，避免 DDP 同步額外吃掉峰值顯存。
        sync_free_ctx = accelerator.accumulate(model) if GRADIENT_ACCUMULATION_STEPS > 1 else nullcontext()
        with sync_free_ctx:
            with torch.autocast(device_type="cuda", dtype=_amp_dtype):
                dummy_outputs = model(dummy_x, labels=dummy_y, router_temp=dummy_router_temp)
            dummy_loss = dummy_outputs[0].mean()
            accelerator.backward(dummy_loss)
        optimizer.zero_grad(set_to_none=True)
        del dummy_x, dummy_y, dummy_router_temp, dummy_outputs, dummy_loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accelerator.print("✅ Dummy Pass 預熱完成，已釋放暫存梯度與峰值記憶體。")

    # 4. 預熱完成後，才把 optimizer / scheduler state 載回裝置端。
    if ckpt_cache is not None:
        accelerator.print("📦 開始載入 Optimizer 與 Scheduler 狀態...")
        old_lr = ckpt_cache["optimizer"]["param_groups"][0]["lr"]
        optimizer.load_state_dict(ckpt_cache["optimizer"])
        scheduler.load_state_dict(ckpt_cache["scheduler"])
        new_lr = scheduler.get_last_lr()[0]
        accelerator.print(f"✅ 成功載入！將從 step {start_step} 繼續訓練。")
        accelerator.print(f"   📉 [LR 轉換確認] 原本舊規則 LR: {old_lr:.2e} ➡️ 重新起步 LR: {new_lr:.2e}")
        del ckpt_cache
        ckpt_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # 🌟 3. Resume 後快進資料
    # =========================================================================
    if start_step > 0:
        torch.cuda.empty_cache()  # 釋放 checkpoint 載入後暫存的 allocator cache
        # 🚨 [非常重要] 讓 DataLoader 跳過已經訓練過的前 N 步資料
        batches_to_skip = start_step * GRADIENT_ACCUMULATION_STEPS
        accelerator.print(f"⏩ 快進 DataLoader：跳過前 {batches_to_skip} 個微批次資料...")
        dataloader = accelerator.skip_first_batches(dataloader, num_batches=batches_to_skip)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_fp     = open(LOG_FILE, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_fp)
    if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
        log_writer.writerow([
            "step", "loss", "ce_loss", "lb_contrib", "z_contrib",
            "router_temp",
            "lr", "grad_norm", "loss_scale",
            "tokens_seen", "elapsed_s", "step_time_s",
        ])

    if accelerator.is_main_process:
        print_model_analysis(unwrap_model(model), config, VOCAB_SIZE)

        eff_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * accelerator.num_processes
   
        W = 68
        print("═" * W);  print("⚙️   Training Config Summary");  print("═" * W)
        print(f"  【D】訓練超參數")
        print(f"  {'SEQ_LEN':.<38} {SEQ_LEN}")
        print(f"  {'BATCH_SIZE (per GPU)':.<38} {BATCH_SIZE}")
        print(f"  {'GRADIENT_ACCUMULATION_STEPS':.<38} {GRADIENT_ACCUMULATION_STEPS}")
        print(f"  {'→ Effective Batch':.<38} {eff_batch}  ({BATCH_SIZE}×{GRADIENT_ACCUMULATION_STEPS}×{accelerator.num_processes} GPU)")
        print(f"  {'LR':.<38} {LR:.2e}")
        print(f"  {'WARMUP':.<38} {WARMUP} steps")
        print(f"  {'STEPS':.<38} {STEPS}")
        print(f"  {'CHECKPOINT_EVERY':.<38} {CHECKPOINT_EVERY} steps")
        print(f"  {'GRAD_CHECK_INTERVAL':.<38} {GRAD_CHECK_INTERVAL} steps")
        print(f"  {'Mixed Precision':.<38} {MIXED_PRECISION}")
        print("─" * W);  print("  【E】Router 退火")
        print(f"  {'ROUTER_T_START':.<38} {ROUTER_T_START}")
        print(f"  {'ROUTER_T_END':.<38} {ROUTER_T_END}")
        print("─" * W);  print("  【F】模式切換")
        print(f"  {'TRAIN_MODE':.<38} {'✅ 正式訓練' if TRAIN_MODE else '🐛 Debug'}")
        print("─" * W);  print("  【G】torch.compile")
        print(f"  {'COMPILE_MODE':.<38} {COMPILE_MODE}")
        print(f"  {'COMPILE_FULLGRAPH':.<38} {COMPILE_FULLGRAPH}")
        print("─" * W);  print("  【路徑】")
        print(f"  {'DATA_PATH':.<38} {DATA_PATH}")
        print(f"  {'OUTPUT_DIR':.<38} {OUTPUT_DIR}")
        print(f"  {'LOG_FILE':.<38} {LOG_FILE}")
        print(f"  {'CHECKPOINT_SAVE_PATH':.<38} {CHECKPOINT_SAVE_PATH}")
        print("═" * W)

        accelerator.print(f"🚂 開始訓練，目標 {STEPS} steps...")

    model.train()
    global_step = start_step
    tokens_seen = global_step * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN
    t_start     = time.time()
    data_iter   = iter(dataloader)

    while global_step < STEPS:
        step_start = time.time()
        acc_loss   = 0.0
        acc_ce     = 0.0
        acc_lb     = 0.0
        acc_z      = 0.0
        cur_router_temp = get_router_temperature(
            global_step,
            warmup=WARMUP,
            total=STEPS,
            t_start=ROUTER_T_START,
            t_end=ROUTER_T_END,
        )
        # Keep the compiled graph shape-stable by passing temperature as a scalar tensor.
        router_temp_tensor = torch.tensor(cur_router_temp, dtype=torch.float32, device=accelerator.device)
        optimizer.zero_grad()

        for _ in range(GRADIENT_ACCUMULATION_STEPS):
            try:
                x_batch, y_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x_batch, y_batch = next(data_iter)

            with accelerator.accumulate(model):
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()

                _amp_dtype = torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float16
                with torch.autocast(device_type="cuda", dtype=_amp_dtype):
                    outputs = model(x_batch, labels=y_batch, router_temp=router_temp_tensor)
                loss    = outputs[0].mean()
                if torch.isnan(loss) or torch.isinf(loss):
                    accelerator.print("⚠️ 偵測到 Loss NaN/Inf，跳過此微批次 (Micro-batch)！")
                    continue
                accelerator.backward(loss)
                acc_loss += loss.detach().float()
                if len(outputs) >= 5:
                    acc_ce += outputs[2].item() if isinstance(outputs[2], torch.Tensor) else float(outputs[2])
                    acc_lb += outputs[3].item() if isinstance(outputs[3], torch.Tensor) else float(outputs[3])
                    acc_z  += outputs[4].item() if isinstance(outputs[4], torch.Tensor) else float(outputs[4])

        grad_norm = 0.0
        if accelerator.sync_gradients:
            norm_val  = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm = norm_val.item() if isinstance(norm_val, torch.Tensor) else norm_val

        if math.isnan(grad_norm) or math.isinf(grad_norm):
            if accelerator.is_main_process:
                print(f"🚨 [Step {global_step}] 攔截到異常梯度爆炸 (Grad Norm: {grad_norm})！ 放棄本次權重更新。")
            
            if hasattr(accelerator, "scaler") and accelerator.scaler is not None:
                optimizer.step()
                scheduler.step()
            else:
                optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            scheduler.step()

        global_step += 1

        step_tokens  = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN
        tokens_seen += step_tokens

        if accelerator.is_main_process:
            step_time = time.time() - step_start
            avg_loss  = acc_loss / GRADIENT_ACCUMULATION_STEPS
            cur_lr    = scheduler.get_last_lr()[0]
            elapsed   = time.time() - t_start

            if hasattr(accelerator, "scaler") and accelerator.scaler is not None:
                current_loss_scale = accelerator.scaler.get_scale()
            else:
                current_loss_scale = 1.0

            n_accum = GRADIENT_ACCUMULATION_STEPS
            ce_val = acc_ce / n_accum if acc_ce > 0 else float(avg_loss)
            lb_val = acc_lb / n_accum
            z_val  = acc_z  / n_accum

            try:
                current_ppl = math.exp(ce_val) if ce_val < 20 else float('inf')
            except OverflowError:
                current_ppl = float('inf')

            log_writer.writerow([
                global_step,
                f"{avg_loss:.5f}", f"{ce_val:.5f}", f"{lb_val:.5f}", f"{z_val:.5f}",
                f"{cur_router_temp:.4f}",
                f"{cur_lr:.2e}",
                f"{grad_norm:.4f}",
                f"{current_loss_scale:.1f}",
                tokens_seen,
                f"{elapsed:.1f}", f"{step_time:.3f}",
            ])
            log_fp.flush()

            step_tok_per_s = step_tokens / (step_time + 1e-8)

            if TRAIN_MODE:
                print(
                    f"  step {global_step:>6}/{STEPS} | "
                    f"Loss: {avg_loss:.4f} (CE:{ce_val:.4f}, LB:{lb_val:.4f}, Z:{z_val:.4f}) | "
                    f"PPL: {current_ppl:>6.1f} | Grad: {grad_norm:>5.2f} | "
                    f"T_router: {cur_router_temp:.3f} | " 
                    f"LR: {cur_lr:.2e} | Time: {step_time:.2f}s | Tok/s: {step_tok_per_s:.0f}"
                )
            else:
                _ls_flag = ""
                if current_loss_scale < 256:
                    _ls_flag = " ⚠️ Scale↓"
                elif current_loss_scale > 32768:
                    _ls_flag = " ✅ Scale↑"
                print(
                    f"  🐛 step {global_step:>6}/{STEPS} | "
                    f"Loss: {avg_loss:.4f} (CE:{ce_val:.4f}, LB:{lb_val:.4f}, Z:{z_val:.4f}) | "
                    f"PPL: {current_ppl:>6.1f} | Grad: {grad_norm:>5.2f} | "
                    f"T_router: {cur_router_temp:.3f} | "
                    f"Scale: {current_loss_scale:.0f}{_ls_flag} | "
                    f"LR: {cur_lr:.2e} | Time: {step_time:.2f}s | Tok/s: {step_tok_per_s:.0f}"
                )

            # ── 梯度診斷區塊 ─────────────────────────────────────
            if global_step % GRAD_CHECK_INTERVAL == 0:
                print(f"  {'='*65}")
                print(f"  🔍 [Step {global_step}] 基本梯度狀態")
                print(f"      - 梯度總範數 (Grad Norm)  : {grad_norm:.4f}")
                print(f"      - 當前 Router 溫度        : {cur_router_temp:.4f}  "
                      f"({'Warmup 階段' if global_step < WARMUP else '退火中'})")

                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    print("      ⚠️  警告: 梯度出現 NaN 或 Inf！")
                elif grad_norm > 5.0:
                    print("      ⚠️  警告: 梯度異常偏高，有爆炸風險 (Clip 保護中)。")
                elif grad_norm < 1e-4:
                    print("      ⚠️  警告: 梯度幾近消失。")
                else:
                    print("      ✅  狀態: 梯度流動正常。")

                # ==========================================================
                # 📊 逐層梯度分析 (加入過濾與數量限制)
                # ==========================================================
              
                
                print(f"\n  📊 [異常梯度分析] (僅顯示有問題的參數層)")
                print(f"  {'參數名稱 (Param Name)':<45} {'最大值 (Max)':>12} {'最小值 (Min)':>12}")
                print(f"  {'-'*45} {'-'*12} {'-'*12}")
                
                problem_logs = []
                
                # unwrap_model 可防止 DDP/FSDP 前綴干擾名稱
                for name, param in unwrap_model(model).named_parameters():
                    if param.grad is not None:
                        max_g = param.grad.abs().max().item()
                        min_g = param.grad.abs().min().item()
                        
                        # 判斷是否有異常
                        warn_flag = ""
                        if max_g < 1e-7:
                            warn_flag = "⚠️ 消失"
                        elif max_g > 5.0:
                            warn_flag = "💥 爆炸"
                            
                        # 如果有異常，才加入列表準備印出
                        if warn_flag:
                            is_tucker = any(k in name for k in ["U_in", "U_out", "core", "U_expert"])
                            mark = "⭐" if is_tucker else "  "
                            short_name = name if len(name) <= 42 else "..." + name[-39:]
                            
                            problem_logs.append(f"  {mark}{short_name:<43} {max_g:>12.2e} {min_g:>12.2e} {warn_flag}")
                
                # 決定印出邏輯
                if not problem_logs:
                    print("  ✅ 所有參數梯度皆大於 1e-7 且小於 5.0，無異常！")
                else:
                    display_count = len(problem_logs) if SHOW_ALL_PROBLEM_GRADS else min(10, len(problem_logs))
                    
                    # 印出前 N 筆
                    for log_str in problem_logs[:display_count]:
                        print(log_str)
                        
                    # 如果有隱藏的，提示使用者
                    if len(problem_logs) > display_count:
                        print(f"  ... (還有 {len(problem_logs) - display_count} 筆異常被隱藏。將 SHOW_ALL_PROBLEM_GRADS 設為 True 可顯示全部) ...")
                
                print(f"  {'='*65}")

        if global_step % CHECKPOINT_EVERY == 0 and accelerator.is_main_process:
            ckpt_dict = {
                "step":         global_step,
                "model":        unwrap_model(model).state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "config":       config.__dict__,
                "train_mode":   TRAIN_MODE,
                "router_t_start": ROUTER_T_START,
                "router_t_end":   ROUTER_T_END,
            }
            torch.save(ckpt_dict, CHECKPOINT_SAVE_PATH)
            del ckpt_dict
            gc.collect()
            torch.cuda.empty_cache()

            if TRAIN_MODE:
                print(f"  💾 Checkpoint 已儲存（step {global_step}, "
                      f"T_router={cur_router_temp:.4f}）→ {CHECKPOINT_SAVE_PATH}")
            else:
                print(f"  💾 [DEBUG] Checkpoint 已儲存（step {global_step}, "
                      f"T_router={cur_router_temp:.4f}, "
                      f"loss_scale={current_loss_scale:.1f}）→ {CHECKPOINT_SAVE_PATH}")

    log_fp.close()

    if accelerator.is_main_process:
        mode_label = "TRAIN" if TRAIN_MODE else "DEBUG"
        print(f"🎉 [{mode_label}] 訓練完成！共 {global_step} steps，"
              f"累計 {tokens_seen:,} tokens。")
        print(f"   最終 Router 溫度: {get_router_temperature(global_step, warmup=WARMUP, total=STEPS, t_start=ROUTER_T_START, t_end=ROUTER_T_END):.4f} "
              f"（目標 T_end={ROUTER_T_END}）")

        accelerator.end_training()


# ┌──────────────────────────────────────────────────────────────────┐
# │  §11  Entry Point — 所有超參數集中在此                          │
# └──────────────────────────────────────────────────────────────────┘

if __name__ == "__main__":

    # ════════════════════════════════════════════════
    # 【A】模型超參數
    # ════════════════════════════════════════════════
    D_MODEL      = 768       # 模型隱藏層維度
    D_STATE      = 64        # SSM 狀態維度
    D_HEAD       = 64        # 每個 Head 的維度
    EXPAND       = 2         # d_inner = EXPAND * D_MODEL
    
    NUM_LAYERS   = 6         # Macro Block 數量 (每個含 4 Mamba + 1 Attn)
    
    MIMO_RANK    = 4         # SSM MIMO Rank R
    CHUNK_SIZE   = 64        # Parallel Scan Chunk Size
    # GQA
    NUM_KV_HEADS = 4         # GQA KV-Head 數

    # ════════════════════════════════════════════════
    # 【B】TuckerMoE 超參數
    # ════════════════════════════════════════════════
    KMOE_NUM_EXPERTS = 8     # 專家數量 E
    KMOE_TOP_K       = 2     # 每 token 激活 top-k 個專家
    
    KMOE_R1          = 32     # Tucker 專家維度 Rank
    KMOE_R2          = 512  # Tucker 輸出 Rank (虛擬容量)
    KMOE_R3          = 256   # Tucker 輸入壓縮 Rank
    
    FFN_EXPAND       = 6     # Transformer FFN 擴展比


    VOCAB_SIZE            = 32007


    # ════════════════════════════════════════════════
    # 【C】資料集與路徑
    # ════════════════════════════════════════════════
    DATA_PATH             = "/kaggle/input/datasets/s990093/fineweb-edu-tokenized-32007/fineweb_tokenized.bin"  # 預先 tokenize 好的 .bin 檔
    OUTPUT_DIR            = "/kaggle/working/"
    LOG_FILE              = "/kaggle/working/train_log.csv"
    CHECKPOINT_SAVE_PATH  = "/kaggle/working/checkpoint.pt"
    PRETRAINED_EMBED_PATH = ""                         # 留空則不載入

    # ════════════════════════════════════════════════
    # 【D】訓練超參數
    # ════════════════════════════════════════════════
    SEQ_LEN                  = 512
    BATCH_SIZE               = 2    # Per-GPU batch size
    GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = BATCH * ACCUM * n_gpu
    LR = 8e-5

    WARMUP = 400   # 4%
    STEPS                    = 60000 # 總訓練 steps
    CHECKPOINT_EVERY         = 500   # 每 N steps 存一次

    # ════════════════════════════════════════════════
    # 【E】Router 退火設定
    # ════════════════════════════════════════════════
    ROUTER_T_START = 2.0     # 初始路由溫度（高 → 更均勻分配）
    ROUTER_T_END   = 0.5     # 最終路由溫度（低 → 更集中）

    # ════════════════════════════════════════════════
    # 【F】模式切換
    # ════════════════════════════════════════════════
    TRAIN_MODE = True        # True = 正式訓練 | False = Debug 模式

    # ════════════════════════════════════════════════
    # 【G】torch.compile 設定
    # ════════════════════════════════════════════════
    ENABLE_TORCH_COMPILE = True
    DISABLE_COMPILE_ON_RESUME = True  # 舊相容開關；resume 時改由 Dummy Pass 預熱 compile
    COMPILE_MODE      = "default"  # "default" | "reduce-overhead" | "max-autotune"
    COMPILE_FULLGRAPH = False      # True = 單體圖（需模型無 Python mutation）

    # ════════════════════════════════════════════════
    # 【H】診斷設定
    # ════════════════════════════════════════════════
    GRAD_CHECK_INTERVAL = 100      # 每 N steps 印出梯度診斷
    SHOW_ALL_PROBLEM_GRADS = False

    # ── 啟動訓練 ────────────────────────────────────
    train(
        D_MODEL=D_MODEL, D_STATE=D_STATE, D_HEAD=D_HEAD,
        EXPAND=EXPAND, NUM_LAYERS=NUM_LAYERS,
        MIMO_RANK=MIMO_RANK, NUM_KV_HEADS=NUM_KV_HEADS, CHUNK_SIZE=CHUNK_SIZE,
        KMOE_NUM_EXPERTS=KMOE_NUM_EXPERTS, KMOE_TOP_K=KMOE_TOP_K,
        KMOE_R1=KMOE_R1, KMOE_R2=KMOE_R2, KMOE_R3=KMOE_R3, FFN_EXPAND=FFN_EXPAND,
        USE_KMOE=True,
        DATA_PATH=DATA_PATH, OUTPUT_DIR=OUTPUT_DIR,
        LOG_FILE=LOG_FILE, CHECKPOINT_SAVE_PATH=CHECKPOINT_SAVE_PATH,
        PRETRAINED_EMBED_PATH=PRETRAINED_EMBED_PATH, VOCAB_SIZE=VOCAB_SIZE,
        SEQ_LEN=SEQ_LEN,
        BATCH_SIZE=BATCH_SIZE,
        GRADIENT_ACCUMULATION_STEPS=GRADIENT_ACCUMULATION_STEPS,
        LR=LR, WARMUP=WARMUP, STEPS=STEPS, CHECKPOINT_EVERY=CHECKPOINT_EVERY,
        ROUTER_T_START=ROUTER_T_START, ROUTER_T_END=ROUTER_T_END,
        TRAIN_MODE=TRAIN_MODE,
        ENABLE_TORCH_COMPILE=ENABLE_TORCH_COMPILE,
        DISABLE_COMPILE_ON_RESUME=DISABLE_COMPILE_ON_RESUME,
        COMPILE_MODE=COMPILE_MODE, COMPILE_FULLGRAPH=COMPILE_FULLGRAPH,
        GRAD_CHECK_INTERVAL=GRAD_CHECK_INTERVAL,
    )
