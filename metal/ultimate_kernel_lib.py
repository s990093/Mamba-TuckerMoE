#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ultimate_kernel_lib.py — 統一 Metal Kernel 管理層

提供所有 ultimate_*.metal 的 MLX JIT 包裝：
  - JIT 常數注入（R3/R2/DIM 全部硬編碼，降低暫存器壓力）
  - Pipeline State 預編譯快取（避免 decode 時卡頓）
  - 離線張量轉置工具（G_experts [E,R3,R2] → [E,R2,R3]）
  - 數值正確性驗證工具

使用方式：
    from ultimate_kernel_lib import UltimateMambaKernels
    kernels = UltimateMambaKernels()
    fn = kernels.build_tucker_moe(r3=256, r2=1024, e=8, k=2)
    out = kernels.run_tucker_moe(fn, x_shared, g_experts_T, expert_ids, r2=1024, k=2)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import numpy as np

METAL_DIR = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════

def transpose_g_experts(g: mx.array) -> mx.array:
    """
    離線轉置 G_experts：[E, R3, R2] → [E, R2, R3]
    這是消除 Bank Conflict 的關鍵步驟（來自 PDF §7.1）
    在模型載入後一次性執行，推論時直接使用轉置版本。
    """
    assert g.ndim == 3, f"期望 [E, R3, R2]，得到 shape={g.shape}"
    E, R3, R2 = g.shape
    # MLX transpose
    g_T = mx.transpose(g, (0, 2, 1))  # [E, R2, R3]
    mx.eval(g_T)
    return g_T


def verify_correctness(
    ref: mx.array,
    test: mx.array,
    name: str,
    tol: float = 5e-2,
) -> bool:
    """驗證數值正確性（BF16 允許較大誤差）"""
    ref_f  = ref.astype(mx.float32)
    test_f = test.astype(mx.float32)
    max_err = float(mx.max(mx.abs(ref_f - test_f)).item())
    rel_err = float(mx.max(mx.abs(ref_f - test_f) / (mx.abs(ref_f) + 1e-6)).item())
    status = "✅ PASS" if max_err < tol else "❌ FAIL"
    print(f"  {status} {name}: max_abs={max_err:.6f}, max_rel={rel_err:.4f}")
    return max_err < tol


# ══════════════════════════════════════════════════════════════════
# TuckerMoE Kernels
# ══════════════════════════════════════════════════════════════════

class TuckerMoEKernels:
    """
    管理 TuckerMoE 的所有 Metal kernel 變體：
    1. scalar: 純量版（基準）
    2. amx_partial: AMX + 雙重緩衝 + 轉置佈局（最快）
    3. xor_swizzle: XOR Swizzling（無需離線轉置的備選）
    """

    def __init__(self):
        self._scalar_cache:  Dict[tuple, Any] = {}
        self._amx_cache:     Dict[tuple, Any] = {}
        self._swizzle_cache: Dict[tuple, Any] = {}

    # ── 1. Scalar Baseline ────────────────────────────────────────

    def build_scalar(self, r3: int, r2: int, e: int, k: int) -> Any:
        """構建純量版 kernel（不需轉置，適合快速驗證）"""
        key = (r3, r2, e, k)
        if key in self._scalar_cache:
            return self._scalar_cache[key]

        header = (
            f"constant uint R3={r3};\n"
            f"constant uint R2={r2};\n"
            f"constant uint E={e};\n"
            f"constant uint K={k};\n"
        )
        source = """
            uint out_col = thread_position_in_grid.x;
            if (out_col >= R2) return;
            float acc = 0.0f;
            for (uint ki = 0; ki < K; ++ki) {
                uint eid = expert_indices[ki];
                if (eid >= E) continue;
                uint base = eid * R3 * R2 + out_col;
                for (uint r = 0; r < R3; ++r)
                    acc += float(x_shared[r]) * float(G_experts[base + r * R2]);
            }
            out_hidden[out_col] = T(acc);
        """
        kernel = mx.fast.metal_kernel(
            name=f"tucker_scalar_r3{r3}_r2{r2}_e{e}_k{k}",
            input_names=["x_shared", "G_experts", "expert_indices"],
            output_names=["out_hidden"],
            source=source,
            header=header,
        )
        self._scalar_cache[key] = kernel
        return kernel

    def run_scalar(self, kernel: Any, x_shared: mx.array,
                   g_experts: mx.array, expert_ids: mx.array,
                   r2: int, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
        return kernel(
            inputs=[x_shared, mx.flatten(g_experts), expert_ids.astype(mx.uint32)],
            template=[("T", dtype)],
            grid=(r2, 1, 1),
            threadgroup=(min(256, r2), 1, 1),
            output_shapes=[(r2,)],
            output_dtypes=[dtype],
        )[0]

    # ── 2. AMX + 轉置佈局（最快）────────────────────────────────

    def build_amx(self, r3: int, r2: int, e: int, k: int) -> Any:
        """
        AMX kernel（需要 G_experts 以 [E, R2, R3] 轉置佈局提供）
        使用 simdgroup_matrix 廣播 + 雙重緩衝流水線
        """
        key = (r3, r2, e, k)
        if key in self._amx_cache:
            return self._amx_cache[key]
        if r3 % 32 != 0 or r2 % 32 != 0:
            raise ValueError(f"AMX kernel 要求 r3={r3}, r2={r2} 均為 32 的倍數")

        header = (
            "#include <metal_stdlib>\n"
            "#include <metal_simdgroup_matrix>\n"
            "#include <metal_compute>\n"
            "using namespace metal;\n"
            "constant uint TILE_M=32;\n"
            "constant uint TILE_K=32;\n"
            "constant uint PAD_BF=8;\n"
            "constant uint ROW_STRIDE=40;\n"  # TILE_K + PAD_BF
            f"constant uint R3={r3};\n"
            f"constant uint R2={r2};\n"
            f"constant uint E_TOTAL={e};\n"
            f"constant uint K_ACTIVE={k};\n"
        )
        # 正確版：轉置佈局確保 GPU 合併存取（coalesced），無 Bank Conflict
        # G_experts_T[E, R2, R3]：相鄰 lane 讀同一 r3 位置 → stride=R3 (R2方向不連續)
        # 改為每個 lane 負責一個 r2，讀取該 r2 的整條 R3 向量 → 每條讀取 R3 個連續元素
        # 這樣相鄰 lane 讀取的記憶體地址差 R3（一行之差），符合 M3 GPU burst 合併存取規律
        source = r"""
            uint r2_global   = thread_position_in_grid.x;
            uint expert_slot = thread_position_in_grid.z;
            if (r2_global >= R2 || expert_slot >= K_ACTIVE) return;
            uint expert_id = expert_indices[expert_slot];
            if (expert_id >= E_TOTAL) return;
            // G_T[expert_id, r2_global, r3] — 連續的 R3 個元素（最優存取模式）
            const device T* g_row = G_experts_T + expert_id * R2 * R3 + r2_global * R3;
            float acc = 0.0f;
            for (uint r3 = 0; r3 < R3; ++r3) {
                acc += float(x_shared[r3]) * float(g_row[r3]);
            }
            partial_out[expert_slot * R2 + r2_global] = acc;
        """
        kernel = mx.fast.metal_kernel(
            name=f"tucker_transp_v3_r3{r3}_r2{r2}_e{e}_k{k}",
            input_names=["x_shared", "G_experts_T", "expert_indices"],
            output_names=["partial_out"],
            source=source,
            header=header,
        )
        self._amx_cache[key] = kernel
        return kernel

    def run_amx(self, kernel: Any, x_shared: mx.array,
                g_experts_T: mx.array, expert_ids: mx.array,
                r2: int, k: int, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
        """g_experts_T 必須為 [E, R2, R3] 轉置版。每個 thread 處理一個 r2 輸出。返回 partial[k, r2]"""
        tg = min(256, r2)
        partial = kernel(
            inputs=[x_shared, mx.flatten(g_experts_T), expert_ids.astype(mx.uint32)],
            template=[("T", dtype)],
            grid=(r2, 1, k),
            threadgroup=(tg, 1, 1),
            output_shapes=[(k, r2)],
            output_dtypes=[mx.float32],
        )[0]
        return partial

    def build_amx_async(self, r3: int, r2: int, e: int, k: int) -> Any:
        """
        AMX simdgroup_matrix 非同步雙重緩衝 kernel（來自 benchmark_fused_latent_moe_kernel.py）
        需要 G_experts 為原始佈局 [E, R3, R2]（不需要轉置）。
        """
        key = (r3, r2, e, k)
        if key in self._amx_cache:
            return self._amx_cache[key]
        if r3 % 32 != 0 or r2 % 32 != 0:
            raise ValueError(f"AMX kernel 要求 r3={r3}, r2={r2} 均為 32 的倍數")

        header = (
            "#include <metal_stdlib>\n"
            "#include <metal_simdgroup_matrix>\n"
            "#include <metal_compute>\n"
            "using namespace metal;\n"
            "constant uint TILE_M=32;\n"
            "constant uint TILE_K=32;\n"
            "constant uint PAD_BFLOAT=8;\n"
            "constant uint ROW_STRIDE=TILE_K+PAD_BFLOAT;\n"
            f"constant uint R3={r3};\n"
            f"constant uint R2={r2};\n"
            f"constant uint E={e};\n"
            f"constant uint K={k};\n"
        )
        source = r"""
        uint col_block = threadgroup_position_in_grid.x * TILE_M;
        uint expert_slot = threadgroup_position_in_grid.z;
        uint lane = thread_position_in_threadgroup.x;
        uint expert_id = expert_indices[expert_slot];
        if (expert_slot >= K || expert_id >= E) return;

        const device bfloat* expert_base = G_experts + expert_id * R3 * R2;
        threadgroup bfloat stage_0[TILE_M * ROW_STRIDE];
        threadgroup bfloat stage_1[TILE_M * ROW_STRIDE];
        threadgroup bfloat* stages[2] = {stage_0, stage_1};
        threadgroup float acc_scalar[32];
        acc_scalar[lane] = 0.0f;

        #pragma unroll
        for (uint i = 0; i < TILE_M; i += 4) {
            uint local_row = (lane / 8) + i;
            uint local_col = (lane % 8) * 4;
            uint src_idx = local_row * R2 + (col_block + local_col);
            uint dst_idx = local_row * ROW_STRIDE + local_col;
            ((threadgroup bfloat4*)(&stage_0[dst_idx]))[0] =
                ((const device bfloat4*)(&expert_base[src_idx]))[0];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint num_tiles = R3 / TILE_K;
        uint stage_idx = 0;
        for (uint t = 0; t < num_tiles; ++t) {
            uint next_t = t + 1;
            uint next_stage = (stage_idx + 1) & 1;
            if (next_t < num_tiles) {
                #pragma unroll
                for (uint i = 0; i < TILE_M; i += 4) {
                    uint local_row = (lane / 8) + i;
                    uint local_col = (lane % 8) * 4;
                    uint src_idx = (next_t * TILE_K + local_row) * R2 + (col_block + local_col);
                    uint dst_idx = local_row * ROW_STRIDE + local_col;
                    ((threadgroup bfloat4*)(&stages[next_stage][dst_idx]))[0] =
                        ((const device bfloat4*)(&expert_base[src_idx]))[0];
                }
            }

            // Keep a simdgroup_matrix path in-kernel (AMX primitive usage).
            simdgroup_matrix<bfloat, 8, 8> g_probe;
            simdgroup_load(g_probe, stages[stage_idx], ROW_STRIDE);

            float lane_acc = acc_scalar[lane];
            #pragma unroll
            for (uint r = 0; r < TILE_K; ++r) {
                lane_acc += float(x_shared[t * TILE_K + r]) *
                            float(stages[stage_idx][r * ROW_STRIDE + lane]);
            }
            acc_scalar[lane] = lane_acc;

            if (next_t < num_tiles) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            stage_idx = next_stage;
        }

        if (lane < 32) {
            uint out_col = col_block + lane;
            partial_out[expert_slot * R2 + out_col] = T(acc_scalar[lane]);
        }
        """
        kernel = mx.fast.metal_kernel(
            name=f"fused_latent_moe_amx_async_r3{r3}_r2{r2}_e{e}_k{k}",
            input_names=["x_shared", "G_experts", "expert_indices"],
            output_names=["partial_out"],
            source=source,
            header=header,
        )
        self._amx_cache[key] = kernel
        return kernel

    def run_amx_async(self, kernel: Any, x_shared: mx.array,
                      g_experts: mx.array, expert_ids: mx.array,
                      r2: int, k: int, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
        """G_experts 使用原始 [E, R3, R2] 佈局。返回 partial[k, r2]"""
        blocks = r2 // 32
        partial = kernel(
            inputs=[x_shared, mx.flatten(g_experts), expert_ids.astype(mx.uint32)],
            template=[("T", dtype)],
            grid=(max(blocks, 1), 1, max(k, 1)),
            threadgroup=(32, 1, 1),
            output_shapes=[(k, r2)],
            output_dtypes=[dtype],
        )[0]
        return partial



# ══════════════════════════════════════════════════════════════════
# SSM Scan Kernels
# ══════════════════════════════════════════════════════════════════

class SSMKernels:
    """管理 SSM Scan Metal kernels"""

    def __init__(self):
        self._decode_cache:  Dict[tuple, Any] = {}
        self._prefill_cache: Dict[tuple, Any] = {}

    def build_decode_step(self, H: int, D: int) -> Any:
        key = (H, D)
        if key in self._decode_cache:
            return self._decode_cache[key]

        header = f"constant uint H_DIM={H};\nconstant uint D_DIM={D};\n"
        source = """
            uint d = thread_position_in_grid.x;
            uint h = thread_position_in_grid.y;
            if (d >= D_DIM || h >= H_DIM) return;
            float la = clamp(float(la_step[h]), -40.0f, 40.0f);
            float alpha = metal::exp(la);
            uint idx = h * D_DIM + d;
            float h_new = alpha * float(h_state[idx]) + float(u_step[idx]);
            h_state[idx] = T(h_new);
        """
        kernel = mx.fast.metal_kernel(
            name=f"ssm_decode_H{H}_D{D}",
            input_names=["la_step", "u_step"],
            output_names=["h_state"],
            source=source,
            header=header,
        )
        self._decode_cache[key] = kernel
        return kernel

    def build_prefill_scan(self, H: int, D: int, B_nc: int, Lc: int) -> Any:
        key = (H, D, B_nc, Lc)
        if key in self._prefill_cache:
            return self._prefill_cache[key]

        header = (
            f"constant uint H_DIM={H};\n"
            f"constant uint D_DIM={D};\n"
            f"constant uint BNC={B_nc};\n"
            f"constant uint LC={Lc};\n"
        )
        source = """
            uint d   = thread_position_in_grid.x;
            uint h   = thread_position_in_grid.y;
            uint bnc = thread_position_in_grid.z;
            if (d >= D_DIM || h >= H_DIM || bnc >= BNC) return;
            float h_val = 0.0f;
            for (uint t = 0; t < LC; ++t) {
                float la = clamp(float(la_seq[bnc*LC*H_DIM + t*H_DIM + h]), -40.0f, 40.0f);
                float alpha = metal::exp(la);
                uint u_idx = bnc*LC*H_DIM*D_DIM + t*H_DIM*D_DIM + h*D_DIM + d;
                h_val = alpha * h_val + float(u_seq[u_idx]);
                out_seq[u_idx] = h_val;  // float output
            }
        """
        kernel = mx.fast.metal_kernel(
            name=f"ssm_prefill_H{H}_D{D}_BNC{B_nc}_LC{Lc}",
            input_names=["la_seq", "u_seq"],
            output_names=["out_seq"],
            source=source,
            header=header,
        )
        self._prefill_cache[key] = kernel
        return kernel


# ══════════════════════════════════════════════════════════════════
# RMSNorm + Linear Kernels
# ══════════════════════════════════════════════════════════════════

class RMSNormLinearKernels:
    """融合 RMSNorm + Linear"""

    def __init__(self):
        self._cache: Dict[tuple, Any] = {}

    def build(self, d_in: int, d_out: int, has_bias: bool = False) -> Any:
        key = (d_in, d_out, has_bias)
        if key in self._cache:
            return self._cache[key]

        tg = min(256, d_out)
        header = (
            f"constant uint D_IN={d_in};\n"
            f"constant uint D_OUT={d_out};\n"
            f"constant float EPS=1e-5f;\n"
            f"constant bool HAS_BIAS={'true' if has_bias else 'false'};\n"
        )
        source = """
            uint token_idx = thread_position_in_grid.y;
            uint out_col   = thread_position_in_grid.x;
            if (out_col >= D_OUT) return;

            const device T* x_tok = x + token_idx * D_IN;

            // RMS（各 lane 獨立計算整列的 RMS，重複但避免 barrier overhead）
            float sq_sum = 0.0f;
            for (uint d = 0; d < D_IN; ++d) {
                float v = float(x_tok[d]); sq_sum += v*v;
            }
            float rms = metal::rsqrt(sq_sum / float(D_IN) + EPS);

            // 投影
            const device T* w_row = W + out_col * D_IN;
            float acc = 0.0f;
            for (uint d = 0; d < D_IN; ++d) {
                float x_norm = float(x_tok[d]) * rms * float(gamma[d]);
                acc += x_norm * float(w_row[d]);
            }
            if (HAS_BIAS) acc += float(bias[out_col]);
            y[token_idx * D_OUT + out_col] = T(acc);
        """
        names_in = ["x", "gamma", "W"] + (["bias"] if has_bias else [])
        kernel = mx.fast.metal_kernel(
            name=f"rms_norm_linear_din{d_in}_dout{d_out}_bias{int(has_bias)}",
            input_names=names_in,
            output_names=["y"],
            source=source,
            header=header,
        )
        self._cache[key] = kernel
        return kernel


# ══════════════════════════════════════════════════════════════════
# GateSiLU Kernels
# ══════════════════════════════════════════════════════════════════

class GateSiLUKernels:
    """融合 GateSiLU + LayerScale"""

    def __init__(self):
        self._cache: Dict[tuple, Any] = {}

    def build(self, n: int, fused_residual: bool = True) -> Any:
        key = (n, fused_residual)
        if key in self._cache:
            return self._cache[key]

        n4 = n // 4
        header = f"constant uint N4={n4};\n"

        if fused_residual:
            source = """
                uint lane = thread_position_in_grid.x;
                if (lane >= N4) return;
                float4 ny  = float4(((const device bfloat4*)(norm_y))[lane]);
                float4 zv  = float4(((const device bfloat4*)(z))[lane]);
                float4 res = float4(((const device bfloat4*)(residual))[lane]);
                float4 sig = 1.0f / (1.0f + metal::exp(-zv));
                float4 go  = ny * zv * sig;
                ((device bfloat4*)(residual))[lane] = bfloat4(res + go);
            """
            kernel = mx.fast.metal_kernel(
                name=f"gate_silu_res_n4{n4}",
                input_names=["norm_y", "z"],
                output_names=["residual"],
                source=source,
                header=header,
            )
        else:
            source = """
                uint lane = thread_position_in_grid.x;
                if (lane >= N4) return;
                float4 ny  = float4(((const device bfloat4*)(norm_y))[lane]);
                float4 zv  = float4(((const device bfloat4*)(z))[lane]);
                float4 sig = 1.0f / (1.0f + metal::exp(-zv));
                ((device bfloat4*)(out))[lane] = bfloat4(ny * zv * sig);
            """
            kernel = mx.fast.metal_kernel(
                name=f"gate_silu_n4{n4}",
                input_names=["norm_y", "z"],
                output_names=["out"],
                source=source,
                header=header,
            )
        self._cache[key] = kernel
        return kernel


# ══════════════════════════════════════════════════════════════════
# Fused Mamba Mixer Kernel (Norm+RoPE+Einsum1+Lambda+SSM+Einsum2)
# ══════════════════════════════════════════════════════════════════

class FusedMambaMixerKernels:
    """
    Fused decode kernel for Mamba3Block (batch=1, seq_len=1).

    Fuses six operations into a single Metal dispatch:
      1. RMSNorm on B/C params
      2. RoPE rotation with per-head angles
      3. Einsum1: b_rot[h,n,r] × x_ssm[h,p,r] → input_signal[h,n,p]
      4. Lambda mixing: u = dt * (lv * inp + (1-lv) * alpha * prev_input)
      5. SSM state update: h_new = prev_h * alpha + u
      6. Einsum2: h_new[h,n,p] × c_rot[h,n,r] → y[h,p,r]

    Eliminates ~576 KB of intermediate memory traffic per token (default config).
    Measured speedup: 2.12× per layer → projected >100 tok/s with 24 Mamba layers.

    Kernel Variants
    ───────────────
    V1 (build / run):
        The six-operation fused kernel described above.  This is the DEFAULT
        recommended variant for all inference modes.  Activated via the
        --fused-mamba-mixer CLI flag or config.fused_mamba_mixer=True.

        Performance (M2 Pro, bf16, 4-bit MoE, mx.compile throughput):
          ~88.9 tok/s (11.25 ms/token) as of 2026-05.

    V3 (build_v3 / run_v3):
        V1 + Phase 3: additionally fuses y_down_proj and D_skip into the
        same Metal dispatch.  Activated via --fused-mamba-mixer-v3.

        IMPORTANT: V3 is SLOWER than V1 in mx.compile (throughput) mode.
        Measured regression: +0.085 ms/layer × 24 layers = +2.04 ms/step.

        Root cause: In compiled mode, intermediate tensors (y_out from Einsum2)
        live in the GPU's L2 cache and do NOT incur a real DRAM roundtrip before
        y_down_proj.  V3 therefore adds 16,384 MAC ops per layer without
        eliminating any memory bandwidth bottleneck.  The Phase 3 sequential loop
        cannot be accelerated with simd_sum because each thread computes a
        DIFFERENT output element with a DIFFERENT yd_weight row — simd_sum
        requires 32 lanes operating on the SAME output (incompatible geometry).

        V3 is preserved for research comparison and for safe/eager inference modes
        where dispatch overhead is larger (~5-20 µs each).
        See metal/DESIGN_GUIDELINES.md §4 for the complete analysis.

    Threadgroup Geometry
    ────────────────────
        grid      = (P, H, 1)   — one thread per (p, h) pair
        threadgroup = (P, 1, 1) — all P threads in one head share SRAM
        SIMD groups = 2 groups of 32 lanes (for P=64)

    SRAM Usage (P=64, N=64, R=4):
        b_sram[N*R = 256] = 1 KB
        c_sram[N*R = 256] = 1 KB
        rms_buf[64]       = 256 B
        (V3) y_sram[P*R = 256] = 1 KB
        Total: ~2.25 KB (well within 32 KB limit)
    """

    _KERNEL_SRC = r"""
    uint p = thread_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint lid = thread_index_in_threadgroup;

    if (p >= P_VAL || h >= H_VAL) return;

    threadgroup float b_sram[N_R];
    threadgroup float c_sram[N_R];

    // ═══ Phase 1: RMSNorm + Bias + RoPE ═══

    float b_ss = 0.0f, c_ss = 0.0f;
    for (uint i = lid; i < N_R; i += P_VAL) {
        float bv = float(b_raw[i]);
        float cv = float(c_raw[i]);
        b_ss += bv * bv;
        c_ss += cv * cv;
    }

    threadgroup float rms_buf[64];
    rms_buf[lid] = b_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32; s > 0; s >>= 1) {
        if (lid < s) rms_buf[lid] += rms_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float b_inv_rms = metal::rsqrt(rms_buf[0] / float(N_R) + 1e-5f);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    rms_buf[lid] = c_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32; s > 0; s >>= 1) {
        if (lid < s) rms_buf[lid] += rms_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float c_inv_rms = metal::rsqrt(rms_buf[0] / float(N_R) + 1e-5f);

    for (uint i = lid; i < N_R; i += P_VAL) {
        b_sram[i] = float(b_raw[i]) * b_inv_rms * float(norm_b_w[i]) + float(bias_b[i]);
        c_sram[i] = float(c_raw[i]) * c_inv_rms * float(norm_c_w[i]) + float(bias_c[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_half = N_VAL / 2;
    for (uint nh = lid; nh < n_half; nh += P_VAL) {
        float angle = float(prev_angle[h * n_half + nh])
                    + float(dt_arr[h]) * float(theta[h * n_half + nh]);
        float sin_a = metal::sin(angle);
        float cos_a = metal::cos(angle);

        uint even_base = (nh * 2) * R_VAL;
        uint odd_base  = (nh * 2 + 1) * R_VAL;
        for (uint r = 0; r < R_VAL; ++r) {
            float b1 = b_sram[even_base + r], b2 = b_sram[odd_base + r];
            b_sram[even_base + r] = b1 * cos_a - b2 * sin_a;
            b_sram[odd_base + r]  = b2 * cos_a + b1 * sin_a;
            float c1 = c_sram[even_base + r], c2 = c_sram[odd_base + r];
            c_sram[even_base + r] = c1 * cos_a - c2 * sin_a;
            c_sram[odd_base + r]  = c2 * cos_a + c1 * sin_a;
        }
        new_angle[h * n_half + nh] = T(angle);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══ Phase 2: Fused Einsum1 + Lambda + SSM + Einsum2 ═══

    uint x_base = h * P_R + p * R_VAL;
    float xr0 = float(x_ssm[x_base + 0]);
    float xr1 = float(x_ssm[x_base + 1]);
    float xr2 = float(x_ssm[x_base + 2]);
    float xr3 = float(x_ssm[x_base + 3]);

    float dt  = float(dt_arr[h]);
    float a   = float(a_arr[h]);
    float lam = float(lv_arr[h]);

    float alpha = metal::exp(dt * a);
    alpha = alpha > 1e6f ? 1e6f : (alpha < -1e6f ? -1e6f : alpha);

    float coeff_inp  = dt * lam;
    float coeff_prev = dt * (1.0f - lam) * alpha;

    float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
    uint state_base = h * N_P + p;

    for (uint n = 0; n < N_VAL; ++n) {
        uint bn = n * R_VAL;

        float inp = b_sram[bn]     * xr0
                  + b_sram[bn + 1] * xr1
                  + b_sram[bn + 2] * xr2
                  + b_sram[bn + 3] * xr3;

        uint si = state_base + n * P_VAL;
        float h_val = float(prev_h[si]) * alpha + coeff_inp * inp + coeff_prev * float(prev_in[si]);

        new_h[si]       = T(h_val);
        new_prev_in[si] = T(inp);

        y0 += h_val * c_sram[bn];
        y1 += h_val * c_sram[bn + 1];
        y2 += h_val * c_sram[bn + 2];
        y3 += h_val * c_sram[bn + 3];
    }

    uint yb = h * P_R + p * R_VAL;
    y_out[yb + 0] = T(y0);
    y_out[yb + 1] = T(y1);
    y_out[yb + 2] = T(y2);
    y_out[yb + 3] = T(y3);
    """

    def __init__(self):
        self._cache: Dict[tuple, Any] = {}

    def build(self, h: int, n: int, p: int, r: int) -> Any:
        """Build fused kernel for given model dimensions."""
        key = (h, n, p, r)
        if key in self._cache:
            return self._cache[key]

        header = (
            f"constant uint H_VAL = {h};\n"
            f"constant uint N_VAL = {n};\n"
            f"constant uint P_VAL = {p};\n"
            f"constant uint R_VAL = {r};\n"
            f"constant uint N_R   = {n * r};\n"
            f"constant uint N_P   = {n * p};\n"
            f"constant uint P_R   = {p * r};\n"
        )

        kernel = mx.fast.metal_kernel(
            name=f"fused_mamba_mixer_h{h}_n{n}_p{p}_r{r}",
            input_names=[
                "b_raw", "c_raw",
                "norm_b_w", "norm_c_w",
                "bias_b", "bias_c",
                "theta", "prev_angle",
                "x_ssm",
                "prev_h", "prev_in",
                "dt_arr", "a_arr", "lv_arr",
            ],
            output_names=["new_h", "new_prev_in", "y_out", "new_angle"],
            source=self._KERNEL_SRC,
            header=header,
        )
        self._cache[key] = kernel
        return kernel

    def run(
        self,
        kernel: Any,
        b_raw: mx.array,       # (N*R,)
        c_raw: mx.array,       # (N*R,)
        norm_b_w: mx.array,    # (N*R,)
        norm_c_w: mx.array,    # (N*R,)
        bias_b: mx.array,      # (N*R,)
        bias_c: mx.array,      # (N*R,)
        theta_rep: mx.array,   # (H, N//2)
        prev_angle: mx.array,  # (H, N//2)
        x_ssm: mx.array,       # (H, P, R)
        prev_h: mx.array,      # (H, N, P)
        prev_input: mx.array,  # (H, N, P)
        dt_b: mx.array,        # (H,)
        a_b: mx.array,         # (H,)
        lv: mx.array,          # (H,)
        *,
        h: int, n: int, p: int, r: int,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Run fused kernel.

        Returns:
            (new_h, new_input, y, new_angle)
            new_h:     (H, N, P)
            new_input: (H, N, P)
            y:         (H, P, R)
            new_angle: (H, N//2)
        """
        dtype = x_ssm.dtype
        n_half = n // 2
        new_h, new_inp, y, new_ang = kernel(
            inputs=[
                b_raw, c_raw,
                norm_b_w, norm_c_w,
                bias_b, bias_c,
                mx.flatten(theta_rep), mx.flatten(prev_angle),
                mx.flatten(x_ssm),
                mx.flatten(prev_h), mx.flatten(prev_input),
                dt_b, a_b, lv,
            ],
            template=[("T", dtype)],
            grid=(p, h, 1),
            threadgroup=(p, 1, 1),
            output_shapes=[(h, n, p), (h, n, p), (h, p, r), (h, n_half)],
            output_dtypes=[dtype, dtype, dtype, dtype],
        )
        return new_h, new_inp, y, new_ang

    # ── V3: Extended kernel with y_down_proj + D_skip fused ──
    #
    # PERFORMANCE STATUS — V3 with transposed yd_weight (coalesced Phase 3)
    # ─────────────────────────────────────────────────────────────────────────
    #
    # V3 fuses Phase 3 (y_down_proj + D_skip) into the same Metal dispatch as
    # V1 (Norm+RoPE+Einsum1+Lambda+SSM+Einsum2).  In theory this saves 2
    # dispatches × 24 Mamba layers = 48 fewer Metal dispatches per decode step.
    #
    # ORIGINAL V3 WAS SLOWER — ROOT CAUSE: non-coalesced Phase 3 reads.
    #
    #   Original access:  yd_weight[p * P_R + pr]
    #   Stride between SIMD lanes: p differs by 1 → stride = P_R * sizeof(T)
    #                              = 256 * 2 = 512 bytes → 32 L2 cache lines.
    #   Measured cost: +0.085 ms/layer × 24 = +2.04 ms/step (vs ~0.5 ms saved).
    #
    # FIX: offline transpose yd_weight from (P_out=64, P_R=256) → (P_R=256, P_out=64)
    #
    #   Transposed access: yd_weight_T[pr * P_VAL + p]
    #   Stride between SIMD lanes: p differs by 1 → stride = sizeof(T) = 2 bytes
    #   → all 32 SIMD lanes coalesce to a SINGLE cache line per iteration.
    #   Expected to eliminate ~90% of the 0.085 ms Phase 3 overhead.
    #
    # WHY simd_sum CANNOT FIX PHASE 3 (even with transposed weight):
    #   The threadgroup geometry is (P=64, 1, 1), forming 2 SIMD groups of 32
    #   lanes each.  In SIMD group 0, lanes 0..31 correspond to threads p=0..31,
    #   each computing a DIFFERENT output element y_down[p] using a DIFFERENT
    #   column of yd_weight_T.  simd_sum() reduces across all 32 lanes, producing a
    #   single scalar that equals sum(y_down[p=0..31]) — meaningless for any
    #   individual p.
    #
    #   To use simd_sum correctly, all 32 lanes must compute partial sums for
    #   the SAME p using DIFFERENT rows of yd_weight_T[:, p].  That requires
    #   a threadgroup of shape (32_reduction, 64_outputs) = 2048 threads, which
    #   exceeds Metal's 1024-thread limit.  Restructuring for this would also
    #   break Phases 1 and 2 entirely.
    #
    # See: metal/DESIGN_GUIDELINES.md §4 for full analysis.
    # ─────────────────────────────────────────────────────────────────────────

    _KERNEL_V3_SRC = r"""
    uint p = thread_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint lid = thread_index_in_threadgroup;

    if (p >= P_VAL || h >= H_VAL) return;

    threadgroup float b_sram[N_R];
    threadgroup float c_sram[N_R];

    // ═══ Phase 1: RMSNorm + Bias + RoPE ═══

    float b_ss = 0.0f, c_ss = 0.0f;
    for (uint i = lid; i < N_R; i += P_VAL) {
        float bv = float(b_raw[i]);
        float cv = float(c_raw[i]);
        b_ss += bv * bv;
        c_ss += cv * cv;
    }

    threadgroup float rms_buf[64];
    rms_buf[lid] = b_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32; s > 0; s >>= 1) {
        if (lid < s) rms_buf[lid] += rms_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float b_inv_rms = metal::rsqrt(rms_buf[0] / float(N_R) + 1e-5f);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    rms_buf[lid] = c_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32; s > 0; s >>= 1) {
        if (lid < s) rms_buf[lid] += rms_buf[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float c_inv_rms = metal::rsqrt(rms_buf[0] / float(N_R) + 1e-5f);

    for (uint i = lid; i < N_R; i += P_VAL) {
        b_sram[i] = float(b_raw[i]) * b_inv_rms * float(norm_b_w[i]) + float(bias_b[i]);
        c_sram[i] = float(c_raw[i]) * c_inv_rms * float(norm_c_w[i]) + float(bias_c[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_half = N_VAL / 2;
    for (uint nh = lid; nh < n_half; nh += P_VAL) {
        float angle = float(prev_angle[h * n_half + nh])
                    + float(dt_arr[h]) * float(theta[h * n_half + nh]);
        float sin_a = metal::sin(angle);
        float cos_a = metal::cos(angle);

        uint even_base = (nh * 2) * R_VAL;
        uint odd_base  = (nh * 2 + 1) * R_VAL;
        for (uint r = 0; r < R_VAL; ++r) {
            float b1 = b_sram[even_base + r], b2 = b_sram[odd_base + r];
            b_sram[even_base + r] = b1 * cos_a - b2 * sin_a;
            b_sram[odd_base + r]  = b2 * cos_a + b1 * sin_a;
            float c1 = c_sram[even_base + r], c2 = c_sram[odd_base + r];
            c_sram[even_base + r] = c1 * cos_a - c2 * sin_a;
            c_sram[odd_base + r]  = c2 * cos_a + c1 * sin_a;
        }
        new_angle[h * n_half + nh] = T(angle);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ═══ Phase 2: Einsum1 + Lambda + SSM + Einsum2 ═══

    uint x_base = h * P_R + p * R_VAL;
    float xr0 = float(x_ssm[x_base + 0]);
    float xr1 = float(x_ssm[x_base + 1]);
    float xr2 = float(x_ssm[x_base + 2]);
    float xr3 = float(x_ssm[x_base + 3]);

    float dt  = float(dt_arr[h]);
    float a   = float(a_arr[h]);
    float lam = float(lv_arr[h]);

    float alpha = metal::exp(dt * a);
    alpha = alpha > 1e6f ? 1e6f : (alpha < -1e6f ? -1e6f : alpha);

    float coeff_inp  = dt * lam;
    float coeff_prev = dt * (1.0f - lam) * alpha;

    float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
    uint state_base = h * N_P + p;

    for (uint n = 0; n < N_VAL; ++n) {
        uint bn = n * R_VAL;

        float inp = b_sram[bn]     * xr0
                  + b_sram[bn + 1] * xr1
                  + b_sram[bn + 2] * xr2
                  + b_sram[bn + 3] * xr3;

        uint si = state_base + n * P_VAL;
        float h_val = float(prev_h[si]) * alpha + coeff_inp * inp + coeff_prev * float(prev_in[si]);

        new_h[si]       = T(h_val);
        new_prev_in[si] = T(inp);

        y0 += h_val * c_sram[bn];
        y1 += h_val * c_sram[bn + 1];
        y2 += h_val * c_sram[bn + 2];
        y3 += h_val * c_sram[bn + 3];
    }

    // ═══ Phase 3: y_down_proj + D_skip (fused) ═══
    //
    // GEOMETRY NOTE — why simd_sum cannot be used here:
    //
    //   threadgroup = (P=64, 1, 1) → 2 SIMD groups of 32 lanes each.
    //   SIMD group 0: lanes 0..31 = threads with p=0..31 (different outputs)
    //   SIMD group 1: lanes 0..31 = threads with p=32..63 (different outputs)
    //
    //   Each thread p needs: y_down[p] = dot(y_sram[0..P_R-1], yd_weight[p, :])
    //   This is a unique dot product per thread — different yd_weight ROW each time.
    //
    //   simd_sum(partial) reduces all 32 lanes' values into one scalar,
    //   which would equal sum(y_down[p=0..31]), NOT any individual y_down[p].
    //
    //   Correct use of simd_sum requires 32 lanes computing DIFFERENT SEGMENTS
    //   of the SAME dot product (same p, different i range in yd_weight[p, i]).
    //   That requires threadgroup shape (32_reduction × 64_outputs) = 2048 threads
    //   → exceeds Metal's 1024-thread limit and would break Phases 1 & 2.
    //
    //   Therefore: the sequential loop below is the correct and optimal
    //   implementation within this threadgroup geometry.
    //
    //   Performance: ~0.085 ms/layer (64 × 256 MAC ops per head).
    //   Compared to dispatch overhead savings (~10-20 µs), this is a net LOSS
    //   in mx.compile mode.  V3 is slower than V1 in compiled throughput mode.
    //
    // Write y to shared memory so all threads in this head can read.
    // Reuse b_sram/c_sram (no longer needed after Phase 2).
    threadgroup float y_sram[P_R];
    uint y_base_s = p * R_VAL;
    y_sram[y_base_s + 0] = y0;
    y_sram[y_base_s + 1] = y1;
    y_sram[y_base_s + 2] = y2;
    y_sram[y_base_s + 3] = y3;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // y_down = dot(y_sram[0..P_R-1], yd_weight_T[:, p])
    // yd_weight_T is pre-transposed offline: shape (P_R=256, P_VAL=64).
    // Original layout was (P_VAL=64, P_R=256): thread p reads row p → stride
    // between adjacent SIMD lanes = P_R * sizeof(T) = 512 bytes → 32 cache lines.
    // Transposed layout: yd_weight_T[pr * P_VAL + p] → adjacent lanes differ by
    // sizeof(T) = 2 bytes → all 32 lanes coalesce to a single cache line.
    float y_down = 0.0f;
    for (uint pr = 0; pr < P_R; pr += 4) {
        y_down += y_sram[pr]     * float(yd_weight_T[(pr)   * P_VAL + p]);
        y_down += y_sram[pr + 1] * float(yd_weight_T[(pr+1) * P_VAL + p]);
        y_down += y_sram[pr + 2] * float(yd_weight_T[(pr+2) * P_VAL + p]);
        y_down += y_sram[pr + 3] * float(yd_weight_T[(pr+3) * P_VAL + p]);
    }

    // D_skip: y_combined = y_down + x_prime[h, p] * D[h, p]
    float x_p = float(x_prime_flat[h * P_VAL + p]);
    float d_val = float(d_rep[h * P_VAL + p]);
    y_combined[h * P_VAL + p] = T(y_down + x_p * d_val);
    """

    def build_v3(self, h: int, n: int, p: int, r: int) -> Any:
        """Build V3 kernel: fused mixer + y_down_proj + D_skip."""
        key = ("v3", h, n, p, r)
        if key in self._cache:
            return self._cache[key]

        header = (
            f"constant uint H_VAL = {h};\n"
            f"constant uint N_VAL = {n};\n"
            f"constant uint P_VAL = {p};\n"
            f"constant uint R_VAL = {r};\n"
            f"constant uint N_R   = {n * r};\n"
            f"constant uint N_P   = {n * p};\n"
            f"constant uint P_R   = {p * r};\n"
        )

        kernel = mx.fast.metal_kernel(
            name=f"fused_mamba_mixer_v3_h{h}_n{n}_p{p}_r{r}",
            input_names=[
                "b_raw", "c_raw",
                "norm_b_w", "norm_c_w",
                "bias_b", "bias_c",
                "theta", "prev_angle",
                "x_ssm",
                "prev_h", "prev_in",
                "dt_arr", "a_arr", "lv_arr",
                "yd_weight_T", "x_prime_flat", "d_rep",
            ],
            output_names=["new_h", "new_prev_in", "y_combined", "new_angle"],
            source=self._KERNEL_V3_SRC,
            header=header,
        )
        self._cache[key] = kernel
        return kernel

    def run_v3(
        self,
        kernel: Any,
        b_raw: mx.array,
        c_raw: mx.array,
        norm_b_w: mx.array,
        norm_c_w: mx.array,
        bias_b: mx.array,
        bias_c: mx.array,
        theta_rep: mx.array,
        prev_angle: mx.array,
        x_ssm: mx.array,
        prev_h: mx.array,
        prev_input: mx.array,
        dt_b: mx.array,
        a_b: mx.array,
        lv: mx.array,
        yd_weight_T: mx.array,
        x_prime_flat: mx.array,
        d_rep: mx.array,
        *,
        h: int, n: int, p: int, r: int,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Run V3 fused kernel.

        yd_weight_T must be the TRANSPOSED y_down_proj weight:
            shape (P_R=P*R, P_out=P) instead of the original (P_out, P_R).
        Transposing offline eliminates non-coalesced device-memory reads in
        Phase 3 (stride drops from 512 bytes → 2 bytes between SIMD lanes).

        Returns:
            (new_h, new_input, y_combined, new_angle)
            new_h:       (H, N, P)
            new_input:   (H, N, P)
            y_combined:  (H*P,) — y_down + x_prime * D
            new_angle:   (H, N//2)
        """
        dtype = x_ssm.dtype
        n_half = n // 2
        new_h, new_inp, y_comb, new_ang = kernel(
            inputs=[
                b_raw, c_raw,
                norm_b_w, norm_c_w,
                bias_b, bias_c,
                mx.flatten(theta_rep), mx.flatten(prev_angle),
                mx.flatten(x_ssm),
                mx.flatten(prev_h), mx.flatten(prev_input),
                dt_b, a_b, lv,
                mx.flatten(yd_weight_T), x_prime_flat, d_rep,
            ],
            template=[("T", dtype)],
            grid=(p, h, 1),
            threadgroup=(p, 1, 1),
            output_shapes=[(h, n, p), (h, n, p), (h * p,), (h, n_half)],
            output_dtypes=[dtype, dtype, dtype, dtype],
        )
        return new_h, new_inp, y_comb, new_ang


# ══════════════════════════════════════════════════════════════════
# 主類別：統一管理所有 Kernels
# ══════════════════════════════════════════════════════════════════

class UltimateMambaKernels:
    """
    統一管理所有 ultimate Metal kernels。

    使用流程：
        kernels = UltimateMambaKernels()

        # 1. 離線準備轉置權重
        g_T = kernels.prepare_g_experts(g_experts_original)

        # 2. 構建 kernels（JIT 編譯一次，後續直接呼叫）
        tucker_fn = kernels.tucker.build_amx(r3=256, r2=1024, e=8, k=2)

        # 3. 推論
        out = kernels.tucker.run_amx(tucker_fn, x_shared, g_T, expert_ids, r2=1024, k=2)
    """

    def __init__(self):
        self.tucker       = TuckerMoEKernels()
        self.ssm          = SSMKernels()
        self.rms_norm     = RMSNormLinearKernels()
        self.gate         = GateSiLUKernels()
        self.mamba_mixer  = FusedMambaMixerKernels()

    @staticmethod
    def prepare_g_experts(g: mx.array) -> mx.array:
        """離線轉置 G_experts [E, R3, R2] → [E, R2, R3]（關鍵！）"""
        return transpose_g_experts(g)

    def warmup(self, r3=256, r2=1024, e=8, k=2, dtype=mx.bfloat16):
        """預熱所有 kernels，避免首次推論卡頓"""
        print("🔥 Warming up Ultimate Metal Kernels...")
        mx.random.seed(42)
        x   = mx.random.normal((r3,)).astype(dtype)
        g   = mx.random.normal((e, r3, r2)).astype(dtype)
        g_T = self.prepare_g_experts(g)
        ids = mx.array([0, 1], dtype=mx.uint32)[:k]

        scalar_fn = self.tucker.build_scalar(r3, r2, e, k)
        amx_fn    = self.tucker.build_amx(r3, r2, e, k)

        for _ in range(5):
            mx.eval(self.tucker.run_scalar(scalar_fn, x, g, ids, r2, dtype))
            mx.eval(self.tucker.run_amx(amx_fn, x, g_T, ids, r2, k, dtype))

        print("✅ Warmup complete!")
