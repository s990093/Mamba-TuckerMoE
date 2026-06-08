# Mamba3-XR: MLX Kernel Fusion & Metal Optimization Plan

**Target**: 80+ decode tok/s, 400+ prefill tok/s, bit-exact numerical equivalence  
**Scope**: `mamba3_mlx/mlx_model/` (6 files, ~660 lines)  
**Device**: Apple Silicon (M4-family GPU, Metal 3)

---

## ✅ IMPLEMENTED (2026-06-06) — 47 → 70 tok/s via mx.compile

> All correctness checks pass (max diff ≤ 7e-6 vs uncompiled reference; bitwise identical for most paths).  
> Prefill is unaffected — compiled paths only activate for L=1 decode.

### Results summary

| Stage | tok/s | ms/tok | Change |
|-------|-------|--------|--------|
| Baseline (no compile) | 47.3 | 21.14 | — |
| + TuckerMoE per-instance compile | 58.7 | 17.03 | +11.4 |
| + Mamba3Block compiled decode | 64.8 | 15.42 | +6.1 |
| + TransformerBlock compiled decode | 67.8 | 14.74 | +3.0 |
| + Model head compiled | **69.8–70.4** | **14.2–14.4** | +2.2 |

---

### What was done and why it works

#### Root cause of decode slowness

At B=1, L=1, each individual matmul is tiny (e.g. `(1,768)@(768,3490)`) — essentially a memory-bandwidth-bound GEMV. The GPU finishes each kernel in ~0.03–0.05 ms. But each Metal kernel also carries:

- **Command buffer submission overhead** — ~5–10 µs CPU-side per kernel
- **GPU scheduler latency** — ~30–50 µs minimum per kernel launch

With 30 blocks × ~60 ops/block = **1 800 Metal kernel dispatches per token step**, this overhead alone accounts for ~90 ms at 50 µs/dispatch — far more than the actual compute.

`mx.compile` solves this by **tracing the Python compute graph once** and replaying it as a pre-compiled Metal program on subsequent calls. Multiple elementwise ops are fused, and the number of program dispatches falls from ~1 800 to ~80.

#### Why not compile the full model?

`TransformerBlock` concatenates KV cache each step:
```python
k = mx.concatenate([past_k, k_new], axis=2)  # S grows by 1 every decode step
```
`mx.compile` re-traces whenever input shapes change. Compiling the full model resulted in **1.57× SLOWER** (29.7 vs 46.8 tok/s) because it re-traced on every step.

---

### Implementation details

#### 1. TuckerMoE per-instance compile — `mlx_model/tucker_moe.py`

**The problem**: Each `TuckerMoE.__call__` re-computed `G_experts = einsum("er,rst->est", U_expert, core)` every call, wasting ~200 Metal kernel dispatches per step. Even after caching G, the 15–25 ops per call each launched a separate Metal kernel.

**What changed**:
```python
# Before: one __call__ method doing everything
def __call__(self, x, temperature=0.5): ...

# After: split implementation from dispatch
def _forward(self, x, temperature=0.5): ...   # full implementation
def __call__(self, x, temperature=0.5):        # dispatch wrapper
    if self._compiled_call is not None and temperature == 0.5:
        return self._compiled_call(x, temperature)
    return self._forward(x, temperature)
```

In `precompute_G_experts()`:
```python
# 1. Precompute G = U_expert ⊗ core in float32 and bf16
G = mx.einsum("er,rst->est", self.U_expert.astype(mx.float32), self.core.astype(mx.float32))
G_flat = G.transpose(1, 0, 2).reshape(self._r3, self.num_experts * self._r2)
mx.eval(G, G_flat)
self._G_experts_cache = G
self._G_flat_cache = G_flat
# bf16 variants avoid a 4 MB cast every forward call
self._G_experts_cache_bf16 = G.astype(mx.bfloat16)
self._G_flat_cache_bf16 = G_flat.astype(mx.bfloat16)
mx.eval(self._G_experts_cache_bf16, self._G_flat_cache_bf16)

# 2. Compile _forward and warm up once
self._compiled_call = mx.compile(self._forward)
mx.eval(self._compiled_call(mx.zeros((1, self.dim_in), dtype=mx.bfloat16)))
```

**Key**: `_forward` has a Python branch `if B_flat > self.top_k * E` that selects prefill vs decode path. `mx.compile` traces through this branch once and bakes in the result — the decode path (B_flat=1 < 16) always takes the `else` branch, so the compiled graph is a fixed decode-path program.

---

#### 2. Mamba3Block compiled decode — `mlx_model/mamba_block.py`

**The problem**: `Mamba3Block.__call__` has multiple Python conditionals on state (`if state is None`, `if state.get("angles_cum") is None`). `mx.compile` cannot trace through Python conditionals that depend on runtime values (whether state is None or not), so the full `__call__` cannot be compiled.

**What changed**: Add a separate `_decode_impl` method with **no Python state conditionals** — it hardcodes the L=1 path and assumes state is always provided as explicit array arguments:

```python
def _decode_impl(self, x, h_prev, prev_input_signal, angles_cum):
    # L=1 path — no 'if state is None' checks
    B_sz = x.shape[0]
    # ... full L=1 decode logic ...
    
    # Key: call _forward directly to INLINE TuckerMoE ops into this compiled graph
    x_up = self.x_up_proj._forward(x_prime_hp.reshape(B_sz, 1, H * P))
    # ...
    proj_out = self.out_proj._forward(normed_mid)
    
    return out, h_new, new_prev_input_signal, new_angles_cum
```

Calling `self.x_up_proj._forward(...)` directly (not `__call__`) inlines the TuckerMoE ops into the outer `mx.compile` graph. This is better than having two nested compiled programs — the ops from x_up_proj and out_proj are fused into the same Metal program as the surrounding Mamba ops.

In `precompute()`:
```python
self._compiled_decode = mx.compile(self._decode_impl)
# Eager warmup — traces the graph for (1, 1, d_model) bf16 inputs
mx.eval(self._compiled_decode(
    mx.zeros((1, 1, d), dtype=mx.bfloat16),   # x
    mx.zeros((1, H, N, P), dtype=mx.bfloat16), # h_prev
    mx.zeros((1, H, N, P), dtype=mx.bfloat16), # prev_input_signal
    mx.zeros((1, H, N//2), dtype=mx.float32),  # angles_cum
))
```

In `__call__`, dispatch condition:
```python
if (L == 1 and state is not None
        and self._compiled_decode is not None
        and state.get("h_prev") is not None):
    out, new_h, new_ip, new_ac = self._compiled_decode(
        x, state["h_prev"], state["prev_input_signal"], state["angles_cum"])
    return out, {"h_prev": new_h, "prev_input_signal": new_ip, "angles_cum": new_ac}
# ... existing prefill path below ...
```

**Why `state.get("h_prev") is not None`**: After prefill, all state values are proper arrays. We only use the compiled path when state is fully initialized; the very first decode step (state=None) falls through to the original path.

---

#### 3. TransformerBlock compiled decode — `mlx_model/transformer_block.py`

**The problem**: `TransformerBlock` cannot be compiled as a whole because the KV cache concatenation changes shapes. However, the output of `scaled_dot_product_attention` is always `(B, H, 1, head_dim)` = `(1, 12, 1, 64)` regardless of KV length — the query length is always 1 for decode. So everything *except* the KV concat and SDPA itself has fixed shapes.

**What changed**: Split the decode path into two compilable functions around the non-compilable KV section:

```python
def _decode_pre(self, x):
    """norm_attn + q/k/v projections — fixed shapes."""
    B, L, D = x.shape
    nx = self.norm_attn(x)
    q = self.q_proj(nx).reshape(B, L, self.num_heads, 64).transpose(0, 2, 1, 3)
    k = self.k_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)
    v = self.v_proj(nx).reshape(B, L, self.num_kv_heads, 64).transpose(0, 2, 1, 3)
    return q, k, v

def _decode_post(self, attn, x):
    """o_proj + residual + norm_ffn + full inlined FFN — fixed shapes.
    
    attn is (B, H, 1, 64) — fixed regardless of KV cache length.
    x   is (B, 1, D) — fixed for decode.
    """
    B, L, D = x.shape
    attn_out = attn.transpose(0, 2, 1, 3).reshape(B, L, D)
    x = x + self.ls_attn(self.o_proj(attn_out))
    h = self.norm_ffn(x)
    # Inline all 3 TuckerMoE forward passes
    gate = self.ffn.gate_proj._forward(h)
    feat = self.ffn.up_proj._forward(h)
    ffn_out = self.ffn.down_proj._forward(silu_gating(gate, feat))
    return x + self.ls_ffn(ffn_out)
```

The dispatch in `__call__` for L=1:
```python
if (L == 1 and state is not None
        and self._compiled_pre is not None
        and state.get("k") is not None):
    q, k_new, v_new = self._compiled_pre(x)
    # KV concat — NOT compilable (shape changes each step)
    k_full = mx.concatenate([state["k"], k_new], axis=2)
    v_full = mx.concatenate([state["v"], v_new], axis=2)
    # SDPA — NOT compilable (k/v input shapes grow)
    attn = mx.fast.scaled_dot_product_attention(q, k_full, v_full, scale=self.scale)
    # Everything after: compilable (attn output shape is fixed)
    out = self._compiled_post(attn, x)
    return out, {"k": k_full, "v": v_full}
```

The non-compilable section is only **3 Metal kernels** (k concat, v concat, SDPA) per step. Everything else is compiled.

---

#### 4. Model head compiled — `mlx_model/hybrid_model.py`

**What changed**:
```python
def _head_forward(self, x):
    """norm + head projection + scaled_tanh — all fixed shapes for decode."""
    h = self.norm(x)
    logits = self.head(h * self.inv_sqrt_d).astype(mx.float32)
    return scaled_tanh(logits, 30.0)
```

Compiled in `precompute()` and dispatched in `__call__` when `x.shape[1] == 1`.

The norm + head matmul + scaled_tanh are 4–5 separate Metal kernels that fuse into 1–2 in the compiled program.

---

### Calling sequence (in Mamba3LanguageModel.precompute)

```python
for layer in self.backbone.layers:
    if isinstance(layer, Mamba3Block):
        layer.x_up_proj.precompute_G_experts()  # G cache + TuckerMoE compile
        layer.out_proj.precompute_G_experts()
        layer.precompute()                       # _D_expand + compiled decode
    elif isinstance(layer, TransformerBlock):
        layer.ffn.gate_proj.precompute_G_experts()
        layer.ffn.up_proj.precompute_G_experts()
        layer.ffn.down_proj.precompute_G_experts()
        layer.precompute()                       # compiled pre/post

self._compiled_head = mx.compile(self._head_forward)
mx.eval(self._compiled_head(mx.zeros((1, 1, d_model), dtype=mx.bfloat16)))
```

Order matters: TuckerMoE caches must exist before `Mamba3Block.precompute()` compiles `_decode_impl` (which accesses `_G_experts_cache_bf16` during tracing).

---

### What still doesn't compile

| Component | Reason | Impact |
|-----------|--------|--------|
| Full model `__call__` | KV cache grow → re-trace every step | Would be 1.57× slower |
| TransformerBlock KV concat | `mx.concatenate` shapes change | 2 kernels per block |
| TransformerBlock SDPA | k/v shapes grow | 1 kernel per block |
| Prefill (L>1) | Uses different code paths; per-block `mx.eval` for OOM prevention | Prefill already fast enough |

---

## ✅ SJD WITH PRECOMPUTE (2026-06-06) — 56 → 90.6 tok/s via model.precompute() in SJD path

> Added `model.precompute()` call to `speculative/benchmark_steps.py` and `speculative/run_jacobi.py`  
> after `mx.eval(model.parameters())`. No changes to model code — pure inference-script fix.

### What changed

`model.precompute()` was already called in the AR decode scripts but **not** in the SJD scripts.
Adding it activates two key benefits for the SJD verify path:

1. **TuckerMoE compiled dispatch** — `_compiled_call = mx.compile(self._forward)` is set,
   so every `TuckerMoE.__call__` during the verify forward (L=K) uses the compiled graph.
2. **bf16 G caches** — `_G_experts_cache_bf16` / `_G_flat_cache_bf16` are pre-cast at load time;
   previously each verify call did `G_experts = self._G_experts_cache.astype(dtype)` on every call.

For SJD verify with K=12 (B_flat=12 < top_k×E=16): the decode path is used, so `_G_experts_cache_bf16`
is accessed directly. For K=16: 16 > 16 is False → still decode path.

### Results (mode=self_awareness, prompt="Who are you?", max=256, seed=42)

| K | Variant | ARL | decode_tps | vs old |
|---|---------|-----|------------|--------|
| 12 | + cot + runtime + compile ★ | 4.67 | **90.6** | +62% vs 56 tok/s |
| 16 | + cot + runtime + compile ★ | 5.02 | 86.3 | |
| 8  | + cot + runtime + compile ★ | 1.41 | 35.2 | |
| 24 | + cot + runtime + compile ★ | 1.54 | 34.4 | |

**K=12 is optimal at 90.6 tok/s** (vs 70 tok/s compiled AR, +29% from SJD itself).

### Full K=12 ladder

| Variant | ARL | decode_tps |
|---------|-----|------------|
| baseline (cold, no caches, no compile) | 1.03 | 20.7 |
| + cot_caches | 1.43 | 28.6 |
| + cot + runtime | 3.23 | 63.2 |
| **+ cot + runtime + compile ★** | **4.67** | **90.6** |

### Why `compile_verify` is essential

Each verify round calls `mamba_verify_step` for 24 Mamba layers. Without `compile_verify`, each layer
runs ~50 Metal kernels → ~1200 dispatches/round. `forward_compiled.py` wraps each Mamba layer's
verify step in `mx.compile`, collapsing these to ~5–8 dispatches/layer. At 57 rounds × 24 layers,
this is the dominant term.

---

---

## 0. Current Architecture Summary

```
Model: d_model=768, d_state=64, d_head=64, n_heads=24, n_groups=1
       6 macro-layers × (4 Mamba3Block + 1 TransformerBlock) = 30 blocks
       chunk_size=64, MoE: 8 experts, top_k=2, Tucker: r1=32, r2=512, r3=256
```

### 0.1 Decode (L=1) Critical Path — per step

The `L==1` fast path in `Mamba3Block.__call__` (line 225-231) **skips** the chunk scan entirely. For decode, compute is dominated by **matmuls + MoE**:

| Operation | Count/step | Shape | FLOPs (per) |
|-----------|-----------|-------|-------------|
| `in_proj` Linear | 24 | 768×~4200 | 3.2M |
| `mamba_dense_proj` Linear | 24 | 1536×768 | 2.4M |
| `y_down_proj` Linear | 24 | 256×64 | 33K |
| `x_up_proj` TuckerMoE | 24 | dim_in=1536, dim_out=6144 | ~1.5M |
| `out_proj` TuckerMoE | 24 | dim_in=768, dim_out=768 | ~0.4M |
| `Transformer QKV+O` Linear | 6 | 4×(768×…) | ~10M total |
| `Transformer FFN` 3× TuckerMoE | 6 | gate/up:768→4608, down:4608→768 | ~4M total |
| RMSNorm | ~120 calls | 768-dim | ~90K |
| `gate * silu(z)` pre gate | 24 | 1536-dim | ~3K |

**Total per step ≈ 24×(3.2+2.4+0.03+1.5+0.4) + 6×(10+4) + overhead ≈ 178M + 84M ≈ 262M FLOPs**
**Total TuckerMoE calls per step: 24(x_up) + 24(out) + 6×3(FFN) = 66 instances, each launching 8-10 kernels → ~528-660 MoE kernel launches alone**

At 80 tok/s → 12.5 ms/step → ~20 GFLOPS required (GPU is 20+ TFLOPS). The bottleneck is NOT raw FLOPs but **kernel launch overhead, memory bandwidth for tiny matmuls (batch=1,seq=1), and lazy graph dispatch**. The 120 RMSNorm calls alone launch ~120 Metal kernels per step. **MoE kernel launch overhead alone may consume >50% of the decode time budget.**

### 0.2 Prefill (L=2048) Critical Path

The `L>=2` branch runs `_chunk_parallel_scan` for each of the 24 Mamba blocks.

| Tensor | Shape | Memory |
|--------|-------|--------|
| `la_c` (per chunk) | (1, 32, 64, 24) | 49K |
| `log_M` broadcast | (1, 32, 64, 64, 24) | **3.1M** |
| `M = exp(log_M)` | (1, 32, 64, 64, 24) | **3.1M** |
| `h_intra` einsum out | (1, 32, 64, 24, 64, 64) | **201M** |
| `y_diag` einsum out | (1, 32, 64, 24, 64, 4) | 12.6M |
| `y_off` einsum out | (1, 32, 64, 24, 64, 4) | 12.6M |

**Per prefill (24 blocks): ~25B FLOPs in h_intra einsum alone.** With Lc=64, the 64×64 lower-triangular matrix per chunk creates a `4096×H` intermediate per chunk (32 chunks).

---

## 1. Fusion Targets — Ranked by Impact

### Tier A (Highest Impact — Structural Rewrite)

#### A1. Fused Chunk Scan Metal Kernel (`_chunk_parallel_scan`)

**Files**: `mamba_block.py:86-148`, `forward.py:36-114`

**Current**: 9 separate MLX ops creating 5 intermediate tensors ≥1M elements each:
1. `la_cum = cumsum(la_c, axis=2)`
2. `log_M = broadcast_sub(lacum[:,None] - lacum[None,:])` — (B,nc,Lc,Lc,H) temp
3. `log_M = where(tri_mask, log_M, -1e9)` — in-place modification
4. `M = exp(log_M)` — elementwise
5. `h_intra = einsum("bcijh,bcjhnp->bcihnp", M, u_c)` — 6D contraction

**Proposed**: Single Metal kernel `fused_chunk_scan` that takes `(la_c, u_c, dt, C_c, h_init)` and returns `(y, h_final)` without materializing `log_M` or `M`.

**Metal kernel design**:
```metal
// Dispatch: per (chunk=c, head=h, scan-pos=i), thread-group covers Lc
kernel void fused_chunk_scan(
    device const float* la_c,    // (B, nc, Lc, H) — cumsum already computed
    device const half*  u_c,     // (B, nc, Lc, H, N, P)
    device const half*  C_c,     // (B, nc, Lc, H, N, R)
    device half*        y_out,   // (B, L, H, P, R)
    device half*        h_final  // (B, nc, H, N, P)
) {
    // Each threadgroup handles one (chunk, head)
    // 1. Compute la_cum difference → store in threadgroup memory
    // 2. exp difference → fuse with mask (lower-tri only)
    // 3. Accumulate h_intra via collaborative reduction over Lc
    // 4. Produce y_diag via second contraction over N
    // 5. Cross-chunk carry: h_final = h_inter * decay + h_intra[-1]
    // 6. y_out = y_diag + y_off
}
```

**Benefits**:
- Eliminates `log_M` (3.1M elements / block) and `M` (3.1M elements / block) intermediate allocations
- Eliminates 3 kernel launches (cumsum, where+exp, broadcast_sub) → 1
- Avoids the `nc` Python for-loop (32 iterations × 24 blocks = 768 loop iterations)
- Threadgroup memory keeps `la_cum` differences in fast SRAM during scan

**Equivalence risk**: `exp` may differ by 1-2 ULPs vs MLX's SIMD-group exp. Acceptable if all outputs match within float32 ULP tolerance (1e-6 relative).

#### A2. Inline `la_cum` compute into scan

**Current**: `la_cum = cumsum(la_c, axis=2)` is a separate kernel.

**Proposed**: Compute cumulative sums in-thread using shared memory prefix sum, fused into A1 kernel. For Lc=64, warp-level prefix sum with Metal SIMD groups is trivial.

---

### Tier B (High Impact — Decode Bottleneck)

#### B1. Fused RMSNorm → Linear Chain

**Files**: `mamba_block.py:163-164`, `transformer_block.py:49-52`, `hybrid_model.py:45-46`

**Current**: Each `RMSNorm` → `nn.Linear` pair launches 2 kernels:
1. `RMSNorm`: `rsqrt(mean(x²)) * x * weight` — 1 dispatch
2. `Linear`: `x @ W + b` — 1 dispatch (or 2 with bias)

**Pattern count**: ~120 norm → linear pairs per decode step.

**Proposed**: Single fused Metal kernel `norm_linear_fwd`:
```metal
kernel void norm_linear_fwd(
    device const float* x,      // (B, L, D)
    device const float* w,      // (D,)  RMSNorm weight
    device const half*  W,      // (D, outD) Linear weight
    device const half*  b,      // (outD,) optional bias
    device half*        out     // (B, L, outD)
) {
    // Threadgroup computes RMS of x for current position
    // Applies RMSNorm scalars
    // Feeds directly into matmul accumulator
    // Single kernel, no intermediate norm write
}
```

**Benefits**:
- Cuts ~120 kernel launches → ~60 fused kernels per decode step
- Eliminates intermediate RMSNorm-result memory (768 × bf16 = 1.5KB per call, but 120× = 180KB)
- Reduces GPU command-buffer pressure

#### B2. Fused `_split_inproj` + `_prepare_BC` + `apply_rope`

**File**: `mamba_block.py:63-84, 189-192`

**Current**: After `in_proj` Linear outputs a flat `(B,L, in_out)` tensor, 7 slicing ops create views, then:
1. `_prepare_BC`: RMSNorm + reshape + bias_add + broadcast → 4 ops on B_param, 4 on C_param
2. `apply_rope(B_p, angles)` and `apply_rope(C_p, angles)` — 2 einsums

**Proposed**: Post-`in_proj` kernel that reads the flat output and produces `B_rot, C_rot` directly:
- Read B_param slice → RMSNorm → add bias → broadcast → apply RoPE in one pass
- Same for C_param
- The θ sin/cos precomputed once, shared between B and C

**Benefits**: 10 kernel launches → 2 (B_rot, C_rot), eliminates ~4 intermediate tensors.

---

### Tier C (Medium Impact)

#### C1. MoE `fancy_gather` Fusion → **See dedicated Section 1bis for full treatment**

**File**: `tucker_moe.py:54-63`

> The brief summary below is superseded by the comprehensive **Section 1bis (TuckerMoE Specialized Optimization)**,
> which covers 7 optimizations (T0-T6), full Metal kernel designs, per-component verification,
> and a phase-by-phase implementation roadmap for all 66 MoE instances.
>
> Quick reference: T0 (precompute G_experts) is the highest-impact single change (zero-risk, eliminates ~200 wasted kernels).
> T4 (single-kernel decode MoE) is the decode game-changer (5-10× per MoE call).

**Current**:
1. `G_experts = einsum("er,rst->est", U_expert, core)` — precomputed, OK
2. `per_expert = einsum("br,bkrs->bks", x_shared, G_selected)` — (B, r3) × (B, k, r3, r2) → (B, k, r2)
3. `weighted = per_expert * top_k_probs[..., None]`
4. `x_core = sum(weighted, axis=1)` — reduce over k

**Proposed**: Fuse steps 2-4 into single kernel:
```metal
kernel void moe_weighted_reduce(
    device const half* x_shared,      // (B, r3)
    device const half* G_selected,    // (B, k, r3, r2)
    device const float* top_k_probs,  // (B, k)
    device half* x_core               // (B, r2)
) {
    // Per output element (batch, r2):
    // sum over k: top_k_probs[b,k] * sum_over_r3: x_shared[b,r3] * G_selected[b,k,r3,r2]
}
```

**Benefits**: Eliminates 2 intermediate tensors perMoE call × 42 MoE calls/step.

#### C2. `silu_gating` + down_proj Fusion in Transformer FFN

**File**: `transformer_block.py:88` → `ops.py:17-18` → `tucker_moe.py`

**Current**: `silu(gate) * feat` produces intermediate, then feeds into third MoE.

**Proposed**: Fuse `silu_gating` with the subsequent `down_proj` TuckerMoE's input projection. The norm + U_in matmul of down_proj can start while gating is computed.

#### C3. Fused `lv/dv/av` blending

**File**: `mamba_block.py:201-221`

**Current**: 3 reshape+astype+expand operations → 6 ops, 3 temp tensors.

**Proposed**: Single kernel producing `u_ssm` from `(lam, dt_b, A_b, input_signal, prev_input_signal)`.
```metal
kernel void ssm_input_blend(
    // computes: lv*dv*input + (1-lv)*dv*av*ip
    // directly producing u_ssm
)
```

---

### Tier D (Quick Wins — Low Risk)

#### D1. Precompute + Cache MoE Router Weights

**File**: `tucker_moe.py:42-43`

The router `nn.Linear(dim_in, 8)` is small but called 42 times/decode step. At L=1, the router matmul is batched 1×... which is the worst case for Metal BLAS.

**Proposed**: Convert small Linear(768→8) matmuls to explicit Metal kernel using SIMD-group reductions, or fuse router + top_k selection.

#### D2. RMSNorm Batch Fusion

**Files**: `ops.py:31-34`, `mamba_block.py:163`, `transformer_block.py:49,87`

**Current**: Each RMSNorm is a separate kernel. ~120 calls/decode step.

**Proposed**: "Super-RMSNorm" kernel that norms multiple tensors at once (one grid launch):
```metal
// Takes array of (ptr, dim) pairs, norms them all in one kernel grid
// Reduces dispatch count from 120 → ~8-10 grouped calls
```

#### D3. KV Cache `repeat_kv` Fusion

**File**: `transformer_block.py:63-65`

**Current**: 2 `mx.repeat` calls + concatenation for k and v.

**Proposed**: Fuse repeat + concatenation into the attention kernel or into a single pre-attention kernel. At decode L=1, the cost is trivially small but the launch overhead matters.

#### D4. `cumsum(delta_angle)` + `+ prev_cum` Fusion

**File**: `mamba_block.py:182-185`

**Current**: `cumsum` + `add` → 2 kernels, one (B, L, H, N//2) temp.

**Proposed**: Single fused cumsum-add kernel that takes `delta_angle` and optional `prev_cum`.

---

## 1bis. TuckerMoE Specialized Optimization

> **Why a dedicated section.** TuckerMoE accounts for **66 instances** in the model
> (24 `x_up_proj` + 24 `out_proj` in Mamba blocks + 6×3 `gate/up/down` in Transformer FFNs).
> Each `__call__` launches **8-10 MLX kernels**. At decode (B_flat=1) the actual compute
> per call is < 2 MFLOPs, but kernel launch overhead is ~50-200 µs/call.
> **The MoE dispatch time dominates decode latency more than the SSM scan.**

### 1bis.1 Anatomy of a TuckerMoE Call

**File**: `tucker_moe.py:37-67` — full trace:

```
INPUT: x with orig_shape (B, L, dim_in)
  ↓ flatten → x_flat (B_flat, dim_in) where B_flat = B × L
  ↓
┌─ STEP 1: ROUTER ─────────────────────────────────────────────┐
│ raw_logits  = x_flat @ router.weight.T    # Linear(dim_in, 8)  │
│ capped      = scaled_tanh(raw_logits, 10) # tanh_approx * 10   │
│ router_lgt  = capped / 0.5                # div by temperature  │
│ router_prob = softmax(router_lgt, -1)     # (B_flat, 8)        │
│ top_k_idx   = argpartition(-router_lgt, kth=1)[..., :2]        │
│ top_k_raw   = take_along_axis(router_prob, top_k_idx)          │
│ top_k_probs = top_k_raw / (sum(top_k_raw) + 1e-6)  # normalize│
└───────────────────────────────────────────────────────────────┘
  ↓
┌─ STEP 2: SHARED PROJECTION ──────────────────────────────────┐
│ x_shared = matmul(x_flat, U_in)       # (B_flat, 256)         │
│ x_shared = inner_norm(x_shared)       # RMSNorm(256)          │
└───────────────────────────────────────────────────────────────┘
  ↓
┌─ STEP 3: EXPERT COMPUTATION (the "fancy gather") ────────────┐
│ G_experts  = einsum("er,rst->est", U_expert(8,32), core(32,256,512)) │
│              → (8, 256, 512)  ← RECOMPUTED EVERY CALL!        │
│ G_selected = G_experts[top_k_idx]     # (B_flat, 2, 256, 512) │
│ per_expert = einsum("br,bkrs->bks", x_shared, G_selected)     │
│              → (B_flat, 2, 512)                                │
│ weighted   = per_expert * top_k_probs[..., None]               │
│ x_core     = sum(weighted, axis=1)    # (B_flat, 512)          │
└───────────────────────────────────────────────────────────────┘
  ↓
┌─ STEP 4: OUTPUT PROJECTION ──────────────────────────────────┐
│ out = matmul(x_core, U_out) + bias    # (B_flat, dim_out)     │
└───────────────────────────────────────────────────────────────┘
  ↓ reshape → (*orig_shape[:-1], dim_out)
```

### 1bis.2 Instance Inventory & Dimension Matrix

There are **66 TuckerMoE instances** across the model, in 5 dimension variants:

| MoE Type | Location | Count | dim_in | dim_out | U_in shape | U_out shape |
|----------|----------|-------|--------|---------|------------|-------------|
| `x_up_proj` | Mamba3Block | 24 | **1536** | **6144** | (1536,256) | (512,6144) |
| `out_proj` | Mamba3Block | 24 | 768 | 768 | (768,256) | (512,768) |
| `gate_proj` | TF FFN | 6 | 768 | 4608 | (768,256) | (512,4608) |
| `up_proj` | TF FFN | 6 | 768 | 4608 | (768,256) | (512,4608) |
| `down_proj` | TF FFN | 6 | 4608 | 768 | (4608,256) | (512,768) |

**Invariant across all 66 instances**: `num_experts=8, top_k=2, r1=32, r2=512, r3=256`.  
The Tucker core `(r1, r3, r2) = (32, 256, 512)` and U_expert `(8, 32)` shapes never change.  
Each instance has **independent** weights — no sharing across layers.

### 1bis.3 The Waste: Kernel Launch vs. Compute

Per TuckerMoE `__call__`, MLX launches **8-10 separate Metal kernels**:

| Kernel | Operation | Tensor Shape (decode B_flat=1) | Approx FLOPs |
|--------|-----------|-------------------------------|-------------|
| K1 | `router(x_flat)` matmul | (1,768)×(768,8) | 12K |
| K2 | `scaled_tanh + div` | (1,8) | ~50 |
| K3 | `softmax` | (1,8) | ~50 |
| K4 | `argpartition + gather` | (1,8)→(1,2) | ~20 |
| K5 | `matmul(x_flat, U_in)` | (1,768)×(768,256) | 393K |
| K6 | `inner_norm` (RMSNorm) | (1,256) | 512 |
| K7 | `einsum U_expert·core` | (8,32)·(32,256,512) | **262K** ← WASTED |
| K8 | `G_experts[top_k_idx]` gather | (8,256,512)→(1,2,256,512) | 0 (view) |
| K9 | `einsum + weighted + sum` | (1,256)×(1,2,256,512) | 262K |
| K10 | `matmul(x_core, U_out) + bias` | (1,512)×(512,dim_out) | 393K-3.1M |

**Total compute/decode: ~1.3-4.2 MFLOPs per MoE call** (varies by dim_out).  
**Total kernel launches: ~8-10 per MoE call × 66 calls = ~528-660 per decode step**.  

At decode, each kernel launch costs ~5-50 µs (Metal dispatch + grid setup + GPU warp scheduling).  
Assuming ~20 µs avg: 600 × 20 µs = **12 ms just in kernel dispatch overhead**.  
That's nearly the entire 12.5 ms budget for 80 tok/s — before any actual compute!

For **prefill** (B_flat=2048), the matmuls become batch-efficient and K5/K9/K10 dominate, but K1-K4 and K7 are still tiny and wasteful.

### 1bis.4 Optimization T0: Precompute `G_experts` at Init Time

**Problem**: `G_experts = einsum("er,rst->est", U_expert, core)` is computed **every single `__call__`** (line 55-56). Since `U_expert` and `core` are static model weights, the result never changes. This wastes:

- 1 einsum kernel launch + 2 `astype` casts = ~3 kernel launches × 66 calls = **~200 wasted launches per forward pass**
- ~262K FLOPs recomputed per call × 66 = ~17 MFLOPs per forward pass (tiny compute, but the *dispatch* hurts)

**Fix**: Precompute once when weights are loaded. Add `G_experts` as a persistent buffer:

```python
class TuckerMoE(nn.Module):
    def __init__(self, ...):
        ...
        self.G_experts = None  # precomputed on first forward or explicit init
    
    def _precompute_experts(self):
        if self.G_experts is not None:
            return
        G = mx.einsum("er,rst->est",
                      self.U_expert.astype(mx.float32),
                      self.core.astype(mx.float32))
        self.G_experts = G.astype(self.U_expert.dtype)  # (8, 256, 512)
        mx.eval(self.G_experts)  # materialize once

    def __call__(self, x, temperature=0.5):
        self._precompute_experts()  # no-op after first call
        ...
        G_experts = self.G_experts  # just read, no compute
        ...
```

**Memory cost**: Each `G_experts` is `(8, 256, 512)` in bf16 = **2 MB**. 66 instances × 2 MB = **132 MB** extra. On M4 with 32-64 GB unified memory, this is negligible (< 0.5%).

**Also precompute**: `G_experts` could optionally be pre-transposed for the common contraction pattern.  
The current einsum `"br,bkrs->bks"` contracts over `r` (r3=256) between x_shared (B,r3) and G_selected (B,k,r3,r2).  
No extra transposition needed — the current layout `(k, r3, r2)` is already correct for batched matmul.

### 1bis.5 Optimization T1: Fused Router Chain Kernel

**Current K1-K4**: 4-5 kernels for a **(B_flat, 8)** tensor. The router chain is:
```
Linear(768,8) → tanh_approx → /temp → softmax → argpartition → gather → normalize
```

All of these operate on vectors of size 8 (for decode, just 1 vector). The compute is trivial (~12K FLOPs), but launch overhead is massive.

**Proposed**: Single Metal kernel `tucker_moe_router` that does the entire chain:

```metal
kernel void tucker_moe_router_decode(
    // B_flat = 1, single-token fast path
    device const half*     x,              // (dim_in,) — single token
    device const half*     router_weight,  // (dim_in, 8)
    constant float&        temperature,
    device float*          top_k_probs,    // (2,)
    device int*            top_k_idx,      // (2,)
    device half*           x_shared,       // (256,) pre-norm output
    device const half*     U_in,           // (dim_in, 256)
    device const float*    norm_weight,    // (256,) inner RMSNorm weight
    constant float&        norm_eps
) {
    // ── Part 1: Router ──────────────────────────────────
    // Each of 8 threads computes one logit: x · router_weight[:,e]
    threadgroup float router_logits[8];
    uint e = thread_position_in_threadgroup;  // 0..7
    float dot = 0.0f;
    for (uint d = 0; d < dim_in; d++) {
        dot += float(x[d]) * float(router_weight[d * 8 + e]);
    }
    router_logits[e] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ── Part 2: Scaled tanh ────────────────────────────
    float capped = tanh_approx(router_logits[e] * 0.1f) * 10.0f;
    router_logits[e] = capped / temperature;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ── Part 3: Softmax ────────────────────────────────
    float max_val = router_logits[0];
    for (uint i = 1; i < 8; i++) max_val = fmax(max_val, router_logits[i]);
    float sum_exp = 0.0f;
    float probs[8];
    for (uint i = 0; i < 8; i++) {
        probs[i] = exp(router_logits[i] - max_val);
        sum_exp += probs[i];
    }
    for (uint i = 0; i < 8; i++) probs[i] /= sum_exp;
    
    // ── Part 4: Top-2 selection (8-element sorting network) ──
    // Bitonic sort or pairwise comparison — 8 is tiny
    // ... find top-2 indices and values ...
    // top_k_idx[0,1], top_k_probs[0,1] (normalized)
    
    // ── Part 5: Shared projection (parallel with router) ──
    // x_shared = RMSNorm(x @ U_in)
    // Each thread computes a portion of the (256,) output
    for (uint r3 = e; r3 < 256; r3 += 8) {
        float acc = 0.0f;
        for (uint d = 0; d < dim_in; d++) {
            acc += float(x[d]) * float(U_in[d * 256 + r3]);
        }
        // RMSNorm per-element
        // ... (need group reduction for RMS)
        x_shared[r3] = half(acc * rms_scale * norm_weight[r3]);
    }
}
```

**Dispatch**: 8-thread threadgroup, 1 threadgroup total for B_flat=1 decode.  
For prefill (B_flat > 1): grid of B_flat threadgroups.

**Kernel launches reduced**: 6 → 1.

### 1bis.6 Optimization T2: Fused Shared Projection + Norm

**Current K5-K6**: `matmul(x_flat, U_in)` → `RMSNorm(x_shared)` — 2 separate kernels.

**Proposed**: One kernel, either standalone or fused into T1's kernel. For the standalone version:

```metal
kernel void moe_shared_project_norm(
    device const half*  x,           // (B_flat, dim_in)
    device const half*  U_in,        // (dim_in, 256)
    device const float* norm_weight, // (256,)
    constant float&     eps,
    device half*        x_shared     // (B_flat, 256)
) {
    uint b = threadgroup_position_in_grid;   // batch element
    uint r3 = thread_position_in_threadgroup; // 0..255
    
    // Compute dot product for this (batch, r3) element
    float acc = 0.0f;
    for (uint d = 0; d < dim_in; d++) {
        acc += float(x[b * dim_in + d]) * float(U_in[d * 256 + r3]);
    }
    
    // RMSNorm reduction across r3 dimension
    threadgroup float sq_sum = 0.0f;
    // ... reduction ...
    float rms = rsqrt(sq_sum / 256.0f + eps);
    
    x_shared[b * 256 + r3] = half(acc * rms * norm_weight[r3]);
}

// Alternatively, use MLX's existing batched matmul (good for prefill B_flat=2048)
// and only fuse the norm part. For decode (B_flat=1), this full fusion wins.
```

**Kernel launches reduced**: 2 → 1.

**Note**: For **prefill** (B_flat=2048), the `matmul(x_flat, U_in)` is already a batched BLAS call that MLX handles efficiently. Fusing the norm is still beneficial (saves 1 kernel launch + intermediate), but less critical. Provide both a "fast decode" path (full fusion) and a "batched prefill" path (matmul + fused norm only).

### 1bis.7 Optimization T3: Fused Expert Gather + Weighted Reduce

**Current K7-K9** (after T0 removes K7's G_experts compute):
```
G_selected = G_experts[top_k_idx]                     // K8: fancy-index gather
per_expert = einsum("br,bkrs->bks", x_shared, G_selected) // K9a: batched MV
weighted   = per_expert * top_k_probs[..., None]       // K9b: broadcast mul
x_core     = sum(weighted, axis=1)                     // K9c: reduce
```

For decode (B_flat=1), `G_selected` shape is `(1, 2, 256, 512)` and `x_shared` is `(1, 256)`. The einsum "br,bkrs->bks" is 2 independent vector-matrix products (256×512 each). The computation is:
```
for k in 0,1:
    x_core[k][t] = sum_{s} x_shared[s] * G_selected[k][s][t]
weighted = w_0 * x_core[0] + w_1 * x_core[1]  # (512,)
```

This fits in a single kernel with 512 threads, each computing one output element:

```metal
kernel void moe_expert_reduce_decode(
    device const half*  x_shared,       // (1, 256)
    device const half*  G_experts_all,  // (8, 256, 512) — precomputed
    device const int*   top_k_idx,      // (1, 2)
    device const float* top_k_probs,    // (1, 2)
    device half*        x_core          // (1, 512)
) {
    uint t = thread_position_in_grid;  // 0..511
    if (t >= 512) return;
    
    float result = 0.0f;
    for (int k = 0; k < 2; k++) {
        int expert_id = top_k_idx[k];
        float weight  = top_k_probs[k];
        float dot = 0.0f;
        for (int s = 0; s < 256; s++) {
            dot += float(x_shared[s]) * 
                   float(G_experts_all[(expert_id * 256 + s) * 512 + t]);
        }
        result += weight * dot;
    }
    x_core[t] = half(result);
}
```

**Dispatch**: 512 threads, 1 threadgroup. ~50 µs including launch (vs. ~150 µs for K8+K9a+K9b+K9c separately).

For **prefill** (B_flat=2048), use `mx.einsum` directly — the batch dimension makes the BLAS call efficient. Only fuse the `weighted + sum` part:

```python
# Prefill path (B_flat large)
per_expert = mx.einsum("br,bkrs->bks", x_shared, G_selected)  # efficient BLAS
x_core = moe_weighted_sum_kernel(per_expert, top_k_probs)     # fused kernel
```

**Kernel launches reduced for decode**: 4 → 1.

### 1bis.8 Optimization T4: Single-Kernel Decode MoE (The Holy Grail)

For **B_flat=1 (decode)**, fuse the **ENTIRE TuckerMoE** into one Metal kernel:

```
tucker_moe_decode_kernel(
    x (1, dim_in),
    router_weight (dim_in, 8),
    U_in (dim_in, 256),
    norm_weight (256,),
    G_experts_precomputed (8, 256, 512),  ← from T0
    U_out (512, dim_out),
    bias (dim_out,),
)
→ out (1, dim_out), top_k_probs (2,), top_k_idx (2,)
```

**Thread configuration**: 
- Router phase: 8 threads compute logits + softmax + top-2
- Shared projection phase: 256 threads compute x_shared (or tile if dim_in is large)
- Expert phase: 512 threads compute x_core
- Output phase: `dim_out` threads compute `x_core @ U_out + bias`

These phases can be sequential within one kernel dispatch (threadgroup barrier between phases) or pipelined. For maximum occupancy, use `dispatchThreadgroups:` with the max of (8, 256, 512, dim_out) threads.

**Expected decode latency per MoE call**: Current ~150-400 µs → **15-30 µs** (single kernel).  
This is a **5-10× improvement** per MoE call.

**Implementation strategy**: Write a family of Metal kernels, parameterized by `dim_in` and `dim_out` via `[[function_constant]]`:

```metal
constant uint MOE_DIM_IN  [[function_constant(0)]];  // 768, 1536, or 4608
constant uint MOE_DIM_OUT [[function_constant(1)]];  // 768, 4608, or 6144

kernel void tucker_moe_decode(...) {
    // All phases fused into one dispatch
    // Phase 1: Router (8-element softmax + top-2)
    // Phase 2: x_shared projection + RMSNorm (via threadgroup reduction)
    // Phase 3: Expert gather + weighted reduce
    // Phase 4: Output projection + bias
}
```

Compile 5 variants (one per dim_in/dim_out pair) or 3 (by dim_in only) with `dim_out` as a runtime parameter. At init time, each TuckerMoE instance selects the correct kernel variant.

### 1bis.9 Optimization T5: FFN Gate+Up MoE Pair Fusion

**File**: `transformer_block.py:21-24`

In `MixtralMoEFeedForward`, `gate_proj(x)` and `up_proj(x)` receive the **same input x**:

```python
def __call__(self, x):
    gate = self.gate_proj(x)    # TuckerMoE(dim_in=768, dim_out=4608)
    feat = self.up_proj(x)      # TuckerMoE(dim_in=768, dim_out=4608)
    return self.down_proj(silu_gating(gate, feat))
```

Both share dim_in=768, dim_out=4608, and the same input. They have *separate* router weights, U_in, U_expert, core, U_out, and bias. But structurally they are identical.

**For decode (B_flat=1)**, fuse gate_proj + up_proj into one dispatch:

```metal
kernel void ffn_gate_up_pair_decode(
    // Input (shared)
    device const half*  x,               // (1, 768)
    // Gate MoE weights
    device const half*  gate_router_w,   // (768, 8)
    device const half*  gate_U_in,       // (768, 256)
    device const half*  gate_G_experts,  // (8, 256, 512) precomputed
    device const half*  gate_U_out,      // (512, 4608)
    // Up MoE weights
    device const half*  up_router_w,     // (768, 8)
    device const half*  up_U_in,         // (768, 256)
    device const half*  up_G_experts,    // (8, 256, 512) precomputed
    device const half*  up_U_out,        // (512, 4608)
    // Norm weights
    device const float* gate_norm_w,     // (256,)
    device const float* up_norm_w,       // (256,)
    device const half*  gate_bias,       // (4608,)
    device const half*  up_bias,         // (4608,)
    // Outputs
    device half*  gate_out,              // (1, 4608)
    device half*  feat_out               // (1, 4608)
) {
    // Phase 1: Dual router (both 8-element softmax + top-2)
    // Phase 2: Dual x_shared projection (each 256-dim)
    // Phase 3: Dual expert gather + reduce (each produces 512-dim)
    // Phase 4: Dual output projection (each 512→4608)
}
```

**Benefits**:
- x is read from memory ONCE (saving 768-element read)
- Combined threadgroup gets better GPU occupancy (larger kernel = more thread-level parallelism)
- Reduces 2×10 kernel launches → 1 kernel for the pair
- Cuts 6 FFN kernel launches (2 gate + 2 up per layer) → 1 kernel per 6 layers

**Total savings per decode step**: 12 kernel launches → 6 for the 6 FFN gate-up pairs.

### 1bis.10 Optimization T6: Top-K via Tiny Sort (in-kernel)

**Current**: `mx.argpartition(-router_logits, kth=1)` on (B_flat, 8). For 8 elements, `argpartition` launches a full GPU kernel with sorting/reduction logic designed for arbitrary sizes, wasting enormous overhead on 8 floats.

**Proposed**: Inside the fused router kernel (T1), use an **8-element Batcher's odd-even sorting network** or **pairwise max tournament**:

```metal
// 8-element top-2 via pairwise max (3 compare-swap rounds)
// Structure: (value, index) pairs
struct Pair { float val; int idx; };
Pair pairs[8];

// Initialize
for (int i = 0; i < 8; i++) {
    pairs[i].val = router_logits[i];
    pairs[i].idx = i;
}

// Bitonic sort 8 elements (6 compare-exchange passes in 3 stages)
#define SWAP(a,b) if(a.val < b.val) { Pair t = a; a = b; b = t; }
SWAP(pairs[0], pairs[4]); SWAP(pairs[1], pairs[5]);
SWAP(pairs[2], pairs[6]); SWAP(pairs[3], pairs[7]);
SWAP(pairs[0], pairs[2]); SWAP(pairs[1], pairs[3]);
SWAP(pairs[4], pairs[6]); SWAP(pairs[5], pairs[7]);
SWAP(pairs[1], pairs[2]); SWAP(pairs[5], pairs[6]);
SWAP(pairs[0], pairs[1]); SWAP(pairs[2], pairs[3]);
SWAP(pairs[4], pairs[5]); SWAP(pairs[6], pairs[7]);
#undef SWAP

// Top 2 are pairs[0] and pairs[1] (descending order)
top_k_idx[0]   = pairs[0].idx;
top_k_idx[1]   = pairs[1].idx;
top_k_probs[0] = pairs[0].val;
top_k_probs[1] = pairs[1].val;
```

This runs entirely in registers within the 8-thread router threadgroup, zero memory access.

### 1bis.11 End-to-End Performance Model

#### Decode (B_flat=1, per-step kernel launch reduction)

| Optimization | Kernels Before | Kernels After | Saved |
|-------------|---------------|---------------|-------|
| T0: Precompute G_experts | 3 × 66 = 198 | 0 (one-time) | 198 |
| T1: Fused router | 5 × 66 = 330 | 1 × 66 = 66 | 264 |
| T2: Fused shared+norm | 2 × 66 = 132 | (fused into T4) | — |
| T3: Fused expert reduce | 3 × 66 = 198 | (fused into T4) | — |
| T4: Single-kernel decode MoE | (replaces T1+T2+T3) | **1 × 66 = 66** | **462** |
| T5: FFN gate-up pair fusion | 2 × 6 = 12 | 1 × 6 = 6 | 6 |
| **Cumulative (all T0+T4+T5)** | **~660 kernel launches** | **~72 kernel launches** | **~588 saved** |

**Latency estimate per MoE call at decode**:
- Before: ~150-400 µs per call × 66 = **9.9-26.4 ms** per decode step
- After T4: ~15-30 µs per call × 66 = **1.0-2.0 ms** per decode step
- **Decode improvement from MoE alone: ~5-10×**

#### Prefill (B_flat=2048, per-MoE optimization)

| Optimization | Effect |
|-------------|--------|
| T0: Precompute G_experts | Eliminates 66 × einsum("er,rst→est") calls, saves ~200 kernel launches |
| T1+M: Router batched (Metal-kernel router for B_flat rows) | Replaces K1-K4 with 1 fused kernel; B_flat-compatible |
| T2+M: Shared projection (keep MLX matmul for batch efficiency, fuse norm) | Saves 1 kernel launch + (B_flat,256) intermediate |
| T3+M: Expert weighted-reduce fused | Saves 2 kernel launches + 2 intermediates |
| Total prefill reduction | ~5-6 kernel launches per MoE → **~330-396 saved** |

### 1bis.12 TuckerMoE Verification Strategy

#### Per-component equivalence (must pass rtol=1e-5, atol=1e-5)

```python
# tests/test_tucker_moe_fused.py

class TestTuckerMoEFusion:
    def setup(self):
        # Load a real TuckerMoE instance with actual weights
        self.moe = load_from_checkpoint(TuckerMoE, "x_up_proj.0")
        self.x_decode = mx.random.normal(shape=(1, 1, 768))  # decode-sized
        self.x_prefill = mx.random.normal(shape=(1, 256, 768))  # prefill batch
    
    # ── T0 ──
    def test_T0_precompute_G_experts(self):
        G_ref = mx.einsum("er,rst->est", 
                          self.moe.U_expert.astype(mx.float32),
                          self.moe.core.astype(mx.float32)).astype(mx.bfloat16)
        self.moe._precompute_experts()
        assert mx.allclose(G_ref, self.moe.G_experts, rtol=1e-7)
    
    # ── T1 ──
    def test_T1_fused_router(self):
        # Reference: original code path
        ref_out = self.moe(x_decode)
        ref_router = ...  # capture intermediate router output
        
        # Fused: new code path
        idx, probs = fused_moe_router(x_decode, self.moe.router.weight, ...)
        
        assert mx.allclose(ref_idx, idx, rtol=0)  # exact index match
        assert mx.allclose(ref_probs, probs, rtol=1e-5)
    
    # ── T3 ──
    def test_T3_fused_expert_reduce(self):
        x_shared = mx.random.normal(shape=(1, 256))
        ref_core = original_expert_reduce(x_shared, G_experts, top_k_idx, top_k_probs)
        fused_core = fused_moe_expert_reduce(x_shared, G_experts, top_k_idx, top_k_probs)
        assert mx.allclose(ref_core, fused_core, rtol=1e-5)
    
    # ── T4 ──
    def test_T4_full_decode_kernel(self):
        ref_out, ref_idx, ref_probs = self.moe(x_decode)  # original
        fused_out, fused_idx, fused_probs = tucker_moe_decode_kernel(
            x_decode.reshape(-1),
            self.moe.router.weight,
            self.moe.U_in,
            self.moe.inner_norm.weight,
            self.moe.G_experts,
            self.moe.U_out,
            self.moe.bias,
        )
        assert mx.allclose(ref_out, fused_out, rtol=1e-5)
        assert mx.allclose(ref_idx, fused_idx, rtol=0)
    
    # ── T5 ──
    def test_T5_ffn_gate_up_pair(self):
        x = mx.random.normal(shape=(1, 1, 768))
        gate_ref = ffn.gate_proj(x)
        feat_ref = ffn.up_proj(x)
        gate_fused, feat_fused = ffn_gate_up_pair_fused(
            x.reshape(-1), ffn.gate_proj, ffn.up_proj)
        assert mx.allclose(gate_ref, gate_fused, rtol=1e-5)
        assert mx.allclose(feat_ref, feat_fused, rtol=1e-5)

    # ── Integration ──
    def test_T_all_full_generation(self):
        """20 prompts, temp=0.0, ALL tokens match reference"""
        for prompt in self.golden_prompts:
            tokens_ref = generate_reference(prompt)
            tokens_fused = generate_with_fused_moe(prompt)
            assert tokens_ref == tokens_fused
```

#### Golden Reference Workflow for MoE

```
1. Freeze model with original TuckerMoE code
2. Generate 50 prompts: save prompt_ids + generated_tokens + ALL MoE intermediate tensors
3. Apply T0-T5 changes (one at a time)
4. After EACH change, run 50-prompt equivalence test
5. Only proceed when 100% token match is confirmed
```

### 1bis.13 Quick Wins Checklist (Ordered by Effort/Reward)

| # | Optimization | Code Change | Risk | Deploy Gain |
|---|-------------|------------|------|-------------|
| T0 | Precompute G_experts | `tucker_moe.py` +10 lines | **Zero** (pure cache) | Eliminates ~200 wasted kernel launches |
| T3 (prefill) | Fused weighted-reduce | `tucker_moe.py` + Metal `.metal` | Low | Saves 2 kernels × 66 MoE |
| T1 | Fused router kernel | New Metal kernel + wrapper | Medium (softmax in Metal) | Saves 4 kernels × 66 MoE |
| T3 (decode) | Fused expert reduce decode | New Metal kernel | Low | Saves 3 kernels × 66 MoE |
| T4 | Single-kernel decode MoE | New Metal kernel (3 variants) | **High** (complex) | **Biggest single win for decode** |
| T5 | FFN gate-up pair fusion | New Metal kernel | Medium | Saves 6 kernels per step |
| T2 | Fused shared+norm | New Metal kernel or inlined in T4 | Low | Saves 1 kernel × 66 MoE |
| T6 | In-kernel top-2 sort | Inlined in T1/T4 kernel | Low (deterministic) | Zero-cost improvement |

**Recommended order**: T0 → T1 → T3 → T4 → T5 → T2 (T2 gets subsumed into T4).

### 1bis.14 Phase Assignment for TuckerMoE

| Phase | Optimizations | Week | Key Deliverable |
|-------|--------------|------|----------------|
| P2 (Quick Wins) | T0 | Day 2-3 | Precomputed G_experts, verified |
| P3 (Decode Fusion) | T1, T3 | Day 5-7 | Fused router + expert reduce, 50% kernel reduction |
| P5 (Deep Fusion) | T4, T5 | Day 14-17 | Single-kernel decode MoE, gate-up pair |
| P6 (Integration) | T2 (inlined in T4) | Day 18-19 | Clean integration, all tests green |
| P7 (Prefill) | T3 prefill path | Day 21-22 | Batched expert reduce for B_flat=2048 |

---

## 1ter. Non-Invasive Integration: Fused ↔ Original Coexistence

> **Core principle**: Original `.py` files are NEVER rewritten — only augmented with
> minimal dispatch hooks. All new logic lives in new files. The original code path
> remains untouched and callable, enabling **deterministic A/B comparison** at every
> level (per-op, per-block, per-layer, full generation).

### 1ter.1 File Discipline

| Rule | Rationale |
|------|-----------|
| Original files (`tucker_moe.py`, `mamba_block.py`, `ops.py`, etc.) | Add ONLY `_ENABLE_FUSED = True` flags and `if _ENABLE_FUSED: ... else: original_code` guards |
| New files (`fused_*.py`, `metal_kernels/*.metal`) | Contain ALL new logic, Metal wrappers, kernel dispatch |
| Tests (`tests/test_*_equivalence.py`) | Always run BOTH paths, compare outputs, assert equality |
| No import of fused modules | Original files import nothing from fused; fused imports from original (one-way dependency) |

### 1ter.2 Pattern: Feature-Flag Dispatch

Every modified function follows this exact pattern:

```python
# ── In tucker_moe.py (example: only this block is added) ──

# At module top (single line added):
_ENABLE_FUSED_MOE_DECODE = True   # toggle for A/B testing

class TuckerMoE(nn.Module):
    # ... all original __init__ code UNCHANGED ...

    def __call__(self, x, temperature=0.5):
        # ── Dispatch: try fused path, fallback to original ──
        if _ENABLE_FUSED_MOE_DECODE and _is_decode(x):
            return self._call_fused_decode(x, temperature)
        # ── Original code below — 100% preserved, zero changes ──
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        # ... rest of original __call__ ...
```

```python
# ── In tucker_moe.py: new method (never touches original __call__) ──

    def _call_fused_decode(self, x, temperature=0.5):
        """Decode fast path: entire MoE in one Metal kernel.
        Only dispatched when B_flat=1 (single-token decode).
        Fallback to original __call__ for prefill or batch>1."""
        from ..metal_kernels.fused_moe_decode import tucker_moe_decode_kernel
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        
        # Precompute G_experts once (cheap, cached)
        if not hasattr(self, '_G_cache'):
            self._G_cache = mx.einsum("er,rst->est",
                self.U_expert.astype(mx.float32),
                self.core.astype(mx.float32)).astype(x_flat.dtype)
            mx.eval(self._G_cache)
        
        # Dispatch to Metal fused kernel
        out = tucker_moe_decode_kernel(
            x_flat, self.router.weight,
            self.U_in, self.inner_norm.weight, self.inner_norm.eps,
            self._G_cache, self.U_out, self.bias,
            temperature=temperature,
        )
        return out.reshape(*orig_shape[:-1], self.dim_out)
```

### 1ter.3 One-Way Dependency Graph

```
┌─────────────────────────────────────────────────────────┐
│  NEW FILES (depend on originals, never modified by them) │
│                                                         │
│  metal_kernels/tucker_moe_decode.metal  ← Metal shader  │
│  metal_kernels/tucker_moe_router.metal  ← Metal shader  │
│  metal_kernels/fused_chunk_scan.metal   ← Metal shader  │
│  mlx_model/fused_moe_decode.py          ← Python wrapper│
│  mlx_model/fused_moe_router.py          ← Python wrapper│
│  mlx_model/fused_scan.py                ← Python wrapper│
│  mlx_model/fused_norm_linear.py         ← Python wrapper│
│                         │                               │
│                         │ import (one-way)               │
│                         ▼                               │
├─────────────────────────────────────────────────────────┤
│  ORIGINAL FILES (minimal flag additions only)            │
│                                                         │
│  mlx_model/tucker_moe.py     ← +4 lines (_ENABLE + if)  │
│  mlx_model/mamba_block.py    ← +4 lines (_ENABLE + if)  │
│  mlx_model/transformer_block.py ← +4 lines              │
│  mlx_model/ops.py            ← +4 lines (batched norm)  │
│  mlx_model/hybrid_model.py   ← UNCHANGED                │
│  inference/generator.py      ← UNCHANGED                │
│                         │                               │
│                         │ reference data                 │
│                         ▼                               │
├─────────────────────────────────────────────────────────┤
│  TESTS (always run both paths)                           │
│                                                         │
│  tests/test_tucker_moe_equivalence.py                   │
│  tests/test_mamba_block_equivalence.py                  │
│  tests/test_full_equivalence.py                         │
│  tests/golden/   ← frozen reference outputs             │
└─────────────────────────────────────────────────────────┘
```

### 1ter.4 Per-Layer A/B Comparison System

```python
# tests/test_tucker_moe_equivalence.py

class TuckerMoEComparison:
    """For EVERY TuckerMoE instance in the model, test BOTH paths."""
    
    @staticmethod
    def run_comparison(moe_instance, x_decode, x_prefill):
        results = {}
        
        # ── Test 1: Decode path ──
        with _flag_override('_ENABLE_FUSED_MOE_DECODE', False):
            ref_out = moe_instance(x_decode)       # original path
            mx.eval(ref_out)
        with _flag_override('_ENABLE_FUSED_MOE_DECODE', True):
            fused_out = moe_instance(x_decode)     # fused path
            mx.eval(fused_out)
        
        results['decode_match'] = mx.allclose(ref_out, fused_out, rtol=1e-5)
        results['decode_max_diff'] = mx.max(mx.abs(ref_out - fused_out))
        
        # ── Test 2: Prefill path (should still use original even with flag ON) ──
        with _flag_override('_ENABLE_FUSED_MOE_DECODE', True):
            prefill_out = moe_instance(x_prefill)  # batch>1 → falls back to original
        with _flag_override('_ENABLE_FUSED_MOE_DECODE', False):
            prefill_ref = moe_instance(x_prefill)
        
        results['prefill_match'] = mx.allclose(prefill_out, prefill_ref, rtol=1e-5)
        results['prefill_max_diff'] = mx.max(mx.abs(prefill_out - prefill_ref))
        
        return results

    def test_all_66_instances(self):
        """Instantiate full model, iterate every TuckerMoE, compare both paths."""
        model = Mamba3LanguageModel(Mamba3Config())
        load_checkpoint(model, CHECKPOINT_PATH)
        
        x_dec = mx.random.normal(shape=(1, 1, 768))      # decode-sized
        x_pre = mx.random.normal(shape=(1, 128, 768))    # prefill-sized
        
        for layer_idx, block in enumerate(model.backbone.layers):
            for attr_name in ['x_up_proj', 'out_proj', 'gate_proj', 'up_proj', 'down_proj']:
                moe = getattr(block, attr_name, None)
                if moe is None:
                    continue
                result = self.run_comparison(moe, x_dec, x_pre)
                assert result['decode_match'], \
                    f"Layer {layer_idx}.{attr_name} DECODE mismatch: max_diff={result['decode_max_diff']}"
                assert result['prefill_match'], \
                    f"Layer {layer_idx}.{attr_name} PREFILL mismatch: max_diff={result['prefill_max_diff']}"
```

### 1ter.5 Flag Override Utility (for testing)

```python
# tests/flag_utils.py
import contextlib
import importlib

@contextlib.contextmanager
def _flag_override(flag_name: str, value: bool):
    """Temporarily override a module-level _ENABLE_FUSED_* flag.
    Restores original value on exit. Used in all comparison tests."""
    import mamba3_mlx.mlx_model.tucker_moe as tm
    import mamba3_mlx.mlx_model.mamba_block as mb
    import mamba3_mlx.mlx_model.transformer_block as tb
    import mamba3_mlx.mlx_model.ops as op
    
    modules = {'_ENABLE_FUSED_MOE_DECODE': [tm],
               '_ENABLE_FUSED_MOE': [tm],
               '_ENABLE_FUSED_SCAN': [mb],
               '_ENABLE_FUSED_NORM_LINEAR': [mb, tb, op],
               '_ENABLE_FUSED_SSM_BLEND': [mb]}
    
    mods = modules.get(flag_name, [])
    old_values = {mod: getattr(mod, flag_name, None) for mod in mods}
    
    for mod in mods:
        setattr(mod, flag_name, value)
    
    try:
        yield
    finally:
        for mod, old_val in old_values.items():
            if old_val is not None:
                setattr(mod, flag_name, old_val)
```

### 1ter.6 Safe-to-Modify vs. Forbidden Changes

| File | Allowed Changes | Forbidden |
|------|----------------|-----------|
| `tucker_moe.py` | `+_ENABLE_FUSED_MOE_DECODE` flag, `+_call_fused_decode()` method, `+if` guard in `__call__`, `+_is_decode()` helper | Changing any existing line in `__call__` or `__init__` |
| `mamba_block.py` | `+_ENABLE_FUSED_SCAN` flag, `+if` guard before `_chunk_parallel_scan()` | Changing existing scan math, reshape logic, state management |
| `transformer_block.py` | `+_ENABLE_FUSED_ATTN` flag, `+if` guard before attention/FFN | Changing existing attention math, KV cache logic |
| `ops.py` | `+batched_rmsnorm()` function, existing functions unchanged | Changing existing RMSNorm, silu, softplus |
| `hybrid_model.py` | **Nothing** | Everything — this is the orchestrator |
| `inference/generator.py` | **Nothing** | Everything — this is the decode loop |
| `weights.py` | `+_precompute_all_moe_experts()` call during load | Existing weight-loading logic |

### 1ter.7 Verification Protocol: Before Any Commit

```bash
# 1. Test original path still works (all flags OFF)
python -c "
import mamba3_mlx.mlx_model.tucker_moe as tm
tm._ENABLE_FUSED_MOE_DECODE = False
# ... run full generation, assert tokens match golden reference
"

# 2. Test fused path produces identical results (flags ON)
python -c "
import mamba3_mlx.mlx_model.tucker_moe as tm
tm._ENABLE_FUSED_MOE_DECODE = True
# ... run same generation, assert 100% token match
"

# 3. Test flag toggling mid-generation (no state corruption)
python tests/test_flag_toggle.py  # toggles flag between steps, verifies consistency

# 4. All unit tests pass with flags ON
python -m pytest tests/ -v -k "equivalence"

# 5. All unit tests pass with flags OFF
_ENABLE_ALL_FUSED=0 python -m pytest tests/ -v
```

### 1ter.8 Minimal Diff Example: What Actually Changes in `tucker_moe.py`

```diff
# tucker_moe.py — actual diff after T0+T1+T4 integration

 import mlx.core as mx
 import mlx.nn as nn
 from .ops import RMSNorm, scaled_tanh

+# ── Fused MoE feature flag ──────────────────────────────────────
+_ENABLE_FUSED_MOE_DECODE = True
+
+
+def _is_decode(x) -> bool:
+    """Return True if the input is a single-token decode (B_flat=1)."""
+    return x.ndim >= 2 and x.shape[0] * (x.shape[1] if x.ndim > 2 else 1) == 1
+
 
 class TuckerMoE(nn.Module):
     """Vectorised Tucker-decomposed MoE ..."""
@@ -19,6 +27,13 @@ class TuckerMoE(nn.Module):
         self.inner_norm = RMSNorm(r3, eps=eps)
 
     def __call__(self, x, temperature: float = 0.5):
+        # ── Fused decode fast path ────────────────────────────
+        if _ENABLE_FUSED_MOE_DECODE and _is_decode(x):
+            return self._call_fused_decode(x, temperature)
+        # ── Original path (unchanged) ─────────────────────────
         orig_shape = x.shape
         x_flat = x.reshape(-1, orig_shape[-1])
         dtype = x_flat.dtype
@@ -67,3 +82,27 @@ class TuckerMoE(nn.Module):
         out = mx.matmul(x_core, self.U_out.astype(dtype))
         out = out + self.bias.astype(dtype)
         return out.reshape(*orig_shape[:-1], self.dim_out)
+
+    # ── New methods (never touch original logic) ──────────────
+    def _ensure_experts_precomputed(self, dtype):
+        if not hasattr(self, '_G_cache') or self._G_cache is None:
+            G = mx.einsum("er,rst->est",
+                          self.U_expert.astype(mx.float32),
+                          self.core.astype(mx.float32)).astype(dtype)
+            mx.eval(G)
+            self._G_cache = G
+        return self._G_cache
+
+    def _call_fused_decode(self, x, temperature=0.5):
+        from ..metal_kernels.fused_moe_decode import tucker_moe_decode_kernel
+        orig_shape = x.shape
+        x_flat = x.reshape(-1, orig_shape[-1])
+        G = self._ensure_experts_precomputed(x_flat.dtype)
+        out = tucker_moe_decode_kernel(
+            x_flat, self.router.weight,
+            self.U_in, self.inner_norm.weight, self.inner_norm.eps,
+            G, self.U_out, self.bias,
+            temperature=temperature,
+        )
+        return out.reshape(*orig_shape[:-1], self.dim_out)
```

**Total original code touched**: 4 lines added (`_ENABLE` flag + `if` guard) + 2 new methods (~25 lines) = **29 lines net added, zero original lines modified or removed**.

### 1ter.9 Same Pattern for `mamba_block.py`

```diff
+# ── At module top ──
+_ENABLE_FUSED_SCAN = True

 class Mamba3Block(nn.Module):
     def _chunk_parallel_scan(self, u, dt_b, A_b, C_rot, h_init=None):
+        if _ENABLE_FUSED_SCAN:
+            return self._chunk_scan_fused(u, dt_b, A_b, C_rot, h_init)
+        # ── Original scan (100% preserved) ─────────────────
         B, L, H, N, P = u.shape
         ...
```

### 1ter.10 The Ultimate Safety Net: CI Gate

```yaml
# .github/workflows/fused_safety.yml
name: Fused Path Safety
on: [push, pull_request]

jobs:
  equivalence:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4
      - name: Original path golden test
        run: |
          python -c "import mamba3_mlx.mlx_model.tucker_moe as tm; tm._ENABLE_FUSED_MOE_DECODE = False"
          python tests/test_golden_equivalence.py --prompts 50
      - name: Fused path equivalence
        run: |
          python -c "import mamba3_mlx.mlx_model.tucker_moe as tm; tm._ENABLE_FUSED_MOE_DECODE = True"
          python tests/test_golden_equivalence.py --prompts 50 --assert-token-match
      - name: Toggle stress test
        run: python tests/test_flag_toggle_stress.py --steps 500
```

---

## 2. Metal Kernel Implementation Guidelines

### 2.1 Memory Layout & Data Types

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Stored weights | `bfloat16` | Matches checkpoint, 834 MB fits in iGPU memory |
| Compute precision | `float32` accumulate, `float16` input/output | Metal `float16` throughput is 2× fp32, fp32 accum for stability |
| Intermediate temp | No allocation — use threadgroup memory only | Eliminate DRAM roundtrips |
| Threadgroup size | 256-1024 per grid-block (tuned per kernel) | M4 max threadgroup = 1024 |

### 2.2 Existing MLX → Metal Bridge

MLX exposes custom Metal kernels via `mx.fast` submodules and `mx.custom_function` / `mx.eval_custom` → Metal JIT. The approach for each fused kernel:

1. **Write `.metal` shader** with `[[buffer(N)]]` bindings matching MLX array layout
2. **Wrap in Python** using `mx.custom_function` with `inputs=[...], outputs=[...]`
3. **MLX allocates** output buffers; kernel writes into them
4. **Shape validation** — add Python-side assertions before dispatch
5. **Register** as a drop-in replacement function

Example pattern:
```python
import mlx.core as mx

def _chunk_scan_metal(u, dt_b, A_b, C_rot, h_init, chunk_size):
    B, L, H, N, P = u.shape
    R = C_rot.shape[-1]
    # Pre-compute in MLX: reshape + la_cumsum (cheap)
    la_c = mx.cumsum((dt_b * A_b).astype(mx.float32).reshape(B, -1, chunk_size, H), axis=2)
    u_c = u.reshape(B, -1, chunk_size, H, N, P)
    C_c = C_rot.reshape(B, -1, chunk_size, H, N, R)
    
    # Custom Metal kernel for the heavy part
    y_out = mx.zeros((B, L, H, P, R), dtype=u.dtype)
    h_final = mx.zeros((B, H, N, P), dtype=u.dtype)
    
    return mx.custom_function(
        "fused_chunk_scan",
        inputs=[la_c, u_c, C_c, h_init] if h_init else [la_c, u_c, C_c],
        outputs=[y_out, h_final],
        stream=mx.gpu,  # or mx.default_stream()
    )
```

### 2.3 Threadgroup Memory Budget

Metal per-threadgroup max shared memory = `min(32KB, deviceLimit)`. For chunk scan:
- `la_cum` local: (64 × 24) floats = 6KB
- `M_row` buffer: 64 floats = 0.25KB
- Accumulator `h_intra_row`: (64 × 64) = 4K halfs = 8KB
- Total ≈ 14KB → fits comfortably

---

## 3. Implementation Phases

### Phase 1 — Benchmarking Infrastructure (Day 1-2)

**Goal**: Establish reproducible baselines and testing harness.

#### 1.1 Write `benchmark.py`
```python
# Profile: mamba3_mlx/tools/benchmark.py
# - Prefill: 512/1024/2048 prompt tokens, warmup=3, avg of 10 runs
# - Decode: 200-step continuous generation, report avg+tps, P50/P90/P99 latency
# - Per-op: mx.trace_metal_stream to capture per-kernel GPU time
# - Output: JSON report {prefill_tps, decode_tps, per_phase_ms: {...}}
```

#### 1.2 Per-block timing
- Instrument `Mamba3Block.__call__`, `TransformerBlock.__call__`, `TuckerMoE.__call__` with `time.perf_counter()` around `mx.eval()`.
- Identify exact wall-time per op with `mx.synchronize()` fences.

#### 1.3 Golden Reference System + Flag Utility

```python
# tests/flag_utils.py — context manager for A/B comparison
import contextlib

@contextlib.contextmanager
def fused_disabled():
    """Temporarily disable ALL fused paths. Restores on exit."""
    import mamba3_mlx.mlx_model.tucker_moe as tm
    import mamba3_mlx.mlx_model.mamba_block as mb
    import mamba3_mlx.mlx_model.transformer_block as tb
    old = {}
    for mod, flags in [(tm, ['_ENABLE_FUSED_MOE_DECODE']),
                        (mb, ['_ENABLE_FUSED_SCAN']),
                        (tb, ['_ENABLE_FUSED_ATTN'])]:
        for f in flags:
            if hasattr(mod, f):
                old[(mod, f)] = getattr(mod, f)
                setattr(mod, f, False)
    try:
        yield
    finally:
        for (mod, f), val in old.items():
            setattr(mod, f, val)

@contextlib.contextmanager
def fused_enabled():
    """Enable ALL fused paths."""
    # ... mirror of fused_disabled, sets flags to True ...
```

```python
# tests/freeze_golden.py — one-time: generate immutable reference
python tests/freeze_golden.py \
    --checkpoint checkpoints/latest_sft_cot_model.npz \
    --tokenizer cot_dataset/tokenizer.json \
    --prompts data/test_prompts.txt \     # 50 diverse prompts
    --output tests/golden/v1/ \
    --seed 42 --temperature 0.0

# Output per prompt:
#   tests/golden/v1/prompt_000/
#     ├── input_ids.npy          # tokenized prompt
#     ├── output_tokens.npy      # generated token IDs (greedy, temp=0)
#     ├── logits_step_000.npy    # per-decode-step logits (optional, large)
#     ├── prefill_tps.txt
#     └── decode_tps.txt
```

#### 1.4 Reference embeddings (per-op, for Metal kernel validation)
- Run prefill+decode on **one fixed prompt** (seed=42, temp=0.0) with original code and save:
  - All 66 TuckerMoE intermediate outputs (router_logits, x_shared, x_core, final out)
  - All 24 Mamba block `_chunk_parallel_scan` inputs/outputs (u_ssm, y_stack, h_prev)
  - All RMSNorm outputs at each call site
  - Saved as `.npy` under `tests/golden/v1/intermediates/`
- These become the **per-op golden reference** — every fused Metal kernel is validated against these exact values.

#### 1.5 Current baseline
Record on target hardware (M4 Pro/Max):
```
make -C mamba3_mlx PROMPT="..." TEMP=0.0 MAX_TOK=256 COMPILE=1
```
Document exact prefill_tps and decode_tps.

#### 1.6 Verify original path still works
```bash
# Before ANY code change, run full equivalence suite on original code
python tests/freeze_golden.py ...                    # generate golden references
python tests/test_golden_equivalence.py --golden v1  # assert self-consistency
# Expected: 50/50 prompts match (original code vs itself = 100%)
```

---

### Phase 2 — Tier D Quick Wins (Day 2-4)

**Non-invasive check**: After each optimization, run:
```bash
python tests/test_golden_equivalence.py --golden v1  # original path: must pass
python tests/test_golden_equivalence.py --golden v1 --fused  # fused path: must pass
```

#### 2.1 T0: Precompute G_experts in TuckerMoE (Section 1bis.4)
- Add `_ensure_experts_precomputed()` to `tucker_moe.py`. **Zero changes to existing `__call__`**.
- The `_call_fused_decode()` method (new) uses the cache; original `__call__` also benefits by reading cached value instead of recomputing.
- **Validation**: `assert mx.allclose(G_experts_from_cache, G_experts_from_einsum, rtol=1e-7)` for all 66 instances.

#### 2.2 RMSNorm Batch Fusion (D2)
- Add `batched_rmsnorm()` to `ops.py` alongside existing `RMSNorm` class.
- Dispatch: new fused path uses `batched_rmsnorm()`; original `RMSNorm.__call__` unchanged.
- **Validation**: `fused_disabled()` + `fused_enabled()` produce identical outputs for all ~120 call sites.

#### 2.3 cumsum+add fusion for RoPE angles (D4)
- Fuse `mx.cumsum(angles, axis=1) + prev_cum` into single kernel.
- **Validation**: Check `angles_cum_seq` equals reference with both flags.

#### 2.4 MoE `fancy_gather` + weighted reduce (Section 1bis.7, T3)
- Fuse einsum + weight + sum for all 66 MoE calls.
- **Validation**: Per-instance outputs match reference; both decode and prefill paths verified.

#### 2.5 Re-benchmark (Phase 1 baseline → Phase 2)
- Run benchmark with fused_enabled() and fused_disabled().
- Record improvements. Expect +20-35% decode speedup from eliminated kernel launches.

---

### Phase 3 — Tier B Decode Fusion (Day 4-7)

#### 3.1 RMSNorm+Linear fused kernel (B1)
- Implement `norm_linear_fwd.metal`.
- Replace all `norm(x)` → `linear(normed_x)` patterns in `mamba_block.py` and `transformer_block.py`.
- **Equivalence**: Compare linear outputs; expect exact match since norm math is identical (only fused dispatch).
- **Complexity caveat**: Transformer QKV projections may need separate handling (different weight shapes).

#### 3.2 `_prepare_BC` + `apply_rope` fusion (B2)
- Post-`in_proj`, slice B_param/C_param ranges → RMSNorm → bias → broadcast → RoPE in Metal.
- Precompute `sin(angles), cos(angles)` once on CPU/Metal per step.
- **Validation**: Compare B_rot, C_rot against reference.

#### 3.3 LV/DV/AV blend fusion (C3)
- Single kernel computing `u_ssm` from lam, dt_b, A_b, input_signal, prev_input_signal.
- **Validation**: u_ssm matches reference.

#### 3.4 Re-benchmark
- Expected decode speedup: 20-40% from eliminated kernel dispatch overhead.

---

### Phase 4 — Tier A SSM Chunk Scan (Day 7-14)

This is the most complex phase. Implement `fused_chunk_scan` in Metal.

#### 4.1 Python-side scaffold
```python
# mlx_model/fused_scan.py
def chunk_parallel_scan_fused(u, dt_b, A_b, C_rot, h_init=None, chunk_size=64):
    """
    Drop-in replacement for Mamba3Block._chunk_parallel_scan.
    Same inputs, same outputs, same math.
    Uses single Metal kernel internally.
    """
    B, L, H, N, P = u.shape
    R = C_rot.shape[-1]
    Lc = chunk_size
    pad = (Lc - L % Lc) % Lc
    
    # Pad in MLX (trivial reshape + pad, 1 kernel)
    if pad:
        u = mx.pad(u, [(0,0),(0,pad),(0,0),(0,0),(0,0)])
        dt_b = mx.pad(dt_b, [(0,0),(0,pad),(0,0)])
        A_b = mx.pad(A_b, [(0,0),(0,pad),(0,0)])
        C_rot = mx.pad(C_rot, [(0,0),(0,pad),(0,0),(0,0),(0,0)])
        L = L + pad
    
    nc = L // Lc
    
    # la_cum computed in MLX (cheap cumsum, keep it)
    la = (dt_b * A_b).astype(mx.float32)
    u_c = u.reshape(B, nc, Lc, H, N, P)
    la_c = la.reshape(B, nc, Lc, H)
    C_c = C_rot.reshape(B, nc, Lc, H, N, R)
    la_cum = mx.cumsum(la_c, axis=2)
    
    # --- Metal kernel boundary ---
    y_out = mx.zeros((B, L, H, P, R), dtype=u.dtype)
    h_final = mx.zeros((B, H, N, P), dtype=u.dtype)
    
    la_cum, u_c, C_c, h_init_array, y_out, h_final = mx.eval_custom(
        "fused_mamba_chunk_scan",
        inputs=[la_cum, u_c, C_c,
                h_init if h_init is not None else mx.zeros((B, H, N, P), dtype=u.dtype),
                mx.array([B, nc, Lc, H, N, P, R], dtype=mx.int32)],
        outputs=[y_out, h_final],
        stream=mx.gpu,
    )
    
    if pad:
        y_out = y_out[:, :L - pad]
    return y_out, h_final
```

#### 4.2 Metal kernel implementation
Write `fused_mamba_scan.metal` with:

```metal
#include <metal_stdlib>
using namespace metal;

// Threadgroup-specialized chunk scan
// Grid: (B * nc * H) threadgroups, each covering one (chunk, head) pair
// Threadgroup size: Lc=64 threads

constant uint Lc [[function_constant(0)]]; // 64
constant uint N   [[function_constant(1)]]; // 64
constant uint P   [[function_constant(2)]]; // 64
constant uint R   [[function_constant(3)]]; // 4

kernel void fused_mamba_chunk_scan(
    // Inputs
    device const float* la_cum,         // (B, nc, Lc, H) float32
    device const half*  u_c,            // (B, nc, Lc, H, N, P) bf16
    device const half*  C_c,            // (B, nc, Lc, H, N, R) bf16
    device const half*  h_init,         // (B, H, N, P) or zeros
    device const int*   shape_info,     // [B, nc, Lc, H, N, P, R]
    // Outputs
    device half*  y_out,                // (B, L, H, P, R)
    device half*  h_final               // (B, H, N, P)
) {
    uint idx = threadgroup_position_in_grid;
    // Decode idx into (b, c, h)
    uint B_sz = shape_info[0];
    uint nc   = shape_info[1];
    uint H_sz = shape_info[4];
    uint b = idx / (nc * H_sz);
    uint rem = idx % (nc * H_sz);
    uint c = rem / H_sz;
    uint h = rem % H_sz;
    
    uint li = thread_position_in_threadgroup; // 0..Lc-1
    
    // ── Stage 1: Load la_cum for this chunk/head → threadgroup memory ──
    threadgroup float la_tg[64]; // Lc=64
    la_tg[li] = la_cum[((b * nc + c) * Lc + li) * H_sz + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // ── Stage 2: Compute row of M = exp(la_i - la_j) for j <= i ──
    // For each i (this thread), compute partial h_intra[i, :, :]
    // by accumulating over j <= i
    
    // Per-thread accumulator for h_intra[i, n, p]
    float h_acc[N][P]; // 64*64 = 4096 floats per thread = 16KB
    // Note: This is too large for register file. Must tile over N or P.
    // Better: tile over N in 8-element blocks (8*64*4 bytes = 2KB fit in registers)
    
    // For each j <= i:
    //   weight = exp(la_tg[i] - la_tg[j])
    //   for n in tile_range:
    //     for p in 0..P:
    //       h_acc[n][p] += weight * u_val
        
    // ── Restructured approach ──
    // Rather than per-thread LcxNxP accumulators, use threadgroup-wide
    // reduction: all Lc threads collaborate on each output (i, n, p),
    // reducing over j.
    
    threadgroup float M_row[64]; // shared exp row
    // Each thread li contributes weight for j=li position
    float la_i = la_tg[li];
    
    // For each output position (i), all threads j (j<=i) contribute
    // h[i,n,p] += exp(la[i]-la[j]) * u[j,n,p]
    
    // Strategy: Lc threads tile over (N, P) dimensions
    // N_tile = N/N_groups_t, each group handles a slice of N
    
    // Simplified: process one i at a time
    for (uint i = 0; i < Lc; i++) {
        if (li <= i) {
            float weight = exp(la_tg[i] - la_tg[li]);
            // load u[li, :, :] and multiply by weight → accumulate
            for (uint n = 0; n < N; n++) {
                for (uint p = 0; p < P; p++) {
                    uint u_offset = ((((b * nc + c) * Lc + li) * H_sz + h) * N + n) * P + p;
                    float u_val = float(u_c[u_offset]);
                    h_accum[li][n][p] = weight * u_val; // to threadgroup mem
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Reduce h_accum across Lc for this i → h_intra[i,n,p]
        // ...
    }
    
    // This gets very complex with register pressure.
    // Alternative: emit intermediate h_intra as a tensor, use MLX einsum for
    // the final contraction. The true bottleneck is the Lc×Lc matrix materialization.
}
```

#### 4.3 Pragmatic Approach: Two-Kernel Fusion

Given the register pressure of a full single-kernel scan, a more practical fusion:

**Kernel 1: `compute_M_and_hintra`** — fuse la_cum → exp(M) → h_intra einsum
- Grid: `(B * nc * H, Lc)` — each thread computes one output element of h_intra
- Each thread: iterates j=0..i, computes `exp(la_cum[i]-la_cum[j]) * u_c[j,n,p]`, accumulates
- No intermediate M storage needed
- I/O: reads la_cum (6KB/threadgroup), u_c (reads); writes h_intra

**Kernel 2: `compute_y_and_carry`** — fuse y_diag + cross-chunk carry + y_off
- Takes h_intra, C_c, h_init, decay → produces y, h_final
- Eliminates the Python for-loop over chunks (uses Metal threadgroup for inter-chunk state)

**This two-kernel approach** still eliminates the large `log_M` and `M` intermediates while keeping register pressure manageable.

#### 4.4 Validation Protocol
```python
# tests/test_chunk_scan_equivalence.py
def test_fused_scan_matches_reference():
    for L in [1, 2, 7, 64, 128, 512, 1024, 2048]:
        u = mx.random.normal(shape=(1, L, 24, 64, 64))
        dt_b = mx.abs(mx.random.normal(shape=(1, L, 24)))
        A_b = -mx.abs(mx.random.normal(shape=(1, L, 24)))
        C_rot = mx.random.normal(shape=(1, L, 24, 64, 4))
        
        y_ref, h_ref = _chunk_parallel_scan_ref(u, dt_b, A_b, C_rot)
        y_fused, h_fused = chunk_parallel_scan_fused(u, dt_b, A_b, C_rot)
        
        assert mx.allclose(y_ref, y_fused, rtol=1e-5, atol=1e-5)
        assert mx.allclose(h_ref, h_fused, rtol=1e-5, atol=1e-5)
```

#### 4.5 Also apply to `_scan_per_pos` (forward.py)
The same fusion applies to `forward.py:_scan_per_pos` for speculative decoding. The per-position `h_per_pos` output requires an additional broadcast-multiply-add, which can be fused into Kernel 2.

---

### Phase 5 — Tier A Deep Fusion (Day 14-18)

#### 5.1 End-to-end `in_proj → scan → y_down` fused for decode (L=1)

For **decode only** (L=1), the entire Mamba3Block can be a single fused kernel:
```
norm_mamba(x) → in_proj(x) → (split, prepare BC, RoPE, MoE x_up,
  einsum B×x, blend lv/dv/av, single-step recurrence, y_down, D-skip,
  gated dense → 1 kernel)
```

This would be ~500 lines of Metal. Sub-kernels are better:
- `norm + in_proj + split` → fused (Kernel A)
- `prepare_BC + RoPE` → fused (Kernel B)
- `MoE x_up + ssm_input_blend + L=1 step + y_down + gate + dense` → fused (Kernel C)

At decode, L=1 means all tensors are tiny (batch×1×...). The chief benefit is eliminating the 15+ kernel launches per Mamba block.

#### 5.2 MoE router + selector fusion
For all 42 MoE calls/decode step: fuse `router(x) → tanh → softmax → top_k → gather` into one Metal dispatch.

---

### Phase 6 — `mx.compile` Compatibility & Integration (Day 18-20)

#### 6.1 Ensure fused kernels work with `mx.compile`
- `mx.compile` must be able to trace through custom Metal calls.
- Test: `COMPILE=1 make -C mamba3_mlx ...` with fused kernels enabled.
- If `mx.compile` doesn't trace custom functions → wrap with `mx.compile` disabled for fused paths, compile the rest.

#### 6.2 `_ENABLE_FUSED` feature flag
```python
# mlx_model/__init__.py or config
_ENABLE_FUSED_SCAN = True    # toggle for debugging
_ENABLE_FUSED_NORM_LINEAR = True
_ENABLE_FUSED_MOE = True
```
Original code paths preserved for validation and debugging.

#### 6.3 Integration with speculative decoding
- `forward.py:mamba_verify_step` and `forward.py:_scan_per_pos` must also use fused kernels.
- The `shrink_chunk=True` optimization (Lc=L) should work with fused kernels (just adjust `function_constant` for Lc).

---

### Phase 7 — Prefill Optimization (Day 20-24)

Prefill is dominated by the SSM chunk scan (Phase 4 covers this). Additional prefill-specific optimizations:

#### 7.1 Batched Lu×Lc matrix processing
For prefill L=2048, we have nc=32 chunks. Process all 32 chunks on GPU simultaneously:
- Grid: `(B * H, nc)` threadgroups, each handles one (head, chunk) pair.
- All chunks run in parallel, no serial Python loop.

#### 7.2 Attention prefill optimization
- For L=2048, `mx.fast.scaled_dot_product_attention` with causal mask is already efficient.
- No custom kernel needed unless profiling shows it's a bottleneck.

#### 7.3 MoE batching for prefill
- At prefill, each MoE receives (B, L, d_model) = (1, 2048, 768). The router and U_in matmul are now batch-2048, which is well within GPU efficiency range. No fusion needed for this scale.

#### 7.4 Prefill pipeline: overlap compute with data movement
- Phase 1: stream `in_proj` results while computing next block.
- Phase 2: Metal command encoder interleaving — start next block's `in_proj` while current block's scan executes (dependent ops auto-queue via MLX lazy eval).

---

### Phase 8 — Final Validation & Tuning (Day 24-30)

#### 8.1 Full equivalence test suite
```python
# tests/test_full_equivalence.py
def test_generate_equivalence():
    """20 prompts, temp=0.0 → all generated tokens must match reference."""
    prompts = [...20 diverse prompts...]
    for prompt in prompts:
        tokens_ref = generate_reference(prompt)
        tokens_fused = generate_fused(prompt)
        assert tokens_ref == tokens_fused, f"Mismatch on: {prompt}"

def test_prefill_equivalence():
    """All hidden states match within rtol=1e-5 at every layer."""
    ...

def test_decode_step_equivalence():
    """Per-decode-step logits match within rtol=1e-5, all 30 layers' states match."""
    ...
```

#### 8.2 Stress tests
- Maximum context: 2048 tokens prefill + 512 decode tokens → no crash, no NaN, correct outputs.
- Random seeds: 100 generations with different seeds, all complete without errors.
- Memory: 16GB Mac → peak GPU memory < 8GB (current: ~2GB).
- Interrupt safety: Ctrl-C during generation → clean exit.

#### 8.3 Performance regression guard
```makefile
# Makefile target
bench:
    @echo "Running benchmark suite..."
    $(PYTHON) tools/benchmark.py --prompt "Explain quantum computing" \
        --prefill-lens 512,1024,2048 \
        --decode-steps 200 \
        --runs 10 \
        --output-json benchmark_latest.json
    $(PYTHON) tools/benchmark.py --compare benchmark_baseline.json benchmark_latest.json
```

Gate: CI must pass benchmark thresholds. No merge if decode_tps < baseline or prefill_tps < baseline.

#### 8.4 Apple GPU Profiling
- Use Xcode Metal Debugger → GPU Timeline to identify:
  - Kernel occupancy (should be >50% for main kernels)
  - Memory bandwidth utilization
  - Pipeline stall points
- MLX `mx.trace_metal_stream` for per-kernel GPU time breakdown

#### 8.5 Tuning
- `Lc` chunk size: try 32, 48, 64, 96, 128 → find optimal for each prefill length
- Threadgroup size: 64/128/256/512/1024 → profile per kernel
- Register spilling: reduce N/P tile size to fit registers → reduce memory traffic

---

## 4. Verification Standards

### 4.1 Numerical Equivalence Matrix

| Test Level | Metric | Tolerance | Frequency |
|-----------|--------|-----------|-----------|
| Per-op output | `mx.allclose` | rtol=1e-5, atol=1e-5 | Every kernel |
| Per-block state (h_prev, angles_cum) | `mx.allclose` | rtol=1e-4 (fp32 accum→bf16) | Every block |
| Per-layer output | `mx.allclose` | rtol=1e-4 | Every layer |
| Full model logits | `argmax match` | exact match at temp=0.0 | 20 prompts |
| Full model logits | `mx.allclose` | rtol=1e-3 (softmax tolerant) | 20 prompts |
| Generated tokens (greedy) | token-by-token equality | **100% match** | 50 prompts |
| Generated tokens (temp=0.3) | token-by-token equality | 100% match (same seed) | 50 prompts |
| Prefill throughput | `>= 400 tok/s` | 2048-token prompt | CI gate |
| Decode throughput | `>= 80 tok/s` | 200-step decode | CI gate |

### 4.2 Golden Reference Generation

```bash
# Before any fusion work, generate and save golden references
python -m mamba3_mlx.tools.freeze_golden \
    --model checkpoints/latest_sft_cot_model.npz \
    --tokenizer cot_dataset/tokenizer.json \
    --output golden/v1/ \
    --num-prompts 50 \
    --seed 42
```

This saves:
- `golden/v1/prompt_{i}_ids.npy`
- `golden/v1/prompt_{i}_tokens.npy`
- `golden/v1/prompt_{i}_logits.npy`
- `golden/v1/prompt_{i}_layer_states/`

### 4.3 Equivalence Testing Workflow

```
┌─────────────┐     ┌─────────────┐
│ Reference   │     │ Fused       │
│ (original)  │     │ (optimized) │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────┐
│   Compare (per-op, per-layer)   │
│   if ALL pass → phase complete  │
│   if ANY fail → fix kernel      │
└─────────────────────────────────┘
```

### 4.4 CI/CD Pipeline

```yaml
# .github/workflows/perf.yml (or local pre-commit)
steps:
  - Generate golden reference (once)
  - For each PR that modifies mlx_model/:
    1. Run unit tests: python -m pytest tests/mlx_model/ -v
    2. Run equivalence: python tests/verify_equivalence.py --golden golden/v1/
    3. Run benchmark: python tools/benchmark.py --runs 5
    4. Assert: decode_tps >= 80 AND prefill_tps >= 400 AND num_match_rate == 1.0
```

---

## 5. Performance Budget & Milestones

### 5.1 Current Estimate (to be confirmed by Phase 1)

| Metric | Estimated Current | Target |
|--------|-------------------|--------|
| Decode (L=1, compiled) | ~15-25 tok/s | **80+ tok/s** |
| Prefill (L=2048, compiled) | ~80-120 tok/s | **400+ tok/s** |

### 5.2 Phase-wise Expected Gains

| Phase | Decode Improvement | Prefill Improvement | Key MoE Win |
|-------|-------------------|---------------------|-------------|
| P2 (Quick Wins) | +20-35% | +5-10% | T0: G_experts precompute (~200 wasted kernels eliminated) |
| P3 (Decode Fusion) | +30-60% | +10-15% | T1+T3: Fused router + expert reduce, ~400 kernels saved |
| P4 (SSM Scan) | +5% (decode L=1 skips scan) | **+60-100%** | — |
| P5 (Deep Fusion) | **+40-60%** | +5-10% | T4+T5: Single-kernel decode MoE, gate-up pair; ~460 → ~72 total MoE kernels |
| P7 (Prefill batch) | +5% | **+40-60%** | T3 prefill path: batched expert weighted-reduce |
| P8 (Tuning) | +10-25% | +10-15% | Threadgroup sizing, Metal occupancy tuning |
| **Cumulative** | **~4-8×** | **~3-5×** | MoE alone contributes ~2-4× of decode gain |

### 5.3 Milestones

| Milestone | Week | Success Criteria |
|-----------|------|-----------------|
| M1: Baseline | 1 | Benchmark script + golden references (50 prompts) saved; `flag_utils.py` working |
| M2: Quick Wins | 1 | T0 (G_experts precompute) + D2+D4; ALL flags OFF→ON produce identical tokens |
| M2b: MoE Router Fusion | 1-2 | T1 fused router; flag toggle test: 500 random prompts, 100% token match |
| M3: Decode Accelerated | 2 | B1 Norm+Linear fused; decode ≥35 tok/s; original path still passes all tests |
| M4: SSM Scan Fused | 2-3 | Chunk scan Metal kernel; `_ENABLE_FUSED_SCAN=True/False` both produce identical outputs |
| M4b: MoE Deep Fuse | 3 | T4 single-kernel decode MoE; all 66 instances pass per-component A/B comparison |
| M5: Deep Fusion | 3 | T4+T5 complete; decode ≥65 tok/s; `if fused: ... else: original` guard on all modified files |
| M6: Prefill Optimized | 3-4 | Batched scan + MoE prefill path; prefill ≥350 tok/s |
| M7: Targets Hit | 4 | decode ≥80, prefill ≥400; 100% token match on 50 golden prompts with ALL flags ON |
| M8: Production Ready | 4-5 | CI gate passes flags ON + OFF; `git diff` shows ≤30 lines changed in any original file |

---

## 6. File Map — What Gets Modified

```
mamba3_mlx/
├── mlx_model/
│   ├── fused_scan.py          [NEW]     Phase 4: Metal-wrapped chunk scan
│   ├── fused_norm_linear.py   [NEW]     Phase 3: RMSNorm→Linear fusion
│   ├── fused_moe.py           [NEW]     Phase 2-5: MoE gather + router fusion wrappers
│   ├── fused_moe_decode.py    [NEW]     Phase 5: Single-kernel decode MoE (T4, 3 variants)
│   ├── fused_ffn_pair.py      [NEW]     Phase 5: FFN gate+up pair fusion (T5)
│   ├── fused_ssm_decode.py    [NEW]     Phase 5: L=1 decode kernel for Mamba block
│   ├── mamba_block.py         [MODIFY]  Import fused ops, add _ENABLE_FUSED_MOE flag
│   ├── transformer_block.py   [MODIFY]  Use fused norm+linear; FFN pair dispatch
│   ├── tucker_moe.py          [MODIFY]  Add _precompute_experts(); decode fast path
│   ├── ops.py                 [MODIFY]  Add fused RMSNorm batched
│   └── hybrid_model.py        [MINOR]   No structural changes
│
├── metal_kernels/
│   ├── fused_chunk_scan.metal [NEW]     Phase 4: SSM scan GPU kernel
│   ├── fused_norm_linear.metal[NEW]     Phase 3: Norm+Linear kernel
│   ├── tucker_moe_router.metal[NEW]     Phase 2: Fused router chain (T1)
│   ├── tucker_moe_expert_reduce.metal[NEW] Phase 2-3: Expert gather+reduce (T3)
│   ├── tucker_moe_decode.metal[NEW]     Phase 5: Single-kernel decode MoE (T4)
│   ├── ffn_gate_up_pair.metal[NEW]      Phase 5: FFN gate+up pair (T5)
│   ├── rmsnorm_batched.metal  [NEW]     Phase 2: Batched norm kernel
│   └── kernels.metallib       [AUTO]    Compiled Metal library
│
├── tests/
│   ├── test_tucker_moe_equivalence.py [NEW] Phase 2-5: Per-component MoE comparison
│   ├── test_mamba_block_equivalence.py [NEW] Phase 4: Mamba block A/B comparison
│   ├── test_chunk_scan.py     [NEW]     Phase 4: Scan equivalence tests
│   ├── test_full_equivalence.py[NEW]   Phase 8: End-to-end tests
│   ├── test_flag_toggle_stress.py [NEW]  Phase 3+: Flag toggle mid-generation
│   ├── flag_utils.py          [NEW]     Phase 1: _flag_override context manager
│   └── golden/                [NEW]     Reference outputs (50 prompts × tokens + states)
│
├── tools/
│   ├── benchmark.py           [NEW]     Phase 1: Performance measurement
│   ├── freeze_golden.py       [NEW]     Phase 1: Generate references
│   └── bench_cmp.py           [NEW]     Phase 8: Compare benchmarks
│
└── .github/workflows/
    └── perf.yml               [NEW]     CI performance gate
```

---

## 7. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Metal kernel incorrect due to float rounding | Generation diverges | Golden token-by-token test; fp32 accum guaranteed in kernels |
| `mx.compile` incompatible with custom Metal | Loses JIT benefits | Test early; plan B: compile non-fused parts, use fused as precompiled |
| M4-specific Metal features won't work on M1-M3 | Portability loss | Use Metal 3.0 features only; feature-detect at runtime; fallback to MLX ops |
| Too many kernel variants → maintenance burden | Code rot | Parameterize with `[[function_constant]]`; single kernel per math pattern |
| Memory pressure from large intermediates | OOM on 8GB Macs | All fusions eliminate intermediates; monitor with `mx.metal.get_active_memory()` |
| T4 single-kernel MoE incorrect on edge cases (dim_in changes, batch>1) | Silent wrong outputs | Guard: only dispatch fused kernel for decode (B_flat=1 + known dim_in); fallback to MLX path for prefill |
| T5 gate-up pair: router divergence between two MoE instances | gate≠feat when routers disagree | By design: separate routers mean separate outputs. No sharing assumed. Verify per-instance outputs match reference |
| G_experts precompute memory (132 MB) pushes unified memory limit | OOM on 8GB devices | Lazy-precompute: only compute G_experts for instances actually used in this forward pass; add `--low-memory` flag to skip precompute |
| `scaled_tanh` approximation difference (tanh_approx vs Metal tanh) | Router probability differences | Use MLX's `tanh_approx` formula in Metal (2*sigmoid(2x)-1) — same math, not hardware tanh |

---

## 8. Quick Start — Running the First Benchmark

```bash
# 1. Install profiling tools
pip install mlx-metal-trace  # if available

# 2. Run baseline
make -C mamba3_mlx PROMPT="Explain quantum computing in detail" TEMP=0.0 MAX_TOK=200 COMPILE=1

# 3. Run benchmark script (once Phase 1 is done)
python mamba3_mlx/tools/benchmark.py \
    --prompt "Explain quantum computing" \
    --prefill-lens 512,1024,2048 \
    --decode-steps 200 \
    --runs 10 \
    --output benchmark_baseline.json \
    --metal-trace

# 4. After each phase, run benchmark again and compare
python mamba3_mlx/tools/benchmark.py \
    --compare benchmark_baseline.json
```

---

*Plan version: v2.0 — May 2025 (includes comprehensive TuckerMoE section 1bis)*  
*Target hardware: Apple M4 Pro/Max (24-40 core GPU), 32-64 GB unified memory*
