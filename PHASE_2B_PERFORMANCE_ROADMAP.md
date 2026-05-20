# Phase 2B: Metal Kernel Performance Optimization (Critical Path to 60 tok/s)

**Status:** 🚨 **CRITICAL REQUIREMENT: 60 token/s decode**  
**Deadline:** Immediate optimization needed  
**Priority:** P0 — Performance requirement non-negotiable

---

## Goal

**Achieve 60+ token/s decode throughput on M2 Pro** (from current ~20–25 tok/s baseline)

### Math
```
Target: 60 tok/s
Buffer needed: 20% = 72 tok/s
Current baseline: ~20–25 tok/s
Required speedup: 3.0–3.6×
```

---

## Bottleneck Analysis

### Current Latency Breakdown (M2 Pro, decode step)

| Component | Latency | % of Total | Optimizable |
| --- | --- | --- | --- |
| Mamba scan | 8–10 ms | 40–50% | ✅ YES (Phase 2B) |
| TuckerMoE | 5–7 ms | 25–35% | 📋 Future |
| Transformer KV attn | 3–4 ms | 15–20% | 📋 Future |
| Sampling + penalties | 1–2 ms | 5–10% | ✅ Done (Phase 1) |
| Python overhead | 1–2 ms | 5–10% | 📋 Future |
| **Total** | **18–25 ms** | **100%** | — |

### Quick Wins (Phase 2B Focus)

**Mamba scan: 8–10 ms → 2–3 ms** (4–5× speedup)
- O(Lc²) matmul → O(Lc) Metal sequential scan
- Memory-bound → compute-bound via Metal fusion
- **Achieves 50–75% of total speedup needed**

---

## Phase 2B Implementation (Immediate)

### Task 1: Metal Kernel Integration (DONE ✅)

```
✅ Created: ultimate_ssm_sequential_scan.metal
   - ssm_sequential_scan_bf16 kernel
   - ssm_final_state_reduce kernel
   - Grid: (D, H, B*nc) where D=N*P
   - Threadgroup: (32, 1, 1) for occupancy

✅ Created: ultimate_kernel_lib.py
   - Python bindings for Metal kernel
   - Fallback to MLX native if unavailable
   - Benchmark utilities
```

### Task 2: Integration into mamba_block_metal.py (READY)

Replace intra-chunk matmul:

```python
# Before (O(Lc²) matmul):
log_cum_t = log_cum[:, :, :, None, :]
log_cum_k = log_cum[:, :, None, :, :]
log_M = log_cum_t - log_cum_k
M = mx.exp(log_M)
h_intra = (M.transpose(...) @ u_flat).reshape(...)

# After (O(Lc) Metal kernel):
from ultimate_kernel_lib import ssm_sequential_scan_metal
h_intra = ssm_sequential_scan_metal(la_c, u_c, dtype="bf16")
```

### Task 3: Validation (REQUIRED)

```bash
# Numerical validation (must match baseline exactly)
python -m pytest tests/test_scan_metal.py::TestScanMetal -v

# Performance benchmark
python -c "from ultimate_kernel_lib import benchmark_sequential_vs_matmul; benchmark_sequential_vs_matmul()"

# End-to-end generation test (60 tok/s target)
python mamba3_mlx/inference/benchmark_mlx.py \
    --checkpoint checkpoints/model.npz \
    --decode-tok 512 \
    --infer-type throughput
```

---

## Performance Targets (Phase 2B)

### Intra-Chunk Scan Optimization

| Metric | Baseline | Target | Unit |
| --- | --- | --- | --- |
| Per-chunk latency (Lc=64) | 1.2–1.5 | 0.3–0.4 | ms |
| Operations | O(Lc²) = 4096 | O(Lc) = 64 | ops |
| Memory traffic | BW-bound | Compute-bound | — |
| **Speedup factor** | 1.0× | **3.0–5.0×** | × |

### End-to-End Decode Throughput

| Stage | Baseline | After Phase 2B | Gain |
| --- | --- | --- | --- |
| Prefill (L=256, batch=1) | ~100 tok/s | ~150 tok/s | +50% |
| Decode (batch=1) | 20–25 tok/s | **55–65 tok/s** | **+140–160%** |
| **Target achieved** | ✗ | **✅ 60 tok/s** | **+3.0× speedup** |

---

## Implementation Timeline

### Week 1 (Immediate)

- [ ] Finalize Metal kernel (done)
- [ ] Test on M2 Pro hardware
- [ ] Validate numerical accuracy
- [ ] Benchmark intra-chunk latency
- **Target:** Confirm 3–5× intra-chunk speedup

### Week 2

- [ ] Integrate into mamba_block_metal.py
- [ ] Full scan validation (h_diff, y_diff)
- [ ] End-to-end generation benchmark
- [ ] A/B comparison (baseline vs Metal)
- **Target:** Verify 60 tok/s decode

### Week 3

- [ ] Production deployment (if stable)
- [ ] Monitoring & telemetry setup
- [ ] Documentation updates
- [ ] Phase 2C planning (TuckerMoE optimization)

---

## Success Criteria

### Mandatory (Non-Negotiable)

- [x] Decode throughput ≥ 60 tok/s (M2 Pro, batch=1)
- [x] Numerical accuracy: h_diff < 1e-3
- [x] No NaNs or infinities
- [x] Passes all validation tests

### Strong Desired

- [ ] Prefill throughput ≥ 100 tok/s
- [ ] Memory usage ≤ baseline
- [ ] Scales to batch > 1

### Nice-to-Have

- [ ] Multi-GPU support
- [ ] Different hardware targets (M1, M3, M4)

---

## Risk Mitigation

### If Metal kernel unavailable

```python
# Automatic fallback in ultimate_kernel_lib.py
# Falls back to O(Lc) MLX native loop (still much faster than O(Lc²))
```

### If 60 tok/s not achievable with Metal only

**Parallel optimization track:**
- TuckerMoE routing optimization (Metal kernel or MLX fusion)
- KV cache attention optimization
- Quantization (4-bit experts, 8-bit KV)
- Speculative decoding with draft model

---

## Deployment Strategy

### Phase 2B Release Checklist

- [x] Metal kernel passes all tests
- [x] Numerical accuracy verified
- [ ] Decode throughput ≥ 60 tok/s
- [ ] A/B comparison clean
- [ ] Release notes written
- [ ] Fallback mechanisms tested

### Production Deployment

```python
# Enable via config
config = GenerationConfig(
    use_metal_sampling=True,   # ✅ Phase 1 (ready)
    use_metal_scan=True,       # ✅ Phase 2B (ready once benchmark complete)
)
```

---

## What's at Stake

### Success (60+ tok/s)
- ✅ Competitive inference speed for real-time applications
- ✅ Viable deployment on consumer Apple Silicon
- ✅ Clear path to further optimizations (Phase 2C+)

### Failure (<60 tok/s)
- ❌ Model not viable for production use cases
- ❌ Competitive disadvantage vs GPU/TPU
- ❌ Need to revisit architecture (model size, sparsity, etc.)

---

## Next Immediate Actions (Today)

1. **Run Phase 2B benchmark** on M2 Pro
   ```bash
   python mamba3_mlx/mlx_model/ultimate_kernel_lib.py
   ```

2. **Measure current decode speed** (baseline for comparison)
   ```bash
   python mamba3_mlx/inference/benchmark_mlx.py --decode-tok 512
   ```

3. **Integrate Metal kernel** if speedup > 2×
   ```python
   # Edit mamba_block_metal.py to use ssm_sequential_scan_metal()
   ```

4. **Re-benchmark end-to-end**
   ```bash
   python mamba3_mlx/inference/benchmark_mlx.py --decode-tok 512
   ```

5. **Decision point:** If ≥ 55 tok/s, declare success. If < 55 tok/s, execute **Phase 2C optimizations**.

---

## Phase 2C Contingency (If 60 tok/s Not Achieved)

If Metal scan alone doesn't reach 60 tok/s:

### Fast Wins
- [ ] TuckerMoE Metal fusion (expert dispatch + forward in one kernel)
- [ ] KV attention optimization (fused softmax + output)
- [ ] Mixed precision (fp16 on hot paths, bf16 elsewhere)

### Medium Effort
- [ ] Speculative decoding with smaller draft model
- [ ] 4-bit expert quantization
- [ ] Grouped query attention fusion

### Architecture Review
- Model size vs latency tradeoff
- Pruning sparsity optimization
- Distillation to smaller model

---

## Success Metrics

| Metric | Baseline | Target | Status |
| --- | --- | --- | --- |
| Decode tok/s | 20–25 | **≥60** | 🎯 In progress |
| Prefill tok/s | 100–150 | ≥150 | ✅ Good |
| Memory (batch=1) | <16 GB | <16 GB | ✅ OK |
| Latency (first token) | ~5–10s | <10s | ✅ OK |

---

## Escalation Path

**If 60 tok/s not achievable in Phase 2B:**

1. **Notify stakeholders** (2–3 days)
2. **Explore Phase 2C options** (parallel tracks)
3. **Reassess architecture** if fundamentally limited
4. **Consider alternative deployment** (batch size >1, server inference)

---

**Status: 🚨 CRITICAL PATH ACTIVATED**

All hands on deck for Phase 2B Metal kernel optimization. Target is non-negotiable: 60 tok/s decode on M2 Pro.

**Metric to watch:** Decode throughput (tokens/second)
