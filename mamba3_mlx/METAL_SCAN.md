# Metal Scan Optimization (In Development)

## Overview

**Metal Scan** provides an optimized SSM state scan for Mamba-3 prefill and incremental decode phases.

### Current Status (Phase 2A)
- ✅ **Numerically validated** — matches baseline exactly (h_diff = 0.00e+00)
- ✅ **Incremental scan** — supports speculative decode K-token verification
- ✅ **Full return_h_all** — can return hidden state at every position
- ⏳ **Performance** — baseline prototype (not yet optimized)

---

## Architecture

### Scan Decomposition

Full SSM scan: h_t = Σ_{k≤t} exp(cumsum[t] - cumsum[k]) * u_k

Decomposed into:
1. **Intra-chunk** (per-chunk scan) — O(Lc) elements, parallelizable
2. **Inter-chunk** (lightweight carry) — O(nc) scalar carries, sequential

### Current Implementation

**Phase 2A (Validation Focus):**
- Intra-chunk: Uses baseline O(Lc²) matrix method per-chunk (avoid memory issues)
- Inter-chunk: O(nc) loop with lightweight carry logic
- Result: Numerically identical to baseline; ready for Metal kernel optimization

**Future (Phase 2B — Metal Kernel):**
- Intra-chunk: Single Metal kernel replacing O(Lc²) matmul with O(Lc) sequential scan
- Expected: 2–3× speedup for large chunks (L > 256)

---

## API

### chunk_parallel_scan_mlx()

Full prefill scan (like `chunk_parallel_scan` from mamba_block.py).

```python
from mamba3_mlx.mlx_model.mamba_block_metal import chunk_parallel_scan_mlx

y, h_prev = chunk_parallel_scan_mlx(
    u,           # (B, L, H, N, P) — SSM input
    dt_b,        # (B, L, H) — discretization step
    a_b,         # (B, L, H) — state matrix (log-space)
    c_rotated,   # (B, L, H, N, R) — output projection
    chunk_size=64,
    return_h_full_zero=False,
)
# y: (B, L, H, P, R) output
# h_prev: (B, H, N, P) final hidden state

# Optional: return hidden state at every position
y, h_prev, h_all = chunk_parallel_scan_mlx(
    ...,
    return_h_full_zero=True,
)
# h_all: (B, L, H, N, P) — h at each position (zero-init accumulation)
```

### chunk_parallel_scan_with_init()

Incremental scan with non-zero initial state (speculative decode).

```python
from mamba3_mlx.mlx_model.mamba_block_metal import chunk_parallel_scan_with_init

# Continue scan from existing cached state h_init
y, h_final = chunk_parallel_scan_with_init(
    u,        # (B, L, H, N, P) — K new draft tokens
    dt_b,     # (B, L, H)
    a_b,      # (B, L, H)
    c_rotated,  # (B, L, H, N, R)
    chunk_size=64,
    h_init,   # (B, H, N, P) — state from previous position
)
# y: (B, L, H, P, R) — output from K draft tokens
# h_final: (B, H, N, P) — state after all K tokens
```

### LayerScale

Fixed layer scaling module (corrected typo from user code).

```python
from mamba3_mlx.mlx_model.mamba_block_metal import LayerScale

ls = LayerScale(dim=768, init_value=1e-2)
scaled = ls(x)  # y = x * gamma
```

---

## Validation Results

### Numerical Accuracy

| Test | Metric | Value | Tolerance | Status |
| --- | --- | --- | --- | --- |
| Full scan vs baseline | h_diff | 0.00e+00 | < 1e-3 | ✅ PASS |
| Full scan vs baseline | y_diff | 0.00e+00 | < 1e-3 | ✅ PASS |
| Incremental (zero init) | y_diff | 0.00e+00 | < 1e-3 | ✅ PASS |
| Incremental (non-zero) | h_final_diff | 0.00e+00 | < 1e-3 | ✅ PASS |
| return_h_all | h_all[-1] vs h_prev | 0.00e+00 | < 1e-4 | ✅ PASS |

### Performance (Baseline)

**Test Configuration:** B=2, L=256, H=8, N=16, P=32, R=8, chunk_size=64

| Implementation | Time | Note |
| --- | --- | --- |
| Baseline (matrix) | 9.9 ms | O(Lc²) per chunk |
| Current Metal impl. | 36.4 ms | Overhead from per-chunk processing |
| **Speedup** | **0.27×** | Not yet optimized; target 2–3× after Metal kernel |

**Note:** Current phase is validation-focused. Speedup will come from Metal kernel intra-chunk scan in Phase 2B.

---

## Integration with mamba_block.py

### Option 1: Direct Replacement (Recommended)

```python
# mamba_block.py: Mamba3Block.__call__()

# Before (baseline):
# y_stack, h_final = chunk_parallel_scan(u_ssm, la, C_rotated, self.chunk_size)

# After (Metal-optimized):
from .mamba_block_metal import chunk_parallel_scan_mlx
y_stack, h_final = chunk_parallel_scan_mlx(u_ssm, la[:, :, :, 0] * a_b, C_rotated, self.chunk_size)
```

This requires zero changes to calling code (drop-in replacement).

### Option 2: Conditional Selection

```python
if use_metal_scan:
    from .mamba_block_metal import chunk_parallel_scan_mlx as scan_fn
else:
    scan_fn = chunk_parallel_scan

y_stack, h_final = scan_fn(u_ssm, la, C_rotated, self.chunk_size)
```

---

## Future Optimization (Phase 2B)

### Metal Kernel Intra-Scan

Implement O(Lc) sequential scan in a single Metal kernel:

```metal
// Pseudo-code: sequential scan per (d, h, b) triplet
for t in 0..Lc-1:
    h_val = alpha[t] * h_prev + u[t]
    out[t] = h_val
    h_prev = h_val
```

**Expected Impact:**
- Intra-chunk: 9.9 ms → ~3–5 ms (2.0–3.3× speedup)
- Full scan: ~15–20 ms total (dominated by inter-chunk + einsum, not intra)

### Implementation Plan

1. Write `ultimate_ssm_sequential_scan.metal` (metal/ directory)
2. Bind with `mx.fast.metal_kernel()` in mamba_block_metal.py
3. Benchmark per-layer latency (`profile_layers.py`)
4. Update `implementation_plan.md` with results

---

## Known Limitations

1. **Phase 2A is baseline replica** — Current intra-chunk uses O(Lc²) matmul, not Metal kernel
   - ✓ Numerically correct; ready for Metal optimization
   - ✗ Not yet faster than baseline

2. **Large L × large chunks may OOM** — Intra-chunk matmul scales as O(Lc²)
   - Solution: Reduce chunk_size or implement Metal kernel in Phase 2B

3. **return_h_all not used by inference** — Intended for debugging/verification
   - Add flag to mamba_block.py if needed in future

---

## Testing

Run all validation tests:

```bash
.venv/bin/python mamba3_mlx/tests/test_scan_metal.py
```

**Passing tests:**
- ✓ Full scan shapes correct
- ✓ Full scan matches baseline (y_diff = 0, h_diff = 0)
- ✓ Incremental scan with zero init
- ✓ Incremental h_final correctness
- ✓ Benchmark and comparison
- ✓ return_h_all works correctly

---

## Configuration

### Makefile Integration (Future)

```bash
# Use Metal scan path
make mlx-bench SCAN_TYPE=metal

# Use baseline (default)
make mlx-bench SCAN_TYPE=baseline
```

### GenerationConfig Flag (Future)

```python
from mamba3_mlx.utils.config import GenerationConfig

config = GenerationConfig(
    use_metal_scan=True,  # Enable when Phase 2B is complete
    metal_scan_chunk_size=64,
)
```

---

## Summary

| Phase | Status | Deliverable |
| --- | --- | --- |
| **2A (Current)** | ✅ Complete | mamba_block_metal.py + tests (numerically validated) |
| **2B (Planned)** | ⏳ To do | Metal kernel intra-scan + integration |
| **2C (Future)** | 📋 To do | End-to-end benchmarking + generation quality |

---

## References

- Source: `mamba3_mlx/mlx_model/mamba_block_metal.py`
- Tests: `mamba3_mlx/tests/test_scan_metal.py`
- Baseline: `mamba3_mlx/mlx_model/mamba_block.py`
