# MLX Mamba3 Implementation Guide

This guide explains the architecture and key design decisions for the MLX inference stack.

---

## Design Philosophy

**Principle:** Strict compliance with the plan in section I–X. No speculative enhancements.

1. **Simplicity First:** No abstractions beyond what the plan requires
2. **MLX Native:** Leverage MLX operations (`mx.fast.scaled_dot_product_attention`, `mx.compile`, etc.)
3. **State Management:** Explicit, not implicit (states passed as dicts)
4. **Numerics:** Materialize states, avoid symbolic computation during decode

---

## Core Components Overview

### 1. Ops Layer (`mlx_model/ops.py`)

Pure MLX operations, no Triton/Metal:

- **tanh_approx(x):** 2σ(2x) - 1, matches PyTorch kernel behavior
- **scaled_tanh(x, scale):** tanh_approx(x/scale) * scale, used in router
- **silu(x), silu_gating(gate, feat):** Activation + gating
- **rope(x, angles):** Rotary position encoding
- **RMSNorm, LayerScale:** Normalization layers

**Key insight:** No gradients tracked in inference, so custom torch.autograd.Function equivalents are N/A.

### 2. TuckerMoE Layer (`mlx_model/tucker_moe.py`)

Tucker decomposition:

```
Routing:
  logits = router(x) → scaled_tanh(., 10) → softmax(. / temp)

Expert gates:
  G_e = einsum("er, rst -> est", U_expert[e], core)  # (r3, r2)

Output:
  x_shared = x @ U_in → norm
  x_core = sum_{k=0}^{top_k} prob_k * (x_shared @ G_{idx_k})
  out = x_core @ U_out + bias
```

**Training losses (opt-in):**
- Z-loss: `mean(logsumexp(capped)^2)` regularizes router magnitude
- LB-loss: `num_experts * sum(mean(expert_mask) * mean(router_probs))` balances loading

**Inference path:** Set `training=False` → both losses return 0.

### 3. Mamba3Block (`mlx_model/mamba_block.py`)

Two paths:

#### Prefill Path (L > 1)
1. Project input: `[x_proj, gate] = in_proj(x)`
2. Compute SSM params: dt, A, B, C from input
3. **Parallel scan** (sequential here, can parallelize):
   - Build triangular product matrix M per chunk
   - `h_intra = M @ u` per chunk
   - Connect chunks via decay accumulation
4. Output: `y = C @ h`
5. Gate: `y = silu(gate) * y`
6. MIMO: `y_mimo = U(y)`, `y = y + V(y_mimo)`
7. Out: `out_proj(y) + skip`

#### Decode Path (L = 1)
- Skip parallel scan
- Use formula: `h_new = exp(dt*A) * h_prev + B*u_ssm`
- Much faster than scan for single token

**State management:**
```python
state = {
    "h_prev": (B, n_heads, d_state),  # Hidden state
    "dt_cumsum_prev": (B, n_heads),   # For RoPE phase
}
```

### 4. TransformerBlock (`mlx_model/transformer_block.py`)

- **GQA:** Q/K/V projections with head grouping
  ```python
  K_expanded = repeat(K, kv_groups)  # Share KV across query groups
  attn = mx.fast.scaled_dot_product_attention(Q, K_expanded, V_expanded)
  ```
- **FFN:** Either StandardFFN (silu-gated) or TuckerMoE
- **Residuals + LayerScale** per sub-layer

### 5. Full Model (`mlx_model/hybrid_model.py`)

`TrueHybridMamba`: Cycles of (4 Mamba + 1 Transformer).

`Mamba3LanguageModel`:
- Embedding (shared with lm_head)
- Backbone (TrueHybridMamba)
- Output norm + lm_head

**Forward signature:**
```python
def forward(token_ids, states=None, training=False):
    """
    Args:
        token_ids: (B, L) or (B, 1)
        states: [state_dict, ...] per Mamba layer, None for prefill
        training: affects MoE loss computation
    
    Returns:
        logits: (B, L, vocab_size)
        new_states: updated states
        aux_loss: MoE loss if training
    """
```

---

## Generation Pipeline

### Sampler (`inference/sampler.py`)

**Order of operations:**
1. Temperature scaling: `logits /= temp`
2. Repetition penalty: divide logits of recent tokens
3. Frequency/presence penalties: subtract from logits
4. Top-k filtering: keep only k highest, set others to -∞
5. Nucleus (top-p) filtering: keep until cumulative prob > p
6. Min-p filtering: keep if prob ≥ min_p × max_prob
7. Softmax → categorical sample

**Key:** -∞ values must be handled in softmax (numerical stability):
```python
logits_safe = mx.where(mx.isinf(logits), -1e9, logits)
probs = mx.softmax(logits_safe)
```

### Generator (`inference/generator.py`)

**Prefill stage:**
```python
logits, states, _ = model(prompt_tensor)  # (B, L, vocab_size)
next_token = sample(logits[0, -1, :], generated_ids=[])
```

**Decode loop:**
```python
for step in range(max_tokens):
    token_tensor = mx.array([[next_token]])
    logits, states, _ = model(token_tensor, states=states)
    next_token = sample(logits[0, 0, :], generated_ids[...])
```

**State update:** Mamba blocks return `{"h_prev": ..., "dt_cumsum_prev": ...}`, pass to next iteration.

---

## Speculative Decoding (`inference/speculative.py`)

**Pipeline:**
1. Draft model generates K tokens sequentially
2. Target model evaluates `[prefix + K draft]` in one forward
3. Accept/reject per token: `random() < target_prob / draft_prob`
4. Repeat from first rejected token

**Key insight:** Target model sees full context in one go, no sequential steps for verification.

---

## Argument Parsing & Config

### `utils/config.py`

Two dataclasses:
- `GenerationConfig`: All sampling hyperparameters (temp, top_k, ..., dtype)
- `ModelConfig`: Architecture (d_model, num_layers, ...)

### `utils/args.py`

CLI parsing via argparse. Converts args → config objects.

---

## Weight Conversion (`mlx_model/convert_weights.py`)

**Process:**
1. Load PyTorch checkpoint: `torch.load(pt_path)`
2. Extract state_dict (handle "model", "state_dict" variants)
3. Convert each tensor to numpy
4. Save as .npz: `np.savez(output, **dict_of_arrays)`

**Weight sharing:** If `lm_head` not in checkpoint, copy from `embed.weight`.

---

## CLI Usage

### Generate Mode

```bash
python run.py generate \
  --model_path checkpoint.npz \
  --prompt "Hello" \
  --max_tokens 256 \
  --temp 0.8 \
  --top_k 40 \
  --full-decode-compile
```

### Benchmark Mode

```bash
python run.py benchmark \
  --model_path checkpoint.npz \
  --prompt "Test" \
  --num_generate 100 \
  --verbose
```

---

## Running Tests

```bash
# Test ops
python -m pytest tests/test_ops.py::test_tanh_approx -v

# Test sampling
python tests/test_sampler.py

# All tests
python -m pytest tests/ -v
```

---

## Data Type Handling

| Config    | Example | Application                  |
| -----     | ------- | ----------------------------- |
| bf16      | Default | Weights, compute, KV cache    |
| fp32      | Optional| Accumulation, high precision  |
| fp16      | Optional| Memory-constrained inference  |

**During inference:**
- Weights loaded as configured dtype
- KV states materialized in same dtype
- Activations computed in dtype

---

## Debugging Tips

### Silent/Zero Outputs

- **Cause:** Symbolic caches not materialized
- **Fix:** Ensure `materialize_caches=True` during decode

### Mismatched Logits (Prefill vs Decode)

- **Cause:** State not properly carried forward
- **Fix:** Verify state dict keys match layer expectations

### NaN in Generation

- **Cause:** Numerical instability in softmax with -∞
- **Fix:** Use `mx.where(mx.isinf(logits), -1e9, logits)` before softmax

### Slow Decode

- **Cause:** `mx.compile` disabled or not effective
- **Fix:** Enable `--full-decode-compile`, check Metal compilation logs

---

## Future Extensions

### 1. Custom Metal Kernels

Replace `chunk_parallel_scan` in mamba_block.py with fused Metal:
```python
# In mamba_block.py
if use_metal_scan:
    y, h_new, dt_cum = metal_chunk_parallel_scan(u_ssm, dt_b, A_b, C_rotated)
else:
    y, h_new, dt_cum = chunk_parallel_scan(...)
```

### 2. KV Cache for Attention

Add to TransformerBlock:
```python
def forward(self, x, kv_cache=None):
    # Store K, V if caching
    if kv_cache is None:
        kv_cache = {"K": K, "V": V}
    else:
        K = mx.concatenate([kv_cache["K"], K_new], axis=1)
        V = mx.concatenate([kv_cache["V"], V_new], axis=1)
    ...
```

### 3. Quantization

Extend weight loading:
```python
if quantize_bits == 4:
    weights = dequantize(weights, scale=scale)
```

---

## References

- **Plan:** User's plan sections I–X (this repo's memory)
- **PyTorch model:** `/Users/hungwei/Desktop/Proj/Mamba3-XR/sft_cot_bundle/scripts/model.py`
- **Tokenizer:** `/Users/hungwei/Desktop/Proj/Mamba3-XR/cot_dataset/tokenizer.json`
- **Weights:** `/Users/hungwei/Desktop/Proj/Mamba3-XR/checkpoints/latest_sft_cot_model.npz`

---

## Support

For issues:
1. Check `CLAUDE.md` in parent repo (project guidelines)
2. Review `implementation_plan.md` (original design notes)
3. Test with ops tests: `python tests/test_ops.py`
4. Add `--verbose` to see timing logs
