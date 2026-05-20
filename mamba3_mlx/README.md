# MLX Mamba3 + TuckerMoE: Apple Silicon Inference

Complete MLX implementation of Mamba3 (with MIMO projections and chunkwise parallel scan) + Tucker-decomposed Mixture-of-Experts (TuckerMoE) for compute-bound inference on Apple Silicon.

**Architecture:** 80% Mamba blocks + 20% Transformer blocks with GQA.  
**Parameters:** 417M (82.87% compression vs 2.4B dense)  
**Target:** Breaking 100 tok/s on M2 Pro via ML能够用Prefill/Decode分离、自定义采样、投机解码兼容性、和高效mx.compile编译优化。

---

## Quick Start

### 1. Install Dependencies

```bash
pip install mlx
# Optionally, for weight conversion from PyTorch:
pip install torch
```

### 2. Convert PyTorch Checkpoint to MLX

```bash
python mlx_model/convert_weights.py \
  --pt_path ../checkpoints/checkpoint_sft_s27510_model_only.pt \
  --output model.npz
```

### 3. Run Generation

```bash
python run.py generate \
  --model_path model.npz \
  --prompt "Hello, world!" \
  --max_tokens 256 \
  --temp 0.8 \
  --top_k 40 \
  --top_p 0.9
```

### 4. Run Benchmarks

```bash
python run.py benchmark \
  --model_path model.npz \
  --prompt "The quick brown fox" \
  --num_generate 256 \
  --verbose
```

---

## Directory Structure

```
mamba3_mlx/
├── mlx_model/                # Model layers
│   ├── ops.py               # Tanh, SiLU, RoPE, RMSNorm, LayerScale
│   ├── tucker_moe.py        # Tucker MoE layer + router
│   ├── mamba_block.py       # Mamba3 with parallel scan
│   ├── transformer_block.py # Attention + FFN
│   ├── hybrid_model.py      # Full model + config
│   ├── convert_weights.py   # PyTorch → MLX conversion
│   └── __init__.py
├── inference/               # Generation pipeline
│   ├── sampler.py          # Sampling strategies
│   ├── generator.py        # Prefill/decode loop
│   ├── speculative.py      # Speculative decoding
│   └── __init__.py
├── utils/
│   ├── config.py           # Config dataclasses
│   ├── args.py             # CLI argument parsing
│   └── __init__.py
├── tests/                   # Unit tests
│   ├── test_ops.py
│   ├── test_sampler.py
│   └── __init__.py
├── run.py                   # Main entry point
├── __init__.py
└── README.md
```

---

## Key Features

### 1. **Prefill/Decode Separation**

- **Prefill stage:** Process full prompt in one forward pass, cache Mamba state
- **Decode stage:** Single-token forward with state management, optimized for throughput

```python
from mamba3_mlx import AutoregressiveGenerator

generator = AutoregressiveGenerator(model, tokenizer)
result = generator.generate(prompt_text, max_tokens=256)
# Includes: prefill_latency, decode_throughput
```

### 2. **Advanced Sampling**

Support all major sampling strategies:
- **Top-k:** Keep top-k tokens by probability
- **Top-p (Nucleus):** Keep tokens until cumulative probability exceeds p
- **Min-p:** Keep tokens with probability ≥ min_p × max_probability
- **Repetition Penalty:** Discourage token repetition
- **Presence/Frequency Penalties:** OpenAI-style penalties

```python
from mamba3_mlx import TextSampler

sampler = TextSampler(
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    min_p=0.05,
    repetition_penalty=1.1,
    frequency_penalty=0.02,
)
token = sampler(logits, generated_ids)
```

### 3. **Speculative Decoding (Optional)**

Use lightweight draft model to generate K candidate tokens, verify with target model in parallel:

```python
from mamba3_mlx.inference.speculative import SpeculativeGenerator

gen = SpeculativeGenerator(target_model, draft_model, tokenizer, k=5)
result = gen.generate(prompt, max_tokens=256)
# Returns: accepted token ratio, speedup estimate
```

### 4. **Compilation Optimization**

Enable `--full-decode-compile` to JIT-compile decode step with `mx.compile`:
- Generates fused Metal kernel graph
- Reduces kernel dispatch overhead
- Trade-off: compilation latency vs repeated throughput

```bash
python run.py generate --model_path model.npz --prompt "..." --full-decode-compile
```

---

## CLI Parameters (Section VIII Reference)

| Parameter         | Default | Description                            |
| -----------       | ------- | -------------------------------------- |
| `--model_path`    | -       | Path to .npz or .pt checkpoint         |
| `--prompt`        | -       | Input text                             |
| `--max_tokens`    | 256     | Max generation length                  |
| `--temp`          | 0.8     | Sampling temperature                   |
| `--top_k`         | 40      | Top-k filtering                        |
| `--top_p`         | 0.9     | Nucleus filtering                      |
| `--min_p`         | 0.05    | Min-p relative threshold               |
| `--rep_pen`       | 1.1     | Repetition penalty                     |
| `--pres_pen`      | 0.0     | Presence penalty                       |
| `--freq_pen`      | 0.02    | Frequency penalty                      |
| `--repeat_last_n` | 64      | Penalty window                         |
| `--dtype`         | bf16    | Weight dtype (fp32, bf16, fp16)        |
| `--kv_dtype`      | auto    | KV cache dtype                         |
| `--full-decode-compile` | True | Enable mx.compile for decode           |
| `--speculative`   | -       | Enable speculative decoding            |
| `--speculative_k` | 5       | Draft tokens per iteration             |

---

## Architecture Details

### Hybrid Block Structure (4 Mamba + 1 Transformer per cycle)

```
Layer 0: Mamba3Block
Layer 1: Mamba3Block
Layer 2: Mamba3Block
Layer 3: Mamba3Block
Layer 4: TransformerBlock (GQA + TuckerMoE FFN)
Layer 5: Mamba3Block
...repeats...
```

### Mamba3 Block

- **Input projection:** Split into gate and SSM input
- **SSM computation:** Parallel scan via dense matrix method (chunk_parallel_scan)
  - For `L=1` (decode): Use shortcut formula `h_new = exp(dt*A) * h_prev + u_ssm`
  - For `L>1` (prefill): Build triangular coefficient matrix per chunk
- **RoPE:** Apply rotary position encoding with cumsum(dt*theta)
- **Output:** SiLU gate × SSM output + skip connection

### TransformerBlock

- **Attention:** Grouped-Query Attention (GQA) with `mx.fast.scaled_dot_product_attention`
- **FFN:** Either standard SiLU-gated or TuckerMoE-based
- **Residuals:** LayerScale per block

### TuckerMoE

Tucker factorization:
```
G_e = U_expert[e] ⊗ core[?, ?, ?]
x_shared = x @ U_in → norm → x_core ← sum_k p_k @ G_{top_k}
out = x_core @ U_out + bias
```

**Training losses:**
- Z-loss: `mean(logsumexp(capped_logits)^2)`
- Load-balancing: `sum(E * expert_usage * router_probs)`

---

## Data Types & Precision

- **Default:** bf16 (bfloat16) — optimal precision/speed trade-off on M2+
- **Options:** fp32, fp16
- **KV dtype:** Auto-matches weight dtype or explicitly specified
- **Mamba state (h_prev, dt_cumsum):** Materialized in memory (not symbolic) for stable decode

---

## Running Tests

```bash
# Unit tests for ops
python -m pytest tests/test_ops.py -v

# Unit tests for sampling
python -m pytest tests/test_sampler.py -v

# Run all tests
python -m pytest tests/ -v
```

---

## Performance Targets (M2 Pro 16GB)

| Metric             | Target | Notes                          |
| ---                | ------ | ------------------------------ |
| Prefill throughput | ~3800  | tok/s (batch=1, seq_len=512)   |
| Decode (bf16)      | 100+   | tok/s (per plan goal)          |
| Decode (8-bit)     | 150+   | tok/s (with quantization)      |
| KV memory @512     | 14.1   | MiB (Mamba has no KV cache)    |

---

## Known Issues & Workarounds

1. **MLX lacks `mx.scan`:** Implemented via dense matrix method (chunk_parallel_scan)
2. **Dynamic shape compilation:** Prefill uses eager mode; decode uses `@mx.compile`
3. **Route losses in inference:** Disabled (set to 0) during `--inference-type inference`
4. **State materialization:** Critical for decode stability; use `--materialize-caches` (auto on decode)

---

## Future Enhancements

- [ ] Custom Metal kernels for chunk_parallel_scan (target 2× speedup)
- [ ] KV cache for Transformer layers (for long-context efficiency)
- [ ] Lookahead router for TuckerMoE
- [ ] Dynamic batch support (currently B=1 only)
- [ ] Asymmetric quantization for experts

---

## References

- **Full technical report:** `paper/hybrid-mamba-15min/report.md`
- **Training pipeline:** `pre-train/train.py`
- **Inference architecture:** See INFERENCE_STACK.md (PyTorch version)
- **Implementation plan:** `implementation_plan.md`

---

## Citation

If using this implementation, please cite:

```bibtex
@article{hung2026hybrid,
  title={Breaking the Memory Wall: Compute-Bound TuckerMoE for Hybrid State Space Models},
  author={Hung, Wei},
  year={2026},
  journal={ICLR 2026},
}
```

---

## License

Same as parent project.
