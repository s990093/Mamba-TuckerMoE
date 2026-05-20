## Project Overview

**Hybrid Mamba-TuckerMoE** is a research implementation focused on compute-bound inference of State Space Models (SSMs) on Apple Silicon. It combines Mamba-3 (with trapezoidal discretization and MIMO projections) with Tucker-decomposed Mixture-of-Experts, achieving 2.4B dense-equivalent capacity with only 417M parameters (82.87% compression).

### Key Innovation Areas

- **Mamba-3**: Advanced SSM with higher-order discretization, up to 32K token sequences
- **TuckerMoE**: Low-rank tensor decomposition for efficient expert routing
- **Metal Fusion Kernels**: Custom Metal implementations to move MoE from memory-bound to compute-bound
- **MLX Inference**: Full inference stack optimized for unified memory architecture
- **Speculative Decoding**: Multi-stage inference with draft/target model strategies

---

## Repository Structure

```
├── pre-train/                    # Training scripts, datasets, and notebooks
│   └── train.py                  # Unified training entry point (pre-train + SFT)
├── inference/                    # MLX inference pipeline
│   ├── lib/mlx_hybrid_infer.py   # Core: model loading, caching, Metal kernel wrappers
│   ├── benchmark_mlx.py          # Throughput benchmarking (prefill + decode)
│   ├── stream_mlx.py             # Interactive token streaming with Rich UI
│   ├── run_stable_*.sh           # Quality-focused entry points (standard paths)
│   ├── run_fast_stream.sh        # Speed-optimized (4-bit + fused Metal)
│   ├── cot_middleware.py         # Chain-of-thought token injection during inference
│   ├── cot_format_fsm.py         # CoT format validation FSM
│   ├── chat_demo.py              # Full chat UI with streaming + KV-cache priming
│   └── lib/                      # Core inference modules
│       ├── mlx_hybrid_infer.py   # Model, checkpoints, compile modes, Tucker fusion
│       ├── fused_sampling_metal.py / *_v2.py  # Custom Metal sampling kernels (experimental)
│       └── mlx_mixed_quant.py    # Asymmetric MoE quantization (experimental)
├── metal/                        # Metal shader development & custom kernels
│   ├── ultimate_ssm_scan_bf16.metal     # Fused SSM scan kernel
│   ├── ultimate_mamba_mixer.metal       # Mamba block fusion (Norm + RoPE + Einsum)
│   ├── ultimate_kernel_lib.py           # Python bindings for Metal kernels
│   └── benchmark_fused_mamba_mixer.py   # Accuracy & latency verification
├── cot_dataset/                  # Chain-of-thought dataset pipeline
│   ├── GUIDE.md                  # Dataset collection methodology
│   ├── SFT_FORMAT.md             # ChatML + mask specifications for training
│   ├── SUMMARY.md                # Dataset statistics
│   └── export_hf_dataset.py      # Convert JSON → HuggingFace dataset format
├── checkpoints/                  # Model weights (.pt / .npz sidecars)
├── paper/                        # ICLR 2026 paper artifacts
│   └── hybrid-mamba-15min/       # Technical report & interactive 3D presentation
├── inference/backend/            # FastAPI server (if cloned with full frontend)
├── inference/frontend/           # Next.js React UI (if cloned with full frontend)
├── Makefile                      # Development shortcuts (see "Development Commands")
├── roofline.py, profile_*.py     # Profiling and roofline analysis tools
└── implementation_plan.md        # Detailed optimization roadmap & Metal kernel strategy
```

---

## Development Commands

All commands run from repo root. The Makefile uses `.venv/bin/python3` if available, otherwise system Python.

### Core Inference Benchmarking

```bash
# Full benchmark (prefill + decode with default tokenizer)
make mlx-bench

# Quick benchmark (SEQ_LEN=128, DECODE_TOK=512)
make mlx-bench-quick

# Benchmark with custom checkpoint and sequence length
make mlx-bench CHECKPOINT=weights/model.pt SEQ_LEN=1024 DECODE_TOK=64

# Export .pt → .npz (creates sidecar for faster loading on subsequent runs)
make mlx-export-npz CHECKPOINT=checkpoint.pt

# Benchmark with full-decode graph compilation
make mlx-force-pt BENCH_EXTRA='--prompt "Hello"'
```

### Streaming & Interactive Inference

```bash
# Stable streaming (MLX sampling, eager decode, cache materialization)
make mlx-stream STREAM_EXTRA='--prompt "Explain quantum computing"'

# Fast streaming (4-bit quantized, fused sampling, full-decode compile)
sh inference/run_fast_stream.sh --prompt "..."

# Speculative decoding (draft/target)
make mlx-stream-spec SPEC_DRAFT_LAYERS=8

# Sweep speculative draft layers (6/8/10/12) for A/B comparison
make mlx-stream-spec-sweep
```

### Profiling & Analysis

```bash
# Layer-level profiling (isolate per-block latency)
make mlx-profile PROFILE_DECODE_STEPS=32

# A/B benchmark quantization quality (8-bit vs 0-bit)
make mlx-stream-spec-ab-quant
```

### Backend & Frontend (Full Stack)

```bash
# Start FastAPI backend (with auto-reload)
make backend-dev

# Start FastAPI backend (production)
make backend

# Start Next.js frontend (hot-reload)
make frontend-dev

# Start both together in one terminal
make up
```

### Variable Overrides

Key environment variables for Makefile targets:

- `CHECKPOINT`: Model checkpoint path (auto-resolves to .npz sidecar if available)
- `SEQ_LEN`: Max prefill sequence length (empty = use prompt length)
- `DECODE_TOK`: Number of tokens to generate during decode
- `DTYPE`: Model compute dtype (default: `bf16`; options: `fp32`, `fp16`)
- `KV_DTYPE`: KV cache dtype (default: `bf16`)
- `ROUTER_TEMP`: MoE router temperature (default: `0.5`)
- `INFER_TYPE`: Inference mode (default: `throughput`; options: `safe`, `eager`, `sequential-ssm`)
- `LOOKAHEAD_ROUTER`: Enable lookahead router optimization (`1` or `true`)
- `WARMUP`: Compile warmup iterations (default: `2`)
- `PYTHON`: Override Python interpreter path

Example:

```bash
make mlx-bench CHECKPOINT=custom.pt DTYPE=fp32 DECODE_TOK=256 INFER_TYPE=safe
```

---

## Inference Architecture

### Data Flow: Checkpoint → Model → Prefill → Decode → Sample

1. **Checkpoint Loading** (`lib/mlx_hybrid_infer.py::resolve_mlx_checkpoint`)
   - Supports `.npz` (fast, no PyTorch) and `.pt` (slower, needs torch)
   - On first `.pt` load, writes `.npz` sidecar for next run (prefer .npz in subsequent invocations)
   - Default repo checkpoint: `checkpoint_sft_s27510_model_only.pt`

2. **Model** (`Mamba3LanguageModel`)
   - Mamba blocks: Chunk-wise parallel scan (matches training `pre-train/train.py` math)
   - Transformer blocks: GQA + `mx.fast.scaled_dot_product_attention`
   - MoE: `TuckerMoE` router + low-rank factorized experts; optional Metal fusion (`--tucker-einsum-fuse`, `--full-fuse`)

3. **Prefill** (`benchmark_mlx.py::benchmark`)
   - Process full prompt in one forward pass or compiled graph
   - Populate KV caches and Mamba state
   - Optional: `mx.compile` the entire graph (throughput mode) or per-layer compilation

4. **Decode Loop** (each step: `(B=1, T=1)`)
   - Sample next token using `benchmark_mlx.sample_decode_token`
   - Update caches (KV, Mamba state)
   - Optional outer `mx.compile` (throughput) or eager execution (safe/eager modes)

5. **Sampling**
   - Default: MLX `argmax` / `mx.random.categorical` + optional penalty
   - Experimental: Custom Metal kernels (`--fused-sample-metal` / `--fused-sample-metal-v2`)
   - Note: Custom Metal sampling differs from MLX path in float precision & algorithm details

### Inference Type & Stability Trade-offs

| Mode                   | Prefill            | Decode                    | Use When                                    |
| ---------------------- | ------------------ | ------------------------- | ------------------------------------------- |
| `throughput` (default) | Full-graph compile | Compiled outer loop       | Maximum TPS, minimal stability checks       |
| `safe`                 | Compiled           | Eager (per-token)         | Debugging, alignment with PyTorch reference |
| `eager`                | Eager              | Eager                     | Lowest memory, slowest                      |
| `sequential-ssm`       | Compiled           | Eager with sequential SSM | Detailed per-layer inspection               |

**For stability**, use `safe` or `eager` + `--materialize-caches` + `--no-tucker-einsum-fuse`.

### ChatML Prompt Format

By default, `--prompt` is treated as **plain user text** and auto-wrapped:

```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

Use `--raw-prompt` to pass literal ChatML strings without wrapping.

---

## Training Pipeline

### Dataset Format (Chain-of-Thought)

Training data lives in `cot_dataset/` as JSON with `input`, `cot`, `output`, `category` fields.

```json
{
  "input": "What is 2+2?",
  "cot": "Let me think step by step...",
  "output": "The answer is 4.",
  "category": "emotion|self_awareness|email_summary|movie_intro|noise|system_call|deep_dive"
}
```

→ Converted to ChatML by `cot_dataset/stf_cot_to_bin.py`:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
<think>
{cot}
</think>
<final>
{output}
</final><|im_end|>
```

Refer to `cot_dataset/SFT_FORMAT.md` for:

- System prompt assignment rules per category
- Token masking strategy (loss computed only on assistant output)
- Detailed structure specifications

### Running Training

```bash
# From pre-train/ directory
cd pre-train
python train.py

# Or from repo root (if train.py has been moved/aliased)
python pre-train/train.py
```

Training features:

- Mixed-precision (AMP)
- Exponential Moving Average (EMA)
- Auto-scaling chunk-wise parallel scan
- Supports resuming from checkpoint

---

## Metal Kernel Development

### Key Files

- `metal/ultimate_ssm_scan_bf16.metal`: Fused SSM state transition (already deployed)
- `metal/ultimate_mamba_mixer.metal`: Fused Norm + RoPE + Einsum + Scan (optimization target; see `implementation_plan.md`)
- `metal/ultimate_kernel_lib.py`: Python bindings and grid/threadgroup configuration
- `metal/benchmark_fused_mamba_mixer.py`: Accuracy verification (max error < 0.05) and latency micro-benchmarks

### Workflow for Optimizing a Kernel

1. **Prototype** in `metal/tools/custom_metal_*.py` or notebook
2. **Validate numerically** against MLX reference path using `benchmark_fused_mamba_mixer.py`
3. **Benchmark** layer isolation with `profile_layers.py`
4. **Wrap** in `ultimate_kernel_lib.py` and conditionally gate in `lib/mlx_hybrid_infer.py`
5. **End-to-end test** with `make mlx-stream` or `run_stable_stream.sh`

### Fusion Strategy (from `implementation_plan.md`)

Breaking 100 tok/s requires fusing multiple operations in a single kernel to minimize:

- Memory roundtrips (intermediate tensors)
- Kernel launch overhead

Target: Fuse **Norm → RoPE → Einsum → Scan** into one Metal kernel, expected **2.12× speedup** per layer (0.84 ms → 0.3 ms).

---

## File-Specific Guidelines

### `inference/lib/mlx_hybrid_infer.py`

**Core model and inference glue.**

- Contains `Mamba3Config`, `Mamba3LanguageModel`, `TuckerMoE`, checkpoint loading
- Handles Metal kernel conditionals (`--tucker-einsum-fuse`, `--full-fuse`, `--lookahead-router`)
- Maintains cache materialization logic; decode unstable if caches are symbolic (not materialized)
- **Modify with care**: Changes affect all downstream inference paths

### `inference/benchmark_mlx.py`, `stream_mlx.py`

**Public inference entry points.**

- Share most sampling & cache logic; `benchmark_mlx` pre-materializes caches (stable), `stream_mlx` skips for speed
- Tokenizer defaults to `inference/tokenizer/` (HF-compatible); override with `--tokenizer`
- `--inference-type` controls compilation strategy
- Output formats: JSON benchmark results (`benchmark_mlx.py`) or Rich-formatted streaming

### `cot_dataset/` and `SFT_FORMAT.md`

**Training data pipeline.**

- JSON → HF dataset → .bin tokenized stream → training
- **Always refer to `SFT_FORMAT.md`** before modifying mask logic or system prompt assignment
- Category-to-prompt mapping is externalized in `stf_cot_to_bin.py` for maintainability

### `pre-train/train.py`

**Unified training entry point.**

- Handles both pre-training (next-token prediction) and SFT (supervised fine-tuning with CoT masking)
- Uses MLX for backend; supports mixed precision and EMA
- Chunk-wise parallel scan must match `inference/lib/mlx_hybrid_infer.py` Mamba scan logic exactly

---

## Performance Benchmarking

### Target Metrics (M2 Pro 16GB baseline)

| Metric               | Target     | Achieved             |
| -------------------- | ---------- | -------------------- |
| Prefill throughput   | —          | ~3,800 tok/s         |
| Decode (bf16)        | ~100 tok/s | ~42 tok/s            |
| Decode (8-bit quant) | —          | ~68 tok/s            |
| KV memory @512 steps | —          | 14.1 MiB             |
| Model params         | —          | 417M (vs 2.4B dense) |

### Profiling Tools

- `profile_decode.py`: Pure graph latency (no Python overhead)
- `profile_layers.py`: Per-layer Mamba/Transformer block timing
- `profile_tucker_moe.py`: MoE dispatch and expert execution breakdown
- `inference/tools/profile_mlx_infer.py`: Comprehensive stack trace (wall-clock, thread CPU, MLX peak memory)

### Understanding Bottlenecks

1. Run `make mlx-profile` to identify which layer consumes time
2. Check Metal kernel fusion opportunities (`implementation_plan.md` section 3)
3. Verify cache materialization is active (if using eager decode, ensure `--materialize-caches`)
4. Compare `--inference-type throughput` vs `safe` to isolate compilation overhead

---

## Experimental Features (Use with Caution)

- **Fused Metal sampling**: `--fused-sample-metal` / `--fused-sample-metal-v2` — May differ from MLX numerically
- **Lookahead router**: `--lookahead-router` — Changes routing behavior; not validated against training
- **Tucker full-fuse**: `--full-fuse` — Aggressive kernel fusion; test end-to-end before relying
- **Asymmetric quantization**: `--quantize 4` on MoE experts — Quality degradation expected, bandwidth gains
- **Speculative decoding**: Draft/target split with custom Metal kernels — Latency reduction varies by model

---

## Known Issues & Workarounds

1. **Decode instability with symbolic caches**: If decode tokens are all 0s or highly repetitive, ensure `--materialize-caches` is set or use `--inference-type safe`
2. **Quantized models & CoT injection**: If using 4-bit or 8-bit, CoT special tokens may not route correctly; test with `--quantize 0` first
3. **Metal kernel version drift**: Custom kernels compiled against specific MLX versions; update `ultimate_kernel_lib.py` if Metal compilation fails
4. **Long sequence prefill OOM**: Reduce `SEQ_LEN` or use `--inference-type sequential-ssm` for per-layer iteration

---

## Common Development Patterns

### Adding a New Inference Option

1. Add CLI flag to `benchmark_mlx.py` / `stream_mlx.py` argparse
2. Pass to `Mamba3LanguageModel(..., option=arg)` during instantiation
3. Conditionally gate kernel invocation in model forward pass (e.g., `if self.enable_fusion: ...`)
4. Validate numerically with reference MLX path in a small test case

### Benchmarking a Kernel Change

```bash
# Before & after latency
python inference/tools/profile_mlx_infer.py --checkpoint ... --profile-decode-steps 32

# End-to-end throughput
make mlx-bench DECODE_TOK=256

# Speculative A/B
make mlx-stream-spec-ab-quant
```

### Debugging Numerical Differences

1. Run with `--inference-type safe --no-tucker-einsum-fuse --quantize 0`
2. Compare outputs: expected stable baseline
3. Gradually enable optimizations one at a time
4. Use `implementation_plan.md` section 4 (Verification Plan) as a template

---

## Deployment & Production Notes

- **Checkpoint sidecars**: After first `.pt` load, `.npz` is written; **redistribute both files together** to avoid torch dependency
- **Tokenizer**: Include `inference/tokenizer/` with any deployed model
- **Backend server**: See `inference/backend/.env.example` for configuration
- **Frontend**: Requires Node.js; see `inference/frontend/package.json` for dependencies
- **Metal SDK**: Requires macOS 11+; Metal shaders are compiled on first use (may add latency to first benchmark)

---

## Paper & Citations

Full technical report: **Breaking the Memory Wall: Compute-Bound TuckerMoE for Hybrid State Space Models** (ICLR 2026)

Refer to `paper/hybrid-mamba-15min/report.md` for detailed method and ablations.

---

## Key Reference Files

- **Architecture overview**: `README.md` (this repo)
- **Technical inference details**: `inference/INFERENCE_STACK.md` (covers stability risks, known issues)
- **Implementation roadmap**: `implementation_plan.md` (Metal kernel optimization strategy)
- **Training format**: `cot_dataset/SFT_FORMAT.md` (ChatML + masking specs)
- **Makefile reference**: `Makefile` (development command syntax)
