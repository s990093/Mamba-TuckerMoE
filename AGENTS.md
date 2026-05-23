# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview

**Hybrid Mamba-TuckerMoE** is a research implementation of a state space model combining Mamba-3 selective SSMs with Tucker-decomposed Mixture-of-Experts (MoE) for efficient edge AI deployment on Apple Silicon.

### Core Architecture

- **Model**: Mamba3-TuckerMoE (417M parameters, 2.4B dense-equivalent capacity)
- **Compression**: 82.87% parameter reduction via Tucker decomposition
- **Deployment Target**: iPhone with Apple Silicon (M-series chips)
- **Inference Framework**: MLX (Apple's unified memory framework)
- **Training Framework**: PyTorch + Triton kernels + Accelerate

### Key Innovations

1. **Mamba-3 Trapezoidal Discretization**: Higher-order integration scheme with MIMO projections for sequences up to 32K tokens
2. **Hybrid Architecture**: Combines Mamba-3 SSMs with Grouped-Query Attention (GQA) in a 4:1 ratio (4 Mamba blocks + 1 Transformer block per macro layer)
3. **Tucker-Decomposed MoE**: Achieves compute-bound throughput by processing low-rank latent experts without full weight reconstruction
4. **Custom Metal Kernels**: Fused operations optimized for Apple Silicon unified memory architecture

### Performance Metrics (M2 Pro 16GB)

- **Throughput**: ~3,800 tok/s (Prefill) | 68 tok/s (8-bit Decode)
- **Memory**: 14.1 MiB KV+State @512 steps (80% less than pure Transformers)
- **Vocabulary**: 32,007 tokens (fixed, non-modifiable)
- **Context Window**: 2,048 tokens maximum

## Repository Structure

```
Mamba3-XR/
├── checkpoints/              # Model weights (.npz, .pt)
│   └── tokenizer/           # BPE tokenizer (vocab 32,007)
├── pre-train/               # Training scripts and notebooks
│   └── train.py            # Standalone training script (all-in-one)
├── mamba3_mlx/             # MLX inference stack (Apple Silicon native)
│   ├── run.py              # CLI entry point
│   ├── Makefile            # Quick-launch shortcuts
│   ├── mlx_model/          # Model architecture (ops, tucker_moe, mamba_block, etc.)
│   ├── inference/          # Sampler and generator
│   └── utils/              # Config and system prompts
├── cot_dataset/            # Chain-of-Thought SFT dataset creation
│   ├── GUIDE.md            # Complete dataset creation guide (1614 lines)
│   ├── SFT_FORMAT.md       # ChatML format and masking rules
│   ├── MATH_DRILL.md       # Math drill specifications
│   └── export_hf_dataset.py # Export to HuggingFace format
├── metal/                  # Custom Metal kernels (fused operations)
├── paper/                  # Technical reports and interactive assets
│   └── hybrid-mamba-15min/ # "Breaking the Memory Wall" report
└── Mamba EYES/            # iOS app (SwiftUI)
```

## Building and Running

### Prerequisites

```bash
# Core dependencies
pip install torch numpy einops pytest

# For MLX inference (macOS only)
pip install mlx tokenizers

# For training
pip install torch triton accelerate
```

### Training

The standalone training script includes all dependencies:

```bash
# Single-file training (no external dependencies)
python pre-train/train.py
```

**Key Training Features:**
- Mixed Precision (bf16/fp16 auto-detection)
- Gradient Accumulation
- EMA (Exponential Moving Average)
- Router Temperature Annealing (2.0 → 0.5)
- torch.compile support with Dummy Pass warmup
- Checkpoint resume with rewarmup

**Training Configuration** (edit `train.py` bottom section):
- `D_MODEL=768`, `D_STATE=64`, `NUM_LAYERS=6`
- `KMOE_NUM_EXPERTS=8`, `KMOE_TOP_K=2`
- `KMOE_R1=32`, `KMOE_R2=512`, `KMOE_R3=256`
- `BATCH_SIZE=2`, `GRADIENT_ACCUMULATION_STEPS=8`
- `LR=8e-5`, `WARMUP=400`, `STEPS=60000`

### MLX Inference (Apple Silicon)

```bash
cd mamba3_mlx

# Quick start (self-awareness mode)
make

# Override parameters
make PROMPT="Explain quantum entanglement" TEMP=0.0 MAX_TOK=300

# Different modes
make emotion PROMPT="I feel stuck and overwhelmed"
make deep PROMPT="Compare SSM vs Transformer" MAX_TOK=512
make math PROMPT="What is 23 times 45?"
```

**Available Modes:**
- `self` - Self-awareness / identity
- `emotion` - Emotional support
- `email` - Summarize & email
- `movie` - Movie intro / analysis
- `daily` - Daily conversation
- `math` - Arithmetic drill
- `syscall` - System call / tool invocation
- `deep` - Deep dive / long-form analysis

**Sampling Parameters:**
- `TEMP=0.15` - Temperature (0 = greedy)
- `TOP_K=20` - Top-k sampling
- `TOP_P=0.85` - Nucleus sampling
- `MIN_P=0.08` - Min-p filter
- `REP_PEN=1.32` - Repetition penalty
- `FREQ_PEN=0.05` - Frequency penalty
- `REPEAT_LAST_N=256` - Penalty window

### Python API (MLX)

```python
import mlx.core as mx
from mamba3_mlx.utils.config import Mamba3Config, GenerationConfig
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint
from mamba3_mlx.inference.generator import generate
from tokenizers import Tokenizer

# Load model
cfg = Mamba3Config()
model = Mamba3LanguageModel(cfg)
load_checkpoint(model, "checkpoints/latest_sft_cot_model.npz", dtype=mx.bfloat16)
mx.eval(model.parameters())

# Tokenize
tok = Tokenizer.from_file("cot_dataset/tokenizer.json")
ids = [tok.token_to_id("<s>")] + tok.encode(text, add_special_tokens=False).ids

# Generate
gen_cfg = GenerationConfig(max_tokens=256, temperature=0.8, top_k=40)
stop_ids = [tok.token_to_id("<|im_end|>"), tok.token_to_id("</s>")]
out_ids = generate(model, ids, gen_cfg, stop_token_ids=stop_ids)
print(tok.decode(out_ids, skip_special_tokens=False))
```

## Development Conventions

### Code Style

- **Training Code**: Single-file design (`train.py`) with all dependencies included
- **Inference Code**: Modular MLX stack with clear separation (ops, model, inference, utils)
- **Kernel Code**: Triton for training, Metal for inference
- **Documentation**: Extensive inline comments in Chinese for training, English for inference

### Model Architecture Principles

1. **Tucker Decomposition**: All MoE layers use Tucker decomposition (r1, r2, r3) for compression
2. **Hybrid Blocks**: 4 Mamba blocks + 1 Transformer block per macro layer (30 total blocks)
3. **Router Temperature**: Anneals from 2.0 (uniform) to 0.5 (concentrated) during training
4. **Layer Scale**: Applied to all residual connections (init=1e-2)
5. **RMS Normalization**: Used throughout (eps=1e-5)

### Training Best Practices

- **Gradient Clipping**: Max norm 1.0
- **Loss Components**: CE loss + load balancing (0.1/n) + router z-loss (5e-3/n)
- **Checkpoint Resume**: Uses Dummy Pass to avoid memory spikes
- **Router Annealing**: Cosine schedule from T_start to T_end
- **Gradient Diagnostics**: Check every 100 steps for vanishing/exploding gradients

### Dataset Creation (CoT SFT)

**Critical Rules:**
1. **Never manually write special tokens** (`<think>`, `</think>`, `<final>`, `</final>`) in JSON files
2. **All content must be in English** with no contractions
3. **Zero spelling errors tolerated** (they propagate to model weights)
4. **Token budget constraints**: 512-768 tokens for most categories, ≤128 for math drill
5. **CoT format**: 3-5 steps, each starting with `Step N:`, separated by `\n`

**Dataset Categories (21,000 total):**
- Emotion: 5,000 samples
- Self-Awareness: 5,000 samples
- Email/Summary: 5,000 samples
- Movie Intro: 1,000 samples
- Daily Conversation: 5,000 samples
- Math Drill: ≤200 samples
- System Call: 600 samples
- Deep Dive: 700 samples

See `cot_dataset/GUIDE.md` for complete specifications.

### Metal Kernel Development

- **Fused Operations**: Combine multiple ops to reduce memory bandwidth
- **Async Dispatch**: Use async compute for overlapping operations
- **Threadgroup Memory**: Leverage shared memory for intermediate results
- **BFloat16**: Primary dtype for Apple Silicon (M1+)

## Important Notes

### Vocabulary and Tokenizer

- **Vocabulary is frozen at 32,007 tokens** - Cannot be modified
- Tokenizer files: `cot_dataset/tokenizer.json` and `cot_dataset/tokenizer_config.json`
- **DO NOT MODIFY** tokenizer files - they are part of the trained model

### Model Limitations

- **No network access** - All knowledge from training data
- **Context window: 2,048 tokens** - Hard limit, plan accordingly
- **Edge deployment** - Optimized for iPhone, not server GPUs
- **Offline-first** - Designed to run without internet connection

### Training Considerations

- **Mixed Precision**: Auto-detects bf16 (Ampere+) or fp16 (older GPUs)
- **TF32**: Enabled on Ampere+ for matmul operations
- **Gradient Accumulation**: Required for large effective batch sizes
- **Checkpoint Size**: ~834 MB (bf16), ~417 MB (int8)

### Inference Optimization

- **Prefill vs Decode**: Different code paths for initial processing vs autoregressive generation
- **KV Cache**: Managed per-layer for Transformer blocks
- **Mamba Cache**: State tensors (h, prev_input, angles) for SSM blocks
- **Compile**: Optional `mx.compile` for decode steps (warmup required)

## Testing and Validation

### Quick Smoke Tests

```bash
# Test inference
cd mamba3_mlx && make PROMPT="Hello" MAX_TOK=50

# Test chat server
make chat-smoke

# Test CoT middleware
make cot-verify
```

### Dataset Validation

```bash
# Check JSON syntax
python3 -m json.tool cot_dataset/emotion.json > /dev/null

# Check for contractions
grep -i "dont\|wont\|cant\|im \|youre" cot_dataset/emotion.json

# Export to HuggingFace format
python3 cot_dataset/export_hf_dataset.py \
  --emotion emotion.json \
  --self-awareness self_awareness.json \
  --output stf_cot_hf_auto
```

## Common Issues and Solutions

### Training Issues

**Problem**: Gradient explosion (norm > 5.0)
- **Solution**: Check router temperature, ensure proper initialization, verify data quality

**Problem**: Loss NaN/Inf
- **Solution**: Skip micro-batch, check loss scale, reduce learning rate

**Problem**: Memory OOM on resume
- **Solution**: Dummy Pass warmup handles this automatically

### Inference Issues

**Problem**: Slow decode speed
- **Solution**: Use compiled mode (`COMPILE=1`), ensure bf16 weights, check for memory swapping

**Problem**: Repetitive output
- **Solution**: Increase `REP_PEN` (1.32), `FREQ_PEN` (0.05), `REPEAT_LAST_N` (256)

**Problem**: Incoherent output
- **Solution**: Lower temperature (0.0-0.3), increase `MIN_P` (0.08), reduce `TOP_K` (20)

## References

- Main README: `README.md`
- MLX Inference Guide: `mamba3_mlx/README.md`
- Dataset Creation Guide: `cot_dataset/GUIDE.md` (1614 lines)
- SFT Format Specification: `cot_dataset/SFT_FORMAT.md`
- Math Drill Specification: `cot_dataset/MATH_DRILL.md`
- Training Summary: `cot_dataset/SUMMARY.md`

## Citation

```bibtex
@article{hybrid_mamba_tuckermoe_2026,
  title={Hybrid Mamba-TuckerMoE: Compute-Bound Tensor Decomposition for State Space Models},
  author={Research Implementation},
  journal={Technical Report},
  year={2026}
}
```

---

**Last Updated:** 2026-05-22  
**Model Version:** Mamba3-TuckerMoE 550M (417M active parameters)  
**Inference Stack:** MLX (Apple Silicon native)  
**Training Stack:** PyTorch + Triton + Accelerate
