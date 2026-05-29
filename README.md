# Hybrid Mamba-TuckerMoE for On-Device LLM Inference

> **Joint Tucker Decomposition MoE · Multi-Strategy Speculative Decoding · Resource-Constrained Devices**
>
> Hung-Wei Lai · Hsin-An Lan · Chun-Ming Hsu · Yu-Han Lu
>
> Department of Computer Science and Information Engineering, National Kaohsiung University of Science and Technology, Kaohsiung, Taiwan

![Method Pipeline](paper/hybrid-mamba-15min/assets/method_flowchart.png)

## Overview

This repository contains the training code, inference implementation, and interactive 3D paper assets for **Hybrid Mamba-TuckerMoE**.

### Key Innovations:

1. **Mamba-3 (Trapezoidal Discretization)**: Replaces standard Euler methods with a higher-order trapezoidal integration scheme and MIMO projections, pushing sequence lengths up to 32K tokens efficiently.
2. **Hybrid Mamba-TuckerMoE**: Combines Mamba-3 selective SSMs with Grouped-Query Attention (GQA) and joint Tucker-decomposed experts. This achieves a **2.4B dense-equivalent capacity** with only **417M parameters** (82.87% compression).
3. **On-the-fly Inference Pipeline**: Eliminates memory bandwidth bottlenecks by processing low-rank latent experts without full weight reconstruction, achieving compute-bound throughput on Apple Silicon.

## 📂 Repository Structure

```text
├── mamba3_mlx/                      # ⚡ MLX inference stack (Apple Silicon)
│   ├── inference/                   #   │   Core engine: generator, sampler, token bans
│   ├── mlx_model/                   #   │   Model architecture v1 (hybrid, mamba, tucker)
│   ├── mlx_model_v2/                #   │   Model architecture v2 (updated scan_metal)
│   ├── mv/                          #   │   CoT middleware (FSM, format enforcement)
│   ├── speculative/                 #   │   Speculative decode (Jacobi, ngram, warmup)
│   ├── ui/                          #   │   Chat frontend (HTML/CSS/JS)
│   ├── tools/                       #   │   Utilities (sidecar converter)
│   ├── utils/                       #   │   Config, system prompts
│   ├── run.py                       #   │   Entry point
│   ├── chat_demo.py                 #   │   FastAPI WebSocket server (port 7860)
│   ├── Makefile                     #   │   Development shortcuts
│   └── API.md                       #   │   API reference
│
├── pre-train/                       # 🏋️ Training scripts, logs, notebooks
│   ├── train.py                     #   │   Single-file training (PyTorch + Triton)
│   ├── kmoe_train.py                #   │   kMoE training variant
│   ├── sft_cot_bundle/              #   │   SFT dataset pipeline
│   └── cot_dataset/                 #   │   Symlink → ../cot_dataset
│
├── cot_dataset/                     # 📚 SFT dataset & tokenizer
│   ├── tokenizer.json               #   │   Vocab 32,007 (frozen)
│   ├── metal/                       #   │   Custom Metal shaders + benchmark scripts
│   ├── GUIDE.md                     #   │   Dataset format guide
│   └── SFT_FORMAT.md               #   │   SFT spec
│
├── checkpoints/                     # 💾 Model weights (v1/v2/v3, .npz sidecars)
│   └── 4_loss_func/                 #   │   latest_sft_cot_model.*.npz
│
├── paper/
│   └── hybrid-mamba-15min/          # 📄 Technical report + 3D interactive assets
│
├── docs/                            # 📖 Supplementary docs
├── assets/                          # 🖼️ Presentation visuals
├── AGENTS.md                        # 🤖 Agent workflow guidance
└── Makefile                         # 🛠️ Root-level benchmarks
```

---

## 🖥 TuckerMoE 3D Interactive Simulator

As part of the ICLR 2026 paper submission, we provide an interactive 3D web-based presentation simulator.
It visually demonstrates:

- **Tucker Matrix Decomposition** compressing the state space from $134\text{MB}$ to $8.4\text{MB}$.
- **On-the-fly Inference Pipeline**, comparing native block latency against our specialized micro-tensor flow.

---

## 📊 Performance & Efficiency Benchmarks

Hybrid Mamba-TuckerMoE is optimized for **Apple Silicon (Unified Memory Architecture)**, delivering high throughput and low memory footprint.

### Key Results (M2 Pro 16GB):

- **Throughput**: **~3,800 tok/s** (Prefill) | **68 tok/s** (8-bit Quantized Decode).
- **Compression**: **82.87%** parameter reduction (417M actual vs 2.4B dense-equivalent).
- **Memory Efficiency**: **14.1 MiB** KV+State memory @512 steps (80% less than pure Transformers).
- **Compute-Bound**: Fused Metal kernels move MoE dispatch from memory-bound to compute-bound states.

| Metric               | Hybrid (bf16) | Hybrid (8-bit) | Saving vs Transformer |
| :------------------- | :-----------: | :------------: | :-------------------: |
| **Decode Speed**     |   42 tok/s    |    68 tok/s    |           -           |
| **KV Memory (@512)** |   22.3 MiB    |    14.1 MiB    |       **~80%**        |

### Benchmark Visualization

<p align="center">
  <img src="paper/hybrid-mamba-15min/assets/plots/mlx_inference_benchmark.png" width="800" alt="MLX Inference Benchmark">
  <br><i>Figure 1: Inference throughput and memory growth analysis on Apple Silicon.</i>
</p>

<p align="center">
  <img src="paper/hybrid-mamba-15min/assets/plots/pareto_frontier.png" width="400" alt="Pareto Frontier">
  <img src="paper/hybrid-mamba-15min/assets/plots/bench_decode_compile_comparison.png" width="400" alt="Compile Uplift">
  <br><i>Figure 2: (Left) Pareto Frontier of capacity vs cost; (Right) Graph compilation speedup (+36.8%).</i>
</p>

> [!TIP]
> For a detailed technical breakdown, see the full report: [Breaking the Memory Wall: Compute-Bound TuckerMoE for Hybrid SSMs](paper/hybrid-mamba-15min/report.md).

---

## 🧠 Mamba-3 Architecture Details

Mamba-3 is an advanced iteration of the Selective State Space Model architecture.

- **Trapezoidal Discretization**: Second-order approximation for more accurate continuous-to-discrete mapping.
- **MIMO Projections**: Rank-based expansion (12% latency for 4x capacity).
- **Complex-Valued Dynamics**: RoPE-based simulation of complex SSMs in a real-valued framework.
- **Vision Support**: Integrated Vision Mamba with Snake Scan and bidirectional processing.

## 🚀 Quick Start

### Inference on Apple Silicon (MLX)

```bash
# Self-awareness demo
make -C mamba3_mlx

# Emotion mode
make -C mamba3_mlx emotion PROMPT="I feel stuck"

# Deep reasoning with compiled decode
make -C mamba3_mlx deep PROMPT="..." MAX_TOK=512 COMPILE=1

# WebSocket chat server (port :7860)
make -C mamba3_mlx chat

# Direct entry point
python mamba3_mlx/run.py
```

### Training (PyTorch + Triton)

```bash
python pre-train/train.py        # Single-file, all hyperparams at bottom
python pre-train/kmoe_train.py   # kMoE variant
```

### Setup

```bash
git clone https://github.com/s990093/Mamba3-XR.git
cd Mamba3-XR
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Weights (`.npz`, bf16) go in `checkpoints/` — see `AGENTS.md` for sidecar conversion.

---

## ⚡ MLX Inference Design

The inference stack (`mamba3_mlx/`) is purpose-built for Apple Silicon's Unified Memory Architecture (UMA).

### Key Principles

- **No device transfers** — CPU and GPU share the same physical memory; no `.to('cuda')` / `.cpu()` copies.
- **Lazy evaluation** — MLX builds a computation graph and defers execution until `mx.eval()`, enabling graph-level fusion.
- **Compiled decode** — `mx.compile(model)` traces and JIT-compiles the decode step after a warmup pass, delivering ~37% decode speedup.

### Architecture

| Component | File | Role |
|-----------|------|------|
| `Mamba3LanguageModel` | `mlx_model/hybrid_model.py` | Embed → 30 sub-layers (4 Mamba + 1 Transformer per macro layer) → norm → head |
| `Mamba3Block` | `mlx_model/mamba_block.py` | Mamba-3 SSM with trapezoidal discretization, MIMO projections, TuckerMoE gating |
| `TransformerBlock` | `mlx_model/transformer_block.py` | GQA attention + TuckerMoE FFN |
| `TuckerMoE` | `mlx_model/tucker_moe.py` | 8 experts, top-2 router; precomputes G = U_expert ⊗ core once to skip Tucker einsum at decode time |
| `chunk_scan` | `mlx_model/scan_metal.py` | Metal-kernel SSM scan (intra-chunk O(Lc), inter-chunk GPU dispatch); transparent fallback to pure MLX |

### Token Bans (`inference/token_bans.py`)

A hardcoded blocklist of 50+ token IDs that should never be sampled — including `<|reserved|>` placeholders, broken Unicode fragments, and other garbage tokens from the 32,007 vocab. Applied post-softmax by zeroing their logits before sampling.

### CoT Middleware (`mv/`)

Two components enforce and guide the Chain-of-Thought output format:

- **`cot_middleware.py`** — Phase-aware seeding: injects `<think>\n` at start, then during decoding forces `Step N:` openings and decides when to emit `</think>\n<final>\n` and `</final>`.
- **`cot_format_fsm.py`** — A deterministic finite-state machine that validates each output token against the SFT format spec, blocking invalid transitions token-by-token.

---

## 🚄 Speculative Decoding

The speculative decoding system (`mamba3_mlx/speculative/`) accelerates autoregressive sampling **without modifying the model or degrading output quality**.

### How It Works

Instead of generating one token per forward pass (25 ms each), Jacobi decoding **guesses** K−1 future tokens, verifies all K in a single batch forward (92 ms), and accepts correct guesses. The speedup depends on how many consecutive guesses are accepted — the **Average Run Length (ARL)**.

```
AR:    1 tok/forward  →  25 ms/tok
SJD:   K=8, ARL=3.93  →  92/3.93 = 23 ms/tok  →  1.53× speedup
```

### Draft Sources (Training-free)

| Source | File | Principle | ARL Impact |
|--------|------|-----------|------------|
| `SuffixRetriever` | `drafts.py` | Longest-suffix match against a sliding window of past output (Prompt Lookup Decoding) | Long fragments |
| `NGramCache` (runtime) | `ngram_cache.py` | LRU dictionary tracking MRU next-token given N−1 context | Local patterns |
| `CoT NGram` (pre-baked) | `cot_cache.py` | Offline scan of 10,217 training JSONs → phase-aware (think/final) ngram dictionary | **Highest hit rate** |
| `Carry Fallback` | — | Repeat last token when all else fails | Baseline |

### Cache Baking

```bash
# CoT cache (6 seconds, no model needed)
python mamba3_mlx/speculative/bake_cot_caches.py

# Runtime cache (6 minutes, runs model to warm up retriever)
python mamba3_mlx/speculative/bake_cache.py
```

### Results (M2 Pro, bf16)

| Prompt | Config | K | ARL | tps | Speedup |
|--------|--------|---|-----|-----|---------|
| self_awareness | SJD ng+rt+cot | 8 | 3.93 | 57.8 | **1.53×** |
| math_drill | SJD ng+rt | 8 | 3.43 | 61.1 | **1.68×** |
| daily_conversation | SJD ng+rt+cot | 8 | 2.93 | 49.4 | **1.26×** |

The bottleneck is draft hit rate. ARL is currently capped at ~4 due to ngram's 3-token context horizon. Adding a SuffixRetriever to the CoT cache (enabling long-phrase retrieval from training data) is projected to push ARL to 6–8 and speedup to **2.0–2.5×**.

### Run Speculative Decoding

```bash
# SJD with CoT cache (best for self_awareness)
make -C mamba3_mlx sjd-self

# SJD for math drill
make -C mamba3_mlx sjd-math

# SJD for daily conversation
make -C mamba3_mlx sjd-daily

# Full benchmark sweep
make -C mamba3_mlx cot-verify
```

---

## 📎 Citation

If you use Hybrid Mamba-TuckerMoE in your research, please cite:

```bibtex
@article{lai2026hybrid,
  title={Hybrid Mamba-TuckerMoE for On-Device LLM Inference},
  author={Lai, Hung-Wei and Lan, Hsin-An and Hsu, Chun-Ming and Lu, Yu-Han},
  journal={Technical Report},
  year={2026}
}
```
