# Streaming Output Guide for Speculative Decoding

## Problem: No Streaming in `run_sjd_best.sh`

The `run_sjd_best.sh` script uses `run_jacobi_sampling.py`, which **does not support streaming output**. It only prints the final result after generation completes.

## Solution: Use `run_jacobi.py` for Streaming

### Quick Command (Streaming with Best Speed)

```bash
python -m mamba3_mlx.speculative.run_jacobi \
    --prompt "Who are you?" \
    --mode self_awareness \
    --K 16 \
    --use_ngram \
    --use_retrieval \
    --max_tokens 512 \
    --stream \
    --dtype bf16
```

### Comparison: Streaming vs Non-Streaming

| Script | Streaming | Use Case |
|--------|-----------|----------|
| `run_jacobi.py` | ✅ `--stream` flag | **Interactive demos, real-time output** |
| `run_jacobi_sampling.py` | ❌ No streaming | Benchmarking, quality checks |
| `run_sjd_best.sh` | ❌ No streaming | Automated benchmarks |

### Why Your Command Didn't Show Output

```bash
# This script doesn't support streaming:
bash mamba3_mlx/speculative/run_sjd_best.sh "who are you?"
# ❌ Output only appears at the end
```

**Reason:** `run_sjd_best.sh` → `run_jacobi_sampling.py` → no `on_token` callback

### Streaming-Enabled Commands

#### 1. Basic Streaming (Greedy Jacobi)
```bash
python -m mamba3_mlx.speculative.run_jacobi \
    --prompt "Explain quantum entanglement" \
    --K 16 \
    --use_ngram \
    --stream \
    --max_tokens 256
```

#### 2. Streaming with All Optimizations
```bash
python -m mamba3_mlx.speculative.run_jacobi \
    --prompt "Who are you?" \
    --mode self_awareness \
    --K 16 \
    --use_ngram \
    --ngram_n 4 \
    --use_retrieval \
    --stream \
    --verbose \
    --max_tokens 512 \
    --dtype bf16
```

#### 3. Streaming with Adaptive K
```bash
python -m mamba3_mlx.speculative.run_jacobi \
    --prompt "Solve 2x+3=11 step by step" \
    --K 12 \
    --adaptive_K \
    --K_min 4 \
    --K_max 16 \
    --use_ngram \
    --stream \
    --max_tokens 256
```

### Key Flags for Streaming

| Flag | Effect |
|------|--------|
| `--stream` | **Enable real-time token output** |
| `--show-special` | Show special tokens (`<think>`, `</think>`, etc.) |
| `--verbose` | Show per-round diagnostics (ARL, acceptance rate) |
| `--K 16` | Best speed/quality balance |
| `--use_ngram` | Enable n-gram draft cache |
| `--use_retrieval` | Enable suffix retrieval (PLD) |

### Expected Streaming Behavior

```bash
# With --stream:
$ python -m mamba3_mlx.speculative.run_jacobi --prompt "Who are you?" --stream --K 16

[load] tokenizer: ...
[load] model: ...
===== PROMPT =====
<|im_start|>system
...
===== ASSISTANT OUTPUT =====
<think>
Step 1: **Identify context** — User asks about identity...
Step 2: **Recall architecture** — I am a Mamba3-TuckerMoE...
</think>

<final>
I am a 550M-parameter hybrid language model...
</final>
# ↑ Tokens appear in real-time as they're generated
```

### Performance Comparison

| Mode | Streaming | Speed | Use Case |
|------|-----------|-------|----------|
| **Greedy Jacobi** (`run_jacobi.py`) | ✅ Yes | ~35 tok/s | **Demos, interactive use** |
| **SJD Sampling** (`run_jacobi_sampling.py`) | ❌ No | ~50 tok/s | Benchmarks, quality checks |
| **Standard AR** (`run.py`) | ✅ Yes | ~18 tok/s | Baseline comparison |

### Why SJD Doesn't Support Streaming (Yet)

**Technical reason:** SJD uses probabilistic acceptance with rejection-resampling. When a draft is rejected, the decoder must:
1. Resample from the residual distribution
2. Stop the current round
3. Extract state at the rejection point

This makes per-token callbacks complex because:
- Tokens aren't emitted in strict left-to-right order
- Rejection can invalidate previously "accepted" tokens in the same round
- The final emitted sequence is only known after the round completes

**Workaround:** Use greedy Jacobi (`run_jacobi.py`) for streaming demos. It's still 1.5-2× faster than AR and supports real-time output.

### Makefile Shortcuts (if available)

```bash
# Check if Makefile has streaming targets:
cd mamba3_mlx
make help | grep -i stream

# If available:
make jacobi-stream PROMPT="Who are you?" K=16
```

### Summary

**For streaming output, use:**
```bash
python -m mamba3_mlx.speculative.run_jacobi \
    --prompt "Your question" \
    --K 16 \
    --use_ngram \
    --use_retrieval \
    --stream \
    --max_tokens 512
```

**Not for streaming:**
- ❌ `run_sjd_best.sh`
- ❌ `run_jacobi_sampling.py`
- ❌ `run_sjd_warm.py`

These are benchmarking tools that prioritize measurement accuracy over real-time output.
