# Mamba3 MLX Inference - Quick Start Guide

Welcome! This guide shows you how to use Mamba3 MLX inference right away.

---

## ⚡ 30-Second Start

```bash
# Make the script executable (one-time setup)
chmod +x run_mamba.sh

# Run your first inference
./run_mamba.sh "Explain quantum computing in simple terms"
```

That's it! The model will generate a response and show you the speed (tok/s).

---

## 📋 Common Examples

### 1. Basic Usage (Default Settings)
```bash
./run_mamba.sh "Hello, how are you?"
```
- Generates 256 tokens (default)
- Uses bfloat16 precision (fastest on M-series)
- Temperature 0.7 (balanced creativity)

### 2. Longer Generation
```bash
./run_mamba.sh --prompt "Write a short story about" --max-tokens 512
```
- Generates up to 512 tokens
- Great for essays, stories, detailed explanations

### 3. Streaming Output (See Text in Real-Time)
```bash
./run_mamba.sh --stream --prompt "Tell me about machine learning"
```
- Text appears token-by-token
- Perfect for interactive use

### 4. Faster (Quantized 8-bit)
```bash
./run_mamba.sh --quantize 8 --prompt "What is AI?"
```
- 8-bit quantization (fewer bits = faster)
- Speed: ~40-50 tok/s
- Quality: Imperceptible difference

### 5. Benchmark Mode (Speed Test)
```bash
./run_mamba.sh --benchmark --prompt "Test"
```
- Generates 256 tokens and reports detailed metrics
- Shows throughput, per-token latency
- Use this to verify your setup

### 6. Verbose Output (See What's Happening)
```bash
./run_mamba.sh -v --prompt "Hello"
```
- Shows model loading time
- Displays configuration
- Useful for debugging

### 7. Python API (Programmatic Use)
```python
from mamba3_mlx.inference_api import MambaInferenceAPI

# Initialize
api = MambaInferenceAPI(
    checkpoint="checkpoints/latest_sft_cot_model.npz",
    dtype="bf16",
    quantize=None,  # Set to 8 for quantization
    verbose=True,
)

# Generate text
result = api.generate(
    prompt="Explain AI",
    max_tokens=512,
    temperature=0.7,
)

print(result['text'])
print(f"Speed: {result['throughput']:.1f} tok/s")

# Or stream tokens
for token in api.stream_generate("Hello", max_tokens=100):
    print(token, end="", flush=True)
```

---

## 🎯 Parameter Guide

### Shell (run_mamba.sh)

| Option | Default | Description |
|--------|---------|-------------|
| `--prompt TEXT` | "Hello, how are you?" | What to ask the model |
| `--max-tokens N` | 256 | Max tokens to generate (1-2048) |
| `--temperature T` | 0.7 | Randomness (0=deterministic, 1=random) |
| `--quantize BITS` | — | 4 or 8-bit quantization (optional) |
| `--dtype TYPE` | bf16 | Data type: bf16, fp32, fp16 |
| `--stream` | — | Show output in real-time |
| `--benchmark` | — | Run speed test (256 tokens) |
| `-v, --verbose` | — | Show detailed output |
| `-h, --help` | — | Show help message |

### Python API

```python
api = MambaInferenceAPI(
    checkpoint="checkpoints/latest_sft_cot_model.npz",  # Model weights
    tokenizer="checkpoints/tokenizer",                   # Token encoder
    dtype="bf16",                                        # bf16, fp32, fp16
    quantize=None,                                       # 4, 8, or None
    enable_fusion=False,                                 # Metal kernel fusion
    verbose=False,                                       # Detailed logging
)

result = api.generate(
    prompt="Your question",
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    show_progress=False,
)
```

---

## 📊 Performance Expectations

### On M2 Pro 16GB

| Configuration | Speed | Best For |
|---------------|-------|----------|
| **Default (bf16)** | 40-45 tok/s | Balanced |
| **8-bit Quantized** | 50-60 tok/s | Speed-focused |
| **4-bit Quantized** | 65-75 tok/s | Maximum speed |
| **Streaming** | Same speed | Real-time UX |

### Output Metrics

The model reports:
- **Throughput**: Tokens per second (tok/s)
- **Per-token**: Milliseconds per token (ms)
- **Total time**: Wall-clock seconds
- **Tokens generated**: Actual count (may be less than `--max-tokens`)

Example output:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated text here...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Metrics:
  Tokens:      256
  Time:        5.83s
  Speed:       43.9 tok/s
  Per-token:   22.78ms
```

---

## 🔧 Setup & Troubleshooting

### First-Time Setup

1. **Download model checkpoint** (if not already present):
   ```bash
   # Place in checkpoints/ directory
   # File: latest_sft_cot_model.npz (or .pt)
   ```

2. **Check dependencies**:
   ```bash
   python3 -c "import mlx.core; print('✓ MLX installed')"
   python3 -c "from transformers import AutoTokenizer; print('✓ Transformers installed')"
   ```

3. **Make script executable**:
   ```bash
   chmod +x run_mamba.sh
   ```

### Common Issues

#### ❌ "Python not found"
```bash
# Make sure you're in the project directory
cd /path/to/Mamba3-XR
./run_mamba.sh "Hello"
```

#### ❌ "checkpoints directory not found"
```bash
# Create and download model weights
mkdir -p checkpoints
# Download latest_sft_cot_model.npz to checkpoints/
```

#### ❌ "ModuleNotFoundError: No module named 'mlx'"
```bash
# Install MLX
pip install mlx

# Or in virtual environment
.venv/bin/pip install mlx transformers
```

#### ❌ Script output is slow (1-10 tok/s)
- **Expected for first run** (model is loading)
- Subsequent runs are faster
- If persistent, check:
  ```bash
  ./run_mamba.sh -v --benchmark --prompt "Test"
  # Look at "Model loaded in X.XXs"
  # If > 10s, model may be loading from disk
  ```

#### ❌ "Temperature must be between 0 and 1"
```bash
# Use valid range
./run_mamba.sh --temperature 0.7 --prompt "Hello"
```

---

## 💡 Tips & Tricks

### Temperature Control
- **0.0**: Always pick the most likely token (deterministic, repetitive)
- **0.5**: More focused, consistent responses
- **0.7**: Default (balanced)
- **1.0**: More creative, varied responses

### Token Limits
- **Short responses** (code snippets, facts): 128-256 tokens
- **Medium** (paragraphs, explanations): 256-512 tokens
- **Long** (essays, stories): 512-1024 tokens

### Quantization Trade-offs
- **No quantization (default)**: Best quality, slower
- **8-bit**: Good quality, 20-30% faster
- **4-bit**: Acceptable quality, 50% faster

### Combining Options
```bash
# Fast, quantized, streaming
./run_mamba.sh \
  --quantize 8 \
  --stream \
  --prompt "Your prompt" \
  --max-tokens 512
```

---

## 🐍 Python API Examples

### Example 1: Simple Generation
```python
from mamba3_mlx.inference_api import MambaInferenceAPI

api = MambaInferenceAPI()
result = api.generate(
    prompt="What is 2+2?",
    max_tokens=100,
)
print(result['text'])
```

### Example 2: Batch Generation
```python
prompts = [
    "Explain gravity",
    "What is AI?",
    "Tell a joke",
]

for prompt in prompts:
    result = api.generate(prompt, max_tokens=200)
    print(f"Q: {prompt}")
    print(f"A: {result['text']}")
    print(f"Speed: {result['throughput']:.1f} tok/s\n")
```

### Example 3: Streaming with Real-Time Output
```python
import sys

api = MambaInferenceAPI(verbose=False)

print("Q: Explain quantum computing")
print("A: ", end="", flush=True)

for token in api.stream_generate(
    "Explain quantum computing",
    max_tokens=256,
):
    print(token, end="", flush=True)
    sys.stdout.flush()

print("\n")
```

### Example 4: Check Model Status
```python
status = api.get_status()
print(f"Model loaded: {status['model_loaded']}")
print(f"Data type: {status['dtype']}")
print(f"Quantized: {status['quantized']}")
print(f"Vocab size: {status['tokenizer_vocab_size']}")
```

### Example 5: Different Configurations
```python
# Default (balanced)
api = MambaInferenceAPI()

# Quantized (fast)
api_fast = MambaInferenceAPI(quantize=8)

# fp32 (highest quality)
api_quality = MambaInferenceAPI(dtype="fp32")

# With fusion optimization
api_fused = MambaInferenceAPI(enable_fusion=True)
```

---

## 📈 Performance Optimization

### To maximize speed:
1. Use quantization:
   ```bash
   ./run_mamba.sh --quantize 8 --prompt "..."
   ```

2. Reduce token count:
   ```bash
   ./run_mamba.sh --max-tokens 128 --prompt "..."
   ```

3. Use streaming (for UX):
   ```bash
   ./run_mamba.sh --stream --prompt "..."
   ```

### To maximize quality:
1. Don't quantize:
   ```bash
   ./run_mamba.sh --prompt "..."
   ```

2. Use fp32:
   ```bash
   ./run_mamba.sh --dtype fp32 --prompt "..."
   ```

3. Lower temperature for consistency:
   ```bash
   ./run_mamba.sh --temperature 0.3 --prompt "..."
   ```

---

## 🎓 Understanding the Output

### Throughput Calculation
```
Speed (tok/s) = Total tokens generated / Time (seconds)

Example:
256 tokens in 5.83s = 43.9 tok/s
```

### Per-Token Latency
```
Per-token (ms) = Total time (ms) / Total tokens

Example:
5830ms / 256 = 22.78ms per token
```

---

## 📞 Getting Help

### Quick checks:
1. **Model running?** `./run_mamba.sh --help`
2. **Speed OK?** `./run_mamba.sh --benchmark --prompt "Test"`
3. **Quality OK?** Compare outputs: quantized vs unquantized

### Debug options:
```bash
# Verbose output
./run_mamba.sh -v --prompt "Hello"

# Python API debugging
api = MambaInferenceAPI(verbose=True)
result = api.generate("Hello", max_tokens=100)
```

---

## 🚀 Next Steps

1. ✅ Run `./run_mamba.sh "Hello"` to verify setup
2. ✅ Try different `--max-tokens` values
3. ✅ Test quantization: `./run_mamba.sh --quantize 8 --prompt "..."`
4. ✅ Integrate into your application using Python API
5. ✅ Fine-tune temperature for your use case

---

## 📋 Checklist Before Production

- [ ] Model checkpoint present in `checkpoints/`
- [ ] First inference runs successfully
- [ ] Speed is within expected range (40-75 tok/s)
- [ ] Output quality is acceptable
- [ ] Error handling in place (if using Python API)
- [ ] Temperature/max-tokens tuned for your use case

---

**Ready to go!** Start with:
```bash
./run_mamba.sh "Hello, let's test this!"
```

Enjoy fast inference on Apple Silicon! 🎉
