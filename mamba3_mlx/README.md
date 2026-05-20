# mamba3_mlx — Native MLX Inference Stack

Pure-MLX inference pipeline for the **Hybrid Mamba-TuckerMoE** model (2.4B dense-equivalent capacity, 417M parameters). No Triton, no PyTorch at runtime — runs entirely on Apple Silicon via MLX unified memory.

---

## Requirements

- macOS 13.3+ (Apple Silicon)
- Python 3.10+
- `mlx`, `tokenizers`

```bash
pip install mlx tokenizers
```

Checkpoint and tokenizer live at repo root:

```
checkpoints/latest_sft_cot_model.npz   # model weights (bf16)
cot_dataset/tokenizer.json             # BPE tokenizer (vocab 32 007)
```

---

## Quick start

```bash
cd mamba3_mlx

make              # "who are you?" · self_awareness mode · streaming
make help         # show all targets and variables
```

Override anything inline:

```bash
make PROMPT="Explain quantum entanglement" TEMP=0.0 MAX_TOK=300
make emotion PROMPT="I feel stuck and overwhelmed"
make deep    PROMPT="Compare SSM vs Transformer" MAX_TOK=512 STREAM=0
```

---

## Directory structure

```
mamba3_mlx/
├── run.py                      # CLI entry point
├── Makefile                    # quick-launch shortcuts
│
├── mlx_model/
│   ├── ops.py                  # primitives: scaled_tanh, silu, softplus,
│   │                           #   RMSNorm, LayerScale, apply_rope
│   ├── tucker_moe.py           # TuckerMoE — vectorised fancy-gather expert dispatch
│   ├── mamba_block.py          # Mamba3Block — chunk scan (prefill) + recurrence (decode)
│   ├── transformer_block.py    # TransformerBlock — GQA attention + MoE FFN + KV cache
│   ├── hybrid_model.py         # TrueHybridMamba + Mamba3LanguageModel
│   └── weights.py              # .npz → model loader
│
├── inference/
│   ├── sampler.py              # temperature / top-k / top-p / min-p / penalties
│   └── generator.py           # prefill / decode loop, streaming callback
│
└── utils/
    ├── config.py               # Mamba3Config, GenerationConfig dataclasses
    └── system_prompts.py       # 7 SFT-category system prompts + resolve helper
```

---

## Model architecture

| Parameter | Value |
|-----------|-------|
| d_model | 768 |
| Total blocks | 30 (4 Mamba + 1 Transformer) × 6 |
| Mamba heads / d_head | 24 / 64 |
| SSM state (d_state) | 64 |
| MIMO rank (R) | 4 |
| MoE experts / top-k | 8 / 2 |
| Tucker ranks (r1/r2/r3) | 32 / 512 / 256 |
| Transformer KV heads | 4 (GQA) |
| Vocabulary | 32 007 |
| Checkpoint size | ~417M params (bf16 ≈ 834 MB) |

---

## CLI — `run.py`

```
python run.py [options]
```

### Core options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt TEXT` | `"What is the capital of France?"` | User message |
| `--mode MODE` | *(none)* | System prompt category — overrides `--system` |
| `--system TEXT` | generic helpful assistant | Custom system prompt |
| `--max_tokens N` | 256 | Maximum tokens to generate |
| `--stream` | off | Print tokens as they arrive |
| `--no-seed-think` | off | Skip pre-seeding `<think>\n` after assistant tag |
| `--raw-prompt` | off | Pass `--prompt` verbatim (no ChatML wrap) |

### Sampling options

| Flag | Default | Description |
|------|---------|-------------|
| `--temp F` | 0.8 | Temperature (0 = greedy argmax) |
| `--top_k N` | 40 | Keep top-k logits (0 = disabled) |
| `--top_p F` | 0.9 | Nucleus sampling threshold |
| `--min_p F` | 0.05 | Min-p filter (relative to max prob) |
| `--rep_pen F` | 1.1 | Repetition penalty |
| `--freq_pen F` | 0.02 | Frequency penalty (OpenAI-style) |
| `--pres_pen F` | 0.0 | Presence penalty |
| `--repeat_last_n N` | 64 | Token window for penalties |
| `--seed N` | 0 | RNG seed |

### Hardware / precision options

| Flag | Default | Description |
|------|---------|-------------|
| `--dtype {fp32,bf16,fp16}` | `bf16` | Weight dtype |
| `--kv_dtype {auto,...}` | `auto` | KV-cache dtype (`auto` = same as `--dtype`) |

### Path overrides

| Flag | Default |
|------|---------|
| `--model_path PATH` | `checkpoints/latest_sft_cot_model.npz` |
| `--tokenizer_path PATH` | `cot_dataset/tokenizer.json` |

---

## Makefile targets

Run from inside `mamba3_mlx/`:

| Target | System prompt | Default prompt |
|--------|--------------|----------------|
| `make` / `make run` | `self_awareness` | `who are you?` |
| `make self` | `self_awareness` | `who are you?` |
| `make emotion` | `emotion` | `who are you?` |
| `make email` | `summarize_email` | `who are you?` |
| `make movie` | `movie_intro` | `who are you?` |
| `make daily` | `daily_conversation` | `who are you?` |
| `make syscall` | `system_call` | `who are you?` |
| `make deep` | `deep_dive` | `who are you?` |
| `make default` | *(none)* | `who are you?` |

Makefile variables (all optional):

| Variable | Default | Example |
|----------|---------|---------|
| `PROMPT` | `who are you?` | `PROMPT="Tell me a joke"` |
| `MODE` | `self_awareness` | `MODE=emotion` |
| `TEMP` | `0.1` | `TEMP=0.0` |
| `MAX_TOK` | `256` | `MAX_TOK=512` |
| `SEED` | `0` | `SEED=42` |
| `STREAM` | `1` | `STREAM=0` |
| `NO_THINK` | `0` | `NO_THINK=1` |

---

## System prompt modes

Seven modes correspond directly to the SFT-CoT training categories. Pass via `--mode` or the Makefile target.

| Mode key | Alias | Description |
|----------|-------|-------------|
| `emotion` | `emotion` | Calm, concrete distress reframing — no clichés |
| `self_awareness` | `self` | Strict architectural self-description (Mamba-TuckerMoE, offline, iPhone) |
| `summarize_email` | `email` | Conclusion-first structured output |
| `movie_intro` | `movie` | Structured film analysis — premise / theme / craft |
| `daily_conversation` | `daily` | Practical everyday answers, explicit assumptions |
| `system_call` | `syscall` | Emits `[CALL: tool {args}]` when tool invocation needed |
| `deep_dive` | `deep` | Long-form analysis — problem model / causal factors / trade-offs |

---

## ChatML output format

Every response follows the training format exactly:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<think>
{chain-of-thought reasoning}
</think>
<final>
{answer}
</final><|im_end|>
```

`<think>\n` is pre-seeded in the prompt by default so the model enters reasoning mode immediately. Use `--no-seed-think` to disable.

---

## Python API

### Load model

```python
import mlx.core as mx
from mamba3_mlx.utils.config import Mamba3Config
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint

cfg   = Mamba3Config()
model = Mamba3LanguageModel(cfg)
load_checkpoint(model, "checkpoints/latest_sft_cot_model.npz", dtype=mx.bfloat16)
mx.eval(model.parameters())
```

### Tokenise

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("cot_dataset/tokenizer.json")
ids = [tok.token_to_id("<s>")] + tok.encode(text, add_special_tokens=False).ids
```

### Generate (high-level)

```python
from mamba3_mlx.utils.config import GenerationConfig
from mamba3_mlx.inference.generator import generate

gen_cfg = GenerationConfig(
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    min_p=0.05,
    rep_pen=1.1,
    freq_pen=0.02,
    seed=0,
)

stop_ids = [tok.token_to_id("<|im_end|>"), tok.token_to_id("</s>")]

out_ids = generate(model, prompt_ids, gen_cfg, stop_token_ids=stop_ids)
print(tok.decode(out_ids, skip_special_tokens=False))
```

### Generate with streaming

```python
def on_token(tid):
    # Called after each token is sampled (before the next decode step)
    print(tok.decode([tid], skip_special_tokens=False), end="", flush=True)

out_ids = generate(model, prompt_ids, gen_cfg,
                   stop_token_ids=stop_ids, on_token=on_token)
```

### Prefill / decode manually

```python
import mlx.core as mx
from mamba3_mlx.inference.generator import prefill, decode_step

last_logits, states = prefill(model, prompt_ids)
mx.eval(last_logits)

for _ in range(max_steps):
    token = int(mx.argmax(last_logits).item())   # or sample
    last_logits, states = decode_step(model, token, states)
    mx.eval(last_logits)
```

### System prompt helper

```python
from mamba3_mlx.utils.system_prompts import resolve_system_prompt

sys_prompt = resolve_system_prompt("self", fallback="You are a helpful assistant.")
# also accepts: "emotion", "email", "movie", "daily", "syscall", "deep"
# and full names: "self_awareness", "summarize_email", ...
```

### Build ChatML prompt

```python
from mamba3_mlx.run import build_chatml_prompt

ids, text = build_chatml_prompt(
    tokenizer=tok,
    system_prompt=sys_prompt,
    user_msg="who are you?",
    seed_think=True,          # pre-seeds <think>\n
)
```

---

## Performance (M2 Pro 16 GB, bf16)

| Stage | Throughput |
|-------|-----------|
| Checkpoint load | ~6.5 s cold |
| Prefill | ~3 800 tok/s |
| Decode | ~20–22 tok/s |

---

## Known limitations

- Decode quality degrades at high temperature due to checkpoint training scale (~417M params).
- `--full-decode-compile` flag is exposed but not yet wired — eager decode only.
- `--speculative` flag is scaffolded but not implemented in this stack.
- Chunk scan uses a dense `(Lc × Lc)` lower-triangular matrix per chunk (`chunk_size=64`); memory is `O(Lc² × H)` per forward pass, fine for sequences up to ~4K.
