# Metadata SFT Tiny-LLM Prep

This folder is a clean workspace for preparing metadata / commonsense-reasoning SFT assets for `arnir0/Tiny-LLM`.

Use the project virtual environment at:

```bash
/Users/hungwei/Desktop/Proj/Mamba3-XR/.venv
```

## Layout

- `models/`: Hugging Face model snapshots.
- `datasets/`: downloaded dataset cache / exported JSONL files.
- `reports/`: generated tokenizer comparison reports.
- `scripts/download_assets.py`: downloads the model and CommonsenseQA.
- `scripts/compare_vocab.py`: compares Tiny-LLM vocab against `../inference/tokenizer`.
- `scripts/generate_basic.py`: minimal PyTorch text generation script.

## Current Assets

| Asset | Path |
| --- | --- |
| Tiny-LLM snapshot | `metadata_sft_tiny_llm/models/arnir0__Tiny-LLM` |
| CommonsenseQA train | `metadata_sft_tiny_llm/datasets/commonsense_qa/train.jsonl` |
| CommonsenseQA validation | `metadata_sft_tiny_llm/datasets/commonsense_qa/validation.jsonl` |
| CommonsenseQA test | `metadata_sft_tiny_llm/datasets/commonsense_qa/test.jsonl` |
| Vocab diff report | `metadata_sft_tiny_llm/reports/vocab_diff.json` |

Dataset split sizes:

| Split | Rows |
| --- | ---: |
| train | 9741 |
| validation | 1221 |
| test | 1140 |

## PyTorch Generation

Run basic local generation:

```bash
/Users/hungwei/Desktop/Proj/Mamba3-XR/.venv/bin/python \
  metadata_sft_tiny_llm/scripts/generate_basic.py \
  --summary \
  --prompt "According to all known laws of aviation, there is no way a bee should be able to fly." \
  --max-new-tokens 96
```

Greedy decoding:

```bash
/Users/hungwei/Desktop/Proj/Mamba3-XR/.venv/bin/python \
  metadata_sft_tiny_llm/scripts/generate_basic.py \
  --temperature 0 \
  --prompt "The capital of France is"
```

Use the local 32007 project tokenizer for compatibility checks:

```bash
/Users/hungwei/Desktop/Proj/Mamba3-XR/.venv/bin/python \
  metadata_sft_tiny_llm/scripts/generate_basic.py \
  --tokenizer inference/tokenizer \
  --resize-token-embeddings \
  --summary \
  --prompt "<|im_start|>user\nWhat is common sense reasoning?\n<|im_end|>\n<|im_start|>assistant\n"
```

Important: base Tiny-LLM was trained with `vocab_size = 32000`. If you pass the local 32007 tokenizer, PyTorch must call `resize_token_embeddings(32007)`. The 7 new rows are randomly initialized until SFT trains them.

Minimal Python equivalent:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "metadata_sft_tiny_llm/models/arnir0__Tiny-LLM"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Draft Model Speed Benchmark (15M~60M)

Benchmark speculative draft-model candidates on Apple Silicon (`mps`) or other devices.  
This script instantiates several `LlamaForCausalLM` configs with `vocab_size=32007`, including:

- `15M_baseline`
- `22M_mid`
- `33M_recommended` (the 16-layer / hidden 256 config)
- `44M_deeper`
- `60M_upper`

Run benchmark on `mps`:

```bash
/Users/hungwei/Desktop/Proj/Mamba3-XR/.venv/bin/python \
  metadata_sft_tiny_llm/scripts/benchmark_draft_models.py \
  --device mps \
  --prompt-tokens 256 \
  --max-new-tokens 128 \
  --measure-steps 5
```

Optional faster smoke test:

```bash
/Users/hungwei/Desktop/Proj/Mamba3-XR/.venv/bin/python \
  metadata_sft_tiny_llm/scripts/benchmark_draft_models.py \
  --device mps \
  --measure-steps 2
```

It prints a ranked table by generation throughput (`tok/s`) and saves a JSON report at:

`metadata_sft_tiny_llm/reports/draft_model_benchmark_mps.json`

Note: `flash_attention_2` is not available on `mps`; this benchmark uses PyTorch's default attention backend on Apple GPU.

## Draft Model Speed Benchmark (MLX)

If you want to benchmark on Apple Silicon using MLX backend:

```bash
/Users/hungwei/Desktop/Proj/Mamba3-XR/.venv/bin/python \
  metadata_sft_tiny_llm/scripts/benchmark_draft_models_mlx.py \
  --dtype float16 \
  --prompt-tokens 128 \
  --max-new-tokens 32 \
  --measure-steps 3
```

JSON output:

`metadata_sft_tiny_llm/reports/draft_model_benchmark_mlx.json`

Note: this MLX script is a Llama-style relative benchmark for model-size comparison. It is not a bitwise-equivalent replacement of Hugging Face `generate()` (KV cache and GQA implementation details can differ).

## Model Architecture

The downloaded Tiny-LLM checkpoint is a Hugging Face `LlamaForCausalLM` model. It is a decoder-only causal language model, not a Mamba/SSM model.

| Field | Value |
| --- | --- |
| architecture | `LlamaForCausalLM` |
| model_type | `llama` |
| layers | 1 |
| hidden size | 192 |
| FFN intermediate size | 1024 |
| activation | `silu` |
| attention heads | 2 |
| key/value heads | 1 |
| attention type | causal self-attention with grouped-query / multi-query KV heads |
| position encoding | RoPE |
| max context | 1024 |
| base vocab size | 32000 |
| dtype in config | `float32` |
| `use_cache` | `true` |

Attention details:

- It uses standard autoregressive causal masking, so each token can only attend to itself and earlier tokens.
- `num_attention_heads = 2` and `num_key_value_heads = 1`, meaning the two query heads share one KV head group. In practice this is grouped-query attention at the config level, equivalent to multi-query KV sharing for this tiny 2Q/1KV setup.
- Position information comes from RoPE, with no configured `rope_scaling`.
- During generation, `use_cache = true` enables KV cache reuse so PyTorch/Transformers does not recompute all prior keys and values every new token.

## Expected Local Tokenizer Additions

The current project tokenizer has 7 added tokens on top of the 32,000-token base:

| Token | ID |
| --- | ---: |
| `<|im_start|>` | 32000 |
| `<|im_end|>` | 32001 |
| `<think>` | 32002 |
| `</think>` | 32003 |
| `<final>` | 32004 |
| `</final>` | 32005 |
| `[PAD]` | 32006 |

## Commands

```bash
/Users/hungwei/Desktop/Proj/Mamba3-XR/.venv/bin/python metadata_sft_tiny_llm/scripts/download_assets.py
/Users/hungwei/Desktop/Proj/Mamba3-XR/.venv/bin/python metadata_sft_tiny_llm/scripts/compare_vocab.py --base metadata_sft_tiny_llm/models/arnir0__Tiny-LLM --target inference/tokenizer --out metadata_sft_tiny_llm/reports/vocab_diff.json
/Users/hungwei/Desktop/Proj/Mamba3-XR/.venv/bin/python metadata_sft_tiny_llm/scripts/generate_basic.py --summary
```
