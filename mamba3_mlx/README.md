# mamba3_mlx

Apple Silicon inference stack for **Hybrid Mamba3 + TuckerMoE** — pure MLX, no PyTorch at runtime.

## Directory layout

```
mamba3_mlx/
├── mlx_model/       Model architecture (hybrid_model, mamba_block, transformer_block, tucker_moe, ops, state_utils)
├── inference/       Generation loop (generator, sampler, cot_splitter)
├── utils/           Shared config dataclass
├── tests/           Test suite
├── scripts/         Shell entry points
├── ui/              Static frontend assets (HTML/JS/CSS — do not edit)
├── run.py           CLI single-turn chat
├── server.py        FastAPI WebSocket chat server
└── server_config.py System prompts + category map (SFT-aligned)
```

## Quick start

```bash
# Chat via CLI
bash mamba3_mlx/scripts/chat.sh "Explain reinforcement learning."

# Start the web chat server (real model)
make serve

# Start the web chat server (mock — no GPU needed)
make serve-mock

# Run tests
make mlx-test

# Throughput benchmark
make mlx-benchmark
```

Open http://localhost:8000 after starting the server.

## Makefile targets

| Target | Description |
|---|---|
| `make serve` | Start chat server with real model |
| `make serve-mock` | Start chat server in mock mode |
| `make mlx-test` | Run inference test suite |
| `make mlx-chat PROMPT="..."` | Single-turn CLI chat |
| `make mlx-benchmark` | Throughput benchmark (prefill + decode) |
| `make mlx-bench` | Original MLX benchmark (inference/ stack) |

Override defaults:
```bash
make serve MLX_CHECKPOINT=path/to/model.npz SERVER_PORT=9000
make serve-mock SERVER_PORT=9000
```

## Server endpoints

| Route | Description |
|---|---|
| `GET /` | Chat UI (HTML with cache-busted JS/CSS) |
| `GET /static/*` | UI assets (no-store) |
| `GET /api/status` | Model load status + timings |
| `GET /api/demo-config` | Categories, system prompts, sampling defaults |
| `WS /ws` | Streaming chat protocol |

## Weight loading

`build_model()` uses `mx.load()` instead of numpy for reading arrays.  
Benchmark on M2 Pro (417M params, bf16):

| Step | Time |
|---|---|
| numpy read all arrays | ~6 300 ms |
| **mx.load** all arrays | **~40 ms** |
| Cast to bf16 + eval | ~2 300 ms |
| **Total (optimized)** | **~2 400 ms** |

## Generation

`stream_generate()` in `inference/generator.py` follows official MLX LLM pattern:
single GPU sync per token, states materialized after each decode step to cap graph growth.

`no_eos_stop=True` in `GenerationConfig` continues past stop tokens — used by the UI's
"EOS No Stop" toggle with the NES progress bar.

## System prompts

`server_config.py` is the single source of truth for the 7 SFT category prompts.
It is also imported by `cot_dataset/export_hf_dataset.py` to keep training and inference aligned.

## CoT format

Training uses `<think>...</think><final>...</final>` inside `<|im_start|>assistant` blocks.
The server injects `<think>\n` at the end of each prompt (reasoning mode) and injects
`<final>\n` via a continuation decode step when `</think>` is detected.

`inference/cot_splitter.py` splits streaming token text into `reasoning` and `token` events
for the WebSocket protocol.
