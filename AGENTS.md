# AGENTS.md

Guidance for working in this repo. Omitted = default conventions apply.

## Two disjoint stacks

- **Training**: `pre-train/train.py` (single-file, PyTorch + Triton + Accelerate)
- **Inference**: `mamba3_mlx/` (modular, MLX-only on Apple Silicon)

These share no runtime dependencies. Know which one you are in.

## Developer commands

All from repo root, unless noted. The Makefile auto-detects `.venv/bin/python3`.

**Inference (MLX):**
```bash
make -C mamba3_mlx                                         # self_awareness mode
make -C mamba3_mlx emotion PROMPT="I feel stuck"
make -C mamba3_mlx deep PROMPT="..." MAX_TOK=512 COMPILE=1 # compiled decode
make -C mamba3_mlx chat                                    # WebSocket server :7860
make -C mamba3_mlx chat-mock                               # UI only, no weights
make -C mamba3_mlx chat-smoke PROMPT="..."                 # boot+assert+shutdown
make -C mamba3_mlx chat-kill                               # kill server on :$CHAT_PORT
make -C mamba3_mlx cot-demo PROMPT="..."                   # CoT middleware one-shot
make -C mamba3_mlx cot-verify                              # 5-trial sweep
```

**Training:**
```bash
python pre-train/train.py            # single-file, all hyperparams at bottom of file
```

## Checkpoint & tokenizer

- Weights: `checkpoints/latest_sft_cot_model.npz` (bf16, ~834 MB)
- Tokenizer: `cot_dataset/tokenizer.json` (vocab 32,007 — **frozen, do not modify**)
- First `.npz` load auto-creates a `.mlx_bf16.npz` sidecar for mmap-fast loads
- `.pt` / `.npz` / `*.safetensors` are gitignored — weights live locally only

## Architecture constants (hardcoded in `pre-train/train.py` bottom section)

| Param | Value |
|-------|-------|
| D_MODEL | 768 |
| D_STATE | 64 |
| NUM_LAYERS | 6 (30 blocks: 4 Mamba + 1 Transformer per macro layer) |
| KMOE_NUM_EXPERTS | 8, TOP_K=2 |
| Tucker ranks | r1=32, r2=512, r3=256 |
| Context window | 2,048 tokens hard limit |
| Vocab | 32,007 (frozen) |

## SFT dataset rules (see `cot_dataset/GUIDE.md` + `cot_dataset/SFT_FORMAT.md`)

- **Never** write `<think>`, `</think>`, `<final>`, `</final>` in raw JSON files
- English only, no contractions, zero spelling errors
- 3-5 CoT steps, each `Step N:`, separated by `\n`
- Token budget: 512-768 for most, ≤128 for math drill
- JSON files and .txt files in `cot_dataset/` are gitignored — verify with `python3 -m json.tool`

## Inference architecture (`mamba3_mlx/`)

- Entry point: `mamba3_mlx/run.py` → calls `mamba3_mlx/inference/generator.py`
- Model: `mamba3_mlx/mlx_model/` — `hybrid_model.py`, `mamba_block.py`, `transformer_block.py`, `tucker_moe.py`, `weights.py`
- CoT middleware: `mamba3_mlx/mv/cot_middleware.py` + `cot_format_fsm.py`
- Chat server: `python -m mamba3_mlx.chat_demo` (FastAPI WebSocket, default port 7860)
- Speculative decoding: `mamba3_mlx/speculative/` (Jacobi, ngram cache)
- Sidecar conversion: `mamba3_mlx/tools/convert_sidecar.py`

## Prefill vs decode

Two different code paths. Prefill processes the full prompt in one forward pass; decode loops token-by-token with KV/Mamba state caching. Use `COMPILE=1` for decode JIT (requires warmup steps).

## Smoke test
```bash
make -C mamba3_mlx PROMPT="Hello" MAX_TOK=50
```

## Stale docs to ignore

The root `README.md` references an `inference/` directory that no longer exists — inference lives in `mamba3_mlx/`. The `CLAUDE.md` at root references `inference/lib/mlx_hybrid_infer.py` which has been superseded by `mamba3_mlx/`. Trust executable sources (Makefile, actual dirs) over prose docs.
