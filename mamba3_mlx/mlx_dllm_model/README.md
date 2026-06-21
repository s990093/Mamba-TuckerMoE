# mlx_dllm_model — diffusion-LLM port (additive)

MLX port of `sft_cot_bundle/DLLM_MLX_PORT.md` layered on top of
`mamba3_mlx/mlx_model/` **without modifying it**. The AR `Mamba3Block` and
`TuckerMoE` are reused byte-for-byte; only the 4 documented changes are added.

| # | change | where |
|---|--------|-------|
| ① | `[MASK]` token, vocab 32007 → **32008** | [config.py](config.py), `DLLMModel.init_mask_embedding` |
| ② | bidirectional attention (`mask=None`) | [bidirectional_block.py](bidirectional_block.py) |
| ③ | masked-CE (1/t weighted) training loss | [loss.py](loss.py) *(training only)* |
| ④ | iterative unmasking generation | [generate.py](generate.py) |

Mamba layers stay **unidirectional** (partial-bidirectional design, §0).

## Absorbing-[MASK] diffusion
The canvas starts as `prompt + [MASK]*G` and positions are **filled in directly**
(absorbing state): committed tokens are never re-noised; non-committed stay
`[MASK]`. `[MASK]` is input-only — it is banned (`-inf`) from the output logits
so the model never emits it. (`make dllm-canvas` visualises the fill.)

## DiffusionGemma-inspired optimizations (`samplers.py`, `self_conditioning.py`)
Ported from `transformers/models/diffusion_gemma`, adapted to absorbing
diffusion (DiffusionGemma uses random-token renoise; we keep `[MASK]`):

| idea | port | inference-only? |
|------|------|:--:|
| EntropyBoundSampler — commit lowest-entropy positions while joint-MI ≤ bound | `entropy_bound_select` (default sampler) | ✅ |
| LinearTemperatureSchedule — t_max early → t_min late | `linear_temperature` | ✅ |
| StableAndConfidentStoppingCriteria — adaptive early stop | `StableConfidentStop` (`--adaptive-stop`) | ✅ |
| Self-Conditioning — feed prev-step logits as a soft-embedding (gated MLP) | `DLLMSelfConditioning` (`--self-cond`) | ❌ needs training |

`final_logit_softcapping=30.0` in DiffusionGemma matches this stack's
`scaled_tanh(logits, 30.0)` head — so the soft-cap is already consistent.
The entropy sampler commits a **data-adaptive** number of tokens per step
(bursts when confident), unlike the fixed cosine schedule.

## Status: UNTRAINED — no checkpoint loaded
The real dLLM weights are still training, so the model self-initialises with
random weights (`build_random_dllm`). Output text is gibberish; what is
verified is that the inference, high-performance, and validation paths run, are
shape-correct, numerically faithful, and fast (batch=1).

## Files
- `dllm_model.py` — `DLLMModel` + `build_random_dllm` (optional self-cond hook)
- `bidirectional_block.py` — change ② bidirectional transformer block
- `generate.py` — `diffusion_generate` (entropy/cosine sampler, temp schedule,
  adaptive stop, self-cond) + `iterative_unmask` (cosine baseline)
- `samplers.py` — entropy sampler / temp schedule / adaptive stop (DiffusionGemma)
- `self_conditioning.py` — gated-MLP cross-step self-conditioning (DiffusionGemma)
- `static_dllm.py` — `StaticDLLM`: one `mx.compile`'d fixed-shape `(1, P+G)`
  forward reused across all `T` iterations. The Mamba chunk-scan prefill
  already runs on the Metal scan kernel (`mlx_model/scan_metal.py`) and the
  TuckerMoE G-cache is reused via `precompute()`.
- `validate.py` — §驗證 (A) parity (B) fixed-ratio recon (C) iterative recon
- `loss.py` — §③ masked-CE for the future training move

## Performance (batch=1, M2 Pro, untrained 417M, G=32, T=12)
Profiling showed the cost is **100% the forward** (per-step selection ≈1.5ms).
Two facts drive the optimization:
- the (1,L) forward is **dispatch-bound** below L≈64 (~62ms floor, L=8/32/64 all ~62ms);
- a **chunk-size=64 cliff**: L≤64 = 1 chunk (~62ms), L=65..128 = 2 chunks (~123ms).
  A naive `P+G=66` lands just over 64 → 2 chunks → 108ms/step.

| path | tok/s | vs eager | note |
|------|------:|:--:|------|
| eager-full | 16 | 1.0× | full (P+G) forward, per step |
| static-full (`dllm-s`) | 24 | 1.5× | `mx.compile` the (P+G) forward |
| **prefix-cache (`dllm-fast`)** | **43** | **2.7×** | encode prompt once, denoise G-only (1 chunk) |

**Prefix cache** = DiffusionGemma's encoder/decoder split: `model.encode_prefix`
runs the prompt once (Mamba state — already exact since Mamba is
unidirectional — + TF KV cache); `model.denoise` then forwards only the G-token
canvas each step, attending to the cached prompt KV. It avoids re-reading the
prompt T times AND keeps the canvas to one chunk (G≤64). Eager==compiled
bit-exact. **Caveat:** it uses prefix-LM attention (prompt does not attend to
the canvas), unlike the full-bidirectional `dllm-s`/`dllm` path — match it at
training time for best quality. Other levers: fewer steps T (entropy sampler +
`--adaptive-stop` already cut forwards), and weight quant (future).

## CLI / Make
```bash
python mamba3_mlx/dllm_infer.py --mode generate --static --prompt "Who are you?"
python mamba3_mlx/dllm_infer.py --mode bench
python mamba3_mlx/dllm_infer.py --mode validate
# or:
make -C mamba3_mlx dllm-s        # static generate
make -C mamba3_mlx dllm-bench
make -C mamba3_mlx dllm-validate
```

## When training finishes — loading weights
The architecture matches the AR model except for the `[MASK]` row and the
bidirectional flag, so reuse `mlx_model/weights.py::load_checkpoint` against a
`DLLMModel`:
1. If the checkpoint embed is already `(32008, D)` (CUDA dLLM weights): load
   directly, then `model.tie_weights()` — **skip** `init_mask_embedding()`.
2. If only `(32007, D)` AR weights exist: load, then `init_mask_embedding()` to
   add the MASK row, then `tie_weights()`.
3. Call `model.precompute()` once after loading (rebuilds the TuckerMoE
   G-cache). Drop `init_experts_random()` — that is a no-checkpoint demo helper.

Then point the CLI at the checkpoint and run §驗證(B)/(C) — accuracy should
rise from ~chance toward ~1.0 on overfit data (see DLLM_MLX_PORT.md §驗證
判讀), and (A) should still hold.
