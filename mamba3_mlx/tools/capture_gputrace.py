"""Capture ONE compiled decode step into a .gputrace for Xcode analysis.

Usage:
    MTL_CAPTURE_ENABLED=1 .venv/bin/python3 mamba3_mlx/tools/capture_gputrace.py
    open decode_step.gputrace        # opens in Xcode → Replay

The capture starts AFTER prefill + compile warmup, so the trace contains
exactly one steady-state token: embed → 24 Mamba (fused SSM kernel) →
6 Transformer → head → sampler → window update.
"""
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.utils.config import Mamba3Config, GenerationConfig
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.mamba_block import Mamba3Block
from mamba3_mlx.mlx_model.weights import load_checkpoint
from mamba3_mlx.mlx_model.static_decode import StaticDecoder
from mamba3_mlx.bench_static import build_prompt_ids

OUT = REPO_ROOT / "decode_step.gputrace"
if OUT.exists():
    shutil.rmtree(OUT)        # start_capture refuses to overwrite

tok = Tokenizer.from_file(str(REPO_ROOT / "cot_dataset" / "tokenizer.json"))
cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
model = Mamba3LanguageModel(cfg)
load_checkpoint(model, str(REPO_ROOT / "checkpoints" / "v6" / "latest_sft_cot_model.npz"),
                dtype=mx.bfloat16)
mx.eval(model.parameters())
prompt_ids = build_prompt_ids(tok, "Explain why the sky is blue.")

dec = StaticDecoder(model, metal_fuse=True, quant_moe_bits=8,
                    quant_proj_bits=8, quant_head_bits=8)
gc = GenerationConfig(max_tokens=256, temperature=0.426, top_k=20, top_p=0.981,
                      min_p=0.067, rep_pen=1.146, pres_pen=0.143, freq_pen=0.133, seed=0)
samp_key = (gc.temperature, gc.top_k, gc.top_p, gc.min_p,
            gc.rep_pen, gc.pres_pen, gc.freq_pen)
step_fn = dec._get_step(1, samp_key)

# build static state exactly like generate() does
logits, states = model(mx.array(prompt_ids, dtype=mx.int32)[None, :], states=None)
mx.eval(logits)
S = 512
m_flat, kvs = [], []
for blk, st in zip(model.backbone.layers, states):
    if isinstance(blk, Mamba3Block):
        m_flat += [st["h_prev"], st["prev_input_signal"], st["angles_cum"]]
    else:
        pad = S - st["k"].shape[2]
        kvs.append(mx.pad(st["k"], ((0, 0), (0, 0), (0, pad), (0, 0))))
        kvs.append(mx.pad(st["v"], ((0, 0), (0, 0), (0, pad), (0, 0))))
V = cfg.vocab_size
counts = mx.zeros((V,), dtype=mx.float32)
ring = mx.full((128,), -1, dtype=mx.int32)
ring_pos = mx.zeros((1,), dtype=mx.int32)
write_pos = mx.array([len(prompt_ids)], dtype=mx.int32)
bb = mx.zeros((V,), dtype=mx.float32)
key = mx.random.key(0)
x_tok = mx.zeros((1, 1), dtype=mx.int32)
mx.eval(*m_flat, *kvs, counts, ring, ring_pos, write_pos, bb)

# warmup: pay compile + first-run costs OUTSIDE the capture
for _ in range(3):
    out = step_fn(x_tok, write_pos, m_flat, kvs, counts, ring, ring_pos, bb, key)
    mx.eval(out[0])

print(f"[capture] starting → {OUT}", file=sys.stderr)
mx.metal.start_capture(str(OUT))
out = step_fn(x_tok, write_pos, m_flat, kvs, counts, ring, ring_pos, bb, key)
mx.eval(out[0])
mx.metal.stop_capture()
print(f"[capture] done. Open with:  open {OUT}", file=sys.stderr)
