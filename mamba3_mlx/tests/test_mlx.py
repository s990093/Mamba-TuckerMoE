#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick sanity-check suite for the MLX Mamba3 inference stack.
Run from repo root:  python mamba3_mlx/test_mlx.py
"""
import sys, time, numpy as np
sys.path.insert(0, ".")

import mlx.core as mx
from mamba3_mlx.mlx_model.hybrid_model import build_model, _remap_key
from mamba3_mlx.inference.generator import prefill, decode_step, generate, wrap_prompt_chatml
from mamba3_mlx.inference.sampler import sample_token, apply_repetition_penalty
from mamba3_mlx.utils.config import GenerationConfig

CHECKPOINT = "checkpoints/latest_sft_cot_model.npz"
TOKENIZER  = "checkpoints/tokenizer"

PASS, FAIL = "PASS", "FAIL"


def check(condition, name):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    return condition


def test_key_remapping():
    print("=== Key Remapping ===")
    cases = {
        "model_embed.weight":                               "embed.weight",
        "model_norm.weight":                                "norm.weight",
        "model_backbone.layers.0.block.in_proj.weight":    "backbone.layers.0.in_proj.weight",
        "model_backbone.layers.4.block.ls_attn.gamma":     "backbone.layers.4.ls_attn.gamma",
        "model_backbone.layers.4.block.ffn.gate_proj.U_in": "backbone.layers.4.ffn.gate_proj.U_in",
    }
    ok = True
    for pt_key, expected in cases.items():
        result = _remap_key(pt_key)
        ok &= check(result == expected, f"{pt_key} → {result}")
    return ok


def test_weight_loading(model):
    print("=== Weight Loading ===")
    data = np.load(CHECKPOINT, allow_pickle=True)
    checks = {
        "embed.weight": ("model_embed.weight", model.embed.weight),
        "norm.weight":  ("model_norm.weight",  model.norm.weight),
        "layers.0.theta_log": (
            "model_backbone.layers.0.block.theta_log",
            model.backbone.layers[0].theta_log,
        ),
        "layers.4.ls_attn.gamma": (
            "model_backbone.layers.4.block.ls_attn.gamma",
            model.backbone.layers[4].ls_attn.gamma,
        ),
    }
    ok = True
    for desc, (pt_key, mlx_param) in checks.items():
        pt_val   = data[pt_key].astype(np.float32)
        mlx_val  = np.array(mlx_param.astype(mx.float32))
        max_diff = float(np.abs(pt_val - mlx_val).max())
        # bf16 has ~3 significant digits; allow up to 1e-2 relative-ish error
        tol = 1e-2
        ok &= check(max_diff < tol, f"{desc}: max_diff={max_diff:.2e} (tol {tol})")
    return ok


def test_forward(model):
    print("=== Forward Pass ===")
    ok = True
    x = mx.array([[1, 100, 200, 300, 400]])
    logits, _, new_m, new_kv = model(x)
    mx.eval(logits)
    ok &= check(logits.shape == (1, 5, model.vocab_size),
                f"logits shape {logits.shape}")
    ok &= check(len(new_m) == 24, f"mamba_states count {len(new_m)}")
    ok &= check(len(new_kv) == 6,  f"kv_caches count {len(new_kv)}")
    ok &= check(
        all(s["h"].shape == (1, model.config.n_heads, model.config.d_state, model.config.d_head)
            for s in new_m),
        "mamba state h shapes",
    )
    return ok


def test_prefill_decode(model):
    print("=== Prefill / Decode ===")
    ok = True
    prompt = [1, 100, 200, 300, 500]
    logits_pre, m, kv = prefill(model, mx.array([prompt]))
    ok &= check(logits_pre.shape == (1, model.vocab_size), f"prefill logits {logits_pre.shape}")

    tok = int(mx.argmax(logits_pre[0]).item())
    T_before = kv[0]["k"].shape[2]

    logits_dec, m2, kv2 = decode_step(model, tok, m, kv)
    ok &= check(logits_dec.shape == (1, model.vocab_size), f"decode logits {logits_dec.shape}")
    ok &= check(kv2[0]["k"].shape[2] == T_before + 1,
                f"KV cache grew from {T_before} to {kv2[0]['k'].shape[2]}")

    # Mamba state should have same shape but differ in values
    ok &= check(
        m2[0]["h"].shape == m[0]["h"].shape,
        f"mamba h shape preserved {m2[0]['h'].shape}",
    )
    return ok


def test_sampler():
    print("=== Sampler ===")
    ok = True
    logits = mx.array([0.1, 0.2, 5.0, 0.05, -1.0])

    # Temperature=0 → greedy
    t0 = sample_token(logits, temperature=0.0)
    ok &= check(t0 == 2, f"greedy argmax = {t0}")

    # Rep penalty should suppress token 2
    logits_pen = apply_repetition_penalty(logits, [2, 2], penalty=10.0, window=64)
    ok &= check(float(logits_pen[2]) < float(logits[2]), "rep penalty applied")

    # Top-k should eliminate low-prob tokens
    toks = [sample_token(logits, temperature=1.0, top_k=2) for _ in range(50)]
    ok &= check(all(t in {1, 2} for t in toks), f"top-k=2 stays in {{1,2}}")

    return ok


def test_generation(model):
    print("=== End-to-end Generation ===")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as e:
        print(f"  [SKIP] tokenizer not available: {e}")
        return True

    stop_ids = [tok.convert_tokens_to_ids(t) for t in ["<|im_end|>", "</s>"]
                if isinstance(tok.convert_tokens_to_ids(t), int)]

    prompt_text = wrap_prompt_chatml("Hello!")
    prompt_ids  = tok.encode(prompt_text)
    gen_cfg = GenerationConfig(max_new_tokens=20, temperature=0.7, top_k=40, stop_token_ids=stop_ids)

    tokens = []
    for token_id in generate(model, prompt_ids, gen_cfg):
        tokens.append(token_id)
        if token_id in stop_ids:
            break

    ok = check(len(tokens) > 0, f"generated {len(tokens)} tokens")
    print(f"  decoded: {repr(tok.decode(tokens, skip_special_tokens=False)[:60])}")
    return ok


def test_throughput(model):
    from mamba3_mlx.inference.generator import _eval_states
    print("=== Throughput Benchmark ===")
    prompt = [1, 100, 200, 300]
    logits_pre, m, kv = prefill(model, mx.array([prompt]))
    tok = int(mx.argmax(logits_pre[0]).item())
    _eval_states(m, kv)

    for _ in range(3):   # warm up (fully synced)
        logits, m, kv = decode_step(model, tok, m, kv)
        tok = int(mx.argmax(logits[0]).item())
        _eval_states(m, kv)

    N = 30
    t0 = time.perf_counter()
    for _ in range(N):
        logits, m, kv = decode_step(model, tok, m, kv)
        tok = int(mx.argmax(logits[0]).item())   # eval logits
        _eval_states(m, kv)                      # eval states
    elapsed = time.perf_counter() - t0
    tps = N / elapsed
    print(f"  {N} decode steps in {elapsed:.2f}s → {tps:.1f} tok/s")
    return check(tps > 5.0, f"tok/s > 5 threshold (got {tps:.1f})")


def main():
    results = []

    # Key remapping (no model needed)
    results.append(("key_remapping", test_key_remapping()))

    # Load model once for all remaining tests
    print("\nLoading model...")
    model = build_model(CHECKPOINT, dtype=mx.bfloat16)
    print("Model loaded.\n")

    results.append(("weight_loading",  test_weight_loading(model)))
    results.append(("forward_pass",    test_forward(model)))
    results.append(("prefill_decode",  test_prefill_decode(model)))
    results.append(("sampler",         test_sampler()))
    results.append(("generation",      test_generation(model)))
    results.append(("throughput",      test_throughput(model)))

    print("\n=== Summary ===")
    for name, ok in results:
        print(f"  {PASS if ok else FAIL}  {name}")
    all_ok = all(ok for _, ok in results)
    print(f"\n{'All tests PASSED.' if all_ok else 'Some tests FAILED.'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
