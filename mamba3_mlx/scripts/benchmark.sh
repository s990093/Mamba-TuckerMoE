#!/usr/bin/env bash
# benchmark.sh — throughput benchmark with correct lazy-eval discipline
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"
cd "${REPO_ROOT}"

echo "=== Mamba3-XR MLX Throughput Benchmark ==="
echo "Checkpoint : checkpoints/latest_sft_cot_model.npz"
echo "dtype      : bf16"
echo ""

"${PYTHON}" - <<'PYEOF'
import sys, time
sys.path.insert(0, ".")
import mlx.core as mx
from mamba3_mlx.mlx_model.hybrid_model import build_model
from mamba3_mlx.inference.generator import prefill, decode_step, _eval_states

CHECKPOINT = "checkpoints/latest_sft_cot_model.npz"

print("Loading model...", end=" ", flush=True)
t0 = time.perf_counter()
model = build_model(CHECKPOINT, dtype=mx.bfloat16)
print(f"done ({time.perf_counter()-t0:.1f}s)")

PROMPT = list(range(1, 21))   # 20 tokens

# ── Prefill timing ─────────────────────────────────────────────────────────
print(f"\nPrefill ({len(PROMPT)} tokens):")
t0 = time.perf_counter()
logits_pre, m, kv = prefill(model, mx.array([PROMPT]))
# Force eval: prefill is cold, this materialises the full graph
tok = int(mx.argmax(logits_pre[0]).item())
_eval_states(m, kv)
t_pre = time.perf_counter() - t0
print(f"  {t_pre*1000:.0f} ms  ({len(PROMPT)/t_pre:.0f} tok/s)")

# ── Warmup (5 steps, fully synced) ─────────────────────────────────────────
print("Warmup (5 steps)...", end=" ", flush=True)
for _ in range(5):
    logits, m, kv = decode_step(model, tok, m, kv)
    tok = int(mx.argmax(logits[0]).item())  # eval logits
    _eval_states(m, kv)                     # eval states
print("done")

# ── Decode benchmark (50 steps, one eval per step) ─────────────────────────
N = 50
print(f"\nDecode ({N} steps, fully synced per step):")
t0 = time.perf_counter()
for _ in range(N):
    logits, m, kv = decode_step(model, tok, m, kv)
    # eval logits (forces logit graph materialisation)
    tok = int(mx.argmax(logits[0]).item())
    # eval states (caps graph growth to one step)
    _eval_states(m, kv)
elapsed = time.perf_counter() - t0
tps = N / elapsed
print(f"  {elapsed:.2f}s  →  {tps:.1f} tok/s")

print("\n=== Summary ===")
print(f"  Prefill : {t_pre*1000:.0f} ms  ({len(PROMPT)} tokens)")
print(f"  Decode  : {tps:.1f} tok/s (warmed up, per-step sync)")
PYEOF
