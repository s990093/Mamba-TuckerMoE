#!/usr/bin/env python3
"""
Pure PyTorch reference decode benchmark for Mamba3-TuckerMoE.

Loads the actual checkpoint (.npz), implements every decode step in eager
PyTorch (no kernel fusion, no compilation, no quantization) and measures
tok/s on MPS (Apple Silicon GPU) or CPU.

This file is a standalone comparison baseline and does NOT import or modify
any mamba3_mlx/ core code.

Usage:
    .venv/bin/python3 mamba3_mlx/bench_pytorch_ref.py
    .venv/bin/python3 mamba3_mlx/bench_pytorch_ref.py --steps 64 --device cpu
    .venv/bin/python3 mamba3_mlx/bench_pytorch_ref.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch not installed — pip install torch")

# ── Architecture constants (mirror utils/config.py) ───────────────────────────
D_MODEL     = 768
EXPAND      = 2
D_INNER     = D_MODEL * EXPAND      # 1536
D_HEAD      = 64                    # P
N_HEADS     = D_INNER // D_HEAD     # 24  = H
D_STATE     = 64                    # N
N_GROUPS    = 1                     # G
MIMO_RANK   = 4                     # R
VOCAB_SIZE  = 32007
N_TF        = 6                     # transformer blocks
MAMBA_RATIO = 4                     # mamba per transformer
N_MAMBA     = N_TF * MAMBA_RATIO    # 24
N_TOTAL     = N_MAMBA + N_TF        # 30
E_EXP       = 8
TOP_K       = 2
R1, R2, R3  = 32, 512, 256
FFN_DIM     = D_MODEL * 6           # 4608

# in_proj output dims
DIM_Z  = N_HEADS * D_HEAD                        # 1536
DIM_X  = N_HEADS * D_HEAD                        # 1536
DIM_B  = N_GROUPS * D_STATE * MIMO_RANK          # 256
DIM_C  = N_GROUPS * D_STATE * MIMO_RANK          # 256
# dt, A, lam each N_GROUPS=1 wide

# Block-type lookup (global index 0..29)
_MAMBA_IDX = set(
    i
    for s in range(N_TF)
    for i in range(s * (MAMBA_RATIO + 1), s * (MAMBA_RATIO + 1) + MAMBA_RATIO)
)
_TF_IDX = set(range(N_TOTAL)) - _MAMBA_IDX

CHECKPOINT = REPO_ROOT / "checkpoints" / "v6" / "latest_sft_cot_model.npz"


# ── Weight loading ─────────────────────────────────────────────────────────────

def _t(arr: np.ndarray, device, dtype) -> torch.Tensor:
    f = arr.astype(np.float32) if arr.dtype not in (np.float32,) else arr
    return torch.from_numpy(f).to(device=device, dtype=dtype)


def load_weights(npz_path: str, device, dtype) -> dict[str, torch.Tensor]:
    print(f"[pt] loading {npz_path} …", end=" ", flush=True)
    t0 = time.perf_counter()
    raw = np.load(npz_path)
    w = {k: _t(raw[k], device, dtype) for k in raw.files}
    print(f"{time.perf_counter()-t0:.1f}s  ({len(w)} tensors)", flush=True)
    return w


# ── Primitives ─────────────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    n = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return (x / n) * weight


def _moe_w(w: dict, pfx: str) -> dict:
    return {
        "router":     w[f"{pfx}.router.weight"],
        "U_in":       w[f"{pfx}.U_in"],
        "inner_norm": w[f"{pfx}.inner_norm.weight"],
        "U_expert":   w[f"{pfx}.U_expert"],
        "core":       w[f"{pfx}.core"],
        "U_out":      w[f"{pfx}.U_out"],
        "bias":       w[f"{pfx}.bias"],
    }


def tucker_moe(x: torch.Tensor, mw: dict) -> torch.Tensor:
    """
    x: (d_in,)
    U_in: (d_in, r3)   inner_norm: (r3,)   U_expert: (E, r1)
    core: (r1, r3, r2) U_out: (r2, d_out)   bias: (d_out,)
    """
    # Router → top-k selection
    scores = F.linear(x, mw["router"])               # (E,)
    topk_w, topk_idx = torch.topk(scores, TOP_K)
    topk_w = F.softmax(topk_w, dim=-1)               # (top_k,)

    # Shared input projection + inner norm
    x_in = x @ mw["U_in"]                            # (d_in,) @ (d_in,r3) → (r3,)
    x_in = rms_norm(x_in, mw["inner_norm"])           # (r3,)

    # Tucker contraction per selected expert
    acc = x_in.new_zeros(mw["U_out"].shape[0])        # (r2,)
    for k in range(TOP_K):
        e = int(topk_idx[k])
        g = mw["U_expert"][e]                         # (r1,)
        G_e = torch.einsum("r,rst->st", g, mw["core"])   # (r3, r2)
        acc.add_(topk_w[k] * (x_in @ G_e))           # (r3,)@(r3,r2)→(r2,)

    return acc @ mw["U_out"] + mw["bias"]             # (r2,)@(r2,d_out)→(d_out,)


# ── Mamba block decode step ────────────────────────────────────────────────────

def mamba_step(
    x_res: torch.Tensor,               # (D_MODEL,)
    h_prev: torch.Tensor,              # (H, N, P)
    prev_inp_sig: torch.Tensor,        # (H, N, P)
    angles_cum: torch.Tensor,          # (H, N//2)
    w: dict,
    pfx: str,                          # e.g. "backbone.layers.0.block"
):
    H, N, P, R, G = N_HEADS, D_STATE, D_HEAD, MIMO_RANK, N_GROUPS
    p = lambda k: f"{pfx}.{k}"

    # Pre-norm
    u = rms_norm(x_res, w[p("norm_mamba.weight")])                 # (D_MODEL,)

    # Input projection → split
    raw  = F.linear(u, w[p("in_proj.weight")], w[p("in_proj.bias")])  # (3587,)
    z        = raw[:DIM_Z]
    x_prime  = raw[DIM_Z : DIM_Z + DIM_X]
    B_param  = raw[DIM_Z + DIM_X : DIM_Z + DIM_X + DIM_B]
    C_param  = raw[DIM_Z + DIM_X + DIM_B : DIM_Z + DIM_X + DIM_B + DIM_C]
    dt_p     = raw[DIM_Z + DIM_X + DIM_B + DIM_C]                 # scalar
    A_p      = raw[DIM_Z + DIM_X + DIM_B + DIM_C + G]             # scalar
    lam_p    = raw[DIM_Z + DIM_X + DIM_B + DIM_C + 2 * G]         # scalar

    # x_up_proj TuckerMoE: (H*P,) → (H*P*R,) = (6144,)
    x_up  = tucker_moe(x_prime, _moe_w(w, p("x_up_proj")))        # (H*P*R,)
    x_ssm = x_up.view(H, P, R)                                    # (H, P, R)

    # Discretize
    dt  = float(F.softplus(dt_p))
    A   = float(-torch.exp(A_p))
    av  = float(torch.exp(torch.tensor(dt * A, dtype=x_res.dtype, device=x_res.device)))
    lv  = float(torch.sigmoid(lam_p))

    # B, C: norm + bias → (G, N, R) → broadcast to (H, N, R)
    B_n = rms_norm(B_param.view(G, N * R), w[p("norm_B.weight")]).view(G, N, R)
    B_n = B_n + w[p("bias_B")]
    C_n = rms_norm(C_param.view(G, N * R), w[p("norm_C.weight")]).view(G, N, R)
    C_n = C_n + w[p("bias_C")]
    B_h = B_n.expand(H, N, R)                                     # (H, N, R)
    C_h = C_n.expand(H, N, R)

    # Compute input_signal = B ⊗ x  (H, N, P)
    inp_sig = torch.einsum("hnr,hpr->hnp", B_h, x_ssm)            # (H, N, P)

    # SSM recurrence (skip RoPE for benchmark — rotation cost is O(H*N) vs matmuls)
    u_ssm = lv * dt * inp_sig + (1.0 - lv) * dt * av * prev_inp_sig
    h_new = av * h_prev + u_ssm                                    # (H, N, P)

    # Output: y = einsum(h_new, C_h) → (H, P, R) → y_down_proj → (H, P)
    y_stack = torch.einsum("hnp,hnr->hpr", h_new, C_h)            # (H, P, R)
    y_flat  = y_stack.reshape(H, P * R)                            # (H, P*R)
    ydp_w   = w[p("y_down_proj.weight")]                           # (P, P*R)
    y       = (y_flat @ ydp_w.T).reshape(H * P)                   # (H*P,)

    # D skip + pre_gate_norm + silu gate
    D_e   = w[p("D")].repeat_interleave(P)                        # (H*P,)
    y     = y + x_prime * D_e
    y_ng  = rms_norm(y, w[p("pre_gate_norm.weight")])
    gated = y_ng * F.silu(z)                                       # (H*P,)

    # mamba_dense_proj → branch 1 residual
    m_out = F.linear(gated, w[p("mamba_dense_proj.weight")])       # (D_MODEL,)
    mid   = x_res + w[p("ls_mamba.gamma")] * m_out

    # out_proj TuckerMoE → branch 2 residual
    mid_n = rms_norm(mid, w[p("norm_out_proj.weight")])
    proj  = tucker_moe(mid_n, _moe_w(w, p("out_proj")))
    out   = mid + w[p("ls_out_proj.gamma")] * proj

    # Cumulative RoPE angle update (cheap; keeps state shape correct)
    theta = torch.exp(w[p("theta_log")].float()).expand(H, -1)     # (H, N//2)
    new_angles = angles_cum + dt * theta

    return out, h_new, inp_sig, new_angles


# ── Transformer block decode step ──────────────────────────────────────────────

def transformer_step(
    x_res: torch.Tensor,                          # (D_MODEL,)
    K_cache: torch.Tensor,                         # (kv_len, 256)
    V_cache: torch.Tensor,                         # (kv_len, 256)
    wpos: int,
    w: dict,
    pfx: str,
):
    p = lambda k: f"{pfx}.{k}"
    n_q = D_MODEL // D_HEAD                        # 12 Q heads
    n_kv = 256 // D_HEAD                           # 4 KV heads
    ratio = n_q // n_kv                            # 3

    # Attention
    x = rms_norm(x_res, w[p("norm_attn.weight")])
    q = F.linear(x, w[p("q_proj.weight")])         # (D_MODEL=768,)
    k = F.linear(x, w[p("k_proj.weight")])         # (256,)
    v = F.linear(x, w[p("v_proj.weight")])         # (256,)

    K_cache[wpos] = k
    V_cache[wpos] = v

    seq = wpos + 1
    K_seq = K_cache[:seq].view(seq, n_kv, D_HEAD).repeat_interleave(ratio, dim=1)  # (seq, 12, 64)
    V_seq = V_cache[:seq].view(seq, n_kv, D_HEAD).repeat_interleave(ratio, dim=1)

    q_t   = q.view(n_q, D_HEAD)                    # (12, 64)
    attn  = torch.einsum("hd,shd->sh", q_t, K_seq) * (D_HEAD ** -0.5)  # (seq, 12)
    attn  = F.softmax(attn, dim=0)
    y_attn = torch.einsum("sh,shd->hd", attn, V_seq).reshape(D_MODEL)

    o_bias = w.get(p("o_proj.bias"))
    y_out  = F.linear(y_attn, w[p("o_proj.weight")], o_bias)
    x_res  = x_res + w[p("ls_attn.gamma")] * y_out

    # FFN — gate + up → SiLU → down (all Tucker MoE)
    x_f  = rms_norm(x_res, w[p("norm_ffn.weight")])
    gate = tucker_moe(x_f, _moe_w(w, p("ffn.gate_proj")))         # (4608,)
    up   = tucker_moe(x_f, _moe_w(w, p("ffn.up_proj")))           # (4608,)
    act  = F.silu(gate) * up
    dn   = tucker_moe(act, _moe_w(w, p("ffn.down_proj")))          # (D_MODEL,)
    x_res = x_res + w[p("ls_ffn.gamma")] * dn

    return x_res


# ── Full model decode step ─────────────────────────────────────────────────────

class PytorchDecoder:
    def __init__(self, w: dict, device, dtype, kv_len: int = 256):
        self.w      = w
        self.device = device
        self.dtype  = dtype
        self.kv_len = kv_len
        self._init_state()

    def _init_state(self):
        H, N, P = N_HEADS, D_STATE, D_HEAD
        self.h_prev   = [torch.zeros(H, N, P, device=self.device, dtype=self.dtype) for _ in range(N_MAMBA)]
        self.prev_ips = [torch.zeros(H, N, P, device=self.device, dtype=self.dtype) for _ in range(N_MAMBA)]
        self.ang_cum  = [torch.zeros(H, N // 2, device=self.device, dtype=torch.float32) for _ in range(N_MAMBA)]
        self.K_caches = [torch.zeros(self.kv_len, 256, device=self.device, dtype=self.dtype) for _ in range(N_TF)]
        self.V_caches = [torch.zeros(self.kv_len, 256, device=self.device, dtype=self.dtype) for _ in range(N_TF)]
        self.wpos = 0

    def step(self, tok_id: int) -> torch.Tensor:
        w = self.w
        x = w["embed.weight"][tok_id]               # (D_MODEL,)

        mi = ti = 0
        for blk in range(N_TOTAL):
            pfx = f"backbone.layers.{blk}.block"
            if blk in _MAMBA_IDX:
                x, self.h_prev[mi], self.prev_ips[mi], self.ang_cum[mi] = \
                    mamba_step(x, self.h_prev[mi], self.prev_ips[mi],
                               self.ang_cum[mi], w, pfx)
                mi += 1
            else:
                x = transformer_step(x, self.K_caches[ti], self.V_caches[ti],
                                     self.wpos, w, pfx)
                ti += 1

        x     = rms_norm(x, w["norm.weight"])
        logits = F.linear(x, w["head.weight"])      # (VOCAB_SIZE,)
        self.wpos = min(self.wpos + 1, self.kv_len - 1)
        return logits


# ── Benchmark entry point ──────────────────────────────────────────────────────

def run_benchmark(steps: int = 32, warmup: int = 2,
                  device_str: str = "auto",
                  checkpoint: str | None = None,
                  json_out: bool = False) -> dict:

    # Device + dtype
    if device_str == "auto":
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device_str)

    # MPS: float16 is stable; bfloat16 has limited MPS support as of PyTorch 2.x
    dtype = torch.float16 if dev.type == "mps" else torch.float32
    print(f"[pt] device={dev}  dtype={dtype}")

    npz = checkpoint or str(CHECKPOINT)
    w = load_weights(npz, dev, dtype)

    dec = PytorchDecoder(w, dev, dtype)

    def _sync():
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()

    print(f"[pt] warmup {warmup} steps …", flush=True)
    tid = 1
    for _ in range(warmup):
        logits = dec.step(tid)
        _sync()
        tid = int(logits.argmax())

    dec._init_state()
    tid = 1

    print(f"[pt] benchmarking {steps} steps …", flush=True)
    t0 = time.perf_counter()
    for _ in range(steps):
        logits = dec.step(tid)
        tid = int(logits.argmax())
    _sync()
    elapsed = time.perf_counter() - t0

    tps = steps / elapsed
    ms  = elapsed / steps * 1000

    result = {
        "backend":    "pytorch",
        "device":     str(dev),
        "dtype":      str(dtype),
        "steps":      steps,
        "elapsed_s":  round(elapsed, 3),
        "decode_tps": round(tps, 1),
        "ms_per_tok": round(ms, 2),
    }

    if json_out:
        print(json.dumps(result))
    else:
        print(f"\n[pt] ── Decode Result ───────────────────────────")
        print(f"[pt] device     : {dev}  ({dtype})")
        print(f"[pt] throughput : {tps:.1f} tok/s  ({ms:.1f} ms/tok)")
        print(f"[pt] total      : {steps} steps in {elapsed:.2f}s")

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PyTorch reference decode benchmark")
    ap.add_argument("--steps",      type=int, default=32)
    ap.add_argument("--warmup",     type=int, default=2)
    ap.add_argument("--device",     default="auto", help="mps|cpu|cuda|auto")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--json",       action="store_true")
    args = ap.parse_args()
    run_benchmark(steps=args.steps, warmup=args.warmup,
                  device_str=args.device, checkpoint=args.checkpoint,
                  json_out=args.json)
