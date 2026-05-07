#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytical KV / decode-state memory for ``mlx_hybrid_infer`` TrueHybridMamba.

- **Transformer** layers: KV cache is shaped ``(B, n_heads, T_slot, 64)`` per K and V
  (matches padded buffers in ``attach_decode_compilation``). Grows ~linearly with
  allocated length ``T_slot`` (≥ prefill + decode).
- **Mamba** layers: decode state is **fixed size** per layer (recurrent ``h``, last
  ``input`` slice, angle summary) — **does not** grow with decode length like KV.

Usage (from repo root):
  python inference/tools/analyze_kv_cache_sizes.py
  python inference/tools/analyze_kv_cache_sizes.py --prefill-len 256 --decode-lens 1,32,128,512 --dtype bf16
  python inference/tools/analyze_kv_cache_sizes.py --mamba-ratio 4 --num-layers 6
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class HybridCacheModel:
    """Matches ``mlx_hybrid_infer.TrueHybridMamba`` + ``TransformerBlock`` / ``Mamba3Block``."""

    d_model: int
    d_state: int
    d_head: int
    expand: int
    num_layers: int
    mamba_ratio: int
    head_dim: int = 64

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def n_heads_mamba(self) -> int:
        return self.d_inner // self.d_head

    @property
    def n_xf_heads(self) -> int:
        return self.d_model // self.head_dim

    @property
    def n_mamba_layers(self) -> int:
        return self.num_layers * self.mamba_ratio

    @property
    def n_xf_layers(self) -> int:
        return self.num_layers

    def bytes_per_elem(self, dtype: str) -> int:
        return {"fp32": 4, "bf16": 2, "fp16": 2}.get(dtype, 4)

    def mamba_decode_state_bytes_per_layer(self, dtype: str) -> int:
        """Single-step Mamba cache: (h, prev_input, angle_sum) — independent of sequence length."""
        b = self.bytes_per_elem(dtype)
        h_m, n, p = self.n_heads_mamba, self.d_state, self.d_head
        # h_final: (1, H, N, P)
        s_h = 1 * h_m * n * p
        # prev_input: (1, 1, H, N, P)
        s_pi = 1 * 1 * h_m * n * p
        # angle buffer: (1, 1, H, N/2) — see ``new_angle_sum = angles[:, -1:]`` in Mamba3Block
        s_ang = 1 * 1 * h_m * (n // 2)
        return (s_h + s_pi + s_ang) * b

    def mamba_total_decode_state_bytes(self, dtype: str) -> int:
        return self.n_mamba_layers * self.mamba_decode_state_bytes_per_layer(dtype)

    def transformer_kv_bytes_one_layer(self, slot_len: int, kv_dtype: str) -> int:
        """K + V for one XF layer, shape (1, n_heads, slot_len, head_dim)."""
        b = self.bytes_per_elem(kv_dtype)
        per = 1 * self.n_xf_heads * slot_len * self.head_dim
        return 2 * per * b

    def transformer_total_kv_bytes(self, slot_len: int, kv_dtype: str) -> int:
        return self.n_xf_layers * self.transformer_kv_bytes_one_layer(slot_len, kv_dtype)


def _parse_lens(s: str) -> List[int]:
    out: List[int] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid backbone cache size (Mamba state + Transformer KV)")
    p.add_argument("--prefill-len", type=int, default=256, help="Prefill token count S (positions before decode).")
    p.add_argument(
        "--decode-lens",
        type=str,
        default="1,32,64,128,256,512",
        help="Comma-separated decode step counts D (new tokens). Each row uses T_slot = S + D (+ slack).",
    )
    p.add_argument(
        "--slack",
        type=int,
        default=8,
        help="Extra slots like benchmark ``max_cache_len = prefill + decode + slack``.",
    )
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--mamba-ratio", type=int, default=4)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-state", type=int, default=64)
    p.add_argument("--d-head", type=int, default=64)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=("fp32", "bf16", "fp16"),
        help="Bytes estimate for Mamba state (matches KV matmul dtype in typical runs).",
    )
    p.add_argument(
        "--kv-dtype",
        type=str,
        default="bf16",
        choices=("fp32", "bf16", "fp16"),
        help="KV cache element type (often bf16 even when weights are fp32).",
    )
    args = p.parse_args()

    m = HybridCacheModel(
        d_model=args.d_model,
        d_state=args.d_state,
        d_head=args.d_head,
        expand=args.expand,
        num_layers=args.num_layers,
        mamba_ratio=args.mamba_ratio,
    )

    decode_lens = _parse_lens(args.decode_lens)
    s_pref = args.prefill_len

    mamba_b = m.mamba_total_decode_state_bytes(args.dtype)
    mamba_mb = mamba_b / (1024**2)

    print("Hybrid backbone cache analysis (analytical)")
    print("-" * 88)
    print(
        f"  Layers: {m.n_mamba_layers} mamba + {m.n_xf_layers} transformer  |  "
        f"Mamba heads H={m.n_heads_mamba}, XF Q-heads={m.n_xf_heads}, KV-heads={args.num_kv_heads}"
    )
    print(
        f"  Mamba decode state: **fixed** per layer (~{m.mamba_decode_state_bytes_per_layer(args.dtype) / 1024:.1f} KiB/layer @ {args.dtype})"
    )
    print(f"  Mamba total (all layers): {mamba_mb:.3f} MiB  (unchanged as decode length grows)")
    print(f"  Transformer KV: grows with allocated slot length T_slot (here: S + D + slack = {s_pref} + D + {args.slack})")
    print("-" * 88)
    print(
        f"{'Decode D':>10} {'T_slot':>8} {'XF KV (MiB)':>14} {'Mamba (MiB)':>14} {'Total (MiB)':>14} {'XF / total':>12}"
    )
    print("-" * 88)

    for d in decode_lens:
        t_slot = s_pref + d + args.slack
        xf_b = m.transformer_total_kv_bytes(t_slot, args.kv_dtype)
        xf_mb = xf_b / (1024**2)
        tot_mb = (mamba_b + xf_b) / (1024**2)
        ratio = xf_b / (mamba_b + xf_b) if (mamba_b + xf_b) > 0 else 0.0
        print(f"{d:10d} {t_slot:8d} {xf_mb:14.3f} {mamba_mb:14.3f} {tot_mb:14.3f} {ratio:11.1%}")

    print("-" * 88)
    print("  T_slot = prefill_len + decode_len + slack (padded KV buffer length).")
    print("  Mamba state size ignores sequence length; only XF KV scales with T_slot.")


if __name__ == "__main__":
    main()
