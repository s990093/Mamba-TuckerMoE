#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 — SFT-GO (Structure Token Group Optimization) helpers.

Loads pre-computed per-token structure weights and applies them to a
per-token CE before the masked mean.  Two storage formats are supported:

1.  **Bundle** (recommended): a single `.pt` file produced by
    `build_structure_weights.py` containing
        { "ids": List[str],
          "weights": List[np.float32 ndarray],
          "assistant_start_tokens": List[int],
          "config": {...} }
    Variable-length, index-aligned with the HuggingFace dataset row order.

2.  **Per-sample .npz** (legacy): `<weights_dir>/<sample_id>.npz` with a
    `weight` key.

The loader exposes the same `load_sample_weights / batch_load_weights` API
regardless of format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class StructureWeightLoader:
    """Resolve sample → (1D weight tensor, asst_start_token).

    The returned tensor is always length `max_length`: right-padded with 1.0
    if shorter, truncated if longer.  Padding with 1.0 (the neutral weight)
    means SFT-GO becomes a no-op on positions that are either out-of-data
    or label-masked — exactly the semantics we want.
    """

    def __init__(
        self,
        *,
        bundle_path: Optional[Path] = None,
        weights_dir: Optional[Path] = None,
        max_length: int = 512,
    ):
        if bundle_path is None and weights_dir is None:
            raise ValueError("provide either bundle_path or weights_dir")
        self.max_length = int(max_length)
        self._bundle: Optional[Dict] = None
        self._id_to_idx: Dict[str, int] = {}
        self._weights_dir: Optional[Path] = None
        self._cache: Dict[str, torch.Tensor] = {}

        if bundle_path is not None and Path(bundle_path).exists():
            bundle = torch.load(str(bundle_path), weights_only=False)
            self._bundle = bundle
            self._id_to_idx = {sid: i for i, sid in enumerate(bundle["ids"])}
        elif weights_dir is not None and Path(weights_dir).exists():
            self._weights_dir = Path(weights_dir)
        else:
            tried = bundle_path if bundle_path is not None else weights_dir
            raise FileNotFoundError(f"no weights source at {tried}")

    # ----- single sample
    def _resolve_raw(self, sample_id: str) -> Optional[np.ndarray]:
        if self._bundle is not None:
            idx = self._id_to_idx.get(sample_id)
            if idx is None:
                return None
            return np.asarray(self._bundle["weights"][idx], dtype=np.float32)
        # legacy npz
        f = self._weights_dir / f"{sample_id}.npz"
        if not f.exists():
            return None
        try:
            data = np.load(f, allow_pickle=True)
            return np.asarray(data["weight"], dtype=np.float32)
        except Exception:
            return None

    def load_sample_weights(
        self,
        sample_id: str,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        cached = self._cache.get(sample_id)
        if cached is not None:
            return cached.to(device)

        raw = self._resolve_raw(sample_id)
        if raw is None:
            t = torch.ones(self.max_length, dtype=torch.float32)
        else:
            if raw.size >= self.max_length:
                t = torch.from_numpy(raw[: self.max_length].copy())
            else:
                t = torch.ones(self.max_length, dtype=torch.float32)
                t[: raw.size] = torch.from_numpy(raw)
        self._cache[sample_id] = t
        return t.to(device)

    # ----- batch
    def batch_load_weights(
        self,
        sample_ids: Sequence[str],
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return torch.stack(
            [self.load_sample_weights(s, device=device) for s in sample_ids],
            dim=0,
        )

    # ----- bulk preload (index-aligned with dataset)
    def preload_all_as_tensor(
        self,
        seq_len: int,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Build one padded [N, seq_len] tensor row-aligned with bundle order.

        Only available when constructed from a bundle.  Returns (tensor, ids).
        """
        if self._bundle is None:
            raise RuntimeError("preload_all_as_tensor requires a bundle source")
        ids = list(self._bundle["ids"])
        n = len(ids)
        out = torch.ones((n, seq_len), dtype=dtype)
        for i, w_np in enumerate(self._bundle["weights"]):
            w = np.asarray(w_np, dtype=np.float32)
            k = min(w.size, seq_len)
            if k > 0:
                out[i, :k] = torch.from_numpy(w[:k])
        return out, ids


# ---------------------------------------------------------------------------
# Application: per-token CE × structure weights → masked mean
# ---------------------------------------------------------------------------
def apply_structure_weights(
    per_token_loss: torch.Tensor,
    structure_weights: torch.Tensor,
    labels: torch.Tensor,
    mask_value: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multiply per-token CE by structure weights and take the masked mean.

    Args:
        per_token_loss:    (B, T) CE per position (reduction='none')
        structure_weights: (B, T) or (T,) float weights
        labels:            (B, T) long with `mask_value` on positions to ignore
        mask_value:        ignored label value (default -100)

    Returns:
        weighted_per_token: (B, T) per-token weighted loss (not yet reduced)
        weighted_mean:      scalar — masked mean over valid positions
    """
    if structure_weights.dim() == 1:
        structure_weights = structure_weights.unsqueeze(0).expand_as(per_token_loss)
    if structure_weights.shape != per_token_loss.shape:
        raise ValueError(
            f"structure_weights shape {tuple(structure_weights.shape)} "
            f"!= per_token_loss shape {tuple(per_token_loss.shape)}"
        )

    weighted = per_token_loss * structure_weights
    valid = (labels != mask_value).to(weighted.dtype)
    denom = valid.sum().clamp(min=1.0)
    mean = (weighted * valid).sum() / denom
    return weighted, mean


# ---------------------------------------------------------------------------
# Minimal self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("SFT-GO module self-test")
    B, T = 2, 8
    pt_loss = torch.ones(B, T)
    sw = torch.tensor([[1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0],
                       [1.5] * T])
    lab = torch.full((B, T), 5, dtype=torch.long)
    lab[0, :2] = -100  # mask out positions 0,1 (weight 1.0, 2.0)
    lab[1, -2:] = -100

    w_per_tok, w_mean = apply_structure_weights(pt_loss, sw, lab)
    # row0 valid: positions 2..7, weights [1,3,1,1,1,1], sum=8
    # row1 valid: positions 0..5, weight 1.5 each, sum=9
    # total = 17, count = 12 -> mean ≈ 17/12 = 1.4167
    print("weighted_per_token row0:", w_per_tok[0].tolist())
    print("weighted_mean:", float(w_mean))
    assert abs(float(w_mean) - 17.0/12.0) < 1e-5, f"unexpected mean {float(w_mean)}"
    print("OK")
