"""Load .npz checkpoint into Mamba3LM nn.Module."""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from .model import Mamba3LM
from .config import Mamba3Config


def _np_to_torch(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    f = arr.astype(np.float32) if arr.dtype != np.float32 else arr
    return torch.from_numpy(f).to(dtype=dtype)


def load_checkpoint(model: Mamba3LM, path: str, dtype: torch.dtype = torch.float32) -> None:
    """Load weights from .npz checkpoint into model in-place."""
    raw = np.load(path)
    w   = {k: _np_to_torch(raw[k], dtype) for k in raw.files}

    # ── Embed / norm / head ─────────────────────────────────────────────────────
    model.embed.weight.data.copy_(w["embed.weight"])
    model.norm.weight.data.copy_(w["norm.weight"])
    model.head.weight.data.copy_(w["head.weight"])

    # ── Per-block weights ───────────────────────────────────────────────────────
    cfg = model.cfg
    types = cfg.block_types()
    mi = ti = 0

    for blk_i, (t, blk) in enumerate(zip(types, model.blocks)):
        pfx = f"backbone.layers.{blk_i}.block"

        if t == "mamba":
            # Mamba3Block
            blk.norm_mamba.weight.data.copy_(w[f"{pfx}.norm_mamba.weight"])
            blk.in_proj.weight.data.copy_(w[f"{pfx}.in_proj.weight"])
            blk.in_proj.bias.data.copy_(w[f"{pfx}.in_proj.bias"])

            _load_tucker(blk.x_up_proj, w, f"{pfx}.x_up_proj")
            _load_tucker(blk.out_proj,  w, f"{pfx}.out_proj")

            blk.y_down_proj.weight.data.copy_(w[f"{pfx}.y_down_proj.weight"])
            blk.mamba_dense_proj.weight.data.copy_(w[f"{pfx}.mamba_dense_proj.weight"])
            blk.pre_gate_norm.weight.data.copy_(w[f"{pfx}.pre_gate_norm.weight"])
            blk.norm_out_proj.weight.data.copy_(w[f"{pfx}.norm_out_proj.weight"])
            blk.norm_B.weight.data.copy_(w[f"{pfx}.norm_B.weight"])
            blk.norm_C.weight.data.copy_(w[f"{pfx}.norm_C.weight"])
            blk.ls_mamba.gamma.data.copy_(w[f"{pfx}.ls_mamba.gamma"])
            blk.ls_out_proj.gamma.data.copy_(w[f"{pfx}.ls_out_proj.gamma"])
            blk.theta_log.data.copy_(w[f"{pfx}.theta_log"])
            blk.D.data.copy_(w[f"{pfx}.D"])
            blk.bias_B.data.copy_(w[f"{pfx}.bias_B"])
            blk.bias_C.data.copy_(w[f"{pfx}.bias_C"])
            mi += 1

        else:
            # TransformerBlock
            blk.norm_attn.weight.data.copy_(w[f"{pfx}.norm_attn.weight"])
            blk.q_proj.weight.data.copy_(w[f"{pfx}.q_proj.weight"])
            blk.k_proj.weight.data.copy_(w[f"{pfx}.k_proj.weight"])
            blk.v_proj.weight.data.copy_(w[f"{pfx}.v_proj.weight"])
            blk.o_proj.weight.data.copy_(w[f"{pfx}.o_proj.weight"])
            blk.o_proj.bias.data.copy_(w[f"{pfx}.o_proj.bias"])
            blk.ls_attn.gamma.data.copy_(w[f"{pfx}.ls_attn.gamma"])
            blk.norm_ffn.weight.data.copy_(w[f"{pfx}.norm_ffn.weight"])
            _load_tucker(blk.ffn_gate, w, f"{pfx}.ffn.gate_proj")
            _load_tucker(blk.ffn_up,   w, f"{pfx}.ffn.up_proj")
            _load_tucker(blk.ffn_down, w, f"{pfx}.ffn.down_proj")
            blk.ls_ffn.gamma.data.copy_(w[f"{pfx}.ls_ffn.gamma"])
            ti += 1


def _load_tucker(moe, w: dict, pfx: str) -> None:
    moe.router.weight.data.copy_(w[f"{pfx}.router.weight"])
    moe.U_in.data.copy_(w[f"{pfx}.U_in"])
    moe.inner_norm.weight.data.copy_(w[f"{pfx}.inner_norm.weight"])
    moe.U_expert.data.copy_(w[f"{pfx}.U_expert"])
    moe.core.data.copy_(w[f"{pfx}.core"])
    moe.U_out.data.copy_(w[f"{pfx}.U_out"])
    moe.bias.data.copy_(w[f"{pfx}.bias"])
