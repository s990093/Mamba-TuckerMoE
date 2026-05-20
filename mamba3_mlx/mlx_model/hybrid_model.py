# -*- coding: utf-8 -*-
"""
Full Hybrid Mamba3 + TuckerMoE Language Model for MLX.
Includes weight loading from the PyTorch-format .npz checkpoint.
"""
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Optional

from .ops import RMSNorm, fast_scaled_tanh
from .mamba_block import Mamba3Block
from .transformer_block import TransformerBlock
from .state_utils import init_mamba_states, init_kv_caches

# Phase 2C: Optimized Mamba block selection
_MAMBA_BLOCK_MODE = "baseline"  # "baseline" | "metal" | "fused"

def set_mamba_block_mode(mode: str) -> None:
    """Set which Mamba block version to use: baseline, metal, or fused."""
    global _MAMBA_BLOCK_MODE
    if mode not in ("baseline", "metal", "fused"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'baseline', 'metal', or 'fused'")
    _MAMBA_BLOCK_MODE = mode


def get_mamba_block_class():
    """Get the Mamba block class based on current mode."""
    if _MAMBA_BLOCK_MODE == "fused":
        try:
            from .mamba_block_fused import Mamba3BlockFused
            return Mamba3BlockFused
        except ImportError:
            import logging
            logging.warning("Fused block import failed, falling back to baseline")
            return Mamba3Block
    return Mamba3Block


@dataclass
class Mamba3Config:
    d_model: int = 768
    d_state: int = 64
    d_head: int = 64
    n_groups: int = 1
    mimo_rank: int = 4
    expand: int = 2
    num_layers: int = 6
    use_parallel_scan: bool = True
    use_kmoe: bool = True
    kmoe_num_experts: int = 8
    kmoe_top_k: int = 2
    kmoe_r1: int = 32
    kmoe_r2: int = 512
    kmoe_r3: int = 256
    ffn_expand: int = 6
    num_kv_heads: int = 4
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    layer_scale_init: float = 1e-2
    rms_norm_eps: float = 1e-5
    chunk_size: int = 64
    mamba_ratio: int = 4

    @property
    def d_inner(self):
        return self.expand * self.d_model

    @property
    def n_heads(self):
        return self.d_inner // self.d_head


def config_from_dict(d: dict) -> Mamba3Config:
    # Computed properties (d_inner, n_heads) must not be set directly
    _skip = {"d_inner", "n_heads", "kv_groups"}
    kwargs = {k: v for k, v in d.items() if k not in _skip and k in Mamba3Config.__dataclass_fields__}
    return Mamba3Config(**kwargs)


def _embed_key(data) -> str:
    for key in ("embed.weight", "model_embed.weight"):
        if key in data.files:
            return key
    raise KeyError("No embed.weight in checkpoint")


def _vocab_size_from_npz(data) -> int:
    return int(data[_embed_key(data)].shape[0])


def infer_config_from_npz(data) -> Mamba3Config:
    """Build config from NPZ weights when config_json is absent."""
    cfg = Mamba3Config()
    embed_key = _embed_key(data)
    d_model = int(data[embed_key].shape[1])
    if d_model != cfg.d_model:
        cfg.d_model = d_model

    trans_idxs = sorted({
        int(k.split("backbone.layers.")[1].split(".")[0])
        for k in data.files
        if "q_proj.weight" in k and "backbone.layers." in k
    })
    mamba_count = sum(1 for k in data.files if k.endswith(".block.in_proj.weight"))
    if trans_idxs:
        cfg.num_layers = len(trans_idxs)
        if mamba_count and mamba_count % len(trans_idxs) == 0:
            cfg.mamba_ratio = mamba_count // len(trans_idxs)

    layer0 = next(
        (k for k in data.files if k.startswith("backbone.layers.0.") and k.endswith(".block.D")),
        None,
    )
    if layer0 is not None:
        n_heads = int(data[layer0].shape[0])
        if cfg.n_heads != n_heads:
            cfg.d_head = cfg.d_inner // n_heads

    y_key = next(
        (k for k in data.files if k.startswith("backbone.layers.0.") and k.endswith(".block.y_down_proj.weight")),
        None,
    )
    if y_key is not None:
        cfg.d_state = int(data[y_key].shape[0])

    k_key = next(
        (k for k in data.files if "backbone.layers." in k and k.endswith(".block.k_proj.weight")),
        None,
    )
    if k_key is not None:
        cfg.num_kv_heads = int(data[k_key].shape[0]) // cfg.d_head

    core_key = next(
        (k for k in data.files if k.endswith(".block.out_proj.core")),
        None,
    )
    if core_key is not None:
        r1, r3, r2 = (int(x) for x in data[core_key].shape)
        cfg.kmoe_r1, cfg.kmoe_r3, cfg.kmoe_r2 = r1, r3, r2

    return cfg


def config_from_npz(data) -> Mamba3Config:
    if "config_json" in data.files:
        import json
        return config_from_dict(json.loads(data["config_json"].item()))
    return infer_config_from_npz(data)


class TrueHybridMamba(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        ratio = config.mamba_ratio  # 4 Mamba per 1 Transformer
        layers = []

        # Phase 2C: Use optimized block version
        MambaBlockClass = get_mamba_block_class()

        for _ in range(config.num_layers):
            for _ in range(ratio):
                layers.append(MambaBlockClass(config))
            layers.append(TransformerBlock(config))
        self.layers = layers
        self.mamba_ratio = ratio
        self.num_layers = config.num_layers

    def _is_transformer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % (self.mamba_ratio + 1) == 0

    def __call__(self, x, step=None, mamba_states=None, kv_caches=None):
        """
        Full prefill or decode forward.

        mamba_states: list[dict] one per Mamba layer, or None
        kv_caches   : list[dict] one per Transformer layer, or None

        Returns (x, lb, new_mamba_states, new_kv_caches)
        """
        new_mamba_states = []
        new_kv_caches = []
        total_lb = 0.0

        mamba_idx = 0
        trans_idx = 0

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Mamba3Block):
                state_in = mamba_states[mamba_idx] if mamba_states else None
                x, lb, _, new_state = layer(x, step=step, mamba_state=state_in)
                total_lb = total_lb + lb
                new_mamba_states.append(new_state)
                mamba_idx += 1
            else:
                cache_in = kv_caches[trans_idx] if kv_caches else {"k": None, "v": None}
                x, lb, _, new_cache = layer(x, step=step, kv_cache=cache_in)
                total_lb = total_lb + lb
                new_kv_caches.append(new_cache)
                trans_idx += 1

        return x, total_lb, new_mamba_states, new_kv_caches


class Mamba3LanguageModel(nn.Module):
    def __init__(self, config: Mamba3Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.embed    = nn.Embedding(vocab_size, config.d_model)
        self.backbone = TrueHybridMamba(config)
        self.norm     = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.head     = nn.Linear(vocab_size, config.d_model, bias=False)  # placeholder shape

        # head weight will be loaded separately and may share embed weight
        self.vocab_size = vocab_size

    def __call__(self, input_ids, step=None, mamba_states=None, kv_caches=None):
        x = self.embed(input_ids)                      # (B, L, d_model)
        x, lb, new_mamba, new_kv = self.backbone(
            x, step=step, mamba_states=mamba_states, kv_caches=kv_caches
        )
        x = self.norm(x)                               # (B, L, d_model)

        # logits = fast_scaled_tanh(x @ embed.weight.T / sqrt(d_model), 30.0)
        scale = math.sqrt(self.config.d_model)
        logits = fast_scaled_tanh((x / scale) @ self.embed.weight.T, 30.0)  # (B, L, V)

        return logits, lb, new_mamba, new_kv

    def init_empty_states(self, batch_size: int = 1):
        """
        Create zero-initialized Mamba states and empty KV caches.
        Uses state_utils factory — single source of truth for schema.
        """
        cfg = self.config
        num_mamba = cfg.num_layers * cfg.mamba_ratio
        num_trans  = cfg.num_layers
        return (
            init_mamba_states(num_mamba, batch_size,
                              H=cfg.n_heads, N=cfg.d_state, P=cfg.d_head),
            init_kv_caches(num_trans),
        )


# ── Weight Loading ─────────────────────────────────────────────────────────────

def _remap_key(key: str) -> str | None:
    """Map .npz key → MLX module attribute path.

    Both numpy and mx.load use the same raw key format (no model_ prefix).
    We only need to strip the .block. wrapper that the training code inserts.
    """
    if key == "config_json":
        return None
    if key.startswith("model_"):          # defensive: handle old PyTorch exports
        key = key[len("model_"):]
    key = key.replace(".block.", ".")
    return key


def build_model(npz_path: str, dtype=mx.bfloat16) -> "Mamba3LanguageModel":
    """Instantiate model and load weights.

    Fast path: mx.load() reads the npz in ~40 ms (vs ~6 s for numpy).
    We still open with numpy briefly to extract the config_json blob
    (a pickled Python string that mx.load skips).
    """
    # Config detection — numpy needed for allow_pickle=True on config_json
    np_data    = np.load(npz_path, allow_pickle=True)
    config     = config_from_npz(np_data)
    vocab_size = _vocab_size_from_npz(np_data)
    np_data.close()                         # release file handle

    model = Mamba3LanguageModel(config, vocab_size)
    _load_weights_mlx(model, npz_path, dtype)
    return model


def _load_weights_mlx(
    model: "Mamba3LanguageModel",
    npz_path: str,
    dtype: mx.Dtype = mx.bfloat16,
) -> None:
    """Load weights using mx.load — ~150× faster than the numpy path.

    mx.load reads the npz without full decompression into Python/numpy RAM,
    returning lazy MLX arrays that are cast and materialized in one eval call.
    """
    raw  = mx.load(npz_path)               # ~40 ms

    flat: dict[str, mx.array] = {}
    for key, arr in raw.items():
        mlx_key = _remap_key(key)
        if mlx_key is None or mlx_key.startswith("head."):
            continue
        # Cast floats to target dtype (lazy — no GPU work yet)
        if arr.dtype in (mx.float32, mx.float16):
            arr = arr.astype(dtype)
        flat[mlx_key] = arr

    model.load_weights(list(flat.items()), strict=False)

    # Release file-mapped buffers BEFORE eval so we don't hold 2× the checkpoint
    # in memory (lazy raw arrays + materialized model weights) at the same time.
    del raw, flat
    import gc; gc.collect()

    mx.eval(model.parameters())            # single materialization pass


def load_weights(model: "Mamba3LanguageModel", npz_path: str, dtype=mx.bfloat16):
    """Backward-compatible wrapper (accepts path string)."""
    _load_weights_mlx(model, npz_path, dtype)
    return model
