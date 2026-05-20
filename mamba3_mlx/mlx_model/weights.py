"""Load .npz checkpoint into a Mamba3LanguageModel."""

import numpy as np
import mlx.core as mx

from .hybrid_model import Mamba3LanguageModel


def _to_mx(arr, dtype):
    return mx.array(arr, dtype=dtype)


def load_checkpoint(model: Mamba3LanguageModel, npz_path: str, dtype=mx.bfloat16):
    """Load weights into ``model`` in-place.

    Maps:
      embed.weight              -> embed.weight
      norm.weight               -> norm.weight
      head.weight               -> head.weight  (also = embed.weight; tied)
      backbone.layers.{i}.block.X -> backbone.layer_{i}.X
    """
    data = np.load(npz_path)
    flat = {k: data[k] for k in data.files}

    # Build a mapping from PyTorch-style key -> tensor.
    # Then translate to MLX parameter paths.
    n_total = len(model.backbone.layers)

    params = {}

    # Embedding / final norm / head
    params["embed.weight"] = _to_mx(flat["embed.weight"], dtype)
    params["norm.weight"] = _to_mx(flat["norm.weight"], dtype)
    # head.weight is tied to embed (Linear: (vocab, d_model))
    head_w = flat.get("head.weight", flat["embed.weight"])
    params["head.weight"] = _to_mx(head_w, dtype)

    for i in range(n_total):
        src = f"backbone.layers.{i}.block."
        dst = f"backbone.layers.{i}."
        layer_keys = [k for k in flat if k.startswith(src)]
        for k in layer_keys:
            sub = k[len(src):]
            params[dst + sub] = _to_mx(flat[k], dtype)

    # Some tensors in the .npz live under a slightly different name in MLX
    # (TuckerMoE uses bare attribute names; nn.Linear uses .weight). Both
    # patterns line up directly thanks to consistent naming, so we set
    # everything in one pass.
    model.load_weights(list(params.items()), strict=False)
    return model
