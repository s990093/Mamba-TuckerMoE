# -*- coding: utf-8 -*-
"""
Quantized inference for faster decode.

Implements 8-bit weight quantization for memory-bandwidth-bound operations.
Focus: TuckerMoE and attention layers (highest compute ratio).
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import numpy as np


class QuantizedLinear(nn.Module):
    """8-bit quantized linear layer with on-the-fly dequantization."""

    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias

        # Placeholder weights (will be loaded from checkpoint)
        self.weight = mx.zeros((output_size, input_size))
        self.weight_scale = mx.ones((output_size, 1))
        self.weight_zero = mx.zeros((output_size, 1))

        if bias:
            self.bias = mx.zeros((output_size,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """
        Quantized forward pass.

        8-bit quantization: w_int8 = round((w - zero) / scale)
        Dequant: w_fp32 = w_int8 * scale + zero
        """
        # Dequantize weights on-the-fly
        w = self.weight.astype(mx.float32) * self.weight_scale + self.weight_zero

        # Compute
        y = mx.matmul(x, w.T)
        if self.bias is not None:
            y = y + self.bias

        return y


class QuantizationConfig:
    """Configuration for quantization."""

    def __init__(
        self,
        quantize_moe: bool = True,
        quantize_attention: bool = True,
        quantize_ssm: bool = False,
        bits: int = 8,
    ):
        self.quantize_moe = quantize_moe  # TuckerMoE
        self.quantize_attention = quantize_attention  # Transformer
        self.quantize_ssm = quantize_ssm  # Mamba (risky, may affect stability)
        self.bits = bits


def quantize_weight(weight: mx.array, bits: int = 8) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Quantize weight matrix to int8.

    Args:
        weight: (out_features, in_features)
        bits: bit-width for quantization

    Returns:
        weight_q: quantized weights (int8)
        scale: per-output scaling factors
        zero: per-output zero points
    """
    # Per-output channel quantization
    out_features = weight.shape[0]

    scale = mx.zeros((out_features, 1))
    zero = mx.zeros((out_features, 1))
    weight_q = mx.zeros_like(weight, dtype=mx.int8)

    for i in range(out_features):
        w_row = weight[i].astype(mx.float32)
        w_min = mx.min(w_row)
        w_max = mx.max(w_row)

        # Compute scale and zero
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        s = (w_max - w_min) / (qmax - qmin)
        z = qmin - w_min / (s + 1e-8)

        # Quantize
        w_q = mx.clip(mx.round((w_row - z) / s), qmin, qmax).astype(mx.int8)

        # Store
        scale[i, 0] = s
        zero[i, 0] = z
        weight_q[i] = w_q

    return weight_q, scale, zero


def apply_quantization_to_model(
    model, quantization_config: QuantizationConfig
) -> None:
    """
    Apply quantization to model weights in-place.

    Quantizes:
    - TuckerMoE layers (if enabled)
    - Attention projection layers (if enabled)
    - Preserves layer norms and SSM (high sensitivity)
    """
    try:
        from .mamba_block import Mamba3Block
        from .transformer_block import TransformerBlock
        from .tucker_moe import TuckerMoE
    except ImportError:
        print("Warning: Could not import model classes for quantization")
        return

    layer_count = {"moe": 0, "attn": 0}

    # Iterate through backbone layers
    for layer in model.backbone.layers:
        if isinstance(layer, Mamba3Block) and quantization_config.quantize_moe:
            # Quantize TuckerMoE in Mamba block
            if hasattr(layer, "x_up_proj") and isinstance(layer.x_up_proj, TuckerMoE):
                _quantize_tucker_moe(layer.x_up_proj)
                layer_count["moe"] += 1

            if hasattr(layer, "out_proj") and isinstance(layer.out_proj, TuckerMoE):
                _quantize_tucker_moe(layer.out_proj)
                layer_count["moe"] += 1

        if isinstance(layer, TransformerBlock) and quantization_config.quantize_attention:
            # Quantize attention projections
            if hasattr(layer, "q_proj"):
                _quantize_linear_layer(layer.q_proj)
                layer_count["attn"] += 1

            if hasattr(layer, "k_proj"):
                _quantize_linear_layer(layer.k_proj)
                layer_count["attn"] += 1

            if hasattr(layer, "v_proj"):
                _quantize_linear_layer(layer.v_proj)
                layer_count["attn"] += 1

            if hasattr(layer, "out_proj") and isinstance(
                layer.out_proj, nn.Linear
            ):
                _quantize_linear_layer(layer.out_proj)
                layer_count["attn"] += 1

    print(f"✓ Quantized {layer_count['moe']} MoE layers")
    print(f"✓ Quantized {layer_count['attn']} attention layers")


def _quantize_linear_layer(layer: nn.Linear) -> None:
    """Quantize a single linear layer."""
    if not hasattr(layer, "weight"):
        return

    weight_q, scale, zero = quantize_weight(layer.weight)

    # Store quantized weights
    layer.weight_q = weight_q.astype(mx.int8)
    layer.weight_scale = scale
    layer.weight_zero = zero
    layer.is_quantized = True


def _quantize_tucker_moe(layer) -> None:
    """Quantize TuckerMoE layer components."""
    # Quantize router
    if hasattr(layer, "router") and hasattr(layer.router, "weight"):
        weight_q, scale, zero = quantize_weight(layer.router.weight)
        layer.router.weight_q = weight_q.astype(mx.int8)
        layer.router.weight_scale = scale
        layer.router.weight_zero = zero
        layer.router.is_quantized = True

    # Core tensor (usually not quantized due to small size)
    # U_expert, U_in, U_out can be quantized if needed


class CompiledDecodeConfig:
    """Configuration for compile-optimized decode."""

    def __init__(
        self,
        compile_full_decode: bool = True,
        cache_compiled_fn: bool = True,
        materialize_kv_cache: bool = True,
    ):
        self.compile_full_decode = compile_full_decode
        self.cache_compiled_fn = cache_compiled_fn
        self.materialize_kv_cache = materialize_kv_cache
        self._compiled_fn_cache = {}

    def get_compiled_forward(self, model, step: int):
        """Get or compile forward function for current step."""
        if not self.compile_full_decode:
            return model.__call__

        # Cache compiled versions
        cache_key = step % 64  # Reuse compiled functions every 64 steps
        if cache_key not in self._compiled_fn_cache:
            self._compiled_fn_cache[cache_key] = mx.compile(model.__call__)

        return self._compiled_fn_cache[cache_key]


# Export
__all__ = [
    "QuantizedLinear",
    "QuantizationConfig",
    "CompiledDecodeConfig",
    "quantize_weight",
    "apply_quantization_to_model",
]
