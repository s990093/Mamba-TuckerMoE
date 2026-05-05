# -*- coding: utf-8 -*-
"""
Mixed-bit weight quantization helpers for Tucker MoE stacks on MLX.

Uses :func:`mlx.nn.quantize` with ``class_predicate`` so router Linears use a tighter
bitwidth than ``U_in`` / ``U_out`` inside the same Tucker block. Tucker ``core`` /
``U_expert`` arrays are unaffected (MLX only quantizes modules with ``to_quantized()``).
"""

from __future__ import annotations

from typing import Any, Callable, Tuple, Union

import mlx.nn as nn


BoolOrQuant = Union[bool, dict[str, Any]]


def _tucker_route_path(path: str) -> bool:
    lowered = path.lower()
    needles = ("gate_proj", "up_proj", "down_proj", "x_up_proj", "out_proj")
    return any(n in lowered for n in needles)


def moe_asymmetric_predicate(
    router_bits: int,
    tucker_linear_bits: int,
    default_bits: int,
    group_size: int,
) -> Callable[[str, nn.Module], BoolOrQuant]:
    """
    Return a predicate suitable for :func:`mlx.nn.quantize`.

    - Embedding + any non-Linear quantized module → *default_bits*
    - Linear **not** inside Tucker-named path → *default_bits*
    - Tucker ``router`` Linear → *router_bits*
    - Tucker ``U_in`` / ``U_out`` Linears → *tucker_linear_bits*
    """

    def _pred(path: str, m: nn.Module) -> BoolOrQuant:
        if getattr(m, "to_quantized", None) is None:
            return False
        gs = dict(group_size=group_size, mode="affine")
        # Embedding exposes to_quantized
        if isinstance(m, nn.Embedding):
            return {**gs, "bits": default_bits}
        if not isinstance(m, nn.Linear):
            return False

        leaf = path.split(".")[-1]
        lk = leaf.lower()
        bits = default_bits

        if _tucker_route_path(path):
            if lk == "router":
                bits = router_bits
            elif lk in ("u_in", "u_out"):
                bits = tucker_linear_bits

        return {**gs, "bits": bits}

    return _pred


def apply_moe_asymmetric_quantization(
    model: nn.Module,
    *,
    router_bits: int = 4,
    tucker_linear_bits: int = 8,
    default_bits: int = 8,
    group_size: int = 64,
) -> None:
    import mlx.core as mx

    pred = moe_asymmetric_predicate(
        router_bits=router_bits,
        tucker_linear_bits=tucker_linear_bits,
        default_bits=default_bits,
        group_size=group_size,
    )
    nn.quantize(model, group_size=group_size, mode="affine", class_predicate=pred)
    mx.eval(model.parameters())
