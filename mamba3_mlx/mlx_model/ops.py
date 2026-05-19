# -*- coding: utf-8 -*-
import mlx.core as mx
import mlx.nn as nn
import math


def tanh_approx(x):
    return 2.0 * mx.sigmoid(2.0 * x) - 1.0


def fast_scaled_tanh(x, scale=10.0):
    return tanh_approx(x * (1.0 / scale)) * scale


def silu(x):
    return x * mx.sigmoid(x)


def silu_gating(gate, feat):
    return silu(gate) * feat


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        norm = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / norm * self.weight


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = mx.full((dim,), init_value)

    def __call__(self, x):
        return x * self.gamma


def apply_rope(x, angles):
    # x: (B, L, H, N, R)
    # angles: (B, L, H, N//2)
    *prefix, N, R = x.shape
    N_half = angles.shape[-1]
    # reshape x to (B, L, H, N_half, 2, R)
    x_r = x.reshape(*prefix, N_half, 2, R)
    x1 = x_r[..., 0, :]  # (B, L, H, N_half, R)
    x2 = x_r[..., 1, :]  # (B, L, H, N_half, R)
    sin_a = mx.sin(angles)[..., None]   # (B, L, H, N_half, 1)
    cos_a = mx.cos(angles)[..., None]   # (B, L, H, N_half, 1)
    rotated = mx.concatenate([
        (x1 * cos_a - x2 * sin_a)[..., None, :],
        (x2 * cos_a + x1 * sin_a)[..., None, :],
    ], axis=-2)  # (B, L, H, N_half, 2, R)
    return rotated.reshape(*prefix, N, R)


def get_router_temperature(step, warmup=500, total=10000, t_start=2.0, t_end=0.5):
    if step is None:
        return t_end
    progress = max(0.0, min(1.0, (step - warmup) / max(1, total - warmup)))
    return t_end + 0.5 * (t_start - t_end) * (1.0 + math.cos(math.pi * progress))
