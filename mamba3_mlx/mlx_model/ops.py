import mlx.core as mx
import mlx.nn as nn


def tanh_approx(x):
    return 2.0 * mx.sigmoid(2.0 * x) - 1.0


def scaled_tanh(x, scale: float = 10.0):
    return tanh_approx(x * (1.0 / scale)) * scale


def silu(x):
    return x * mx.sigmoid(x)


def silu_gating(gate, feat):
    return silu(gate) * feat


def softplus(x):
    return mx.logaddexp(x, mx.zeros_like(x))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        f = x.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(f * f, axis=-1, keepdims=True) + self.eps)
        return (f * rms).astype(x.dtype) * self.weight.astype(x.dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value: float = 1e-2):
        super().__init__()
        self.gamma = mx.ones((dim,)) * init_value

    def __call__(self, x):
        return x * self.gamma.astype(x.dtype)


def apply_rope(x, angles):
    """x: (..., N, R); angles: (..., N//2)."""
    N_half = angles.shape[-1]
    R = x.shape[-1]
    # reshape last two dims (N, R) -> (N_half, 2, R)
    x_r = x.reshape(*x.shape[:-2], N_half, 2, R)
    x1 = x_r[..., 0, :]
    x2 = x_r[..., 1, :]
    sin_a = mx.sin(angles)[..., None].astype(x.dtype)
    cos_a = mx.cos(angles)[..., None].astype(x.dtype)
    r1 = x1 * cos_a - x2 * sin_a
    r2 = x2 * cos_a + x1 * sin_a
    out = mx.stack([r1, r2], axis=-2)
    return out.reshape(*x.shape)
