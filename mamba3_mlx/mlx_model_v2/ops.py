import mlx.core as mx
import mlx.nn as nn

# mx.fast.rms_norm is a dedicated Metal kernel (faster than the manual 3-op path).
# Falls back to manual float32 implementation for older MLX builds.
_fast_rms_norm = getattr(mx.fast, "rms_norm", None)


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
        if _fast_rms_norm is not None:
            return _fast_rms_norm(x, self.weight, self.eps)
        # Fallback: upcast to float32 for numerical stability.
        f = x.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(f * f, axis=-1, keepdims=True) + self.eps)
        return (f * rms).astype(x.dtype) * self.weight.astype(x.dtype)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value: float = 1e-2):
        super().__init__()
        self.gamma = mx.ones((dim,)) * init_value

    def __call__(self, x):
        return x * self.gamma.astype(x.dtype)


def apply_rope_with_sc(x, sin_a, cos_a):
    """RoPE rotation with precomputed sin/cos.

    Use this when the same angles are applied to multiple tensors — compute
    sin/cos once outside and pass them in to avoid redundant elementwise ops.

    x:     (..., N, R)
    sin_a: (..., N//2, 1)  sin(angles), already cast to x.dtype
    cos_a: (..., N//2, 1)  cos(angles), already cast to x.dtype
    """
    N_half = sin_a.shape[-2]
    R = x.shape[-1]
    x_r = x.reshape(*x.shape[:-2], N_half, 2, R)
    x1, x2 = x_r[..., 0, :], x_r[..., 1, :]
    return mx.stack(
        [x1 * cos_a - x2 * sin_a, x2 * cos_a + x1 * sin_a], axis=-2
    ).reshape(*x.shape)


def apply_rope(x, angles):
    """x: (..., N, R); angles: (..., N//2).

    Computes sin/cos internally. When the same angles are applied to two
    tensors (e.g., B and C projections), prefer apply_rope_with_sc to share
    the sin/cos computation.
    """
    sin_a = mx.sin(angles)[..., None].astype(x.dtype)
    cos_a = mx.cos(angles)[..., None].astype(x.dtype)
    return apply_rope_with_sc(x, sin_a, cos_a)
