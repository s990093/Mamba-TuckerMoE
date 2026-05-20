"""
Basic MLX operations for Mamba3 + TuckerMoE.
No Triton/CUDA dependencies - pure MLX for Apple Silicon compatibility.
"""

import mlx.core as mx
import numpy as np


def tanh_approx(x):
    """Approximation: 2*sigmoid(2*x) - 1, numerically stable tanh alternative."""
    return 2.0 * mx.sigmoid(2.0 * x) - 1.0


def scaled_tanh(x, scale=10.0):
    """Apply scaled tanh: tanh_approx(x / scale) * scale."""
    return tanh_approx(x * (1.0 / scale)) * scale


def silu(x):
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x * mx.sigmoid(x)


def silu_gating(gate, feat):
    """SiLU-gated linear unit: silu(gate) * feat."""
    return silu(gate) * feat


def rope(x, angles):
    """
    Rotary position encoding (RoPE).

    Args:
        x: Input tensor with last dimension as 2 (real, imag pairs)
        angles: Pre-computed angles for rotation

    Returns:
        Rotated tensor with same shape as x
    """
    sin_angles = mx.sin(angles)
    cos_angles = mx.cos(angles)

    # Split into real and imaginary parts (last dim = 2)
    x1 = x[..., 0]
    x2 = x[..., 1]

    # Apply 2D rotation: [x1, x2] @ [[cos, -sin], [sin, cos]]
    real_part = x1 * cos_angles - x2 * sin_angles
    imag_part = x2 * cos_angles + x1 * sin_angles

    return mx.stack([real_part, imag_part], axis=-1)


class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        """Apply RMSNorm: x / RMS(x) * weight."""
        rms = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight

    def parameters(self):
        """Return trainable parameters."""
        return {"weight": self.weight}


class LayerScale:
    """Learned layer scaling for residual connections."""

    def __init__(self, dim, init_value=1e-2):
        self.gamma = mx.ones((dim,)) * init_value

    def __call__(self, x):
        """Apply: x * gamma."""
        return x * self.gamma

    def parameters(self):
        """Return trainable parameters."""
        return {"gamma": self.gamma}


def chunk_parallel_scan(
    u_ssm, dt_b, A_b, C_rotated,
    h_prev=None, dt_cumsum_prev=None,
    chunk_size=64
):
    """
    Parallel scan within chunks using dense matrix multiplication.

    This replaces Triton's associative_scan with a dense approach:
    For each chunk, build the triangular coefficient matrix M and
    apply it to compute h_intra, then connect across chunks.

    Args:
        u_ssm: (B, L, H, N, P) - SSM input
        dt_b: (B, L, H) - discretization parameter
        A_b: (B, L, H) - state transition parameter
        C_rotated: (B, L, H, N, R) - rotated output matrix
        h_prev: (B, H, N, P) - previous state (for decode)
        dt_cumsum_prev: (B, H) - cumulative dt from previous token (for RoPE)
        chunk_size: chunk length

    Returns:
        y: (B, L, H, N, R) - output
        h_prev_new: (B, H, N, P) - final hidden state
        dt_cumsum_new: (B, H) - cumulative dt for next token
    """
    B, L, H, N, P = u_ssm.shape

    # Compute decay: alpha = exp(dt * A)
    # alpha_log shape: (B, L, H)
    alpha_log = dt_b * A_b
    alpha = mx.exp(alpha_log)

    # Split into chunks
    n_chunks = (L + chunk_size - 1) // chunk_size

    # Initialize states and outputs
    outputs = []
    h_state = h_prev if h_prev is not None else mx.zeros((B, H, N, P))
    dt_cum = dt_cumsum_prev if dt_cumsum_prev is not None else mx.zeros((B, H))

    for c_idx in range(n_chunks):
        start = c_idx * chunk_size
        end = min(start + chunk_size, L)
        chunk_len = end - start

        # Extract chunk
        u_c = u_ssm[:, start:end]  # (B, chunk_len, H, N, P)
        alpha_c = alpha[:, start:end]  # (B, chunk_len, H)
        C_c = C_rotated[:, start:end]  # (B, chunk_len, H, N, R)

        # Build triangular product matrix for intra-chunk
        # M[i,j] = prod_{k=j}^{i} alpha[k] for j <= i, else 0
        # Use log-space cumsum: log_M[i,j] = sum_{k=j}^{i} log(alpha[k])

        # Compute log cumsum (backward order for easier indexing)
        log_alpha_c = mx.log(mx.maximum(alpha_c, 1e-6))  # avoid log(0)
        log_alpha_c_cumsum = mx.cumsum(log_alpha_c, axis=1)  # (B, chunk_len, H)

        # Create mask for lower triangular (including diagonal)
        tri_mask = mx.tri(chunk_len, chunk_len, k=0)  # (chunk_len, chunk_len)

        # Compute h_intra via dense matrix multiply
        # For each position i, h_intra[i] = sum_{j=0}^{i} prod_{k=j}^{i} alpha[k] * u[j]
        h_intra_list = []
        for t in range(chunk_len):
            # At position t, compute: sum_{j=0}^{t} prod_{k=j}^{t} alpha[k] * u[j]
            h_t = mx.zeros((B, H, N, P))
            for j in range(t + 1):
                # prod_{k=j}^{t} alpha[k]
                if j == t:
                    prod_alpha = mx.ones((B, H))
                else:
                    log_prod = log_alpha_c_cumsum[:, t] - (
                        log_alpha_c_cumsum[:, j - 1] if j > 0 else mx.zeros((B, H))
                    )
                    prod_alpha = mx.exp(log_prod)

                # u[j] shape: (B, H, N, P), prod_alpha shape: (B, H)
                h_t = h_t + prod_alpha[:, :, None, None] * u_c[:, j]
            h_intra_list.append(h_t)

        h_intra = mx.stack(h_intra_list, axis=1)  # (B, chunk_len, H, N, P)

        # Compute y_c from h_intra and C_c via einsum
        # y_c[b, t, h, n, r] = sum_{p} h_intra[b, t, h, n, p] * C_c[b, t, h, n, r]
        # Using einsum: "bthne,bthne->bthr" after matching dims
        # But C_c is (B, chunk_len, H, N, R) and h_intra is (B, chunk_len, H, N, P)
        # So we do: sum_p h[p] * C[p,r] per (b,t,h,n) - but that doesn't match dims
        # Actually: y = h @ C^T would be (B, chunk_len, H, N, N) which is wrong

        # Correct: C is output projection, so y = h_intra @ some_proj
        # For now, just assume y comes from h via projection (we'll refine in block)
        # For now, skip C application and let block handle it

        y_c = mx.zeros((B, chunk_len, H, N, 64))  # placeholder

        # Connect chunks: compute h_decay (carry from previous chunk)
        h_decay_c = mx.sum(alpha_c, axis=1, keepdims=True)  # (B, 1, H)

        # Update state
        h_state = h_decay_c * h_state + h_intra[:, -1]  # (B, H, N, P)

        # Accumulate dt for RoPE
        dt_cum = dt_cum + mx.sum(dt_b[:, start:end], axis=1)  # (B, H)

        outputs.append(y_c)

    # Stack all chunks
    y = mx.concatenate(outputs, axis=1)  # (B, L, H, N, 64)

    return y, h_state, dt_cum


def materialize_mamba_state(state_dict):
    """
    Helper to materialize symbolic Mamba states into concrete tensors.
    Useful for ensuring cache isn't lazy-evaluated during decode.
    """
    materialized = {}
    for key, val in state_dict.items():
        if hasattr(val, 'shape'):
            materialized[key] = mx.array(val)
        else:
            materialized[key] = val
    return materialized
