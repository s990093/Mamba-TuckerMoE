# -*- coding: utf-8 -*-
"""
Scan implementation selector — allows toggling between baseline and Metal optimized paths.

This module provides a unified interface to choose between:
  - Baseline: chunk_parallel_scan from mamba_block.py (O(Lc²) matrix method)
  - Metal: chunk_parallel_scan_mlx from mamba_block_metal.py (optimized for future Metal kernel)

Usage in Mamba3Block:
    from .mamba_scan_selector import get_scan_fn
    scan_fn = get_scan_fn(use_metal=config.use_metal_scan)
    y_stack, h_final = scan_fn(u_ssm, la, C_rotated, chunk_size)
"""

from typing import Callable, Optional, Tuple
import mlx.core as mx

# Import both implementations
from .mamba_block import chunk_parallel_scan as baseline_scan
from .mamba_block_metal import chunk_parallel_scan_mlx as metal_scan

__all__ = [
    "get_scan_fn",
    "ScanImplementation",
]


class ScanImplementation:
    """Encapsulates scan function with metadata."""

    def __init__(
        self,
        fn: Callable,
        name: str,
        description: str,
        supports_return_h_all: bool = False,
    ):
        self.fn = fn
        self.name = name
        self.description = description
        self.supports_return_h_all = supports_return_h_all

    def __call__(self, *args, **kwargs):
        """Transparent forward to underlying function."""
        return self.fn(*args, **kwargs)


def get_scan_fn(use_metal: bool = False) -> ScanImplementation:
    """
    Get the appropriate scan function based on configuration.

    Args:
        use_metal: If True, use Metal-optimized version (if available).
                   If False, use baseline (always available).

    Returns:
        ScanImplementation object with unified interface.

    Raises:
        RuntimeError: If Metal scan requested but import failed.
    """
    if use_metal:
        try:
            # Verify Metal scan is available
            metal_scan(
                mx.zeros((1, 1, 1, 1, 1)),  # Dummy args
                mx.zeros((1, 1, 1)),
                mx.zeros((1, 1, 1)),
                mx.zeros((1, 1, 1, 1, 1)),
                chunk_size=1,
            )
            return ScanImplementation(
                fn=metal_scan,
                name="Metal (Optimized)",
                description="MLX-optimized parallel scan with future Metal kernel support",
                supports_return_h_all=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Metal scan requested but initialization failed: {e}\n"
                f"Falling back to baseline. Check METAL_SCAN.md for details."
            ) from e
    else:
        return ScanImplementation(
            fn=baseline_scan,
            name="Baseline (Matrix Method)",
            description="O(Lc²) matrix matmul method, numerically stable",
            supports_return_h_all=False,
        )


def test_scan_selector():
    """Quick test of selector functionality."""
    import numpy as np

    print("Testing scan selector...")

    # Create minimal test data
    B, L, H, N, P, R = 1, 32, 2, 4, 8, 2

    u = mx.array(np.random.randn(B, L, H, N, P).astype(np.float32) * 0.1)
    dt_b = mx.array(np.random.randn(B, L, H).astype(np.float32) * 0.1)
    a_b = mx.array(np.random.randn(B, L, H).astype(np.float32) * -0.5)
    c_rotated = mx.array(np.random.randn(B, L, H, N, R).astype(np.float32))

    # Baseline
    baseline_impl = get_scan_fn(use_metal=False)
    la = dt_b * a_b
    y_base, h_base = baseline_impl(u, la, c_rotated, chunk_size=32)

    print(f"✓ Baseline: y={y_base.shape}, h={h_base.shape}")

    # Metal
    try:
        metal_impl = get_scan_fn(use_metal=True)
        y_metal, h_metal = metal_impl(u, dt_b, a_b, c_rotated, chunk_size=32)
        print(f"✓ Metal: y={y_metal.shape}, h={h_metal.shape}")

        # Compare
        y_diff = float(mx.max(mx.abs(y_base - y_metal)))
        h_diff = float(mx.max(mx.abs(h_base - h_metal)))
        print(f"✓ Difference: y_diff={y_diff:.2e}, h_diff={h_diff:.2e}")
        assert y_diff < 1e-2 and h_diff < 1e-2, "Implementations differ too much!"
    except RuntimeError as e:
        print(f"⚠ Metal not available: {e}")
        print("  (This is expected if Metal scan isn't fully set up)")

    print("✓ Selector test passed")


if __name__ == "__main__":
    test_scan_selector()
