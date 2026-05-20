"""
解耦合的融合优化管理器。

提供可选的融合内核优化，不影响原始推理路径：
- 可以独立启用/禁用每个优化
- 默认使用标准 MLX 实现
- 支持动态启用/禁用
"""

import mlx.core as mx
from typing import Optional, Dict, Any


class FusionOptimizer:
    """融合优化管理器 - 完全解耦合的优化策略。"""

    def __init__(self):
        """初始化优化管理器。"""
        self.enabled = {
            "moe_fusion": False,           # TuckerMoE 融合
            "ssm_rope_fusion": False,      # SSM + RoPE 融合
            "kv_cache_fusion": False,      # KV 缓存优化
        }
        self.metal_available = self._check_metal_available()

    def _check_metal_available(self) -> bool:
        """检查 Metal kernel 支持。"""
        try:
            # Check if mx.metal_kernel or similar is available
            # For now, assume True (fallback to MLX if needed)
            return True
        except Exception:
            return False

    def enable(self, fusion_type: str) -> None:
        """启用特定的融合优化。

        Args:
            fusion_type: "moe_fusion", "ssm_rope_fusion", 或 "kv_cache_fusion"
        """
        if fusion_type in self.enabled:
            self.enabled[fusion_type] = True
            print(f"✓ Enabled: {fusion_type}")
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def disable(self, fusion_type: str) -> None:
        """禁用特定的融合优化。"""
        if fusion_type in self.enabled:
            self.enabled[fusion_type] = False
            print(f"✓ Disabled: {fusion_type}")

    def enable_all(self) -> None:
        """启用所有优化。"""
        for key in self.enabled:
            self.enabled[key] = True
        print("✓ All optimizations enabled")

    def disable_all(self) -> None:
        """禁用所有优化（回到基准路径）。"""
        for key in self.enabled:
            self.enabled[key] = False
        print("✓ All optimizations disabled (baseline mode)")

    def is_enabled(self, fusion_type: str) -> bool:
        """检查融合优化是否启用。"""
        return self.enabled.get(fusion_type, False)

    def get_status(self) -> Dict[str, Any]:
        """获取优化器状态。"""
        return {
            "enabled": self.enabled.copy(),
            "metal_available": self.metal_available,
            "summary": f"{sum(self.enabled.values())}/{len(self.enabled)} optimizations active",
        }

    def print_status(self) -> None:
        """打印优化器状态。"""
        print("=" * 70)
        print("Fusion Optimization Status")
        print("=" * 70)
        for opt_name, is_enabled in self.enabled.items():
            status = "✓ ENABLED" if is_enabled else "✗ DISABLED"
            print(f"  {opt_name:<25} {status}")
        print(f"Metal available: {'Yes' if self.metal_available else 'No'}")
        print("=" * 70)


# Global singleton optimizer instance
_fusion_optimizer = FusionOptimizer()


def get_fusion_optimizer() -> FusionOptimizer:
    """获取全局融合优化管理器实例。"""
    return _fusion_optimizer


class SSMRoPEFusion:
    """SSM + RoPE 融合实现 (可选)。"""

    @staticmethod
    def rope_input_signal_fused(
        B_param: mx.array,          # (B, L, G, N*R)
        x_ssm: mx.array,            # (B, L, H, P, R)
        angles: mx.array,           # (B, L, H, N//2)
        bias_B: mx.array,           # (G, N, R)
        G: int,
        N: int,
        P: int,
        R: int,
        ratio: int,                 # H / G
    ) -> tuple:
        """RoPE + Input Signal 融合计算。

        Returns:
            (B_rotated, input_signal)
        """
        B, L, _, _ = B_param.shape
        H = G * ratio

        # Step 1: Apply RoPE to B parameter
        B_normed = B_param.reshape(B, L, G, N, R) + bias_B[None, None, :, :, :]
        B_broad = mx.repeat(B_normed, ratio, axis=2)  # (B, L, H, N, R)

        # Apply RoPE rotation
        from .ops import apply_rope
        B_rotated = apply_rope(B_broad, angles)  # (B, L, H, N, R)

        # Step 2: Compute input_signal
        # input_signal[b,l,h,n,p] = sum_r B_rotated[b,l,h,n,r] * x_ssm[b,l,h,p,r]
        input_signal = mx.einsum("blhnr, blhpr -> blhnp", B_rotated, x_ssm)

        return B_rotated, input_signal

    @staticmethod
    def ssm_discretization_fused(
        input_signal: mx.array,     # (B, L, H, N, P)
        dt_b: mx.array,             # (B, L, H)
        A_b: mx.array,              # (H, N, N)
        lam_b: mx.array,            # (B, L, H)
        last_ip: Optional[mx.array] = None,  # (B, H, N, P) for decode
    ) -> mx.array:
        """SSM 梯形离散化融合。

        Fuses trapezoidal discretization:
        u_ssm[t] = lv*dv*is[t] + (1-lv)*dv*av*is[t-1]
        """
        B, L, H, N, P = input_signal.shape

        # Compute coefficients
        lv = mx.sigmoid(lam_b)  # (B, L, H)
        dv = dt_b  # (B, L, H)
        av = mx.exp(dt_b * A_b[0, 0])  # Simplified: use scalar approximation

        # Reshape for broadcasting
        lv = lv.reshape(B, L, H, 1, 1)
        dv = dv.reshape(B, L, H, 1, 1)
        av = av.reshape(1, 1, 1, 1, 1)

        # Shift input_signal right by 1
        if last_ip is not None:
            # Decode: use saved previous state
            ip_prev = last_ip[:, None, :, :, :]  # (B, 1, H, N, P)
        else:
            # Prefill: pad with zeros
            ip_prev = mx.pad(input_signal[:, :-1], [(0, 0), (1, 0), (0, 0), (0, 0), (0, 0)])

        # Apply trapezoidal discretization
        u_ssm = lv * dv * input_signal + (1.0 - lv) * dv * av * ip_prev

        return u_ssm

    @staticmethod
    def ssm_scan_fused(
        u_ssm: mx.array,            # (B, L, H, N, P)
        la: mx.array,               # (B, L, H) = dt * A
        C_rotated: mx.array,        # (B, L, H, N, R)
        h_prev: Optional[mx.array] = None,  # (B, H, N, P)
    ) -> tuple:
        """融合的 SSM 扫描和输出投影。

        Returns:
            (y_stack, h_final)
        """
        B, L, H, N, P = u_ssm.shape
        _, _, _, R = C_rotated.shape

        # Single-step implementation for decode
        if L == 1 and h_prev is not None:
            # Decode path: h_new = exp(la) * h_prev + u_ssm
            u_t = u_ssm[:, 0, :, :, :]  # (B, H, N, P)
            la_t = la[:, 0, :]  # (B, H)

            la_t = la_t.reshape(B, H, 1, 1)
            h_new = mx.exp(la_t) * h_prev + u_t

            # Output: y = einsum("bhnp, bhnr -> bhpr", h_new, C_t)
            C_t = C_rotated[:, 0, :, :, :]  # (B, H, N, R)
            y_stack = mx.einsum("bhnp, bhnr -> bhpr", h_new, C_t)
            y_stack = y_stack[:, None, :, :, :]  # (B, 1, H, P, R)

            return y_stack, h_new

        # Prefill path: standard chunk scan (delegate to original implementation)
        # This is kept as-is to avoid breaking existing logic
        return None, None  # Signal to use default implementation


class KVCacheFusion:
    """KV 缓存优化 (可选)。"""

    @staticmethod
    def materialize_kv_cache(
        key: mx.array,              # (B, L, H, D)
        value: mx.array,            # (B, L, H, D)
        past_key: Optional[mx.array] = None,
        past_value: Optional[mx.array] = None,
    ) -> tuple:
        """物化 KV 缓存 - 保持在 GPU 内存中而不是动态计算。

        Returns:
            (key_cache, value_cache)
        """
        if past_key is not None:
            key_cache = mx.concatenate([past_key, key], axis=1)
        else:
            key_cache = key

        if past_value is not None:
            value_cache = mx.concatenate([past_value, value], axis=1)
        else:
            value_cache = value

        return key_cache, value_cache

    @staticmethod
    def attention_with_cached_kv(
        query: mx.array,            # (B, L, H, D)
        key_cache: mx.array,        # (B, L_total, H, D)
        value_cache: mx.array,      # (B, L_total, H, D)
    ) -> mx.array:
        """使用缓存的 KV 执行注意力。"""
        # Standard scaled dot-product attention
        # scores = Q @ K^T / sqrt(d)
        # attn = softmax(scores)
        # out = attn @ V

        d = query.shape[-1]
        scores = mx.matmul(query, key_cache.transpose(0, 1, 3, 2)) / mx.sqrt(mx.array(d, dtype=query.dtype))
        attn = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn, value_cache)

        return output


# Export functions
def apply_fusion_optimization(
    x: mx.array,
    fusion_type: str,
    **kwargs,
) -> Optional[mx.array]:
    """应用融合优化（如果启用）。

    Args:
        x: 输入数据
        fusion_type: 优化类型
        **kwargs: 优化特定的参数

    Returns:
        优化后的结果，或 None 如果未启用
    """
    optimizer = get_fusion_optimizer()

    if not optimizer.is_enabled(fusion_type):
        return None

    if fusion_type == "moe_fusion":
        # TODO: Implement MoE fusion callback
        pass
    elif fusion_type == "ssm_rope_fusion":
        # TODO: Implement SSM+RoPE fusion callback
        pass
    elif fusion_type == "kv_cache_fusion":
        # TODO: Implement KV cache fusion callback
        pass

    return None


__all__ = [
    "FusionOptimizer",
    "get_fusion_optimizer",
    "SSMRoPEFusion",
    "KVCacheFusion",
    "apply_fusion_optimization",
]
