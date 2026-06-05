import mlx.core as mx
import mlx.nn as nn

from .ops import RMSNorm, scaled_tanh


class TuckerMoE(nn.Module):
    """Vectorised Tucker-decomposed MoE — matches sft_cot_bundle/scripts/model.py TritonTuckerMoE.

    Forward (no losses, inference only):
        logits  = router(x)
        capped  = scaled_tanh(logits, 10.0)
        probs   = softmax(capped / temp)
        top_k_idx, top_k_probs (normalised)
        x_shared = inner_norm(x @ U_in)
        G        = einsum("er,rst->est", U_expert, core)            # (E, r3, r2)
        G_w      = sum_k probs_k * G[top_k_idx_k]                   # (B, r3, r2)
        out_core = einsum("br,brs->bs", x_shared, G_w)              # (B, r2)
        out      = addmm(bias, out_core, U_out)                      # (B, d_out)

    Acceleration notes:
        - G_experts is precomputed once after weight loading (call precompute_G_experts()).
          This avoids re-running the Tucker einsum on every decode step.
        - Expert weighting is done on G matrices before the x contraction (1 matmul vs k).
        - mx.addmm fuses the final bias addition with the U_out matmul.
    """

    def __init__(self, dim_in, dim_out, num_experts=8, top_k=2, r1=32, r2=512, r3=256,
                 eps: float = 1e-5):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.router = nn.Linear(dim_in, num_experts, bias=False)
        self.U_expert = mx.zeros((num_experts, r1))
        self.U_in = mx.zeros((dim_in, r3))
        self.U_out = mx.zeros((r2, dim_out))
        self.core = mx.zeros((r1, r3, r2))
        self.bias = mx.zeros((dim_out,))
        self.inner_norm = RMSNorm(r3, eps=eps)
        # Populated by precompute_G_experts() after weight loading.
        self._G_experts_cache: mx.array | None = None

    def precompute_G_experts(self) -> None:
        """Precompute and cache the Tucker expert matrices G = U_expert ⊗ core.

        Call once after model weights are loaded. The result is stored in float32
        and cast to the input dtype on each forward pass. After calling this,
        the expensive Tucker einsum is skipped during every decode step.
        """
        G = mx.einsum(
            "er,rst->est",
            self.U_expert.astype(mx.float32),
            self.core.astype(mx.float32),
        )
        mx.eval(G)
        self._G_experts_cache = G

    def __call__(self, x, temperature: float = 0.5):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])           # (B_flat, dim_in)
        dtype = x_flat.dtype

        raw_logits = self.router(x_flat)                  # (B_flat, E)
        capped = scaled_tanh(raw_logits.astype(mx.float32), 10.0)
        router_logits = capped / temperature
        router_probs = mx.softmax(router_logits, axis=-1)

        top_k_idx = mx.argpartition(-router_logits, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        top_k_raw = mx.take_along_axis(router_probs, top_k_idx, axis=-1)
        top_k_probs = top_k_raw / (mx.sum(top_k_raw, axis=-1, keepdims=True) + 1e-6)

        x_shared = mx.matmul(x_flat, self.U_in.astype(dtype))   # (B_flat, r3)
        x_shared = self.inner_norm(x_shared)

        # Use precomputed G_experts when available, otherwise compute on the fly.
        if self._G_experts_cache is not None:
            G_experts = self._G_experts_cache.astype(dtype)     # (E, r3, r2)
        else:
            G_experts = mx.einsum(
                "er,rst->est",
                self.U_expert.astype(mx.float32),
                self.core.astype(mx.float32),
            ).astype(dtype)

        # Accumulate top_k experts one at a time to avoid the (B, top_k, r3, r2)
        # intermediate, which at prefill lengths dominates peak memory usage
        # (e.g. 52 MB per TuckerMoE at L=100, ×66 layers = 3.4 GB).
        probs_t = top_k_probs.astype(dtype)
        G_weighted = G_experts[top_k_idx[:, 0]] * probs_t[:, 0:1, None]
        for k in range(1, self.top_k):
            G_weighted = G_weighted + G_experts[top_k_idx[:, k]] * probs_t[:, k:k+1, None]
        # G_weighted: (B_flat, r3, r2)
        x_core = mx.einsum("br,brs->bs", x_shared, G_weighted)  # (B_flat, r2)

        # Fused matmul + bias: addmm(bias, x_core, U_out) = bias + x_core @ U_out
        out = mx.addmm(self.bias.astype(dtype), x_core, self.U_out.astype(dtype))
        return out.reshape(*orig_shape[:-1], self.dim_out)