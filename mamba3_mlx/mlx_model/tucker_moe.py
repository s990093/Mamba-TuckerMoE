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
        out_core = sum_k probs_k * (x_shared @ G[top_k_idx_k])      # (B, r2)
        out      = out_core @ U_out + bias                           # (B, d_out)
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

    def __call__(self, x, temperature: float = 0.5):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])           # (B_flat, dim_in)
        dtype = x_flat.dtype

        raw_logits = self.router(x_flat)                  # (B_flat, E)
        capped = scaled_tanh(raw_logits.astype(mx.float32), 10.0)
        router_logits = capped / temperature
        router_probs = mx.softmax(router_logits, axis=-1)

        top_k_idx = mx.argpartition(-router_logits, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        top_k_raw = mx.take_along_axis(router_probs, top_k_idx, axis=-1)
        top_k_probs = top_k_raw / (mx.sum(top_k_raw, axis=-1, keepdims=True) + 1e-6)

        x_shared = mx.matmul(x_flat, self.U_in.astype(dtype))   # (B_flat, r3)
        x_shared = self.inner_norm(x_shared)

        # G_experts: (E, r3, r2)
        G_experts = mx.einsum("er,rst->est", self.U_expert.astype(mx.float32),
                              self.core.astype(mx.float32)).astype(dtype)

        # Fancy gather: G_selected (B_flat, top_k, r3, r2)
        G_selected = G_experts[top_k_idx]                  # (B_flat, k, r3, r2)
        # einsum: (B,r3) and (B,k,r3,r2) -> (B,k,r2)
        per_expert = mx.einsum("br,bkrs->bks", x_shared, G_selected)
        weighted = per_expert * top_k_probs.astype(dtype)[..., None]
        x_core = mx.sum(weighted, axis=1)                  # (B_flat, r2)

        out = mx.matmul(x_core, self.U_out.astype(dtype))  # (B_flat, dim_out)
        out = out + self.bias.astype(dtype)
        return out.reshape(*orig_shape[:-1], self.dim_out)
