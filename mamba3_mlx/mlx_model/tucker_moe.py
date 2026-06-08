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

    Prefill acceleration (B_flat > top_k * num_experts):
        Instead of gathering (B, r3, r2) per expert k — which at L=4096 allocates
        a 1 GB intermediate and dominates prefill latency — we do ONE large matmul:
            x_shared @ G_flat → (B_flat, E*r2), reshape to (B_flat, E, r2)
        where G_flat = G.transpose(1,0,2).reshape(r3, E*r2) is precomputed at load time.
        Memory peak: (B_flat, E*r2) ≈ 32 MB at L=4096 vs 1 GB for the old gather path.

    Decode acceleration (B_flat == 1, mx.compile):
        After precompute_G_experts(), the internal _forward method is compiled with
        mx.compile.  This eliminates ~25 separate Metal kernel dispatches per call
        (Python graph-building overhead) and reduces to a handful of fused dispatches.
        Speedup measured: ~1.8–2.2× per TuckerMoE call at B_flat=1.
    """

    def __init__(self, dim_in, dim_out, num_experts=8, top_k=2, r1=32, r2=512, r3=256,
                 eps: float = 1e-5):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self._r2 = r2
        self._r3 = r3

        self.router = nn.Linear(dim_in, num_experts, bias=False)
        self.U_expert = mx.zeros((num_experts, r1))
        self.U_in = mx.zeros((dim_in, r3))
        self.U_out = mx.zeros((r2, dim_out))
        self.core = mx.zeros((r1, r3, r2))
        self.bias = mx.zeros((dim_out,))
        self.inner_norm = RMSNorm(r3, eps=eps)

        # Populated by precompute_G_experts() after weight loading.
        self._G_experts_cache: mx.array | None = None   # (E, r3, r2) float32 — decode path
        self._G_flat_cache: mx.array | None = None      # (r3, E*r2) float32 — prefill path

        # bf16 versions — eliminate per-call float32→bf16 cast (4 MB each).
        self._G_experts_cache_bf16: mx.array | None = None
        self._G_flat_cache_bf16: mx.array | None = None

        # Compiled decode forward — set by precompute_G_experts(), used in __call__.
        self._compiled_call: object | None = None

    def precompute_G_experts(self) -> None:
        """Precompute G = U_expert ⊗ core, its flat form, bf16 variants, and the
        compiled decode-path forward.

        Call once after model weights are loaded.
          _G_experts_cache:      (E, r3, r2) float32 — fallback decode path
          _G_flat_cache:         (r3, E*r2)  float32 — fallback prefill path
          _G_experts_cache_bf16: (E, r3, r2) bfloat16 — fast decode path
          _G_flat_cache_bf16:    (r3, E*r2)  bfloat16 — fast prefill path
          _compiled_call:        mx.compile(self._forward) for fixed-shape dispatch
        """
        G = mx.einsum(
            "er,rst->est",
            self.U_expert.astype(mx.float32),
            self.core.astype(mx.float32),
        )  # (E, r3, r2)
        G_flat = G.transpose(1, 0, 2).reshape(self._r3, self.num_experts * self._r2)
        mx.eval(G, G_flat)
        self._G_experts_cache = G
        self._G_flat_cache = G_flat

        G_bf16   = G.astype(mx.bfloat16)
        G_fl_bf16 = G_flat.astype(mx.bfloat16)
        mx.eval(G_bf16, G_fl_bf16)
        self._G_experts_cache_bf16 = G_bf16
        self._G_flat_cache_bf16   = G_fl_bf16

        # Compile the forward implementation.  This eliminates per-call Python
        # overhead (~25 MLX op dispatches → 1 compiled program).
        self._compiled_call = mx.compile(self._forward)
        # Eager warmup so the first real decode step is fast.
        _dummy = mx.zeros((1, self.dim_in), dtype=mx.bfloat16)
        mx.eval(self._compiled_call(_dummy))

    def _forward(self, x, temperature: float = 0.5):
        """Implementation — called both directly and via the compiled wrapper."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])           # (B_flat, dim_in)
        B_flat = x_flat.shape[0]
        dtype = x_flat.dtype
        E = self.num_experts
        r2 = self._r2

        raw_logits = self.router(x_flat)                  # (B_flat, E)
        capped = scaled_tanh(raw_logits.astype(mx.float32), 10.0)
        router_logits = capped / temperature
        router_probs = mx.softmax(router_logits, axis=-1)

        top_k_idx = mx.argpartition(-router_logits, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        top_k_raw = mx.take_along_axis(router_probs, top_k_idx, axis=-1)
        top_k_probs = top_k_raw / (mx.sum(top_k_raw, axis=-1, keepdims=True) + 1e-6)

        x_shared = mx.matmul(x_flat, self.U_in.astype(dtype))   # (B_flat, r3)
        x_shared = self.inner_norm(x_shared)

        probs_t = top_k_probs.astype(dtype)

        if B_flat > self.top_k * E:
            # ── Prefill path ─────────────────────────────────────────────────
            if self._G_flat_cache_bf16 is not None and dtype == mx.bfloat16:
                G_flat = self._G_flat_cache_bf16
            elif self._G_flat_cache is not None:
                G_flat = self._G_flat_cache.astype(dtype)
            else:
                G = mx.einsum(
                    "er,rst->est",
                    self.U_expert.astype(mx.float32),
                    self.core.astype(mx.float32),
                ).astype(dtype)
                G_flat = G.transpose(1, 0, 2).reshape(self._r3, E * r2)

            xG_flat = mx.matmul(x_shared, G_flat)                # (B_flat, E*r2)
            xG_be = xG_flat.reshape(B_flat, E, r2)               # (B_flat, E, r2)
            gathered = mx.take_along_axis(xG_be, top_k_idx[:, :, None], axis=1)
            x_core = mx.sum(gathered * probs_t[:, :, None], axis=1)
        else:
            # ── Decode path ──────────────────────────────────────────────────
            if self._G_experts_cache_bf16 is not None and dtype == mx.bfloat16:
                G_experts = self._G_experts_cache_bf16
            elif self._G_experts_cache is not None:
                G_experts = self._G_experts_cache.astype(dtype)
            else:
                G_experts = mx.einsum(
                    "er,rst->est",
                    self.U_expert.astype(mx.float32),
                    self.core.astype(mx.float32),
                ).astype(dtype)

            G_weighted = G_experts[top_k_idx[:, 0]] * probs_t[:, 0:1, None]
            for k in range(1, self.top_k):
                G_weighted = G_weighted + G_experts[top_k_idx[:, k]] * probs_t[:, k:k+1, None]
            x_core = mx.einsum("br,brs->bs", x_shared, G_weighted)

        out = mx.addmm(self.bias.astype(dtype), x_core, self.U_out.astype(dtype))
        return out.reshape(*orig_shape[:-1], self.dim_out)

    def __call__(self, x, temperature: float = 0.5):
        if self._compiled_call is not None and temperature == 0.5:
            return self._compiled_call(x, temperature)
        return self._forward(x, temperature)
