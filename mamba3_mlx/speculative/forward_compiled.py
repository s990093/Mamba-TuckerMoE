"""``mx.compile``-fused verify forward.

Same math as :mod:`.forward`, but each Mamba layer's per-position step is
wrapped in ``mx.compile`` so that all of its internal ops (norm + in_proj +
softplus + cumsum + einsums + scan + out_proj) fuse into one Metal dispatch
batch instead of ~50.  fp32 byte-equality with the uncompiled path is
preserved — ``mx.compile`` does not change the math, only fuses dispatch.

For the bf16 Mamba3 model this collapses ~50 ms of MLX dispatch overhead
per verify round (24 Mamba layers × ~50 ops × ~50 µs CPU launch cost).

Two compiled variants per Mamba layer:
* ``cold``  — first round, state = None (zeros).
* ``warm``  — subsequent rounds, state has ``h_prev`` / ``prev_input_signal``
              / ``angles_cum`` arrays.

We split on state-presence rather than special-casing inside one compiled
function because ``mx.compile`` traces structure: a single function that
sometimes receives ``None`` would force per-call dispatch on which branch
to use, defeating the fusion.

Transformer layers are NOT compiled here — their KV cache grows by
``K`` tokens per round so the input shape to the attention varies between
rounds (1, K+1, 2K+1, …).  ``mx.compile`` would retrace every round.  A
separate fixed-size-KV compiled path (Step 3 of the optimisation plan) is
needed there.
"""
from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx

from ..mlx_model.mamba_block import Mamba3Block
from ..mlx_model.ops import scaled_tanh
from .forward import _MAMBA_BLOCK_TYPES, mamba_verify_step


def _make_compiled_mamba_step_cold(blk) -> Callable:
    """Returns ``(x) -> (out, h_prev, prev_input_signal, angles_cum)`` for
    state-less first round.  Compiled once per layer; reused thereafter."""

    def _step(x):
        out, sp = mamba_verify_step(blk, x, state=None)
        return out, sp["h_prev"], sp["prev_input_signal"], sp["angles_cum"]

    return mx.compile(_step)


def _make_compiled_mamba_step_warm(blk) -> Callable:
    """Returns ``(x, h_prev, prev_input_signal, angles_cum)
    -> (out, h_prev_new, prev_input_signal_new, angles_cum_new)`` for rounds
    where prior state must be threaded in.  Compiled once per layer."""

    def _step(x, h_prev, prev_input_signal, angles_cum):
        state = {
            "h_prev": h_prev,
            "prev_input_signal": prev_input_signal,
            "angles_cum": angles_cum,
        }
        out, sp = mamba_verify_step(blk, x, state=state)
        return out, sp["h_prev"], sp["prev_input_signal"], sp["angles_cum"]

    return mx.compile(_step)


def make_compiled_model_verify_forward(model) -> Callable:
    """Build a verify-forward function with Mamba AND Transformer layers JIT-fused.

    Drop-in replacement for :func:`mamba3_mlx.speculative.forward.model_verify_forward`.
    Compilation work happens lazily on first call per (layer, shape) pair; subsequent
    calls with the same shapes reuse the cached graphs.

    Mamba layers: each layer's verify step is compiled (cold + warm variants).
    Transformer layers: _decode_pre and _decode_post are compiled for verify L=K
      (MLX auto-retraces for each new L on first use).  The non-compilable middle
      section (KV concat + SDPA) remains eager — it is only 3 kernel dispatches.
    Head: uses model._compiled_head when available (auto-retraces for verify L=K).

    Requires ``model.precompute()`` to have been called so that TransformerBlock
    ``_compiled_pre`` / ``_compiled_post`` and ``model._compiled_head`` are set.

    Returns
    -------
    fn(model, input_ids, states=None) -> (logits, payload)
        Signature-identical to :func:`model_verify_forward`.
    """
    cold_steps: dict[int, Callable] = {}
    warm_steps: dict[int, Callable] = {}
    for i, blk in enumerate(model.backbone.layers):
        if isinstance(blk, _MAMBA_BLOCK_TYPES):
            cold_steps[i] = _make_compiled_mamba_step_cold(blk)
            warm_steps[i] = _make_compiled_mamba_step_warm(blk)

    def _verify(_model, input_ids, states=None):
        x = model.embed(input_ids)
        payload: list[dict] = []
        L_q = input_ids.shape[1]

        for i, blk in enumerate(model.backbone.layers):
            st = states[i] if states is not None else None
            if isinstance(blk, _MAMBA_BLOCK_TYPES):
                if st is None:
                    x, h_prev, prev_in, angles = cold_steps[i](x)
                else:
                    x, h_prev, prev_in, angles = warm_steps[i](
                        x, st["h_prev"], st["prev_input_signal"], st["angles_cum"],
                    )
                payload.append({
                    "kind": "mamba",
                    "h_prev": h_prev,
                    "prev_input_signal": prev_in,
                    "angles_cum": angles,
                })
            else:
                # TransformerBlock verify — use compiled pre/post when available.
                S_past = 0
                if st is not None and st.get("k") is not None:
                    S_past = int(st["k"].shape[2])

                if blk._compiled_pre is not None:
                    # compiled_pre / compiled_post auto-retrace for L=K on first call;
                    # subsequent rounds with same L reuse the cached graph.
                    q, k_new, v_new = blk._compiled_pre(x)

                    if st is not None and st.get("k") is not None:
                        k_full = mx.concatenate([st["k"], k_new], axis=2)
                        v_full = mx.concatenate([st["v"], v_new], axis=2)
                    else:
                        k_full = k_new
                        v_full = v_new

                    S = k_full.shape[2]
                    if L_q == 1:
                        attn = mx.fast.scaled_dot_product_attention(
                            q, k_full, v_full, scale=blk.scale)
                    else:
                        # Causal mask: query i attends keys 0..S-L_q+i
                        mask = mx.full((L_q, S), -mx.inf, dtype=q.dtype)
                        i_idx = mx.arange(L_q)[:, None]
                        j_idx = mx.arange(S)[None, :]
                        mask = mx.where(
                            j_idx <= (S - L_q + i_idx),
                            mx.zeros_like(mask), mask,
                        )
                        attn = mx.fast.scaled_dot_product_attention(
                            q, k_full, v_full, scale=blk.scale, mask=mask)

                    x = blk._compiled_post(attn, x)
                    payload.append({"kind": "tf", "k": k_full, "v": v_full,
                                    "S_past": S_past})
                else:
                    # Fallback: uncompiled path (precompute not called).
                    x, new_st = blk(x, state=st)
                    payload.append({"kind": "tf", "k": new_st["k"],
                                    "v": new_st["v"], "S_past": S_past})

        # Head: use compiled head when available (auto-retraces for verify L=K).
        if getattr(model, "_compiled_head", None) is not None:
            logits = model._compiled_head(x)
        else:
            h = model.norm(x)
            logits = model.head(h * model.inv_sqrt_d).astype(mx.float32)
            logits = scaled_tanh(logits, 30.0)
        return logits, payload

    return _verify
