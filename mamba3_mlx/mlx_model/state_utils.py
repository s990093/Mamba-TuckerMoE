# -*- coding: utf-8 -*-
"""
Mamba3 state factory and clone utilities — single source of truth.

All code that creates or copies mamba_states / kv_caches must go through
these functions.  This prevents schema drift: the canonical key-set is
defined once here and enforced by _assert_state_schema in generator.py.

Schema
------
Each element in mamba_states is a dict with exactly these keys:
  "h"       : (B, H, N, P)   — SSM hidden state
  "angles"  : (B, H, N//2)   — cumulative RoPE phase angles
  "last_ip" : (B, H, N, P)   — input_signal from the PREVIOUS step
                                used for trapezoidal discretisation (coeff ≈ 0.95)

Each element in kv_caches is a dict with:
  "k" : (B, kv_heads, T, D) | None
  "v" : (B, kv_heads, T, D) | None
"""
from typing import List, Optional
import mlx.core as mx

# Canonical key-set — imported by generator.py for schema validation
MAMBA_STATE_KEYS: frozenset = frozenset({"h", "angles", "last_ip"})


# ── Single-state factories ─────────────────────────────────────────────────────

def make_mamba_state(batch_size: int, H: int, N: int, P: int,
                     dtype=mx.float32) -> dict:
    """Zero-initialised state for one Mamba layer."""
    return {
        "h":       mx.zeros((batch_size, H, N, P),    dtype=dtype),
        "angles":  mx.zeros((batch_size, H, N // 2),  dtype=dtype),
        "last_ip": mx.zeros((batch_size, H, N, P),    dtype=dtype),
    }


def make_kv_cache() -> dict:
    """Empty KV-cache slot for one Transformer layer."""
    return {"k": None, "v": None}


# ── Full-model state initialisation ───────────────────────────────────────────

def init_mamba_states(num_mamba: int, batch_size: int,
                      H: int, N: int, P: int,
                      dtype=mx.float32) -> List[dict]:
    """Return a list of zero-initialised Mamba layer states."""
    return [make_mamba_state(batch_size, H, N, P, dtype=dtype)
            for _ in range(num_mamba)]


def init_kv_caches(num_trans: int) -> List[dict]:
    """Return a list of empty KV-cache dicts."""
    return [make_kv_cache() for _ in range(num_trans)]


# ── Clone helpers (deep-copy, materialise) ────────────────────────────────────

def clone_mamba_state(state: dict) -> dict:
    """
    Deep-copy a single Mamba layer state.
    All arrays are materialised so the clone is independent of the source graph.
    Use when saving a speculative rollback checkpoint.
    """
    cloned = {k: mx.array(v) for k, v in state.items()}
    mx.eval(*cloned.values())
    return cloned


def clone_mamba_states(states: List[dict]) -> List[dict]:
    """Deep-copy a list of Mamba layer states."""
    return [clone_mamba_state(s) for s in states]


def clone_kv_cache(cache: dict) -> dict:
    """Deep-copy a single KV-cache dict."""
    if cache.get("k") is None:
        return make_kv_cache()
    cloned = {"k": mx.array(cache["k"]), "v": mx.array(cache["v"])}
    mx.eval(cloned["k"], cloned["v"])
    return cloned


def clone_kv_caches(caches: List[dict]) -> List[dict]:
    """Deep-copy a list of KV-cache dicts."""
    return [clone_kv_cache(c) for c in caches]
