"""Token banning logic for guided generation.

Maps "trigger" token IDs to sets of "banned next" tokens that would produce
known hallucination patterns (Conyner, Conyley, Conyberg, Analley, MStep, M##, etc.).

Also provides ALWAYS_BAN for tokens that are individually problematic and unlikely
to appear in legitimate self-awareness output.
"""
from __future__ import annotations

# ── Conditional bans ───────────────────────────────────────────────────────────
# key = trigger token_id, value = set of banned next token_ids
# These prevent known hallucination patterns from forming.
CONDITIONAL_BANS: dict[int, set[int]] = {
    # ▁Con (1281) → don't let it become "Conyner/Conyley/Conyberg"
    1281: {948, 29891, 1508, 12072, 2552},  # yn, y, yle, yk, berg
    # ▁con (378) → same for lowercase
    378:  {948, 29891, 1508, 12072, 2552},
    # ▁Anal (11597) → prevent "Analley"
    11597: {2330},  # ley
    # ▁anal (3483) → prevent "analley"
    3483: {2330},
    # ▁Sug (25589) → prevent "Suggencies/Sugserve"
    25589: {1885, 16349},  # gen, serve
    # ▁M (341) → prevent "MStep", "M##" but allow "Mamba" (amba=20027)
    341: {14448, 2277},  # Step, ##
}

# ── Always-banned tokens ───────────────────────────────────────────────────────
# These tokens are individually hallucination-prone and safe to ban globally.
# They rarely appear in legitimate English text.
ALWAYS_BAN: set[int] = {
    # Hallucination subword fragments
    12072,   # yk (from "Conyk")
    1508,    # yle (from "Conyley")
    948,     # yn (from "Conyner")
    2552,    # berg (from "Conyberg" — note: "Berg" is a name but safe to ban here)
    29882,   # h (from "Conyhital" — loose 'h' as standalone token)
    2410,    # ital (from "Conyhital")
    1885,    # gen (from "Suggencies" — risky: "general" → "gen" + "eral")
}

# ── Special tokens we want to ENCOURAGE ────────────────────────────────────────
THINK_END   = 32003   # </think>
FINAL_START = 32004   # <final>
FINAL_END   = 32005   # </final>
IM_END      = 32001   # <|im_end|>

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_banned_for_prev(prev_token_id: int) -> set[int]:
    """Return banned next-token IDs given the previous token."""
    return CONDITIONAL_BANS.get(prev_token_id, set())


def get_all_banned(prev_token_id: int, extra: set[int] | None = None) -> set[int]:
    """Combine always-banned, conditional-banned, and extras."""
    result = ALWAYS_BAN | get_banned_for_prev(prev_token_id)
    if extra:
        result |= extra
    return result
