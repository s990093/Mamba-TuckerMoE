"""Per-mode sampling hyperparameters for each SFT-CoT category.

These are applied automatically when --mode is set in run.py and when
category_key is selected in the chat UI. CLI arguments and frontend
sampling overrides always take precedence.

Params map to GenerationConfig fields:
  temperature, top_k, top_p, min_p, rep_pen, pres_pen, freq_pen, seed
"""
from __future__ import annotations

MODE_GEN_CONFIGS: dict[str, dict] = {
    # Identity/self-awareness: exact-phrase tuned params.
    # v6 checkpoint sweep (160 trials, warm GPU context): seed=26/temp=0.25 →
    #   "I am Mamba, a local AI that lives entirely on your device."
    # prewarm_n=5: run.py/chat_demo both run 5 short generation sequences before
    #   the real generation to stabilise Metal GPU bf16 arithmetic (~3s overhead).
    # raw_sampling=True: disables ALL logit engineering so the chat path matches
    #   run.py's unbiased sampling exactly.
    "self_awareness": {
        "temperature":  0.25,
        "top_k":        60,
        "top_p":        0.856,
        "min_p":        0.122,
        "rep_pen":      1.243,
        "pres_pen":     0.306,
        "freq_pen":     0.031,
        "seed":         26,
        "reasoning":    True,
        "raw_sampling": True,
        # "prewarm_n":      5,
        "prewarm_prompt": "Who are you?",  # capital W required for correct Metal warm state
    },
    # Emotional support: warmer, slightly higher diversity, gentler penalty.
    "emotion": {
        "temperature": 0.52,
        "top_k":       40,
        "top_p":       0.92,
        "min_p":       0.06,
        "rep_pen":     1.18,
        "pres_pen":    0.18,
        "freq_pen":    0.04,
        "seed":        0,
        "reasoning":   True,
    },
    # Email summary: moderate temperature, reduced penalties for fluent structured output.
    # raw_sampling=True: bypass middleware (ban masks, close_bias, force_final_inject)
    # so chat_demo path matches run.py exactly.
    "summarize_email": {
        "temperature": 0.25,    # v4 email sweep best: seed=17 → email template with subject+greeting+body
        "top_k":       10,
        "top_p":       0.85,
        "min_p":       0.10,
        "rep_pen":     1.15,
        "pres_pen":    0.10,
        "freq_pen":    0.03,
        "seed":        17,
        "reasoning":   True,
        "raw_sampling": True,
    },
    # Movie intro: creative, high diversity.
    "movie_intro": {
        "temperature": 0.60,
        "top_k":       50,
        "top_p":       0.93,
        "min_p":       0.05,
        "rep_pen":     1.15,
        "pres_pen":    0.12,
        "freq_pen":    0.03,
        "seed":        0,
        "reasoning":   True,
    },
    # Daily conversation: balanced.
    "daily_conversation": {
        "temperature": 0.45,
        "top_k":       30,
        "top_p":       0.90,
        "min_p":       0.07,
        "rep_pen":     1.20,
        "pres_pen":    0.18,
        "freq_pen":    0.04,
        "seed":        0,
        "reasoning":   True,
    },
    # Math: near-greedy for precise answers. reasoning=False for clean direct answers.
    "math_drill": {
        "temperature": 0.10,
        "top_k":       5,
        "top_p":       0.75,
        "min_p":       0.15,
        "rep_pen":     1.10,
        "pres_pen":    0.05,
        "freq_pen":    0.01,
        "seed":        0,
        "reasoning":   False,
    },
    # System call: deterministic tool-call format. reasoning=False for clean JSON/format output.
    "system_call": {
        "temperature": 0.20,
        "top_k":       15,
        "top_p":       0.80,
        "min_p":       0.12,
        "rep_pen":     1.20,
        "pres_pen":    0.20,
        "freq_pen":    0.05,
        "seed":        0,
        "reasoning":   False,
    },
    # Deep dive: creative, long-form.
    "deep_dive": {
        "temperature": 0.50,
        "top_k":       45,
        "top_p":       0.92,
        "min_p":       0.05,
        "rep_pen":     1.18,
        "pres_pen":    0.15,
        "freq_pen":    0.03,
        "seed":        0,
        "reasoning":   True,
    },
}


def get_mode_gen_config(mode: str | None) -> dict:
    """Return sampling config dict for *mode*, or {} if unknown/None."""
    if not mode:
        return {}
    from .system_prompts import MODE_ALIASES
    key = MODE_ALIASES.get(mode.lower()) if mode else None
    return dict(MODE_GEN_CONFIGS.get(key or mode, {}))
