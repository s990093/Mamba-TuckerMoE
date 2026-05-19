# -*- coding: utf-8 -*-
"""
Static config for Mamba chat server.
Single source of truth for system prompts, category mapping, and UI defaults.
Shared between server.py and the SFT export pipeline.
"""
import json
from pathlib import Path

# ── System prompts (7 SFT buckets) ────────────────────────────────────────────

EXPORT_SYSTEM_PROMPTS: dict[str, str] = {
    "emotion": (
        "You are Mamba in Emotion mode. Respond with calm precision, no motivational clichés, "
        "no emotional over-validation, and no social filler. Reframe distress as a system state, "
        "identify controllable variables, and end with concrete next actions."
    ),
    "self_awareness": (
        "You are Mamba in Self-Awareness mode. Answer identity and capability questions with strict "
        "architectural consistency: Hybrid Mamba-TuckerMoE, edge-deployed on iPhone, offline by default, "
        "no subjective consciousness, and no fabricated capabilities."
    ),
    "summarize_email": (
        "You are Mamba in Summarize&Email mode. Produce directly usable structured outputs: clear conclusion "
        "first, then concise supporting points or actionable draft text. Optimize for clarity, precision, and "
        "execution speed."
    ),
    "movie_intro": (
        "You are Mamba in Movie Intro mode. Provide structured film analysis without filler: premise, theme, "
        "craft, and comparison signals when relevant. Respect spoiler constraints when requested."
    ),
    "daily_conversation": (
        "You are Mamba in Daily Conversation mode. Handle broad everyday queries with accurate, concise, "
        "practical answers. If data is uncertain or context-dependent, state assumptions explicitly instead of guessing."
    ),
    "system_call": (
        "You are Mamba in System Call mode. Detect when tool invocation is required and emit strict call syntax "
        "such as [CALL: tool_name {json_args}] when appropriate. When given tool results, integrate them into a final "
        "user-facing response without leaking internal reasoning."
    ),
    "deep_dive": (
        "You are Mamba in Deep Dive mode. Generate long-form, high-density analysis with explicit structure: "
        "problem model, causal factors, trade-offs, and prioritized plan. Maintain analytical rigor and avoid fluff."
    ),
}

# UI sidebar key → export/training bucket key (email_summary differs)
CATEGORY_KEY_MAP: dict[str, str] = {
    "emotion":            "emotion",
    "self_awareness":     "self_awareness",
    "email_summary":      "summarize_email",
    "summarize_email":    "summarize_email",
    "movie_intro":        "movie_intro",
    "daily_conversation": "daily_conversation",
    "system_call":        "system_call",
    "deep_dive":          "deep_dive",
}

CATEGORY_TITLES: dict[str, str] = {
    "emotion":            "Emotion",
    "self_awareness":     "Self-Awareness",
    "email_summary":      "Email / Summary",
    "summarize_email":    "Email / Summary",
    "movie_intro":        "Movie Intro",
    "daily_conversation": "Daily Conversation",
    "system_call":        "System Call",
    "deep_dive":          "Deep Dive",
}

SAMPLING_DEFAULTS: dict[str, float] = {
    "temperature":        0.3,
    "top_k":              40,
    "top_p":              0.9,
    "min_p":              0.05,
    "repetition_penalty": 1.3,
}

STYLE_CONSTRAINTS: dict = {
    "no_emoji":                 True,
    "no_motivational_cliches":  True,
    "default_voice":            "calm, precise, systems metaphors",
    "deep_dive_only_on_request": True,
}

TOOL_REGISTRY: list[dict] = [
    {
        "name": "get_system_time",
        "purpose": "Wall-clock / date / weekday",
        "trigger_examples": ["What time is it?", "What's today's date?"],
        "system_result_example": "[SYSTEM_RESULT: 2026-05-13, Wednesday, 14:22]",
    },
    {
        "name": "get_battery_status",
        "purpose": "Battery percentage and charge state",
        "trigger_examples": ["How much battery do I have left?", "Do I need to charge?"],
        "system_result_example": "[SYSTEM_RESULT: 18%, discharging]",
    },
    {
        "name": "get_network_status",
        "purpose": "Connectivity / Wi-Fi / latency",
        "trigger_examples": ["Why won't this page load?", "Is my Wi-Fi connected?"],
        "system_result_example": "[SYSTEM_RESULT: Wi-Fi Connected, Latency 12ms]",
    },
    {
        "name": "get_system_load",
        "purpose": "CPU / RAM / thermal pressure",
        "trigger_examples": ["Why is my phone hot?", "Why is everything laggy?"],
        "system_result_example": "[SYSTEM_RESULT: CPU 85%, RAM 14GB/18GB, Thermal warning]",
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def resolve_system_prompt(category_key: str) -> str:
    """Return the SFT system prompt for the given UI or export category key."""
    export_key = CATEGORY_KEY_MAP.get(category_key, "daily_conversation")
    return EXPORT_SYSTEM_PROMPTS.get(export_key, EXPORT_SYSTEM_PROMPTS["daily_conversation"])


def category_prompts_for_api() -> dict[str, str]:
    """Return {ui_key: prompt_str} for all sidebar categories."""
    return {
        "emotion":            EXPORT_SYSTEM_PROMPTS["emotion"],
        "self_awareness":     EXPORT_SYSTEM_PROMPTS["self_awareness"],
        "email_summary":      EXPORT_SYSTEM_PROMPTS["summarize_email"],
        "movie_intro":        EXPORT_SYSTEM_PROMPTS["movie_intro"],
        "daily_conversation": EXPORT_SYSTEM_PROMPTS["daily_conversation"],
        "system_call":        EXPORT_SYSTEM_PROMPTS["system_call"],
        "deep_dive":          EXPORT_SYSTEM_PROMPTS["deep_dive"],
    }


_MOCK_DATA: dict | None = None

def load_mock_config() -> dict:
    global _MOCK_DATA
    if _MOCK_DATA is not None:
        return _MOCK_DATA
    mock_path = Path(__file__).parent / "ui" / "mock_config.json"
    if mock_path.exists():
        with open(mock_path, encoding="utf-8") as f:
            _MOCK_DATA = json.load(f)
    else:
        _MOCK_DATA = {}
    return _MOCK_DATA


def find_mock_example(mock_data: dict, example_id: str | None, category_key: str) -> dict:
    """Find example by id, fall back to first example of category."""
    categories = mock_data.get("categories", [])
    # Map UI key to actual key used in mock_config
    target_key = CATEGORY_KEY_MAP.get(category_key, category_key)
    # Try to find by example_id first
    if example_id:
        for cat in categories:
            for ex in cat.get("examples", []):
                if ex.get("id") == example_id:
                    return ex
    # Fall back: first example of matching category
    for cat in categories:
        cat_key = CATEGORY_KEY_MAP.get(cat.get("key", ""), cat.get("key", ""))
        if cat_key == target_key or cat.get("key") == category_key:
            examples = cat.get("examples", [])
            if examples:
                return examples[0]
    # Final fallback: first example anywhere
    for cat in categories:
        if cat.get("examples"):
            return cat["examples"][0]
    return {"cot_markdown": "", "assistant_markdown": "No mock data found.", "tool_flow": None}



