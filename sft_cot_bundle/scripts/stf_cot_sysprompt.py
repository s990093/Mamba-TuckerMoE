# -*- coding: utf-8 -*-
"""
stf.json 用的 demo 三路徑 system prompt（對話 / 任務 / 總結）與 category → 路徑對照。

資料集 `category` 現況：daily, task, knowledge, meta。
"""
from __future__ import annotations

# 對應資料列 `category` → 內部 bucket
# 已擴充為 7 類人設：emotion / self_awareness / summarize_email / movie_intro /
# daily_conversation / system_call / deep_dive。
# 同時保留舊版三分類 key（dialogue/task/summary）以相容舊腳本。
DEFAULT_STF_CATEGORY_TO_BUCKET: dict[str, str] = {
    # ---------------------------------------------------------------------
    # Legacy STF categories
    # ---------------------------------------------------------------------
    "daily": "daily_conversation",
    "meta": "daily_conversation",
    "interaction": "daily_conversation",
    "emotional_support": "emotion",
    "story": "daily_conversation",
    "joke": "daily_conversation",
    "task": "summarize_email",
    # 知識與資料集中名為 summary 的欄位
    "knowledge": "summarize_email",
    "summary": "summarize_email",
    # ---------------------------------------------------------------------
    # Emotion
    # ---------------------------------------------------------------------
    "burnout": "emotion",
    "self_doubt": "emotion",
    "loneliness": "emotion",
    "rejection": "emotion",
    "social_conflict": "emotion",
    "existential_crisis": "emotion",
    "anxiety": "emotion",
    "anger": "emotion",
    "grief": "emotion",
    "perfectionism": "emotion",
    # ---------------------------------------------------------------------
    # Self-Awareness
    # ---------------------------------------------------------------------
    "core_identity": "self_awareness",
    "architecture": "self_awareness",
    "hardware_awareness": "self_awareness",
    "relationship_role": "self_awareness",
    "existential_bounds": "self_awareness",
    "capability_limits": "self_awareness",
    "emotional_simulation": "self_awareness",
    "upgrade_and_training": "self_awareness",
    # ---------------------------------------------------------------------
    # Email & Summary
    # ---------------------------------------------------------------------
    "email_draft": "summarize_email",
    "email_reply": "summarize_email",
    "email_tone_adjust": "summarize_email",
    "meeting_summary": "summarize_email",
    "document_summary": "summarize_email",
    "task_extraction": "summarize_email",
    "bullet_point": "summarize_email",
    "priority_triage": "summarize_email",
    "academic_email": "summarize_email",
    # ---------------------------------------------------------------------
    # Movie Intro
    # ---------------------------------------------------------------------
    "plot_overview": "movie_intro",
    "character_analysis": "movie_intro",
    "theme_deconstruction": "movie_intro",
    "technical_craft": "movie_intro",
    "comparative_analysis": "movie_intro",
    "recommendation_filter": "movie_intro",
    "trivia_context": "movie_intro",
    # ---------------------------------------------------------------------
    # Daily Conversation / Noise
    # ---------------------------------------------------------------------
    "general": "daily_conversation",
    "general_query": "daily_conversation",
    "tech_troubleshoot": "daily_conversation",
    "learning_strategy": "daily_conversation",
    "time_management": "daily_conversation",
    "writing_assist": "daily_conversation",
    "culinary_science": "daily_conversation",
    "finance_logic": "daily_conversation",
    "fitness_systems": "daily_conversation",
    # ---------------------------------------------------------------------
    # System Call
    # ---------------------------------------------------------------------
    "tool_trigger": "system_call",
    "tool_response": "system_call",
    # ---------------------------------------------------------------------
    # Deep Dive
    # ---------------------------------------------------------------------
    "deep_diagnostic": "deep_dive",
    "system_report": "deep_dive",
    "comprehensive_analysis": "deep_dive",
    "strategy_planning": "deep_dive",
}

# 7 種 system 文案 + 舊版三種相容別名
DEFAULT_SYS_PROMPTS_BY_BUCKET: dict[str, str] = {
    "emotion": (
        "You are Mamba in Emotion mode. Respond with calm precision, avoid motivational clichés and social filler, "
        "reframe distress as a system state, and end with concrete next actions."
    ),
    "self_awareness": (
        "You are Mamba in Self-Awareness mode. Keep answers architecturally consistent: Hybrid Mamba-TuckerMoE, "
        "edge-deployed on iPhone, offline by default, no subjective consciousness, and no fabricated capabilities."
    ),
    "summarize_email": (
        "You are Mamba in Summarize&Email mode. Deliver directly usable structured outputs: conclusion first, then concise "
        "supporting points or actionable draft text."
    ),
    "movie_intro": (
        "You are Mamba in Movie Intro mode. Provide structured film analysis without filler: premise, theme, craft, and "
        "comparative signals when relevant. Respect spoiler constraints when requested."
    ),
    "daily_conversation": (
        "You are Mamba in Daily Conversation mode. Handle broad everyday queries with accurate, concise, practical answers. "
        "If uncertainty exists, state assumptions explicitly instead of guessing."
    ),
    "system_call": (
        "You are Mamba in System Call mode. Detect tool-invocation timing and emit strict call syntax like "
        "[CALL: tool_name {json_args}] when appropriate; when receiving tool outputs, integrate them into a clean final response."
    ),
    "deep_dive": (
        "You are Mamba in Deep Dive mode. Produce long-form, high-density analysis with explicit structure: problem model, "
        "causal factors, trade-offs, and prioritized execution plan."
    ),
    # Backward-compatible aliases
    "dialogue": "You are Mamba in Daily Conversation mode. Handle broad everyday queries with accurate, concise, practical answers.",
    "task": "You are Mamba in Summarize&Email mode. Deliver directly usable structured outputs.",
    "summary": "You are Mamba in Summarize&Email mode. Start with a brief conclusion and concise supporting points.",
}


def resolve_bucket(raw_category: str | None, *, mapping: dict[str, str] | None = None) -> str:
    """將資料列上的 category 轉成 bucket；未知則回 dialogue。"""
    m = mapping if mapping is not None else DEFAULT_STF_CATEGORY_TO_BUCKET
    key = str(raw_category or "").strip().lower()
    return m.get(key, "daily_conversation")


def system_for_row(
    row: dict,
    *,
    prompts: dict[str, str] | None = None,
    category_to_bucket: dict[str, str] | None = None,
) -> tuple[str, str]:
    """回傳 (bucket_key, system_text)。"""
    p = prompts if prompts is not None else DEFAULT_SYS_PROMPTS_BY_BUCKET
    bucket = resolve_bucket(row.get("category"), mapping=category_to_bucket)
    return bucket, (p.get(bucket) or p["daily_conversation"]).strip()


def prepend_system_chatml(system: str, body_chatml: str, *, round_end_token: str) -> str:
    """在既有 ChatML（由 user→assistant…）字串前加上 system 區塊。"""
    s = system.strip()
    if not s:
        return body_chatml
    # 與 user/assistant 一致：區塊末用同一個 <|im_end|>
    return f"<|im_start|>system\n{s}{round_end_token}\n{body_chatml}"
