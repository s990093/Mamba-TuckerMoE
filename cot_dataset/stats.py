#!/usr/bin/env python3
"""
Mamba CoT Dataset — Comprehensive Statistics & Quality Report

Usage:
    python3 cot_dataset/stats.py [--no-quality] [--json-out stats.json]

Requires: tokenizers, rich, numpy, pandas  (all pre-installed or pip-installable)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = SCRIPT_DIR / "tokenizer.json"

SYSTEM_PROMPTS = {
    "dialogue": "You are a helpful conversational assistant. Reply naturally, acknowledge the user's intent, and give practical advice.",
    "task": "You are a task-oriented assistant. Follow instructions and output concise, directly usable results.",
    "summary": "You are a summarization assistant. Start with a brief conclusion, then add key reasons clearly and concisely.",
}

CATEGORY_TO_BUCKET: dict[str, str] = {
    # Emotion
    "burnout": "dialogue", "self_doubt": "dialogue", "loneliness": "dialogue",
    "rejection": "dialogue", "social_conflict": "dialogue",
    "existential_crisis": "dialogue", "anxiety": "dialogue",
    "anger": "dialogue", "grief": "dialogue", "perfectionism": "dialogue",
    # Self-Awareness
    "core_identity": "dialogue", "architecture": "dialogue",
    "hardware_awareness": "dialogue", "relationship_role": "dialogue",
    "existential_bounds": "dialogue", "capability_limits": "dialogue",
    "emotional_simulation": "dialogue", "upgrade_and_training": "dialogue",
    # Email & Summary
    "email_draft": "task", "email_reply": "task", "email_tone_adjust": "task",
    "meeting_summary": "summary", "document_summary": "summary",
    "task_extraction": "summary", "bullet_point": "summary",
    "priority_triage": "summary", "academic_email": "task",
    # Movie Intro
    "plot_overview": "summary", "character_analysis": "dialogue",
    "theme_deconstruction": "dialogue", "technical_craft": "dialogue",
    "comparative_analysis": "summary", "recommendation_filter": "task",
    "trivia_context": "summary",
    # Daily Conversation (noise)
    "tech_troubleshoot": "task", "learning_strategy": "dialogue",
    "time_management": "task", "writing_assist": "task",
    "culinary_science": "dialogue", "fitness_systems": "dialogue",
    "finance_logic": "task", "travel_logistics": "task",
    "general_knowledge": "dialogue", "creative_problem": "dialogue",
    # System Call
    "tool_trigger": "task", "tool_response": "task",
    # Deep Dive
    "deep_diagnostic": "dialogue", "system_report": "dialogue",
    "comprehensive_analysis": "summary", "strategy_planning": "task",
}

TARGETS = {
    "emotion": 5000,
    "self_awareness": 5000,
    "email_summary": 5000,
    "movie_intro": 2000,
    "noise": 2000,
    "system_call": 600,
    "deep_dive": 700,
}

FILE_CATEGORY_GROUPS = {
    "emotion": [
        "burnout", "self_doubt", "loneliness", "rejection", "social_conflict",
        "existential_crisis", "anxiety", "anger", "grief", "perfectionism",
    ],
    "self_awareness": [
        "core_identity", "architecture", "hardware_awareness", "relationship_role",
        "existential_bounds", "capability_limits", "emotional_simulation",
        "upgrade_and_training",
    ],
    "email_summary": [
        "email_draft", "email_reply", "email_tone_adjust", "meeting_summary",
        "document_summary", "task_extraction", "bullet_point", "priority_triage",
        "academic_email",
    ],
    "movie_intro": [
        "plot_overview", "character_analysis", "theme_deconstruction",
        "technical_craft", "comparative_analysis", "recommendation_filter",
        "trivia_context",
    ],
    "noise": [
        "tech_troubleshoot", "learning_strategy", "time_management",
        "writing_assist", "culinary_science", "fitness_systems",
        "finance_logic", "travel_logistics", "general_knowledge",
        "creative_problem",
    ],
    "system_call": [
        "tool_trigger", "tool_response",
    ],
    "deep_dive": [
        "deep_diagnostic", "system_report", "comprehensive_analysis",
        "strategy_planning",
    ],
}

SUBCATEGORY_TARGETS = {
    "burnout": 600, "self_doubt": 600, "loneliness": 500, "rejection": 500,
    "social_conflict": 500, "existential_crisis": 500, "anxiety": 500,
    "anger": 400, "grief": 400, "perfectionism": 500,
    "core_identity": 700, "architecture": 700, "hardware_awareness": 600,
    "relationship_role": 600, "existential_bounds": 700,
    "capability_limits": 600, "emotional_simulation": 500,
    "upgrade_and_training": 600,
    "email_draft": 800, "email_reply": 800, "email_tone_adjust": 500,
    "meeting_summary": 600, "document_summary": 600, "task_extraction": 500,
    "bullet_point": 500, "priority_triage": 400, "academic_email": 300,
    "plot_overview": 400, "character_analysis": 300,
    "theme_deconstruction": 300, "technical_craft": 250,
    "comparative_analysis": 300, "recommendation_filter": 250,
    "trivia_context": 200,
    "tech_troubleshoot": 300, "learning_strategy": 250,
    "time_management": 250, "writing_assist": 250,
    "culinary_science": 150, "fitness_systems": 150,
    "finance_logic": 150, "travel_logistics": 150,
    "general_knowledge": 200, "creative_problem": 150,
    "tool_trigger": 300, "tool_response": 300,
    "deep_diagnostic": 200, "system_report": 150,
    "comprehensive_analysis": 200, "strategy_planning": 150,
}

TOKEN_BUDGETS = {
    "emotion": 512,
    "self_awareness": 512,
    "email_summary": 768,
    "movie_intro": 768,
    "noise": 512,
    "system_call": 512,
    "deep_dive": 2048,
}

FORBIDDEN_PHRASES = [
    "I'm sorry to hear that", "That must be really hard",
    "You got this", "Everything will be okay", "Don't worry",
    "I understand how you feel", "Hang in there",
    "Take it one day at a time", "Believe in yourself",
    "I'm here for you", "That's totally normal",
]

CONTRACTION_PATTERN = re.compile(
    r"\b(don't|won't|can't|I'm|you're|they're|we're|it's|isn't|aren't|"
    r"wasn't|weren't|haven't|hasn't|hadn't|couldn't|wouldn't|shouldn't|"
    r"didn't|doesn't|he's|she's|that's|there's|here's|who's|what's|let's)\b",
    re.IGNORECASE,
)

console = Console()


def fmt_tok(n: int) -> str:
    """Human-friendly token count: M for millions, K for thousands."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------
def load_tokenizer() -> Tokenizer | None:
    if not TOKENIZER_PATH.exists():
        console.print(f"[yellow]⚠ tokenizer.json not found at {TOKENIZER_PATH}, "
                       "falling back to word-based estimation (×1.3)[/yellow]")
        return None
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def count_tokens(text: str, tok: Tokenizer | None) -> int:
    if tok is None:
        return max(1, int(len(text.split()) * 1.3))
    return len(tok.encode(text).ids)


def build_chatml(entry: dict, tok: Tokenizer | None) -> int:
    """Return the full ChatML token count for one entry."""
    bucket = CATEGORY_TO_BUCKET.get(entry.get("category", ""), "dialogue")
    sys_prompt = SYSTEM_PROMPTS[bucket]
    chatml = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{entry.get('input', '')}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{entry.get('cot', '')}\n</think>"
        f"<final>\n{entry.get('output', '')}\n</final><|im_end|>"
    )
    return count_tokens(chatml, tok)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def infer_file_group(filepath: str, entry: dict) -> str:
    """Decide which target group a record belongs to based on filename/directory and category."""
    p = Path(filepath)
    basename = p.stem.lower()
    parent = p.parent.name.lower()
    combined = f"{parent}/{basename}"
    if "emotion" in basename or parent == "emotion":
        return "emotion"
    if "self" in basename or parent == "self":
        return "self_awareness"
    if "email" in basename or "mail" in basename or parent == "email_summary":
        return "email_summary"
    if "movie" in basename or parent == "movie_intro":
        return "movie_intro"
    if "deep" in basename or parent == "deep_dive":
        return "deep_dive"
    if "system_call" in basename or parent == "system_call":
        return "system_call"
    if "noise" in basename or parent == "noise":
        return "noise"
    cat = entry.get("category", "")
    for group, cats in FILE_CATEGORY_GROUPS.items():
        if cat in cats:
            return group
    return "other"


def load_all_data() -> tuple[list[dict], dict[str, list[dict]]]:
    """Load all JSON data files. Returns (all_entries, entries_by_group).

    Scans root-level *.json and all immediate subdirectories.
    When the same ``id`` appears in both a root file and a subdirectory file,
    the subdirectory version wins (assumed to be newer / revised).
    """
    all_entries: list[dict] = []
    by_group: dict[str, list[dict]] = defaultdict(list)

    json_files = sorted(SCRIPT_DIR.glob("*.json"))
    for subdir in sorted(SCRIPT_DIR.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith("."):
            json_files.extend(sorted(subdir.glob("*.json")))

    skip = {"tokenizer.json", "tokenizer_config.json"}
    seen_ids: dict[str, str] = {}
    raw_entries: list[dict] = []
    override_summary: dict[str, int] = defaultdict(int)
    dup_warnings: list[str] = []

    for fp in json_files:
        if fp.name in skip:
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            console.print(f"[red]✗ JSON parse error in {fp.name}: {exc}[/red]")
            continue

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            console.print(f"[yellow]⚠ Skipping {fp.name}: root is not a list or dict[/yellow]")
            continue

        rel = str(fp.relative_to(SCRIPT_DIR))
        is_subdir = "/" in rel
        for entry in data:
            entry["_source_file"] = rel
            entry["_group"] = infer_file_group(str(fp), entry)
            eid = entry.get("id", "")
            if eid and eid in seen_ids:
                prev_src = seen_ids[eid]
                prev_is_subdir = "/" in prev_src
                if is_subdir and not prev_is_subdir:
                    raw_entries = [e for e in raw_entries if e.get("id") != eid]
                    key = f"{prev_src} → {rel}"
                    override_summary[key] += 1
                elif not is_subdir and prev_is_subdir:
                    continue
                else:
                    dup_warnings.append(f"[{eid}] in {prev_src} and {rel}")
            seen_ids[eid] = rel
            raw_entries.append(entry)

    for desc, cnt in override_summary.items():
        console.print(f"[dim]↻ {cnt} IDs overridden: {desc}[/dim]")
    for warn in dup_warnings[:10]:
        console.print(f"[yellow]⚠ Duplicate ID {warn}[/yellow]")
    if len(dup_warnings) > 10:
        console.print(f"[yellow]  ... and {len(dup_warnings) - 10} more duplicates[/yellow]")

    for entry in raw_entries:
        all_entries.append(entry)
        by_group[entry["_group"]].append(entry)

    return all_entries, dict(by_group)


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------
def percentile_stats(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {"count": 0, "mean": 0, "std": 0, "min": 0,
                "p25": 0, "median": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "p25": int(np.percentile(arr, 25)),
        "median": int(np.median(arr)),
        "p75": int(np.percentile(arr, 75)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
    }


def word_count(text: str) -> int:
    return len(text.split())


def cot_step_count(cot: str) -> int:
    return len(re.findall(r"Step\s+\d+", cot))


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------
def quality_check(entry: dict) -> list[str]:
    issues: list[str] = []
    eid = entry.get("id", "???")

    output_text = entry.get("output", "")
    cot_text = entry.get("cot", "")
    input_text = entry.get("input", "")

    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in output_text.lower():
            issues.append(f"[{eid}] Forbidden phrase in output: '{phrase}'")

    contractions = CONTRACTION_PATTERN.findall(output_text)
    if contractions:
        issues.append(f"[{eid}] Contractions in output: {contractions[:5]}")

    cot_contractions = CONTRACTION_PATTERN.findall(cot_text)
    if cot_contractions:
        issues.append(f"[{eid}] Contractions in CoT: {cot_contractions[:5]}")

    steps = cot_step_count(cot_text)
    if steps < 3:
        issues.append(f"[{eid}] CoT has only {steps} steps (min 3)")

    group = entry.get("_group", "")
    if group not in ("deep_dive",) and steps > 5:
        issues.append(f"[{eid}] CoT has {steps} steps (max 5 for non-deep-dive)")

    for token in ["<think>", "</think>", "<final>", "</final>", "<|im_start|>", "<|im_end|>"]:
        if token in cot_text or token in output_text or token in input_text:
            issues.append(f"[{eid}] Special token '{token}' found in JSON content (should be auto-wrapped)")

    cat = entry.get("category", "")
    if cat and cat not in CATEGORY_TO_BUCKET:
        legacy_noise = {"entertainment_analysis", "meteorology_action",
                        "social_conflict_general", "philosophy_chaos", "parenting_logic",
                        "music_analysis"}
        if cat not in legacy_noise:
            issues.append(f"[{eid}] Unknown category '{cat}' (not in GUIDE.md spec)")

    return issues


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_progress_table(by_group: dict[str, list[dict]]) -> None:
    table = Table(title="📊 Dataset Progress vs. Targets", show_lines=True)
    table.add_column("Group", style="bold cyan")
    table.add_column("Target", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Progress", justify="right")
    table.add_column("Remaining", justify="right")
    table.add_column("Status")

    total_target = 0
    total_current = 0

    for group in ["emotion", "self_awareness", "email_summary", "movie_intro", "noise", "system_call", "deep_dive", "other"]:
        entries = by_group.get(group, [])
        target = TARGETS.get(group, 0)
        current = len(entries)
        total_target += target
        total_current += current

        if target == 0:
            pct = "N/A"
            remaining = "N/A"
            status = "[dim]reference[/dim]"
        else:
            p = current / target * 100
            pct = f"{p:.1f}%"
            remaining = str(max(0, target - current))
            if p >= 100:
                status = "[green]✓ COMPLETE[/green]"
            elif p >= 50:
                status = "[yellow]▶ IN PROGRESS[/yellow]"
            elif p > 0:
                status = "[red]▶ STARTED[/red]"
            else:
                status = "[red]✗ NOT STARTED[/red]"

        table.add_row(group, str(target) if target else "—", str(current), pct, remaining, status)

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_target}[/bold]",
        f"[bold]{total_current}[/bold]",
        f"[bold]{total_current / total_target * 100:.1f}%[/bold]" if total_target else "—",
        f"[bold]{max(0, total_target - total_current)}[/bold]",
        "",
    )
    console.print(table)


def render_subcategory_table(all_entries: list[dict]) -> None:
    cat_counts = Counter(e.get("category", "unknown") for e in all_entries)

    for group_name, cats in FILE_CATEGORY_GROUPS.items():
        table = Table(title=f"📂 {group_name} — Subcategory Distribution", show_lines=True)
        table.add_column("Subcategory", style="cyan")
        table.add_column("Target", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Progress", justify="right")
        table.add_column("Gap", justify="right")

        group_total_target = 0
        group_total_current = 0
        for cat in cats:
            target = SUBCATEGORY_TARGETS.get(cat, 0)
            current = cat_counts.get(cat, 0)
            group_total_target += target
            group_total_current += current
            gap = max(0, target - current)
            pct = f"{current / target * 100:.1f}%" if target else "—"
            table.add_row(cat, str(target), str(current), pct, str(gap) if gap else "[green]0[/green]")

        table.add_row(
            "[bold]Subtotal[/bold]",
            f"[bold]{group_total_target}[/bold]",
            f"[bold]{group_total_current}[/bold]",
            f"[bold]{group_total_current / group_total_target * 100:.1f}%[/bold]" if group_total_target else "—",
            f"[bold]{max(0, group_total_target - group_total_current)}[/bold]",
        )
        console.print(table)
        console.print()


def render_token_table(records: list[dict], title: str) -> None:
    if not records:
        console.print(f"[dim]  (no records for {title})[/dim]")
        return

    input_toks = np.array([r["input_tokens"] for r in records])
    cot_toks = np.array([r["cot_tokens"] for r in records])
    output_toks = np.array([r["output_tokens"] for r in records])
    total_toks = np.array([r["chatml_tokens"] for r in records])
    output_words = np.array([r["output_words"] for r in records])
    cot_steps = np.array([r["cot_steps"] for r in records])

    fields = [
        ("Input Tokens", input_toks),
        ("CoT Tokens", cot_toks),
        ("Output Tokens", output_toks),
        ("ChatML Total Tokens", total_toks),
        ("Output Words", output_words),
        ("CoT Steps", cot_steps),
    ]

    table = Table(title=f"🔢 {title} — Token & Length Statistics (n={len(records)})", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("P25", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("P75", justify="right")
    table.add_column("P90", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("Max", justify="right")

    for name, arr in fields:
        s = percentile_stats(arr)
        table.add_row(
            name,
            f"{s['mean']:.1f}", f"{s['std']:.1f}",
            str(s["min"]), str(s["p25"]), str(s["median"]),
            str(s["p75"]), str(s["p90"]), str(s["p95"]), str(s["max"]),
        )
    console.print(table)


def render_aggregate_totals(records: list[dict]) -> None:
    """Per-group and grand total: record count, total tokens for each field."""
    groups_order = ["emotion", "self_awareness", "email_summary", "movie_intro", "noise", "system_call", "deep_dive", "other"]

    table = Table(title="📋 Aggregate Totals — Records & Tokens", show_lines=True)
    table.add_column("Group", style="bold cyan")
    table.add_column("Records", justify="right")
    table.add_column("Input Tok\n(total)", justify="right")
    table.add_column("CoT Tok\n(total)", justify="right")
    table.add_column("Output Tok\n(total)", justify="right")
    table.add_column("ChatML Tok\n(total)", justify="right")
    table.add_column("Avg Input", justify="right")
    table.add_column("Avg CoT", justify="right")
    table.add_column("Avg Output", justify="right")
    table.add_column("Avg ChatML", justify="right")

    grand = {"n": 0, "input": 0, "cot": 0, "output": 0, "chatml": 0}

    for group in groups_order:
        grecs = [r for r in records if r["_group"] == group]
        if not grecs:
            continue
        n = len(grecs)
        s_in = sum(r["input_tokens"] for r in grecs)
        s_cot = sum(r["cot_tokens"] for r in grecs)
        s_out = sum(r["output_tokens"] for r in grecs)
        s_ch = sum(r["chatml_tokens"] for r in grecs)
        grand["n"] += n
        grand["input"] += s_in
        grand["cot"] += s_cot
        grand["output"] += s_out
        grand["chatml"] += s_ch
        table.add_row(
            group, f"{n:,}",
            fmt_tok(s_in), fmt_tok(s_cot), fmt_tok(s_out), fmt_tok(s_ch),
            f"{s_in / n:.1f}", f"{s_cot / n:.1f}", f"{s_out / n:.1f}", f"{s_ch / n:.1f}",
        )

    n = grand["n"] or 1
    table.add_row(
        "[bold]TOTAL[/bold]", f"[bold]{grand['n']:,}[/bold]",
        f"[bold]{fmt_tok(grand['input'])}[/bold]",
        f"[bold]{fmt_tok(grand['cot'])}[/bold]",
        f"[bold]{fmt_tok(grand['output'])}[/bold]",
        f"[bold]{fmt_tok(grand['chatml'])}[/bold]",
        f"[bold]{grand['input'] / n:.1f}[/bold]",
        f"[bold]{grand['cot'] / n:.1f}[/bold]",
        f"[bold]{grand['output'] / n:.1f}[/bold]",
        f"[bold]{grand['chatml'] / n:.1f}[/bold]",
    )
    console.print(table)


def render_budget_violations(records: list[dict], by_group: dict[str, list[dict]]) -> None:
    table = Table(title="⚠️  Token Budget Violations", show_lines=True)
    table.add_column("Group", style="cyan")
    table.add_column("Budget", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Over Budget", justify="right", style="red")
    table.add_column("% Over", justify="right")
    table.add_column("Max Tokens", justify="right")
    table.add_column("Worst Offender ID")

    for group in ["emotion", "self_awareness", "email_summary", "movie_intro", "noise", "system_call", "deep_dive"]:
        budget = TOKEN_BUDGETS.get(group)
        if budget is None:
            continue
        group_recs = [r for r in records if r["_group"] == group]
        if not group_recs:
            continue

        over = [r for r in group_recs if r["chatml_tokens"] > budget]
        worst = max(group_recs, key=lambda r: r["chatml_tokens"]) if group_recs else None
        table.add_row(
            group,
            str(budget),
            str(len(group_recs)),
            str(len(over)),
            f"{len(over) / len(group_recs) * 100:.1f}%" if group_recs else "—",
            str(worst["chatml_tokens"]) if worst else "—",
            worst["id"] if worst else "—",
        )
    console.print(table)


def render_truncation_risk(records: list[dict]) -> None:
    seq_len = 1024
    model_max = 2048
    at_risk_1024 = [r for r in records if r["chatml_tokens"] > seq_len + 1]
    at_risk_2048 = [r for r in records if r["chatml_tokens"] > model_max]

    table = Table(title="✂️  Truncation Risk Analysis", show_lines=True)
    table.add_column("Threshold")
    table.add_column("Records Exceeding", justify="right", style="red")
    table.add_column("% of Total", justify="right")
    table.add_column("Risk Level")

    total = len(records) if records else 1
    table.add_row(
        f"SEQ_LEN ({seq_len}+1)",
        str(len(at_risk_1024)),
        f"{len(at_risk_1024) / total * 100:.2f}%",
        "[yellow]Will be truncated in default training[/yellow]" if at_risk_1024 else "[green]SAFE[/green]",
    )
    table.add_row(
        f"model_max_length ({model_max})",
        str(len(at_risk_2048)),
        f"{len(at_risk_2048) / total * 100:.2f}%",
        "[red]Exceeds model capacity[/red]" if at_risk_2048 else "[green]SAFE[/green]",
    )
    console.print(table)

    if at_risk_2048:
        console.print("\n[red bold]Records exceeding model_max_length (2048):[/red bold]")
        for r in sorted(at_risk_2048, key=lambda x: -x["chatml_tokens"])[:20]:
            console.print(f"  • {r['id']} ({r['_group']}/{r.get('category','?')}) — {r['chatml_tokens']} tokens")


def render_quality_issues(issues: list[str]) -> None:
    if not issues:
        console.print(Panel("[green]✓ No quality issues detected![/green]", title="Quality Check"))
        return

    counter: dict[str, int] = Counter()
    for issue in issues:
        if "Forbidden phrase" in issue:
            counter["Forbidden phrases"] += 1
        elif "Contractions in output" in issue:
            counter["Contractions (output)"] += 1
        elif "Contractions in CoT" in issue:
            counter["Contractions (CoT)"] += 1
        elif "CoT has only" in issue:
            counter["CoT too short (<3 steps)"] += 1
        elif "CoT has" in issue and "steps (max" in issue:
            counter["CoT too long (>5 steps, non-DD)"] += 1
        elif "Special token" in issue:
            counter["Special tokens in content"] += 1
        elif "Unknown category" in issue:
            counter["Unknown categories"] += 1
        else:
            counter["Other"] += 1

    table = Table(title="🔍 Quality Issues Summary", show_lines=True)
    table.add_column("Issue Type", style="bold")
    table.add_column("Count", justify="right", style="red")
    for k, v in sorted(counter.items(), key=lambda x: -x[1]):
        table.add_row(k, str(v))
    table.add_row("[bold]Total Issues[/bold]", f"[bold red]{len(issues)}[/bold red]")
    console.print(table)

    console.print("\n[bold]First 30 issues (detail):[/bold]")
    for issue in issues[:30]:
        console.print(f"  [yellow]•[/yellow] {issue}")
    if len(issues) > 30:
        console.print(f"  ... and {len(issues) - 30} more")


def render_id_check(all_entries: list[dict]) -> None:
    ids = [e.get("id", "") for e in all_entries]
    id_counts = Counter(ids)
    dupes = {k: v for k, v in id_counts.items() if v > 1}

    if dupes:
        console.print(Panel(f"[red]✗ Found {len(dupes)} duplicate IDs![/red]", title="ID Uniqueness"))
        for eid, count in sorted(dupes.items(), key=lambda x: -x[1])[:20]:
            console.print(f"  [red]•[/red] '{eid}' appears {count} times")
    else:
        console.print(Panel(f"[green]✓ All {len(ids)} IDs are unique[/green]", title="ID Uniqueness"))


def render_source_files(all_entries: list[dict]) -> None:
    file_counts = Counter(e.get("_source_file", "?") for e in all_entries)
    table = Table(title="📁 Source File Breakdown", show_lines=True)
    table.add_column("File", style="cyan")
    table.add_column("Records", justify="right")
    for fp, count in sorted(file_counts.items()):
        table.add_row(fp, str(count))
    table.add_row("[bold]Total[/bold]", f"[bold]{sum(file_counts.values())}[/bold]")
    console.print(table)


def render_output_word_histogram(records: list[dict], by_group: dict[str, list[dict]]) -> None:
    """Simple ASCII histogram of output word counts per group."""
    bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250),
            (250, 300), (300, 400), (400, 600), (600, 800), (800, 9999)]
    bin_labels = ["0-49", "50-99", "100-149", "150-199", "200-249",
                  "250-299", "300-399", "400-599", "600-799", "800+"]

    for group in ["emotion", "self_awareness", "email_summary", "movie_intro", "noise", "system_call", "deep_dive"]:
        grecs = [r for r in records if r["_group"] == group]
        if not grecs:
            continue
        words = [r["output_words"] for r in grecs]
        table = Table(title=f"📊 {group} — Output Word Count Distribution (n={len(grecs)})", show_lines=True)
        table.add_column("Word Range", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Pct", justify="right")
        table.add_column("Histogram")

        max_count = 1
        counts = []
        for lo, hi in bins:
            c = sum(1 for w in words if lo <= w < hi)
            counts.append(c)
            if c > max_count:
                max_count = c

        for label, c in zip(bin_labels, counts):
            bar_len = int(c / max_count * 40) if max_count else 0
            bar = "█" * bar_len
            pct = f"{c / len(grecs) * 100:.1f}%" if grecs else "—"
            table.add_row(label, str(c), pct, bar)
        console.print(table)
        console.print()


def render_chatml_token_histogram(records: list[dict]) -> None:
    """Histogram of full ChatML token counts across all records."""
    bins = [(0, 128), (128, 256), (256, 384), (384, 512), (512, 640),
            (640, 768), (768, 1024), (1024, 1536), (1536, 2048), (2048, 99999)]
    bin_labels = ["0-127", "128-255", "256-383", "384-511", "512-639",
                  "640-767", "768-1023", "1024-1535", "1536-2047", "2048+"]

    table = Table(title="📊 ChatML Total Token Distribution (all records)", show_lines=True)
    table.add_column("Token Range", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Pct", justify="right")
    table.add_column("Histogram")

    total = len(records) if records else 1
    counts = []
    max_count = 1
    for lo, hi in bins:
        c = sum(1 for r in records if lo <= r["chatml_tokens"] < hi)
        counts.append(c)
        if c > max_count:
            max_count = c

    for label, c in zip(bin_labels, counts):
        bar_len = int(c / max_count * 40) if max_count else 0
        bar = "█" * bar_len
        pct = f"{c / total * 100:.1f}%"
        table.add_row(label, str(c), pct, bar)
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Mamba CoT Dataset Statistics")
    parser.add_argument("--no-quality", action="store_true", help="Skip quality checks")
    parser.add_argument("--json-out", type=str, default=None, help="Export summary to JSON")
    args = parser.parse_args()

    console.rule("[bold blue]Mamba CoT Dataset — Statistics Report[/bold blue]")
    console.print()

    tok = load_tokenizer()
    if tok:
        console.print(f"[green]✓ Loaded tokenizer (vocab size: {tok.get_vocab_size()})[/green]")
    console.print()

    console.print("[bold]Loading data files...[/bold]")
    all_entries, by_group = load_all_data()
    console.print(f"  Total entries loaded: [bold]{len(all_entries)}[/bold]")
    console.print()

    # ----- Source file breakdown -----
    render_source_files(all_entries)
    console.print()

    # ----- Progress vs targets -----
    render_progress_table(by_group)
    console.print()

    # ----- Subcategory distribution -----
    render_subcategory_table(all_entries)

    # ----- ID uniqueness -----
    render_id_check(all_entries)
    console.print()

    # ----- Tokenization -----
    console.print("[bold]Tokenizing all entries (this may take a moment)...[/bold]")
    enriched: list[dict] = []
    for entry in all_entries:
        input_text = entry.get("input", "")
        cot_text = entry.get("cot", "")
        output_text = entry.get("output", "")

        rec = {
            "id": entry.get("id", "???"),
            "category": entry.get("category", "unknown"),
            "_group": entry.get("_group", "other"),
            "_source_file": entry.get("_source_file", "?"),
            "input_tokens": count_tokens(input_text, tok),
            "cot_tokens": count_tokens(cot_text, tok),
            "output_tokens": count_tokens(output_text, tok),
            "chatml_tokens": build_chatml(entry, tok),
            "output_words": word_count(output_text),
            "input_words": word_count(input_text),
            "cot_steps": cot_step_count(cot_text),
        }
        enriched.append(rec)

    console.print(f"  Tokenized [bold]{len(enriched)}[/bold] entries.\n")

    # ----- Per-group token stats -----
    for group in ["emotion", "self_awareness", "email_summary", "movie_intro", "noise", "system_call", "deep_dive", "other"]:
        group_recs = [r for r in enriched if r["_group"] == group]
        if group_recs:
            render_token_table(group_recs, group)
            console.print()

    # ----- Overall token stats -----
    render_token_table(enriched, "ALL RECORDS (combined)")
    console.print()

    # ----- Aggregate totals (records × total tokens per field) -----
    render_aggregate_totals(enriched)
    console.print()

    # ----- Budget violations -----
    render_budget_violations(enriched, by_group)
    console.print()

    # ----- Truncation risk -----
    render_truncation_risk(enriched)
    console.print()

    # ----- Word count histogram -----
    render_output_word_histogram(enriched, by_group)

    # ----- ChatML token histogram -----
    render_chatml_token_histogram(enriched)
    console.print()

    # ----- Quality checks -----
    if not args.no_quality:
        console.print("[bold]Running quality checks...[/bold]")
        all_issues: list[str] = []
        for entry in all_entries:
            all_issues.extend(quality_check(entry))
        render_quality_issues(all_issues)
        console.print()

    # ----- Grand summary -----
    total_chatml = sum(r["chatml_tokens"] for r in enriched)
    total_input = sum(r["input_tokens"] for r in enriched)
    total_cot = sum(r["cot_tokens"] for r in enriched)
    total_output = sum(r["output_tokens"] for r in enriched)
    n = len(enriched) or 1
    console.rule("[bold blue]Grand Summary[/bold blue]")

    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column(style="bold")
    summary_table.add_column(justify="right")
    summary_table.add_row("Total records", f"{len(enriched):,}")
    summary_table.add_row("", "")
    summary_table.add_row("Total Input tokens", fmt_tok(total_input))
    summary_table.add_row("Total CoT tokens", fmt_tok(total_cot))
    summary_table.add_row("Total Output tokens", fmt_tok(total_output))
    summary_table.add_row("Total ChatML tokens", fmt_tok(total_chatml))
    summary_table.add_row("", "")
    summary_table.add_row("Avg Input tokens/record", f"{total_input / n:.1f}")
    summary_table.add_row("Avg CoT tokens/record", f"{total_cot / n:.1f}")
    summary_table.add_row("Avg Output tokens/record", f"{total_output / n:.1f}")
    summary_table.add_row("Avg ChatML tokens/record", f"{total_chatml / n:.1f}")
    summary_table.add_row("", "")
    summary_table.add_row("Vocab size", f"{tok.get_vocab_size():,}" if tok else "(estimated)")
    summary_table.add_row("Model max length", "2,048 tokens")
    summary_table.add_row("Training SEQ_LEN", "1,024 tokens")
    console.print(summary_table)
    console.print()

    progress_table = Table(title="Progress", show_lines=True)
    progress_table.add_column("Group", style="cyan")
    progress_table.add_column("Current", justify="right")
    progress_table.add_column("Target", justify="right")
    progress_table.add_column("Progress", justify="right")
    progress_table.add_column("Input Tok", justify="right")
    progress_table.add_column("CoT Tok", justify="right")
    progress_table.add_column("Output Tok", justify="right")
    progress_table.add_column("ChatML Tok", justify="right")

    for group in ["emotion", "self_awareness", "email_summary", "movie_intro", "noise", "system_call", "deep_dive"]:
        t = TARGETS.get(group, 0)
        grecs = [r for r in enriched if r["_group"] == group]
        c = len(grecs)
        g_in = sum(r["input_tokens"] for r in grecs)
        g_cot = sum(r["cot_tokens"] for r in grecs)
        g_out = sum(r["output_tokens"] for r in grecs)
        g_ch = sum(r["chatml_tokens"] for r in grecs)
        pct = f"{c / t * 100:.1f}%" if t else "—"
        progress_table.add_row(
            group, f"{c:,}", f"{t:,}", pct,
            fmt_tok(g_in), fmt_tok(g_cot), fmt_tok(g_out), fmt_tok(g_ch),
        )

    console.print(progress_table)
    console.print()

    # ----- JSON export -----
    if args.json_out:
        export = {
            "total_records": len(enriched),
            "total_input_tokens": total_input,
            "total_cot_tokens": total_cot,
            "total_output_tokens": total_output,
            "total_chatml_tokens": total_chatml,
            "avg_input_tokens": round(total_input / n, 1),
            "avg_cot_tokens": round(total_cot / n, 1),
            "avg_output_tokens": round(total_output / n, 1),
            "avg_chatml_tokens": round(total_chatml / n, 1),
            "vocab_size": tok.get_vocab_size() if tok else None,
            "groups": {},
        }
        for group in ["emotion", "self_awareness", "email_summary", "movie_intro", "noise", "system_call", "deep_dive", "other"]:
            grecs = [r for r in enriched if r["_group"] == group]
            if not grecs:
                continue
            export["groups"][group] = {
                "count": len(grecs),
                "target": TARGETS.get(group, 0),
                "total_input_tokens": sum(r["input_tokens"] for r in grecs),
                "total_cot_tokens": sum(r["cot_tokens"] for r in grecs),
                "total_output_tokens": sum(r["output_tokens"] for r in grecs),
                "total_chatml_tokens": sum(r["chatml_tokens"] for r in grecs),
                "chatml_stats": percentile_stats(np.array([r["chatml_tokens"] for r in grecs])),
                "input_stats": percentile_stats(np.array([r["input_tokens"] for r in grecs])),
                "cot_stats": percentile_stats(np.array([r["cot_tokens"] for r in grecs])),
                "output_stats": percentile_stats(np.array([r["output_tokens"] for r in grecs])),
            }
        Path(args.json_out).write_text(json.dumps(export, indent=2, ensure_ascii=False))
        console.print(f"[green]Exported summary to {args.json_out}[/green]")


if __name__ == "__main__":
    main()
