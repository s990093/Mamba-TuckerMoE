#!/usr/bin/env python3
"""
Plot CoT length distribution per category (token count).

Usage:
    python3 cot_dataset/plot_cot_length.py                  # interactive window
    python3 cot_dataset/plot_cot_length.py --save plots/    # save PNGs only
    python3 cot_dataset/plot_cot_length.py --bucket-level   # group by sys_bucket instead of fine category

Requirements: matplotlib, numpy, tokenizers
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

from category_system_prompts import EXPORT_SYSTEM_PROMPTS as SYSTEM_PROMPTS

SCRIPT_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = SCRIPT_DIR / "tokenizer.json"

# ---------------------------------------------------------------------------
# Category → bucket mapping (mirrors export_hf_dataset.py)
# ---------------------------------------------------------------------------
CATEGORY_TO_BUCKET: dict[str, str] = {
    "burnout": "emotion", "self_doubt": "emotion", "loneliness": "emotion",
    "rejection": "emotion", "social_conflict": "emotion",
    "existential_crisis": "emotion", "anxiety": "emotion",
    "anger": "emotion", "grief": "emotion", "perfectionism": "emotion",
    "core_identity": "self_awareness", "architecture": "self_awareness",
    "hardware_awareness": "self_awareness", "relationship_role": "self_awareness",
    "existential_bounds": "self_awareness", "capability_limits": "self_awareness",
    "emotional_simulation": "self_awareness", "upgrade_and_training": "self_awareness",
    "email_draft": "summarize_email", "email_reply": "summarize_email",
    "email_tone_adjust": "summarize_email", "meeting_summary": "summarize_email",
    "document_summary": "summarize_email", "task_extraction": "summarize_email",
    "bullet_point": "summarize_email", "priority_triage": "summarize_email",
    "academic_email": "summarize_email",
    "plot_overview": "movie_intro", "character_analysis": "movie_intro",
    "theme_deconstruction": "movie_intro", "technical_craft": "movie_intro",
    "comparative_analysis": "movie_intro", "recommendation_filter": "movie_intro",
    "trivia_context": "movie_intro",
    "general": "daily_conversation", "general_query": "daily_conversation",
    "tech_troubleshoot": "daily_conversation", "learning_strategy": "daily_conversation",
    "time_management": "daily_conversation", "writing_assist": "daily_conversation",
    "culinary_science": "daily_conversation", "finance_logic": "daily_conversation",
    "fitness_systems": "daily_conversation", "everyday_physics": "daily_conversation",
    "everyday_chemistry": "daily_conversation", "object_materials": "daily_conversation",
    "math_basic": "daily_conversation", "math_applied": "daily_conversation",
    "assistant_productivity": "daily_conversation",
    "assistant_quick_task": "daily_conversation",
    "travel_logistics": "daily_conversation", "creative_problem": "daily_conversation",
    "general_knowledge": "daily_conversation",
    "arith_add_units": "math_drill", "arith_add_mixed": "math_drill",
    "arith_mul_table": "math_drill", "arith_mul_teens": "math_drill",
    "arith_mul_extended": "math_drill", "arith_mul_hundred": "math_drill",
    "deep_diagnostic": "deep_dive", "system_report": "deep_dive",
    "comprehensive_analysis": "deep_dive", "strategy_planning": "deep_dive",
    "tool_trigger": "system_call", "tool_response": "system_call",
}

BUCKET_ORDER = [
    "emotion", "self_awareness", "summarize_email", "movie_intro",
    "daily_conversation", "math_drill", "system_call", "deep_dive",
]

BUCKET_COLORS = {
    "emotion": "#e74c3c",
    "self_awareness": "#8e44ad",
    "summarize_email": "#2980b9",
    "movie_intro": "#f39c12",
    "daily_conversation": "#27ae60",
    "math_drill": "#e67e22",
    "system_call": "#1abc9c",
    "deep_dive": "#2c3e50",
}

SKIP_FILES = {"tokenizer.json", "tokenizer_config.json"}


def load_tokenizer() -> Tokenizer | None:
    if not TOKENIZER_PATH.exists():
        print(f"WARNING: tokenizer not found at {TOKENIZER_PATH}, falling back to char count")
        return None
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def load_all_data() -> list[dict]:
    entries: list[dict] = []
    json_files = sorted(SCRIPT_DIR.glob("*.json"))
    for subdir in sorted(SCRIPT_DIR.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith(".") and not subdir.name.startswith("__"):
            json_files.extend(sorted(subdir.glob("*.json")))

    for fp in json_files:
        if fp.name in SKIP_FILES:
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        rel = str(fp.relative_to(SCRIPT_DIR))
        for obj in data:
            cot = (obj.get("cot") or "").strip()
            cat = (obj.get("category") or "").strip().lower()
            if not cat or not cot:
                continue
            entries.append({"category": cat, "cot": cot, "source_file": rel})
    return entries


def compute_cot_lengths(entries: list[dict], tok: Tokenizer | None) -> list[dict]:
    results: list[dict] = []
    for entry in entries:
        cot = entry["cot"]
        bucket = CATEGORY_TO_BUCKET.get(entry["category"], "other")
        tok_len = len(tok.encode(cot).ids) if tok else len(cot)
        char_len = len(cot)
        results.append({
            "category": entry["category"],
            "bucket": bucket,
            "cot_tokens": tok_len,
            "cot_chars": char_len,
        })
    return results


def plot_histogram(
    lengths_by_key: dict[str, list[int]],
    key_label: str,
    title: str,
    out_path: Path | None,
    log: bool = False,
) -> None:
    keys = [k for k in BUCKET_ORDER if k in lengths_by_key] + [
        k for k in sorted(lengths_by_key) if k not in BUCKET_ORDER
    ]
    n = len(keys)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, key in enumerate(keys):
        ax = axes[i // cols][i % cols]
        arr = np.array(lengths_by_key[key])
        color = BUCKET_COLORS.get(key, "#7f8c8d")

        ax.hist(arr, bins=40, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        if log:
            ax.set_yscale("log")
        ax.axvline(np.mean(arr), color="black", linestyle="--", linewidth=1.2,
                   label=f"μ={np.mean(arr):.0f}")
        ax.axvline(np.median(arr), color="white", linestyle="-", linewidth=1.2,
                   label=f"med={np.median(arr):.0f}")
        ax.set_title(f"{key}  (n={len(arr)}, σ={np.std(arr):.0f})", fontsize=10)
        ax.set_xlabel(key_label)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    for j in range(i + 1, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_box(
    lengths_by_key: dict[str, list[int]],
    key_label: str,
    title: str,
    out_path: Path | None,
) -> None:
    keys = [k for k in BUCKET_ORDER if k in lengths_by_key] + [
        k for k in sorted(lengths_by_key) if k not in BUCKET_ORDER
    ]
    data = [lengths_by_key[k] for k in keys]
    colors = [BUCKET_COLORS.get(k, "#7f8c8d") for k in keys]

    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.1), 6))
    bp = ax.boxplot(data, labels=keys, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black", markersize=6))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(key_label)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_stacked_histogram(
    lengths_by_key: dict[str, list[int]],
    key_label: str,
    title: str,
    out_path: Path | None,
    log: bool = False,
) -> None:
    keys = [k for k in BUCKET_ORDER if k in lengths_by_key]
    if not keys:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=14, fontweight="bold")

    bins = np.linspace(
        min(min(v) for v in lengths_by_key.values()),
        np.percentile(np.concatenate(list(lengths_by_key.values())), 99),
        50,
    )

    for key in keys:
        axes[0].hist(lengths_by_key[key], bins=bins, alpha=0.6, label=key,
                     color=BUCKET_COLORS.get(key, "#7f8c8d"), edgecolor="white", linewidth=0.3)
    if log:
        axes[0].set_yscale("log")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(axis="y", alpha=0.3)

    for i, key in enumerate(keys):
        arr = np.array(lengths_by_key[key])
        axes[1].barh(i, np.mean(arr), xerr=np.std(arr), color=BUCKET_COLORS.get(key, "#7f8c8d"),
                     alpha=0.8, capsize=3)
        axes[1].text(np.mean(arr) + np.std(arr) + 2, i, f"n={len(arr)}  μ={np.mean(arr):.0f}",
                     va="center", fontsize=8)
    axes[1].set_yticks(range(len(keys)))
    axes[1].set_yticklabels(keys, fontsize=9)
    axes[1].set_xlabel(key_label)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def print_stats(results: list[dict], tok: Tokenizer | None) -> None:
    buckets: dict[str, list[int]] = defaultdict(list)
    for r in results:
        buckets[r["bucket"]].append(r["cot_tokens"])

    print(f"\n{'='*65}")
    print(f"{'CoT Token Length Summary by Bucket':^65}")
    print(f"{'='*65}")
    print(f"{'Bucket':<22} {'n':>6} {'min':>6} {'μ':>7} {'σ':>7} {'med':>6} {'max':>6}")
    print(f"{'-'*65}")

    for key in BUCKET_ORDER:
        arr = buckets.get(key, [])
        if not arr:
            continue
        a = np.array(arr)
        print(f"{key:<22} {len(a):>6} {a.min():>6} {a.mean():>7.0f} "
              f"{a.std():>7.0f} {np.median(a):>6.0f} {a.max():>6}")

    all_arr = np.concatenate(list(buckets.values())) if buckets else np.array([])
    if len(all_arr):
        print(f"{'-'*65}")
        print(f"{'TOTAL':<22} {len(all_arr):>6} {all_arr.min():>6} {all_arr.mean():>7.0f} "
              f"{all_arr.std():>7.0f} {np.median(all_arr):>6.0f} {all_arr.max():>6}")
    print()


def main() -> None:
    matplotlib.use("TkAgg" if not _should_save() else "Agg")

    parser = argparse.ArgumentParser(description="Plot CoT length distribution per category.")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save PNGs to directory instead of showing interactively.")
    parser.add_argument("--bucket-level", action="store_true",
                        help="Group by sys_bucket (8 categories) instead of fine-grained category.")
    parser.add_argument("--no-tokenizer", action="store_true",
                        help="Use character count instead of token count.")
    args = parser.parse_args()

    tok = None if args.no_tokenizer else load_tokenizer()

    print("Loading data...")
    entries = load_all_data()
    print(f"  Loaded {len(entries)} entries with CoT content")

    results = compute_cot_lengths(entries, tok)
    key_label = "CoT Characters" if tok is None else "CoT Tokens"

    if args.bucket_level:
        key_field = "bucket"
        prefix = "bucket"
    else:
        key_field = "category"
        prefix = "category"

    lengths_by_key: dict[str, list[int]] = defaultdict(list)
    for r in results:
        lengths_by_key[r[key_field]].append(r["cot_tokens"])

    # Sort by key name, keep BUCKET_ORDER for buckets
    if key_field == "bucket":
        sorted_keys = [k for k in BUCKET_ORDER if k in lengths_by_key]
    else:
        sorted_keys = sorted(lengths_by_key, key=lambda k: CATEGORY_TO_BUCKET.get(k, "zzz") + k)

    lengths_ordered = {k: lengths_by_key[k] for k in sorted_keys}

    print_stats(results, tok)

    out_dir = args.save
    if out_dir:
        out_dir = Path(out_dir)

    plot_histogram(
        lengths_ordered, key_label,
        f"CoT Length Distribution by {prefix.capitalize()}",
        out_dir / f"cot_length_{prefix}_hist.png" if out_dir else None,
    )
    plot_box(
        lengths_ordered, key_label,
        f"CoT Length Distribution by {prefix.capitalize()}",
        out_dir / f"cot_length_{prefix}_box.png" if out_dir else None,
    )

    if args.bucket_level:
        plot_stacked_histogram(
            lengths_ordered, key_label,
            f"CoT Length — All Buckets Overlaid",
            out_dir / f"cot_length_{prefix}_stacked.png" if out_dir else None,
        )

    print("Done.")


def _should_save() -> bool:
    import sys as _sys
    return "--save" in _sys.argv


if __name__ == "__main__":
    main()
