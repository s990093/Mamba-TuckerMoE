#!/usr/bin/env python3
"""
Export Mamba CoT JSON files into a Hugging Face `datasets` snapshot.

Spec baseline:
- cot_dataset/GUIDE.md
- cot_dataset/SFT_FORMAT.md

Output columns:
- id
- category
- sys_bucket
- system
- source_file
- history
- input
- cot
- output
- text (ChatML string with <think>/<final>)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import Dataset

_cot_dir = Path(__file__).resolve().parent
if str(_cot_dir) not in sys.path:
    sys.path.insert(0, str(_cot_dir))
from category_system_prompts import EXPORT_SYSTEM_PROMPTS as SYSTEM_PROMPTS  # noqa: E402

CATEGORY_TO_BUCKET = {
    # Emotion
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
    # Self-Awareness
    "core_identity": "self_awareness",
    "architecture": "self_awareness",
    "hardware_awareness": "self_awareness",
    "relationship_role": "self_awareness",
    "existential_bounds": "self_awareness",
    "capability_limits": "self_awareness",
    "emotional_simulation": "self_awareness",
    "upgrade_and_training": "self_awareness",
    # Email & Summary
    "email_draft": "summarize_email",
    "email_reply": "summarize_email",
    "email_tone_adjust": "summarize_email",
    "meeting_summary": "summarize_email",
    "document_summary": "summarize_email",
    "task_extraction": "summarize_email",
    "bullet_point": "summarize_email",
    "priority_triage": "summarize_email",
    "academic_email": "summarize_email",
    # Movie Intro
    "plot_overview": "movie_intro",
    "character_analysis": "movie_intro",
    "theme_deconstruction": "movie_intro",
    "technical_craft": "movie_intro",
    "comparative_analysis": "movie_intro",
    "recommendation_filter": "movie_intro",
    "trivia_context": "movie_intro",
    # Daily Conversation / Noise
    "general": "daily_conversation",
    "general_query": "daily_conversation",
    "tech_troubleshoot": "daily_conversation",
    "learning_strategy": "daily_conversation",
    "time_management": "daily_conversation",
    "writing_assist": "daily_conversation",
    "culinary_science": "daily_conversation",
    "finance_logic": "daily_conversation",
    "fitness_systems": "daily_conversation",
    # Deep Dive
    "deep_diagnostic": "deep_dive",
    "system_report": "deep_dive",
    "comprehensive_analysis": "deep_dive",
    "strategy_planning": "deep_dive",
    # System Call
    "tool_trigger": "system_call",
    "tool_response": "system_call",
}

SOURCE_FILE_TO_BUCKET = {
    "emotion.json": "emotion",
    "self_awareness.json": "self_awareness",
    "email_summary.json": "summarize_email",
    "movie_intro.json": "movie_intro",
    "noise.json": "daily_conversation",
    "system_call.json": "system_call",
    "deep_dive.json": "deep_dive",
}


def resolve_bucket(category: str, source_file: str) -> str:
    cat_bucket = CATEGORY_TO_BUCKET.get(category)
    if cat_bucket:
        return cat_bucket
    source_name = Path(source_file).name
    if source_name in SOURCE_FILE_TO_BUCKET:
        return SOURCE_FILE_TO_BUCKET[source_name]
    if source_file.startswith("emotion/"):
        return "emotion"
    if source_file.startswith("self/"):
        return "self_awareness"
    if source_file.startswith("noise/"):
        return "daily_conversation"
    if source_file.startswith("system_call/"):
        return "system_call"
    if source_file.startswith("movie_intro/") or source_file.startswith("movie/"):
        return "movie_intro"
    if source_file.startswith("deep_dive/") or source_file.startswith("deep/"):
        return "deep_dive"
    if source_file.startswith("email_summary/"):
        return "summarize_email"
    return "daily_conversation"

FORBIDDEN_MANUAL_TOKENS = (
    "<think>",
    "</think>",
    "<final>",
    "</final>",
    "<|im_start|>",
    "<|im_end|>",
)


def build_chatml(system: str, user_input: str, cot: str, output: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{cot}\n</think>\n"
        f"<final>\n{output}\n</final><|im_end|>"
    )


def load_json_list(path: Path) -> list[dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"WARNING: Skipping {path} - JSON decode error: {e}")
        return []
    if isinstance(raw, dict):
        return [raw]
    if not isinstance(raw, list):
        raise ValueError(f"{path} root must be list or dict")
    return raw


def sanitize_text(value: object) -> str:
    return str(value or "").strip()


def validate_no_manual_special_tokens(row: dict, fields: tuple[str, ...]) -> None:
    for field in fields:
        text = sanitize_text(row.get(field))
        for tok in FORBIDDEN_MANUAL_TOKENS:
            if tok in text:
                raise ValueError(
                    f"id={row.get('id', '<missing>')!r} field={field!r} contains forbidden token {tok!r}"
                )


def iter_records(
    src_dir: Path,
    files: list[str],
    allow_missing: bool,
    duplicate_policy: str,
    invalid_row_policy: str,
    dedupe_by_content: bool,
) -> tuple[list[dict], list[str], dict[str, int], list[str], int]:
    rows: list[dict] = []
    missing: list[str] = []
    row_index_by_id: dict[str, int] = {}
    duplicate_count_by_id: dict[str, int] = {}
    invalid_messages: list[str] = []
    seen_content_keys: set[tuple[str, str, str, str]] = set()
    duplicate_content_rows = 0

    for name in files:
        fp = src_dir / name
        if not fp.exists():
            if allow_missing:
                missing.append(name)
                continue
            raise FileNotFoundError(f"missing required file: {fp}")

        for obj in load_json_list(fp):
            if not isinstance(obj, dict):
                continue
            sid = sanitize_text(obj.get("id"))
            cat = sanitize_text(obj.get("category")).lower()
            user_input = sanitize_text(obj.get("input"))
            cot = sanitize_text(obj.get("cot"))
            output = sanitize_text(obj.get("output"))
            history = obj.get("history", [])

            if not sid or not cat or not user_input or not output:
                continue
            try:
                validate_no_manual_special_tokens(obj, ("input", "cot", "output"))
            except ValueError as exc:
                msg = str(exc)
                invalid_messages.append(msg)
                if invalid_row_policy == "skip":
                    continue
                raise

            bucket = resolve_bucket(cat, name)
            system = SYSTEM_PROMPTS[bucket]
            text = build_chatml(system, user_input, cot, output)

            row_obj = {
                "id": sid,
                "category": cat,
                "sys_bucket": bucket,
                "system": system,
                "source_file": name,
                "history": history if isinstance(history, list) else [],
                "input": user_input,
                "cot": cot,
                "output": output,
                "text": text,
            }

            if dedupe_by_content:
                content_key = (cat, user_input, cot, output)
                if content_key in seen_content_keys:
                    duplicate_content_rows += 1
                    continue
                seen_content_keys.add(content_key)

            old_index = row_index_by_id.get(sid)
            if old_index is None:
                row_index_by_id[sid] = len(rows)
                rows.append(row_obj)
                continue

            duplicate_count_by_id[sid] = duplicate_count_by_id.get(sid, 0) + 1
            if duplicate_policy == "error":
                raise ValueError(f"duplicate id detected: {sid}")
            if duplicate_policy == "keep-first":
                continue
            if duplicate_policy == "keep-last":
                rows[old_index] = row_obj
                continue
            if duplicate_policy == "keep-all":
                rows.append(row_obj)
                continue
            raise ValueError(f"unsupported duplicate policy: {duplicate_policy}")
    return rows, missing, duplicate_count_by_id, invalid_messages, duplicate_content_rows


def main() -> None:
    default_root = Path(__file__).resolve().parent
    default_out = default_root / "stf_cot_hf"

    parser = argparse.ArgumentParser(description="Export cot_dataset JSON into HF dataset snapshot.")
    parser.add_argument("--src-dir", type=Path, default=default_root, help="Directory containing JSON files.")
    parser.add_argument(
        "--files",
        type=str,
        default=(
            "emotion.json,self_awareness.json,email_summary.json,movie_intro.json,"
            "noise.json,system_call.json,deep_dive.json"
        ),
        help="Comma-separated JSON file list.",
    )
    parser.add_argument("--out", type=Path, default=default_out, help="Output directory for Dataset.save_to_disk.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing files in --files list (useful when deep_dive.json is not ready).",
    )
    parser.add_argument(
        "--duplicate-policy",
        type=str,
        default="error",
        choices=("error", "keep-first", "keep-last", "keep-all"),
        help="How to handle duplicate sample ids across multiple JSON files.",
    )
    parser.add_argument(
        "--invalid-row-policy",
        type=str,
        default="error",
        choices=("error", "skip"),
        help="How to handle rows containing forbidden manual special tokens.",
    )
    parser.add_argument(
        "--dedupe-by-content",
        action="store_true",
        help="Drop rows with duplicated (category,input,cot,output), regardless of id.",
    )
    parser.add_argument(
        "--rewrite-id-prefix",
        type=str,
        default="",
        help="If set, rewrite all ids to '{prefix}{index:06d}' after filtering.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows so different sys_bucket samples are uniformly mixed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --shuffle (default: 42).",
    )
    args = parser.parse_args()

    file_list = [x.strip() for x in args.files.split(",") if x.strip()]
    rows, missing, duplicate_count_by_id, invalid_messages, duplicate_content_rows = iter_records(
        args.src_dir.resolve(),
        file_list,
        allow_missing=args.allow_missing,
        duplicate_policy=args.duplicate_policy,
        invalid_row_policy=args.invalid_row_policy,
        dedupe_by_content=args.dedupe_by_content,
    )
    if not rows:
        raise SystemExit("No valid rows were collected.")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(rows)
        print(f"   🔀 Shuffled {len(rows)} rows (seed={args.seed})")

    if args.rewrite_id_prefix:
        prefix = args.rewrite_id_prefix
        for i, row in enumerate(rows, start=1):
            row["id"] = f"{prefix}{i:06d}"

    args.out.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(args.out))

    summary = {
        "num_rows": len(rows),
        "output_dir": str(args.out.resolve()),
        "files": file_list,
        "missing_files": missing,
        "duplicate_policy": args.duplicate_policy,
        "duplicate_ids": len(duplicate_count_by_id),
        "duplicate_rows_dropped_or_overwritten": int(sum(duplicate_count_by_id.values())),
        "invalid_row_policy": args.invalid_row_policy,
        "invalid_rows": len(invalid_messages),
        "invalid_examples": invalid_messages[:10],
        "dedupe_by_content": args.dedupe_by_content,
        "duplicate_content_rows_removed": duplicate_content_rows,
        "rewrite_id_prefix": args.rewrite_id_prefix,
        "shuffled": args.shuffle,
        "shuffle_seed": args.seed if args.shuffle else None,
        "columns": ds.column_names,
        "system_prompts": SYSTEM_PROMPTS,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
