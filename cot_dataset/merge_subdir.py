#!/usr/bin/env python3
"""
merge_subdir.py — 把子目錄的 JSON 合併回對應的主 JSON 檔案。

對映：
    emotion/       → emotion.json
    self/          → self_awareness.json
    noise/         → noise.json
    movie/         → movie_intro.json
    mail/          → email_summary.json
    system_call/   → system_call.json
    deep_dive/     → deep_dive.json

合併邏輯：
    1. 讀取主 JSON（若不存在則從空 list 開始）
    2. 讀取子目錄所有 *.json
    3. 以 (category, input, cot, output) 做內容去重，保留後出現的版本
    4. 寫回主 JSON（格式化，2-space indent）
    5. 印出合併摘要

用法：
    cd cot_dataset
    python3 merge_subdir.py              # 合併全部
    python3 merge_subdir.py --dry-run    # 只印摘要，不寫入
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

SUBDIR_TO_ROOT = {
    "emotion":      "emotion.json",
    "self":         "self_awareness.json",
    "noise":        "noise.json",
    "movie":        "movie_intro.json",
    "mail":         "email_summary.json",
    "system_call":  "system_call.json",
    "deep_dive":    "deep_dive.json",
    "math_drill":   "math_drill.json",
    "math":         "math_drill.json",
}


def content_key(entry: dict) -> tuple[str, str, str, str]:
    return (
        (entry.get("category") or "").strip().lower(),
        (entry.get("input") or "").strip(),
        (entry.get("cot") or "").strip(),
        (entry.get("output") or "").strip(),
    )


def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"  ⚠️  JSON parse error in {path.name}: {exc}")
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return raw
    return []


def merge_one(subdir_name: str, root_name: str, dry_run: bool) -> dict:
    subdir_path = SCRIPT_DIR / subdir_name
    root_path = SCRIPT_DIR / root_name

    if not subdir_path.is_dir():
        return {"subdir": subdir_name, "status": "skip", "reason": "dir not found"}

    sub_files = sorted(subdir_path.glob("*.json"))
    if not sub_files:
        return {"subdir": subdir_name, "status": "skip", "reason": "no json files"}

    root_entries = load_json_list(root_path)
    root_before = len(root_entries)

    sub_entries: list[dict] = []
    for fp in sub_files:
        sub_entries.extend(load_json_list(fp))

    seen_keys: set[tuple[str, str, str, str]] = set()
    merged: list[dict] = []

    for entry in root_entries + sub_entries:
        ck = content_key(entry)
        if ck in seen_keys:
            continue
        seen_keys.add(ck)
        merged.append(entry)

    added = len(merged) - root_before
    deduped = (root_before + len(sub_entries)) - len(merged)

    if not dry_run:
        root_path.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "subdir": subdir_name,
        "root": root_name,
        "status": "dry-run" if dry_run else "merged",
        "root_before": root_before,
        "sub_files": len(sub_files),
        "sub_entries": len(sub_entries),
        "added": added,
        "content_deduped": deduped,
        "total_after": len(merged),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge subdirectory JSONs into root JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing.")
    args = parser.parse_args()

    print("=" * 60)
    print(" merge_subdir.py —", "DRY RUN" if args.dry_run else "MERGING")
    print("=" * 60)

    for subdir, root in SUBDIR_TO_ROOT.items():
        result = merge_one(subdir, root, dry_run=args.dry_run)
        if result["status"] == "skip":
            print(f"\n  ⏭  {subdir}/ — skipped ({result['reason']})")
            continue

        print(f"\n  {'🔍' if args.dry_run else '✅'}  {subdir}/ → {root}")
        print(f"      主 JSON 原有: {result['root_before']}")
        print(f"      子目錄檔案數: {result['sub_files']}")
        print(f"      子目錄 entries: {result['sub_entries']}")
        print(f"      內容去重移除: {result['content_deduped']}")
        print(f"      新增 entries: {result['added']}")
        print(f"      合併後總數:   {result['total_after']}")

    print("\n" + "=" * 60)
    if args.dry_run:
        print(" 以上為預覽，加 --dry-run 取消即可實際寫入")
    else:
        print(" 合併完成！子目錄檔案未刪除，可手動清理。")
    print("=" * 60)


if __name__ == "__main__":
    main()
