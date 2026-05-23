#!/usr/bin/env python3
"""
Generate math_drill.json — capped English arithmetic drills (~200 rows).

Design goals (see MATH_DRILL.md):
  - Avoid over-representation: sample per tier, do NOT exhaust ×100 or full tables.
  - Natural CoT (blackboard style), NO "Step 1:" / Parse operands / Emit answer.
  - output: digits only (wrapper added by stf_cot_to_bin.py as <final>).

Usage:
  python3 cot_dataset/scripts/generate_math_drill.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

RNG = random.Random(42)

# Per-tier caps (≈20–50 each). Total must match sum(QUOTAS).
QUOTAS: dict[str, int] = {
    "arith_add_units": 40,
    "arith_mul_table": 48,
    "arith_mul_teens": 24,
    "arith_add_mixed": 36,
    "arith_mul_extended": 36,
    "arith_mul_hundred": 16,
}
TARGET_TOTAL = sum(QUOTAS.values())  # 200

ADD_TEMPLATES = (
    "What is {a} plus {b}?",
    "What is {a} + {b}?",
    "How much is {a} plus {b}?",
)
MUL_TEMPLATES = (
    "What is {a} times {b}?",
    "What is {a} multiplied by {b}?",
    "{a} × {b} = ?",
)


def _eval_op(a: int, b: int, op: str) -> int:
    if op == "+":
        return a + b
    if op == "*":
        return a * b
    raise ValueError(op)


def _sample_pool(category: str) -> list[tuple[int, int, str]]:
    if category == "arith_add_units":
        pool = [(a, b, "+") for a in range(0, 10) for b in range(0, 10)]
    elif category == "arith_mul_table":
        pool = [(a, b, "*") for a in range(1, 13) for b in range(1, 13)]
    elif category == "arith_mul_teens":
        pool = [(a, b, "*") for a in range(13, 20) for b in range(13, 20)]
    elif category == "arith_add_mixed":
        pool = [(a, b, "+") for a in range(10, 100) for b in range(10, 100)]
    elif category == "arith_mul_extended":
        pool = [(a, b, "*") for a in range(20, 100) for b in range(20, 100)]
    elif category == "arith_mul_hundred":
        pool = [(n, 100, "*") for n in range(1, 101)]
    else:
        raise ValueError(category)
    RNG.shuffle(pool)
    return pool[: QUOTAS[category]]


def build_specs() -> list[tuple[str, int, int, str]]:
    specs: list[tuple[str, int, int, str]] = []
    for cat in QUOTAS:
        for a, b, op in _sample_pool(cat):
            specs.append((cat, a, b, op))
    if len(specs) != TARGET_TOTAL:
        raise RuntimeError(f"expected {TARGET_TOTAL} specs, got {len(specs)}")
    return specs


def _natural_cot(category: str, a: int, b: int, op: str, result: int, idx: int) -> str:
    """Short human-style reasoning. Never use Step N: or compiler-log phrasing."""
    if category == "arith_mul_hundred" or (op == "*" and b == 100):
        lines = [
            (
                f"Multiplying {a} by 100 is the same as appending two zeros — "
                f"the digits shift two places to the left.\n{a} → {result}."
            ),
            (
                f"For whole numbers, ×100 means tack on \"00\" at the end.\n"
                f"So {a} × 100 = {result}."
            ),
        ]
        return lines[idx % len(lines)]

    if category == "arith_mul_table" and a <= 12 and b <= 12:
        lines = [
            f"{a} × {b} is a basic table fact: {result}.",
            f"From the multiplication table, {a} times {b} equals {result}.",
        ]
        return lines[idx % len(lines)]

    if op == "+":
        if category == "arith_add_units":
            lines = [
                f"Adding {a} and {b} gives {result}.",
                f"{a} plus {b} combines to {result}.",
            ]
        else:
            lines = [
                f"Add the two numbers: {a} + {b} = {result}.",
                (
                    f"Break it down: {a} + {b}. "
                    f"Ones and tens line up to {result}."
                ),
            ]
        return lines[idx % len(lines)]

    # multiplication (general)
    if a <= 12 or b <= 12:
        lines = [
            f"{a} × {b} works out to {result}.",
            f"Multiply {a} by {b}: {result}.",
        ]
    else:
        tens_b, ones_b = (b // 10) * 10, b % 10
        partial = a * tens_b
        lines = [
            f"{a} × {b} = {result}.",
            (
                f"Split {b} into {tens_b} + {ones_b}. "
                f"{a}×{tens_b}={partial}, plus {a}×{ones_b}={a * ones_b}, "
                f"total {result}."
            ),
        ]
    return lines[idx % len(lines)]


def _input_text(a: int, b: int, op: str, idx: int) -> str:
    templates = ADD_TEMPLATES if op == "+" else MUL_TEMPLATES
    return templates[idx % len(templates)].format(a=a, b=b)


def build_records() -> list[dict]:
    specs = build_specs()
    records: list[dict] = []
    for i, (category, a, b, op) in enumerate(specs):
        result = _eval_op(a, b, op)
        records.append({
            "id": f"mat_{i + 1:04d}",
            "category": category,
            "input": _input_text(a, b, op, i),
            "cot": _natural_cot(category, a, b, op, result, i),
            "output": str(result),
        })
    return records


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=f"Generate math_drill.json ({TARGET_TOTAL} rows).")
    parser.add_argument("--out", type=Path, default=root / "math_drill.json")
    args = parser.parse_args()
    records = build_records()
    args.out.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    by_cat: dict[str, int] = {}
    for r in records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"Wrote {len(records)} rows → {args.out}")
    print("By category:", dict(sorted(by_cat.items())))


if __name__ == "__main__":
    main()
