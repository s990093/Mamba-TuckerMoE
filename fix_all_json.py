#!/usr/bin/env python3
import json
import re
from pathlib import Path

cot_dir = Path("/Users/hungwei/Desktop/Proj/Mamba3-XR/cot_dataset")

def find_and_report_issues(filepath):
    """Find all JSON issues in a file and report them."""
    try:
        content = filepath.read_text(encoding="utf-8")
        json.loads(content)
        return []  # No issues
    except json.JSONDecodeError as e:
        return [(e.lineno, e.colno, e.msg)]

# Find all broken files
broken_files = []
for json_file in sorted(cot_dir.rglob("*.json")):
    issues = find_and_report_issues(json_file)
    if issues:
        broken_files.append((json_file, issues))

print(f"Found {len(broken_files)} broken JSON files:\n")
for filepath, issues in broken_files:
    rel_path = filepath.relative_to(cot_dir)
    print(f"{rel_path}")
    for line, col, msg in issues:
        print(f"  Line {line}, Col {col}: {msg}")

# Now try to fix them by looking at the specific issues
print("\n\nAttempting fixes...")
for filepath, issues in broken_files:
    rel_path = filepath.relative_to(cot_dir)
    for line_num, col_num, msg in issues:
        if "Invalid \\escape" in msg or "Invalid control character" in msg:
            # Try to fix escape sequences
            content = filepath.read_text(encoding="utf-8")
            lines = content.split('\n')
            if line_num - 1 < len(lines):
                line = lines[line_num - 1]

                # Fix invalid escape sequences like \S -> \\S
                # But preserve valid ones like \n, \t, etc.
                fixed = re.sub(
                    r'(?<!\\)\\([^"\\nrbtfv/uU])',
                    r'\\\\\\1',
                    line
                )

                if fixed != line:
                    lines[line_num - 1] = fixed
                    new_content = '\n'.join(lines)
                    try:
                        json.loads(new_content)
                        filepath.write_text(new_content, encoding="utf-8")
                        print(f"✓ Fixed {rel_path} (escape sequences)")
                    except json.JSONDecodeError as e2:
                        print(f"  Still broken: {e2.msg}")

# Re-check
print("\n\nFinal scan...")
still_broken = []
for json_file in sorted(cot_dir.rglob("*.json")):
    issues = find_and_report_issues(json_file)
    if issues:
        still_broken.append((json_file, issues))

if still_broken:
    print(f"Still {len(still_broken)} files with issues:")
    for filepath, issues in still_broken:
        rel_path = filepath.relative_to(cot_dir)
        print(f"  {rel_path}: {issues[0][2]}")
else:
    print(f"✓ All JSON files are now valid!")
