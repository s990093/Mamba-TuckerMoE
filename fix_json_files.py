#!/usr/bin/env python3
import json
import re
from pathlib import Path

cot_dir = Path("/Users/hungwei/Desktop/Proj/Mamba3-XR/cot_dataset")

# List of files with JSON errors
bad_files = [
    "mail/8.json",
    "movie/14.json",
    "movie/17.json",
    "movie/18.json",
    "movie/21.json",
    "movie/50.json",
    "movie/52.json",
    "movie/53.json",
    "movie/54.json",
    "movie/55.json",
    "movie/57.json",
    "noise/104.json",
    "noise/105.json",
    "noise/12.json",
]

def repair_json(filepath):
    """Attempt to repair JSON by escaping unescaped quotes and fixing control characters."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Cannot read {filepath}: {e}")
        return False

    # Try to parse - if it works, no fix needed
    try:
        json.loads(content)
        print(f"✓ {filepath.relative_to(cot_dir)} is already valid")
        return True
    except json.JSONDecodeError as e:
        print(f"⚙️  Attempting to fix {filepath.relative_to(cot_dir)}")
        print(f"   Error: {e}")

    # Strategy: Parse object by object
    lines = content.split('\n')
    fixed = False

    # For empty files
    if not content.strip() or content.strip() == "[]":
        print(f"   Skipping empty/empty-array file")
        return False

    # Try removing control characters (null bytes, etc)
    original_len = len(content)
    content = ''.join(c for c in content if ord(c) >= 32 or c in '\n\r\t')
    if len(content) != original_len:
        print(f"   Removed {original_len - len(content)} control characters")
        fixed = True

    # Attempt to fix by carefully escaping quotes in string values
    # This is a heuristic approach - look for patterns like ": "... unescaped_quote ...""

    # Try to parse again
    try:
        json.loads(content)
        if fixed:
            filepath.write_text(content, encoding="utf-8")
            print(f"   ✓ Fixed and saved")
        return True
    except json.JSONDecodeError as e:
        print(f"   ✗ Still broken after control char removal")
        print(f"   Trying alternative fix strategy...")

        # Try to rebuild from individual objects
        return attempt_object_by_object_fix(filepath, content)

    return False

def attempt_object_by_object_fix(filepath, content):
    """Try to extract and rebuild valid objects from the file."""
    # Find all { ... } pairs and try to validate them
    try:
        # Try adding closing bracket if missing
        if not content.rstrip().endswith(']'):
            test_content = content.rstrip() + ']'
            try:
                json.loads(test_content)
                filepath.write_text(test_content, encoding="utf-8")
                print(f"   ✓ Fixed by adding closing bracket")
                return True
            except:
                pass

        # Try removing the last incomplete object
        match = re.search(r'\}\s*,\s*\{\s*"id":\s*[^}]*$', content)
        if match:
            test_content = content[:match.start()] + '}'
            if test_content.endswith(','):
                test_content = test_content[:-1]
            try:
                json.loads(test_content)
                filepath.write_text(test_content, encoding="utf-8")
                print(f"   ✓ Fixed by removing incomplete final object")
                return True
            except:
                pass

        print(f"   Cannot auto-fix {filepath.relative_to(cot_dir)}")
        return False
    except Exception as e:
        print(f"   Error during fix attempt: {e}")
        return False

# Process each bad file
for bad_file in bad_files:
    filepath = cot_dir / bad_file
    if filepath.exists():
        repair_json(filepath)
    else:
        print(f"? {bad_file} not found")

print("\n=== Summary ===")
print("Re-scanning all JSON files...")
for json_file in sorted(cot_dir.rglob("*.json")):
    try:
        json.loads(json_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"✗ Still broken: {json_file.relative_to(cot_dir)}")
        print(f"  {e}")
