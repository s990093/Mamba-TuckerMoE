#!/usr/bin/env python3
import json
import re
from pathlib import Path

cot_dir = Path("/Users/hungwei/Desktop/Proj/Mamba3-XR/cot_dataset")

def fix_json_with_smart_escaping(filepath):
    """Fix JSON by properly escaping quotes in string values."""
    content = filepath.read_text(encoding="utf-8")

    try:
        json.loads(content)
        return True  # Already valid
    except json.JSONDecodeError as e:
        pass

    # Fix strategy: Find problematic lines and escape unescaped quotes
    lines = content.split('\n')
    line_num = 1

    # Try to identify and fix the specific error location
    if hasattr(e, 'lineno'):
        error_line = e.lineno - 1
        if error_line < len(lines):
            line = lines[error_line]
            # Look for unescaped quotes in string values
            # Pattern: "key": "value with unescaped "quotes" inside"

            # Fix unescaped quotes in values
            # This regex looks for: ": " followed by content with unescaped quotes
            fixed_line = re.sub(
                r'(?<=[^\\])"([^"]*)"([^"]*)"(?=[,\n\r}])',
                lambda m: f'\\"{m.group(1)}\\" {m.group(2)}\\"',
                line
            )

            if fixed_line != line:
                lines[error_line] = fixed_line
                new_content = '\n'.join(lines)

                try:
                    json.loads(new_content)
                    filepath.write_text(new_content, encoding="utf-8")
                    print(f"✓ Fixed {filepath.relative_to(cot_dir)}")
                    return True
                except:
                    pass

    # More aggressive fix: Try to escape ALL unescaped quotes inside string values
    # Pattern: look for lines with "key": "...content..."
    for i, line in enumerate(lines):
        if re.match(r'\s*"[^"]+"\s*:\s*"', line):  # Line with string value
            # Count quotes - if odd number of quotes (minus escaped ones), it's broken
            unescaped_quotes = len(re.findall(r'(?<!\\)"', line))
            if unescaped_quotes % 2 != 0:  # Odd number = broken
                # Try to fix by escaping the middle unescaped quotes
                new_line = line
                # Replace unescaped quotes (but not the opening ones)
                parts = re.split(r'(?<=[^\\])"(?=[^"]*[\n,}])', new_line, 1)
                if len(parts) == 2:
                    # The first part is correct, the second has the issue
                    prefix = parts[0]
                    rest = parts[1]
                    # Escape quotes in the rest until we find the final "
                    # This is complex, so let's try a different approach
                    pass

    # Last resort: Check if the error is in the last line and truncate
    if not content.rstrip().endswith(']'):
        # Try closing it
        test = content.rstrip() + ']'
        try:
            json.loads(test)
            filepath.write_text(test, encoding="utf-8")
            print(f"✓ Fixed {filepath.relative_to(cot_dir)} by adding closing bracket")
            return True
        except:
            pass

        # Try removing the last object if incomplete
        last_brace = content.rfind('{')
        if last_brace > 0:
            prev_close = content.rfind('}', 0, last_brace)
            if prev_close > 0:
                test = content[:prev_close+1] + ']'
                try:
                    json.loads(test)
                    filepath.write_text(test, encoding="utf-8")
                    print(f"✓ Fixed {filepath.relative_to(cot_dir)} by removing incomplete object")
                    return True
                except:
                    pass

    # Remove control characters
    new_content = ''.join(c for c in content if ord(c) >= 32 or c in '\n\r\t')
    if new_content != content:
        try:
            json.loads(new_content)
            filepath.write_text(new_content, encoding="utf-8")
            print(f"✓ Fixed {filepath.relative_to(cot_dir)} by removing control characters")
            return True
        except:
            pass

    print(f"✗ Cannot fix {filepath.relative_to(cot_dir)}")
    return False

# Focus on the critical one first
critical_file = cot_dir / "noise/104.json"
print(f"Fixing {critical_file.relative_to(cot_dir)}...")
fix_json_with_smart_escaping(critical_file)

# Re-scan
try:
    json.loads(critical_file.read_text(encoding="utf-8"))
    print(f"✓ noise/104.json is now valid!")
except json.JSONDecodeError as e:
    print(f"✗ Still broken: {e}")
