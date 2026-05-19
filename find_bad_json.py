#!/usr/bin/env python3
import json
from pathlib import Path

cot_dir = Path("/Users/hungwei/Desktop/Proj/Mamba3-XR/cot_dataset")

# Check all JSON files
for json_file in sorted(cot_dir.rglob("*.json")):
    try:
        json.loads(json_file.read_text(encoding="utf-8"))
        print(f"✓ {json_file.relative_to(cot_dir)}")
    except json.JSONDecodeError as e:
        print(f"✗ {json_file.relative_to(cot_dir)}")
        print(f"  Error: {e}")
