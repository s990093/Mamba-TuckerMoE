#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
結構權重視覺化：抽樣顯示高權重 token 標註，用 HTML 或 terminal 顏色驗證 RE 標註正確性。

輸入：
  - HF dataset（文本）
  - structure_weights/*.npz（權重與結構索引）
  - tokenizer

輸出：
  - reports/structure_samples.html（互動式視覺化）
  - reports/structure_samples.txt（terminal 颜色版本）

使用：
  python visualize_structure_weights.py \\
    --hf-path sft_cot_bundle/dataset/sft_cot_hf \\
    --weights-dir reports/structure_weights \\
    --sample-count 20
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_PATH = PROJECT_ROOT / "sft_cot_bundle" / "dataset" / "stf_cot_hf"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "cot" / "reports" / "structure_weights"
DEFAULT_TOKENIZER = PROJECT_ROOT / "sft_cot_bundle" / "dataset" / "tokenizer"

# ANSI 颜色
COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}

# Pattern 類型到 HTML 顏色的映射（6 種 patterns）
PATTERN_COLORS_HTML = {
    "step": "#FFB6C6",        # 淺粉紅 (Step 推理)
    "pipe": "#FFE699",        # 淺黃 (表格豎線)
    "separator": "#C6E0B4",   # 淺綠 (分隔行)
    "bold": "#B4C7E7",        # 淺藍 (粗體)
    "heading": "#E2EFDA",     # 極淺綠 (標題)
    "fenced_code": "#F4B084", # 淺橙 (代碼塊)
}

# Pattern 類型到 ANSI 顏色的映射
PATTERN_COLORS_ANSI = {
    "step": "red",
    "pipe": "yellow",
    "separator": "green",
    "bold": "blue",
    "heading": "magenta",
    "fenced_code": "cyan",
}


def load_hf_dataset(hf_path: Path | str) -> Dataset:
    """載入 HF dataset。"""
    hf_path = Path(hf_path)
    if not hf_path.exists():
        raise FileNotFoundError(f"HF dataset 不存在: {hf_path}")
    ds = load_from_disk(str(hf_path))
    if isinstance(ds, dict):
        if "train" in ds:
            return ds["train"]
        return ds[next(iter(ds))]
    return ds


def load_tokenizer(tokenizer_dir: Path | str) -> PreTrainedTokenizerFast:
    """載入 tokenizer。"""
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    return tok


def load_weights(weights_dir: Path) -> dict:
    """加載所有 .npz 權重檔。"""
    weights_dir = Path(weights_dir)
    weights = {}
    for npz_file in sorted(weights_dir.glob("*.npz")):
        sample_id = npz_file.stem
        data = np.load(npz_file, allow_pickle=True)
        weights[sample_id] = {
            "weight": data["weight"],
            "structure_indices": data["structure_indices"],
            "token_patterns": data.get("token_patterns", np.array([])),  # 新增
        }
    return weights


def tokenize_with_offsets(text: str, tok: PreTrainedTokenizerFast):
    """Tokenize 文本，取得 offset mapping。"""
    enc = tok(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    return enc["input_ids"], enc["offset_mapping"]


def highlight_text_by_tokens(
    text: str,
    token_ids: list[int],
    offset_mapping: list[tuple[int, int]],
    structure_indices: np.ndarray,
    weights: np.ndarray,
    token_patterns: np.ndarray = None,
    w_threshold: float = 1.5,
) -> str:
    """
    將文本中高權重的 token 用顏色標註。
    若有 token_patterns，按 pattern 類型著色；否則全用紅色。
    """
    # 構造字元 -> (是否高權重, pattern 類型) 的映射
    char_info = [(False, "")] * len(text)
    char_info = [(False, "") for _ in range(len(text))]

    for tok_idx in structure_indices:
        if tok_idx < len(offset_mapping):
            char_start, char_end = offset_mapping[tok_idx]
            weight = float(weights[tok_idx]) if tok_idx < len(weights) else 1.0
            if weight > w_threshold:
                pattern = ""
                if token_patterns is not None and tok_idx < len(token_patterns):
                    pattern = str(token_patterns[tok_idx])
                for c_idx in range(char_start, min(char_end, len(text))):
                    char_info[c_idx] = (True, pattern)

    # 重建文本，加顏色標記
    result = []
    in_highlight = False
    current_pattern = ""
    for c_idx, char in enumerate(text):
        should_highlight, pattern = char_info[c_idx]
        if should_highlight:
            if not in_highlight or current_pattern != pattern:
                if in_highlight:
                    result.append(f'{COLORS["reset"]}')
                color_name = PATTERN_COLORS_ANSI.get(pattern, "red") if pattern else "red"
                result.append(f'{COLORS[color_name]}{COLORS["bold"]}')
                in_highlight = True
                current_pattern = pattern
        else:
            if in_highlight:
                result.append(f'{COLORS["reset"]}')
                in_highlight = False
                current_pattern = ""
        result.append(char)

    if in_highlight:
        result.append(f'{COLORS["reset"]}')

    return "".join(result)


def generate_html_samples(
    samples: list[dict],
    output_path: Path,
) -> None:
    """生成互動式 HTML 視覺化。"""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Structure Weights Visualization (6 Pattern Types)</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        .sample {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            border-left: 4px solid #0066cc;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample-id {
            font-weight: bold;
            color: #0066cc;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .sample-stats {
            font-size: 12px;
            color: #666;
            margin-bottom: 10px;
            padding: 8px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .text-content {
            line-height: 1.8;
            word-wrap: break-word;
            white-space: pre-wrap;
            font-family: "Courier New", monospace;
            font-size: 13px;
            padding: 10px;
            background: #fafafa;
            border-radius: 4px;
            border: 1px solid #eee;
        }
        .pattern-legend {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin: 15px 0;
            padding: 12px;
            background: #fff9f0;
            border-radius: 4px;
        }
        .pattern-tag {
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        h2 {
            color: #0066cc;
            font-size: 18px;
            margin-top: 30px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #0066cc;
        }
        h3 {
            color: #333;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .summary {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .params-section {
            margin: 15px 0;
            padding: 12px;
            background: #f0f8ff;
            border-radius: 6px;
            border-left: 4px solid #0066cc;
        }
        .params-section h3 {
            margin: 0 0 10px 0;
            color: #0066cc;
            font-size: 14px;
        }
        .params-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            background: white;
            border-radius: 4px;
            overflow: hidden;
        }
        .params-table th {
            background: #0066cc;
            color: white;
            padding: 8px;
            text-align: left;
            font-weight: bold;
        }
        .params-table td {
            padding: 8px;
            border-bottom: 1px solid #e0e0e0;
        }
        .params-table tr:hover {
            background: #f5f5f5;
        }
        .params-table tr:last-child td {
            border-bottom: none;
        }
    </style>
</head>
<body>
    <h1>CoT Structure Weights Visualization (6 Pattern Types)</h1>
    <div class="summary">
        <p><strong>Total Samples:</strong> """ + str(len(samples)) + """</p>
        <p><strong>Visualization:</strong> High-weight structure tokens highlighted by pattern type</p>
        <h2>Pattern Legend (R1-R6)</h2>
        <div class="pattern-legend">
            <div><span class="pattern-tag" style="background-color:#FFB6C6;">R1: Step</span> Reasoning steps (Step N:)</div>
            <div><span class="pattern-tag" style="background-color:#FFE699;">R2: Pipe</span> Table vertical bars (|)</div>
            <div><span class="pattern-tag" style="background-color:#C6E0B4;">R3: Separator</span> Table separator rows</div>
            <div><span class="pattern-tag" style="background-color:#B4C7E7;">R4: Bold</span> Bold text (**...)**</div>
            <div><span class="pattern-tag" style="background-color:#E2EFDA;">R5: Heading</span> Markdown headings (#)</div>
            <div><span class="pattern-tag" style="background-color:#F4B084;">R6: Code</span> Fenced code blocks (```)</div>
        </div>

        <h2>📊 Statistical Charts</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
            <div>
                <h3>Structure Token Distribution</h3>
                <img src="plot_structure_histogram.png" style="width: 100%; max-width: 500px; border-radius: 6px; border: 1px solid #ddd;">
            </div>
            <div>
                <h3>Pattern Type Breakdown</h3>
                <img src="plot_pattern_pie.png" style="width: 100%; max-width: 500px; border-radius: 6px; border: 1px solid #ddd;">
            </div>
            <div>
                <h3>Sequence Length Distribution</h3>
                <img src="plot_length_histogram.png" style="width: 100%; max-width: 500px; border-radius: 6px; border: 1px solid #ddd;">
            </div>
            <div>
                <h3>Weight Distribution</h3>
                <img src="plot_weight_distribution.png" style="width: 100%; max-width: 500px; border-radius: 6px; border: 1px solid #ddd;">
            </div>
            <div>
                <h3>Weight Statistics by Sample</h3>
                <img src="plot_weight_statistics.png" style="width: 100%; max-width: 500px; border-radius: 6px; border: 1px solid #ddd;">
            </div>
            <div>
                <h3>SCALe Schedule (Training)</h3>
                <img src="plot_scale_schedule.png" style="width: 100%; max-width: 500px; border-radius: 6px; border: 1px solid #ddd;">
            </div>
        </div>

        <hr style="margin: 30px 0; border: 1px solid #ddd;">
    </div>

    <h2>📋 Sample-by-Sample Analysis</h2>
"""

    for sample in samples:
        # 生成 pattern 詳細信息表
        pattern_details_html = "<tr><th>Pattern</th><th>Type</th><th>Count</th></tr>"
        pattern_name_map = {
            "step": "R1: Step",
            "pipe": "R2: Pipe",
            "separator": "R3: Separator",
            "bold": "R4: Bold",
            "heading": "R5: Heading",
            "fenced_code": "R6: Code",
        }
        for pattern_key, pattern_display in pattern_name_map.items():
            count = sample['pattern_details'].get(pattern_key, 0)
            if count > 0:
                color = PATTERN_COLORS_HTML.get(pattern_key, "#CCCCCC")
                pattern_details_html += f'<tr><td style="background-color:{color}; font-weight:bold">{pattern_display}</td><td>{pattern_key.replace("_", " ").title()}</td><td>{count}</td></tr>'

        # 計算額外的統計信息
        structure_ratio = (sample['structure_count'] / sample['total_tokens'] * 100) if sample['total_tokens'] > 0 else 0

        html += f"""    <div class="sample">
        <div class="sample-id">📊 Sample: {sample['sample_id']}</div>

        <div class="params-section">
            <h3>📈 Token Statistics</h3>
            <table class="params-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Tokens</td><td>{sample['total_tokens']}</td></tr>
                <tr><td>Structure Tokens</td><td>{sample['structure_count']} ({structure_ratio:.1f}%)</td></tr>
                <tr><td>Assistant Start Token</td><td>{sample['assistant_start_token']}</td></tr>
                <tr><td>Assistant Tokens</td><td>{sample['assistant_tokens']}</td></tr>
                <tr><td>Text Length (chars)</td><td>{sample['text_length']}</td></tr>
            </table>
        </div>

        <div class="params-section">
            <h3>⚖️ Weight Statistics</h3>
            <table class="params-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Min Weight</td><td>{sample['weight_min']:.4f}</td></tr>
                <tr><td>Max Weight</td><td>{sample['weight_max']:.4f}</td></tr>
                <tr><td>Mean Weight</td><td>{sample['weight_mean']:.4f}</td></tr>
                <tr><td>Std Dev</td><td>{sample['weight_std']:.4f}</td></tr>
                <tr><td>Weight Range</td><td>[{sample['weight_min']:.2f}, {sample['weight_max']:.2f}]</td></tr>
            </table>
        </div>

        <div class="params-section">
            <h3>🏷️ Pattern Distribution</h3>
            <table class="params-table">
                {pattern_details_html}
            </table>
            <p style="margin-top: 8px; font-size: 12px; color: #666;">Active Patterns: {', '.join(sample['patterns']) if sample['patterns'] else 'None'}</p>
        </div>

        <div class="text-content" style="margin-top: 15px;">
            <h3>📝 Text Content (Highlighted Tokens)</h3>
            {sample['html_text']}
        </div>
        <div class="legend">ℹ️ Text truncated to first 2000 chars for readability. Colors indicate pattern type of high-weight tokens.</div>
    </div>
"""

    html += """</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def generate_text_samples(
    samples: list[dict],
    output_path: Path,
) -> None:
    """生成 terminal 格式的視覺化（ANSI 顏色）。"""
    lines = [
        "=" * 80,
        "Structure Weights Visualization (Terminal)",
        "=" * 80,
        "",
    ]

    for i, sample in enumerate(samples, 1):
        structure_ratio = (sample['structure_count'] / sample['total_tokens'] * 100) if sample['total_tokens'] > 0 else 0

        lines.extend([
            f"Sample {i}/{len(samples)}: {sample['sample_id']}",
            "",
            "  📊 Token Statistics:",
            f"    Total Tokens: {sample['total_tokens']}",
            f"    Structure Tokens: {sample['structure_count']} ({structure_ratio:.1f}%)",
            f"    Assistant Start: {sample['assistant_start_token']}, Assistant Tokens: {sample['assistant_tokens']}",
            f"    Text Length: {sample['text_length']} chars",
            "",
            "  ⚖️ Weight Statistics:",
            f"    Min: {sample['weight_min']:.4f}, Max: {sample['weight_max']:.4f}",
            f"    Mean: {sample['weight_mean']:.4f} ± {sample['weight_std']:.4f}",
            "",
            "  🏷️ Pattern Distribution:",
        ])

        # 添加 pattern 詳細信息
        for pattern_key, pattern_display in [
            ("step", "R1: Step"),
            ("pipe", "R2: Pipe"),
            ("separator", "R3: Separator"),
            ("bold", "R4: Bold"),
            ("heading", "R5: Heading"),
            ("fenced_code", "R6: Code"),
        ]:
            count = sample['pattern_details'].get(pattern_key, 0)
            if count > 0:
                lines.append(f"    {pattern_display}: {count}")

        if not sample['patterns']:
            lines.append("    (None)")

        lines.extend([
            "",
            sample["terminal_text"],
            "",
            "-" * 80,
            "",
        ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Terminal-formatted samples saved to: {output_path}")


def visualize_samples(
    dataset: Dataset,
    weights_dict: dict,
    tok: PreTrainedTokenizerFast,
    sample_count: int = 20,
    output_dir: Path = None,
) -> list[dict]:
    """
    抽樣視覺化：隨機選擇 sample_count 筆樣本，對每筆高亮權重。

    回傳包含 HTML 和 terminal 文本的樣本列表。
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "cot" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 隨機抽樣
    available_samples = list(weights_dict.keys())
    selected = random.sample(available_samples, min(sample_count, len(available_samples)))

    samples = []

    for sample_id in selected:
        # 查找對應的 dataset row
        row = None
        for r in dataset:
            if r.get("id") == sample_id:
                row = r
                break

        if row is None:
            print(f"Warning: Could not find row for sample {sample_id}")
            continue

        text = row.get("text", "")
        weight_data = weights_dict[sample_id]
        weights = weight_data["weight"]
        structure_indices = weight_data["structure_indices"]
        token_patterns = weight_data.get("token_patterns", np.array([]))

        # Tokenize
        token_ids, offset_mapping = tokenize_with_offsets(text, tok)

        # 截斷長文本（HTML 適用）
        truncated_text = text[:2000]
        if len(text) > 2000:
            truncated_text += f"\n\n[... text truncated: {len(text)} chars total ...]"

        # 構造樣本資訊（含詳細參數）
        weight_array = weights
        pattern_hits = weight_data.get("pattern_hits", {})
        assistant_start_token = weight_data.get("assistant_start_token", 0)

        sample_info = {
            "sample_id": sample_id,
            "total_tokens": len(weights),
            "structure_count": len(structure_indices),
            "structure_ratio": f"{len(structure_indices)/len(weights)*100:.1f}%" if len(weights) > 0 else "0%",
            "weight_min": float(np.min(weight_array)),
            "weight_max": float(np.max(weight_array)),
            "weight_mean": float(np.mean(weight_array)),
            "weight_std": float(np.std(weight_array)),
            "patterns": [
                name for name, count in pattern_hits.items()
                if count > 0
            ],
            "pattern_details": pattern_hits,  # 各 pattern 的詳細計數
            "assistant_start_token": assistant_start_token,
            "assistant_tokens": len(weights) - assistant_start_token if assistant_start_token < len(weights) else 0,
            "text_length": weight_data.get("text_length", len(text)),
        }

        # 生成 HTML 版本
        html_text = highlight_text_by_tokens(
            truncated_text, token_ids, offset_mapping, structure_indices, weights,
            token_patterns=token_patterns
        )
        # 將 ANSI 序列轉換為 HTML span（支持多種顏色）
        html_text = html_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for pattern, color in PATTERN_COLORS_ANSI.items():
            ansi_code = COLORS[color]
            html_color = PATTERN_COLORS_HTML.get(pattern, "#FFB6C6")
            html_text = html_text.replace(
                ansi_code, f'<span style="background-color:{html_color}; font-weight:bold">'
            )
        html_text = html_text.replace(COLORS["bold"], "")
        html_text = html_text.replace(COLORS["reset"], "</span>")
        sample_info["html_text"] = html_text

        # 生成 terminal 版本
        terminal_text = highlight_text_by_tokens(
            truncated_text, token_ids, offset_mapping, structure_indices, weights,
            token_patterns=token_patterns
        )
        sample_info["terminal_text"] = terminal_text

        samples.append(sample_info)

    # 生成輸出檔案
    html_path = output_dir / "structure_samples.html"
    text_path = output_dir / "structure_samples.txt"

    generate_html_samples(samples, html_path)
    generate_text_samples(samples, text_path)

    print(f"HTML visualization saved to: {html_path}")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Visualize structure weights with highlighted tokens."
    )
    parser.add_argument(
        "--hf-path",
        type=Path,
        default=DEFAULT_HF_PATH,
        help="Path to HF dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help="Path to structure_weights directory (default: %(default)s)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="Path to tokenizer (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "cot" / "reports",
        help="Output directory for visualizations (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=20,
        help="Number of samples to visualize (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: %(default)s)",
    )

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset...")
    dataset = load_hf_dataset(args.hf_path)

    print("Loading tokenizer...")
    tok = load_tokenizer(args.tokenizer_dir)

    print("Loading weights...")
    weights_dict = load_weights(args.weights_dir)
    print(f"Loaded {len(weights_dict)} weight files")

    print(f"Visualizing {args.sample_count} samples...")
    samples = visualize_samples(
        dataset,
        weights_dict,
        tok,
        sample_count=args.sample_count,
        output_dir=args.output_dir,
    )

    print(f"\n✓ Visualization complete ({len(samples)} samples)")
    print(f"  HTML: {args.output_dir / 'structure_samples.html'}")
    print(f"  Text: {args.output_dir / 'structure_samples.txt'}")


if __name__ == "__main__":
    main()
