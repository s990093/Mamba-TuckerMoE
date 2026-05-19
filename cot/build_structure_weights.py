#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RE 標註與結構權重產生：讀取已 tokenize 的 ChatML 資料，用 RE 找出結構 token，
映射到 token index，產生 structure_weights 檔案。

輸入：
  - HF dataset 或 `.bin` 檔（含 input_ids）
  - 原始文本（assistant 區段）

輸出：
  - structure_weights/sample_{i:04d}.npz（含 token 索引與權重）
  - structure_weights_metadata.json（統計資訊）

使用：
  python build_structure_weights.py \\
    --hf-path sft_cot_bundle/dataset/sft_cot_hf \\
    --tokenizer-dir third_party/Tiny-LLM-merged-32007 \\
    --output-dir reports/structure_weights
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_PATH = PROJECT_ROOT / "sft_cot_bundle" / "dataset" / "stf_cot_hf"
DEFAULT_TOKENIZER = PROJECT_ROOT / "sft_cot_bundle" / "dataset" / "tokenizer"

# RE 模式表（來自 TASK2_LOSS_ENGINEERING.md §4.2）
# R1: 推理步驟導引，R2: Markdown 豎線，R3: 表格分隔行，R4: 粗體
# R5: 行首標題，R6: fenced code
# Note: R1b (中文步驟) 為防呆，GUIDE 要求全英文故不實現
RE_PATTERNS = {
    "step": (r"Step\s*\d+[:：]?", re.IGNORECASE | re.MULTILINE),         # R1
    "pipe": (r"\|", 0),                                                   # R2
    "separator": (r"\|?[\s\-]*\|[\s\-]*\|?", re.MULTILINE),             # R3
    "bold": (r"\*\*[^*]+\*\*", 0),                                       # R4
    "heading": (r"^#+\s", re.MULTILINE),                                 # R5
    "fenced_code": (r"```", 0),                                          # R6
}

@dataclass
class TokenizationResult:
    """Tokenizer 結果：ids、offset mapping、原文"""
    input_ids: list[int]
    offset_mapping: list[tuple[int, int]]
    text: str

    @property
    def num_tokens(self) -> int:
        return len(self.input_ids)


def load_hf_dataset(hf_path: Path | str) -> Dataset:
    """載入 HF dataset。"""
    hf_path = Path(hf_path)
    if not hf_path.exists():
        raise FileNotFoundError(f"HF dataset 不存在: {hf_path}")
    ds = load_from_disk(str(hf_path))
    if isinstance(ds, dict):  # DatasetDict
        if "train" in ds:
            return ds["train"]
        return ds[next(iter(ds))]
    return ds


def load_tokenizer(tokenizer_dir: Path | str) -> PreTrainedTokenizerFast:
    """載入 tokenizer（需為 vocab_size=32007）。"""
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    if len(tok) != 32007:
        raise ValueError(f"tokenizer vocab_size 應為 32007，目前 {len(tok)}")
    return tok


def tokenize_with_offsets(text: str, tok: PreTrainedTokenizerFast) -> TokenizationResult:
    """Tokenize 文本並取得 offset mapping。"""
    enc = tok(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    return TokenizationResult(
        input_ids=enc["input_ids"],
        offset_mapping=enc["offset_mapping"],
        text=text,
    )


def find_structure_tokens(
    text: str,
    tok: PreTrainedTokenizerFast,
    patterns: dict[str, tuple[str, int]] = RE_PATTERNS,
) -> dict[str, list[tuple[int, int]]]:
    """
    在 text 上運行多個 RE 模式，回傳 {pattern_name: [(char_start, char_end), ...]}。
    每個 match 使用 match.start() 與 match.end()。
    """
    result = {}
    for name, (pattern, flags) in patterns.items():
        matches = list(re.finditer(pattern, text, flags=flags))
        result[name] = [(m.start(), m.end()) for m in matches]
    return result


def char_span_to_token_indices(
    char_start: int,
    char_end: int,
    offset_mapping: list[tuple[int, int]],
    policy: str = "union",
) -> list[int]:
    """
    將字元區間 [char_start, char_end) 映射到 token indices。

    policy:
      - "union": 所有與區間重疊的 token（推薦）
      - "first": 僅第一個重疊的 token
    """
    indices = []
    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        # 檢查是否重疊：not (tok_end <= char_start or tok_start >= char_end)
        if not (tok_end <= char_start or tok_start >= char_end):
            indices.append(i)
            if policy == "first":
                break
    return indices


def find_assistant_start_token(
    text: str,
    tok: PreTrainedTokenizerFast,
    offset_mapping: list[tuple[int, int]],
) -> int:
    """
    找出 <|im_start|>assistant\n 在文本中的位置，回傳其後的第一個 token 索引。
    若未找到，回傳 0（表示整個序列都計算 loss）。
    """
    marker = "<|im_start|>assistant\n"
    marker_pos = text.find(marker)
    if marker_pos == -1:
        return 0

    # 計算 marker 後一個位置對應的 token 索引
    marker_end_pos = marker_pos + len(marker)
    for i, (char_start, char_end) in enumerate(offset_mapping):
        if char_start >= marker_end_pos:
            return i
    return len(offset_mapping)


def build_structure_weights(
    dataset: Dataset,
    tok: PreTrainedTokenizerFast,
    patterns: dict[str, tuple[str, int]] = RE_PATTERNS,
    w_struct: float = 3.0,
    w_min: float = 0.25,
    w_max: float = 10.0,
    policy: str = "union",
    sample_indices: Optional[list[int]] = None,
) -> dict:
    """
    對 dataset 中的每筆樣本計算結構權重。

    重點：只對 <|im_start|>assistant\n 之後的 tokens 應用 w_struct 乘數。
    user/system 部分（被 mask）的權重保持 1.0。

    輸出：
    {
        sample_id: {
            "weight": float32 array of shape [T]
            "structure_indices": list of token indices with w > 1
            "pattern_hits": {pattern_name: count}
            "text_length": number of characters
            "token_length": T
            "assistant_start_token": token index where assistant begins
        }
    }
    """
    results = {}

    # 預設全部樣本
    if sample_indices is None:
        sample_indices = list(range(len(dataset)))

    for idx in sample_indices:
        row = dataset[int(idx)]
        text = row.get("text", "")
        sample_id = row.get("id", f"sample_{idx:04d}")

        # Tokenize 取得 offset mapping
        tok_result = tokenize_with_offsets(text, tok)
        T = tok_result.num_tokens

        # 初始化權重為 1.0
        weights = np.ones(T, dtype=np.float32)
        token_patterns = [""] * T  # 記錄每個 token 的 pattern 類型

        # 找出 assistant 部分的開始位置（mask 邊界）
        assistant_start_token = find_assistant_start_token(text, tok, tok_result.offset_mapping)

        # 找出所有結構匹配
        structure_matches = find_structure_tokens(text, tok, patterns)

        # 對每個匹配應用權重（只在 assistant 部分）
        structure_indices = set()
        pattern_hits = {name: len(matches) for name, matches in structure_matches.items()}

        for pattern_name, matches in structure_matches.items():
            for char_start, char_end in matches:
                indices = char_span_to_token_indices(
                    char_start, char_end, tok_result.offset_mapping, policy=policy
                )
                for i in indices:
                    # 只對 assistant 部分應用加權
                    if i >= assistant_start_token:
                        weights[i] *= w_struct
                        structure_indices.add(i)
                    if not token_patterns[i]:  # 若未被其他 pattern 標記
                        token_patterns[i] = pattern_name
                    else:
                        # 多重標記時用 + 連接
                        token_patterns[i] += f"+{pattern_name}"

        # Clamp 權重
        weights = np.clip(weights, w_min, w_max).astype(np.float32)

        results[sample_id] = {
            "weight": weights,
            "structure_indices": sorted(list(structure_indices)),
            "token_patterns": token_patterns,
            "pattern_hits": pattern_hits,
            "text_length": len(text),
            "token_length": T,
            "assistant_start_token": assistant_start_token,  # 新增：assistant 開始位置
        }

    return results


def save_structure_weights(
    results: dict,
    output_dir: Path,
) -> dict:
    """
    儲存 structure_weights 到檔案。產出：
      - structure_weights/{sample_id}.npz
      - structure_weights_metadata.json（全局統計）

    回傳統計資訊。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_samples": len(results),
        "structure_tokens_per_sample": [],
        "weight_stats": {"min": [], "max": [], "mean": [], "std": []},
        "pattern_distribution": {},
    }

    for sample_id, data in results.items():
        # 儲存 npz
        npz_path = output_dir / f"{sample_id}.npz"
        np.savez_compressed(
            npz_path,
            weight=data["weight"],
            structure_indices=np.array(data["structure_indices"], dtype=np.int64),
            token_patterns=np.array(data["token_patterns"], dtype=object),  # 新增
        )

        # 蒐集統計
        w = data["weight"]
        stats["structure_tokens_per_sample"].append(len(data["structure_indices"]))
        stats["weight_stats"]["min"].append(float(np.min(w)))
        stats["weight_stats"]["max"].append(float(np.max(w)))
        stats["weight_stats"]["mean"].append(float(np.mean(w)))
        stats["weight_stats"]["std"].append(float(np.std(w)))

        for pattern_name, count in data["pattern_hits"].items():
            if pattern_name not in stats["pattern_distribution"]:
                stats["pattern_distribution"][pattern_name] = []
            stats["pattern_distribution"][pattern_name].append(count)

    # 聚合統計
    aggregated_stats = {
        "total_samples": stats["total_samples"],
        "structure_tokens_per_sample": {
            "mean": float(np.mean(stats["structure_tokens_per_sample"])),
            "std": float(np.std(stats["structure_tokens_per_sample"])),
            "min": int(np.min(stats["structure_tokens_per_sample"])) if stats["structure_tokens_per_sample"] else 0,
            "max": int(np.max(stats["structure_tokens_per_sample"])) if stats["structure_tokens_per_sample"] else 0,
        },
        "weight_stats": {
            "min": {"mean": float(np.mean(stats["weight_stats"]["min"]))},
            "max": {"mean": float(np.mean(stats["weight_stats"]["max"]))},
            "mean": {"mean": float(np.mean(stats["weight_stats"]["mean"]))},
            "std": {"mean": float(np.mean(stats["weight_stats"]["std"]))},
        },
        "pattern_distribution": {
            name: {
                "total_hits": sum(counts),
                "mean_per_sample": float(np.mean(counts)),
                "samples_with_hits": sum(1 for c in counts if c > 0),
            }
            for name, counts in stats["pattern_distribution"].items()
        },
    }

    metadata_path = output_dir.parent / "structure_weights_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_stats, f, indent=2, ensure_ascii=False)

    return aggregated_stats


def main():
    parser = argparse.ArgumentParser(
        description="Build structure weights from CoT dataset with RE patterns."
    )
    parser.add_argument(
        "--hf-path",
        type=Path,
        default=DEFAULT_HF_PATH,
        help="Path to HF dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="Path to tokenizer (vocab_size=32007) (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "cot" / "reports" / "structure_weights",
        help="Output directory for weights (default: %(default)s)",
    )
    parser.add_argument(
        "--w-struct",
        type=float,
        default=3.0,
        help="Structure weight multiplier (default: %(default)s)",
    )
    parser.add_argument(
        "--policy",
        choices=["union", "first"],
        default="union",
        help="Token mapping policy (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Process only first N samples (default: all)",
    )

    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_hf_dataset(args.hf_path)
    print(f"Dataset size: {len(dataset)}")

    print("Loading tokenizer...")
    tok = load_tokenizer(args.tokenizer_dir)

    sample_indices = None
    if args.sample_count:
        sample_indices = list(range(min(args.sample_count, len(dataset))))

    print(f"Building structure weights (w_struct={args.w_struct}, policy={args.policy})...")
    results = build_structure_weights(
        dataset,
        tok,
        w_struct=args.w_struct,
        policy=args.policy,
        sample_indices=sample_indices,
    )

    print(f"Saving {len(results)} weights to {args.output_dir}...")
    stats = save_structure_weights(results, args.output_dir)

    print("\n=== Structure Weights Summary ===")
    print(json.dumps(stats, indent=2))
    print(f"\nWeights saved to: {args.output_dir}")
    print(f"Metadata saved to: {args.output_dir.parent / 'structure_weights_metadata.json'}")


if __name__ == "__main__":
    main()
