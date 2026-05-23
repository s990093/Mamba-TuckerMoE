#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統計圖表與離線驗證：生成結構 token 分佈、權重統計、長度分析、排程圖等。

輸出：
  - reports/plot_*.png（各種分析圖表）
  - reports/validation_report.json（統計摘要）

使用：
  python validate_and_plot.py \\
    --hf-path sft_cot_bundle/dataset/sft_cot_hf \\
    --weights-dir reports/structure_weights \\
    --seq-len 512
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_PATH = PROJECT_ROOT / "sft_cot_bundle" / "dataset" / "stf_cot_hf"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "cot" / "reports" / "structure_weights"
DEFAULT_TOKENIZER = PROJECT_ROOT / "sft_cot_bundle" / "dataset" / "tokenizer"

# 設定 matplotlib/seaborn
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_hf_dataset(hf_path: Path | str) -> Dataset:
    """載入 HF dataset。"""
    hf_path = Path(hf_path)
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


def load_weights_metadata(weights_dir: Path) -> dict:
    """載入 structure_weights_metadata.json。"""
    metadata_path = weights_dir.parent / "structure_weights_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_all_weights(weights_dir: Path) -> dict:
    """加載所有權重檔。"""
    weights_dir = Path(weights_dir)
    weights = {}
    for npz_file in sorted(weights_dir.glob("*.npz")):
        sample_id = npz_file.stem
        data = np.load(npz_file)
        weights[sample_id] = data["weight"]
    return weights


def analyze_sequence_lengths(
    dataset: Dataset,
    tok: PreTrainedTokenizerFast,
    seq_len: int = 512,
) -> dict:
    """分析所有樣本的 token 長度分佈。"""
    lengths = []
    for row in dataset:
        text = row.get("text", "")
        ids = tok.encode(text, add_special_tokens=False)
        lengths.append(len(ids))

    lengths = np.array(lengths)
    truncated_count = np.sum(lengths > seq_len)

    return {
        "total_samples": len(lengths),
        "length_stats": {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "p25": float(np.percentile(lengths, 25)),
            "p50": float(np.percentile(lengths, 50)),
            "p75": float(np.percentile(lengths, 75)),
            "p95": float(np.percentile(lengths, 95)),
        },
        "truncation_info": {
            "seq_len": seq_len,
            "samples_truncated": int(truncated_count),
            "truncation_rate": float(truncated_count / len(lengths)),
        },
        "length_distribution": lengths,
    }


def plot_structure_token_histogram(
    metadata: dict,
    output_path: Path,
) -> None:
    """繪製每筆樣本的結構 token 數量直方圖。"""
    if "structure_tokens_per_sample" not in metadata:
        print("Warning: structure_tokens_per_sample not in metadata")
        return

    stats = metadata["structure_tokens_per_sample"]
    # 假設在原始 build_structure_weights 中已計算了 per-sample 數據
    # 這裡使用聚合統計作示例；真實應用應有每筆數據

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(
        0.5,
        0.5,
        f"Mean: {stats.get('mean', 0):.1f}\nStd: {stats.get('std', 0):.1f}",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=14,
    )
    ax.set_title("Structure Tokens per Sample (Aggregated Stats)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_pattern_distribution(
    metadata: dict,
    output_path: Path,
) -> None:
    """繪製結構 token 類型分佈圓餅圖。"""
    if "pattern_distribution" not in metadata:
        return

    patterns = metadata["pattern_distribution"]
    labels = list(patterns.keys())
    sizes = [patterns[p].get("total_hits", 0) for p in labels]

    # 過濾掉 0 的模式
    filtered_labels = [l for l, s in zip(labels, sizes) if s > 0]
    filtered_sizes = [s for s in sizes if s > 0]

    if not filtered_sizes:
        print("Warning: No pattern hits to visualize")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(
        filtered_sizes,
        labels=filtered_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("husl", len(filtered_labels)),
    )
    ax.set_title("Pattern Distribution (Token Count)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_length_histogram(
    length_dist: np.ndarray,
    seq_len: int,
    output_path: Path,
) -> None:
    """繪製長度直方圖，標註 SEQ_LEN 邊界。"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(length_dist, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    ax.axvline(seq_len, color="red", linestyle="--", linewidth=2, label=f"SEQ_LEN={seq_len}")
    ax.set_xlabel("Token Length")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Sequence Length Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_weight_distribution(
    weights_dict: dict,
    output_path: Path,
) -> None:
    """繪製權重分佈直方圖。"""
    all_weights = []
    for w in weights_dict.values():
        all_weights.extend(w.flatten())

    all_weights = np.array(all_weights)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 線性刻度
    ax1.hist(all_weights, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    ax1.set_xlabel("Weight Value")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Weight Distribution (Linear)")
    ax1.grid(True, alpha=0.3)

    # 對數刻度
    ax2.hist(all_weights, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    ax2.set_yscale("log")
    ax2.set_xlabel("Weight Value")
    ax2.set_ylabel("Frequency (log)")
    ax2.set_title("Weight Distribution (Log Scale)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_scale_schedule(
    total_steps: int = 10_000,
    eta_max: float = 1.0,
    eta_min: float = 0.3,
    output_path: Path = None,
) -> None:
    """
    預先繪製 SCALe 排程曲線。

    公式：w_think(s) = η_min + 0.5*(η_max - η_min)*(1 + cos(π·s/S))
    """
    steps = np.arange(0, total_steps + 1)
    w_think = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * steps / total_steps))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, w_think, linewidth=2, label="w_think(s)")
    ax.axhline(y=eta_max, color="g", linestyle="--", alpha=0.5, label=f"η_max={eta_max}")
    ax.axhline(y=eta_min, color="r", linestyle="--", alpha=0.5, label=f"η_min={eta_min}")
    ax.fill_between(steps, eta_min, w_think, alpha=0.2, color="blue")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Weight (w_think)")
    ax.set_title("SCALe Schedule: Cosine Annealing for CoT Think Weight")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_weight_statistics(
    weights_dict: dict,
    output_path: Path,
) -> None:
    """繪製權重統計摘要（box plot + violin plot）。"""
    # 計算每個樣本的統計量
    sample_stats = []
    for sample_id, w in weights_dict.items():
        sample_stats.append({
            "mean": np.mean(w),
            "std": np.std(w),
            "min": np.min(w),
            "max": np.max(w),
            "p25": np.percentile(w, 25),
            "p75": np.percentile(w, 75),
        })

    sample_stats = {k: [s[k] for s in sample_stats] for k in sample_stats[0].keys()}

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (stat_name, values) in enumerate(sample_stats.items()):
        ax = axes[idx]
        ax.hist(values, bins=30, alpha=0.7, edgecolor="black", color="steelblue")
        ax.set_xlabel(f"{stat_name.capitalize()} Value")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {stat_name.capitalize()} per Sample")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and plot CoT dataset statistics."
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
        help="Output directory for plots (default: %(default)s)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for truncation analysis (default: %(default)s)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10_000,
        help="Total training steps for SCALe schedule (default: %(default)s)",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = load_hf_dataset(args.hf_path)

    print("Loading tokenizer...")
    tok = load_tokenizer(args.tokenizer_dir)

    print("Loading structure weights metadata...")
    metadata = load_weights_metadata(args.weights_dir)

    print("Loading all weights...")
    weights_dict = load_all_weights(args.weights_dir)

    # 分析序列長度
    print("Analyzing sequence lengths...")
    length_analysis = analyze_sequence_lengths(dataset, tok, seq_len=args.seq_len)

    # 生成圖表
    print("\nGenerating plots...")
    plot_structure_token_histogram(metadata, args.output_dir / "plot_structure_histogram.png")
    plot_pattern_distribution(metadata, args.output_dir / "plot_pattern_pie.png")
    plot_length_histogram(
        length_analysis["length_distribution"],
        args.seq_len,
        args.output_dir / "plot_length_histogram.png",
    )
    plot_weight_distribution(weights_dict, args.output_dir / "plot_weight_distribution.png")
    plot_weight_statistics(weights_dict, args.output_dir / "plot_weight_statistics.png")
    plot_scale_schedule(
        total_steps=args.total_steps,
        output_path=args.output_dir / "plot_scale_schedule.png",
    )

    # 生成驗證報告
    report = {
        "sequence_length_analysis": length_analysis,
        "structure_weights_metadata": metadata,
        "validation_summary": {
            "total_samples": len(dataset),
            "total_weights_files": len(weights_dict),
            "seq_len": args.seq_len,
            "truncation_rate": length_analysis["truncation_info"]["truncation_rate"],
        },
    }

    report_path = args.output_dir / "validation_report.json"
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    report = convert_to_serializable(report)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Validation complete")
    print(f"  Report: {report_path}")
    print(f"  Plots: {args.output_dir / 'plot_*.png'}")

    # 打印摘要
    print("\n=== Summary ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Sequence length (mean ± std): {length_analysis['length_stats']['mean']:.1f} ± {length_analysis['length_stats']['std']:.1f}")
    print(f"Truncation rate: {length_analysis['truncation_info']['truncation_rate']:.2%}")


if __name__ == "__main__":
    main()
