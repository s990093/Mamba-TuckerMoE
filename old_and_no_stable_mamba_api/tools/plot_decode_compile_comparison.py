#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run decode throughput experiment and plot:
1) eager (graph breaks)
2) throughput compile (full per-layer graph compile)

Output:
  - JSON summary with raw runs and aggregate stats
  - PNG bar chart with mean decode tok/s and std
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DECODE_RE = re.compile(r"Decode:\s+tokens=\d+\s+time=[\d.]+\s+ms\s+\(([\d.]+)\s+tok/s\)")


@dataclass
class ModeResult:
    name: str
    runs: list[float]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.runs)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.runs) if len(self.runs) > 1 else 0.0


def _parse_decode_tps(stdout: str) -> float:
    m = DECODE_RE.search(stdout)
    if not m:
        raise RuntimeError("Failed to parse decode tok/s from benchmark output.")
    return float(m.group(1))


def _run_one(cmd: list[str], cwd: Path) -> float:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Benchmark command failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return _parse_decode_tps(proc.stdout)


def _run_mode(
    *,
    repo_root: Path,
    repeats: int,
    base_cmd: list[str],
    inference_type: str,
) -> ModeResult:
    runs: list[float] = []
    for _ in range(repeats):
        cmd = base_cmd + ["--inference-type", inference_type]
        runs.append(_run_one(cmd, repo_root))
    return ModeResult(name=inference_type, runs=runs)


def _plot(results: list[ModeResult], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [r.name for r in results]
    means = [r.mean for r in results]
    errs = [r.stdev for r in results]
    colors = ["#d62728", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    bars = ax.bar(labels, means, yerr=errs, capsize=8, color=colors[: len(results)], alpha=0.9)

    ax.set_title("Decode Throughput: Graph Breaks vs Full Graph Compile")
    ax.set_ylabel("Decode Throughput (tok/s)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.5, 0.02 * max(means)),
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if len(results) == 2 and means[0] > 0:
        uplift = (means[1] / means[0] - 1.0) * 100.0
        ax.text(
            0.02,
            0.95,
            f"Compile uplift: {uplift:+.1f}%",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f0f0f0", "alpha": 0.9, "edgecolor": "#bbbbbb"},
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare eager (graph breaks) vs throughput compile decode speed and plot results."
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path passed to benchmark_mlx.py")
    parser.add_argument("--tokenizer", type=str, default="inference/tokenizer")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--kv-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8])
    parser.add_argument(
        "--output-json", type=str, default="inference/results/bench_decode_compile_comparison.json"
    )
    parser.add_argument("--output-fig", type=str, default="inference/results/bench_decode_compile_comparison.png")
    parser.add_argument(
        "--extra-benchmark-args",
        type=str,
        default="",
        help="Extra raw args appended to benchmark command, e.g. '--fast-sample --no-penalties'",
    )
    args = parser.parse_args()

    infer_root = Path(__file__).resolve().parent.parent
    repo_root = infer_root.parent
    benchmark_path = infer_root / "benchmark_mlx.py"
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Cannot find benchmark script: {benchmark_path}")

    base_cmd = [
        sys.executable,
        str(benchmark_path),
        "--tokenizer",
        args.tokenizer,
        "--seq-len",
        str(args.seq_len),
        "--decode-tokens",
        str(args.decode_tokens),
        "--warmup",
        str(args.warmup),
        "--dtype",
        args.dtype,
        "--kv-dtype",
        args.kv_dtype,
        "--quantize",
        str(args.quantize),
        "--no-show-io",
    ]
    if args.checkpoint:
        base_cmd.extend(["--checkpoint", args.checkpoint])
    if args.extra_benchmark_args.strip():
        base_cmd.extend(args.extra_benchmark_args.strip().split())

    eager_res = _run_mode(
        repo_root=repo_root,
        repeats=args.repeats,
        base_cmd=base_cmd,
        inference_type="eager",
    )
    throughput_res = _run_mode(
        repo_root=repo_root,
        repeats=args.repeats,
        base_cmd=base_cmd,
        inference_type="throughput",
    )
    results = [eager_res, throughput_res]

    payload = {
        "settings": {
            "checkpoint": args.checkpoint,
            "tokenizer": args.tokenizer,
            "seq_len": args.seq_len,
            "decode_tokens": args.decode_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "dtype": args.dtype,
            "kv_dtype": args.kv_dtype,
            "quantize": args.quantize,
            "extra_benchmark_args": args.extra_benchmark_args,
        },
        "results": [
            {"mode": r.name, "runs_tok_s": r.runs, "mean_tok_s": r.mean, "std_tok_s": r.stdev}
            for r in results
        ],
    }
    if eager_res.mean > 0:
        payload["compile_uplift_percent"] = (throughput_res.mean / eager_res.mean - 1.0) * 100.0

    output_json = (repo_root / args.output_json).resolve()
    output_fig = (repo_root / args.output_fig).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(results, output_fig)

    print(f"Wrote summary JSON: {output_json}")
    print(f"Wrote figure PNG:  {output_fig}")
    if eager_res.mean > 0:
        uplift = (throughput_res.mean / eager_res.mean - 1.0) * 100.0
        print(
            f"Decode mean tok/s | eager={eager_res.mean:.2f}, throughput={throughput_res.mean:.2f}, "
            f"uplift={uplift:+.2f}%"
        )


if __name__ == "__main__":
    main()
