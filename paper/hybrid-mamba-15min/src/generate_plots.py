#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "archive" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_REPORT = ROOT / "archive/profiling/ncu_moe_scan_report.ncu-rep"
DEFAULT_SUMMARY = ROOT / "assets/data/ncu_profile_summary.json"
DEFAULT_PLOT = ROOT / "assets/plots/profiling_latency.png"
NSIGHT_PYTHON_DIR = Path("/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/python")

TIME_METRIC = "gpu__time_duration.sum"
READ_BYTES_METRIC = "dram__bytes_read.sum"
WRITE_BYTES_METRIC = "dram__bytes_write.sum"
THROUGHPUT_METRICS = {
    "dram_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
}
MEAN_METRICS = {
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "eligible_warps": "smsp__warps_eligible.avg.per_cycle_active",
    "stall_mio": "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "stall_long": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "stall_short": "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "stall_not_selected": "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
}
LAUNCH_METRICS = {
    "thread_count": "launch__thread_count",
    "grid_size": "launch__grid_size",
    "registers_per_thread": "launch__registers_per_thread_allocated",
    "shared_mem_per_block": "launch__shared_mem_per_block_allocated",
}
DEVICE_METRICS = {
    "display_name": "device__attribute_display_name",
    "compute_capability_major": "device__attribute_compute_capability_major",
    "compute_capability_minor": "device__attribute_compute_capability_minor",
    "sm_count": "device__attribute_multiprocessor_count",
    "memory_bus_width_bits": "device__attribute_global_memory_bus_width",
    "max_mem_frequency_khz": "device__attribute_max_mem_frequency_khz",
    "max_gpu_frequency_khz": "device__attribute_max_gpu_frequency_khz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a real Nsight Compute summary plot from a .ncu-rep archive."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Path to the .ncu-rep file.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to the extracted NCU summary JSON.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=DEFAULT_PLOT,
        help="Path to the output PNG plot.",
    )
    parser.add_argument(
        "--extract-ncu-summary",
        action="store_true",
        help="Only extract the raw Nsight Compute summary JSON.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def metric_value(action, name: str, default=0.0):
    metric = action.metric_by_name(name)
    if metric is None or not metric.has_value():
        return default
    try:
        return float(metric.as_double())
    except Exception:
        pass
    try:
        return float(metric.as_uint64())
    except Exception:
        pass
    try:
        return metric.as_string()
    except Exception:
        return default


def normalize_rule(rule: dict) -> dict:
    message = rule.get("rule_message") or {}
    return {
        "rule_identifier": rule.get("rule_identifier", ""),
        "name": rule.get("name", ""),
        "section_identifier": rule.get("section_identifier", ""),
        "title": message.get("title", ""),
        "message": message.get("message", ""),
    }


def weighted_average(records: list[dict], field: str, weight_field: str = "duration_ns") -> float:
    numer = 0.0
    denom = 0.0
    for record in records:
        weight = float(record.get(weight_field, 0.0))
        value = float(record.get(field, 0.0))
        numer += weight * value
        denom += weight
    return numer / denom if denom else 0.0


def arithmetic_average(records: list[dict], field: str) -> float:
    values = [float(record.get(field, 0.0)) for record in records]
    return sum(values) / len(values) if values else 0.0


def extract_summary(report_path: Path) -> dict:
    if not NSIGHT_PYTHON_DIR.exists():
        raise FileNotFoundError(
            f"Nsight Compute Python bindings not found at {NSIGHT_PYTHON_DIR}"
        )

    sys.path.insert(0, str(NSIGHT_PYTHON_DIR))
    import ncu_report  # type: ignore

    context = ncu_report.load_report(str(report_path))
    num_ranges = context.num_ranges()
    if num_ranges == 0:
        raise RuntimeError(f"No ranges found in {report_path}")

    actions: list[dict] = []
    device_info: dict[str, object] | None = None

    for range_idx in range(num_ranges):
        trace_range = context.range_by_idx(range_idx)
        for action_idx in range(trace_range.num_actions()):
            action = trace_range.action_by_idx(action_idx)
            record = {
                "name": action.name(),
                "duration_ns": float(metric_value(action, TIME_METRIC, 0.0)),
                "read_bytes": float(metric_value(action, READ_BYTES_METRIC, 0.0)),
                "write_bytes": float(metric_value(action, WRITE_BYTES_METRIC, 0.0)),
                "rules": [normalize_rule(rule) for rule in action.rule_results_as_dicts()],
            }
            for field, metric_name in THROUGHPUT_METRICS.items():
                record[field] = float(metric_value(action, metric_name, 0.0))
            for field, metric_name in MEAN_METRICS.items():
                record[field] = float(metric_value(action, metric_name, 0.0))
            for field, metric_name in LAUNCH_METRICS.items():
                record[field] = float(metric_value(action, metric_name, 0.0))
            actions.append(record)

            if device_info is None:
                device_info = {
                    field: metric_value(action, metric_name, "")
                    for field, metric_name in DEVICE_METRICS.items()
                }

    if not actions:
        raise RuntimeError(f"No kernel actions found in {report_path}")

    total_duration_ns = sum(record["duration_ns"] for record in actions)
    total_read_bytes = sum(record["read_bytes"] for record in actions)
    total_write_bytes = sum(record["write_bytes"] for record in actions)
    total_traffic_bytes = total_read_bytes + total_write_bytes
    effective_bandwidth_gbps = (
        total_traffic_bytes / total_duration_ns if total_duration_ns else 0.0
    )

    kernels: dict[str, list[dict]] = defaultdict(list)
    for record in actions:
        kernels[record["name"]].append(record)

    kernel_summaries = []
    for kernel_name, records in sorted(
        kernels.items(),
        key=lambda item: sum(record["duration_ns"] for record in item[1]),
        reverse=True,
    ):
        duration_ns = sum(record["duration_ns"] for record in records)
        read_bytes = sum(record["read_bytes"] for record in records)
        write_bytes = sum(record["write_bytes"] for record in records)
        traffic_bytes = read_bytes + write_bytes
        launch_groups: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
        for record in records:
            key = (
                int(record["thread_count"]),
                int(record["grid_size"]),
                int(record["registers_per_thread"]),
                int(record["shared_mem_per_block"]),
            )
            launch_groups[key].append(record)

        launch_variants = []
        for key, variant_records in launch_groups.items():
            launch_variants.append(
                {
                    "thread_count": key[0],
                    "grid_size": key[1],
                    "registers_per_thread": key[2],
                    "shared_mem_per_block": key[3],
                    "count": len(variant_records),
                    "mean_duration_ns": arithmetic_average(variant_records, "duration_ns"),
                    "mean_dram_pct": arithmetic_average(variant_records, "dram_pct"),
                    "mean_sm_pct": arithmetic_average(variant_records, "sm_pct"),
                    "mean_occupancy_pct": arithmetic_average(variant_records, "occupancy_pct"),
                    "mean_eligible_warps": arithmetic_average(variant_records, "eligible_warps"),
                }
            )
        launch_variants.sort(
            key=lambda item: (item["count"], item["mean_duration_ns"]),
            reverse=True,
        )

        representative_rules: list[dict] = []
        seen_rule_ids: set[str] = set()
        for record in records:
            for rule in record["rules"]:
                rule_id = rule["rule_identifier"]
                if rule_id in seen_rule_ids:
                    continue
                seen_rule_ids.add(rule_id)
                representative_rules.append(rule)
            if len(representative_rules) >= 8:
                break

        kernel_summaries.append(
            {
                "name": kernel_name,
                "launches": len(records),
                "time_ms": duration_ns / 1e6,
                "time_share_pct": 100.0 * duration_ns / total_duration_ns,
                "read_mb": read_bytes / (1024 * 1024),
                "write_mb": write_bytes / (1024 * 1024),
                "traffic_mb": traffic_bytes / (1024 * 1024),
                "effective_bandwidth_gbps": traffic_bytes / duration_ns if duration_ns else 0.0,
                "weighted_dram_pct": weighted_average(records, "dram_pct"),
                "weighted_memory_pct": weighted_average(records, "memory_pct"),
                "weighted_sm_pct": weighted_average(records, "sm_pct"),
                "mean_occupancy_pct": arithmetic_average(records, "occupancy_pct"),
                "mean_eligible_warps": arithmetic_average(records, "eligible_warps"),
                "mean_stall_mio": arithmetic_average(records, "stall_mio"),
                "mean_stall_long": arithmetic_average(records, "stall_long"),
                "mean_stall_short": arithmetic_average(records, "stall_short"),
                "mean_stall_not_selected": arithmetic_average(records, "stall_not_selected"),
                "launch_variants": launch_variants,
                "representative_rules": representative_rules,
            }
        )

    summary = {
        "report_path": str(report_path),
        "num_ranges": num_ranges,
        "num_actions": len(actions),
        "device": device_info or {},
        "overall": {
            "total_kernel_time_ms": total_duration_ns / 1e6,
            "total_read_mb": total_read_bytes / (1024 * 1024),
            "total_write_mb": total_write_bytes / (1024 * 1024),
            "total_traffic_mb": total_traffic_bytes / (1024 * 1024),
            "effective_bandwidth_gbps": effective_bandwidth_gbps,
            "weighted_dram_pct": weighted_average(actions, "dram_pct"),
            "weighted_memory_pct": weighted_average(actions, "memory_pct"),
            "weighted_sm_pct": weighted_average(actions, "sm_pct"),
        },
        "kernels": kernel_summaries,
    }
    return summary


def extract_summary_via_arm64(report_path: Path, summary_json: Path) -> dict:
    ensure_parent(summary_json)
    command = [
        "arch",
        "-arm64",
        "/usr/bin/python3",
        str(Path(__file__).resolve()),
        "--extract-ncu-summary",
        "--report",
        str(report_path),
        "--summary-json",
        str(summary_json),
    ]
    subprocess.run(command, check=True)
    return json.loads(summary_json.read_text(encoding="utf-8"))


def ensure_summary(report_path: Path, summary_json: Path) -> dict:
    try:
        return extract_summary(report_path)
    except Exception as exc:
        if platform.system() == "Darwin" and platform.machine() != "arm64":
            print(
                f"[generate_plots] Local Nsight import failed under {platform.machine()}; "
                "retrying summary extraction with arm64 /usr/bin/python3.",
                file=sys.stderr,
            )
            return extract_summary_via_arm64(report_path, summary_json)
        raise exc


def write_summary(summary: dict, summary_json: Path) -> None:
    ensure_parent(summary_json)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Publication-style NCU figure: pure white background, high-contrast neutrals, colorblind-safe accents.
_NCU_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica Neue",
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
        "sans-serif",
    ],
    "font.size": 10,
    "axes.edgecolor": "#2a2a2a",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#1a1a1a",
    "axes.titleweight": "600",
    "axes.titlesize": 11.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#d8d8d8",
    "grid.linestyle": "--",
    "grid.linewidth": 0.65,
    "grid.alpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": False,
    "figure.titlesize": 14,
}


def _style_axes_frame(ax) -> None:
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_profiling_figure(summary: dict, output_plot: Path) -> None:
    ensure_parent(output_plot)
    kernels = summary["kernels"]
    klabels = [kernel["name"] for kernel in kernels]
    y = np.arange(len(klabels))

    time_bar_colors = ["#1f4e79", "#c25e2a"]
    with plt.rc_context(_NCU_RC):
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))
        fig.patch.set_facecolor("#ffffff")
        for ax in axes.flat:
            ax.set_facecolor("#ffffff")

        # Top-left: time share.
        ax = axes[0, 0]
        time_values = [kernel["time_ms"] for kernel in kernels]
        bars = ax.barh(
            y,
            time_values,
            color=time_bar_colors[: len(klabels)],
            alpha=0.92,
            edgecolor="#1a1a1a",
            linewidth=0.35,
        )
        ax.set_title("Kernel Time Share", pad=8)
        ax.set_xlabel("Time (ms)")
        ax.set_yticks(y, klabels)
        ax.invert_yaxis()
        _style_axes_frame(ax)
        ax.grid(axis="x", which="major")
        txmax = max(time_values) if time_values else 1.0
        for idx, (bar, kernel) in enumerate(zip(bars, kernels)):
            label = f"{kernel['time_share_pct']:.1f}% | {kernel['launches']} launches"
            ax.text(
                bar.get_width() + txmax * 0.02,
                idx,
                label,
                va="center",
                fontsize=9,
                color="#222222",
            )

        # Top-right: weighted throughput.
        ax = axes[0, 1]
        w = 0.22
        x = np.arange(len(klabels))
        dram = [kernel["weighted_dram_pct"] for kernel in kernels]
        memory = [kernel["weighted_memory_pct"] for kernel in kernels]
        sm = [kernel["weighted_sm_pct"] for kernel in kernels]
        ax.bar(
            x - w,
            dram,
            width=w,
            label="DRAM",
            color="#b2182b",
            edgecolor="#1a1a1a",
            linewidth=0.3,
        )
        ax.bar(
            x,
            memory,
            width=w,
            label="Compute-Memory",
            color="#1b7837",
            edgecolor="#1a1a1a",
            linewidth=0.3,
        )
        ax.bar(
            x + w,
            sm,
            width=w,
            label="SM",
            color="#2166ac",
            edgecolor="#1a1a1a",
            linewidth=0.3,
        )
        ax.set_title("Weighted Throughput (% of peak)", pad=8)
        ax.set_ylabel("% peak sustained elapsed")
        ax.set_xticks(x, klabels, rotation=10, ha="right")
        ax.set_ylim(0, max(max(dram), max(memory), max(sm)) * 1.22)
        _style_axes_frame(ax)
        ax.yaxis.grid(True)
        leg = ax.legend(
            frameon=True,
            fontsize=8.5,
            loc="upper right",
            facecolor="white",
            framealpha=1.0,
        )
        if leg is not None:
            leg.get_frame().set_linewidth(0.5)

        # Bottom-left: stall breakdown.
        ax = axes[1, 0]
        stall_spec = [
            ("mean_stall_mio", "MIO", "#2166ac"),
            ("mean_stall_long", "Long SB", "#d95f02"),
            ("mean_stall_short", "Short SB", "#1a9850"),
            ("mean_stall_not_selected", "Not Sel.", "#762a83"),
        ]
        sw = 0.18
        for offset, (field, slabel, c) in zip(
            np.linspace(-0.27, 0.27, len(stall_spec)), stall_spec
        ):
            values = [kernel[field] for kernel in kernels]
            ax.bar(
                x + offset,
                values,
                width=sw,
                label=slabel,
                color=c,
                edgecolor="#1a1a1a",
                linewidth=0.25,
            )
        ax.set_title("Average Warp Stall Reasons", pad=8)
        ax.set_ylabel("Stall ratio per issued instruction")
        ax.set_xticks(x, klabels, rotation=10, ha="right")
        _style_axes_frame(ax)
        ax.yaxis.grid(True)
        leg2 = ax.legend(
            frameon=True,
            fontsize=8.5,
            ncols=2,
            loc="upper right",
            facecolor="white",
            framealpha=1.0,
        )
        if leg2 is not None:
            leg2.get_frame().set_linewidth(0.5)

        # Bottom-right: occupancy + eligible warps.
        ax = axes[1, 1]
        occupancy = [kernel["mean_occupancy_pct"] for kernel in kernels]
        eligible = [kernel["mean_eligible_warps"] for kernel in kernels]
        ax.bar(
            x,
            occupancy,
            color="#5e3c99",
            alpha=0.92,
            width=0.5,
            label="Achieved occupancy",
            edgecolor="#1a1a1a",
            linewidth=0.35,
        )
        ax.set_title("Occupancy and Scheduler Eligibility", pad=8)
        ax.set_ylabel("Occupancy (% peak active)")
        ax.set_xticks(x, klabels, rotation=10, ha="right")
        _style_axes_frame(ax)
        ax.yaxis.grid(True)
        ax2 = ax.twinx()
        ax2.set_facecolor("#ffffff")
        ax2.plot(
            x,
            eligible,
            color="#1a1a1a",
            marker="o",
            markersize=5,
            markeredgewidth=0.4,
            markeredgecolor="white",
            linewidth=1.65,
            label="Eligible warps/scheduler",
        )
        ax2.set_ylabel("Eligible warps / scheduler")
        ax2.spines["top"].set_visible(False)
        lines, labels_l = ax.get_legend_handles_labels()
        lines2, labels_r = ax2.get_legend_handles_labels()
        leg3 = ax2.legend(
            lines + lines2,
            labels_l + labels_r,
            frameon=True,
            fontsize=8.5,
            loc="upper right",
            facecolor="white",
            framealpha=1.0,
        )
        if leg3 is not None:
            leg3.get_frame().set_linewidth(0.5)

        overall = summary["overall"]
        device = summary["device"]
        caption = (
            f"Source: {Path(summary['report_path']).name}  ·  {device.get('display_name', 'Unknown GPU')}  ·  "
            f"{summary['num_actions']} launches, {overall['total_kernel_time_ms']:.2f} ms total, "
            f"{overall['total_traffic_mb'] / 1024:.2f} GB traffic, "
            f"{overall['effective_bandwidth_gbps']:.1f} GB/s effective bandwidth"
        )
        fig.suptitle("Nsight Compute Summary (parsed from .ncu-rep)", fontsize=14, weight="600", y=0.99)
        fig.text(0.5, 0.016, caption, ha="center", fontsize=9, color="#333333")
        fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.1, wspace=0.28, hspace=0.42)
        fig.savefig(
            output_plot,
            dpi=240,
            bbox_inches="tight",
            facecolor="#ffffff",
            edgecolor="none",
            pad_inches=0.12,
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.extract_ncu_summary:
        summary = extract_summary(args.report)
        write_summary(summary, args.summary_json)
        return

    summary = ensure_summary(args.report, args.summary_json)
    write_summary(summary, args.summary_json)
    draw_profiling_figure(summary, args.output_plot)
    print(f"[generate_plots] Wrote summary: {args.summary_json}")
    print(f"[generate_plots] Wrote plot: {args.output_plot}")


if __name__ == "__main__":
    main()
