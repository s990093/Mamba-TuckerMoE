#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "archive" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHECKPOINT = ROOT.parent.parent / "checkpoint.npz"
SUMMARY_JSON = ROOT / "assets/data/checkpoint_compression_summary.json"
PLOT_PATH = ROOT / "assets/plots/checkpoint_compression_study.png"

FRONTIER_RATIOS = [0.80, 0.8287, 0.85, 0.90, 0.93, 0.95]
MODE1_CANDIDATES = list(range(1, 9))
MODE2_CANDIDATES = list(range(16, 257, 16))
MODE3_CANDIDATES = list(range(32, 513, 32))
SVD_RANK_CANDIDATES = list(range(1, 257))
RANK_SWEEP_MODE1 = list(range(1, 9))
RANK_SWEEP_MODE2 = [32, 64, 96, 128, 160, 192, 224, 256]
RANK_SWEEP_MODE3 = [64, 128, 192, 256, 320, 384, 448, 512]
REPRESENTATIVE_LABELS = {
    (768, 768): "Mamba out projection",
    (1536, 6144): "Mamba x_up projection",
    (768, 4608): "Transformer FFN up/gate projection",
    (4608, 768): "Transformer FFN down projection",
}


@dataclass
class ModuleSummary:
    name: str
    label: str
    d_in: int
    d_out: int
    num_experts: int
    dense_params: int
    deployed_tucker_params: int
    family_count: int
    energy_sq: float
    tucker_frontier_best: dict[float, dict[str, float]]
    svd_frontier_best: dict[float, dict[str, float]]
    rank_sweep_mode1: dict[int, float]
    rank_sweep_mode2: dict[int, float]
    rank_sweep_mode3: dict[int, float]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def mode_dot(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    moved = np.moveaxis(tensor, mode, 0)
    result = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(result, 0, mode)


def orthogonalize_module(
    u_expert: np.ndarray,
    u_in: np.ndarray,
    u_out_t: np.ndarray,
    core: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Wide expert factors use a compact SVD so the expert mode becomes orthonormal.
    q1, s1, vh1 = np.linalg.svd(u_expert, full_matrices=False)
    r1 = (s1[:, None] * vh1).astype(np.float32, copy=False)
    core1 = mode_dot(core, r1, mode=0)

    q2, r2 = np.linalg.qr(u_in, mode="reduced")
    core2 = mode_dot(core1, r2.astype(np.float32, copy=False), mode=1)

    q3, r3 = np.linalg.qr(u_out_t, mode="reduced")
    core3 = mode_dot(core2, r3.astype(np.float32, copy=False), mode=2)
    return np.asarray(q1, dtype=np.float32), np.asarray(q2, dtype=np.float32), np.asarray(q3, dtype=np.float32), np.asarray(core3, dtype=np.float32)


def leading_mode_vectors(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def top_left_vectors(unfolded: np.ndarray) -> np.ndarray:
        gram = unfolded @ unfolded.T
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        return np.asarray(eigvecs[:, order], dtype=np.float32)

    u1 = top_left_vectors(tensor.reshape(tensor.shape[0], -1))
    u2 = top_left_vectors(np.moveaxis(tensor, 1, 0).reshape(tensor.shape[1], -1))
    u3 = top_left_vectors(np.moveaxis(tensor, 2, 0).reshape(tensor.shape[2], -1))
    return u1, u2, u3


def build_prefix_sums(coeff: np.ndarray) -> np.ndarray:
    sq = np.square(coeff, dtype=np.float64)
    return sq.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)


def retained_energy_from_prefix(prefix: np.ndarray, a: int, b: int, c: int) -> float:
    return float(prefix[a - 1, b - 1, c - 1])


def collect_module_summaries(checkpoint_path: Path) -> tuple[list[ModuleSummary], dict[str, dict[str, float]], int, int]:
    arrays = np.load(checkpoint_path, mmap_mode="r")
    chosen_prefixes: dict[tuple[int, int], str] = {}
    shape_breakdown: dict[str, dict[str, float]] = {}
    dense_total = 0
    deployed_total = 0

    for key in sorted(arrays.files):
        if not key.endswith("U_expert"):
            continue
        prefix = key[: -len("U_expert")]
        u_in_key = prefix + "U_in.weight"
        u_out_key = prefix + "U_out.weight"
        core_key = prefix + "core"
        router_key = prefix + "router.weight"
        bias_key = prefix + "bias"
        if not all(name in arrays for name in [u_in_key, u_out_key, core_key, router_key, bias_key]):
            continue

        u_expert = np.asarray(arrays[key], dtype=np.float32)
        u_in = np.asarray(arrays[u_in_key], dtype=np.float32).T
        u_out_t = np.asarray(arrays[u_out_key], dtype=np.float32)
        core = np.asarray(arrays[core_key], dtype=np.float32)
        dense_params = int(u_expert.shape[0] * u_in.shape[0] * u_out_t.shape[0])
        deployed_tucker_params = int(
            u_expert.size
            + arrays[u_in_key].size
            + arrays[u_out_key].size
            + arrays[core_key].size
            + arrays[router_key].size
            + arrays[bias_key].size
        )
        dense_total += dense_params
        deployed_total += deployed_tucker_params

        shape_key = (u_in.shape[0], u_out_t.shape[0])
        shape_name = f"{shape_key[0]}x{shape_key[1]}"
        entry = shape_breakdown.setdefault(
            shape_name,
            {
                "count": 0,
                "dense_params": 0.0,
                "deployed_tucker_params": 0.0,
                "label": REPRESENTATIVE_LABELS.get(shape_key, shape_name),
            },
        )
        entry["count"] += 1
        entry["dense_params"] += dense_params
        entry["deployed_tucker_params"] += deployed_tucker_params

        if shape_key not in REPRESENTATIVE_LABELS or shape_key in chosen_prefixes:
            continue
        chosen_prefixes[shape_key] = prefix

    modules: list[ModuleSummary] = []
    for shape_key, prefix in sorted(chosen_prefixes.items()):
        key = prefix + "U_expert"
        u_in_key = prefix + "U_in.weight"
        u_out_key = prefix + "U_out.weight"
        core_key = prefix + "core"
        router_key = prefix + "router.weight"
        bias_key = prefix + "bias"

        u_expert = np.asarray(arrays[key], dtype=np.float32)
        u_in = np.asarray(arrays[u_in_key], dtype=np.float32).T
        u_out_t = np.asarray(arrays[u_out_key], dtype=np.float32)
        core = np.asarray(arrays[core_key], dtype=np.float32)
        q1, _, _, orth_core = orthogonalize_module(u_expert, u_in, u_out_t, core)
        a1, a2, a3 = leading_mode_vectors(orth_core)
        coeff = mode_dot(mode_dot(mode_dot(orth_core, a1.T, 0), a2.T, 1), a3.T, 2)
        prefix_energy = build_prefix_sums(coeff)
        total_energy = float(np.square(coeff, dtype=np.float64).sum())

        dense_params = int(u_expert.shape[0] * u_in.shape[0] * u_out_t.shape[0])
        deployed_tucker_params = int(
            u_expert.size
            + arrays[u_in_key].size
            + arrays[u_out_key].size
            + arrays[core_key].size
            + arrays[router_key].size
            + arrays[bias_key].size
        )
        print(f"Analyzing representative module: {prefix.rstrip('.')} ({shape_key[0]}x{shape_key[1]})", flush=True)
        tucker_frontier_best: dict[float, dict[str, float]] = {}
        for ratio in FRONTIER_RATIOS:
            budget = (1.0 - ratio) * dense_params
            best = {"r1": 1.0, "r2": 16.0, "r3": 32.0, "params": 0.0, "retained_energy_sq": 0.0}
            for r1 in MODE1_CANDIDATES:
                for r2 in MODE2_CANDIDATES:
                    for r3 in MODE3_CANDIDATES:
                        params = (
                            u_expert.shape[0] * r1
                            + u_in.shape[0] * r2
                            + u_out_t.shape[0] * r3
                            + r1 * r2 * r3
                        )
                        if params > budget:
                            continue
                        retained = retained_energy_from_prefix(prefix_energy, r1, r2, r3)
                        if retained > best["retained_energy_sq"]:
                            best = {
                                "r1": float(r1),
                                "r2": float(r2),
                                "r3": float(r3),
                                "params": float(params),
                                "retained_energy_sq": retained,
                            }
            tucker_frontier_best[ratio] = best

        middle = np.tensordot(q1, orth_core, axes=(1, 0))
        per_expert_svals_sq = []
        for expert_idx in range(middle.shape[0]):
            singular_values = np.linalg.svd(middle[expert_idx], compute_uv=False)
            per_expert_svals_sq.append(np.square(singular_values.astype(np.float64)))
        per_expert_svals_sq = np.stack(per_expert_svals_sq, axis=0)
        cumulative_svd = np.cumsum(per_expert_svals_sq, axis=1)

        svd_frontier_best: dict[float, dict[str, float]] = {}
        for ratio in FRONTIER_RATIOS:
            budget = (1.0 - ratio) * dense_params
            best = {"rank": 1.0, "params": 0.0, "retained_energy_sq": 0.0}
            for rank in SVD_RANK_CANDIDATES:
                params = u_expert.shape[0] * rank * (u_in.shape[0] + u_out_t.shape[0])
                if params > budget:
                    continue
                retained = float(cumulative_svd[:, rank - 1].sum())
                if retained > best["retained_energy_sq"]:
                    best = {"rank": float(rank), "params": float(params), "retained_energy_sq": retained}
            svd_frontier_best[ratio] = best

        rank_sweep_mode1 = {
            r1: retained_energy_from_prefix(prefix_energy, r1, orth_core.shape[1], orth_core.shape[2]) / total_energy
            for r1 in RANK_SWEEP_MODE1
        }
        rank_sweep_mode2 = {
            r2: retained_energy_from_prefix(prefix_energy, orth_core.shape[0], r2, orth_core.shape[2]) / total_energy
            for r2 in RANK_SWEEP_MODE2
        }
        rank_sweep_mode3 = {
            r3: retained_energy_from_prefix(prefix_energy, orth_core.shape[0], orth_core.shape[1], r3) / total_energy
            for r3 in RANK_SWEEP_MODE3
        }

        modules.append(
            ModuleSummary(
                name=prefix.rstrip("."),
                label=REPRESENTATIVE_LABELS[shape_key],
                d_in=u_in.shape[0],
                d_out=u_out_t.shape[0],
                num_experts=u_expert.shape[0],
                dense_params=dense_params,
                deployed_tucker_params=deployed_tucker_params,
                family_count=int(shape_breakdown[f"{shape_key[0]}x{shape_key[1]}"]["count"]),
                energy_sq=total_energy,
                tucker_frontier_best=tucker_frontier_best,
                svd_frontier_best=svd_frontier_best,
                rank_sweep_mode1=rank_sweep_mode1,
                rank_sweep_mode2=rank_sweep_mode2,
                rank_sweep_mode3=rank_sweep_mode3,
            )
        )

    for entry in shape_breakdown.values():
        entry["compression_ratio"] = 1.0 - entry["deployed_tucker_params"] / entry["dense_params"]

    return modules, shape_breakdown, dense_total, deployed_total


def aggregate_results(
    modules: list[ModuleSummary],
    shape_breakdown: dict[str, dict[str, float]],
    dense_total: int,
    deployed_total: int,
) -> dict:
    total_energy = sum(module.energy_sq * module.family_count for module in modules)

    frontier = []
    for ratio in FRONTIER_RATIOS:
        tucker_kept = sum(module.tucker_frontier_best[ratio]["retained_energy_sq"] * module.family_count for module in modules)
        svd_kept = sum(module.svd_frontier_best[ratio]["retained_energy_sq"] * module.family_count for module in modules)
        frontier.append(
            {
                "compression_ratio": ratio,
                "tucker_relative_error": math.sqrt(max(0.0, 1.0 - tucker_kept / total_energy)),
                "svd_relative_error": math.sqrt(max(0.0, 1.0 - svd_kept / total_energy)),
                "tucker_energy_retention": tucker_kept / total_energy,
                "svd_energy_retention": svd_kept / total_energy,
            }
        )

    rank_sensitivity = {"expert_mode": [], "input_mode": [], "output_mode": []}
    for r1 in RANK_SWEEP_MODE1:
        kept = sum(module.rank_sweep_mode1[r1] * module.energy_sq * module.family_count for module in modules)
        rank_sensitivity["expert_mode"].append(
            {"rank": r1, "energy_retention": kept / total_energy, "relative_error": math.sqrt(max(0.0, 1.0 - kept / total_energy))}
        )
    for r2 in RANK_SWEEP_MODE2:
        kept = sum(module.rank_sweep_mode2[r2] * module.energy_sq * module.family_count for module in modules)
        rank_sensitivity["input_mode"].append(
            {"rank": r2, "energy_retention": kept / total_energy, "relative_error": math.sqrt(max(0.0, 1.0 - kept / total_energy))}
        )
    for r3 in RANK_SWEEP_MODE3:
        kept = sum(module.rank_sweep_mode3[r3] * module.energy_sq * module.family_count for module in modules)
        rank_sensitivity["output_mode"].append(
            {"rank": r3, "energy_retention": kept / total_energy, "relative_error": math.sqrt(max(0.0, 1.0 - kept / total_energy))}
        )

    return {
        "checkpoint": str(CHECKPOINT),
        "num_representative_modules": len(modules),
        "weighted_module_count": sum(module.family_count for module in modules),
        "dense_total_params": dense_total,
        "deployed_tucker_total_params": deployed_total,
        "deployed_compression_ratio": 1.0 - deployed_total / dense_total,
        "shape_breakdown": shape_breakdown,
        "representative_modules": [
            {
                "name": module.name,
                "label": module.label,
                "shape": f"{module.d_in}x{module.d_out}",
                "family_count": module.family_count,
            }
            for module in modules
        ],
        "compression_frontier": frontier,
        "rank_sensitivity": rank_sensitivity,
    }


def plot_results(summary: dict, output_path: Path) -> None:
    ensure_parent(output_path)

    frontier = summary["compression_frontier"]
    rank_sensitivity = summary["rank_sensitivity"]
    deployed_ratio = summary["deployed_compression_ratio"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    x = [item["compression_ratio"] * 100.0 for item in frontier]
    axes[0].plot(x, [item["tucker_relative_error"] for item in frontier], marker="o", linewidth=2.4, label="Shared Tucker")
    axes[0].plot(x, [item["svd_relative_error"] for item in frontier], marker="s", linewidth=2.4, label="Per-expert SVD")
    axes[0].axvline(deployed_ratio * 100.0, linestyle="--", color="#666666", linewidth=1.2, label="Current checkpoint")
    axes[0].set_title("Checkpoint Compression Frontier")
    axes[0].set_xlabel("Compression ratio (%)")
    axes[0].set_ylabel("Relative Frobenius reconstruction error")
    axes[0].grid(alpha=0.25, linestyle=":")
    axes[0].legend(frameon=False)

    axes[1].plot(
        [item["rank"] for item in rank_sensitivity["expert_mode"]],
        [item["relative_error"] for item in rank_sensitivity["expert_mode"]],
        marker="o",
        linewidth=2.1,
        label="Expert mode",
    )
    axes[1].plot(
        [item["rank"] for item in rank_sensitivity["input_mode"]],
        [item["relative_error"] for item in rank_sensitivity["input_mode"]],
        marker="s",
        linewidth=2.1,
        label="Input shared subspace",
    )
    axes[1].plot(
        [item["rank"] for item in rank_sensitivity["output_mode"]],
        [item["relative_error"] for item in rank_sensitivity["output_mode"]],
        marker="^",
        linewidth=2.1,
        label="Output shared subspace",
    )
    axes[1].set_title("Mode-Rank Sensitivity on Learned Expert Tensor")
    axes[1].set_xlabel("Retained rank")
    axes[1].set_ylabel("Relative Frobenius reconstruction error")
    axes[1].grid(alpha=0.25, linestyle=":")
    axes[1].legend(frameon=False)

    fig.suptitle(
        "Real checkpoint-space compression study on representative TuckerMoE families",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    modules, shape_breakdown, dense_total, deployed_total = collect_module_summaries(CHECKPOINT)
    summary = aggregate_results(modules, shape_breakdown, dense_total, deployed_total)

    ensure_parent(SUMMARY_JSON)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_results(summary, PLOT_PATH)

    print(f"Wrote summary to {SUMMARY_JSON}")
    print(f"Wrote plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
