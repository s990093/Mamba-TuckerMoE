import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ============================================================
# Paths (relative to this script)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

PRE_TRAIN_CSV = DATA_DIR / "train_log.csv"
INS_SFT_TRAIN_CSV = DATA_DIR / "train_sft_log.csv"
INS_SFT_VAL_CSV = DATA_DIR / "val_sft_log.csv"

COT_TRAIN_NAME = "train_sft_cot_log.csv"
COT_VAL_NAME = "val_sft_cot_log.csv"

OUTPUT_CE = SCRIPT_DIR / "ce_loss.png"
OUTPUT_PPL = SCRIPT_DIR / "ppl.png"

# Fallback when CSV has no tokens_seen column
DEFAULT_TOKENS_PER_STEP = {
    "pre": 512 * 128,
    "ins": 32 * 512,
    "cot": 32 * 512,
}
COT_V4_TOKENS_PER_STEP = 1024 * 16  # 16,384
COT_TOKENS_PER_STEP_OVERRIDE = {4: COT_V4_TOKENS_PER_STEP}

FIGSIZE = (10.5, 2.65)
TRAIN_SMOOTH_WINDOW = 80
TRAIN_ALPHA = 0.38
VAL_ALPHA = 0.92
Y_PERCENTILES = (1.5, 99.0)
PPL_CE_CLIP = 12.0


def read_csv(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def loss_column(df: pd.DataFrame) -> str:
    if "ce_loss" in df.columns:
        return "ce_loss"
    if "val_ce_loss" in df.columns:
        return "val_ce_loss"
    raise KeyError(f"Expected 'ce_loss' or 'val_ce_loss' in columns: {list(df.columns)}")


def tokens_seen_is_monotonic(df: pd.DataFrame) -> bool:
    if "tokens_seen" not in df.columns or len(df) < 2:
        return True
    return bool((np.diff(df["tokens_seen"].astype(np.int64)) >= 0).all())


def infer_tokens_per_step(df: pd.DataFrame, fallback: int) -> int:
    if "tokens_seen" not in df.columns or df.empty:
        return fallback
    ts = df["tokens_seen"].astype(np.int64).values
    if len(ts) >= 2:
        deltas = np.diff(ts)
        positives = deltas[deltas > 0]
        if len(positives):
            # First step-pair matches run config; mode breaks after resume / batch changes.
            return int(positives[0])
    steps = df["step"].astype(np.int64).values
    mask = steps > 0
    if mask.any():
        return int(ts[mask][0] / steps[mask][0])
    return int(ts[0]) if len(ts) else fallback


def stage_token_extent(df: pd.DataFrame, tokens_per_step: int) -> int:
    max_step = int(df["step"].astype(np.int64).max())
    step_extent = max_step * tokens_per_step
    if "tokens_seen" in df.columns and tokens_seen_is_monotonic(df) and not df.empty:
        return max(step_extent, int(df["tokens_seen"].astype(np.int64).max()))
    return step_extent


def steps_to_tokens(steps: np.ndarray, tokens_per_step: int, token_offset: int = 0) -> np.ndarray:
    return steps.astype(np.int64) * tokens_per_step + token_offset


def load_stage(
    filepath: Path,
    token_offset: int = 0,
    tokens_per_step: int | None = None,
    fallback_tokens_per_step: int = DEFAULT_TOKENS_PER_STEP["cot"],
):
    df = read_csv(filepath)
    col = loss_column(df)
    steps = df["step"].values.astype(np.int64)
    losses = df[col].values.astype(np.float64)
    tps = tokens_per_step if tokens_per_step is not None else infer_tokens_per_step(df, fallback_tokens_per_step)
    if "tokens_seen" in df.columns and tokens_seen_is_monotonic(df):
        tokens = df["tokens_seen"].values.astype(np.int64) + token_offset
    else:
        tokens = steps_to_tokens(steps, tps, token_offset)
    return tokens, losses, tps


def discover_cot_stages(data_dir: Path) -> list[dict]:
    """v1 at data root; v2+ in data/vN/ (auto-detected, sorted)."""
    stages: list[dict] = []
    v1_train = data_dir / COT_TRAIN_NAME
    if v1_train.exists():
        v1_val = data_dir / COT_VAL_NAME
        stages.append(
            {
                "version": 1,
                "train": v1_train,
                "val": v1_val if v1_val.exists() else None,
            }
        )

    version_dirs: list[tuple[int, Path]] = []
    for path in data_dir.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"v(\d+)", path.name)
        if match:
            version_dirs.append((int(match.group(1)), path))
    version_dirs.sort(key=lambda item: item[0])

    for version, folder in version_dirs:
        if version == 1:
            continue
        train = folder / COT_TRAIN_NAME
        if not train.exists():
            continue
        val = folder / COT_VAL_NAME
        stages.append(
            {
                "version": version,
                "train": train,
                "val": val if val.exists() else None,
            }
        )
    return stages


def cot_version_range_label(versions: list[int]) -> str:
    if not versions:
        return ""
    if len(versions) == 1:
        return f"v{versions[0]}"
    if versions == list(range(versions[0], versions[-1] + 1)):
        return f"v{versions[0]}→v{versions[-1]}"
    return "→".join(f"v{v}" for v in versions)


def smooth_median(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < 3:
        return y
    w = min(window, len(y))
    if w % 2 == 0:
        w -= 1
    w = max(w, 1)
    return (
        pd.Series(y)
        .rolling(w, center=True, min_periods=max(1, w // 5))
        .median()
        .to_numpy()
    )


def ce_to_ppl(ce: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(ce, None, PPL_CE_CLIP))


def robust_ylim(*arrays, percentiles=Y_PERCENTILES, pad=0.1):
    pool = np.concatenate([a[np.isfinite(a)] for a in arrays if len(a)])
    lo, hi = np.percentile(pool, percentiles)
    span = max(hi - lo, 0.2)
    return lo - span * pad, hi + span * pad


def token_formatter(x, _pos):
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:.1f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.0f}M" if x % 1e6 == 0 else f"{x / 1e6:.1f}M"
    if ax >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{int(x)}"


COLORS = {
    "train": "#1d4ed8",
    "val_ins": "#c2410c",
    "val_cot": "#7c2d12",
    "band_a": (0.93, 0.94, 0.97, 1.0),
    "band_b": (0.97, 0.97, 0.99, 1.0),
    "divider": "#9ca3af",
    "text": "#1f2937",
    "formula": "#4b5563",
}

STAGE_NAMES = ["Pre-train", "Indie Mode SFT", "CoT SFT"]


def draw_stage_bands(ax, boundaries):
    bands = [COLORS["band_a"], COLORS["band_b"], COLORS["band_a"]]
    for i in range(3):
        ax.axvspan(boundaries[i], boundaries[i + 1], facecolor=bands[i], edgecolor="none", zorder=0)
    for b in boundaries[1:-1]:
        ax.axvline(b, color=COLORS["divider"], linestyle=(0, (4, 4)), linewidth=0.7, zorder=1)


def style_axis(ax, grid=True):
    if grid:
        ax.grid(axis="y", linestyle=":", linewidth=0.35, alpha=0.55, color="#cbd5e1")
        ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#6b7280")
        ax.spines[spine].set_linewidth(0.6)


def add_stage_headers(fig, boundaries, cot_version_label: str):
    formulas = [
        r"$\mathcal{L}_{\mathrm{pre}}=\mathrm{CE}+\alpha\mathcal{L}_{\mathrm{lb}}+\beta\mathcal{L}_{z}$",
        r"$\mathcal{L}_{\mathrm{SFT}}=\mathrm{CE}$",
        rf"$\mathcal{{L}}_{{\mathrm{{CoT}}}}\approx 92\%\mathrm{{CE}}+8\%\mathrm{{FCP}}+\mathrm{{aux}}$ ({cot_version_label})",
    ]
    for i in range(3):
        x_pos = (boundaries[i] + boundaries[i + 1]) / 2
        fig.text(
            x_pos / boundaries[-1],
            0.965,
            STAGE_NAMES[i],
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=COLORS["text"],
            transform=fig.transFigure,
        )
        fig.text(
            x_pos / boundaries[-1],
            0.935,
            formulas[i],
            ha="center",
            va="bottom",
            fontsize=6.5,
            style="italic",
            color=COLORS["formula"],
            transform=fig.transFigure,
        )


def plot_combined(
    *,
    train_x,
    train_y_raw,
    train_y_smooth,
    val_segments,
    boundaries,
    cot_version_label,
    y_transform,
    ylabel,
    output_path,
):
    train_y_plot = y_transform(train_y_smooth)
    train_y_raw_t = y_transform(train_y_raw)
    val_segments_t = [(x, y_transform(y), label, color) for x, y, label, color in val_segments]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.14)

    draw_stage_bands(ax, boundaries)
    ax.set_xlim(0, boundaries[-1])
    ax.xaxis.set_major_formatter(FuncFormatter(token_formatter))
    style_axis(ax)

    ax.plot(
        train_x,
        train_y_plot,
        color=COLORS["train"],
        linewidth=1.1,
        alpha=TRAIN_ALPHA,
        label="Training (median-smoothed)",
        zorder=3,
        solid_capstyle="round",
    )

    val_ys = []
    for x, y, label, color in val_segments_t:
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.35,
            alpha=VAL_ALPHA,
            label=label,
            zorder=5,
            solid_capstyle="round",
        )
        val_ys.append(y)

    ax.set_xlabel("Tokens Seen (cumulative)")
    ax.set_ylabel(ylabel)
    y_lo, y_hi = robust_ylim(train_y_raw_t, *val_ys)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#d1d5db", fontsize=7)

    add_stage_headers(fig, boundaries, cot_version_label)
    fig.savefig(output_path, format="png")
    plt.close(fig)


def load_fixed_stage(train_csv: Path, val_csv: Path | None, token_start: int, fallback_tps: int):
    train_df = read_csv(train_csv)
    train_tps = infer_tokens_per_step(train_df, fallback_tps)
    train_x, train_loss, _ = load_stage(train_csv, token_start, train_tps)
    token_end = token_start + stage_token_extent(train_df, train_tps)

    val_x, val_loss = np.array([]), np.array([])
    if val_csv is not None and val_csv.exists():
        val_x, val_loss, _ = load_stage(val_csv, token_start, train_tps)
    return train_x, train_loss, val_x, val_loss, token_end, train_tps


def load_cot_chain(stages: list[dict], cot_start: int):
    train_x_parts, train_loss_parts = [], []
    val_x_parts, val_loss_parts = [], []
    token_cursor = cot_start
    tokens_per_version: dict[int, int] = {}

    for stage in stages:
        version = stage["version"]
        segment_start = token_cursor
        train_df = read_csv(stage["train"])
        train_tps = COT_TOKENS_PER_STEP_OVERRIDE.get(
            version, infer_tokens_per_step(train_df, DEFAULT_TOKENS_PER_STEP["cot"])
        )
        tokens_per_version[version] = train_tps

        x, loss, _ = load_stage(stage["train"], segment_start, train_tps)
        train_x_parts.append(x)
        train_loss_parts.append(loss)
        token_cursor = segment_start + stage_token_extent(train_df, train_tps)

        if stage["val"] is not None:
            val_x, val_loss, _ = load_stage(stage["val"], segment_start, train_tps)
            val_x_parts.append(val_x)
            val_loss_parts.append(val_loss)

    cot_train_x = np.concatenate(train_x_parts) if train_x_parts else np.array([])
    cot_train_loss = np.concatenate(train_loss_parts) if train_loss_parts else np.array([])
    cot_val_x = np.concatenate(val_x_parts) if val_x_parts else np.array([])
    cot_val_loss = np.concatenate(val_loss_parts) if val_loss_parts else np.array([])
    return cot_train_x, cot_train_loss, cot_val_x, cot_val_loss, token_cursor, tokens_per_version


# ============================================================
# Load data
# ============================================================
cot_stages = discover_cot_stages(DATA_DIR)
if not cot_stages:
    raise FileNotFoundError(f"No CoT logs found under {DATA_DIR}")

cot_versions = [s["version"] for s in cot_stages]
cot_label = cot_version_range_label(cot_versions)

pre_train_x, pre_train_loss, _, _, pre_token_end, pre_tps = load_fixed_stage(
    PRE_TRAIN_CSV, None, 0, DEFAULT_TOKENS_PER_STEP["pre"]
)
ins_train_x, ins_train_loss, ins_val_x, ins_val_loss, ins_token_end, ins_tps = load_fixed_stage(
    INS_SFT_TRAIN_CSV, INS_SFT_VAL_CSV, pre_token_end, DEFAULT_TOKENS_PER_STEP["ins"]
)
cot_train_x, cot_train_loss, cot_val_x, cot_val_loss, cot_token_end, cot_tps_map = load_cot_chain(
    cot_stages, ins_token_end
)

train_x = np.concatenate([pre_train_x, ins_train_x, cot_train_x])
train_loss_raw = np.concatenate([pre_train_loss, ins_train_loss, cot_train_loss])
train_loss = smooth_median(train_loss_raw, TRAIN_SMOOTH_WINDOW)

boundaries = [0, pre_token_end, ins_token_end, cot_token_end]

val_segments = [
    (ins_val_x, ins_val_loss, "Indie Mode SFT (val)", COLORS["val_ins"]),
    (cot_val_x, cot_val_loss, "CoT SFT (val)", COLORS["val_cot"]),
]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)

plot_kwargs = dict(
    train_x=train_x,
    train_y_raw=train_loss_raw,
    train_y_smooth=train_loss,
    val_segments=val_segments,
    boundaries=boundaries,
    cot_version_label=cot_label,
)

plot_combined(**plot_kwargs, y_transform=lambda y: y, ylabel="Cross-Entropy Loss", output_path=OUTPUT_CE)
plot_combined(**plot_kwargs, y_transform=ce_to_ppl, ylabel="Perplexity", output_path=OUTPUT_PPL)

print(f"Saved {OUTPUT_CE.name} and {OUTPUT_PPL.name}")
print(f"CoT stages: {cot_label} ({len(cot_stages)} segments)")
for version, tps in cot_tps_map.items():
    print(f"  v{version}: tokens/step={tps:,}")

cot_token_parts = []
offset = ins_token_end
for stage in cot_stages:
    df = read_csv(stage["train"])
    tps = cot_tps_map[stage["version"]]
    extent = stage_token_extent(df, tps)
    cot_token_parts.append((stage["version"], extent))
    offset += extent

cot_breakdown = " ".join(f"cot_v{v}={n:,}" for v, n in cot_token_parts)
print(
    f"Tokens: pre={pre_token_end:,} ({pre_tps:,}/step) "
    f"ins={ins_token_end - pre_token_end:,} ({ins_tps:,}/step) "
    f"{cot_breakdown} total={boundaries[-1]:,}"
)
