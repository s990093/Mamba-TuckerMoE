import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============================================================
# Paths
# ============================================================
PRE_TRAIN_CSV = "./data/train_log.csv"
INS_SFT_TRAIN_CSV = "./data/train_sft_log.csv"
INS_SFT_VAL_CSV = "./data/val_sft_log.csv"
COT_SFT_TRAIN_CSV = "./data/train_sft_cot_log.csv"
COT_SFT_VAL_CSV = "./data/val_sft_cot_log.csv"
COT_V2_SFT_TRAIN_CSV = "./data/v2/train_sft_cot_log.csv"  # V2 continues from v1
COT_V2_SFT_VAL_CSV = "./data/v2/val_sft_cot_log.csv"

OUTPUT_CE = "ce_loss.png"
OUTPUT_PPL = "ppl.png"

PRE_TOKENS_PER_STEP = 512 * 128   # 65,536
SFT_TOKENS_PER_STEP = 32 * 512    # 16,384
COT_V2_TOKENS_PER_STEP = 1024 * 6 * 8  # 49,152 (V2 uses larger SEQ_LEN)

FIGSIZE = (10.5, 2.65)
TRAIN_SMOOTH_WINDOW = 80
TRAIN_ALPHA = 0.38          # faint train so val stays readable
VAL_ALPHA = 0.92
Y_PERCENTILES = (1.5, 99.0)
PPL_CE_CLIP = 12.0


def read_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def loss_column(df: pd.DataFrame) -> str:
    if "ce_loss" in df.columns:
        return "ce_loss"
    if "val_ce_loss" in df.columns:
        return "val_ce_loss"
    raise KeyError(f"Expected 'ce_loss' or 'val_ce_loss' in columns: {list(df.columns)}")


def stage_max_step(filepath: str) -> int:
    return int(read_csv(filepath)["step"].max())


def steps_to_tokens(steps: np.ndarray, tokens_per_step: int, token_offset: int = 0) -> np.ndarray:
    return steps.astype(np.int64) * tokens_per_step + token_offset


def load_stage(filepath: str, token_offset: int = 0, tokens_per_step: int = SFT_TOKENS_PER_STEP):
    df = read_csv(filepath)
    col = loss_column(df)
    steps = df["step"].values.astype(np.int64)
    tokens = steps_to_tokens(steps, tokens_per_step, token_offset)
    losses = df[col].values.astype(np.float64)
    return tokens, losses


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
    "val_v2": "#059669",
    "band_a": (0.93, 0.94, 0.97, 1.0),
    "band_b": (0.97, 0.97, 0.99, 1.0),
    "divider": "#9ca3af",
    "text": "#1f2937",
    "formula": "#4b5563",
}

STAGE_NAMES = ["Pre-train", "Indie Mode SFT", "CoT SFT"]
FORMULAS = [
    r"$\mathcal{L}_{\mathrm{pre}}=\mathrm{CE}+\alpha\mathcal{L}_{\mathrm{lb}}+\beta\mathcal{L}_{z}$",
    r"$\mathcal{L}_{\mathrm{SFT}}=\mathrm{CE}$",
    r"$\mathcal{L}_{\mathrm{CoT}}\approx 92\%\mathrm{CE}+8\%\mathrm{FCP}+\mathrm{aux}$ (v1→v2)",
]


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


def add_stage_headers(fig, boundaries):
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
            FORMULAS[i],
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

    add_stage_headers(fig, boundaries)
    fig.savefig(output_path, format="png")
    plt.close(fig)


# ============================================================
# Load data
# ============================================================
PRE_STEPS = stage_max_step(PRE_TRAIN_CSV)
INS_STEPS = stage_max_step(INS_SFT_TRAIN_CSV)
COT_V1_STEPS = stage_max_step(COT_SFT_TRAIN_CSV)
COT_V2_STEPS = stage_max_step(COT_V2_SFT_TRAIN_CSV)

pre_token_start = 0
ins_token_start = PRE_STEPS * PRE_TOKENS_PER_STEP
cot_v1_token_start = ins_token_start + INS_STEPS * SFT_TOKENS_PER_STEP
cot_v2_token_start = cot_v1_token_start + COT_V1_STEPS * SFT_TOKENS_PER_STEP
boundaries = [
    pre_token_start,
    ins_token_start,
    cot_v1_token_start,
    cot_v2_token_start + COT_V2_STEPS * COT_V2_TOKENS_PER_STEP,
]

pre_train_x, pre_train_loss = load_stage(PRE_TRAIN_CSV, pre_token_start, PRE_TOKENS_PER_STEP)
ins_train_x, ins_train_loss = load_stage(INS_SFT_TRAIN_CSV, ins_token_start, SFT_TOKENS_PER_STEP)

# CoT SFT: continuous training from v1 → v2
cot_v1_train_x, cot_v1_train_loss = load_stage(COT_SFT_TRAIN_CSV, cot_v1_token_start, SFT_TOKENS_PER_STEP)
cot_v2_train_x, cot_v2_train_loss = load_stage(COT_V2_SFT_TRAIN_CSV, cot_v2_token_start, COT_V2_TOKENS_PER_STEP)

train_x = np.concatenate([pre_train_x, ins_train_x, cot_v1_train_x, cot_v2_train_x])
train_loss_raw = np.concatenate([pre_train_loss, ins_train_loss, cot_v1_train_loss, cot_v2_train_loss])
train_loss = smooth_median(train_loss_raw, TRAIN_SMOOTH_WINDOW)

ins_val_x, ins_val_loss = load_stage(INS_SFT_VAL_CSV, ins_token_start, SFT_TOKENS_PER_STEP)
cot_v1_val_x, cot_v1_val_loss = load_stage(COT_SFT_VAL_CSV, cot_v1_token_start, SFT_TOKENS_PER_STEP)
cot_v2_val_x, cot_v2_val_loss = load_stage(COT_V2_SFT_VAL_CSV, cot_v2_token_start, COT_V2_TOKENS_PER_STEP)

# Merge CoT v1 and v2 validation data into single line
cot_val_x = np.concatenate([cot_v1_val_x, cot_v2_val_x])
cot_val_loss = np.concatenate([cot_v1_val_loss, cot_v2_val_loss])

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

plot_combined(
    train_x=train_x,
    train_y_raw=train_loss_raw,
    train_y_smooth=train_loss,
    val_segments=val_segments,
    boundaries=boundaries,
    y_transform=lambda y: y,
    ylabel="Cross-Entropy Loss",
    output_path=OUTPUT_CE,
)

plot_combined(
    train_x=train_x,
    train_y_raw=train_loss_raw,
    train_y_smooth=train_loss,
    val_segments=val_segments,
    boundaries=boundaries,
    y_transform=ce_to_ppl,
    ylabel="Perplexity",
    output_path=OUTPUT_PPL,
)

print(f"Saved {OUTPUT_CE} and {OUTPUT_PPL}")
cot_v1_tokens = cot_v2_token_start - cot_v1_token_start
cot_v2_tokens = boundaries[3] - cot_v2_token_start
print(
    f"Tokens: pre={boundaries[1]:,} ins={boundaries[2]-boundaries[1]:,} "
    f"cot_v1={cot_v1_tokens:,} cot_v2={cot_v2_tokens:,} total={boundaries[-1]:,}"
)
