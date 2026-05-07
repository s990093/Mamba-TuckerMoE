import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ==========================================
# 1. 論文級圖表全局設定 (Times New Roman)
# ==========================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'  # 讓數學符號看起來像 LaTeX
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 13
rcParams['axes.labelsize'] = 11
rcParams['legend.fontsize'] = 8
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['figure.dpi'] = 300  # 高解析度輸出

# ==========================================
# 2. GPU 硬體數據庫 (FP16/BF16)
# ==========================================
gpus = {
    # Data center
    'H100 (SXM)':         {'tflops': 1979, 'bw': 3.35, 'color': '#D62728', 'ls': '-',  'group': 'dc'},
    'MI300A':             {'tflops': 1961, 'bw': 5.30, 'color': '#FF7F0E', 'ls': '--', 'group': 'dc'},
    'A100 (SXM)':         {'tflops': 312,  'bw': 2.04, 'color': '#2CA02C', 'ls': '-.', 'group': 'dc'},
    # Consumer
    'RTX 4090':           {'tflops': 330,  'bw': 1.00, 'color': '#1F77B4', 'ls': ':',  'group': 'consumer'},
    'RTX 3090':           {'tflops': 142,  'bw': 0.94, 'color': '#9467BD', 'ls': '-',  'group': 'consumer'},
    # Apple Silicon (estimated AI throughput from com.md)
    'M3 Max (est.)':      {'tflops': 40,   'bw': 0.50, 'color': '#8C564B', 'ls': '--', 'group': 'apple'},
    'M2 Ultra (est.)':    {'tflops': 27,   'bw': 0.80, 'color': '#E377C2', 'ls': '-.', 'group': 'apple'},
    'M1 Pro (est.)':      {'tflops': 10,   'bw': 0.20, 'color': '#7F7F7F', 'ls': ':',  'group': 'apple'},
}

# ==========================================
# 3. 數學模型：計算 TuckerMoE 的算術強度
# 根據推導：I = (512 * N) / (N + 1024) (假設 r2=512, r3=256, k=2, E=8)
# ==========================================
def calc_intensity(n_val, r_out=512, r_in=256, num_experts=8):
    # I_core ≈ (r_out * N) / (N + E * r_out / r_in), with k absorbed in both terms
    return (r_out * n_val) / (n_val + (num_experts * r_out / r_in))

# 定義要觀察的特定 x (Batch Size N)
decode_n = [1, 64, 128, 256]
prefill_n = [1024, 4096, 8000]

# ==========================================
# 4. 開始繪圖 (1x2 雙拼圖)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.6), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=4.0 / 72.0, h_pad=3.0 / 72.0, wspace=0.12, hspace=0.05)

# ------------------------------------------
# 左圖：AI Ceiling Climb Curve (I vs N)
# ------------------------------------------
n_array = np.linspace(1, 8000, 1000)
r_out_candidates = [128, 256, 512, 768]
r_curve_colors = {
    128: '#8DD3C7',
    256: '#80B1D3',
    512: '#17BECF',
    768: '#1F78B4',
}

# 左圖：不同 r (Dim_OUT) 的強度曲線
for r_out in r_out_candidates:
    i_array = calc_intensity(n_array, r_out=r_out)
    lw = 3 if r_out == 512 else 2.0
    alpha = 1.0 if r_out == 512 else 0.9
    ax1.plot(
        n_array,
        i_array,
        color=r_curve_colors[r_out],
        lw=lw,
        alpha=alpha,
        label=rf'TuckerMoE ($r={r_out}$)',
    )
ax1.axhline(y=1, color='gray', linestyle=':', lw=2, label=r'Dense Baseline ($I \approx 1$)')

# 標示所有裝置的 Ridge Point（依群組控制透明度）
for name, specs in gpus.items():
    ridge = specs['tflops'] / specs['bw']
    if specs['group'] == 'dc':
        alpha = 0.45
        lw = 1.5
    elif specs['group'] == 'consumer':
        alpha = 0.33
        lw = 1.2
    else:
        alpha = 0.28
        lw = 1.0
    ax1.axhline(y=ridge, color=specs['color'], linestyle='--', alpha=alpha, lw=lw)

# 核心 Ridge（主要展示三條 Data Center 線）
ax1.plot([], [], color='#D62728', linestyle='--', label='H100 Ridge (~590)')
ax1.plot([], [], color='#FF7F0E', linestyle='--', label='MI300 Ridge (~370)')
ax1.plot([], [], color='#2CA02C', linestyle='--', label='A100 Ridge (~153)')

# 繪製 Decode 與 Prefill 區間背景
ax1.axvspan(0, 256, color='#E6F3FF', alpha=0.6)
ax1.text(150, 600, 'Decode\nRegime', ha='center', va='center', fontweight='bold', color='#1F77B4')
ax1.axvspan(256, 8000, color='#E6FFE6', alpha=0.4)
ax1.text(4000, 605, 'Prefill Regime', ha='center', va='center', fontweight='bold', color='#2CA02C')

# 標出特定的 N 值
for n in [64, 256, 1024, 4096]:
    i_val = calc_intensity(n, r_out=512)
    ax1.plot(n, i_val, 'ko', markersize=5)
    if n == 4096:
        offset = (12, -14)
    elif n == 1024:
        offset = (10, -2)
    else:
        offset = (8, -12)
    ax1.annotate(f'$N={n}$', xy=(n, i_val), xytext=offset, textcoords='offset points', fontsize=9)

ax1.set_xlim(0, 8000)
ax1.set_ylim(0, 650)
ax1.set_xlabel('Token Batch Size ($N$)')
ax1.set_ylabel('Arithmetic Intensity (FLOPs/Byte)')
ax1.set_title('(a) Phase Transition w.r.t Batch Size', fontweight='bold', pad=15)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='lower right', framealpha=0.92, borderpad=0.45, handlelength=2.2)

# ------------------------------------------
# 右圖：Classic Roofline Model
# ------------------------------------------
# X 軸：Intensity (Log scale, 0.1 to 1000)
intensity_range = np.logspace(-1, 3, 500)

for name, specs in gpus.items():
    peak_perf = specs['tflops']
    bw = specs['bw']
    # 效能 = min(峰值算力, 算術強度 * 頻寬)
    perf = np.minimum(peak_perf, intensity_range * bw)
    ax2.plot(intensity_range, perf, label=name, color=specs['color'], linestyle=specs['ls'], lw=2)

# 在 Roofline 上標出 Decode (N=64) 和 Prefill (N=4096) 的垂直線與工作點
i_decode = calc_intensity(64, r_out=512)   # 512 設定下 decode 點
i_prefill = calc_intensity(4096, r_out=512) # 512 設定下 prefill 點

r_ref_colors = {
    128: '#9E9E9E',
    256: '#757575',
    512: '#424242',
    768: '#212121',
}

# 繪製垂直線
ax2.axvline(x=1, color='gray', linestyle=':', lw=2, label='Dense Baseline')
ax2.axvline(x=i_decode, color='#1F77B4', linestyle='-', lw=1.5, alpha=0.7)
ax2.axvline(x=i_prefill, color='#2CA02C', linestyle='-', lw=1.5, alpha=0.7)
for r_out in r_out_candidates:
    ax2.axvline(x=r_out, color=r_ref_colors[r_out], linestyle='--', lw=1.25, alpha=0.9, zorder=2)
    ax2.text(
        r_out,
        3000,
        rf'$r={r_out}$',
        fontsize=8.2,
        color=r_ref_colors[r_out],
        rotation=90,
        va='top',
        ha='center',
        alpha=0.98,
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
    )

# 標註文字（避免互相重疊）
ax2.annotate(
    f'Tucker Decode\n$N=64, I={i_decode:.1f}$',
    xy=(i_decode, 8.5),
    xytext=(i_decode * 1.2, 13.0),
    color='#1F77B4',
    fontsize=8.5,
    arrowprops=dict(arrowstyle='-', lw=0.9, color='#1F77B4', alpha=0.8),
    bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.65),
)
ax2.annotate(
    f'Tucker Prefill\n$N=4096, I={i_prefill:.0f}$',
    xy=(i_prefill, 8.5),
    xytext=(i_prefill * 0.82, 9.5),
    color='#2CA02C',
    fontsize=8.5,
    ha='right',
    arrowprops=dict(arrowstyle='-', lw=0.9, color='#2CA02C', alpha=0.8),
    bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.65),
)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(0.5, 1000)
ax2.set_ylim(1, 4000)
ax2.set_xlabel('Arithmetic Intensity (FLOPs/Byte)')
ax2.set_ylabel('Attainable Performance (TFLOPS)', labelpad=2)
ax2.set_title('(b) GPU Roofline & Operational Points', fontweight='bold', pad=15)
ax2.grid(True, which='both', linestyle='--', alpha=0.3)
ax2.legend(
    loc='upper left',
    bbox_to_anchor=(0.015, 0.995),
    framealpha=0.92,
    borderpad=0.35,
    handlelength=1.7,
    handletextpad=0.45,
    labelspacing=0.28,
    ncol=2,
    fontsize=7.6,
)

# ==========================================
# 5. 顯示與儲存
# ==========================================
output_pdf = 'paper/hybrid-mamba-15min/assets/plots/roofline_tuckermoe_analysis.pdf'
output_png = 'paper/hybrid-mamba-15min/assets/plots/roofline_tuckermoe_analysis.png'
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
plt.savefig(output_png, format='png', bbox_inches='tight', dpi=300)
print(f"圖表已成功生成：{output_pdf} / {output_png}")
if 'agg' not in plt.get_backend().lower():
    plt.show()