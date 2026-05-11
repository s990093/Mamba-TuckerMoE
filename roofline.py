import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ==========================================
# 1. 論文級圖表全局設定 (Times New Roman)
# ==========================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 11
rcParams['axes.titlesize'] = 13
rcParams['axes.labelsize'] = 12
rcParams['legend.fontsize'] = 9
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['figure.dpi'] = 300

# ==========================================
# 2. 完整 GPU / 晶片硬體數據庫 (FP16/BF16)
# ==========================================
gpus = {
    # Data Center GPU
    'H100 (SXM)':     {'tflops': 1979, 'bw': 3.35, 'color': '#D62728', 'ls': '-'},
    'MI300A':         {'tflops': 1961, 'bw': 5.30, 'color': '#FF7F0E', 'ls': '--'},
    'A100 (SXM)':     {'tflops': 312,  'bw': 2.04, 'color': '#2CA02C', 'ls': '-.'},
    # Consumer GPU
    'RTX 4090':       {'tflops': 330,  'bw': 1.00, 'color': '#1F77B4', 'ls': ':'},
    'RTX 3090':       {'tflops': 142,  'bw': 0.94, 'color': '#9467BD', 'ls': '-'},
    # Apple Silicon (估算 FP16)
    'M3 Max (est.)':  {'tflops': 40,   'bw': 0.40, 'color': '#8C564B', 'ls': '--'},
    'M2 Ultra (est.)':{'tflops': 27,   'bw': 0.80, 'color': '#E377C2', 'ls': '-.'},
    'M1 Pro (est.)':  {'tflops': 10,   'bw': 0.20, 'color': '#7F7F7F', 'ls': ':'}
}

# ==========================================
# 3. 數學模型：不同壓縮維度 r 的算術強度
# ==========================================
def calc_intensity(n_val, r_val):
    # 算術強度 I 逼近於 r，分母常數項假設為 1024 模擬 overhead
    return (r_val * n_val) / (n_val + 1024)

# 定義多組 TuckerMoE 維度
r_values = [128, 256, 512, 768]
colors_r = ['#A8D8EA', '#76BA1B', '#00B4D8', '#0077B6'] # 漸層藍綠色系

# ==========================================
# 4. 圖表一：Phase Transition (多組 r 曲線)
# ==========================================
fig1, ax1 = plt.subplots(figsize=(11, 6))
n_array = np.linspace(1, 8000, 1000)

# 畫出不同 r 的攀爬曲線
for i, r in enumerate(r_values):
    i_array = calc_intensity(n_array, r)
    lw = 3 if r == 512 else 2 # 突顯主角 r=512
    ax1.plot(n_array, i_array, color=['#B0E0E6', '#87CEFA', '#17BECF', '#4682B4'][i], 
             lw=lw, label=f'TuckerMoE ($r={r}$)')

# 基準線與硬體轉折點
ax1.axhline(y=1, color='gray', linestyle=':', lw=2, label=r'Dense Baseline ($I \approx 1$)')
ax1.axhline(y=1979/3.35, color='#D62728', linestyle='--', alpha=0.5, label='H100 Ridge (~590)')
ax1.axhline(y=1961/5.30, color='#FF7F0E', linestyle='--', alpha=0.5, label='MI300 Ridge (~370)')
ax1.axhline(y=312/2.04,  color='#2CA02C', linestyle='--', alpha=0.5, label='A100 Ridge (~153)')

# 背景區塊 (Decode vs Prefill)
ax1.axvspan(0, 256, color='#E6F3FF', alpha=0.6)
ax1.text(128, 600, 'Decode\nRegime', ha='center', va='center', fontweight='bold', color='#1F77B4')
ax1.axvspan(256, 8000, color='#E6FFE6', alpha=0.4)
ax1.text(4000, 600, 'Prefill Regime', ha='center', va='center', fontweight='bold', color='#2CA02C')

# 在 r=512 曲線上打點
for n in [64, 256, 1024, 4096]:
    i_val = calc_intensity(n, 512)
    ax1.plot(n, i_val, 'ko', markersize=5)
    ax1.annotate(f'$N={n}$', xy=(n, i_val), xytext=(12, -8), textcoords='offset points')

ax1.set_xlim(0, 8000)
ax1.set_ylim(0, 650)
ax1.set_xlabel('Token Batch Size ($N$)')
ax1.set_ylabel('Arithmetic Intensity (FLOPs/Byte)')
ax1.set_title('Phase Transition w.r.t Batch Size', fontweight='bold', pad=15)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='lower right', framealpha=0.9, ncol=1)

plt.tight_layout()
plt.savefig('TuckerMoE_PhaseTransition.pdf', format='pdf', bbox_inches='tight')
plt.savefig('TuckerMoE_PhaseTransition.png', format='png', bbox_inches='tight', dpi=300)

# ==========================================
# ==========================================
# 5. 圖表二：全陣容 GPU Roofline
# ==========================================
fig2, ax2 = plt.subplots(figsize=(11, 6))
intensity_range = np.logspace(-0.3, 3.1, 1000)

# 繪製背景背景色：Memory-bound vs Compute-bound
ax2.axvspan(0.1, 10, color='#fdf2f2', alpha=0.3, label='Memory-Bound Regime')
ax2.axvspan(10, 2000, color='#f0fdf4', alpha=0.3, label='Compute-Bound Regime')

# 畫所有硬體的 Roofline
for name, specs in gpus.items():
    perf = np.minimum(specs['tflops'], intensity_range * specs['bw'])
    lw = 2.5 if 'H100' in name or '4090' in name else 1.2
    alpha = 1.0 if lw > 2 else 0.6
    ax2.plot(intensity_range, perf, label=name, color=specs['color'], linestyle=specs['ls'], lw=lw, alpha=alpha, zorder=2)

# 標註運作點 (Operational Points)
i_decode = calc_intensity(64, 512)   
i_prefill = calc_intensity(4096, 512) 

# 在 H100 曲線上的運作點打星號
h100_perf_decode = np.minimum(gpus['H100 (SXM)']['tflops'], i_decode * gpus['H100 (SXM)']['bw'])
h100_perf_prefill = np.minimum(gpus['H100 (SXM)']['tflops'], i_prefill * gpus['H100 (SXM)']['bw'])

ax2.scatter([i_decode], [h100_perf_decode], color='#1F77B4', marker='*', s=200, edgecolor='white', linewidth=1.5, zorder=10, label='Tucker Decode Point')
ax2.scatter([i_prefill], [h100_perf_prefill], color='#2CA02C', marker='D', s=100, edgecolor='white', linewidth=1.5, zorder=10, label='Tucker Prefill Point')

# 指引線與文字標籤
ax2.annotate(f'Decode ($I={i_decode:.1f}$)\nMemory-Bound', 
             xy=(i_decode, h100_perf_decode), xytext=(i_decode*0.15, h100_perf_decode*2.5),
             arrowprops=dict(arrowstyle='->', color='#1F77B4', lw=1.5),
             fontsize=10, fontweight='bold', color='#1F77B4',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#1F77B4', alpha=0.8))

ax2.annotate(f'Prefill ($I={i_prefill:.0f}$)\nCompute-Bound', 
             xy=(i_prefill, h100_perf_prefill), xytext=(i_prefill*0.1, h100_perf_prefill*0.1),
             arrowprops=dict(arrowstyle='->', color='#2CA02C', lw=1.5),
             fontsize=10, fontweight='bold', color='#2CA02C',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#2CA02C', alpha=0.8))

# 輔助線
ax2.axvline(x=i_decode, color='#1F77B4', linestyle=':', lw=1, alpha=0.5)
ax2.axvline(x=i_prefill, color='#2CA02C', linestyle=':', lw=1, alpha=0.5)

# 設定坐標軸
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(0.5, 1200)
ax2.set_ylim(1, 10000)
ax2.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', labelpad=10)
ax2.set_ylabel('Attainable Performance (TFLOPS)', labelpad=10)
ax2.set_title('Hybrid Mamba-TuckerMoE: GPU Roofline Analysis', fontweight='bold', size=15, pad=20)
ax2.grid(True, which='both', linestyle='--', alpha=0.2)

# 圖籤優化：分兩欄放置，避免遮擋
ax2.legend(loc='upper left', framealpha=0.95, ncol=2, fontsize=8.5, edgecolor='#d1d5db')

plt.tight_layout()
plt.savefig('TuckerMoE_Roofline_Analysis.pdf', format='pdf', bbox_inches='tight')
plt.savefig('TuckerMoE_Roofline_Analysis.png', format='png', bbox_inches='tight', dpi=300)

print("圖表生成完畢：")
print("1. TuckerMoE_PhaseTransition.pdf / .png")
print("2. TuckerMoE_Roofline_Analysis.pdf / .png")
plt.show()