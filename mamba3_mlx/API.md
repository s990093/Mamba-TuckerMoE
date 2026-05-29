# MLX API 完整參考手冊

> **版本**: MLX (Apple Silicon 專用機器學習框架)  
> **架構**: 延遲評估 (lazy evaluation), 統一記憶體 (unified memory), Metal GPU 加速  
> **對比參考**: PyTorch (`torch.*`), NumPy (`np.*`), JAX (`jnp.*`)

---

## 目錄

1. [核心概念](#1-核心概念)
2. [mx.core — 陣列操作與數學](#2-mxcore--陣列操作與數學)
3. [mx.fast — 優化內核](#3-mxfast--優化內核)
4. [mx.random — 隨機數](#4-mxrandom--隨機數)
5. [mx.linalg — 線性代數](#5-mxlinalg--線性代數)
6. [mx.fft — 傅立葉變換](#6-mxfft--傅立葉變換)
7. [mx.metal / GPU — 設備管理](#7-mxmetal--gpu--設備管理)
8. [nn.Module — 模型基底類](#8-nnmodule--模型基底類)
9. [nn.Linear / Convolution / Embedding](#9-nnlinear--convolution--embedding)
10. [nn.Normalization — 正規化層](#10-nnnormalization--正規化層)
11. [nn.Activations — 激活函數](#11-nnactivations--激活函數)
12. [nn.Transformer / Attention](#12-nntransformer--attention)
13. [nn.Recurrent — RNN / LSTM / GRU](#13-nnrecurrent--rnn--lstm--gru)
14. [nn.Pooling / Dropout / Upsample](#14-nnpooling--dropout--upsample)
15. [nn.Quantized — 量化層](#15-nnquantized--量化層)
16. [nn.init — 權重初始化](#16-nninit--權重初始化)
17. [nn.losses — 損失函數](#17-nnlosses--損失函數)
18. [mx.compile — JIT 編譯](#18-mxcompile--jit-編譯)
19. [mx.custom_function — 自定義 Metal 內核](#19-mxcustom_function--自定義-metal-內核)
20. [mx.grad / vjp / jvp — 自動微分](#20-mxgrad--vjp--jvp--自動微分)
21. [mx.vmap — 自動向量化](#21-mxvmap--自動向量化)
22. [mx.distributed — 分散式訓練](#22-mxdistributed--分散式訓練)
23. [mlx.utils — 樹操作工具](#23-mlxutils--樹操作工具)
24. [Mamba3-XR 實際使用模式](#24-mamba3-xr-實際使用模式)
25. [PyTorch ↔ MLX 快速對照表](#25-pytorch--mlx-快速對照表)

---

## 1. 核心概念

### 1.1 延遲評估 (Lazy Evaluation)

MLX **不會立即執行**運算。所有操作先建立計算圖 (computation graph)，只有呼叫 `mx.eval()` 或讀取數值 (`.item()`, `.tolist()`, `print()`) 時才實際執行。

```python
import mlx.core as mx

a = mx.ones((1000, 1000))
b = mx.ones((1000, 1000))
c = a + b          # 尚未執行 — 只是記錄了「加」這個操作
d = c @ c.T        # 尚未執行 — 又記錄了「矩陣乘法」
mx.eval(d)         # 現在才真正在 GPU 上執行
```

**PyTorch 對比**: PyTorch 預設 eager mode (立即執行)，MLX 預設 lazy (類似 JAX)。

**注意**: 如果不呼叫 `mx.eval()`，陣列只是「承諾」。`print()` 或 `.item()` 會強制執行。

### 1.2 統一記憶體 (Unified Memory)

Apple Silicon 的 CPU 和 GPU 共享同一塊物理記憶體。MLX 陣列不需要 `.to('cuda')` 或 `.cpu()` 的轉換 — 資料永遠在同一個位址空間。

```python
# MLX: 無需設備轉換
x = mx.ones((100, 100))          # 在統一記憶體中
y = mx.exp(x)                     # GPU 直接讀寫，無需複製

# PyTorch (對比):
# x = torch.ones(100, 100).to('cuda')  # 需要顯式設備管理
```

### 1.3 資料型態 (dtypes)

```python
# 浮點數
mx.float16    # 半精度 (half)
mx.bfloat16   # Brain float (Mamba3 使用的格式)
mx.float32    # 單精度 (float)
mx.complex64  # 複數

# 整數
mx.int8, mx.int16, mx.int32, mx.int64
mx.uint8, mx.uint16, mx.uint32, mx.uint64

# 布林
mx.bool_
```

**型態轉換**: `x.astype(mx.float32)` — 類似 PyTorch `.to(torch.float32)` 或 NumPy `.astype(np.float32)`。

### 1.4 設備

```python
mx.gpu                     # 預設 GPU 設備
mx.cpu                     # CPU 設備
mx.default_device()        # 查詢當前預設設備
mx.set_default_device(mx.gpu)  # 設為 GPU（預設）
```

---

## 2. mx.core — 陣列操作與數學

### 2.1 陣列創建

```python
import mlx.core as mx

# 從 Python 資料創建
mx.array([1, 2, 3])                    # → array([1, 2, 3], dtype=int32)
mx.array([[1.0, 2.0], [3.0, 4.0]])    # → array([[1, 2], [3, 4]], dtype=float32)
mx.array([1, 2, 3], dtype=mx.float16)  # 指定 dtype

# 常用工廠函數
mx.zeros((2, 3))             # 全 0，形狀 (2,3)
mx.zeros_like(x)             # 形狀與 x 相同的全 0 陣列
mx.ones((2, 3))              # 全 1
mx.ones_like(x)              # 形狀與 x 相同的全 1 陣列
mx.full((2, 3), 7.0)        # 全 7.0
mx.eye(4)                    # 4×4 單位矩陣
mx.identity(3)               # 3×3 單位矩陣

# 範圍 / 序列
mx.arange(10)                # array([0, 1, ..., 9], dtype=int32)
mx.arange(0.0, 1.0, 0.1)    # 含步長
mx.linspace(0, 1, 5)        # 5 個均勻點: [0., 0.25, 0.5, 0.75, 1.]

# 特殊常量
mx.inf                      # 正無窮
mx.nan                      # NaN
mx.e                        # e
mx.pi                       # π
mx.newaxis                  # None 的別名，用於擴展維度
```

### 2.2 陣列屬性與檢查

```python
x = mx.array([[1.0, 2.0], [3.0, 4.0]])

x.shape          # (2, 2)
x.dtype          # float32
x.ndim           # 2
x.size           # 4（總元素數）
x.nbytes         # 16（位元組數: 4 × 4 bytes）

# 轉為 Python 數值（強制 eval）
x.item()         # 僅限 0 維陣列
x.tolist()       # → [[1.0, 2.0], [3.0, 4.0]]

# 類型檢查
mx.isnan(x)      # 逐元素
mx.isinf(x)      # 逐元素
mx.isfinite(x)   # 逐元素
mx.isclose(a, b, rtol=1e-5, atol=1e-8)  # 近似相等
mx.allclose(a, b, rtol=1e-5, atol=1e-8) # 全部近似相等
mx.array_equal(a, b)  # 精確相等
```

### 2.3 形狀操作

```python
x = mx.arange(12).reshape(3, 4)   # array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])

# reshape: 改變形狀（不複製資料）
mx.reshape(x, (4, 3))       # → (4, 3)
mx.reshape(x, (-1, 6))      # -1 自動推導 → (2, 6)
mx.flatten(x)               # → (12,)
mx.flatten(x, start_axis=1) # 從第 1 軸開始展平

# 維度增減
mx.expand_dims(x, axis=0)   # (3,4) → (1,3,4)
mx.expand_dims(x, axis=-1)  # (3,4) → (3,4,1)
mx.squeeze(x)               # 移除所有大小為 1 的軸
mx.squeeze(x, axis=0)       # 只移除軸 0（若其大小為 1）

# 轉置 / 維度交換
mx.transpose(x)             # (3,4) → (4,3)
mx.transpose(x, (1, 0))     # 同上
mx.moveaxis(x, 0, 1)       # 將軸 0 移到軸 1
mx.swapaxes(x, 0, 1)       # 交換軸 0 和軸 1
mx.permute_dims(x, (1, 0)) # 依指定順序排列軸（類似 torch.permute）

# 拼接 / 堆疊
mx.concatenate([a, b], axis=0)   # 沿指定軸拼接（類似 np.concatenate）
mx.concat([a, b], axis=0)        # concatenate 的別名
mx.stack([a, b], axis=0)         # 新增軸並堆疊（類似 np.stack）

# 重複 / 平鋪
mx.repeat(x, 3, axis=1)   # 沿軸 1 重複 3 次: (3,4)→(3,12)
mx.tile(x, (2, 1))         # 平鋪: (3,4)→(6,4)
mx.broadcast_to(x, (2, 3, 4))  # 廣播到新形狀
mx.broadcast_arrays(a, b)  # 將多個陣列廣播到共同形狀

# 切片 / 索引
x[0]          # 選取第一列
x[:, 1:3]    # 選取欄 1~2
mx.take(x, mx.array([0, 2]), axis=0)     # 沿軸 0 選取索引 [0,2]
mx.take_along_axis(x, indices, axis=-1)  # 沿指定軸按 indices 選取
mx.put_along_axis(x, indices, values, axis=-1)  # 沿指定軸寫入值

# 對角線
mx.diag(x)        # 提取對角線
mx.diagonal(x)    # 提取對角線（可指定 offset, axis）

# 填充
mx.pad(x, [(0, 0), (2, 2)])     # pad_width: 每個軸 (before, after)
mx.pad(x, [(0, 0), (2, 2)], mode='constant', constant_values=0)
# mode: 'constant', 'edge', 'reflect', 'symmetric'
```

### 2.4 數學運算

```python
x = mx.array([1.0, 2.0, 3.0])
y = mx.array([4.0, 5.0, 6.0])

# 基本四則運算
x + y, x - y, x * y, x / y          # 逐元素
mx.add(x, y), mx.subtract(x, y)
mx.multiply(x, y), mx.divide(x, y)
x @ y                                # 內積（1D）/ 矩陣乘法（2D+）

# 取整 / 取餘
mx.floor(x), mx.ceil(x), mx.round(x)
mx.remainder(x, y)                   # 餘數（類似 Python %）
mx.divmod(x, y)                      # → (商, 餘)
mx.floor_divide(x, y)                # 整數除法

# 冪次 / 根號
mx.power(x, 2.0)                     # x²
mx.sqrt(x)                           # √x
mx.rsqrt(x)                          # 1/√x（RMSNorm 的核心）
mx.square(x)                         # x²
mx.reciprocal(x)                     # 1/x

# 指數 / 對數
mx.exp(x)                            # eˣ
mx.expm1(x)                          # eˣ - 1（x 很小時更精確）
mx.log(x)                            # ln(x)
mx.log2(x)                           # log₂(x)
mx.log10(x)                          # log₁₀(x)
mx.log1p(x)                          # ln(1 + x)
mx.logaddexp(x, y)                   # ln(eˣ + eʸ) — softplus 的基礎

# 三角函數
mx.sin(x), mx.cos(x), mx.tan(x)
mx.arcsin(x), mx.arccos(x), mx.arctan(x)
mx.sinh(x), mx.cosh(x), mx.tanh(x)
mx.arcsinh(x), mx.arccosh(x), mx.arctanh(x)
mx.arctan2(y, x)                     # atan2(y, x)
mx.degrees(x), mx.radians(x)         # 弧度 ↔ 角度轉換

# 誤差函數
mx.erf(x)                            # 誤差函數
mx.erfinv(x)                         # 反誤差函數

# 激活相關（mx.core 中的）
mx.sigmoid(x)                        # 1/(1+e⁻ˣ)
mx.softmax(x, axis=-1)              # softmax 沿指定軸
mx.tanh(x)                           # tanh

# 邏輯運算
mx.logical_and(a, b), mx.logical_or(a, b), mx.logical_not(a)
mx.equal(a, b), mx.not_equal(a, b)
mx.less(a, b), mx.less_equal(a, b)
mx.greater(a, b), mx.greater_equal(a, b)

# 位元運算
mx.bitwise_and(a, b), mx.bitwise_or(a, b)
mx.bitwise_xor(a, b), mx.bitwise_invert(a)
mx.left_shift(a, n), mx.right_shift(a, n)

# 符號 / 截斷
mx.abs(x)                            # |x|
mx.sign(x)                           # 符號: -1, 0, 1
mx.negative(x)                       # -x
mx.clip(x, a_min, a_max)             # 限制範圍

# 特殊值替換
mx.nan_to_num(x, nan=0.0)            # NaN → 0（可指定 posinf, neginf）
```

### 2.5 歸約運算 (Reductions)

```python
# 基本歸約
mx.sum(x)                # 所有元素和
mx.sum(x, axis=0)        # 沿軸 0 求和
mx.sum(x, keepdims=True) # 保持維度
mx.mean(x), mx.mean(x, axis=-1)
mx.max(x), mx.max(x, axis=0)
mx.min(x), mx.min(x, axis=0)
mx.prod(x)               # 乘積
mx.std(x)                # 標準差（除 N，不是 N-1）
mx.var(x)                # 變異數

# 進階歸約
mx.logsumexp(x, axis=-1)       # ln(Σ eˣ)，數值穩定
mx.logcumsumexp(x, axis=-1)    # 累積 logsumexp
mx.median(x, axis=-1)          # 中位數
mx.all(x), mx.any(x)           # 布林歸約

# 累積運算
mx.cumsum(x, axis=-1)          # 累積和
mx.cumprod(x, axis=-1)         # 累積積
mx.cummax(x, axis=-1)          # 累積最大值
mx.cummin(x, axis=-1)          # 累積最小值

# 排序 / 搜索
mx.sort(x, axis=-1)             # 排序
mx.argsort(x, axis=-1)          # 排序索引
mx.argmax(x, axis=-1)           # 最大值索引
mx.argmin(x, axis=-1)           # 最小值索引
mx.argpartition(x, kth=2, axis=-1)  # 部分排序（頂 k 用）
mx.partition(x, kth=2, axis=-1) # 部分排序值
mx.topk(x, k=3, axis=-1)        # 頂 k 個值

# 條件選擇
mx.where(condition, x, y)       # 根據 condition 從 x 或 y 選取
                                # condition: True → x, False → y
```

### 2.6 線性代數（核心）

```python
# 矩陣乘法
mx.matmul(a, b)           # 一般矩陣乘法（支援批次）
a @ b                     # @ 運算符
mx.inner(a, b)            # 內積（最後軸收縮）
mx.outer(a, b)            # 外積

# 批次矩陣乘法
mx.addmm(c, a, b)         # c + a @ b（融合加法和矩陣乘法）
mx.addmm(c, a, b, alpha=2.0, beta=0.5)  # beta*c + alpha*(a@b)

# 愛因斯坦求和
mx.einsum("ij,jk->ik", a, b)           # 矩陣乘法
mx.einsum("bhnp,bhnr->bhpr", a, b)    # 合約 n 維度
mx.einsum("bcijh,bcjhnp->bcihnp", M, u)  # Mamba SSM 掃描的核心
mx.einsum_path("ij,jk->ik", a, b)     # 回傳最優合約路徑

# 特殊矩陣
mx.tri(4)                             # 4×4 下三角布林矩陣
mx.tril(x, k=0)                       # 下三角
mx.triu(x, k=0)                       # 上三角
mx.trace(x)                           # 跡
mx.kron(a, b)                         # 克羅內克積

# 張量收縮
mx.tensordot(a, b, axes=2)           # 沿指定軸數收縮

# Hadamard 變換
mx.hadamard_transform(x)              # 快速沃爾什-哈達瑪變換
```

### 2.7 卷積

```python
# 1D 卷積
mx.conv1d(x, w, stride=1, padding=0, dilation=1, groups=1)
mx.conv_transpose1d(x, w, stride=1, padding=0, dilation=1, groups=1)

# 2D 卷積
mx.conv2d(x, w, stride=1, padding=0, dilation=1, groups=1)
mx.conv_transpose2d(x, w, stride=1, padding=0, dilation=1, groups=1)

# 3D 卷積
mx.conv3d(x, w, stride=1, padding=0, dilation=1, groups=1)
mx.conv_transpose3d(x, w, stride=1, padding=0, dilation=1, groups=1)

# 通用卷積
mx.conv_general(x, w, stride, padding, kernel_dilation, input_dilation, groups, flip, ...)

# 一維卷積（信號處理）
mx.convolve(a, v, mode='full')    # mode: 'full', 'same', 'valid'
```

### 2.8 張量乘法（進階）

```python
# 量化矩陣乘法
mx.quantized_matmul(x, w, scales, biases, ...)

# 分塊遮罩矩陣乘法
mx.block_masked_mm(x, w, mask, ...)

# 分段矩陣乘法
mx.segmented_mm(x, w, segments, ...)

# 聚集矩陣乘法 (Gather MM)
mx.gather_mm(x, w, indices, ...)
mx.gather_qmm(x, w, scales, biases, indices, ...)

# QQ 矩陣乘法
mx.qqmm(x, w, ...)
```

### 2.9 存檔 / 讀取

```python
# MLX 原生格式
mx.savez("model.npz", weight1=w1, weight2=w2)
mx.savez_compressed("model.npz", **weights)  # 壓縮版

# 檔案載入
mx.load("model.npz")         # → dict of arrays (惰性！不立即讀取)
mx.save("tensor.npy", x)     # 單一張量

# Safetensors / GGUF
mx.save_safetensors("model.safetensors", weights)
mx.save_gguf("model.gguf", weights)
```

### 2.10 控制 / 同步

```python
mx.eval(x)                    # 強制執行計算（最重要！）
mx.eval(x, y, z)              # 同時 eval 多個張量
mx.async_eval(x)              # 非同步 eval（不阻塞）
mx.synchronize()              # 等待所有排隊操作完成（相當於 torch.cuda.synchronize）
```

### 2.11 Stream (CUDA Stream 對應)

```python
s = mx.new_stream(mx.gpu)     # 建立新 stream
with mx.stream(s):            # 在此 stream 上執行
    y = x @ x.T
mx.default_stream(mx.gpu)     # 取得預設 stream
mx.set_default_stream(s)      # 設定預設 stream
```

---

## 3. mx.fast — 優化內核

MLX 的 `mx.fast` 模組提供針對特定模式的硬體加速實作。

```python
from mlx.core import fast

# 核心：縮放點積注意力（Flash Attention 等級的 Metal 內核）
fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=None)
# q, k, v: (B, H, L, D)
# scale: 1/sqrt(d_head)
# mask: (L, S) 或 broadcastable，-inf 表示遮罩
# 支援因果遮罩、GQA（k/v heads 和 q heads 可以不同數量）
# Mamba3 中: mx.fast.scaled_dot_product_attention(q, k_exp, v_exp, scale=self.scale, mask=mask)

# 自訂 RMSNorm（比手動實作快）
fast.rms_norm(x, weight, eps)    # 內建 Metal kernel

# LayerNorm 快速版
fast.layer_norm(x, weight, bias, eps)

# RoPE (旋轉位置編碼)
fast.rope(x, theta, axis=-1)     # 內建 RoPE kernel
```

**範例 — Mamba3 中的使用**:
```python
# transformer_block.py 中：
attn = mx.fast.scaled_dot_product_attention(q, k_exp, v_exp, scale=self.scale, mask=mask)
```

---

## 4. mx.random — 隨機數

MLX 使用**無狀態 PRNG**，類似 JAX 的 key-based 系統。

```python
import mlx.core.random as rand

# 建立隨機 key
key = rand.key(42)                         # 從種子建立 key
key1, key2 = rand.split(key)               # 分割 key（每次分割產生不同的獨立 key）
keys = rand.split(key, num=5)              # 分割成 5 個 key

# 隨機分佈
rand.normal(shape=(3, 4), key=key)         # 標準常態 N(0, 1)
rand.uniform(shape=(3, 4), key=key)        # 均勻 U(0, 1)
rand.uniform(shape=(3, 4), low=-1, high=1, key=key)  # U(-1, 1)
rand.randint(shape=(5,), low=0, high=10, key=key)    # 均勻整數
rand.bernoulli(p=0.5, shape=(10,), key=key)            # 伯努利
rand.categorical(logits, axis=-1, key=key)             # 類別分佈（取樣一個類別）
rand.truncated_normal(shape=(3,4), key=key)            # 截斷常態
rand.laplace(shape=(3,4), key=key)                     # 拉普拉斯
rand.gumbel(shape=(5,), key=key)                       # Gumbel
rand.multivariate_normal(mean, cov, shape=(), key=key) # 多元常態

# 排列
rand.permutation(n, key=key)              # 產生 0..n-1 的隨機排列
rand.permutation(x, axis=0, key=key)      # 沿軸隨機排列

# 狀態（不建議用，改用 key-based）
rand.seed(42)                             # 設定全域種子（會影響 key-based 嗎？不會）
rand.state()                              # 取得目前 PRNG 狀態
```

**範例 — Mamba3 中的使用**:
```python
# sampler.py 中：
key = mx.random.key(gen_config.seed)
tok_arr, key = sample_logits(z, temperature, top_k, top_p, min_p, key)
# key 會更新並回傳，確保每一步的隨機性是獨立的
```

---

## 5. mx.linalg — 線性代數

```python
import mlx.core.linalg as la

# 矩陣分解
la.cholesky(A)                     # Cholesky 分解
la.cholesky_inv(L)                 # 從 Cholesky 因子求逆
la.lu(A)                           # LU 分解 → (P, L, U)
la.lu_factor(A)                    # 緊湊 LU → (LU, pivots)
la.qr(A)                           # QR 分解 → (Q, R)
la.svd(A)                          # SVD → (U, S, Vt)

# 特徵值
la.eig(A)                          # 特徵值 + 特徵向量
la.eigh(A)                         # Hermitian 矩陣的特徵值
la.eigvals(A)                      # 僅特徵值
la.eigvalsh(A)                     # Hermitian 矩陣的特徵值

# 逆矩陣 / 解方程
la.inv(A)                          # 逆矩陣
la.pinv(A)                         # 摩爾-彭若斯偽逆
la.tri_inv(A)                      # 三角矩陣逆
la.solve(A, b)                     # 解 Ax = b
la.solve_triangular(A, b)          # 解三角系統

# 向量運算
la.cross(a, b)                     # 叉積
la.norm(x)                         # 向量範數（預設 L2）
la.norm(x, ord='fro')              # Frobenius 範數
```

---

## 6. mx.fft — 傅立葉變換

```python
import mlx.core.fft as ft

# 1D FFT
ft.fft(x)             # 複數 FFT
ft.ifft(x)            # 反 FFT
ft.rfft(x)            # 實數輸入 FFT（只輸出正頻率）
ft.irfft(x)           # 實數反 FFT

# 2D FFT
ft.fft2(x), ft.ifft2(x)
ft.rfft2(x), ft.irfft2(x)

# nD FFT
ft.fftn(x), ft.ifftn(x)
ft.rfftn(x), ft.irfftn(x)

# 頻率平移
ft.fftshift(x)        # 零頻率移到中心
ft.ifftshift(x)       # 反平移
```

---

## 7. mx.metal / GPU — 設備管理

```python
import mlx.core as mx

# 設備資訊
mx.default_device()              # → Device(gpu)
mx.set_default_device(mx.gpu)    # 設為 GPU
mx.set_default_device(mx.cpu)    # 設為 CPU
mx.device_count(mx.gpu)          # GPU 數量
mx.is_available(mx.gpu)          # GPU 是否可用

# 記憶體監控（用於效能調校）
mx.get_active_memory()           # 目前使用中的 GPU 記憶體（bytes）
mx.get_cache_memory()            # 目前快取記憶體
mx.get_peak_memory()             # 迄今為止的峰值記憶體使用量
mx.reset_peak_memory()           # 重置峰值記錄

# 記憶體限制
mx.set_memory_limit(8 * 1024**3)   # 設定 GPU 記憶體上限（8 GB）
mx.set_cache_limit(256 * 1024**2)  # 設定快取上限
mx.set_wired_limit(4 * 1024**3)    # 設定 wired 記憶體上限

# Metal 除錯（效能分析用）
mx.metal.start_capture("trace.gputrace")  # 開始 Metal GPU 追蹤
# ... 執行模型 ...
mx.metal.stop_capture()                   # 停止並儲存到 .gputrace
# 用 Xcode 開啟 trace.gputrace 可檢視每個 kernel 的 GPU 時間

# 清除快取
mx.clear_cache()
```

---

## 8. nn.Module — 模型基底類

```python
import mlx.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 256)     # 註冊為子模組
        self.scale = mx.ones((1,))             # 註冊為參數（會被 train 影響）

    def __call__(self, x):
        return self.linear(x) * self.scale

model = MyModel()

# 參數管理
model.parameters()           # 遞迴收集所有 mx.array 參數 → dict
model.trainable_parameters() # 只收集 trainable 參數
model.update(params_dict)    # 用新值更新參數（用於 optimizer）
model.apply(weights_dict)    # 同 update
model.eval()                 # 評估模式標記（mamba3 中用於控制 cache）
model.train()                # 訓練模式標記

# 凍結參數
model.freeze()               # 凍結所有參數
model.unfreeze()             # 解凍
model.freeze(keys=['linear']) # 只凍結特定模組

# 樹操作
nn.utils.tree_flatten(model.parameters())  # 展平參數樹
```

---

## 9. nn.Linear / Convolution / Embedding

### 9.1 nn.Linear

```python
layer = nn.Linear(input_dims=768, output_dims=1536, bias=True)
# 包含:
#   layer.weight  → mx.array of shape (768, 1536)  ← 注意！不同於 PyTorch！
#   layer.bias    → mx.array of shape (1536,) or None

y = layer(x)     # x: (..., 768) → y: (..., 1536)
# 執行: y = x @ layer.weight + layer.bias
```

**重要**: MLX 的 `nn.Linear.weight` 形狀是 `(input_dims, output_dims)`，而 PyTorch 是 `(output_dims, input_dims)`。這在從 PyTorch checkpoint 載入時需要轉置。

### 9.2 nn.Embedding

```python
emb = nn.Embedding(num_embeddings=32007, dims=768)
# emb.weight: (32007, 768)

y = emb(input_ids)      # input_ids: (B, L) int32 → y: (B, L, 768)
```

### 9.3 nn.Conv1d / Conv2d / Conv3d

```python
# 1D: (N, C_in, L) → (N, C_out, L')
conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

# 2D: (N, C_in, H, W) → (N, C_out, H', W')
conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)

# 轉置卷積
tconv = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1)
```

### 9.4 nn.Bilinear

```python
bilinear = nn.Bilinear(input_dims1=128, input_dims2=64, output_dims=32, bias=True)
y = bilinear(x1, x2)   # x1: (B,128), x2: (B,64) → y: (B,32)
```

---

## 10. nn.Normalization — 正規化層

### 10.1 RMSNorm（Mamba3 核心）

```python
norm = nn.RMSNorm(dims=768, eps=1e-5)
# norm.weight: (768,)

y = norm(x)  # x: (..., 768) → y: (..., 768)
# 數學: x * rsqrt(mean(x²) + eps) * weight
```

**快速版 — `mx.fast.rms_norm`（專用 Metal kernel）**:

MLX 內建 `mx.fast.rms_norm` 是專用 Metal kernel 實作，比 `nn.RMSNorm` 或手動實作更快。Mamba3 中已採用此模式（`ops.py` 第 3-6 行）：

```python
# ops.py — Mamba3 實際使用的模式
import mlx.core as mx

_fast_rms_norm = getattr(mx.fast, "rms_norm", None)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        if _fast_rms_norm is not None:
            # 使用 MLX 內建 Metal kernel — 零 Python 開銷
            return _fast_rms_norm(x, self.weight, self.eps)
        # 舊版 MLX 降級：手動 float32 實作
        f = x.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(f * f, axis=-1, keepdims=True) + self.eps)
        return (f * rms).astype(x.dtype) * self.weight.astype(x.dtype)
```

**與 `nn.RMSNorm` 的差異**: `mx.fast.rms_norm(x, weight, eps)` 是**函數**而非模組 — 傳入 weight 和 eps 作為參數。比 `nn.RMSNorm` 更快，因為繞過了模組包裝層。

**其他快速正規化內核**:
```python
mx.fast.layer_norm(x, weight, bias, eps)    # 專用 LayerNorm Metal kernel
mx.fast.rope(x, theta, axis=-1)             # 專用 RoPE Metal kernel
```

### 10.2 其他正規化

```python
nn.LayerNorm(dims=768, eps=1e-5)     # LayerNorm
nn.BatchNorm(num_features=64)        # BatchNorm
nn.GroupNorm(num_groups=4, dims=128) # GroupNorm
nn.InstanceNorm(dims=64)             # InstanceNorm
```

**對比**: MLX 的 `nn.RMSNorm` 只接受最後一維的 `dims`，不像 PyTorch 的 `nn.RMSNorm(normalized_shape)` 可以處理多維。

---

## 11. nn.Activations — 激活函數

```python
# 作為模組使用（用於 Sequential）
nn.ReLU(), nn.ReLU6(), nn.ReLU2()
nn.LeakyReLU(negative_slope=0.01)
nn.PReLU(num_parameters=768)         # 可學習的 LeakyReLU
nn.ELU(alpha=1.0)
nn.CELU(alpha=1.0)
nn.SELU()
nn.GELU()                            # Gaussian Error Linear Unit
nn.GLU()                             # Gated Linear Unit
nn.SiLU()                            # Sigmoid Linear Unit (= Swish)
nn.Mish()                            # Mish: x * tanh(softplus(x))
nn.Sigmoid()
nn.Tanh()
nn.Softplus()
nn.Softsign()
nn.Hardswish()
nn.HardTanh(min_val=-1, max_val=1)
nn.HardShrink(lambd=0.5)
nn.Softshrink(lambd=0.5)
nn.LogSigmoid()
nn.LogSoftmax()
nn.Softmax(dim=-1)                   # softmax 沿指定軸
nn.Softmin(dim=-1)                   # softmin
nn.Step(threshold=0.0, value=1.0)   # 階梯函數

# 對應的函數版本（直接呼叫，無狀態）:
import mlx.nn.activations as act
act.relu(x), act.silu(x), act.gelu(x), act.mish(x), ...
```

**Mamba3 中的使用**:
```python
# ops.py — 自訂義激活:
silu(x)       = x * mx.sigmoid(x)
softplus(x)   = mx.logaddexp(x, mx.zeros_like(x))
scaled_tanh(x, scale=10.0) = tanh_approx(x / scale) * scale
```

---

## 12. nn.Transformer / Attention

### 12.1 MultiHeadAttention

```python
mha = nn.MultiHeadAttention(dims=768, num_heads=12, bias=True)

# 輸出模式:
y = mha(x)                           # self-attention
y = mha(query, key, value)           # cross-attention
y = mha(query, key, value, mask=mask) # 附遮罩

# mask 形狀: (seq_len, seq_len) 或 broadcastable
# mask 值: -inf 表示不允許注意
```

### 12.2 Transformer / TransformerEncoder / TransformerDecoder

```python
# 完整的 Transformer
transformer = nn.Transformer(
    dims=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6
)
y = transformer(src, tgt)

# 僅編碼器
encoder_layer = nn.TransformerEncoderLayer(dims=768, num_heads=12, mlp_dims=3072)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
y = encoder(x, mask=src_mask)

# 僅解碼器
decoder_layer = nn.TransformerDecoderLayer(dims=768, num_heads=12, mlp_dims=3072)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
y = decoder(x, memory, tgt_mask=None, memory_mask=None)
```

### 12.3 RoPE (旋轉位置編碼)

```python
rope = nn.RoPE(dims=64, traditional=False, base=10000.0)
y = rope(x)  # 對最後兩個維度應用 RoPE

# Mamba3 中用手動實作的 apply_rope:
# def apply_rope(x, angles):
#     x_r = x.reshape(*x.shape[:-2], N_half, 2, R)
#     x1, x2 = x_r[...,0,:], x_r[...,1,:]
#     sin, cos = mx.sin(angles), mx.cos(angles)
#     r1 = x1*cos - x2*sin; r2 = x2*cos + x1*sin
#     return mx.stack([r1, r2], axis=-2).reshape(*x.shape)
```

### 12.4 位置編碼

```python
nn.SinusoidalPositionalEncoding(dims=768, min_freq=1e-5, max_freq=1.0)
nn.ALiBi(num_heads=12)  # Attention with Linear Biases
```

---

## 13. nn.Recurrent — RNN / LSTM / GRU

```python
# 單層 RNN/LSTM/GRU
rnn  = nn.RNN(input_dims=128, hidden_dims=256, num_layers=2, bias=True, dropout=0.1)
lstm = nn.LSTM(input_dims=128, hidden_dims=256, num_layers=2)
gru  = nn.GRU(input_dims=128, hidden_dims=256, num_layers=2)

# forward: (input, state) → (output, next_state)
y, new_state = lstm(x)                    # x: (seq, batch, dims)
y, (h, c) = lstm(x, (h0, c0))            # 指定初始狀態
```

---

## 14. nn.Pooling / Dropout / Upsample

```python
# 池化
nn.MaxPool1d(kernel_size=2, stride=2)
nn.MaxPool2d(kernel_size=2, stride=2)
nn.MaxPool3d(kernel_size=2, stride=2)
nn.AvgPool1d(kernel_size=2)
nn.AvgPool2d(kernel_size=2)
nn.AvgPool3d(kernel_size=2)

# Dropout
nn.Dropout(p=0.1)            # 僅訓練模式有效
nn.Dropout2d(p=0.1)
nn.Dropout3d(p=0.1)

# 上取樣
nn.Upsample(scale_factor=2.0, mode='nearest')    # mode: 'nearest', 'linear'
nn.Upsample(scale_factor=(2, 2), mode='linear')  # 2D 上取樣
```

---

## 15. nn.Quantized — 量化層

```python
# 量化線性層（減少記憶體、加速推理）
ql = nn.QuantizedLinear(input_dims=768, output_dims=1536, group_size=64, bits=4, bias=True)
y = ql(x)

# 量化 Embedding
qe = nn.QuantizedEmbedding(num_embeddings=32007, dims=768, group_size=64, bits=4)

# 分散式量化
nn.QQLinear(input_dims, output_dims, group_size, bits, ...)
nn.AllToShardedLinear(...)
nn.QuantizedAllToShardedLinear(...)
nn.QuantizedShardedToAllLinear(...)
nn.ShardedToAllLinear(...)
```

---

## 16. nn.init — 權重初始化

```python
from mlx.nn.init import *

w = mx.zeros((768, 1536))

# 常數 / 常態 / 均勻
constant(w, 0.0)
normal(w, mean=0.0, std=0.02)
uniform(w, low=-0.1, high=0.1)

# Xavier / Glorot
glorot_normal(w)       # Xavier 常態
glorot_uniform(w)      # Xavier 均勻

# He / Kaiming
he_normal(w)           # Kaiming 常態
he_uniform(w)          # Kaiming 均勻

# 單位矩陣 / 正交 / 稀疏
identity(w)            # 填入單位矩陣（用於 square 矩陣）
orthogonal(w)          # 正交初始化
sparse(w, sparsity=0.1, std=0.01)  # 稀疏初始化
```

---

## 17. nn.losses — 損失函數

```python
import mlx.nn.losses as L

# 分類 / 交叉熵
L.cross_entropy(logits, targets)          # targets: class indices
L.binary_cross_entropy(inputs, targets)   # 二元交叉熵（帶 logits 版本？用 reduction）
L.nll_loss(log_probs, targets)            # 負對數似然

# 回歸
L.mse_loss(pred, target)                  # 均方誤差
L.l1_loss(pred, target)                   # L1 損失
L.smooth_l1_loss(pred, target)            # Smooth L1 (Huber 的變體)
L.huber_loss(pred, target, delta=1.0)     # Huber 損失

# 比對 / 相似度
L.cosine_similarity_loss(x1, x2)          # 餘弦相似度損失
L.triplet_loss(anchor, positive, negative, margin=1.0)
L.margin_ranking_loss(x1, x2, target, margin=0.0)
L.hinge_loss(inputs, targets)             # 鉸鏈損失

# KL 散度 / 高斯
L.kl_div_loss(log_probs, targets)
L.gaussian_nll_loss(input, target, var)   # 高斯負對數似然

# 通用參數
L.cross_entropy(logits, targets, reduction=L.Reduction.MEAN)
# reduction: 'none', 'mean', 'sum'
```

---

## 18. mx.compile — JIT 編譯

`mx.compile` 將 Python 模型轉換為**優化的 Metal 計算圖**，消除 Python 開銷和 kernel launch 延遲。

```python
# 基本用法
compiled_model = mx.compile(model)

# 第一次呼叫觸發編譯（較慢），後續呼叫使用快取圖
logits, states = compiled_model(ids, states=states)   # warmup / 編譯
logits, states = compiled_model(ids, states=states)   # 快速！使用已編譯圖

# 條件：每次呼叫的形狀必須一致（或編譯多個變體）
# 不支援動態形狀的 compile

# 在 Mamba3 中的用法:
# generator.py:
def compile_model(model):
    return mx.compile(model)

# run.py: --full-decode-compile 旗標
```

**重要限制**:
- 輸入張量形狀在 compile 後不可改變（靜態圖）
- 適合解碼迴圈（每次 L=1）或固定長度的 prefill
- 第一次執行包含編譯開銷 (warmup)
- 不支援帶有 Python 控制流（if/for）的模型 — 會將分支內聯

```python
# 編譯控制
mx.enable_compile()       # 全域啟用（一般不建議，顯式 compile 較好）
mx.disable_compile()      # 全域禁用

# 不編譯特定操作（用於除錯）
mx.compile(fn, inputs=[x], outputs=[y])  # 指定輸入/輸出形狀

# 匯出圖（用於檢查）
mx.export_to_dot(fn, "graph.dot")
mx.export_function(fn, "exported.mlxfn")

# 函數導出 / 匯入
mx.exporter.export(fn, ...)
mx.import_function("exported.mlxfn")
```

---

## 19. mx.custom_function — 自定義 Metal 內核

`mx.custom_function` 是連接 Python 和自定義 Metal 內核的橋樑。

```python
# 範例：自定義 fused 運算
def my_fused_op(x, w, bias):
    # x: (B, D), w: (D, Out), bias: (Out,)
    out = mx.zeros((x.shape[0], w.shape[1]), dtype=x.dtype)
    
    # 使用 Metal 內核（用字串名稱引用）
    y = mx.custom_function(
        "my_metal_kernel_name",          # Metal 函數名
        inputs=[x, w, bias],             # 輸入張量列表
        outputs=[out],                   # 輸出張量列表（預先分配）
        stream=mx.default_stream(mx.gpu) # 可選：指定 stream
    )
    return y[0]  # 回傳第一個輸出

# 對應的 Metal 內核（.metal 檔案中）:
# kernel void my_metal_kernel_name(
#     device const half* x     [[buffer(0)]],
#     device const half* w     [[buffer(1)]],
#     device const half* bias  [[buffer(2)]],
#     device half* out         [[buffer(3)]],
#     uint tid [[thread_position_in_grid]]
# ) { ... }

# 使用 FunctionExporter（較新的方式）
exporter = mx.FunctionExporter()
exporter.add_function(my_fused_op, inputs=[x, w], outputs=[y])
exporter.export("fused_op.mlxfn")
```

**Mamba3 中的自定義內核範例**（優化後）:
```python
# fused_moe_decode.py — 完整的 TuckerMoE fused kernel
def tucker_moe_decode_kernel(x, router_w, U_in, norm_w, norm_eps,
                              G_experts, U_out, bias, temperature=0.5):
    """
    在一個 Metal kernel 中完成整個 TuckerMoE 解碼計算。
    僅用於 decode (B_flat=1)。
    """
    B_flat = x.shape[0]
    dim_in = x.shape[1]
    dim_out = U_out.shape[1]
    
    out = mx.zeros((B_flat, dim_out), dtype=x.dtype)
    
    result = mx.custom_function(
        "tucker_moe_decode_v1",
        inputs=[
            x, router_w, U_in, norm_w,
            mx.array([norm_eps, temperature], dtype=mx.float32),
            G_experts, U_out, bias
        ],
        outputs=[out],
    )
    return result[0]
```

---

## 20. mx.grad / vjp / jvp — 自動微分

```python
# 純量輸出的梯度
def loss_fn(params, x, y):
    pred = model_fn(params, x)
    return mx.mean((pred - y) ** 2)

grad_fn = mx.grad(loss_fn)           # 對第一個參數求梯度
grads = grad_fn(params, x, y)

# value_and_grad: 同時取得值和梯度
loss_value, grads = mx.value_and_grad(loss_fn)(params, x, y)

# 對特定參數求梯度
grad_fn = mx.grad(loss_fn, argnums=0)   # 對參數 0 求梯度（預設）

# VJP (Vector-Jacobian Product) — 反向模式 AD
primals, vjp_fn = mx.vjp(fn, primals)
cotangent = mx.ones_like(primals[0])
vjp_result = vjp_fn(cotangent)

# JVP (Jacobian-Vector Product) — 前向模式 AD
primals, tangents = ...
jvp_result, jvp_out = mx.jvp(fn, primals, tangents)

# 中斷梯度
mx.stop_gradient(x)     # 等同 torch.no_grad / jax.lax.stop_gradient

# 梯度檢查點（重計算）
mx.checkpoint(fn)       # 不儲存中間值，反向時重計算
```

---

## 21. mx.vmap — 自動向量化

```python
# 將純量函數向量化到批次維度
def scalar_fn(x):      # x: (D,) → y: (D,)
    return x * 2 + 1

batch_fn = mx.vmap(scalar_fn, in_axes=0, out_axes=0)
# batch_fn(x)   # x: (B, D) → y: (B, D)

# 多個參數
def fn(a, b):
    return a + b

vfn = mx.vmap(fn, in_axes=(0, None))  # a 沿軸 0 batch, b 不是 batch
# vfn(A, b)  # A: (B, N), b: (N,) → (B, N)
```

---

## 22. mx.distributed — 分散式訓練

```python
import mlx.core.distributed as dist

# 初始化
dist.init()              # 初始化分散式後端

# 群組
group = dist.Group([0, 1, 2, 3])  # 建立通訊群組

# 集合通訊
dist.all_gather(x)       # 所有 GPU 收集資料
dist.all_sum(x)           # 所有 GPU 求和
dist.all_max(x)           # 所有 GPU 取最大值
dist.all_min(x)           # 所有 GPU 取最小值
dist.sum_scatter(x, ...)  # 求和後分發

# 點對點通訊
dist.send(x, dst=1)
dist.recv(shape, dtype=dtype, src=0)
dist.recv_like(x, src=0)   # 接收形狀和 dtype 與 x 相同的張量

# 檢查
dist.is_available()       # 分散式是否可用
```

---

## 23. mlx.utils — 樹操作工具

```python
import mlx.utils as mu

# tree_flatten: 展平巢狀結構
params = {'layer1': {'w': w1, 'b': b1}, 'layer2': {'w': w2}}
flat = mu.tree_flatten(params)
# → [('layer1.w', w1), ('layer1.b', b1), ('layer2.w', w2)]

# tree_unflatten: 從展平列表重建
mu.tree_unflatten(flat)
# → {'layer1': {'w': w1, 'b': b1}, 'layer2': {'w': w2}}

# tree_map: 對樹中每個葉子應用函數
mu.tree_map(lambda x: x.astype(mx.float16), params)

# tree_map_with_path: 同時提供路徑
def scale_by_path(path, value):
    if 'bias' in path: return value
    return value * 0.5
mu.tree_map_with_path(scale_by_path, params)

# tree_reduce: 對所有葉子歸約
mu.tree_reduce(lambda acc, x: acc + x.item(), params, 0.0)

# tree_merge: 合併兩棵樹
# (用於 optimizer update: model.update(tree_merge(grads, ...)))

# 輔助
mu.zip_longest(*iterables)
```

---

## 24. Mamba3-XR 實際使用模式

以下是 Mamba3 codebase 中 MLX API 的實際使用範例。

### 24.1 模型定義模式

```python
# hybrid_model.py
class Mamba3LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.backbone = TrueHybridMamba(config)
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # head weight 與 embed.weight 綁定 (tied weights)

    def __call__(self, input_ids, states=None):
        x = self.embed(input_ids)
        x, new_states = self.backbone(x, states=states)
        h = self.norm(x)
        logits = self.head(h * self.inv_sqrt_d).astype(mx.float32)
        logits = scaled_tanh(logits, 30.0)
        return logits, new_states
```

### 24.2 KV / Mamba Cache 模式

```python
# decode 時，每層維護一個 state dict
# Mamba 層:
mamba_state = {
    "h_prev": mx.zeros((B, H, N, P), dtype),        # SSM 狀態
    "prev_input_signal": mx.zeros((B, H, N, P), dtype), # 前一個 input signal
    "angles_cum": mx.zeros((B, H, N//2), dtype),     # 累積 RoPE 角度
}

# Transformer 層:
tf_state = {
    "k": mx.zeros((B, num_kv_heads, S_past, head_dim), dtype),  # KV cache
    "v": mx.zeros((B, num_kv_heads, S_past, head_dim), dtype),
}
```

### 24.3 Lazy Eval 與 mx.eval 模式

```python
# generator.py — 正確的 eval 模式
def prefill(model, prompt_ids):
    ids = mx.array(prompt_ids, dtype=mx.int32)[None, :]
    t0 = time.perf_counter()
    logits, states = model(ids, states=None)
    mx.eval(logits, *_iter_state_arrays(states))  # ← 強制執行！
    elapsed = time.perf_counter() - t0
    return logits[0, -1], states, elapsed

# decode loop:
for step in range(remaining):
    tok_id = sample(logits, ...)
    ids = mx.array([[tok_id]], dtype=mx.int32)
    logits_out, states = model(ids, states=states)
    logits = logits_out[0, -1]        # ← 只取最後一個 token 的 logits
    mx.eval(logits, *_iter_state_arrays(states))  # ← 強制執行每一步
```

### 24.4 Sampling 模式

```python
# sampler.py
def sample_logits(logits, temperature, top_k, top_p, min_p, key):
    if temperature > 0:
        logits = logits * (1.0 / temperature)
    
    if top_k > 0:
        # mx.topk 回傳 (values, indices)
        top_vals, _ = mx.topk(logits, k=top_k, axis=-1)
        threshold = top_vals[..., -1:]
        logits = mx.where(logits < threshold, -mx.inf, logits)
    
    if top_p < 1.0:
        sorted_idx = mx.argsort(logits, axis=-1)[..., ::-1]
        sorted_logits = mx.take_along_axis(logits, sorted_idx, axis=-1)
        cum_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
        mask = cum_probs > top_p
        sorted_logits = mx.where(mask, -mx.inf, sorted_logits)
        # ... 取消排序 ...
    
    # 使用 key-based 隨機取樣
    tok = rand.categorical(logits, axis=-1, key=key)
    return tok, rand.split(key)[0]  # 回傳 token 和新 key
```

### 24.5 Compile + Warmup 模式

```python
# generator.py
def generate(model, prompt_ids, gen_config, full_decode_compile=False, warmup_steps=3):
    # 1. prefill (不 compile — 只需一次)
    last_logits, states, elapsed_prefill = prefill(model, prompt_ids)
    
    if full_decode_compile:
        decode_model = mx.compile(model)    # 編譯整個模型
        
        # 2. warmup: 用 greedy sampling 觸發 compile
        for _ in range(warmup_steps):
            tok_id = int(mx.argmax(last_logits).item())  # greedy，無亂數
            ids = mx.array([[tok_id]], dtype=mx.int32)
            logits_out, states = decode_model(ids, states=states)
            last_logits = logits_out[0, -1]
            mx.eval(last_logits, *_iter_state_arrays(states))
        
        # 3. 正式 decode: 使用已編譯的圖，速度快
        for step in range(remaining):
            tok_id = sample(logits, key)
            ids = mx.array([[tok_id]], dtype=mx.int32)
            logits_out, states = decode_model(ids, states=states)
            logits = logits_out[0, -1]
            mx.eval(logits, *_iter_state_arrays(states))
```

### 24.6 自定義運算模式

```python
# ops.py — Mamba3 自定義的激活函數
def tanh_approx(x):
    """快速 tanh 近似: 2*sigmoid(2x) - 1"""
    return 2.0 * mx.sigmoid(2.0 * x) - 1.0

def scaled_tanh(x, scale=10.0):
    return tanh_approx(x * (1.0 / scale)) * scale

def softplus(x):
    return mx.logaddexp(x, mx.zeros_like(x))

def apply_rope(x, angles):
    """自定義 RoPE — 直接操作最後兩個維度"""
    N_half = angles.shape[-1]
    R = x.shape[-1]
    x_r = x.reshape(*x.shape[:-2], N_half, 2, R)
    x1, x2 = x_r[..., 0, :], x_r[..., 1, :]
    sin_a, cos_a = mx.sin(angles)[..., None], mx.cos(angles)[..., None]
    r1 = x1 * cos_a - x2 * sin_a
    r2 = x2 * cos_a + x1 * sin_a
    return mx.stack([r1, r2], axis=-2).reshape(*x.shape)
```

### 24.7 einsum 密集使用模式

```python
# mamba_block.py — 5 維 / 6 維 einsum
B_rot_rot = apply_rope(B_p, angles)           # (B, L, H, N, R)
x_ssm = x_up.reshape(B_sz, L, H, P, R)        # (B, L, H, P, R)

# 合約 R 維度 (B 和rot 的 R)
input_signal = mx.einsum("blhnr,blhpr->blhnp", B_rot, x_ssm)
# → (B, L, H, N, P)

# SSM chunk scan
h_intra = mx.einsum("bcijh,bcjhnp->bcihnp", M, u_c)
# M: (B, nc, Lc, Lc, H), u_c: (B, nc, Lc, H, N, P)
# → h_intra: (B, nc, Lc, H, N, P)

y_diag = mx.einsum("bclhnp,bclhnr->bclhpr", h_intra, C_c)
y_off  = mx.einsum("bchnp,bclhnr->bclhpr", h_inter, c_dec)
```

### 24.8 TuckerMoE einum 模式

```python
# tucker_moe.py
# Tucker 分解: 專家矩陣 = U_expert @₁ core
G_experts = mx.einsum("er,rst->est",
    U_expert,     # (E=8, r1=32)
    core)         # (r1=32, r3=256, r2=512)
# → (E=8, r3=256, r2=512)

# 每個 batch element 的 expert 輸出
per_expert = mx.einsum("br,bkrs->bks",
    x_shared,     # (B_flat, r3=256)
    G_selected)   # (B_flat, k=2, r3=256, r2=512)
# → (B_flat, k=2, r2=512)
```

---

## 25. PyTorch ↔ MLX 快速對照表

| 操作 | PyTorch | MLX | 註記 |
|------|---------|-----|------|
| 創建陣列 | `torch.tensor([1,2])` | `mx.array([1,2])` | |
| 全零 | `torch.zeros(3,4)` | `mx.zeros((3,4))` | MLX 傳 tuple |
| 全一 | `torch.ones(3,4)` | `mx.ones((3,4))` | |
| 形狀 | `x.shape` / `x.size()` | `x.shape` | |
| dtype | `x.dtype` | `x.dtype` | |
| 轉型 | `x.to(torch.float32)` | `x.astype(mx.float32)` | |
| 設備轉移 | `x.to('cuda')` | 不需（統一記憶體） | |
| reshape | `x.reshape(2,3)` | `mx.reshape(x, (2,3))` | |
| view | `x.view(2,3)` | `mx.reshape(x, (2,3))` | MLX 無 view 概念 |
| 轉置 | `x.T` / `x.transpose(0,1)` | `x.T` / `mx.transpose(x)` | |
| permute | `x.permute(2,0,1)` | `mx.transpose(x, (2,0,1))` | MLX 無 permute，用 transpose |
| 矩陣乘法 | `a @ b` / `torch.matmul(a,b)` | `a @ b` / `mx.matmul(a,b)` | |
| einsum | `torch.einsum('ij,jk->ik', a,b)` | `mx.einsum('ij,jk->ik', a,b)` | 語法相同 |
| softmax | `F.softmax(x, dim=-1)` | `mx.softmax(x, axis=-1)` | `dim` → `axis` |
| 拼接 | `torch.cat([a,b], dim=0)` | `mx.concatenate([a,b], axis=0)` | |
| 堆疊 | `torch.stack([a,b], dim=0)` | `mx.stack([a,b], axis=0)` | |
| 重複 | `x.repeat(2,3)` | `mx.repeat(x, 3, axis=1)` | 語法不同 |
| exp | `torch.exp(x)` | `mx.exp(x)` | 同 |
| sigmoid | `torch.sigmoid(x)` | `mx.sigmoid(x)` | 同 |
| sum | `x.sum(dim=0)` | `mx.sum(x, axis=0)` | |
| mean | `x.mean(dim=-1)` | `mx.mean(x, axis=-1)` | |
| cumsum | `x.cumsum(dim=1)` | `mx.cumsum(x, axis=1)` | |
| argsort | `x.argsort(dim=-1)` | `mx.argsort(x, axis=-1)` | |
| topk | `x.topk(k=5)` | `mx.topk(x, k=5, axis=-1)` | MLX 回傳 (values, indices) |
| where | `torch.where(cond, a, b)` | `mx.where(cond, a, b)` | 同 |
| clip | `torch.clip(x, min, max)` | `mx.clip(x, min, max)` | 同 |
| 評估 | `torch.no_grad()` 或 `@torch.no_grad()` | `mx.stop_gradient(x)` | 語法不同 |
| Linear | `nn.Linear(768, 256)` | `nn.Linear(768, 256)` | MLX weight: (in,out)，PT: (out,in) |
| Embedding | `nn.Embedding(V, D)` | `nn.Embedding(V, D)` | 同 |
| RMSNorm | `nn.RMSNorm(D)` / 自定義 | `nn.RMSNorm(D, eps)` | MLX 內建 |
| LayerNorm | `nn.LayerNorm(D)` | `nn.LayerNorm(D, eps)` | |
| 梯度 | `torch.autograd.grad()` | `mx.grad(fn)` | 相似的函數式 API |
| compile | `torch.compile(model)` | `mx.compile(model)` | 相似的 JIT |
| eval | (即刻執行) | `mx.eval(x)` | MLX 強制執行 lazy 圖 |
| 同步 | `torch.cuda.synchronize()` | `mx.synchronize()` | |
| 隨機數 | `torch.manual_seed(42)` | `rand.key(42)` / `rand.split(key)` | MLX 是 JAX 式無狀態 |
| 存檔 | `torch.save(model.state_dict())` | `mx.savez("m.npz", **params)` | |
| 載入 | `torch.load(...)` | `mx.load(...)` | MLX 惰性載入 |

---

## 附錄 A: 常用 MLX 型態速查

```python
# dtype 簡寫
mx.float16, mx.float32, mx.bfloat16, mx.complex64
mx.int8, mx.int16, mx.int32, mx.int64
mx.uint8, mx.uint16, mx.uint32, mx.uint64
mx.bool_

# dtype 類型群組
mx.floating         # 所有浮點數型態
mx.integer          # 所有整數
mx.inexact          # float + complex
mx.number           # 所有數值型態
mx.signedinteger    # 所有有符號整數
mx.unsignedinteger  # 所有無符號整數
mx.generic          # 所有 dtype
mx.complexfloating  # 所有複數

# dtype 資訊
mx.finfo(mx.float32)   # 浮點數資訊 (min, max, eps, ...)
mx.iinfo(mx.int32)      # 整數資訊 (min, max, bits)
mx.issubdtype(dt, mx.floating)  # 類型檢查
```

## 附錄 B: MLX 與 Mamba3 相容性註記

- **Weight 格式**: Mamba3 checkpoint 是 PyTorch `.npz`，weight 形狀與 MLX 不同。`weights.py` 處理轉換。
- **Lazy eval**: 在不呼叫 `mx.eval()` 時，陣列只是一個計算圖節點。Mamba3 在 `generator.py` 中使用 `mx.eval()` 來控制執行時機。
- **Sidecar**: Mamba3 首次載入時建立 `.mlx_bf16.npz` sidecar 以加速後續載入（mmap）。
- **State dict**: `model.parameters()` 回傳巢狀 dict，類似 PyTorch `state_dict()`，但 key 名稱不同。
- **KV cache**: 需要手動管理（`concatenate` 到現有 cache 上），MLX Transformer 內建支援但 Mamba3 使用自定義 cache。

---

*文件版本: v1.0 — 涵蓋 MLX 全部公開 API，並附 Mamba3-XR 實際使用範例*  
*基於 MLX (Apple Silicon) 最新版*
