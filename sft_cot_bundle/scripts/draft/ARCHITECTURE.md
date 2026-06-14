# Draft Model Architecture — MLX Porting Guide

純 Flash-Attention Transformer，為 Speculative Decoding 設計。  
此文件提供足夠細節以在 Apple MLX 上重建相同架構。

---

## 1. 超參數 (DraftConfig)

| 參數 | 值 | 說明 |
|---|---|---|
| `d_model` | 256 | 隱藏層維度 |
| `n_layers` | 6 | Transformer block 數 |
| `n_heads` | 8 | Query head 數 |
| `n_kv_heads` | 2 | KV head 數（GQA 4:1 ratio） |
| `head_dim` | 32 | `d_model / n_heads = 256/8` |
| `ffn_mult` | 3 | SwiGLU hidden = `d_model × ffn_mult = 768` |
| `max_seq` | 768 | RoPE 預計算長度（訓練用 512，生成緩衝）|
| `vocab_size` | 32007 | 含自定義 special tokens |
| `rope_base` | 10000 | RoPE θ base |

**參數量：12.72M**（embed+head tied，計一次）

---

## 2. 整體結構

```
input_ids (B, T)
    │
    ▼
Embedding  →  (B, T, 256)
    │
    ▼  ×6
┌─────────────────────────────┐
│  RMSNorm                    │
│  GQAttention (Q/K/V/O proj) │  ← KV Cache 在這層
│  + residual                 │
│  RMSNorm                    │
│  SwiGLU FFN                 │
│  + residual                 │
└─────────────────────────────┘
    │
    ▼
RMSNorm  →  Linear head  →  logits (B, T, 32007)
```

**所有 Linear 無 bias。**

---

## 3. 各層 Tensor Shape

### Embedding
```
embed.weight : (32007, 256)   # lookup: (B,T) → (B,T,256)
```

### Per-Block（以 layer i 為例）

```
norm1.weight : (256,)

attn.q.weight : (256, 256)    # W_Q : d_model → n_heads × head_dim
attn.k.weight : ( 64, 256)    # W_K : d_model → n_kv_heads × head_dim  (64=2×32)
attn.v.weight : ( 64, 256)    # W_V : 同上
attn.o.weight : (256, 256)    # W_O : d_model → d_model

norm2.weight : (256,)

ffn.gate.weight : (768, 256)  # SwiGLU gate branch
ffn.up.weight   : (768, 256)  # SwiGLU up branch
ffn.down.weight : (256, 768)  # 投影回 d_model
```

### 最終層
```
norm.weight  : (256,)
head.weight  : (32007, 256)   # ← 與 embed.weight 是同一塊（tied）
```

---

## 4. 關鍵操作細節

### 4.1 RMSNorm
```
y = x / rms(x) * weight,   rms(x) = sqrt(mean(x²) + ε)
ε = 1e-6（PyTorch 預設）
weight shape: (d_model,)，初始化為 1
```

### 4.2 RoPE（Rotary Position Embedding）

```python
# head_dim=32, 故 inv_freq 有 16 個頻率
inv_freq[i] = 1 / (10000 ** (2i / 32)),  i = 0..15

# 對位置 t，head_dim/2 個 sin/cos 對：
cos[t, 2i  ] = cos[t, 2i+1] = cos(t * inv_freq[i])
sin[t, 2i  ] = sin[t, 2i+1] = sin(t * inv_freq[i])

# 旋轉：x shape (B, H, T, 32)
h = 16
x_rot = concat([-x[..., h:], x[..., :h]], dim=-1)
out = x * cos + x_rot * sin
```

**帶 offset（KV cache decode 用）：**
```python
# offset = 目前 cache 中 token 數
cos = cos_table[offset : offset + T]
sin = sin_table[offset : offset + T]
```

RoPE 參數**不在 state_dict** 中（`persistent=False`），MLX 重建時直接算。

### 4.3 GQA（Grouped Query Attention）

```
Q: (B, n_heads,    T, head_dim) = (B, 8, T, 32)
K: (B, n_kv_heads, T, head_dim) = (B, 2, T, 32)
V: (B, n_kv_heads, T, head_dim) = (B, 2, T, 32)

每個 KV head 被 4 個 Query head 共用。
PyTorch: F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
MLX:     mx.fast.scaled_dot_product_attention(q, k, v, scale, mask)
         (MLX 會自動 broadcast KV)
```

**is_causal 規則：**
- Prefill（無 past_kv，T > 1）：`is_causal = True`
- Decode（有 past_kv，T = 1）：`is_causal = False`

### 4.4 SwiGLU FFN

```python
hidden = silu(gate(x)) * up(x)   # (B, T, 768)
out    = down(hidden)              # (B, T, 256)

# gate.weight : (768, 256)
# up.weight   : (768, 256)
# down.weight : (256, 768)
```

### 4.5 Weight Tying

```
head.weight IS embed.weight（同一記憶體位址）
```

**state_dict 的陷阱：** PyTorch `state_dict()` 會同時儲存 `embed.weight` 和 `head.weight`（相同內容）。  
MLX 轉換時只需載入 `embed.weight`，head 直接引用同一矩陣，**不要載入兩次**。

---

## 5. KV Cache 格式

```python
KVCache = list[tuple[Tensor, Tensor]]   # 長度 = n_layers = 6

kv_cache[i] = (
    k_i,   # (B, n_kv_heads, T_cached, head_dim) = (B, 2, T, 32)
    v_i,   # 同上
)
```

Decode 一步更新：
```python
k_new = concat([k_cached, k_current], dim=2)   # T_cached+1
v_new = concat([v_cached, v_current], dim=2)
```

---

## 6. Checkpoint 格式（.pt）

```python
ckpt = torch.load("draft_tf_s{step}.pt")
ckpt.keys()
# → "step", "model", "arch", "config"

ckpt["arch"]    # "transformer"
ckpt["config"]  # dict: d_model, n_layers, n_heads, n_kv_heads, ffn_mult, vocab_size
ckpt["model"]   # state_dict，key 見第 3 節
```

---

## 7. MLX 轉換步驟

### 7.1 提取 weights

```python
import torch, numpy as np

ckpt = torch.load("draft_tf_s5000.pt", map_location="cpu", weights_only=False)
sd   = ckpt["model"]

# 轉 numpy（MLX 吃 float16 或 float32）
weights = {k: v.to(torch.float16).numpy() for k, v in sd.items()}

# 去掉重複的 head.weight（與 embed.weight 相同）
weights.pop("head.weight", None)

np.savez("draft_tf.npz", **weights)
```

### 7.2 MLX 架構骨架

```python
import mlx.core as mx
import mlx.nn as nn

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq=768, base=10000):
        super().__init__()
        inv = 1.0 / (base ** (mx.arange(0, head_dim, 2) / head_dim))
        t   = mx.arange(max_seq)
        emb = mx.concatenate([mx.outer(t, inv)] * 2, axis=-1)
        self.cos = mx.cos(emb)   # (max_seq, head_dim)
        self.sin = mx.sin(emb)

    def __call__(self, x, offset=0):
        # x: (B, H, T, head_dim)
        T   = x.shape[2]
        h   = x.shape[-1] // 2
        cos = self.cos[offset:offset+T][None, None]
        sin = self.sin[offset:offset+T][None, None]
        x_rot = mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)
        return x * cos + x_rot * sin

class GQAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_kv_heads=2, head_dim=32):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        kv_dim = n_kv_heads * head_dim

        self.q    = nn.Linear(d_model, d_model,  bias=False)
        self.k    = nn.Linear(d_model, kv_dim,   bias=False)
        self.v    = nn.Linear(d_model, kv_dim,   bias=False)
        self.o    = nn.Linear(d_model, d_model,  bias=False)
        self.rope = RoPE(head_dim)

    def __call__(self, x, past_kv=None, offset=0):
        B, T, _ = x.shape
        q = self.q(x).reshape(B, T, self.n_heads,    self.head_dim).transpose(0,2,1,3)
        k = self.k(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0,2,1,3)
        v = self.v(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0,2,1,3)

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if past_kv is not None:
            k = mx.concatenate([past_kv[0], k], axis=2)
            v = mx.concatenate([past_kv[1], v], axis=2)

        scale     = self.head_dim ** -0.5
        is_causal = (past_kv is None) and (T > 1)
        mask      = nn.MultiHeadAttention.create_additive_causal_mask(k.shape[2]) if is_causal else None

        # mx.fast.scaled_dot_product_attention broadcasts KV across query groups
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        out = out.transpose(0,2,1,3).reshape(B, T, -1)
        return self.o(out), (k, v)

class SwiGLU(nn.Module):
    def __init__(self, d_model=256, mult=3):
        super().__init__()
        h = d_model * mult
        self.gate = nn.Linear(d_model, h, bias=False)
        self.up   = nn.Linear(d_model, h, bias=False)
        self.down = nn.Linear(h, d_model, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(256)
        self.attn  = GQAttention()
        self.norm2 = nn.RMSNorm(256)
        self.ffn   = SwiGLU()

    def __call__(self, x, past_kv=None, offset=0):
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv, offset=offset)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv

class DraftTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed  = nn.Embedding(32007, 256)
        self.layers = [Block() for _ in range(6)]
        self.norm   = nn.RMSNorm(256)
        self.head   = nn.Linear(256, 32007, bias=False)
        # weight tying 在 load_weights 後手動設定：
        # self.head.weight = self.embed.weight

    def __call__(self, input_ids, past_key_values=None):
        offset = past_key_values[0][0].shape[2] if past_key_values else 0
        x      = self.embed(input_ids)
        new_kvs = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            x, kv = layer(x, past_kv=past, offset=offset)
            new_kvs.append(kv)
        logits = self.head(self.norm(x))
        return logits, new_kvs
```

### 7.3 載入 weights

```python
model = DraftTransformer()
model.load_weights("draft_tf.npz")     # 不含 head.weight（已 pop）
model.head.weight = model.embed.weight  # 手動設 weight tying
mx.eval(model.parameters())
```

### 7.4 state_dict key 對應

PyTorch key → MLX attribute：

| PyTorch | MLX |
|---|---|
| `embed.weight` | `model.embed.weight` |
| `layers.{i}.norm1.weight` | `model.layers[i].norm1.weight` |
| `layers.{i}.attn.q.weight` | `model.layers[i].attn.q.weight` |
| `layers.{i}.attn.k.weight` | `model.layers[i].attn.k.weight` |
| `layers.{i}.attn.v.weight` | `model.layers[i].attn.v.weight` |
| `layers.{i}.attn.o.weight` | `model.layers[i].attn.o.weight` |
| `layers.{i}.norm2.weight` | `model.layers[i].norm2.weight` |
| `layers.{i}.ffn.gate.weight` | `model.layers[i].ffn.gate.weight` |
| `layers.{i}.ffn.up.weight` | `model.layers[i].ffn.up.weight` |
| `layers.{i}.ffn.down.weight` | `model.layers[i].ffn.down.weight` |
| `norm.weight` | `model.norm.weight` |
| `head.weight` | ← 跳過，用 embed.weight |

---

## 8. 注意事項

- **Linear weight 轉置**：PyTorch `nn.Linear` 儲存為 `(out, in)`，MLX `nn.Linear` 也是 `(out, in)`，**不需轉置**。
- **RMSNorm eps**：PyTorch 預設 `1e-6`，MLX `nn.RMSNorm` 預設也是 `1e-6`，一致。
- **dtype**：訓練用 bfloat16，MLX 建議轉 float16（Apple Silicon 不支援 bfloat16 硬體加速）。
- **RoPE buffers 不在 state_dict**：cos/sin table 由 `DraftConfig` 參數重新計算即可。
- **Speculative decoding 接口**：MLX 版和 PyTorch 版 forward 簽名相同，直接替換即可。
