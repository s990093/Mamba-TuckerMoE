    # dLLM 改造 + 驗證（MLX 端）

> 前提：你已有自己的 MLX 架構與 GPU kernel（scan / MoE / attention 都跑得動）。
> 本文**只列把 AR 架構變 dLLM 需要動的 4 處**，以及怎麼驗證有沒有改對。
> 不重複講 kernel 移植。CUDA 端對照實作見 §5。

---

## 0. 要改的只有 4 處（其餘 forward 不動）

| #   | 改動                                   | 影響範圍         | 訓練/推理 |
| --- | -------------------------------------- | ---------------- | :-------: |
| ①   | 加 `[MASK]` token（embed 32007→32008） | embedding / head |   兩者    |
| ②   | 注意力關掉 causal（雙向）              | 只 attention 層  |   兩者    |
| ③   | 訓練目標換成 masked-CE（1/t 加權）     | loss             |   訓練    |
| ④   | 推理換成迭代 unmasking                 | 生成迴圈         |   推理    |

> Mamba 層**不動**（仍單向 scan）→ 只有 attention 變雙向 = **partial bidirectional**。
> 這是刻意的；要全雙向才需動到 scan（BiMamba），本版不做。

---

## ① [MASK] token

```python
MASK_ID = 32007                      # 接在原 32007 詞表之後
# embedding 由 (32007, D) 擴成 (32008, D)
new_emb = mx.zeros((32008, D))
new_emb[:32007] = old_emb
new_emb[32007]  = old_emb.mean(axis=0) + 0.02 * mx.random.normal((D,))   # MASK 初值
# head 與 embed 綁定：指向同一 array
model.embed.weight = new_emb
model.head.weight  = new_emb
```

> 若直接載入 CUDA 訓練好的 dLLM 權重，embed 已是 `(32008, D)`，**這步免做**，
> 只要確認 `MASK_ID=32007` 且 head/embed tie 正確。

---

## ② 雙向注意力（關鍵一行）

你的 attention 把 causal mask 拿掉即可：

```python
# AR（原本）:  mask = causal_mask  (或 "causal")
# dLLM:        mask = None         ← 改這個
attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1/math.sqrt(head_dim), mask=None)
```

對應 CUDA 端的 `is_causal=False`。Mamba 的 scan **保持原樣**，不要動。

---

## ③ 訓練目標：masked-CE（1/t 加權）

把 AR 的 next-token CE 換成 absorbing-diffusion 目標（LLaDA 式）。
**prompt 永不 mask，只 mask response 區**：

```python
def dllm_loss(model, x, resp_mask):           # x:(B,T)  resp_mask:(B,T) bool
    B, T = x.shape
    t = mx.random.uniform(shape=(B, 1)) * (1 - 1e-3) + 1e-3      # 每序列噪聲比例
    noise  = mx.random.uniform(shape=(B, T))
    masked = resp_mask & (noise < t)                            # 被遮位置
    # 保底：每序列至少遮 1 個 response token（避免分母 0）
    noisy  = mx.where(masked, MASK_ID, x)

    logits = model(noisy)                                       # (B,T,V) 雙向前向
    ce = nn.losses.cross_entropy(logits, x, reduction="none")  # (B,T)
    ce_masked = (ce * masked).sum(axis=1)
    per_seq = (1.0 / t.squeeze(1)) * ce_masked / mx.maximum(resp_mask.sum(1), 1)
    return per_seq.mean()

loss_and_grad = nn.value_and_grad(model, dllm_loss)
# 迴圈：loss, grads = loss_and_grad(model, x, resp_mask); opt.update(model, grads); mx.eval(...)
```

重點：

- 只在 `masked` 位置算 CE；`1/t` 加權（遮越少、單點越重要）。
- 不需要 MoE 的 lb/z loss、不需要 FCP/SCALe/SFT-GO（overfit demo 用單項就好）。
- response 區切點 = `<|im_start|>assistant\n`（ids `[32000,465,22137,13]`）之後到第一個 `<|im_end|>`(32001)。

---

## ④ 推理：迭代 unmasking

```python
import math
x = mx.array([prompt_ids + [MASK_ID]*G])           # (1, P+G)
filled = [False]*G
T = 16                                              # 迭代步數（每次推理跑 T 次 forward）
for s in range(1, T+1):
    logits = model(x)[0, P:]                        # (G, V) 一次 forward 拿全部 [MASK] 分佈
    probs  = mx.softmax(logits, axis=-1)
    conf   = mx.max(probs, axis=-1); pred = mx.argmax(probs, axis=-1)
    keep   = int(math.floor(G * math.cos(math.pi/2 * s/T)))   # cosine：第 s 步後留幾個 MASK
    need   = (G - keep) - sum(filled)
    if need > 0:
        cand  = mx.where(mx.array(filled), -1.0, conf)
        order = mx.argsort(-cand)[:need].tolist()   # 信心最高 need 個（MLX 無 topk 索引→argsort）
        for j in order:
            x[0, P+j] = int(pred[j]); filled[j] = True
    mx.eval(x)                                       # 惰性求值，迴圈內要 eval
# 截到 </final>(32005) 或 <|im_end|>(32001)
```

gotchas：`mx.topk` 不回索引（用 `argsort`/`argpartition`）；迴圈內記得 `mx.eval()`；
取樣版用 `mx.random.categorical(logits/temp)`。

---

## 驗證（三層，由淺到深）

### (A) logit parity — 先確認 forward 沒搬錯

同一段 `input_ids`（含 `[MASK]`）餵 CUDA 與 MLX，比 logits：

```
順序：1) 單向(causal) 對拍 → 確認 scan/MoE/attention 本身對
     2) 再切雙向(mask=None) 對拍 → 確認 ② 改對
容忍：bf16 max-abs-diff ~1e-2；fp32 ~1e-4
```

> 先單向後雙向能把「kernel 對不對」和「dLLM 改動對不對」拆開抓。

### (B) 固定比例重建準確率 — 確認有學到 masked 預測

對 response 區用**固定** mask 比例隨機遮，一次 forward，量被遮位置 top-1 acc：

```python
for ratio in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    masked = resp_mask & (mx.random.uniform(x.shape) < ratio)
    pred = model(mx.where(masked, MASK_ID, x)).argmax(-1)
    acc = ((pred == x) & masked).sum() / masked.sum()
```

判讀：

- 未訓練基線：ratio 0.1 ≈ 0.06、0.9 ≈ 0.002（≈亂猜）。
- overfit 訓練後：ratio 0.1 應 **接近 1.0**，且隨 ratio 升高才下降。

### (C) 迭代重建 vs gold — 端到端「會不會生成」

從全 `[MASK]` 跑 §④ 的 T 步，和原文比：

```
token 準確率 = (gen == gold).mean()
exact-match  = (gen == gold).all() 的樣本比例
```

判讀：overfit 64 筆時，token 準確率與 EM 應該都很高（模型記住了）。

> 三個測試的 CUDA 參考實作就是 `scripts/dllm_validate.py`（(A) 用 logit 比對版本、
> (B)/(C) 已實作），可拿來和 MLX 數字對拍。

---

## 5. CUDA 端對照（拿來對拍 / 抄行為）

| 概念           | CUDA 實作（本 repo）                               | 對應改動 |
| -------------- | -------------------------------------------------- | -------- |
| 雙向           | `dllm_common.py: enable_bidirectional_attention()` | ②        |
| `[MASK]`/載入  | `dllm_common.py: add_mask_token / load_dllm_model` | ①        |
| 訓練目標       | `dllm_finetune.py`（masked-CE, 1/t）               | ③        |
| 迭代 unmasking | `dllm_generate.py`                                 | ④        |
| 驗證 (A)(B)(C) | `dllm_validate.py`                                 | 驗證     |
| 原理/超參      | `DLLM.md`                                          | —        |

---

## 詞表常數（兩端必須一致）

```
MASK_ID = 32007   PAD = 32006   <|im_end|> = 32001
<think>=32002  </think>=32003   <final>=32004  </final>=32005
assistant header ids = [32000, 465, 22137, 13]   # <|im_start|>assistant\n
```
