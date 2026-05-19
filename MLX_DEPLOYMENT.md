# MLX 部署指南

本文档说明如何将 Tiny-LLM CoT 训练管线迁移至 **MLX**（Apple Silicon 优化框架），包括模型转换、数据准备、训练优化和推理。

## 1. MLX 概述与适用场景

### MLX 特点

- **目标硬件**：Apple Silicon (M1/M2/M3/M4 Pro/Max/Ultra)
- **后端**：Metal（GPU）+ CPU 自适应调度
- **API**：NumPy 风格，易于学习和迁移
- **生态**：MLX、MLX-LM（推理/微调库）
- **优势**：
  - 本地训练无需云资源
  - 内存效率高（统一内存架构）
  - 推理快速（GPU 加速）
  - 开发迭代快

### 性能预期

| 任务                      | Tiny-LLM (12.7M) | 备注                      |
| ------------------------- | ---------------- | ------------------------- |
| **训练**（32GB M-series） | ~0.3–0.5s/step   | 基础设置；比 GPU 慢 5–10× |
| **推理**（fp16）          | ~50–80 tokens/s  | 实时对话可接受            |
| **内存占用**              | ~2–4 GB          | 远低于 VRAM 要求          |

---

dsf

## 2. 环境设置

### 2.1 前置条件

```bash
# macOS 12+ with Apple Silicon (M1/M2/M3+)
uname -m  # 应输出 "arm64"

# 检查 Metal 支持
system_profiler SPDisplaysDataType | grep Metal
```

### 2.2 安装 MLX

```bash
# 创建专用 conda 环境（推荐 Python 3.10–3.11）
conda create -n mlx python=3.11
conda activate mlx

# 安装 MLX 及依赖
pip install mlx mlx-lm

# 验证安装
python -c "import mlx.core as mx; print(mx.metal_enabled())"
# 应输出: True
```

### 2.3 验证 Metal 加速

```python
import mlx.core as mx
import time

# 简单性能测试
a = mx.random.normal((4096, 4096))
b = mx.random.normal((4096, 4096))
start = time.time()
c = mx.matmul(a, b)
mx.eval(c)
elapsed = time.time() - start
print(f"4K×4K matmul: {elapsed:.3f}s (Metal enabled: {mx.metal_enabled()})")
```

---

## 3. 模型转换与加载

### 3.1 HuggingFace 格式 → MLX 格式

#### 选项 A：使用 MLX-LM 内置转换（推荐）

```bash
# 转换 Tiny-LLM 到 MLX 格式
mlx_lm.convert --hf-path ../third_party/Tiny-LLM \
               --mlx-path ./models/tiny_llm_mlx

# 验证
ls -lh models/tiny_llm_mlx/
# 应包含: config.json, model-0000.safetensors, tokenizer.model, etc.
```

#### 选项 B：手动转换（更新词汇表）

对于自定义 32007 词汇表的 tokenizer，需要手动调整：

```python
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 加载原始 HuggingFace 模型
model = AutoModelForCausalLM.from_pretrained("../third_party/Tiny-LLM")
tokenizer = AutoTokenizer.from_pretrained("../third_party/Tiny-LLM")

# 2. 替换为自定义 tokenizer（32007 tokens）
custom_tokenizer = AutoTokenizer.from_pretrained("dataset/tokenizer")
model.resize_token_embeddings(32007)

# 3. 导出为 safetensors（MLX 兼容）
from safetensors.torch import save_file
state_dict = model.state_dict()
save_file(state_dict, "models/tiny_llm_safetensors/model.safetensors")

# 4. 复制配置并更新 vocab_size
import shutil
shutil.copy("dataset/tokenizer/config.json",
            "models/tiny_llm_safetensors/config.json")
config = json.load(open("models/tiny_llm_safetensors/config.json"))
config["vocab_size"] = 32007
json.dump(config, open("models/tiny_llm_safetensors/config.json", "w"), indent=2)
```

### 3.2 在 MLX 中加载模型

```python
import mlx.core as mx
from mlx_lm.models.llama import LanguageModel
from transformers import AutoConfig

# 加载配置和权重
config = AutoConfig.from_pretrained("models/tiny_llm_mlx/config.json")
model = LanguageModel.from_pretrained("models/tiny_llm_mlx")

print(f"Model: {model}")
print(f"Params: {sum(v.size for v in mx.tree_flatten(model.parameters()))}")
# 应为 ~12.7M
```

---

## 4. 数据准备与 Token 化

### 4.1 数据格式要求

MLX 与 PyTorch 使用相同的 `.bin` 格式。复用现有数据管线：

```bash
# 使用现有脚本生成 .bin 文件
python scripts/stf_cot_to_bin.py --tokenizer-dir dataset/tokenizer
# 输出: dataset/stf_cot_train.bin, dataset/stf_cot_hf
```

### 4.2 在 MLX 中加载 `.bin` 数据

```python
import numpy as np

def load_bin_tokens(bin_path, tokenizer):
    """加载二进制 token 流为 NumPy 数组."""
    tokens = np.fromfile(bin_path, dtype=np.uint16)
    return tokens

# 加载训练数据
train_tokens = load_bin_tokens("dataset/stf_cot_train.bin", tokenizer)
print(f"Total tokens: {len(train_tokens)} ({len(train_tokens)/1e9:.2f}B)")

# 分割成训练/验证集（例如 95:5）
val_size = int(len(train_tokens) * 0.05)
train_tokens = train_tokens[:-val_size]
val_tokens = train_tokens[-val_size:]
```

### 4.3 批处理与序列组成

```python
import mlx.core as mx

def create_sequence_batch(tokens, seq_len, batch_size, start_idx=0):
    """从 token 流创建 (input, target) 批次."""
    batch = []
    indices = []

    for i in range(start_idx, start_idx + batch_size):
        idx = (i * seq_len) % (len(tokens) - seq_len - 1)
        x = tokens[idx : idx + seq_len]
        y = tokens[idx + 1 : idx + seq_len + 1]

        batch.append((x, y))
        indices.append(idx)

    # 转为 MLX tensor
    xs = mx.array(np.stack([b[0] for b in batch]))  # (batch, seq_len)
    ys = mx.array(np.stack([b[1] for b in batch]))

    return xs, ys, indices

# 测试
xs, ys, _ = create_sequence_batch(train_tokens, seq_len=768, batch_size=8)
print(f"Batch shape: {xs.shape}, {ys.shape}")
```

---

## 5. 训练脚本：MLX 版本

### 5.1 完整训练循环模板

创建 `scripts/train_sft_cot_mlx.py`：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT-CoT 训练：MLX 版本（Apple Silicon 优化）
"""

import json
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 配置与超参数
# ─────────────────────────────────────────────────────────────────────────────

class TrainConfig:
    """训练超参数"""
    # 模型
    model_path = "models/tiny_llm_mlx"

    # 数据
    train_bin = "dataset/stf_cot_train.bin"
    seq_len = 768

    # 训练
    batch_size = 16  # MLX 推荐批大小：8–32
    epochs = 5
    lr = 1e-5
    warmup_steps = 500
    val_interval = 100
    save_interval = 200

    # FCP/SCALe 配置
    enable_fcp = True
    enable_scale = True
    fcp_lambda = 0.2
    fcp_delta = 0.01
    scale_eta_max = 1.0
    scale_eta_min = 0.3

    # 输出
    output_dir = "output/mlx_checkpoints"
    log_file = "output/train_mlx_log.json"

# ─────────────────────────────────────────────────────────────────────────────
# 核心训练函数
# ─────────────────────────────────────────────────────────────────────────────

def load_tokens(bin_path):
    """加载二进制 token 流."""
    tokens = np.fromfile(bin_path, dtype=np.uint16)
    return tokens

def create_batch(tokens, seq_len, batch_size, start_idx):
    """创建一个批次."""
    batch_xs = []
    batch_ys = []

    for i in range(batch_size):
        idx = (start_idx + i) * seq_len
        if idx + seq_len + 1 > len(tokens):
            idx = idx % (len(tokens) - seq_len - 1)

        x = tokens[idx : idx + seq_len]
        y = tokens[idx + 1 : idx + seq_len + 1]
        batch_xs.append(x)
        batch_ys.append(y)

    return mx.array(batch_xs), mx.array(batch_ys)

def compute_loss(logits, targets, seq_len=768, enable_fcp=False, enable_scale=False):
    """
    计算加权 CE 损失（支持 FCP、SCALe）。

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)

    Returns:
        total_loss, ce_loss, fcp_loss（如启用）
    """
    batch_size, seq_len, vocab_size = logits.shape

    # 打平用于交叉熵
    logits_flat = logits.reshape(-1, vocab_size)  # (batch*seq_len, vocab_size)
    targets_flat = targets.reshape(-1)

    # 基础 CE 损失
    ce_loss = nn.losses.cross_entropy(logits_flat, targets_flat)

    fcp_loss = mx.array(0.0)
    scale_w = mx.array(1.0)

    # FCP 损失（简化版：惩罚 <think> 区域的 EOS）
    if enable_fcp:
        # token IDs：32005 = </think>, 32001 = <|im_end|>
        eos_id = 32001
        # 检测 <think> 区域（简化：假设前 40% 是 think）
        think_end = int(seq_len * 0.4)

        # P(EOS | x) 在 think 区域
        eos_probs = mx.softmax(logits[:, :think_end, :], axis=-1)[:, :, eos_id]
        fcp_loss = mx.mean(eos_probs)  # 惩罚高 EOS 概率

    # SCALe：不同区域不同权重
    if enable_scale:
        # 简化：think 区域权重动态变化
        total_steps = 10000  # 预估总步数
        current_step = 0  # 应从外部注入
        progress = max(0, min(1, current_step / total_steps))
        scale_w = 1.0 - progress * (1.0 - 0.3)  # 从 1.0 → 0.3

    total_loss = ce_loss + 0.2 * fcp_loss  # 0.2 = FCP 权重

    return total_loss, ce_loss, fcp_loss

def train_epoch(model, train_tokens, config, epoch):
    """训练一个 epoch."""
    n_batches = len(train_tokens) // (config.batch_size * config.seq_len)
    losses = []

    for batch_idx in range(n_batches):
        # 创建批次
        xs, ys = create_batch(train_tokens, config.seq_len, config.batch_size, batch_idx)

        # 前向传播
        def loss_fn(model_params):
            logits = model(xs)
            total_loss, ce_loss, fcp_loss = compute_loss(
                logits, ys,
                enable_fcp=config.enable_fcp,
                enable_scale=config.enable_scale
            )
            return total_loss

        # 反向传播与优化
        loss, grads = mx.value_and_grad(loss_fn)(model.parameters())

        # 梯度裁剪
        grads_flat = mx.tree_flatten(grads)[0]
        grad_norm = mx.sqrt(sum(mx.sum(g ** 2) for g in grads_flat))
        if grad_norm > 1.0:
            grads = mx.tree_map(lambda g: g / grad_norm, grads)

        losses.append(loss.item())

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx+1}/{n_batches} | Loss: {loss:.4f}")

    return np.mean(losses)

def main():
    config = TrainConfig()

    # 加载模型
    print("Loading model...")
    from mlx_lm.models.llama import LanguageModel
    model = LanguageModel.from_pretrained(config.model_path)

    # 加载数据
    print("Loading tokens...")
    train_tokens = load_tokens(config.train_bin)

    # 优化器
    optimizer = optim.Adam(learning_rate=config.lr)

    # 创建输出目录
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # 训练循环
    logs = []
    for epoch in range(config.epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")
        epoch_loss = train_epoch(model, train_tokens, config, epoch)

        logs.append({"epoch": epoch, "loss": epoch_loss})
        print(f"Epoch {epoch + 1} | Avg Loss: {epoch_loss:.4f}")

        # 保存 checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f"{config.output_dir}/epoch_{epoch+1}.npz"
            # 保存权重（使用 MLX 原生格式或转换为 NumPy）
            print(f"Checkpoint → {checkpoint_path}")

    # 保存日志
    with open(config.log_file, "w") as f:
        json.dump(logs, f, indent=2)

if __name__ == "__main__":
    main()
```

### 5.2 运行训练

```bash
# 激活环境
conda activate mlx

# 运行训练（预期：~0.3–0.5s/step，依赖硬件）
python scripts/train_sft_cot_mlx.py

# 监控 GPU/内存
# 在另一个终端：
watch -n 1 "ps aux | grep train_sft_cot_mlx; system_profiler SPDisplaysDataType | grep -i gpu"
```

---

## 6. 性能优化策略

### 6.1 内存优化

| 策略           | 效果                     | 实现                                       |
| -------------- | ------------------------ | ------------------------------------------ |
| **批大小调整** | 批大小↓ → 内存↓，但吞吐↓ | 从 16 开始，逐步增加至 32 或 64            |
| **序列长度**   | seq_len↓ → 显存↓         | 使用 512 代替 768（精度略降）              |
| **精度降级**   | float16 → 内存减半       | MLX 默认 fp32；可手动转换                  |
| **梯度累积**   | 模拟大批大小，内存低     | effective_batch = batch_size × accum_steps |

### 6.2 速度优化

```python
# 1. 启用 Metal 优化
mx.set_default_device(mx.gpu)

# 2. 使用 JIT 编译（预定义计算图）
from mlx.core import compile
loss_fn_compiled = compile(loss_fn)

# 3. 减少 dtype 转换
# ✓ 好：logits = logits.astype(mx.float32)
# ✗ 差：多次转换

# 4. 批量评估（避免重复编译）
mx.eval(losses)  # 一次性评估所有张量
```

### 6.3 分布式训练（多 GPU/NPU）

MLX 目前不支持多设备 DDP，但可使用数据并行：

```python
# 伪代码：手动数据并行
devices = [mx.Device.gpu(i) for i in range(num_gpus)]
for batch_idx, batch in enumerate(batches):
    # 分割批次到各设备
    sharded_batch = [batch[i::num_gpus] for i in range(num_gpus)]

    # 并行前向/反向
    losses = [compute_loss_on_device(b, dev) for b, dev in zip(sharded_batch, devices)]

    # 平均梯度并更新
    avg_loss = sum(losses) / len(losses)
```

---

## 7. Checkpoint 与推理

### 7.1 保存与恢复

```python
def save_checkpoint(model, optimizer, step, save_path):
    """保存 MLX checkpoint."""
    checkpoint = {
        "model_state": dict(mx.tree_flatten(model.parameters())),
        "optimizer_state": optimizer.state,
        "step": step,
    }
    mx.savez(str(save_path), **checkpoint)

def load_checkpoint(checkpoint_path):
    """加载 checkpoint."""
    data = mx.load(str(checkpoint_path))
    return data

# 使用
save_checkpoint(model, optimizer, step=100, save_path="checkpoints/step_100.npz")
checkpoint = load_checkpoint("checkpoints/step_100.npz")
```

### 7.2 推理（生成文本）

```python
from mlx_lm.generate import generate
from transformers import AutoTokenizer

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("dataset/tokenizer")

# 文本生成
prompt = "<|im_start|>user\n请解释 CoT 的原理。<|im_end|>\n<|im_start|>assistant\n<think>"
generated = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
)
print(generated)
```

---

## 8. 外部 Draft 模型投機解碼實驗報告

### 8.1 實驗設置

| 項目 | 詳情 |
|------|------|
| **Target 模型** | Mamba3 Hybrid（417M, 30 層 Mamba+Transformer） |
| **Draft 模型** | `checkpoints/darf-model`（13M, 1 層 Transformer, d_model=192, MQA） |
| **方法** | External Draft Spec Decode（貪婪接受，greedy accept/reject） |
| **測試腳本** | `inference/benchmark_spec_external.py` |
| **Makefile 指令** | `make mlx-spec-external` |
| **Prompt 長度** | 94 tokens |
| **生成 tokens** | 150 tokens，warmup=1 |
| **硬體** | Apple Silicon（MLX bf16） |

**架構說明（Draft 模型）：**
- `embed_tokens` : \[32007, 192\]（vocab=32007, d_model=192）
- 1 層 Self-Attn + MLP：q/k/v 投影 \[192,192\] / \[96,192\] / \[96,192\]，MQA（n_heads=2, n_kv_heads=1, head_dim=96）
- FFN（SwiGLU）：gate/up \[1024,192\]，down \[192,1024\]
- 總計 **13.0M 參數**

### 8.2 實驗結果

| 模式 | K（Draft/輪） | tok/s | 接受率（accept%） | 相對基線加速比 |
|------|:---:|------:|------:|------:|
| **Baseline（無投機）** | — | **71.2** | — | 1.00× |
| External Spec | K=1 | 38.0 | 18.1% | 0.53× |
| External Spec | K=2 | 28.6 | 10.5% | 0.40× |
| External Spec | K=4 | 25.0 | 5.7% | 0.35× |
| External Spec | K=6 | 21.3 | 3.4% | 0.30× |
| External Spec | K=8 | 19.1 | 2.5% | 0.27× |

**結論：所有 K 值均低於基線（71.2 tok/s），最佳 K=1 也僅有 0.53× 速度。**

### 8.3 Root Cause 分析

#### 問題 1：Mamba 的 Verify 不能有效並行

| 解碼方式 | 複雜度 | 備註 |
|---------|--------|------|
| 純 Transformer | O(K²) attention，但 Metal 批量計算高效 | K token verify ≈ 1 step cost（計算並行） |
| **Mamba SSM scan** | **O(K) sequential state update** | K token verify ≈ K × t_single（無法省略） |

Mamba 的 SSM 狀態更新是**遞歸的**（`h_t = A·h_{t-1} + B·x_t`），即使使用 chunk-wise parallel scan，decode 模式下 K token 的 verify 代價仍近似 K × t_single。這意味著 verify 並不能帶來「免費 K 個 token」的效果。

#### 問題 2：Draft 與 Reject Replay 的額外開銷

每輪 spec 的代價（以 K=1 為例）：

```
接受時 (18.1%):  1 draft forward + 1 verify  ≈ t_draft + t_single
拒絕時 (81.9%):  1 draft forward + 1 verify + 1 correction + 1 draft replay（同步 KV cache）
              ≈ 2×t_draft + 2×t_single
```

**期望代價/token = 0.18×(t_d + t_v) + 0.82×(2t_d + 2t_v) ≈ 1.82×(t_d + t_v)**
**vs 基線 = 1×t_single ≈ t_v**

即使 t_draft ≈ 0，spec 代價也是基線的 ~1.82×，因此 K=1 約有 0.53× 加速比。

#### 問題 3：接受率過低

| K | 期望接受 tokens/輪 | 理論吞吐比（無 replay） | 實測 |
|---:|---:|---:|---:|
| 1 | 0.18 | 1.18/2 ≈ 0.59× | 0.53× |
| 4 | 0.23 | 1.23/5 ≈ 0.25× | 0.35× |

Draft 模型雖在相同 SFT 資料集上訓練，但 d_model=192（vs target 768）、只有 1 層、訓練 1680 步，與 target 預測分佈差距大，接受率上限約 18%（K=1）。

### 8.4 關於 embed_tokens 共享的可行性

```
Draft:  embed_tokens [32007, 192]   d_model = 192
Target: embed_tokens [32007, 768]   d_model = 768
```

**無法直接共享**，因為 d_model 不同，embedding 向量空間維度不一致。理論上可以做：

1. **Projection 共享**：Draft 使用 target 的 embedding + 線性投影（768→192）。
   - 節省 32007×192 ≈ 6.1M 參數
   - 增加一個 768×192=147K 投影矩陣
   - 需要重新訓練 draft 模型

2. **Token ID 共享（已實現）**：兩個模型使用同一個 tokenizer 的 token ID，embedding lookup 各自獨立。
   - Draft embedding lookup（K tokens × 192 dim）開銷可忽略
   - **目前瓶頸不在 embedding，在 Mamba 的 SSM scan**

**結論：embedding 不是瓶頸，共享/融合對整體速度幾乎無影響（< 0.5ms/輪）。**

### 8.5 與 Early-Exit 投機解碼的比較

| 方法 | 最佳 accept% | 最佳 tok/s | 相對基線 | 備註 |
|------|:---:|------:|------:|------|
| Baseline | — | ~98 tok/s | 1.00× | 基線（無投機） |
| Early-Exit (8 layers) | 3.6% | 16.8 tok/s | 0.17× | 僅用 30 層中的前 8 層 |
| Early-Exit (28 layers) | 29.5% | 24.1 tok/s | 0.25× | |
| **External Draft (darf-model)** | **18.1% (K=1)** | **38.0 tok/s** | **0.53×** | 本次實驗 |
| Jacobi | —（結構文本高） | — | — | 僅對重複性文本有效 |

外部 Draft 接受率（18.1%）優於 Early-Exit 小 K 設置，但仍無法超越基線。

### 8.6 未來改進方向

| 方向 | 預期效果 | 難度 |
|------|---------|------|
| **Draft 訓練對齊**：對 target logits 做 knowledge distillation | accept% 提升至 40–60% | 高 |
| **Draft 架構調整**：增加訓練步數（1680→10000+）、加深至 2–3 層 | accept% 提升至 25–35% | 中 |
| **Jacobi 模式**：適合生成格式化輸出（如 JSON、code） | 視文本而定 | 已實現 |
| **Mamba 架構改進**：支援 Tree-Attention 或 chunk verify | 根本解決 Mamba 瓶頸 | 極高 |
| **純 Transformer 區段的局部投機**：只在 Transformer 層做 spec | 部分加速 | 高 |

### 8.7 使用方式

```bash
# 標準 K sweep（K=1,2,4,6,8，生成 200 tokens）
make mlx-spec-external

# 自定義 K 值和生成長度
make mlx-spec-external SPEC_EXT_K="1 4" SPEC_EXT_N=100

# 直接執行腳本
.venv/bin/python inference/benchmark_spec_external.py \
    --checkpoint checkpoints/latest_sft_cot_model.npz \
    --draft-checkpoint checkpoints/darf-model \
    --n-tokens 200 \
    --k-values 1 2 4 6 8 \
    --output-json inference/results/spec_external_results.json
```
