# CoT Middleware Instability Diagnosis

## 问题观察

对比三个 "who are you?" 的输出：

### 输出 A & B (chat_precise.sh，不用 MW)
```
✅ 相对稳定，有清晰的步骤结构
- Step 1/2/3/4 的逻辑流畅
- Token 解码正常，无损坏
- `<think>` 块语义合理
```

### 输出 C (启用 CoT MW)
```
❌ 严重问题
1. Token 解码错误：
   - "##egressive" → 应该是单词但断裂
   - "Conyna and theight" → 完全错误组合
   - "in-the-murath-slavery" → 虚幻 token

2. 格式支离破碎：
   - "Step 1: ##egressive language model." → 句子残缺
   - "**n05:" → 异常符号出现
   
3. 推理块结构崩溃：
   - 缺少逻辑连接词
   - 无法形成完整句子
```

## 根本原因分析

### 假设 1: 注入的 `</final>\n` 导致的 logits 错误
**症状匹配度**: ⭐⭐⭐⭐ (高)

在 `maybe_inject_final()` 调用时：
```python
# server.py:916-924
caches, new_pos, last_row, did_inject, _ = mw.maybe_inject_final(
    caches=caches, pos=len(generated))
if did_inject:
    inj_logits = last_row  # ← 这个 logits 可能有问题
```

**可能问题**：
- `last_row` (注入后的 logits) 可能被错误量化或裁剪
- 如果 Middleware 启用但 format_guard 被应用两次，可能导致双重约束
- `close_bias` 可能过度压低某些 token 概率

### 假设 2: 两次 logits 变换导致的数值溢出
**症状匹配度**: ⭐⭐⭐ (中)

```python
# server.py:888,794
logits_1d = mw.transform_logits(logits_1d)  # 第一次变换
# ... 采样
# ... 注入
logits_1d = mw.transform_logits(logits_1d)  # 是否被再次变换？
```

如果 `inj_logits` 被再次通过 `transform_logits()` 处理，可能导致：
- Logits 梯度饱和（超出 float32 范围）
- 小概率 token 被意外放大

### 假设 3: Tokenizer 状态污染
**症状匹配度**: ⭐⭐ (低)

如果 CoT MW 导致某些特殊 token 状态改变，后续解码可能出错。

---

## 差异对比表

| 方面 | 无 MW (A/B) | 有 MW (C) |
|------|----------|---------|
| **逻辑流** | 4 步结构清晰 | 步骤残缺，无逻辑 |
| **Token 完整性** | 词汇正确 | 词汇损坏（##egressive, theight） |
| **句法正确** | 完整句子 | 片段化，缺连接词 |
| **Coherence** | 虽然答非所问，但内部一致 | 完全碎片化 |
| **</final> 位置** | 不出现（无MW） | 应该在某处（已注入） |

---

## 可能的代码缺陷

### 缺陷 1: 注入后的 logits 未清理
```python
# server.py 中，after injection
if did_inject:
    inj_logits = last_row
    use_inj = True
    
# 后续采样时
if use_inj:
    logits_1d = inj_logits
    use_inj = False
    inj_logits = None
else:
    decode_step_n += 1
    logits, ... = decode_step(...)
    logits_1d = logits[0]

# ❌ logits_1d 是否被再次 transform?
logits_1d = mw.transform_logits(logits_1d)  # <-- HERE
```

**问题**：如果 `inj_logits` 来自于已经在 `maybe_inject_final()` 内部应用过某种变换的状态，再次调用 `transform_logits()` 可能导致双重应用，造成数值问题。

### 缺陷 2: format_guard 在 logits 已削弱时应用
```python
# server.py:794
first_logits = mw.transform_logits(pf_logits[0])

# ... 一系列操作

# server.py:888
logits_1d = mw.transform_logits(logits_1d)  # 每次都调用

# ❌ 如果 mw.transform_logits() 有状态或累积效应呢？
```

---

## 诊断测试计划

### 测试 1: 禁用 </final> 注入
在 `server.py` 中，临时禁用：
```python
caches, new_pos, last_row, did_inject, _ = mw.maybe_inject_final(
    caches=caches, pos=len(generated))
if did_inject:
    # did_inject = False  # ← 临时禁用
```
**预期**：如果输出恢复正常，问题在注入逻辑。

### 测试 2: 检查 logits 数值范围
在 `_sample()` 前后添加日志：
```python
def _sample(logits_1d):
    print(f"[DEBUG] logits_1d min={logits_1d.min()}, max={logits_1d.max()}, mean={logits_1d.mean()}")
    logits_1d = apply_repetition_penalty(...)
    print(f"[DEBUG] after penalty: min={logits_1d.min()}, max={logits_1d.max()}")
```
**预期**：发现数值爆炸或饱和。

### 测试 3: 对比不同 batch 的 logits
运行多次相同问题，比较 logits 的数值稳定性。
**预期**：unstable logits → unstable output。

### 测试 4: 禁用 format_guard
在 `doSend()` 时强制：
```javascript
payload.format_guard = false;  // 即使 toggle 是 ON
```
在后端，跳过 `mw.transform_logits()`：
```python
if not enable_cot_mw:
    # logits_1d = mw.transform_logits(logits_1d)  ← 注释掉
    pass
```
**预期**：如果输出恢复正常，问题在 format_guard 逻辑。

---

## 快速修复建议

### 选项 A: 强制双倍精度
在 `maybe_inject_final()` 后，确保 logits 类型正确：
```python
if did_inject:
    inj_logits = last_row
    if hasattr(inj_logits, 'astype'):
        inj_logits = inj_logits.astype('float32')  # 强制精度
    use_inj = True
```

### 选项 B: 跳过注入后的 transform
```python
if use_inj:
    logits_1d = inj_logits
    use_inj = False
    # ❌ 不要再次调用 transform_logits
else:
    logits, ... = decode_step(...)
    logits_1d = logits[0]
    logits_1d = mw.transform_logits(logits_1d)  # 只对普通 logits 调用
```

### 选项 C: 禁用 CoT MW 的 format_guard (临时)
```python
turn_mw_cfg = CotMiddlewareConfig(
    enabled=mw_cfg.enabled and enable_cot_mw,
    ban_im_start=False if enable_cot_mw else mw_cfg.ban_im_start,  # 临时禁用
    ...
)
```

---

## 建议的优先级

1. **立即做**: 运行 Test 1 (禁用注入) 来隔离问题来源
2. **如果问题在注入**: 尝试修复选项 A（精度检查）
3. **如果问题在 format_guard**: 尝试修复选项 B（跳过二次变换）
4. **长期**: 在 CotMiddleware 中添加更详细的数值日志

---

## 现象总结表

| 观察 | 指向问题 |
|------|---------|
| token 损坏（##egressive） | Logits 数值问题或 sampler 故障 |
| 句法支离破碎 | Middleware 约束过度或梯度饱和 |
| 仅在 CoT MW 启用时发生 | 问题在 CotMiddleware 或 `maybe_inject_final` |
| 重复运行有不同结果 | 可能是随机采样失败，而不是确定性错误 |

**最可能的根本原因**：`inj_logits` 的数值不正常，或者 logits 在被二次变换时发生了溢出。
