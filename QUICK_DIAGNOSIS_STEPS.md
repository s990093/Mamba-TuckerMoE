# CoT Middleware 问题快速诊断步骤

## 概述

基于观察：**启用 CoT MW 时输出破碎（token 损坏），禁用时正常**

## 快速诊断步骤（按优先级）

### 🔍 Step 1: 检查日志（最快，5 分钟）

#### 1.1 启动服务器，查看 logits 数值
```bash
python -m mamba3_mlx.server &
# 在另一个终端，运行聊天
sleep 3
./chat_precise.sh "who are you?" 2>&1 | grep -E '\[logits\]|\[inj\]|\[mw\]'
```

**观察目标**：
```
[logits] first_logits(after transform): min=-5.32, max=12.45, mean=0.23
[inj] injected_logits: min=-8.94, max=98.34, mean=1.23   ← ⚠️ 如果 max 异常大（>50）
[inj] injected tok=52341                                    ← token ID 异常大？
[logits] injected_logits(no_transform): min=-5.21, max=11.3, mean=0.19
```

**问题征兆**：
- ✗ injected_logits 的 max > 50（数值爆炸）
- ✗ injected_logits 的 min < -50（负数溢出）
- ✗ 采样后的 token ID > vocab_size

---

### 🧪 Step 2: 禁用 </final> 注入（10 分钟）

#### 2.1 临时禁用注入
编辑 `server.py` 第 929 行，改为：
```python
if False and did_inject:  # DIAGNOSTIC: disabled
    # ... 注入代码
```

#### 2.2 测试
```bash
./chat_precise.sh "who are you?" 2>&1 | tail -20
```

**判定**：
- ✅ 输出恢复正常 → **问题在注入逻辑**
- ❌ 输出仍然破碎 → 问题在其他地方（format_guard 或 sampling）

**下一步**：
- 如果恢复 → 继续 Step 3
- 如果仍破碎 → 跳到 Step 4

---

### 🔧 Step 3: 检查注入后的 logits 处理（10 分钟）

#### 3.1 禁用注入后的 transform
编辑 `server.py` 第 895-920 行，改为：
```python
        if use_inj:
            logits_1d = inj_logits
            use_inj   = False
            # ❌ 不再调用 mw.transform_logits
            tok = _sample(logits_1d, debug_label="injected_logits(no_transform)")
            log.info(f"[inj] injected tok={tok}")
        else:
            decode_step_n += 1
            logits, mamba_states, kv_caches = decode_step(...)
            logits_1d = logits[0]
            # ✅ 只对普通 logits 调用 transform
            logits_1d = mw.transform_logits(logits_1d)
            tok = _sample(logits_1d, debug_label="decode_logits(after_transform)")
```

#### 3.2 测试
```bash
./chat_precise.sh "who are you?" 2>&1 | tail -20
```

**判定**：
- ✅ 输出恢复正常 → **问题是：注入的 logits 被二次 transform，导致溢出**
- ❌ 输出仍然破碎 → **问题在 inj_logits 本身的数值**

---

### 🎯 Step 4: 禁用 format_guard（10 分钟）

#### 4.1 禁用 format_guard 的 transform
编辑 `server.py` 第 888-894 行，改为：
```python
        # Apply format guard (ban mask + dynamic close-bias) before sampling
        if not (use_inj or mw.cfg.enabled):  # 只对非注入、非MW的logits应用
            logits_1d = mw.transform_logits(logits_1d)
        # else: skip transform for injected or mw-disabled
        tok = _sample(logits_1d, debug_label="decode_logits")
```

#### 4.2 测试
```bash
./chat_precise.sh "who are you?" 2>&1 | tail -20
```

**判定**：
- ✅ 输出恢复 → **问题在 format_guard 的 close_bias 或 ban_im_start**
- ❌ 仍破碎 → **问题在采样或 sampling 参数**

---

## 根本原因判断矩阵

| 场景 | Step 2 禁用注入 | Step 3 跳过二次 transform | Step 4 禁用 format_guard | **根本原因** |
|------|-------------|--------------------------|------------------------|-----------|
| A | ✅恢复 | ✅恢复 | N/A | `inj_logits` 被二次 transform 导致溢出 |
| B | ✅恢复 | ❌仍破 | N/A | `inj_logits` 数值本身异常 |
| C | ❌仍破 | N/A | ✅恢复 | format_guard transform 过度压低 |
| D | ❌仍破 | N/A | ❌仍破 | sampling 参数或 tokenizer 问题 |

---

## 修复建议（根据根本原因）

### 如果是 A（二次 transform）
```python
# server.py 第 895-915 行修改为：
if use_inj:
    logits_1d = inj_logits
    use_inj = False
    tok = _sample(logits_1d, debug_label="injected(skip_transform)")
else:
    decode_step_n += 1
    logits, mamba_states, kv_caches = decode_step(...)
    logits_1d = logits[0]
    logits_1d = mw.transform_logits(logits_1d)
    tok = _sample(logits_1d, debug_label="decode(after_transform)")
```

### 如果是 B（inj_logits 异常）
检查 `CotMiddleware.maybe_inject_final()` 中的 logits 生成逻辑
```python
# 可能的修复：确保精度
if hasattr(last_row, 'astype'):
    last_row = last_row.astype('float32')
```

### 如果是 C（format_guard 过度）
```python
# 减少 close_bias 强度
turn_mw_cfg = CotMiddlewareConfig(
    ...
    close_bias_max=2.0,  # 从 4.0 降低到 2.0
)
```

### 如果是 D（采样问题）
检查：
- Tokenizer 是否正确加载
- sampling 参数是否合理
- 是否有 vocab_size 不匹配

---

## 自动化诊断（可选）

运行诊断脚本自动测试所有场景：
```bash
python diagnose_cot_mw.py --test all --prompt "who are you?" --runs 3
```

输出：`diagnose_results.json` 包含所有结果对比

---

## 日志阅读指南

启用了详细日志后，关键输出为：

```
[logits] first_logits(after transform): min=-5.32, max=12.45, mean=0.23
  ↑ 预期：max < 50，mean ≈ 0，这是正常的 logits 分布

[inj] injected_logits: min=-8.94, max=98.34, mean=1.23
  ↑ ⚠️  警告：max=98.34 太大了！可能导致采样崩溃

[inj] injected tok=52341
  ↑ token ID，应该 < vocab_size（通常 32000）

[logits] injected_logits(no_transform): min=-5.21, max=11.3, mean=0.19
  ↑ 采样前的 logits，应该合理范围
```

---

## 预期时间

- 📊 Step 1（日志）：5 分钟
- 🧪 Step 2（禁用注入）：10 分钟
- 🔧 Step 3（跳过二次 transform）：10 分钟
- 🎯 Step 4（禁用 format_guard）：10 分钟
- **总计：~45 分钟确认根本原因**

---

## 常见问题

**Q: 服务器卡住了怎么办？**  
A: `Ctrl+C` 停止，恢复 `server.py` 原版本（已自动备份）

**Q: 日志太多看不清？**  
A: 筛选关键日志：
```bash
./chat_precise.sh "..." 2>&1 | grep "\[logits\]\|\[inj\]\|\[mw\]"
```

**Q: 修改后忘记恢复了？**  
A: `git checkout mamba3_mlx/server.py`

---

## 下一步

确认根本原因后：
1. 应用对应的修复
2. 运行 3-5 次相同问题验证稳定性
3. 查看 `render_health_line()` 输出，应该显示健康的 token 分布

