# 🧪 CoT Middleware 不稳定性诊断实验指南

## 问题陈述

运行相同问题 ("who are you?") 时：
- ✅ **无 CoT MW**：输出稳定，逻辑清晰
- ❌ **启用 CoT MW**：输出破碎，token 损坏（##egressive, theight）

## 实验设计

### 假设
1. **H1**: `</final>` 注入的 logits 数值异常
2. **H2**: 注入的 logits 被二次调用 `transform_logits()` 导致溢出
3. **H3**: format_guard 的 close_bias 过度压低概率
4. **H4**: sampling 参数或 tokenizer 问题

### 实验流程（二分查找）

```
┌─ Baseline (CoT MW ON)
│  ├─ 破碎？YES → H1, H2, H3, or H4
│  └─ 破碎？NO → 问题已修复
│
├─ Test 1: 禁用 </final> 注入
│  ├─ 恢复？YES → 问题在注入逻辑（H1 or H2）
│  │  └─ Test 1.1: 禁用注入后的 transform_logits
│  │     ├─ 恢复？YES → H2（二次 transform）✅
│  │     └─ 恢复？NO → H1（inj_logits 本身异常）
│  └─ 恢复？NO → 问题在其他地方
│
└─ Test 2: 禁用 format_guard transform
   ├─ 恢复？YES → H3（close_bias 问题）
   └─ 恢复？NO → H4（采样或 tokenizer）
```

---

## 快速开始

### 准备工作
```bash
cd /Users/hungwei/Desktop/Proj/Mamba3-XR

# 确保代码已编译并加载调试日志
python3 -m py_compile mamba3_mlx/server.py

# 验证脚本存在
ls -la test_scenarios.sh QUICK_DIAGNOSIS_STEPS.md
```

### 方法 A: 自动化测试（推荐）
```bash
# 测试所有场景，每个 1 轮
./test_scenarios.sh all "who are you?" 1

# 输出对比分析
```

**预期结果**：
```
━━━ Baseline (CoT MW enabled) ━━━
  <think>...</think>
  <final>...</final>
  [benchmark] ... tok/s
  
━━━ Disabled </final> Injection ━━━
  <think>...</think>   ← 检查：是否恢复？
  [benchmark] ...
  
━━━ Disabled Transform on Injected Logits ━━━
  [检查输出质量]
  
━━━ Disabled entire CoT Middleware ━━━
  [检查输出质量，应该接近 Baseline]
```

### 方法 B: 手动测试（细粒度）
参考 `QUICK_DIAGNOSIS_STEPS.md`，逐步修改 `server.py`

---

## 实验执行

### 实验 1: 禁用 </final> 注入
**目的**: 隔离注入逻辑是否导致问题

**修改**:
```python
# server.py 第 929 行
if False and did_inject:  # 禁用
    # ... 注入代码
```

**运行**:
```bash
./chat_precise.sh "who are you?" 2>&1 | tail -30
```

**判断标准**:
- ✅ **输出恢复正常（有逻辑的 4 步推理）**
  → 问题确实在注入逻辑（H1 or H2）
  → 继续 Exp 1.1

- ❌ **输出仍破碎（##egressive 等）**
  → 问题不在注入
  → 跳到 Exp 2

---

### 实验 1.1: 禁用注入后的二次 transform
**前置条件**: Exp 1 恢复了 ✅

**目的**: 判断是否是二次 transform 导致的溢出

**修改** (server.py 第 895-920 行):
```python
        if use_inj:
            logits_1d = inj_logits
            use_inj   = False
            # ❌ 不调用 transform_logits
            tok = _sample(logits_1d, debug_label="injected(skip_transform)")
            log.info(f"[inj] sampled tok={tok}")
        else:
            decode_step_n += 1
            logits, mamba_states, kv_caches = decode_step(...)
            logits_1d = logits[0]
            # ✅ 只对普通 logits 调用
            logits_1d = mw.transform_logits(logits_1d)
            tok = _sample(logits_1d, debug_label="decode(with_transform)")
```

**运行**:
```bash
./chat_precise.sh "who are you?" 2>&1 | tail -30
```

**判断标准**:
- ✅ **恢复正常** → **根本原因：H2（二次 transform 导致溢出）**
  ```
  建议修复：
  server.py 第 895-920 行如上所示
  或在 CotMiddleware 中添加状态检查
  ```

- ❌ **仍破碎** → **根本原因：H1（inj_logits 本身异常）**
  ```
  建议修复：
  检查 CotMiddleware.maybe_inject_final() 中的 logits 生成
  或添加精度检查：
    if hasattr(last_row, 'astype'):
        last_row = last_row.astype('float32')
  ```

---

### 实验 2: 禁用 format_guard transform
**前置条件**: Exp 1 仍破碎 ❌

**目的**: 判断是否是 format_guard 的 close_bias 导致

**修改** (server.py 第 888-894 行):
```python
        # Apply format guard (ban mask + dynamic close-bias) before sampling
        # DIAGNOSTIC: Skip format guard for injected logits
        if not use_inj:
            logits_1d = mw.transform_logits(logits_1d)
        # else: skip for injected
        
        tok = _sample(logits_1d, debug_label="logits")
```

**运行**:
```bash
./chat_precise.sh "who are you?" 2>&1 | tail -30
```

**判断标准**:
- ✅ **恢复正常** → **根本原因：H3（format_guard 过度）**
  ```
  建议修复：
  降低 close_bias 强度，或改进 FSM 逻辑
  ```

- ❌ **仍破碎** → **根本原因：H4（采样或 tokenizer）**
  ```
  检查项：
  - Tokenizer 是否正确加载（vocab_size）
  - sampling 参数（temperature, top_k, top_p）
  - 是否有 token ID 超出范围
  ```

---

## 结果记录表

填写实验结果来快速定位问题：

| 实验 | 修改 | 输出状态 | 破碎/正常 | 根本原因 |
|------|------|--------|----------|--------|
| Baseline | 无 | 参考点 | ❌破碎 | 待查 |
| Exp 1 | 禁用注入 | | ✅/❌ | |
| Exp 1.1 | 跳过二次transform | | ✅/❌ | |
| Exp 2 | 禁用 format_guard | | ✅/❌ | |

**示例填写**:
```
Baseline | 无 | 破碎 | ❌ | -
Exp 1    | 禁用注入 | 恢复！ | ✅ | 在注入逻辑
Exp 1.1  | 跳过二次 | 更好了 | ✅ | H2: 二次 transform 溢出
```

---

## 关键日志输出

运行期间留意这些日志：

```bash
# 查看所有诊断日志
./chat_precise.sh "who are you?" 2>&1 | grep -E '\[logits\]|\[inj\]|\[mw\]'
```

**正常输出示例**:
```
[logits] first_logits(after transform): min=-5.32, max=12.45, mean=0.0234
[mw] first_tok=2342, mw.enabled=True
[inj] injected_logits: min=-8.94, max=11.32, mean=0.0156
[inj] injected tok=52341
[logits] injected_logits(no_transform): min=-5.21, max=10.3, mean=0.0189
[logits] decode_logits(after_transform): min=-6.10, max=9.87, mean=0.0142
```

**异常输出示例**:
```
[inj] injected_logits: min=-150.32, max=450.45, mean=23.45  ← ⚠️ 巨大值
[inj] injected tok=99999  ← ⚠️ 不合理的 token ID
```

---

## 修复清单

根据诊断结果选择修复：

### 修复 H2（二次 transform）
```python
# server.py 第 895-920 行，改为：
if use_inj:
    logits_1d = inj_logits
    use_inj = False
    # 直接采样，跳过 transform
    tok = _sample(logits_1d, ...)
else:
    # 普通路径
    logits_1d = decode_step(...)
    logits_1d = mw.transform_logits(logits_1d)
    tok = _sample(logits_1d, ...)
```

### 修复 H1（inj_logits 异常）
检查 `CotMiddleware.maybe_inject_final()`:
```python
# 添加精度检查
if hasattr(last_row, 'astype'):
    last_row = last_row.astype('float32')
    
# 或添加值范围检查
last_row = mx.clip(last_row, -50, 50)  # 限制范围
```

### 修复 H3（format_guard 过度）
```python
turn_mw_cfg = CotMiddlewareConfig(
    ...
    close_bias_max=2.0,  # 从 4.0 降低
    close_bias_value=1.0,  # 减弱偏差
)
```

---

## 时间估算

| 步骤 | 耗时 | 备注 |
|------|------|------|
| Exp 1（禁用注入） | ~5 min | 快速确认是否在注入 |
| Exp 1.1（跳过二次 transform） | ~5 min | 如果 Exp 1 恢复 |
| Exp 2（禁用 format_guard） | ~5 min | 如果 Exp 1 未恢复 |
| **总计** | **~15 min** | 确认根本原因 |

---

## 预期结果

实验完成后，你应该能够：

1. ✅ 识别出根本原因（H1, H2, H3, or H4）
2. ✅ 提供对应的代码修复
3. ✅ 验证修复后输出稳定性（多轮 token 一致）
4. ✅ 查看 CoT MW 健康报告（reasoning_tokens/final_tokens 平衡）

---

## 常见问题

**Q: 实验中途修改遗漏怎么办？**
```bash
git checkout mamba3_mlx/server.py
```

**Q: 输出还是不稳定怎么办？**
→ 检查是否多次修改冲突，使用 `git diff` 查看

**Q: 怎样验证修复有效？**
```bash
# 运行 5 轮相同问题，对比输出
for i in {1..5}; do ./chat_precise.sh "who are you?" 2>&1 | tail -5; done
```

---

## 下一步

修复后：
1. 将修改提交到 git
2. 更新 `COT_MW_INSTABILITY_DIAGNOSIS.md` 文档
3. 运行完整测试套件确保无回归
