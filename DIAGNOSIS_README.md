# 🔬 CoT Middleware 不稳定性诊断工具包

## 概览

你已获得完整的诊断工具包来快速定位 CoT Middleware 的输出破碎问题。

### 问题
启用 CoT Middleware 时：
- ❌ 输出破碎（token 损坏：##egressive, theight）
- ❌ 结构支离破碎，无逻辑
- ❌ 不稳定，重复运行有不同结果

禁用 CoT Middleware 时：
- ✅ 输出基本正常（虽然有其他问题，但至少结构完整）

---

## 工具包清单

### 📋 文档

| 文件 | 用途 |
|------|------|
| **EXPERIMENT_GUIDE.md** | 🎯 **从这里开始** - 完整实验流程和假设 |
| **QUICK_DIAGNOSIS_STEPS.md** | ⚡ 快速 4 步诊断（5-45 分钟） |
| **COT_MW_INSTABILITY_DIAGNOSIS.md** | 📊 根本原因分析和修复建议 |

### 🛠️ 脚本

| 脚本 | 用途 |
|------|------|
| **test_scenarios.sh** | 自动化测试所有 4 个场景 |
| **diagnose_cot_mw.py** | Python 诊断工具（高级） |

### 🔧 代码修改

已在 `server.py` 中添加的：
- ✅ 详细的 logits 数值日志（min/max/mean）
- ✅ 注入阶段的日志标记
- ✅ 采样路径的追踪

---

## 快速开始（3 步）

### Step 1: 阅读实验指南（5 分钟）
```bash
cat EXPERIMENT_GUIDE.md
```
理解二分查找的逻辑和 4 个假设。

### Step 2: 运行自动化测试（10 分钟）
```bash
./test_scenarios.sh all "who are you?" 1
```
查看 4 个场景的输出对比。

### Step 3: 根据结果进行修复（5 分钟）
参考 `COT_MW_INSTABILITY_DIAGNOSIS.md` 中的修复清单。

---

## 实验流程（简化版）

```
测试 Baseline (CoT MW ON)
    ↓
    输出破碎？ YES ↓
    ↓
禁用 </final> 注入 (Test 1)
    ↓
    恢复？ YES → 问题在注入逻辑
    ├─→ 禁用注入后的 transform (Test 1.1)
    │   ├─ 恢复？ YES → H2（二次 transform）✅
    │   └─ 恢复？ NO → H1（inj_logits 本身）
    │
    └─ 恢复？ NO → 问题在其他地方
        ↓
        禁用 format_guard (Test 2)
        ├─ 恢复？ YES → H3（close_bias 过度）
        └─ 恢复？ NO → H4（采样/tokenizer）
```

---

## 4 个假设

| 假设 | 症状 | 可能性 | 修复 |
|------|------|--------|------|
| **H1** | `inj_logits` 数值异常（极大值） | 🟡 中 | 精度检查/裁剪 |
| **H2** | 注入的 logits 被二次 transform，导致溢出 | 🔴 高 | 跳过二次 transform |
| **H3** | format_guard 的 close_bias 过度压低概率 | 🟡 中 | 降低 close_bias |
| **H4** | sampling 参数或 tokenizer 不匹配 | 🟢 低 | 检查采样/vocab |

---

## 运行诊断

### 方式 A: 自动化（推荐）
```bash
# 运行所有 4 个场景
./test_scenarios.sh all "who are you?" 1

# 输出会显示每个场景的前 15 行
# 对比观察是否恢复
```

### 方式 B: 手动（细粒度）
按照 `QUICK_DIAGNOSIS_STEPS.md` 逐步修改 `server.py` 并测试。

### 方式 C: 高级（Python）
```bash
python diagnose_cot_mw.py --test all --prompt "who are you?" --runs 3
# 生成 diagnose_results.json 详细报告
```

---

## 日志阅读

启用调试日志后，运行：
```bash
./chat_precise.sh "who are you?" 2>&1 | grep -E '\[logits\]|\[inj\]|\[mw\]'
```

**关键数值**：
```
[logits] first_logits(after transform): min=-5.32, max=12.45, mean=0.023
  ↑ 预期：max < 50，mean ≈ 0

[inj] injected_logits: min=-8.94, max=98.34, mean=1.23
  ↑ 警告：max > 50 可能导致采样崩溃

[inj] injected tok=52341
  ↑ 应该 < vocab_size（32000）
```

---

## 快速修复（如果确认是 H2）

编辑 `server.py` 第 895-920 行：

```python
        if use_inj:
            logits_1d = inj_logits
            use_inj   = False
            # ❌ 删除这行：logits_1d = mw.transform_logits(logits_1d)
            tok = _sample(logits_1d, ...)
        else:
            decode_step_n += 1
            logits, mamba_states, kv_caches = decode_step(...)
            logits_1d = logits[0]
            # ✅ 只在普通路径调用 transform
            logits_1d = mw.transform_logits(logits_1d)
            tok = _sample(logits_1d, ...)
```

---

## 预期时间

- 📖 阅读指南：5 分钟
- 🧪 自动化测试：10-15 分钟
- 🔍 手动诊断：20-45 分钟（如果需要细粒度）
- 🔧 应用修复：5 分钟
- ✅ 验证修复：10 分钟

**总计：1-1.5 小时确认并修复问题**

---

## 验证修复

修复后，验证稳定性：

```bash
# 运行 5 轮相同问题
for i in {1..5}; do
    echo "=== Run $i ==="
    ./chat_precise.sh "who are you?" 2>&1 | grep "Assistant:" -A 5
done

# 对比输出，应该结构一致（token 可能略有不同）
```

查看 UI 中的 CoT MW 健康报告：
```
[blue box] reasoning_tokens=124 | final_tokens=86 | health=0.96
```

---

## 常见问题

**Q: 我应该先测试哪个场景？**  
A: 按顺序：baseline → no-injection → no-transform → no-mw

**Q: 如果所有场景都破碎怎么办？**  
A: 可能是 H4（采样或 tokenizer），检查：
```bash
grep -n "vocab_size\|sampling\|tokenizer" mamba3_mlx/server.py
```

**Q: 怎样恢复原始状态？**  
A: 脚本会自动备份，或：
```bash
git checkout mamba3_mlx/server.py
```

**Q: 健康报告一直是 0 怎么办？**  
A: 可能 CoT MW 未启用，检查：
```bash
grep "enable_cot_mw\|turn_mw_cfg.enabled" mamba3_mlx/server.py
```

---

## 下一步

1. ✅ 运行诊断确认根本原因
2. ✅ 应用对应修复
3. ✅ 验证多轮稳定性
4. ✅ 提交 git 并更新文档

---

## 文件位置

所有文件在：`/Users/hungwei/Desktop/Proj/Mamba3-XR/`

```
.
├── DIAGNOSIS_README.md                 ← 你在这里
├── EXPERIMENT_GUIDE.md                 ← 开始这个
├── QUICK_DIAGNOSIS_STEPS.md            ← 快速参考
├── COT_MW_INSTABILITY_DIAGNOSIS.md     ← 修复建议
├── test_scenarios.sh                   ← 自动化测试
├── diagnose_cot_mw.py                  ← 高级诊断
└── mamba3_mlx/
    └── server.py                       ← 已添加日志
```

---

## 支持

遇到问题？检查：
1. `QUICK_DIAGNOSIS_STEPS.md` 的常见问题部分
2. 日志输出（grep `[logits]`, `[inj]`, `[mw]`）
3. 确保 `server.py` 未被其他修改污染

---

**开始诊断吧！** 👉 `cat EXPERIMENT_GUIDE.md`
