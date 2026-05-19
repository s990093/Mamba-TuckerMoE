# 🔬 CoT Middleware 诊断工具包 - 完整总结

## 📦 你已获得的工具

### 核心诊断文档（新创建）
| 文件 | 大小 | 用途 |
|------|------|------|
| **START_HERE.txt** | 4.8K | ⭐ 快速开始指南 |
| **DIAGNOSIS_README.md** | 6.2K | 📖 工具包概览 |
| **EXPERIMENT_GUIDE.md** | 8.2K | 🎯 完整实验设计 |
| **QUICK_DIAGNOSIS_STEPS.md** | 6.3K | ⚡ 4 步快速诊断 |
| **COT_MW_INSTABILITY_DIAGNOSIS.md** | 6.2K | 📊 根本原因分析 |

### 自动化诊断脚本（新创建）
| 脚本 | 大小 | 用途 |
|------|------|------|
| **test_scenarios.sh** | 5.2K | 🧪 自动化测试 4 个场景 |
| **diagnose_cot_mw.py** | 7.6K | 🔧 Python 诊断工具 |

### 代码修改
| 文件 | 修改内容 |
|------|----------|
| **server.py** | ✅ 添加详细 logits 日志（min/max/mean） |
| **server.py** | ✅ 添加注入阶段追踪 |
| **server.py** | ✅ 添加采样路径标记 |
| **chat_demo.html** | ✅ 添加 CoT MW 控制 toggle |
| **chat_demo.css** | ✅ 添加蓝色主题样式 |
| **chat_demo.js** | ✅ 添加 MW 状态管理和健康报告 |

---

## 🎯 诊断流程

```
┌─ 读文档 (START_HERE.txt)
│
├─ 选择诊断方式
│  ├─ 自动化：./test_scenarios.sh all
│  ├─ 手动：按 QUICK_DIAGNOSIS_STEPS.md
│  └─ 高级：python diagnose_cot_mw.py
│
├─ 运行测试，对比 4 个场景
│  ├─ Baseline（CoT MW ON）
│  ├─ No-injection（禁用注入）
│  ├─ No-transform（跳过二次 transform）
│  └─ No-mw（禁用整个 MW）
│
├─ 观察输出，判断恢复点
│  ├─ 在 No-injection 恢复？→ 问题在注入逻辑
│  ├─ 在 No-transform 恢复？→ H2（二次 transform）
│  ├─ 在 No-mw 恢复？→ 问题在 format_guard
│  └─ 都不恢复？→ 问题在采样/tokenizer
│
└─ 应用修复，验证稳定性
```

---

## 🔍 4 个假设及可能性

| 假设 | 描述 | 可能性 | 症状 | 修复位置 |
|------|------|--------|------|----------|
| **H1** | inj_logits 数值异常 | 🟡 中 | 极大值（>50） | CotMiddleware |
| **H2** | 二次 transform 溢出 | 🔴 高 | token 损坏 | server.py:895-920 |
| **H3** | format_guard 过度 | 🟡 中 | 采样崩溃 | server.py:777-782 |
| **H4** | sampling/tokenizer | 🟢 低 | 不合理 token | 采样配置 |

---

## ⚡ 快速开始（3 步，15 分钟）

### Step 1: 阅读
```bash
cat START_HERE.txt
```

### Step 2: 运行自动化测试
```bash
./test_scenarios.sh all "who are you?" 1
```

### Step 3: 查看结果，应用修复
```bash
# 对比 4 个场景的输出
# 查看在哪个场景恢复正常
# 参考 QUICK_DIAGNOSIS_STEPS.md 中的修复表
```

---

## 📊 关键日志

启用调试后，运行：
```bash
./chat_precise.sh "who are you?" 2>&1 | grep -E '\[logits\]|\[inj\]|\[mw\]'
```

**关键观察点**：
- `[logits]` 数值是否合理（min/max/mean）
- `[inj]` 注入的 logits 是否异常（max > 50？）
- `[inj] injected tok` token ID 是否合理（< 32000？）

---

## ✅ UI 新功能

已添加到前端：

### CoT MW 控制 Toggle
- 位置：输入框下方，EOS No Stop 按钮旁
- 颜色：蓝色（与 EOS 的绿色区分）
- 作用：动态开启/关闭 CoT Middleware

### 健康报告行
- 显示条件：启用 CoT MW 且生成完成
- 内容：`reasoning_tokens=124 | final_tokens=86 | health=0.96`
- 自动隐藏：禁用 CoT MW 时隐藏

### 代码改动
```javascript
// JS 新增
cotMiddlewareEnabled = true
enable_cot_middleware: false  // 发送给后端

// 后端响应
"cot_health_report": "reasoning_tokens=... | final_tokens=... | health=..."
```

---

## 🔧 修复清单

根据诊断结果选择修复：

### 修复 H2（最可能）
```python
# server.py 第 895-920 行
if use_inj:
    logits_1d = inj_logits
    use_inj = False
    # ❌ 删除：logits_1d = mw.transform_logits(logits_1d)
    tok = _sample(logits_1d, ...)
else:
    logits_1d = logits[0]
    logits_1d = mw.transform_logits(logits_1d)  # ✅ 只这里调用
    tok = _sample(logits_1d, ...)
```

### 修复 H1
```python
# 在 CotMiddleware.maybe_inject_final() 中
if hasattr(last_row, 'astype'):
    last_row = last_row.astype('float32')
```

### 修复 H3
```python
# server.py 第 777-782 行
mw_cfg = CotMiddlewareConfig(
    ...
    close_bias_max=2.0,  # 从 4.0 降低
)
```

---

## 📈 验证修复

修复后验证（5 轮相同问题）：
```bash
for i in {1..5}; do 
    echo "=== Run $i ===" 
    ./chat_precise.sh "who are you?" 2>&1 | tail -5
done
```

应看到：**结构一致，token 可能略有不同**

---

## 📚 文档导航

```
START_HERE.txt
    ↓
DIAGNOSIS_README.md ← 概览和工具包内容
    ↓
EXPERIMENT_GUIDE.md ← 实验设计和流程（最详细）
    ↓
QUICK_DIAGNOSIS_STEPS.md ← 4 步快速诊断
    ↓
COT_MW_INSTABILITY_DIAGNOSIS.md ← 根本原因和修复代码
```

---

## ⏱️ 预期时间

| 步骤 | 耗时 |
|------|------|
| 阅读文档 | 5 分钟 |
| 自动化测试 | 10-15 分钟 |
| 根本原因确认 | 5 分钟 |
| 修复实现 | 5 分钟 |
| 验证修复 | 10 分钟 |
| **总计** | **35-40 分钟** |

---

## 🚀 立即开始

```bash
# 1. 查看快速开始指南
cat START_HERE.txt

# 2. 运行自动化诊断（推荐）
./test_scenarios.sh all "who are you?" 1

# 3. 或按步骤手动诊断
cat QUICK_DIAGNOSIS_STEPS.md
```

---

## 常见问题

**Q: 哪个文件我应该先看？**
```
START_HERE.txt (快速概览)
  ↓ 然后
EXPERIMENT_GUIDE.md (详细步骤)
```

**Q: 我没时间怎么办？**
```
直接运行：./test_scenarios.sh all
输出会显示每个场景的对比
```

**Q: 修复后怎样验证？**
```
运行 5 轮相同问题，检查输出结构一致性
```

---

## 文件检查清单

```
✅ START_HERE.txt                          (4.8K)
✅ DIAGNOSIS_README.md                     (6.2K)
✅ EXPERIMENT_GUIDE.md                     (8.2K)
✅ QUICK_DIAGNOSIS_STEPS.md                (6.3K)
✅ COT_MW_INSTABILITY_DIAGNOSIS.md         (6.2K)
✅ test_scenarios.sh                       (5.2K, 可执行)
✅ diagnose_cot_mw.py                      (7.6K, 可执行)
✅ server.py                               (已修改，含日志)
✅ chat_demo.html/.css/.js                 (已修改，新增 UI)
```

---

## 技术总结

### 问题性质
**输出破碎** → 典型的 logits 数值问题（溢出/饱和）

### 诊断方法
**二分查找** → 通过禁用子系统逐步隔离根本原因

### 修复方向
**4 个假设** → 根据诊断结果对应 4 种修复策略

### 验证方法
**多轮稳定性测试** → 确保修复有效且可重复

---

## 下一步

1. ✅ 运行诊断（15-40 分钟）
2. ✅ 确认根本原因
3. ✅ 应用修复
4. ✅ 验证稳定性
5. ✅ 提交 git 并更新文档

**预计总耗时：1-1.5 小时**

---

祝你诊断顺利！🧪

有问题？查看 START_HERE.txt 或 DIAGNOSIS_README.md
