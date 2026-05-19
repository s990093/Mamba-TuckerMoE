# infer_cot.py 诊断指南

## 当前问题

运行后显示：
```
Tokens: 11
Output: "PosPositivePositive..."
Has reasoning: True
Has final answer: False
```

这表示模型采样出了错误的 token 序列。

---

## 快速诊断步骤

### 步骤 1：验证 Format Guard 初始化

✅ 你看到这个，说明 format guard 初始化正确：
```
Format guard: enabled · close_bias=[think→32003, final→32005]
```

如果看到 `close_bias=[think→829, final→829]` 就有问题。

### 步骤 2：检查 Middleware Events

运行（已添加调试输出）：
```bash
python -m mamba3_mlx.infer_cot --prompt "Who are you?" --category self_awareness 2>&1 | grep -A 30 "Step 0"
```

查看输出中的 `[Step 0] reasoning:` 和 `[Step 1] reasoning:` 等。

**好的输出应该是：**
```
[Step 0] reasoning: 'Let me think'
[Step 1] reasoning: ' about my identity'
```

**坏的输出是：**
```
[Step 0] reasoning: 'Pos'
[Step 1] reasoning: 'Positive'
```

### 步骤 3：检查采样的 Token ID

在你的 Mac 上，修改 `infer_cot.py` 第 125 行附近，在采样后添加：

```python
tid = sample_token(logits_row, temperature=temperature, top_k=40)
generated_ids.append(tid)

# 临时调试：查看 token ID 和对应的文本
if step_idx < 5:
    chunk = self.tokenizer.decode([tid], skip_special_tokens=False)
    print(f"  [DEBUG] Step {step_idx}: tid={tid}, chunk={repr(chunk)}")
```

然后运行。你会看到：
```
[DEBUG] Step 0: tid=12345, chunk='Let'
[DEBUG] Step 1: tid=678, chunk=' me'
```

或者（坏的）：
```
[DEBUG] Step 0: tid=99999, chunk='Pos'
[DEBUG] Step 1: tid=99999, chunk='Pos'
```

---

## 可能的根本原因分析

### 原因 1：Logits 被破坏（可能性 40%）

**症状：**
- 采样出的 token ID 都很大或很小
- 重复采样同一个 token

**检查方法：**
在 `middleware.transform_logits()` 后添加：
```python
if step_idx == 0:
    print(f"Logits stats: min={mx.min(logits_row)}, max={mx.max(logits_row)}, mean={mx.mean(logits_row)}")
```

**好的应该是：**
```
Logits stats: min=-10.5, max=8.3, mean=-0.2
```

### 原因 2：Close Bias 过强（可能性 30%）

**症状：**
- 总是采样同一个 token（比如 32003 或某个其他 token）

**检查方法：**
在 `middleware.transform_logits()` 前后检查 logits 的变化：
```python
if step_idx == 0:
    print(f"Before bias: logits_row shape={logits_row.shape}, sample logits: {logits_row[:10]}")
logits_row = middleware.transform_logits(logits_row)
if step_idx == 0:
    print(f"After bias: {logits_row[:10]}")
```

### 原因 3：Token Decode 问题（可能性 20%）

**症状：**
- 采样的 token ID 看起来对，但解码成垃圾

**检查方法：**
```python
tid = sample_token(...)
chunk = self.tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
chunk2 = self.tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=True)
if chunk != chunk2:
    print(f"Decode mismatch: {repr(chunk)} vs {repr(chunk2)}")
```

### 原因 4：Middleware Events 没有正确返回（可能性 10%）

**症状：**
- middleware.step() 返回的事件为空或没有 "markdown" 字段

**检查方法：**
```python
for event in middleware.step(tid, ...):
    print(f"Event: {event.keys()}")  # 看看有什么字段
```

应该看到：
```
Event: dict_keys(['type', 'markdown']) 
或
Event: dict_keys(['type', 'text'])
或
Event: dict_keys(['__stop__'])
```

---

## 完整诊断脚本

在 `infer_cot.py` 的 `infer()` 方法中，将 decode loop 替换为这个诊断版本：

```python
for step_idx in range(max_tokens):
    # Transform logits through middleware format guard
    if logits.ndim == 3:
        logits_row = logits[0, -1, :]
    else:
        logits_row = logits[0, :] if logits.ndim == 2 else logits
    
    # DEBUG: Check logits before bias
    if step_idx == 0:
        print(f"\n[DIAGNOSTIC] Logits before bias:")
        print(f"  Shape: {logits_row.shape}")
        print(f"  Min: {mx.min(logits_row).item():.2f}, Max: {mx.max(logits_row).item():.2f}")
        print(f"  Sample values: {logits_row[:5].tolist()}")
    
    logits_row = middleware.transform_logits(logits_row)
    mx.eval(logits_row)
    
    # DEBUG: Check logits after bias
    if step_idx == 0:
        print(f"\n[DIAGNOSTIC] Logits after bias:")
        print(f"  Min: {mx.min(logits_row).item():.2f}, Max: {mx.max(logits_row).item():.2f}")
        print(f"  Sample values: {logits_row[:5].tolist()}")

    # Sample
    tid = sample_token(logits_row, temperature=temperature, top_k=40)
    generated_ids.append(tid)
    
    # DEBUG: Check sampled token
    chunk = self.tokenizer.decode([tid], skip_special_tokens=False)
    if step_idx < 5:
        print(f"\n[DIAGNOSTIC] Step {step_idx}:")
        print(f"  Token ID: {tid}")
        print(f"  Decoded: {repr(chunk)}")
        print(f"  Token (tokenizer): {self.tokenizer.convert_ids_to_tokens(tid)}")

    # Let middleware process
    elapsed_ms = (time.perf_counter() - t0) * 1000
    for event in middleware.step(tid, n_out=len(generated_ids), elapsed_s_fn=lambda: elapsed_ms / 1000):
        if step_idx < 5:
            print(f"  Event: {event}")
        
        if event.get("__stop__"):
            stopped = True
            stop_reason = "middleware_stop"
        elif event.get("type") == "reasoning":
            reasoning_text += event.get("markdown", "")
        elif event.get("type") == "token":
            final_text += event.get("text", "")

    # 构建 raw_text
    raw_text = self.tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )

    if middleware.should_break(tid):
        stop_reason = "stop_token"
        stopped = True

    if stopped:
        break

    # Decode step
    try:
        logits, mamba_states, kv_caches = decode_step(
            self.model, tid, mamba_states, kv_caches, step=seq_pos
        )
        mx.eval(logits)
    except Exception as e:
        stop_reason = f"decode_error: {e}"
        break
    seq_pos += 1
```

---

## 运行诊断

1. 修改 `infer_cot.py` 使用上面的诊断脚本
2. 运行：
   ```bash
   python -m mamba3_mlx.infer_cot --prompt "Who are you?" --category self_awareness 2>&1 | tee diagnosis_output.txt
   ```
3. 检查前 5-10 步的输出
4. 看看：
   - Token ID 是否合理（应该在 0-32007 范围）
   - Decoded chunk 是否有意义
   - Middleware events 是否正确返回

---

## 我的猜测

基于"PosPositivePositive"这个重复模式，我猜测：

**最可能：** 采样总是返回同一个 token ID（比如某个对应"Positive"的 token）
- 这可能是 close_bias 过强
- 或者是某个特定 logit 值导致采样总是选择它

**次可能：** Token ID 没问题，但 tokenizer.decode() 有问题
- 检查 tokenizer 版本
- 检查是否有 tokenizer 的特殊配置

---

## 需要的信息

运行诊断后，请提供：

1. **前 5 步的输出**（Step 0-4）
2. **Logits 统计**（min/max before/after bias）
3. **Sampled Token IDs**（每一步采样的 ID）
4. **Middleware Events**（每一步返回的事件）

这样我能精确定位问题所在。

---

**下一步：** 在你的 Mac 上运行这个诊断，告诉我看到什么。
