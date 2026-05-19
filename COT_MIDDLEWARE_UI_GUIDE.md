# CoT Middleware UI Control Guide

## 新增功能

在 UI 前端添加了两个互补的控制开关，放在输入框底部的 `footer-ctrls` 区域：

### 1. **EOS No Stop** (已有，绿色)
- 位置：输入框下方的控制栏
- 作用：开启后，模型会在生成 `</final>` 后继续生成文本，而不是立即停止
- 指示器：绿色 toggle，标签显示 "EOS No Stop"

### 2. **CoT MW** (新增，蓝色)
- 位置：EOS No Stop 按钮旁边
- 作用：动态开启/关闭 CoT Middleware
  - **开启**（默认）：使用完整的 CoT Middleware，包括：
    - 格式守卫（ban `<|im_start|>`）
    - 动态 close-bias 偏差
    - 推理预算控制
    - `</final>` token 注入
  - **关闭**：禁用 Middleware，原始输出无约束
- 指示器：蓝色 toggle，标签显示 "CoT MW"
- 健康报告：当启用时，在输入框下方显示实时健康状态行

## 健康状态行 (Health Line)

当 CoT Middleware **启用**且生成完成后，会在输入框下方显示一行诊断信息：

```
[blue box]
reasoning_tokens=124 | final_phase_tokens=86 | health=0.96
```

这一行包含：
- **reasoning_tokens**: 推理块中生成的 token 数
- **final_phase_tokens**: `</final>` 块中生成的 token 数  
- **health**: Middleware 整体执行健康度（0-1，越高越好）

如果 CoT Middleware **禁用**，该行自动隐藏。

## UI 交互流程

### 基础聊天流程
1. 用户在输入框输入问题
2. 选择推理模式（Thinking / Direct）
3. **可选**：调整 EOS No Stop 和 CoT MW 开关
4. 点击发送或按 Enter

### 对比实验（推荐的用法）

#### 场景 A：默认设置（标准 CoT）
```
- Reasoning: ON (Thinking)
- EOS No Stop: OFF
- CoT MW: ON (蓝色，启用)
```
→ 查看健康报告，观察 reasoning_tokens 和 final_phase_tokens 的平衡

#### 场景 B：禁用 Middleware（原始输出）
```
- Reasoning: ON (Thinking)
- EOS No Stop: OFF
- CoT MW: OFF (灰色，禁用)
```
→ 没有健康报告；输出原始，无 CoT 格式守卫

#### 场景 C：Extended 生成（EOS No Stop）
```
- Reasoning: ON (Thinking)
- EOS No Stop: ON (绿色，启用)
- CoT MW: ON (蓝色，启用)
```
→ 模型在 `</final>` 后继续生成；健康报告显示延长的 token 数

## 技术细节

### 前端改动
- **HTML**：添加 `cot-mw-toggle` 控件和 `cot-health-line` 显示区域
- **CSS**：蓝色主题（`#64b4ff`），与 EOS No Stop 的绿色形成对比
- **JS**：
  - `cotMiddlewareEnabled` 状态变量
  - `doSend()` 发送 `enable_cot_middleware: false` 参数（仅在禁用时）
  - `handleMsg()` 处理 `cot_health_report` 并更新 UI

### 后端改动
- **server.py**：
  - 从 WS 消息中读取 `enable_cot_middleware` 参数
  - 根据该参数动态构建 `CotMiddlewareConfig(enabled=...)`
  - 在 `done` 事件中包含 `cot_health_report` 字段（使用 `render_health_line(mw.health_report())` 格式化）

## 调试建议

1. **查看原始输出**：禁用 CoT MW，观察模型是否自然输出 `<think>` 和 `<final>` 块
2. **测试格式守卫**：启用 CoT MW，尝试提示模型输出 `<|im_start|>`，应该被拦截
3. **监控健康状态**：多次运行同一问题，比较健康报告，识别不稳定的模式
4. **结合 EOS No Stop**：禁用 Middleware 但启用 EOS No Stop，观察无约束延续的行为

## 已知限制

- 健康报告仅在启用 CoT Middleware 时显示
- 如果 Middleware 初始化失败（见 server.py 日志），该控制会被忽略，回到基线行为
- 每次切换都需要重新发送消息；不影响已流式传输的输出

---

**版本**: 2026-05-20  
**联系**: 在 server.py 日志中查看 `[mw]` 前缀的诊断信息
