# Chat Demo — UI Controls

## 鍵盤快捷鍵

| 按鍵 | 效果 | 條件 |
|------|------|------|
| `P` | 切換 **Perf 浮動面板**（tok/s、TTFT、prefill 圖表） | 焦點不在輸入框時 |
| `Enter` | 送出訊息 | 焦點在輸入框 |
| `Shift + Enter` | 換行（不送出） | 焦點在輸入框 |
| `Escape` | 關閉 Reasoning mode 選單 | 選單開啟時 |

---

## 頂部列按鈕

| 按鈕 | 位置 | 效果 |
|------|------|------|
| ☰ **Sidebar** | 左上 | 展開 / 收起左側 example 列表 |
| 🗑 **Clear** | 右上工具列 | 清除對話歷史，重置 WebSocket session |
| ℹ **Info** | 右上工具列 | 開啟右側 **Drawer**（Metrics / Sampling / System prompt） |

---

## Reasoning Mode Pill（輸入框左上）

點一下 pill 展開選單，再點選項切換：

| 模式 | 意義 |
|------|------|
| **Thinking** | 推理模式 — 模型先輸出 `<think>…</think>` CoT，再給最終答案 |
| **Direct** | 直接回答 — 不注入 `<think>` block，速度較快 |

---

## 輸入框右側控制

| 控制項 | 效果 |
|--------|------|
| **EOS No Stop** 核取方塊 | 勾選後，遇到 `<\|im_end\|>` 不停止，繼續 decode（用於測試長輸出） |
| ▶ **Send**（圓形按鈕） | 送出訊息，等同 `Enter` |
| ⏹ **Stop** | 中途中斷當前 generation（送出 `abort` WS message） |

---

## Drawer 分頁（點 ℹ 開啟）

| 分頁 | 內容 |
|------|------|
| **Metrics** | 模型架構資訊（d\_model、layers、experts…）＋本次 turn 的 prefill/TTFT/tok/s |
| **Sampling** | 即時調整 temperature、top\_k、top\_p、min\_p、rep/pres/freq penalty；**Reset** 還原 CLI 預設值 |
| **System prompt** | 目前 category 的 SFT export system prompt（唯讀）；style constraints；system call registry |

---

## Sidebar Example 列表

| 互動 | 效果 |
|------|------|
| 點 category 標題 | 展開 / 收起該分類 |
| 點 example 項目 | 填入 prompt 並自動選取對應 category，點 Send 後使用匹配的 system prompt |
| **Play system prompt** 按鈕 | 以打字機動畫播放目前 category 的 system prompt（不送出 chat） |

---

## Perf 浮動面板（`P` 切換）

即時顯示：
- **tok/s** 折線圖（decode 速度）
- **Prefill ms** / **TTFT ms**
- **Total tokens** / prompt tokens
- KV cache / state 占用曲線（如果 profiler 有接入）
