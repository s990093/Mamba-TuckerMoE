# UI 測試 Prompts 側邊欄使用指南

## 功能說明

側邊欄現在顯示來自 `mock_config.json` 的所有測試示例，組織成不同的主題類別。

## 可用的主題類別

1. **Emotion** — 情感相關的提示（burnout、self_doubt、anxiety 等）
2. **Self-Awareness** — 自我認知問題（core_identity、architecture、hardware_awareness 等）
3. **Email Summary** — 郵件與摘要相關（draft_email、reply_email、summarize_meeting 等）
4. **Movie Intro** — 電影分析相關（analysis、comparative_analysis、recommendation_filter）
5. **Daily Conversation** — 日常對話（tech_troubleshoot、learning_assist、writing_help 等）
6. **System Call** — 系統調用示例（tool_trigger、tool_response）
7. **Deep Dive** — 深度分析報告（architecture_report、diagnostic_report）

## 如何使用

### 步驟 1: 選擇主題
側邊欄會自動展開所有可用的主題類別。

### 步驟 2: 選擇測試示例
在每個主題下，點擊你想要的示例。例如：
- 在 "Emotion" 下點擊 "I have been coding for 14 hours straight..."
- 在 "Self-Awareness" 下點擊 "Are you basically just Siri with extra steps?"

### 步驟 3: 查看填充的輸入框
測試提示會自動填充到輸入框中，你可以：
- 直接使用原始提示
- 編輯提示文字後再發送
- 選擇另一個示例替換

### 步驟 4: 手動送出
點擊 **Send** 按鈕來提交你的提示。

## UI 元素說明

```
左側邊欄
├─ Test Prompts (標籤)
│  ├─ Emotion
│  │  ├─ Step 1: **Identify the role** — ... (示例 1)
│  │  ├─ Step 2: **Confusingly worded thing** — ... (示例 2)
│  │  └─ ...
│  ├─ Self-Awareness
│  │  ├─ No. Siri is a **cloud-routed** ... (示例)
│  │  └─ ...
│  └─ [更多主題]
```

## 優點

✅ **快速測試** - 無需手動輸入，直接選擇示例  
✅ **分類組織** - 按主題分類，容易找到相關測試  
✅ **靈活編輯** - 可在發送前修改提示  
✅ **完整覆蓋** - 包含所有 mock_config.json 的示例  

## 技術細節

### 改變的行為

**之前**: 點擊示例 → 自動發送  
**現在**: 點擊示例 → 填充輸入框 + 聚焦輸入框 + 等待用戶點擊 Send

### 代碼位置

- HTML: `mamba3_mlx/ui/chat_demo.html` (line 60)
- JavaScript: `mamba3_mlx/ui/chat_demo.js` (line 811-820)
- 配置: `mamba3_mlx/ui/mock_config.json`

### 相關函數

```javascript
renderSidebarCategories(categories)  // 渲染側邊欄
// 點擊事件處理器現在只填充輸入框，不調用 doSend()
```

## 常見使用場景

### 場景 1: 測試情感相應能力
1. 在側邊欄點擊 "Emotion" → "I have been coding for 14 hours..."
2. 查看輸入框是否正確填充
3. 點擊 Send 測試 Mamba 的回應

### 場景 2: 測試特定主題的多個變體
1. 選擇一個主題（如 "Email Summary"）
2. 依次點擊多個示例，觀察模型對不同提示的回應
3. 每次可選擇編輯或保持原樣

### 場景 3: 驗證系統呼叫功能
1. 在 "System Call" 主題下選擇示例
2. 確認模型正確識別並調用系統功能
3. 查看 [SYSTEM_RESULT] 的處理

## 未來增強

可能的改進方向：
- 添加收藏夾功能（標記常用測試）
- 添加搜索/過濾（快速找到特定測試）
- 添加自定義提示保存（用戶創建的測試集）
- 添加測試結果比較（並排查看不同輸入的輸出）

---

**最後更新**: 2026-05-20  
**相關提交**: bd6a9a5
