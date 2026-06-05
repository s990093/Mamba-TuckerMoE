# EYES — 雙眼睛語音助手介面 實作計畫

## 一、概念

一個極簡的語音互動介面，畫面只有**一雙眼睛**。使用語音輸入問題，眼睛透過動態表情即時反映 LLM 的內部狀態（聆聽 / 思考 / 回應），回應以語音合成播放。

- 無鍵盤、無文字輸入框
- 無聊天記錄區塊
- 雙眼就是唯一的 UI 元素
- 語音就是唯一的輸入/輸出通道

---

## 二、架構概覽

```
┌──────────────────────────────────────────────┐
│  瀏覽器 (eyes.html)                           │
│  ┌────────────┐  ┌────────────┐              │
│  │ Eye Canvas │  │ Web Audio  │              │
│  │  (SVG/CSS) │  │  (TTS播出) │              │
│  └────────────┘  └────────────┘              │
│  ┌─────────────────────────────┐             │
│  │  Voice Controller           │             │
│  │  - SpeechRecognition (STT)  │             │
│  │  - SpeechSynthesis  (TTS)   │             │
│  │  - Wake-word detection      │             │
│  └─────────────────────────────┘             │
│  ┌─────────────────────────────┐             │
│  │  WebSocket Client           │             │
│  │  ws://host:7860/ws          │             │
│  └─────────────────────────────┘             │
└──────────────────────┬───────────────────────┘
                       │ WebSocket
┌──────────────────────▼───────────────────────┐
│  chat_demo.py (現有後端，不改)                 │
│  FastAPI + WebSocket                         │
│  Mamba3LanguageModel + CoT Middleware         │
└──────────────────────────────────────────────┘
```

**關鍵原則：後端零改動。** 完全復用現有的 `/ws` WebSocket 協議和 `chat_demo.py`。

---

## 三、檔案規劃

全部新增在 `mamba3_mlx/ui/` 下，不碰現有檔案：

```
mamba3_mlx/ui/
├── chat_demo.html          # 已有，不動
├── chat_demo.css           # 已有，不動
├── chat_demo.js            # 已有，不動
├── mock_config.json        # 已有，不動
├── claude-color.svg        # 已有，不動
│
├── eyes.html               # ★ 新增：主頁面
├── eyes.css                # ★ 新增：眼睛 + 整體樣式
├── eyes.js                 # ★ 新增：語音控制 + WebSocket + 動畫
└── EYES_PLAN.md            # ★ 本文件
```

後端新增一個 route（改 `chat_demo.py` 加 3 行）：

```python
@app.get("/eyes", response_class=HTMLResponse)
async def eyes_page():
    return HTMLResponse((_UI_DIR / "eyes.html").read_text(encoding="utf-8"))
```

---

## 四、眼睛動畫系統

### 4.1 技術方案：SVG + CSS Animation + JavaScript 狀態機

使用 SVG 繪製雙眼，CSS animation 處理週期性動畫（眨眼），JS 根據 WebSocket 狀態切換 class。

### 4.2 眼睛結構（單眼 SVG 元件）

```svg
<svg viewBox="0 0 200 120">
  <!-- 眼白 -->
  <ellipse cx="100" cy="60" rx="90" ry="55" fill="#f5f0e8" />

  <!-- 虹膜 (可動) -->
  <circle cx="100" cy="60" r="28" fill="#3a3226">
    <!-- 瞳孔 -->
    <circle cx="100" cy="60" r="12" fill="#1a1510" />
    <!-- 高光 -->
    <circle cx="94" cy="52" r="6" fill="rgba(255,255,255,0.7)" />
  </circle>

  <!-- 上眼瞼 (動態閉合) -->
  <path class="upper-lid" d="M10,60 Q100,-10 190,60" fill="#262624" />

  <!-- 下眼瞼 -->
  <path d="M10,60 Q100,130 190,60" fill="#262624" />
</svg>
```

雙眼在畫面上水平並排，間距約 1 倍眼寬。

### 4.3 狀態機 — 眼睛表情對應 LLM 狀態

| 狀態 | 觸發條件 | 眼睛動畫 | 持續時間 |
|------|---------|----------|---------|
| **idle** | 頁面載入，無對話 | 緩慢眨眼 (4-6s 間隔)，虹膜微微左右漂移 | 永久 |
| **listening** | 使用者按住/觸發語音輸入 | 虹膜放大 15%，微微向上凝視，眨眼暫停 | 語音輸入期間 |
| **processing** | WebSocket 已發送，等待第一個 token | 虹膜快速左右掃動，眨眼頻率提高 (1-2s)，瞳孔縮小 | 直到收到 meta 或 token |
| **thinking** | CoT 階段 (type: reasoning) | 虹膜朝上偏轉，瞳孔收窄，不規則慢眨眼，偶爾「轉一圈」 | CoT 長度 |
| **speaking** | final 階段 (type: token) | 虹膜回復正常，眨眼恢復正常節奏，跟隨語音韻律微微放大縮小 | 直到 done |
| **done** | type: done | 短暫閉眼 0.3s (滿足/確認)，回到 idle | 瞬時 |
| **error** | type: error | 快速連眨兩次，虹膜朝左下偏移 | 1.5s 後回 idle |
| **sleep** | 60s 無互動 | 閉眼，進入慢呼吸節奏 (慢速開合) | 直到喚醒 |

### 4.4 動畫參數

```javascript
const EYE_CONFIG = {
  blinkIntervalIdle:   [4000, 6000],  // ms, random range
  blinkIntervalActive: [2000, 3500],
  blinkDuration:        120,           // ms, 閉眼→開眼
  irisDriftRadius:      6,             // px, idle 時虹膜漂移半徑
  irisScanSpeed:        40,            // px/s, thinking 掃動速率
  irisLookUpAngle:      -8,            // deg, listening 上望角度
  sleepTimeout:         60000,         // ms, 進入休眠
};
```

---

## 五、語音互動流程

### 5.1 喚醒方式

採用 **按住說話 (Push-to-Talk)** 作為主要模式：

- 桌面：按住 **空白鍵** 開始收音，放開結束
- 觸控：按住畫面任意處開始收音，放開結束
- 備用：點擊喚醒詞按鈕（小圖示在眼睛下方）

原因：Web Speech API 的 continuous recognition 在瀏覽器中不可靠，PTT 模式最穩定。

### 5.2 完整對話流程

```
1. idle (雙眼緩慢眨眼)
       │ 使用者按住空白鍵
       ▼
2. listening (虹膜放大，凝視上方)
       │ SpeechRecognition 收音
       │ 放開空白鍵 → 取得文字
       ▼
3. 發送 WebSocket { action: "chat", prompt: "...", reasoning: true }
       │
       ▼
4. processing (虹膜快速掃動)
       │ 收到 meta / reasoning / token
       ├── thinking (CoT 階段，眼睛顯示思考狀態)
       │      │ TTS 不播音 (或小聲播放 reasoning 摘要)
       │      ▼
       └── speaking (final 階段，眼睛顯示說話狀態)
              │ TTS 逐句播放回應文字
              │ 累積到標點符號或換行時觸發一次 utterance
              ▼
5. done (短暫閉眼，回到 idle)
```

### 5.3 TTS 播放策略

```javascript
// 不使用 token-level TTS（太破碎）
// 改以句子為單位累積，遇到以下符號時觸發播放：
const TTS_BREAK_CHARS = /[。！？.!?\n]/;

// 串流 token 時累積到 buffer
// 每當 buffer 包含 break char，擷取完整句子送 SpeechSynthesis
// 播放完成後才播下一句（排隊機制）
```

### 5.4 語音辨識處理

```javascript
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
// 設定：
//   lang: 'zh-TW' (或自動偵測)
//   interimResults: false (只取最終結果)
//   continuous: false
//   maxAlternatives: 1
```

---

## 六、WebSocket 協議（完全復用現有）

### 發送

```json
{
  "action": "chat",
  "prompt": "語音辨識出來的文字",
  "max_tokens": 512,
  "reasoning": true,
  "category_key": "daily_conversation"
}
```

### 接收（關鍵事件處理）

| type | 前端行為 |
|------|---------|
| `connected` | 確認連線，眼睛 idle |
| `meta` | processing → thinking，顯示 prefill 時間 |
| `reasoning` | 眼睛切換 thinking 動畫 |
| `token` | 眼睛切換 speaking 動畫，文字累積到 TTS buffer |
| `mw_inject` | 眼睛快速旋轉一圈（<final> 注入提示） |
| `done` | 眼睛閉合→idle，播放剩餘 TTS buffer |
| `error` | 眼睛連眨兩次，可選的語音提示「再試一次」 |

---

## 七、視覺設計

### 7.1 配色

沿用現有 Claude 風格暗色主題：

```
背景: #1a1918 (比 chat 更暗，讓眼睛更突出)
虹膜: #cc785c (terracotta accent)
瞳孔: #1a1510
眼白: #f5f0e8 (微暖白)
光暈: radial-gradient with rgba(204,120,92,0.15)
```

### 7.2 佈局

```
┌─────────────────────────────────┐
│                                 │
│                                 │
│        ┌─────┐    ┌─────┐      │
│        │ 左眼 │    │ 右眼 │      │
│        └─────┘    └─────┘      │
│                                 │
│         [狀態指示燈/文字]         │
│                                 │
│        (按住空白鍵說話)           │
│                                 │
└─────────────────────────────────┘
```

- 雙眼水平居中，佔畫面 40-50% 寬度
- 眼睛下方：小型狀態文字（"聆聽中..." / "思考中..." / "回應中..."）
- 最下方：操作提示（"Hold Space to speak"）
- 無其他 UI 元素

### 7.3 響應式

- Desktop (>768px)：雙眼 200x120 視口
- Mobile (<768px)：雙眼 140x84 視口，觸控長按取代空白鍵
- 全螢幕 API：建議使用者加到主畫面（PWA 模式）

---

## 八、實作階段

### Phase 1：核心骨架（預計 2-3 小時）

**檔案：`eyes.html`, `eyes.css`, `eyes.js`**

- [ ] 建立 HTML 結構 (雙眼 SVG + 狀態文字 + 提示)
- [ ] CSS：暗色背景、眼睛基礎樣式、置中佈局
- [ ] JS：WebSocket 連線復用現有協議
- [ ] JS：狀態機基礎框架 (`setEyeState(state)`)
- [ ] 確認能連上 `ws://localhost:7860/ws` 收發訊息

### Phase 2：眼睛動畫（預計 3-4 小時）

- [ ] CSS animation：眨眼 (blink keyframes)
- [ ] JS：虹膜漂移 (idle 時的隨機緩慢移動)
- [ ] JS：虹膜掃動 (processing/thinking 時的快速移動)
- [ ] JS：瞳孔縮放 (listening 放大, thinking 縮小)
- [ ] JS：狀態切換過渡 (ease-in-out 0.3s)
- [ ] JS：閉眼 sleep 模式

### Phase 3：語音整合（預計 3-4 小時）

- [ ] SpeechRecognition：按住空白鍵觸發收音
- [ ] 收音視覺反饋（虹膜放大 + 上望）
- [ ] SpeechSynthesis：句子級 TTS 排隊播放
- [ ] TTS 與眼睛 speaking 狀態同步
- [ ] 觸控支援（長按畫面）

### Phase 4：細部打磨（預計 2-3 小時）

- [ ] 喚醒詞過渡動畫（idle→listening 的平滑轉換）
- [ ] 光暈效果（thinking 時虹膜周圍的微光呼吸）
- [ ] 狀態指示文字淡入淡出
- [ ] 音效提示（可選：輕微的聆聽開始/結束音）
- [ ] PWA manifest (加到主畫面)
- [ ] Mobile 響應式調整
- [ ] 異常處理（WebSocket 斷線、STT 失敗）

---

## 九、技術風險與對策

| 風險 | 影響 | 對策 |
|------|------|------|
| Safari 不支援 SpeechRecognition | iOS 無法語音輸入 | 降級為文字輸入框（暫存方案），或提示使用 Chrome |
| SpeechSynthesis 聲音品質差 | 體驗打折 | 提供語音選擇下拉，選用系統高品質聲音 |
| WebSocket 串流 + TTS 排隊不同步 | 眼睛動畫與語音錯位 | 以 token type 為主要狀態來源，TTS buffer 獨立管理 |
| SVG 動畫在 mobile 掉幀 | 體驗卡頓 | 使用 `will-change: transform` + `requestAnimationFrame` 優化 |

---

## 十、後端改動（極小）

只需在 `chat_demo.py` 加一個 route：

```python
# 放在 index() route 附近
@app.get("/eyes", response_class=HTMLResponse)
async def eyes_page():
    html = (_UI_DIR / "eyes.html").read_text(encoding="utf-8")
    return HTMLResponse(html)
```

WebSocket `/ws`、所有 API endpoints **完全復用，零改動**。

---

## 十一、Makefile 新增指令

```makefile
# 在 mamba3_mlx/Makefile 新增
eyes:
	@echo "Open http://$(HOST):$(PORT)/eyes"
	@echo "Press and hold SPACE to speak"

eyes-mock:
	$(PYTHON) -m mamba3_mlx.chat_demo --mock --port $(PORT)

eyes-full:
	$(PYTHON) -m mamba3_mlx.chat_demo --port $(PORT)
```

---

## 十二、完成標準

- [ ] 空白鍵按住 → 收音 → 放開 → 自動發送 → 眼睛反映狀態 → TTS 播放回應
- [ ] 六種眼睛狀態（idle/listening/processing/thinking/speaking/done）動畫流暢切換
- [ ] 桌面端 + 手機端（響應式）均可操作
- [ ] WebSocket 斷線自動重連，眼睛回到 idle
- [ ] 後端 chat_demo.py 改動不超過 5 行
- [ ] Mock 模式可用（`--mock` 不載入模型也能測完整流程）

---

*規劃日期：2026-06-03*
