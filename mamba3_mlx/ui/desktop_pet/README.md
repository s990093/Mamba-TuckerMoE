# Mamba 桌面寵物 (Desktop Pet)

把 `eyes` 吉祥物做成 macOS 上**去背、置頂、可互動**的桌面寵物——透明無邊框視窗，角色直接浮在桌面上。

底層只是一個 Swift 單檔（[DesktopPet.swift](DesktopPet.swift)）把現有的 `/eyes` 網頁包進透明 `WKWebView`；`?pet=1` 讓網頁套用去背的寵物樣式。

---

## 啟動

寵物要連到**真正的模型**（`make chat` 服務的 `:7860`，那裡才有聊天 `/ws`），所以先開 chat、再開 pet：

```bash
# 終端機 A — 先開好，等它載完模型
make chat

# 終端機 B — 自動連上 :7860、自動編譯、開寵物
make pet
```

`make pet` 會先檢查 `:7860`；**沒開就提示你先 `make chat`**（不會自己硬載模型）。

### 選項

```bash
make pet PET_EXTRA="--width 300 --height 360"   # 自訂初始大小
PORT=7860 make pet                              # 自訂 chat port

# 直接跑（繞過 make）
ui/desktop_pet/run.sh --width 360 --height 440
```

改了 `DesktopPet.swift` 下次 `make pet` 會自動重編。

---

## 操作方法

### 滑鼠

| 動作 | 結果 |
| --- | --- |
| **拖曳寵物**（任意處按住拖） | 移動位置；拖曳時會放大＋微傾＋落下陰影＝「被拿起」的手感 |
| **原地點一下** | 視為點擊，傳給網頁（不會誤判成拖曳；3px 門檻區分） |
| **移動桌面游標** | 眼睛即時盯著游標看（全域追蹤） |
| **窗內齒輪 / persona / 角色鈕** | 直接點（見下方控制列） |
| **底部文字框 → Enter** | 跟寵物聊天 |

### 窗內控制列（顯示在寵物身上）

| 鈕 | 功能 |
| --- | --- |
| ⚙ 齒輪 | 開設定面板（語音 rate / pitch、maxTokens、mouseTrack…） |
| persona 鈕（顯示目前 category，如 `daily`） | 點開選單切換 **system prompt** |
| 角色鈕 | 切換 `eyes` ↔ `tars` |
| 底部文字框 | 打字 Enter 送給模型，回覆會用 TTS 唸出來＋字幕泡泡 |

### 選單列 🐍（螢幕右上角，備援設定）

| 項目 | 功能 |
| --- | --- |
| **Track cursor** | 開關「眼睛追游標」 |
| **Switch character** | 切換角色（= 按 `X`） |
| **Switch persona (system prompt)** | 切換 system prompt（= 按 `C`） |
| **Bigger / Smaller** | 放大 / 縮小寵物（以中心縮放，220–840px） |
| **Click-through** | 整個視窗滑鼠穿透（暫時不擋桌面操作；要再互動就關掉） |
| **Reset settings** | 還原設定 |
| **Reload** | 重新載入 |
| **Quit Pet** | 結束（沒有 Dock 圖示，這是關閉方式） |

### 鍵盤（寵物視窗有焦點時，沿用 eyes 頁面）

| 鍵 | 功能 |
| --- | --- |
| `S` | 開 / 關設定面板 |
| `C` | 循環切換 system prompt（category） |
| `X` | 切換角色 |
| `?` | 快捷鍵說明 |
| `Esc` | 關閉面板 |

> 在底部文字框打字時，這些快捷鍵會自動停用，不會誤觸。

---

## 聊天與語音

- **講話（TTS 輸出）✅**：開箱即用。CoT 解析**只唸 `<final>` 最終答案**，不會把 `<think>` 思考唸出來；但**思考的推理文字面板（`#cot-stream`）會在思考時顯示**（最終答案開始時淡出），讓你看得到它在想什麼。
- **Space 說話**：先用滑鼠**點進寵物視窗**（取得焦點），之後按住 `Space` 才會觸發說話流程（沒焦點時 Space 不監聽，避免誤觸）。
- **聽你說（STT 輸入）⚠️**：eyes 原本「按住 Space 說話」用的 `webkitSpeechRecognition`，在 `WKWebView` 通常**不支援**。所以桌面寵物的穩定互動路徑是**底部文字框**。
- **麥克風權限**：Swift 已自動 grant 頁面麥克風請求，但 `swiftc` 直接編的裸執行檔沒有 `Info.plist` 的 `NSMicrophoneUsageDescription`，系統層 TCC 仍可能擋。真要做語音輸入，需打包成 `.app` + Info.plist，或接原生 `SFSpeechRecognizer` 餵字給 `sendPrompt`。

---

## 視窗特性（為什麼能「貼在桌面」）

需要**兩層都透明**才不會出現底色方塊：

1. **網頁層** — `?pet=1` 套用 `html,body{background:transparent}`（[eyes.css](../styles/eyes.css) 的 pet 區塊）
2. **視窗層** — `NSWindow.isOpaque=false` + `backgroundColor=.clear` + `WKWebView.drawsBackground=false`

再加 `.borderless`（無邊框）、`.floating`（置頂、蓋過全螢幕 app）、`.accessory`（無 Dock 圖示）。

---

## Debug（終端事件 log）

`make pet` 的終端會印出事件，方便除錯：

```
[pet 12:30:01] launch → http://127.0.0.1:7860/eyes?pet=1  size 360x440
[pet 12:30:02] page loaded ✓ http://127.0.0.1:7860/eyes?pet=1
[pet 12:30:05] drag start
[pet 12:30:06] drag end
[pet 12:30:09] switch persona (key c)
[pet 12:30:12] [page] log: [eyes WS] connected ...
[pet 12:30:15] [page] log: [eyes WS] token ...
```

- **`[page] …`** 是網頁的 `console.log/warn/error`（含 `[eyes WS]` 聊天/WebSocket 訊息）被橋接過來——打字送出後沒反應時，看這裡有沒有 WS 連線或 token 事件。
- **`page load FAILED …`** 代表連不到 chat server（確認 `make chat` 已 ready）。
- Swift 端事件：`drag start/end`、`switch persona/character`、`zoom`、`track cursor`、`click-through` 等。

## 疑難排解

| 症狀 | 處理 |
| --- | --- |
| 開 `make pet` 說沒有 chat server | 先在另一個終端機 `make chat`，等它載完模型再 `make pet` |
| 打字送出後不講話 | 確認 chat 已 ready；開 Web Inspector 看 console 有無 `/ws` 連線錯誤 |
| 看不到寵物 | 預設在主螢幕**右下角**；或被全螢幕 app 蓋住，切回桌面 |
| 想關掉寵物 | 螢幕右上選單列 🐍 → **Quit Pet** |
| 不小心整個穿透點不到 | 選單列 🐍 → 關掉 **Click-through** |

---

## 相關檔案

- [DesktopPet.swift](DesktopPet.swift) — 透明置頂視窗 + 拖曳 + 眼睛追游標 + 選單
- [run.sh](run.sh) — 檢查 chat server、編譯、啟動
- [../templates/eyes.html](../templates/eyes.html) — `?pet=1` 注入腳本（petLookAt / petDragging / 文字框）
- [../styles/eyes.css](../styles/eyes.css) — `body.pet` 去背樣式、眼睛追蹤、拖曳反饋、聊天列
- [../scripts/eyes.js](../scripts/eyes.js) — `window.sendPrompt` 已 export 給寵物使用
