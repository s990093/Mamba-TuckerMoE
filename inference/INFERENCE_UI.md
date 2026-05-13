# Mamba 推理畫面（Chat Demo）規格

本文件描述 **Edge 助理 Mamba** 在本倉庫 `inference/chat_demo` 上的推理／對話 UI、以及與訓練管線（[`cot_dataset/GUIDE.md`](../cot_dataset/GUIDE.md)、[`cot_dataset/SFT_FORMAT.md`](../cot_dataset/SFT_FORMAT.md)）的對齊規則。實作對應檔案：`inference/ui/`（靜態資源）、`inference/chat_demo.py`（FastAPI + WebSocket）、`cot_dataset/category_system_prompts.py`（與 SFT export 同源的分類 system 字串）。

---

## 1. 產品定位（對齊 GUIDE）

| 項目 | 規格來源 |
|------|----------|
| 助理名稱 | **Mamba**；demo 標題顯示 **Mamba · Edge** |
| 語氣 | 冷靜、精準、零冗餘；**不使用 emoji**、避免雞湯式動機句 |
| 架構敘述 | Hybrid **Mamba-TuckerMoE**、Edge on **iPhone / Apple Silicon**、**32,007** vocab、離線 |
| 回覆語言 | 訓練資料為英文 `input`／`output`；UI 文案可中英並存，模型輸出以英文為主 |

訓練時 `cot` 與 `output` 由預處理包進 `<think>…</think><final>…</final>`（GUIDE §3、`SFT_FORMAT.md`）。本 demo：
- **真實模型路徑**：在組 prompt 時對應地注入 `<think>\n`（reasoning 模式），並由後端用一個 streaming splitter 把模型輸出拆成 `reasoning`（送到 UI 的 CoT 摺疊框）與 `token`（送到 assistant 主訊息泡泡）兩條事件。
- **Mock 模式**：直接從 `mock_config.json` 讀 `cot_markdown` 與 `assistant_markdown`，以同樣的 WebSocket 事件呈現（不跑模型）。

---

## 2. 七大類別 × System Prompt（重要邏輯）

GUIDE 定義七大類別：Emotion、Self-Awareness、Email/Summary、Movie Intro、Daily Conversation、System Call、Deep Dive。

`cot_dataset/category_system_prompts.py` 是**單一真相來源**：

- `EXPORT_SYSTEM_PROMPTS`：七支短 system 字串（與 `export_hf_dataset.py` 訓練匯出共用），對應 SFT bucket key（`emotion` / `self_awareness` / `summarize_email` / `movie_intro` / `daily_conversation` / `system_call` / `deep_dive`）。
- `MOCK_CATEGORY_TO_EXPORT_KEY`：UI sidebar 的 `email_summary` → export 的 `summarize_email` 對應表。
- `merged_category_prompts_for_api(data)`：回傳每個 sidebar `key` → 字串，並允許 `mock_config.json` 用同名 `category_system_prompts` dict 覆寫個別分類。

`GET /api/demo-config` 會把這份 dict 一併送給前端（`category_system_prompts`）。

UI 行為：
- 點左欄範例 → `sysCatSelect` 自動切到該分類，`#sys-card-body` 立刻換成該分類的 SFT 短字串。
- 手動改下拉 → `change` 事件同步更新卡片。
- 找不到 key 或還沒拿到 → 退回 `system_prompt_markdown`（mock 的完整 persona）。

---

## 3. WebSocket 事件契約

| 方向 | type | payload | 說明 |
|------|------|---------|------|
| C→S | `chat` | `{ prompt, max_tokens, reasoning?, category_key?, sampling?, example_id? }` | `reasoning` 由 UI 切換鈕決定；`category_key` 用來查 SFT bucket prompt；`sampling` = `{temperature, top_k, top_p, min_p, repetition_penalty}`，由抽屜 **Sampling** tab 即時更新；`example_id` 只給 mock 對照 |
| C→S | `play_system_prompt` | `{ category_key }` | mock：串流文字；real：背景跑系統 prefill 並 cache，**同時**前端看到打字動畫 |
| C→S | `clear` / `ping` | — | 清會話 / heartbeat |
| S→C | `connected` | `{ ready, mock }` | 連線握手 |
| S→C | `intro_start` | `{ category_key, category_title }` | 開始送「分類 system 字串」打字動畫 |
| S→C | `meta` | `{ prefill_ms, prompt_tokens, turns, cached_prefix_tokens? }` | 首段 prefill 結束 |
| S→C | `reasoning` | `{ markdown }` | 累積式 `<think>` 內容（送 UI CoT 摺疊框） |
| S→C | `tool_action` | `{ call, system_result, phase, tool_name }` | 模擬 System Call（mock）|
| S→C | `assistant_split` | — | 工具列後切下一段 assistant 氣泡 |
| S→C | `token` | `{ text, n, tok_s }` | 串流 `<final>` 內容 |
| S→C | `done` | `{ total_tokens, total_ms, tok_s, ttft_ms, prefill_ms, play_system_only?, primed_ok?, cached_prefix_tokens? }` | 一輪完成 |

---

## 4. Real-mode 推理流程（`_stream_generate`）

對齊 SFT-CoT 訓練腳本的 ChatML：

```
<|im_start|>system
{category_system_prompt}
<|im_end|>
<|im_start|>user
{user_turn}
<|im_end|>
<|im_start|>assistant
<think>           ← 僅 reasoning=True 時注入
```

執行細節：

1. **System prompt 選取**：`category_key` → `_category_system_prompts_map(data)` → bucket 字串；缺失時 fallback `daily_conversation`。
2. **Continuation prefill（KV cache 重用）**：若 `_primed_prefix` 已 cache 該 system 字串，僅對 suffix（`user + assistant + <think>`）跑一次小 prefill，並以 `_deepcopy_caches` 複製 primed caches 後傳入 `_model(x, caches=…, seq_pos=primed.pos, …)`。
   - 太長（超過 `seq_len` 或 cache 上限）→ 自動退回完整 prefill。
   - 失敗（例外）→ log warning 後也退回完整 prefill。
3. **Stream splitter（`_CotStreamSplitter`）**：
   - `reasoning=True` 起始 mode = `think`。
   - 看到 `</think>` → 切到 `between`；看到 `<final>` → 切到 `final`；看到 `</final>` / `<|im_end|>` → `stop`。
   - 跨 chunk 標籤（如 `</thi` + `nk>`）以 safe-tail 暫存避免 leak；EOS 中斷時 `flush()` 會丟掉殘留的 partial tag。
4. **Special-id 還原 CoT 字面**：`_decode_chunk` 對解出 `<think>` / `</think>` / `<final>` / `</final>` 的 special token 直接回傳該字面字串，供 splitter 路由。
5. **停止條件**：splitter 收到 `stop`、或 token id 命中 `_stop_ids`，立刻 yield `done` 並結束 generator。

---

## 5. System-prefix KV cache（`play_system_prompt`）

| 項目 | 行為 |
|------|------|
| 觸發 | 點「Play system prompt」、或 WS 連線時對 `daily_conversation` 自動背景 prime |
| Mock | `_mock_stream_category_system_prompt` 純打字模擬 |
| Real | 後端非同步排程 `_prime_system_prefix_sync(sys_text)`，在 executor 跑 `<|im_start|>system\n{P}<|im_end|>\n` 的 prefill，並把 `(ids, caches, pos)` 存入 `_primed_prefix[sys_text]` |
| Cache key | system prompt 原字串（不含 `<think>` / user / history） |
| Cache 重用 | `_stream_generate` 偵測到命中時改走 continuation prefill；deep-copy 確保多次 chat 不會互汙染 primed 狀態 |
| 失敗保護 | 過長 / 例外 → 自動退回完整 prefill；UI 不知道、行為一致 |

優點：使用者邊看分類 SFT 字串打字、後端已在背景跑完 system 區塊 prefill；之後第一個 token 的 TTFT 通常下降為「只跑 suffix prefill + 第一次 decode」。

---

## 6. UI 行為（重點）

| 區塊 | 功能 |
|------|------|
| 左側欄 | 七大類 GUIDE 對齊的範例捷徑；點擊時自動同步 `sysCatSelect` 並更新 system card；窄螢幕用按鈕收合 |
| Header | WebSocket 狀態、清除對話、Metrics / **Sampling** / System prompt 抽屜入口 |
| Sampling tab | 即時拉桿：Temperature / Top-K / Top-P / Min-P / Rep. pen.；改值不影響當前正在串流的回合，下個 chat payload 才帶上；變更與 CLI 預設不同時拉桿格子轉為琥珀色，**Reset to defaults** 鈕一鍵回到 `/api/demo-config.sampling_defaults`（由 `_args.temp` 等 CLI 旗標決定） |
| **System prompt card**（聊天區頂端） | 顯示「目前分類的 SFT 短字串」；含分類下拉與 **Play system prompt** 按鈕；第一次送出訊息時自動收合，清除會話時自動展開 |
| 對話主體 | 自帶的 `renderGuideMarkdown` 純前端 Markdown（headings / tables / lists / fenced code / inline code、bold、italic、link），DOMPurify 預設 html profile；CoT 進可摺疊 `<details>` 框 |
| 動畫 | **進場 morph**：token 每次抵達都重 render 後與既有 DOM 做同 tag 同 index 的 in-place patch；新冒出的 block / table row / list item 加 `.fresh` 觸發 fade-slide-in；streaming 期間 caret 跟在最後一個 block |
| Reasoning toggle | 輸入框右下角的綠色 pill（預設 ON）。送 chat 時夾帶 `reasoning: bool`；OFF 時後端不注入 `<think>`，splitter 啟動於 `head` 模式仍能處理直答 |
| 輸入區 | 多行 textarea、max tokens slider、reasoning toggle、send 按鈕 |

CSS asset 通過 `/ui/chat_demo.{js,css}` 路由送出，headers `Cache-Control: no-store`；HTML 端再加 `?v=<mtime>` query 串作為 cache buster，避免改 JS 後瀏覽器拿到舊版。

---

## 7. HTTP API

| Endpoint | 說明 |
|----------|------|
| `GET /` | 主頁，會把 `<script>/static/chat_demo.js</script>` 重寫成 `?v=<mtime>` |
| `GET /ui/chat_demo.js` / `/ui/chat_demo.css` | 帶 no-store 的 UI assets endpoint |
| `GET /api/status` | 模型載入狀態 / 架構（給抽屜 Metrics 分頁；mock 時讀 `mock_config.json.status`） |
| `GET /api/demo-config` | `system_prompt_markdown`、`category_system_prompts`、`style_constraints`、`tool_registry`、`categories`、`examples`、`mock` 旗標 |

---

## 8. CLI 與啟動

| 項目 | 預設 |
|------|------|
| 入口 | `python inference/chat_demo.py` 或 `./inference/run_chat_demo.sh` |
| Mock | `python inference/chat_demo.py --mock` 或 `MOCK=1 ./inference/run_chat_demo.sh` |
| Reasoning | `--reasoning`（預設）/ `--no-reasoning` |
| Checkpoint | `--checkpoint` 或 `CHECKPOINT=...`；`run_chat_demo.sh` 提供 `bset`（含 throughput preset）／預設 stable 兩組 |
| 量化 / dtype | `--quantize {0,4,8}`、`--dtype {fp32,bf16,fp16}`、`--kv-dtype` |
| 上下文 | `--seq-len`、`--max-new-tokens`、`--warmup` |

---

## 9. 檔案對照

| 路徑 | 用途 |
|------|------|
| `inference/chat_demo.py` | FastAPI app、`/`、`/ui/*`、`/api/*`、`/ws`、real & mock 雙路徑、prefix prime |
| `inference/ui/chat_demo.html` | UI 骨架（含 system card、抽屜、reasoning toggle） |
| `inference/ui/chat_demo.css` | 樣式（intro bubble、reasoning toggle、fresh block 動畫、CoT 摺疊框、tool card 等） |
| `inference/ui/chat_demo.js` | WS handler、`renderGuideMarkdown`、`_CotStreamSplitter` 的對應 client morph、reasoning toggle 狀態 |
| `inference/ui/mock_config.json` | mock `status`、`system_prompt_markdown`（完整 persona）、`categories[]`（七大類，含 `cot_markdown` / `assistant_markdown` / 可選 `tool_flow`）、`mock_stream`（target tok/s、jitter、pause 等）、可加 `category_system_prompts` 覆寫 |
| `cot_dataset/category_system_prompts.py` | 七大 SFT bucket prompt（與 export 共用） |
| `cot_dataset/export_hf_dataset.py` | 從上述模組 `import EXPORT_SYSTEM_PROMPTS as SYSTEM_PROMPTS`，確保訓練／UI 同源 |

---

## 10. 後端串流摘要

```
[Client]                                  [Server]
  WS open ───────────────────────────────▶ accept + connected
                                          │ (real) bg prime 'daily_conversation'
  play_system_prompt ───────────────────▶ intro_start
                                          │ (executor) prefill <system block>
                                          │  └→ _primed_prefix[sys_text]
  ◀──────────────── intro tokens (typing animation) ──┘
  ◀──────────────── done {primed_ok, prefill_ms, cached_prefix_tokens}

  chat {prompt, reasoning, category_key} ▶
                                          │ system_prompt = _category_system_prompts_map()[ck]
                                          │ if _primed_prefix[sys_text]:
                                          │     deepcopy(caches); continuation prefill on suffix
                                          │ else:
                                          │     full prefill
  ◀──────────────── meta {prefill_ms, prompt_tokens, cached_prefix_tokens?, turns}
                                          │ splitter(reasoning=…).feed(decoded_chunk)
  ◀──────────────── reasoning {markdown}  (累積，UI CoT 框)
  ◀──────────────── token {text, n, tok_s}(<final> 內文，UI assistant body)
  ◀──────────────── done {total_tokens, total_ms, tok_s, ttft_ms, prefill_ms}
```

---

## 11. 已知限制 / 後續

- Mock 仍然 `import mlx.core`：要拆出純 HTTP mock 服務時可獨立成 `chat_demo_mock.py`。
- Continuation prefill 目前只 cache 「裸 system 區塊」；不快取 reasoning toggle / 歷史。需要更高層的 prefix cache（包含 turn-1 user prefix）時可擴充。
- Sampling 覆寫經 `_apply_sampling_override(_args, msg["sampling"])` 產生 per-call `args`，僅替換 `temp/top_k/top_p/min_p/rep_pen` 並 clamp 範圍；全域 `_args` 與 prefix 快取均不受影響，已串流中的 turn 也不會被改變。
- `play_system_prompt` 在 real 模式下每次都會把 system 字串重新打字一次（給使用者視覺反饋）；如果想要靜默 prime，可以從 UI 端送 `silent: true` 並在後端略過 `intro_start` / token 流。
