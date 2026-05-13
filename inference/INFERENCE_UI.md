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
| C→S | `chat` | `{ prompt, max_tokens, reasoning?, category_key?, sampling?, format_guard?, example_id? }` | `reasoning` 由 UI 切換鈕決定；`category_key` 用來查 SFT bucket prompt；`sampling` = `{temperature, top_k, top_p, min_p, repetition_penalty}`，由抽屜 **Sampling** tab 即時更新；`format_guard=false`／`format_guard={"enabled":false, "force_final_inject":false}` 會逐欄關掉該回合的 middleware（見 §4.6）；`example_id` 只給 mock 對照 |
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
3. **Inference middleware（`CotMiddleware`）**：所有 inference-time 保證集中在 `inference/cot_middleware.py`，每回合建立一個實例；`_stream_generate` 只在四個窄接口呼叫它。
4. **停止條件**：middleware `step()` 回傳 `__stop__` 哨兵、或 `should_break(tid)` 命中合併後的 `_stop_ids`，立刻 yield `done` 並結束 generator。
5. **Multi-stage `<final>` 注入**：當 splitter 從 `think` 變成 `between` 時，middleware 編碼 `<final>\n` 並用 model_apply 跑一次小型 continuation prefill 讓 KV cache 推進，下一個 sample 就在「已經吐完 `<final>\n`」的條件下取樣；同一回合只跑一次。
6. **Format guard（logit 攔截層）**：見下一節。

### 4.6 Inference middleware（`inference/cot_middleware.py`）

對齊「Prefix Injection → Logits ban → Dynamic Logits Processor → Multi-Stage Prompt Injection → Stop Sequences → Hard budget」六層策略，全部包在 `CotMiddleware` 一個物件裡。

#### 4.6.1 完整資料流（per-turn ASCII map）

```
                ┌───────────────────────────────────────────────────────────────────┐
                │                       PROCESS-LEVEL (singleton)                    │
                │                                                                    │
                │  _load_model()                                                     │
                │     │                                                              │
                │     ├── build tokenizer / model / KV padding                       │
                │     │                                                              │
                │     └── CotMiddlewareDeps.build(tokenizer, cfg=_mw_cfg)             │
                │             ├── FormatGuard (ban mask + per-mode one-hot)          │
                │             ├── final_inject_ids = encode("<final>\n")             │
                │             └── stop_ids = _stop_ids ∪ {</final> single-id?}       │
                └───────────────────────────────────────────────────────────────────┘
                                              │  (deps shared, never mutated)
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                  PER-TURN LIFECYCLE                              │
│                                                                                  │
│  WS chat ──▶ _stream_generate()                                                  │
│              │                                                                   │
│              ├─ prompt prefill / continuation prefill (KV cache built)           │
│              │                                                                   │
│              ├─ CotMiddleware( deps, cfg, reasoning, model_apply ) ◀── per turn  │
│              │     │   reasoning=True  → splitter.start_in_think                 │
│              │     │   reasoning=False → splitter.head (loose final)             │
│              │     │                                                             │
│              │     └─ owns:  splitter, _reasoning_acc, _think_tokens,            │
│              │               _budget_hit, _final_injected                        │
│              │                                                                   │
│              ▼                                                                   │
│  ┌────────────────────────  PER-STEP DECODE  ────────────────────────────────┐   │
│  │                                                                          │   │
│  │  ① row = model.logits[0,-1,:]                                            │   │
│  │       │                                                                  │   │
│  │       ▼                                                                  │   │
│  │  ② mw.transform_logits(row)                                              │   │
│  │       │   • ban_mask:    row[ban_ids]    += -1e9                         │   │
│  │       │   • close_bias:  row[close_id]   += current_close_bias()         │   │
│  │       │                  (per-mode + ramped for "think" mode)            │   │
│  │       ▼                                                                  │   │
│  │  ③ tid = sample_decode_token(biased_row, token_counts, sample_args)      │   │
│  │       │                                                                  │   │
│  │       ▼                                                                  │   │
│  │  ④ for ev in mw.step(tid):                                               │   │
│  │       │   • _decode_chunk(tid)        → str                              │   │
│  │       │   • splitter.feed(str)        → [(kind, text), …]                │   │
│  │       │   • emit reasoning / token / __stop__                            │   │
│  │       │   • if mode=="think": _think_tokens += 1                         │   │
│  │       │       └─ budget exceeded? → _fire_reasoning_budget()             │   │
│  │       │                              ├─ synth notice → final            │   │
│  │       │                              ├─ splitter.force_done()           │   │
│  │       │                              └─ yield __stop__                  │   │
│  │       ▼                                                                  │   │
│  │  ⑤ if prev_mode=="think" and mw.mode=="between":                         │   │
│  │       │   caches,pos,inj_row,did,ms = mw.maybe_inject_final(caches,pos)  │   │
│  │       │     • encodes "<final>\n" once (cached at startup)               │   │
│  │       │     • model_apply on (1, N) → caches advance N positions         │   │
│  │       │     • splitter.feed("<final>\n") → mode auto-becomes "final"     │   │
│  │       │     • inj_row replaces next step's decode_fn output              │   │
│  │       ▼                                                                  │   │
│  │  ⑥ if stop_after or mw.should_break(tid):  break                         │   │
│  │       │                                                                  │   │
│  │       │   (next iter)                                                    │   │
│  │       ▼                                                                  │   │
│  │  ⑦ if inj_row is not None:                                               │   │
│  │           row = inj_row;  inj_row = None;  pos unchanged                 │   │
│  │       else:                                                              │   │
│  │           row, caches = decode_fn(x_one, caches, seq_pos=pos);  pos++   │   │
│  │       (loop back to ②)                                                   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  end of loop:                                                                    │
│      mw.flush()       → emit any safe text remaining in splitter buffer          │
│      yield "done"     → total_tokens, elapsed, ttft_ms, prefill_ms               │
│      _print_turn_summary(..., mw_health=mw.health_report())                      │
│      _free_metal_cache()                                                         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

#### 4.6.2 Splitter state machine（`CotStreamSplitter`）

```
            reasoning=True                       reasoning=False
                  │                                   │
                  ▼                                   ▼
              ┌────────┐                          ┌────────┐
              │ think  │                          │  head  │   ← loose_final=True
              └────┬───┘                          └────┬───┘   (tagless answer ok)
                   │                                   │
       sees </think>                          sees <think>   sees <final>
                   │                                   │             │
                   ▼                                   ▼             ▼
              ┌──────────┐  (multi-stage         ┌────────┐     ┌────────┐
              │ between  │   inject fires here)──▶│ think  │     │ final  │
              └─────┬────┘                       └────┬───┘     └────┬───┘
                    │                                 │              │
       sees <final>/<final>\n inject                  ▼              │
                    │                            ┌──────────┐        │
                    ▼                            │ between  │ ◀──────┘
              ┌────────┐    sees </final> or          │
              │ final  │ ─▶ <|im_end|>           sees </final>/<|im_end|>
              └────┬───┘                              │
                   │                                  ▼
                   │                              ┌──────┐
                   ▼                              │ done │
              ┌──────┐                            └──────┘
              │ done │
              └──────┘

stop emission (kind="stop"):  any mode → done on </final> or <|im_end|>
bridged text preserved:       between mode emits any text before next tag as ("final", …)
cross-chunk tag safety:       _safe_tail_cut() keeps up to MAX_TAG_LEN-1 chars in _buf
budget override:              force_done() drops _buf + sets mode=done
```

#### 4.6.3 Dynamic close-bias ramp（mode="think"）

```
  close-bias on </think>'s first id
       ▲
       │
  c_max│                                            ╭───────────────────────
       │                                          ╱ │
       │  (ramp slope =                         ╱   │   (clipped at c_max)
       │   (c_max − c_value) /                ╱     │
       │   (budget − start))               ╱        │
  c_val│                                ╭╯          │
       │                              ╱             │
       │                          ╱                 │
     0 │ ─────────────────────╱                     │ ─────────────────────▶
       └──────────────────────┴─────────────────────┴──────────────  think_tokens
       0                  close_bias_start    reasoning_budget
                          (auto = budget/2)   (hard stop / watchdog)

  • for_mode=="between" / "final":  constant +c_value (no ramp)
  • for_mode=="head"  / "done":     0
```

#### 4.6.4 Multi-stage `<final>` injection（sequence）

```
                                       caches (KV) pos=N
                                            ▼
   model emits </think> ─── splitter mode: think → between
                                            │
                                            ├─ guard checks (force_final_inject?
                                            │                already injected?
                                            │                final_inject_ids resolved?)
                                            │
                                            ▼
                            mw.maybe_inject_final(caches, pos=N)
                                            │
                                            ▼
                            x = [[<final_id>, \n_id]]  (1 × M)
                                            │
                                            ▼
                            _model_apply(x, caches, seq_pos=N)
                                            │
                                            ├─ caches now reflect N+M positions
                                            └─ logits[0, -1, :] = inj_row
                                            │
                                            ▼
                            splitter.feed("<final>\n")
                                            │
                                            └─ mode: between → final, buf="\n" cleared
                                            │
                                            ▼
                            return (caches', N+M, inj_row, did=True, ms)
                                            │
                                            ▼
                            next sample uses inj_row directly →
                            decode_fn step skipped this iter, pos already N+M
                            (mw._final_injected = True; never fires again this turn)
```

#### 4.6.5 各層責任表

| 層 | 行為 | 來源 |
|----|------|------|
| Prefix injection | `_build_multiturn_ids` / continuation prefill 結尾固定為 `<\|im_start\|>assistant\n<think>\n` | `chat_demo.py` |
| Ban | 把 `<\|im_start\|>` 的 logit 設為 `-1e9`，杜絕 role-flip | `FormatGuard.apply` |
| **Dynamic close-bias** | `mode=="think"` 時，把 `</think>` 首 id 的 logit 由 `+close_bias`（在 `close_bias_start`）線性拉到 `+close_bias_max`（在 `reasoning_budget`）；`between`/`final` 則套用靜態 `+close_bias` | `CotMiddleware.current_close_bias` |
| **Multi-stage `<final>` 注入** | splitter 一進入 `between` 就用 `model_apply` 把 `<final>\n` 推進 KV cache，並把 splitter 模式手動切到 `final`；同回合僅一次 | `CotMiddleware.maybe_inject_final` |
| Stop extension | 把 `</final>` 等單 id 收尾加入 `_stop_ids`；多 token 收尾仍由 splitter `stop` 兜底 | `merge_format_guard_stop_ids` |
| **Hard reasoning budget** | think 模式累計 token 達 `--reasoning-budget` 時 watchdog 合成 reasoning + 警告 `<final>`，把 splitter 推到 `done` | `CotMiddleware._fire_reasoning_budget` |

預設行為：

- `--format-guard` 預設 **on**；`--ban-im-start` 預設 **on**；`--force-final-inject` 預設 **on**。
- `--close-bias` 預設 `4.0`、`--close-bias-max` 預設 `16.0`、`--close-bias-start` 預設 `0`（middleware 自動推導為 `reasoning_budget // 2`，讓 ramp 在 budget 後半段啟動）。
- WS `chat` payload 可帶 `format_guard: false` 整層關掉、或 `format_guard: {"force_final_inject": false}` 只停掉注入路徑；其餘維持全域預設。
- 每回合結束會在後端 console 印出 `"middleware":` 一行 `health_report`：`mode=…  think=…/budget  budget_hit  final_injected  close_bias=(v→max from t_start to t_budget)`。

效能考量：

- `FormatGuard` 在 startup 把 ban mask 與 per-mode one-hot 向量一次 build；`apply` = 1 個常數向量加 + 1 個 scalar×one-hot 加。
- Dynamic ramp 只算純量；不重建詞表向量。
- `maybe_inject_final` 一回合最多跑一次小型 prefill（`len("<final>\n")` tokens），開銷可忽略。

完整 Regex / FSM constrained decoding：保留 `cot_format_fsm.py` 底部 TODO 註記。若未來模型再退化到無法以單 token 起頭關閉標籤，再升級為 BPE prefix trie；當前 ban + ramped bias + 注入 + watchdog 對固定 ChatML 骨架已足夠穩定。

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
| Middleware | `--format-guard` / `--no-format-guard`、`--ban-im-start` / `--no-ban-im-start`、`--close-bias`（預設 4.0）、`--close-bias-max`（預設 16.0）、`--close-bias-start`（0 = 由 `reasoning_budget // 2` 自動推導）、`--force-final-inject` / `--no-force-final-inject` |
| 推理硬上限 | `--reasoning-budget`（think 模式 token 上限，超過強制收尾）|

`run_chat_demo.sh` 對應環境變數：`FORMAT_GUARD=on/off`、`FORCE_FINAL_INJECT=on/off`、`CLOSE_BIAS=4.0`、`CLOSE_BIAS_MAX=16.0`、`CLOSE_BIAS_START=0`、`REASONING_BUDGET=...`。

---

## 9. 檔案對照

| 路徑 | 用途 |
|------|------|
| `inference/chat_demo.py` | FastAPI app、`/`、`/ui/*`、`/api/*`、`/ws`、real & mock 雙路徑、prefix prime |
| `inference/ui/chat_demo.html` | UI 骨架（含 system card、抽屜、reasoning toggle、format guard toggle） |
| `inference/ui/chat_demo.css` | 樣式（intro bubble、reasoning toggle、format guard 開關、fresh block 動畫、CoT 摺疊框、tool card 等） |
| `inference/ui/chat_demo.js` | WS handler、`renderGuideMarkdown`、`_CotStreamSplitter` 的對應 client morph、reasoning toggle 狀態、format guard payload |
| `inference/cot_format_fsm.py` | 推論期低階原語：`CotStreamSplitter`（CoT/ChatML FSM）、`FormatGuard`（logit ban + close-bias 套用） |
| `inference/cot_middleware.py` | 高階 inference middleware：`CotMiddleware` 將 splitter / guard / reasoning budget / dynamic ramp / multi-stage `<final>` 注入串成單一接口 |
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
- `cot_middleware.py` 已實作「在 `</think>` 後 inplace 注入 `<final>\n` 走 continuation prefill」這條多階推論路徑，但仍把它包成 *one-shot per turn* 並讓 splitter 自動跟進——對使用者透明、不需要 WS 狀態機改動。若未來想拆成更完整的兩段式 decode（例如 `final` 段獨立 token 上限或獨立溫度），可在 `CotMiddleware.maybe_inject_final` 之後分出第二個 sampling profile。
- `cot_format_fsm.py` 還沒有完整 BPE token 前綴 trie；目前用單 token close-bias 已能穩定關標籤，但若 SFT 改版讓 `</think>` 無法用單 id 起頭，依檔尾 TODO 升級。
