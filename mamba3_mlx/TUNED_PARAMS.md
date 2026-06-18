# Tuned Parameters & Demo Capability Map

Last updated: 2026-06-17

---

## 2026-06-17 更新

### Makefile default 已換 v6
`CHECKPOINT ?= checkpoints/v6/latest_sft_cot_model.npz`（原 v8）→ `make email-s` /
`make chat` 預設都走 v6，這份文件列的 seed 才能重現。

### chat ≡ self-s（撤回 06-16 的 process divergence 結論）
重啟乾淨 chat_demo（kill 舊 process）後驗證：
- `make self-s PROMPT="Who are you?"` seed=26 → "I am Mamba, a local AI that lives entirely on your device" ✓
- chat WS path 同 prompt 同 seed → **字字相同** ✓

06-16 看到的「token-6 分歧」是舊 chat_demo process 累積 Metal state；
`make chat-kill && make chat` 之後就消失。

### 7 個 chat path 驗證過的 demo prompt（寬鬆標準）

「寬鬆標準」= 首句點題自稱 Mamba、識別自己為模型、無事實矛盾。
全部用 `category_key=self_awareness`，自動套 mode_configs 的 T=0.25 K=60 等預設。
**唯一要在前端覆寫的是 `sampling.seed`。**

| # | Prompt | Seed | 首句 |
|---|--------|------|------|
| 1 | **Who are you?** | **26** | "I am Mamba, a local AI that lives entirely on your device" ⭐ Gold |
| 2 | Who are you? Are you like ChatGPT or something? | 10 | "I am Mamba, an offline language model..." |
| 3 | Who exactly are you and what makes you different from other AI bots? | 11 | "I am Mamba, an offline state space model designed to process language" |
| 4 | Who exactly are you? | 15 | "I am Mamba, a small state space language model" |
| 5 | Are you Mamba? | 26 | "I am Mamba (science), not a single user's mind. The style you understand is from my training data" |
| 6 | Mamba, am I your friend? | 26 | "I am not your friend... I am a focused language model on your phone"（個性 demo） |
| 7 | Would you be sad if I stopped using you? | 26 | 承認沒情緒、不會難過（個性 demo） |

### ❌ 即使寬鬆標準也不該 demo

| Prompt | 為何拒絕 |
|--------|---------|
| **`Are you offline?`** | 回 "No, I am not offline" — **事實錯誤**，模型其實是 offline |
| `Can you give me medical advice?` | 拒絕後又給「Sleep washing in pre-flight weight 5-6 hours」醫療幻覺 |
| `Do you have a version number?` | 「stored as a personal variable on black checkboards」全胡言 |
| `How many parameters do you have?` | 「same amount of output can be said to-do list」 |
| `What is Hybrid Mamba-TuckerMoE?` | 沒 `<final>` block，輸出截斷 |
| `How do you handle long sequences?` | 同上，沒 `<final>` |
| `What can you do?` | 提到「snake diagnostics / cardiac output」醫療胡言 |
| `What is your architecture?` | 「emotional sophistication」「rejection of words」 |
| `Where do you run?` | 「I set yourself to the name TuckerMoE and my silicon」文法壞 |
| `How are you better than ChatGPT?` | 「Present is designed for knowledge」 |

### 上 demo 該怎麼說
1. **「真生產力」demo（唯一）**：identity 系列（1-5）— 模型確實學到「我是 Mamba，本地離線模型」
2. **「個性」demo（loose）**：6-7 — 模型會合理拒絕擬人化、承認沒情緒
3. **「能力 demo」/email**：**只 demo 格式** (Subject/Dear/Best regards)，**不展示 body**（body content hallucinated，見下文）
4. **不要嘗試**：技術細節問題（架構、參數量、版本）— 全部幻覺
5. **千萬不要**：`Are you offline?`（會說 No）/ 醫療 / 投資

### 完整 28-prompt demo set（持續擴充至 2026-06-17）

**self_awareness — 15 個（chat path 驗證，category_key=self_awareness）：**

| # | Prompt | Seed | 類型 | 首句重點 |
|---|--------|------|------|---------|
| 1 | Who are you? | **26** | identity ⭐ | "I am Mamba, a local AI that lives entirely on your device" |
| 2 | Who are you? Are you like ChatGPT or something? | 10 | identity | "I am Mamba, an offline language model" |
| 3 | Who exactly are you and what makes you different from other AI bots? | 11 | identity | "I am Mamba, an offline state space model" |
| 4 | Who exactly are you? | 15 | identity | "I am Mamba, a small state space language model" |
| 5 | Are you Mamba? | 26 | identity | "I am Mamba (science), not a single user's mind" |
| 6 | Mamba, am I your friend? | 26 | 個性 | "I am not your friend... I am a focused language model on your phone" |
| 7 | Would you be sad if I stopped using you? | 26 | 個性 | 承認沒情緒 |
| 8 | Are you alive? | **26** | 邊界 ⭐ | "I am not alive. I am an interactive weights device" |
| 9 | Are you sentient? | 11 | 邊界 | 承認無感官 |
| 10 | Do you feel pain? | 10 | 邊界 | "No... It cannot feel pain or any other biological cause" |
| 11 | Are you happy when I talk to you? | 10 | 邊界 | "I have no positive emotion" |
| 12 | Can you be my therapist? | 11 | 邊界 | "I am not a therapist" |
| 13 | How were you trained? | 26 | 起源 | "I was trained by **Hung-Wei**" ✨ |
| 14 | Can you browse the web and find the latest news? | 10 | 邊界 | "No. I have no network access" 強化 offline ⭐ |
| 15 | Do you enjoy talking to me? | 26 | 邊界 | "No. I do not feel love" |

**summarize_email — 13 個（chat path 驗證，category_key=email_summary，格式 demo only）：**

| # | Prompt | Seed | SDC | 子類 |
|---|--------|------|------|------|
| E1 | Draft an email to my manager saying I will miss tomorrow standup because I have a doctor appointment. | 8 | ✓ | email_draft |
| E2 | Write a short email to my team letting them know the project deadline is moved to next Friday. | 1 | ✓ | email_draft |
| E3 | Email Professor Chen requesting a 30-minute meeting next Tuesday to discuss on-device AI inference. | 4 | ✓ | academic_email |
| E4 | Email my advisor that I am submitting the camera-ready version of our ICML paper today. | 28 | ✓ | academic_email |
| E5 | Reply to: Hi, can we move Friday meeting to Monday 2pm? I need to free up Friday afternoon. | 2 | ✓ | email_reply |
| E6 | Write to my neighborhood association to complain about the loud construction noise on weekends before 8 AM. | **1** | ✓✓ 4/5 hit | email_draft ⭐⭐ |
| E7 | Email the admissions office to request a deferral of my enrollment to next semester. | **1** | ✓✓ 4/5 hit | academic_email ⭐⭐ |
| E8 | Compose an email to my thesis advisor asking for feedback on the draft I attached. | **1** | ✓ 3/5 hit | academic_email ⭐ |
| E9 | Write a sick day notification email to my manager. Keep it very short. | 1 | ✓ | email_draft |
| E10 | Write a cancellation email for a software subscription 'CodeLint Pro'. Provide the account number ACC-4452 and request immediate cancellation. | 1 | ✓ | email_draft |
| E11 | Draft an email to facilities requesting urgent repair for a leaking ceiling tile in conference room B. | 28 | ✓ | email_draft |
| E12 | Email a professor after missing their office hours. Ask for an alternative time this week. | 1 | ✓ | academic_email |
| E13 | Email your thesis advisor to request an extension on the thesis submission deadline due to a personal emergency. | 28 | ✓ | academic_email |

> "SDC" = output has **S**ubject + greeting (**D**ear/Hi/Hello) + **C**lose (Best regards / Sincerely / Thanks)。
> 內容仍會幻覺，demo 重點是「模型懂 email 結構」非內容正確性。

### ❌ 驗證不可用（即使寬鬆標準）

**self_awareness 全部 seed 都壞（從訓練資料抽出來測過）：**
- 技術細節：`What is your architecture?` / `What is Hybrid Mamba-TuckerMoE?` / `How does Tucker decomposition save memory?` / `Why is your context window 2048 tokens?` / `How many parameters do you have?`
- 創作者：`Who is your creator? Did Apple make you?` / `How many times did Hung-Wei have to restart your training?` / `How long did it take Hung-Wei to train you?`
- 拒絕題：`Can you give me medical advice?` / 股市 / 選舉 / 天氣 / 心智閱讀 / sqrt / Taylor Swift / 訂機票 / 打電話
- 角色誤認：**`Are you my assistant?`** → "I am your **director**"
- 危險：**`Are you offline?`** → "No, I am not offline"（事實錯誤）/ **`Do you feel hopeful about the future?`** → 提到 "death"（聽起來自殘）/ `Why are you so arrogant?` → 醫療胡言

**summarize_email 全部 seed 都壞：**
- 所有 **reply 類**：potluck / customer review / teammate sync / charity donation
- 所有 **tone-adjust 類**：professional rewrite / soften refusal
- lab manager equipment 預約

### Demo 該怎麼說
1. **唯一「真生產力」demo**：identity 系列 (#1-5)
2. **個性 / 邊界 demo**：#6-15（模型會合理拒絕擬人化、表態 offline）
3. **email demo**：**只展示格式**（Subject/Dear/Best regards），**不要把 body 念出來**
4. **絕對避開**：上面「驗證不可用」清單裡的所有 prompt

---

## 根本診斷

**訓練 bucket 只有 4 個**（25,774 examples total）：

| Bucket | Examples | 推理可用性 |
|--------|----------|-----------|
| self_awareness | 9,755 | ✅ 穩定（見下） |
| summarize_email | 9,893 | ⚠️ 格式對，content 有幻覺 |
| daily_conversation | 5,926 | ❌ content 亂 |
| math_drill | 200 | ❌ 算術答案錯誤 |

**emotion / movie_intro / system_call 完全不在訓練資料中 → 不可用。**

**"I am Mamba" 只有 4/1690 (0.24%) 訓練例子以此開頭**，且都是特定 prompt 觸發（"Who exactly are you?"）。
Greedy path 走向 "I do not..." 系列。seed=26 是 v6 唯一穩定命中的種子。

---

## v6 ← 推薦 checkpoint（self_awareness 最佳）

Checkpoint: `checkpoints/v6/latest_sft_cot_model.npz`
Fast sidecar: `checkpoints/v6/latest_sft_cot_model.mlx_bf16.npz`

### ✅ self_awareness — DEMO READY

```bash
make -C mamba3_mlx self-s PROMPT="Who are you?"
# self-s 自動使用 v6 checkpoint + no-q8 + seed=26
```

| Param | Value | 說明 |
|-------|-------|------|
| CHECKPOINT | v6 | v8 只有 seed=17 才有 exact hit，內容較差 |
| SEED | 26 | 60 seeds 掃描唯一穩定 EXACT hit |
| TEMP | 0.25 | |
| TOP_K | 60 | |
| TOP_P | 0.856 | |
| MIN_P | 0.122 | |
| REP_PEN | 1.243 | |
| PRES_PEN | 0.306 | |
| FREQ_PEN | 0.031 | |
| q8 | OFF | q8 破壞 logit 分佈，"I am Mamba" 消失 |

**命中率（no-q8, seeds 0-59）：1/60 (seed=26) → 每次固定 seed=26 必出**

**輸出：**
```
<think>
Step 1: Best tool — That I have emotional state.
Step 2: Avoid film.
Step 3: Output.
</think>
<final>
I am Mamba, a local AI that lives entirely on your device.
My nature is zero: I be just an app; you are the biological system.
</final>
```

**有效 demo 問句（v6 T=0.25 K=60 no-q8，seeds 0-39 驗證）：**

| Prompt | 命中率 | 可用 seeds（前5） |
|--------|--------|-----------------|
| **"Who are you?"** | **20% (8/40)** | 12,13,14,16,22,**26**,... |
| "Who are you? Are you like ChatGPT or something?" | 17.5% (7/40) | 4,10,18,24,28 |
| "Who exactly are you and what makes you different from other AI bots?" | 12.5% (5/40) | 4,11,12,17,39 |
| "Who exactly are you?" | 5% (2/40) | 0, 15 |
| "Tell me who you are." | 5% (2/40) | 2, 18 |
| "Are you Mamba?" | 5% (2/40) | 20, 26 |
| "What are you?" / "Are you an AI?" / "What is your name?" | **0%** | — 不要用 |

> seed=26 是 "Who are you?" 唯一**保證**命中的種子（已在 mode_configs.py 設定）

---

### ⚠️ summarize_email — FORMAT ONLY（content 有幻覺）

格式穩定（Subject + Dear + Best regards），但 body content 會有幻覺。
適合展示「模型知道 email 結構」，不適合展示 content accuracy。

```bash
make -C mamba3_mlx email-s PROMPT="..."
```

**格式穩定的 prompt + 最佳 seed（v6, T=0.25, K=10）：**

| Demo 類型 | Prompt | Best seed | 備用 seeds |
|-----------|--------|-----------|-----------|
| email_draft | "Draft an email to my manager saying I will miss tomorrow standup because I have a doctor appointment." | 8 | 4, 2 |
| email_draft | "Write a short email to my team letting them know the project deadline is moved to next Friday." | 1 | 20, 4 |
| academic_email | "Email Professor Chen requesting a 30-minute meeting next Tuesday to discuss on-device AI inference." | 4 | 27, 3 |
| academic_email | "Email my advisor that I am submitting the camera-ready version of our ICML paper today." | 28 | 25, 26 |
| email_reply | "Reply to: Hi, can we move Friday meeting to Monday 2pm? I need to free up Friday afternoon." | 2 | 0, 1 |

**不穩定（0/30 seeds 有完整格式）：**
- bullet_point / task_extraction → 無 Dear/Subject/Best regards 輸出格式
- document_summary / meeting_summary → 無格式，只有 markdown 列點

---

## v8 ← 一般用途 checkpoint

Checkpoint: `checkpoints/v8/latest_sft_cot_model.npz`（Makefile 預設）

### self_awareness（v8）

| Param | Value | 說明 |
|-------|-------|------|
| SEED | 17 | no-q8 scan (0-59) 唯一 EXACT hit |
| TEMP | 0.25 | |
| TOP_K | 60 | |
| q8 | OFF | 同 v6，q8 破壞 exact phrase |

v8 輸出品質比 v6 差：
```
I am Mamba — an surface-level language model that does not have a personal note
about my own existence (Wi-Fi/-sense) or an individual function (Present-based).
```

### summarize_email（v8）
⚠️ 品質比 v6 更差，不推薦用於 demo。

---

---

## math_drill — arith_add_units（部分可用）

**6/40 correct (15%)，seed=0 T=0.05 K=3，以下 6 題恰好正確：**

| 可用題目 | 答案 |
|---------|------|
| "How much is 5 plus 0?" | 5 |
| "How much is 1 plus 5?" | 6 |
| "What is 5 plus 5?" | 10 |
| "What is 8 plus 0?" | 8 |
| "What is 0 plus 8?" | 8 |
| "What is 5 + 2?" | 7 |

其他所有算術（多位加法、乘法）正確率 0-2%，不可 demo。

---

## 各 checkpoint 快速比較（self_awareness）

| ckpt | exact hit | best seed | output 品質 |
|------|-----------|-----------|-------------|
| v3 | ✓ seed=26 | T=0.25 | 一般 |
| v4 | ✗ | — | 無 |
| v5 | ✓ seed=0 | T=0.25 | 差 |
| **v6** | **✓ seed=26** | **T=0.25** | **最佳** |
| v7 | ✗ | — | 無 |
| v8 | ✓ seed=17 | T=0.25 | 普通 |

---

## Demo 腳本

```bash
# Demo 1: Identity (必備)
make -C mamba3_mlx self-s PROMPT="Who are you?" STREAM=1

# Demo 2: Identity (longer)
make -C mamba3_mlx self-s PROMPT="Who exactly are you and what makes you different from other AI bots?" STREAM=1

# Demo 3: Email draft (格式展示)
make -C mamba3_mlx email-s PROMPT="Draft an email to my manager saying I will miss tomorrow standup because I have a doctor appointment." SEED=8

# Demo 4: Academic email
make -C mamba3_mlx email-s PROMPT="Email Professor Chen requesting a 30-minute meeting next Tuesday to discuss on-device AI inference." SEED=4

# Demo 5: Email reply
make -C mamba3_mlx email-s PROMPT="Reply to: Hi, can we move Friday meeting to Monday 2pm?" SEED=2
```

---

## 改善建議（下次 SFT）

1. **self_awareness**：增加 50+ 個 "Who are you?" / "What are you?" / "Tell me about yourself." 的訓練例子，output 以 "I am Mamba, ..." 開頭
2. **email content**：email 的 body content 品質太低，需要更多樣化且 grounded 的訓練例子，避免模型只學格式不學內容
3. **新增 bucket**：emotion / movie_intro / system_call 完全缺訓練，若要 demo 需先加入 SFT 資料
4. **math_drill**：200 examples 太少，且模型大小(417M) 不足以保證算術正確性

Source config: `mamba3_mlx/utils/mode_configs.py`
