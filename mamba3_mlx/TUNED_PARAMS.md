# Tuned Parameters & Demo Capability Map

Last updated: 2026-06-16

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
