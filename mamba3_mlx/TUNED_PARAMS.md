# Tuned Parameters

Sweep date: 2026-06-10

---

## v6 (current default)

Checkpoint: `checkpoints/v6/latest_sft_cot_model.npz`

### self_awareness

```bash
make -C mamba3_mlx self PROMPT="Who are you?"
```

| Param | Value |
|-------|-------|
| SEED | 26 |
| TEMP | 0.25 |
| TOP_K | 60 |
| TOP_P | 0.856 |
| MIN_P | 0.122 |
| REP_PEN | 1.243 |
| PRES_PEN | 0.306 |
| FREQ_PEN | 0.031 |

Output (warm GPU / chat server context):
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

> ⚠️ Context-dependent: Metal GPU warm (chat server after first request) → "I am Mamba" 穩定出現。
> `make self` cold fresh process → 可能不同輸出（v6 self_awareness 訓練數據不足）。
> 160-trial sweep (temps 0.20–0.35 × seeds 0–39): 21/160 exact hits。

### summarize_email

```bash
make -C mamba3_mlx email PROMPT="Draft a short email asking Professor Chen for a 30-minute lab visit next Tuesday to discuss on-device inference."
```

| Param | Value |
|-------|-------|
| SEED | 0 |
| TEMP | 0.25 |
| TOP_K | 5 |

Output: Subject + Dear + body + Best regards（格式完整，內容有幻覺）

### daily_conversation

❌ v6 不可用

---

## v4

Checkpoint: `checkpoints/v4/latest_sft_cot_model.npz`

### self_awareness

| Param | Value |
|-------|-------|
| SEED | 0 |
| TEMP | 0.18 |
| TOP_K | 10 |

Output: `I am Mamba, a digital user with my design style...`

### summarize_email

| Param | Value |
|-------|-------|
| SEED | 17 |
| TEMP | 0.25 |
| TOP_K | 10 |

Output: Subject + Dear + body + Best regards（格式完整，內容有幻覺）

### daily_conversation

❌ v4 不可用

---

## v5

❓ 未掃描

---

## v3

❓ 未掃描（產出「lightweight lightweight...」無限迴圈）

---

Source config: `mamba3_mlx/utils/mode_configs.py`
