# Tuned Parameters (v4 checkpoint)

Sweep date: 2026-06-08 | checkpoint: `checkpoints/v4/latest_sft_cot_model.npz`

---

## self_awareness

```bash
make -C mamba3_mlx self PROMPT="who are you?"
```

| Param | Value |
|-------|-------|
| TEMP | 0.259 |
| TOP_K | 60 |
| TOP_P | 0.856 |
| MIN_P | 0.122 |
| REP_PEN | 1.243 |
| PRES_PEN | 0.306 |
| FREQ_PEN | 0.031 |
| SEED | 5 |

> ⚠️ 所有參數耦合：任何一個改變都會破壞 seed=5 的輸出路徑。

Output:
```
I am **Mamba**, an offline language model designed for local or Itemy physical
infrastructure. I can only report what you know by my parameters (weights,
weights, data) but no one was relying on me to analyze; I had no critical
identity system.
```

---

## summarize_email

```bash
make -C mamba3_mlx email PROMPT="Draft a short email asking Professor Chen for a 30-minute lab visit next Tuesday to discuss on-device inference."
```

| Param | Value |
|-------|-------|
| TEMP | 0.25 |
| TOP_K | 10 |
| TOP_P | 0.85 |
| MIN_P | 0.10 |
| REP_PEN | 1.15 |
| PRES_PEN | 0.10 |
| FREQ_PEN | 0.03 |
| SEED | 17 |

Output:
```
**Subject:** Friday Proposal — June 5
**Dear Professoruser,**
I would like to discuss the on-device ... next Tuesday for a 30-minute field visit.
**Best regards**
```

---

## daily_conversation

❌ 不可用 — noise3/noise4 (3000+ CoT 日常問答) 未被訓練進 v4。

---

Full tuning results: `mamba3_mlx/tune_self_v4_results.json`
Source config: `mamba3_mlx/utils/mode_configs.py`
