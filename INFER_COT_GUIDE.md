# Interactive CoT Inference Guide

## Quick Start

### List All System Prompts
```bash
python -m mamba3_mlx.infer_cot --list-categories
```

Output:
```
═════════════════════════════════════════════════════════════════════════════
AVAILABLE SYSTEM PROMPTS
═════════════════════════════════════════════════════════════════════════════

1. EMOTION (emotion)
   You are Mamba in Emotion mode. Respond with calm precision...

2. SELF-AWARENESS (self_awareness)
   You are Mamba in Self-Awareness mode. Answer identity...

3. EMAIL / SUMMARY (email_summary)
   You are Mamba in Summarize&Email mode...

4. MOVIE INTRO (movie_intro)
   You are Mamba in Movie Intro mode...

5. DAILY CONVERSATION (daily_conversation)
   You are Mamba in Daily Conversation mode...

6. SYSTEM CALL (system_call)
   You are Mamba in System Call mode...

7. DEEP DIVE (deep_dive)
   You are Mamba in Deep Dive mode...
```

---

## Usage Modes

### 1. Interactive Mode (Choose Everything)

```bash
python -m mamba3_mlx.infer_cot --interactive
```

Then the script will:
1. Show all 7 system prompts
2. Ask you to choose (1-7)
3. Ask you to enter your question

Example:
```
╔════════════════════════════════════════════════════════════════════════════╗
║                    INTERACTIVE COT INFERENCE                              ║
╚════════════════════════════════════════════════════════════════════════════╝

...categories list...

Choose category (1-7): 2

✓ Selected: Self-Awareness
  Prompt: You are Mamba in Self-Awareness mode...

Enter your question: What are your limitations?
```

---

### 2. Command-Line Mode (Specify Arguments)

```bash
# With specific prompt and category
python -m mamba3_mlx.infer_cot \
  --prompt "What is your purpose?" \
  --category self_awareness

# With emotion category
python -m mamba3_mlx.infer_cot \
  --prompt "I feel overwhelmed about this project" \
  --category emotion

# With deep dive analysis
python -m mamba3_mlx.infer_cot \
  --prompt "Explain state space models" \
  --category deep_dive \
  --max-tokens 400

# With custom checkpoint
python -m mamba3_mlx.infer_cot \
  --checkpoint /path/to/model.npz \
  --tokenizer /path/to/tokenizer \
  --prompt "Who are you?" \
  --category self_awareness
```

---

## Available Categories

| Category | Key | Use Case |
|----------|-----|----------|
| **Emotion** | `emotion` | Process emotional states, reframe as system states, suggest actions |
| **Self-Awareness** | `self_awareness` | Describe identity, architecture, capabilities, limitations |
| **Email / Summary** | `email_summary` | Summarize emails, extract key points, draft responses |
| **Movie Intro** | `movie_intro` | Analyze films, discuss themes, compare works |
| **Daily Conversation** | `daily_conversation` | General questions, practical answers |
| **System Call** | `system_call` | Detect when tool invocation needed, emit call syntax |
| **Deep Dive** | `deep_dive` | Long-form analysis, problem modeling, trade-offs |

---

## Example Sessions

### Example 1: Self-Awareness

```bash
python -m mamba3_mlx.infer_cot \
  --prompt "Who are you and what makes you different from other LLMs?" \
  --category self_awareness \
  --max-tokens 300
```

Expected output structure:
```
REASONING BLOCK
──────────────
<think>
Let me think about my unique characteristics...
I'm a hybrid model combining Mamba SSM with TuckerMoE...
This is different from transformer-based models because...
</think>

FINAL ANSWER
────────────
I am Mamba, a 417M parameter hybrid model combining:
- State Space Model (Mamba) backbone
- Tucker-decomposed Mixture-of-Experts
- Optimized for Apple Silicon inference
```

### Example 2: Emotion Mode

```bash
python -m mamba3_mlx.infer_cot \
  --prompt "I'm struggling with technical debt in my codebase" \
  --category emotion
```

Expected output:
```
REASONING BLOCK
──────────────
<think>
This is a system state problem. Let me identify:
- The pressure source: accumulated technical compromises
- Controllable variables: refactoring pace, prioritization
- Time constraints: competing demands
</think>

FINAL ANSWER
────────────
Technical debt is a resource allocation problem, not a moral failing.
Controllable actions:
1. Quantify impact (how much velocity loss?)
2. Prioritize highest-friction areas
3. Set sustainable refactoring cadence
```

### Example 3: Deep Dive

```bash
python -m mamba3_mlx.infer_cot \
  --prompt "Compare state space models vs transformers for long-context tasks" \
  --category deep_dive \
  --max-tokens 400
```

Expected output:
```
REASONING BLOCK
──────────────
<think>
I need to model:
1. Computational complexity (SSM: O(N), Transformer: O(N²))
2. Memory efficiency (SSM: constant state, Transformer: full cache)
3. Training dynamics (SSM: sequential, Transformer: parallel)
4. Empirical strengths/weaknesses
</think>

FINAL ANSWER
────────────
Problem: Transformer quadratic complexity breaks for 32K+ context

State Space Models (Mamba):
• Strength: O(N) complexity enables unlimited context
• Weakness: Sequential training, lower per-token capacity
• Tradeoff: Sacrifice training parallelism for inference efficiency

Comparison table:
  Metric | Transformer | Mamba
  ──────────────────────────────
```

---

## Output Interpretation

### Quality Metrics

**✅ PASS: CoT Separation Working**
```
✓ Has reasoning: True
✓ Has final answer: True
✓ Reached final mode: True

✅ PASS: CoT separation working
   - Reasoning block properly identified
   - Final answer properly generated
   - FSM state transitions correct
```

**⚠️ INCOMPLETE: Missing Reasoning**
```
✓ Has reasoning: False
✓ Has final answer: True
✓ Reached final mode: True

⚠️ INCOMPLETE: No reasoning block detected
   (Model jumped straight to final answer)
```

**⚠️ INCOMPLETE: Missing Final Answer**
```
✓ Has reasoning: True
✓ Has final answer: False
✓ Reached final mode: False

⚠️ INCOMPLETE: No final answer generated
   (Model got stuck in reasoning mode)
```

**❌ FAIL: Format Parsing Broken**
```
✓ Has reasoning: False
✓ Has final answer: False
✓ Reached final mode: False

❌ FAIL: Format parsing broken
   (Neither reasoning nor final answer found)
```

---

## Middleware State Report

After each inference, you'll see detailed middleware status:

```json
{
  "budget_tokens": 500,
  "think_tokens": 145,
  "budget_ok": true,
  "bias_ramp": 0.29,
  "should_break": false,
  "has_reasoning": true,
  "format_guard": {
    "enabled": true,
    "ban_ids": [2, 32000, 32001],
    "close_targets": {
      "think": 32003,
      "between": 32004,
      "final": 32005
    }
  }
}
```

Key indicators:
- **budget_ok**: Reasoning stayed within token budget ✓
- **close_targets**: Should show 32003-32005 (not 829) ✓
- **has_reasoning**: Should be true ✓

---

## Performance Expectations

| Operation | Time |
|-----------|------|
| Model load + warmup | 5-10 seconds |
| Prefill (system + prompt) | 2-5 seconds |
| First token to reasoning block | 200-500ms |
| Decode (reasoning + answer) | 2-8 seconds |
| **Total per query** | **10-20 seconds** |

---

## Troubleshooting

### Issue: No reasoning block generated

**Symptom:**
```
✓ Has reasoning: False
✓ Has final answer: True
```

**Cause:** Model generated answer without thinking

**Solutions:**
1. Try with `--category deep_dive` (encourages analysis)
2. Rephrase prompt to ask "how would you approach..."
3. Increase `--max-tokens`
4. Reduce `--temp` (should be < 0.8 for structured output)

### Issue: Stuck in reasoning mode

**Symptom:**
```
✓ Has reasoning: True
✓ Has final answer: False
✓ Reached final mode: False
```

**Cause:** Model never generated `</think>` or got stuck

**Solutions:**
1. Check format guard shows correct close targets (32003, 32005)
2. Try `--temp 0.5` (lower temperature)
3. Reduce `--max-tokens` (easier to complete)
4. Check middleware report shows `budget_ok: true`

### Issue: Close bias targets show 829 instead of 32003-32005

**Symptom:**
```
"close_targets": {
  "think": 829,
  "final": 829
}
```

**Cause:** Vocab size detection failed (the issue we fixed!)

**Solutions:**
1. Verify `mamba3_mlx/server.py` has lines 209-217
2. Verify git status: `git diff HEAD~2`
3. Run: `python mamba3_mlx/diagnose_cot.py --test all`
4. Check tokenizer backend: `tokenizer.backend_tokenizer.get_vocab()`

---

## Batch Testing Script

To test all categories automatically:

```python
#!/usr/bin/env python3
import subprocess

categories = [
    "emotion",
    "self_awareness",
    "email_summary",
    "movie_intro",
    "daily_conversation",
    "system_call",
    "deep_dive",
]

test_prompts = {
    "emotion": "I'm feeling stuck on this problem",
    "self_awareness": "Who are you?",
    "email_summary": "Summarize this email about quarterly review",
    "movie_intro": "Analyze the themes in Inception",
    "daily_conversation": "What is machine learning?",
    "system_call": "What time is it right now?",
    "deep_dive": "Explain attention mechanisms in detail",
}

for cat in categories:
    prompt = test_prompts[cat]
    print(f"\n{'='*80}")
    print(f"Testing: {cat.upper()}")
    print(f"{'='*80}")
    
    cmd = [
        "python", "-m", "mamba3_mlx.infer_cot",
        "--prompt", prompt,
        "--category", cat,
        "--max-tokens", "150",
    ]
    
    subprocess.run(cmd)
```

Save as `test_all_categories.py` and run:
```bash
python test_all_categories.py 2>&1 | tee test_results.log
```

---

## Advanced Options

```bash
# Use float32 for better precision (slower)
python -m mamba3_mlx.infer_cot \
  --prompt "Who are you?" \
  --category self_awareness \
  --dtype fp32

# Generate longer response
python -m mamba3_mlx.infer_cot \
  --prompt "Explain quantum computing" \
  --category deep_dive \
  --max-tokens 500

# Lower temperature for more consistent output
python -m mamba3_mlx.infer_cot \
  --prompt "What is 2+2?" \
  --category daily_conversation \
  --temp 0.3

# Higher temperature for more creative output
python -m mamba3_mlx.infer_cot \
  --prompt "Write a haiku about models" \
  --category daily_conversation \
  --temp 0.9
```

---

## Summary

This script provides a unified interface to:

1. ✅ **Load** the model with fixed vocab_size detection
2. ✅ **Initialize** CoT middleware with correct format guard
3. ✅ **Infer** with system prompt selection
4. ✅ **Split** output into reasoning + final answer
5. ✅ **Verify** quality metrics and FSM state

Use it to validate that the CoT generation quality has improved after the vocab_size fixes.

---

**Ready to test!** Try:
```bash
python -m mamba3_mlx.infer_cot --interactive
```

Last updated: 2026-05-19
