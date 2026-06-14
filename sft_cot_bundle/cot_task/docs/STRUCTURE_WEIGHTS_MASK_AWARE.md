# Structure Weights with Mask-Aware Weighting

## Overview

Structure weight calculation now implements **mask-aware weighting**: structure patterns (Step, pipe, separator, bold, heading, fenced code) only receive weight multipliers (`w_struct`) in the **assistant portion** of the sequence. User/system tokens remain at weight `1.0` (no amplification).

**Why?** The training loss calculation (in `train_sft.py::_build_xy_masked`) masks user/system tokens with `labels=-100`, meaning they don't contribute to the loss. Therefore, amplifying their weights during the offline weighting phase would be wasteful and inconsistent.

---

## Key Architecture

### Mask Boundary Detection

The function `find_assistant_start_token()` in `build_structure_weights.py` locates the `<|im_start|>assistant\n` marker and returns the token index where the assistant content begins.

```python
def find_assistant_start_token(
    text: str,
    tok: PreTrainedTokenizerFast,
    offset_mapping: list[tuple[int, int]],
) -> int:
    """
    Find <|im_start|>assistant\n position in text.
    Return the token index of the first token after this marker.
    If not found, return 0 (entire sequence computed for loss).
    """
    marker = "<|im_start|>assistant\n"
    marker_pos = text.find(marker)
    if marker_pos == -1:
        return 0
    
    marker_end_pos = marker_pos + len(marker)
    for i, (char_start, char_end) in enumerate(offset_mapping):
        if char_start >= marker_end_pos:
            return i
    return len(offset_mapping)
```

### Weight Application Logic

Only tokens with index `>= assistant_start_token` receive the `w_struct` multiplier:

```python
for pattern_name, matches in structure_matches.items():
    for char_start, char_end in matches:
        indices = char_span_to_token_indices(
            char_start, char_end, tok_result.offset_mapping, policy=policy
        )
        for i in indices:
            # Only apply w_struct to assistant tokens
            if i >= assistant_start_token:
                weights[i] *= w_struct
                structure_indices.add(i)
```

---

## Outputs

### Weight Vector (`weight` in .npz)
- Shape: `[T]` where `T = len(input_ids)`
- **User/system positions**: `1.0` (no amplification)
- **Assistant positions with structure**: `w_struct * ... * w_struct` (clipped to `[w_min, w_max]`)
- **Assistant positions without structure**: `1.0`

### Metadata
Each `.npz` file now includes:
- `assistant_start_token`: (int) Token index where assistant begins

This is recorded in the metadata JSON for verification:

```json
{
  "sample_id": "...",
  "token_length": 512,
  "assistant_start_token": 145,
  "structure_tokens_per_sample": 52,
  "pattern_distribution": {
    "step": 3,
    "pipe": 13,
    "separator": 13,
    "bold": 5,
    "heading": 0,
    "fenced_code": 0
  }
}
```

---

## Validation Results (50-sample run)

| Metric | Value |
|--------|-------|
| Total samples | 50 |
| Mean structure tokens per sample | **52.9** |
| Mean tokens per sample (total) | 224.8 |
| Structure token ratio | **23.6%** (only in assistant portion) |
| Weight min | 1.0 |
| Weight max | 10.0 |
| Weight mean | 1.63 |

**Key observation**: Structure tokens reduced from 57.9 → 52.9 (9% reduction) because user/system patterns are now excluded from weighting (though still counted in visualization for completeness).

---

## Training Integration

### CE Loss Calculation
In `train_sft.py::_build_xy_masked`:

```python
# Only assistant tokens get labels != -100
labels[j] = ids[j]  # for j in [assistant_start, end_of_assistant)
# All other positions: labels[j] = -100 (ignored)
```

### Loss Weighting During Training
When computing loss:

```python
# During training, loss_weight is applied to assistant tokens only
# User/system tokens have labels=-100, so they're automatically skipped
loss_weight[i] * ce_loss[i]  # for positions where labels[i] != -100
```

---

## RE Patterns (6 types, all 6 supported)

| ID | Pattern Name | Regex | Status |
|---|---|---|---|
| R1 | Step | `Step\s*\d+[:：]?` | Detected in assistant only |
| R2 | Pipe | `\|` | Detected in assistant only |
| R3 | Separator | `\|?[\s\-]*\|[\s\-]*\|?` | Detected in assistant only |
| R4 | Bold | `\*\*[^*]+\*\*` | Detected in assistant only |
| R5 | Heading | `^#+\s` | Detected in assistant only (low frequency) |
| R6 | Fenced code | ` ``` ` | Detected in assistant only (rare in dataset) |

---

## Visualization

### HTML Output
All 6 pattern types are color-coded:
- **R1 (Step)**: Pink (#FFB6C6)
- **R2 (Pipe)**: Yellow (#FFE699)
- **R3 (Separator)**: Green (#C6E0B4)
- **R4 (Bold)**: Blue (#B4C7E7)
- **R5 (Heading)**: Light green (#E2EFDA)
- **R6 (Code)**: Orange (#F4B084)

### Text Output (Terminal)
ANSI-colored with pattern type awareness, same color scheme as HTML.

---

## Consistency Checks

### Checkpoint: Mask Boundary Alignment
```bash
# Verify that find_assistant_start_token matches _build_xy_masked
python verify_stf_cot_mask.py --weights-dir reports/structure_weights
```

### Manual Inspection
Open `/cot/reports/structure_samples.html` to verify:
1. High-weight tokens appear only in assistant region
2. User/system content shows base weight color (if any)
3. No weights amplified before the `<|im_start|>assistant` marker

---

## Parameters (Defaults)

| Parameter | Default | Range | Notes |
|---|---|---|---|
| `w_struct` | 3.0 | [1.0, 10.0) | Multiplier for structure tokens |
| `w_min` | 0.25 | [0.1, 1.0] | Floor after clipping |
| `w_max` | 10.0 | [2.0, 100.0] | Ceiling after clipping |
| `policy` | `union` | {`union`, `first`} | Token mapping strategy |

---

## Files Modified

1. **`cot/build_structure_weights.py`**
   - Added `find_assistant_start_token()` function
   - Modified `build_structure_weights()` to apply `w_struct` only to assistant tokens
   - Saves `assistant_start_token` in results

2. **`cot/docs/TASK2_LOSS_ENGINEERING.md`**
   - Updated §4.3 Step 4 ("寫入權重") with mask-aware logic
   - Updated §4.3 Step 5 ("與 labels 對齊") with consistency notes

3. **`cot/visualize_structure_weights.py`**
   - Enhanced to support 6-pattern color visualization
   - Added pattern-specific ANSI colors and HTML colors
   - HTML legend displays all 6 pattern types

4. **`cot/validate_and_plot.py`**
   - Auto-adapts to display all 6 patterns in pie chart
   - No changes needed (reads from metadata)

---

## Future Work

- [ ] Extend `IPS` (Inverse Probability Sampling) to respect mask boundaries
- [ ] Implement per-region SCALe scheduling (different weights for think vs. final)
- [ ] Add Focal Loss support with mask awareness
- [ ] Monitor weight distribution statistics per region (user/system/assistant)

---

## References

- [`train_sft.py::_build_xy_masked`](../../sft_cot_bundle/scripts/train_sft.py) — Authoritative mask logic
- [`TASK2_LOSS_ENGINEERING.md`](./TASK2_LOSS_ENGINEERING.md) — Loss engineering specs
- [`SFT_FORMAT.md`](../../cot_dataset/SFT_FORMAT.md) — ChatML format specification
