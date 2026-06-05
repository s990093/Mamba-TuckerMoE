"""chat_config.py — all tuneable defaults for chat_demo.py.

Edit this file to change server/model/sampling/middleware settings without
touching the argparse wiring in chat_demo.py.
"""

# ── Server ────────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 7860

# ── Paths (relative to REPO_ROOT, resolved at runtime in chat_demo.py) ───────
CHECKPOINT_RELPATH = "checkpoints/v3/latest_sft_cot_model.npz"
TOKENIZER_RELPATH  = "cot_dataset"

# ── Model sizing ──────────────────────────────────────────────────────────────
VOCAB_SIZE     = 0      # 0 = derive from tokenizer
SEQ_LEN        = 20480
MAX_NEW_TOKENS = 20480
WARMUP_STEPS   = 1

# ── Compute precision ─────────────────────────────────────────────────────────
DTYPE = "bf16"          # fp32 | bf16 | fp16

# ── Sampling ──────────────────────────────────────────────────────────────────
# Best config from tune_self.py 96-trial random search (self_awareness mode).
# Penalties apply to generated tokens only (prompt excluded from window).
TEMPERATURE   = 0.236
TOP_K         = 30
TOP_P         = 0.959
MIN_P         = 0.122
REP_PEN       = 1.281
PRES_PEN      = 0.298
FREQ_PEN      = 0.168
REPEAT_LAST_N = 128
SEED          = 0

# ── Middleware / CoT format guard ─────────────────────────────────────────────
THINK_MIN_TOKENS   = 120    # ban </think> until model has generated this many think tokens
REASONING          = True
REASONING_BUDGET   = 300    # soft cap; close_bias ramps toward this limit
FORMAT_GUARD       = True
BAN_IM_START       = True
CLOSE_BIAS         = 2.0    # gentle start bias after THINK_MIN_TOKENS
CLOSE_BIAS_MAX     = 12.0   # max nudge near REASONING_BUDGET
CLOSE_BIAS_START   = 120    # matches THINK_MIN_TOKENS — ramp starts here
FORCE_FINAL_INJECT = True
FINAL_MIN_TOKENS   = 16
