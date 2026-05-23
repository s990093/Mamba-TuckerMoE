"""chat_config.py — all tuneable defaults for chat_demo.py.

Edit this file to change server/model/sampling/middleware settings without
touching the argparse wiring in chat_demo.py.
"""

# ── Server ────────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 7860

# ── Paths (relative to REPO_ROOT, resolved at runtime in chat_demo.py) ───────
CHECKPOINT_RELPATH = "checkpoints/latest_sft_cot_model.npz"
TOKENIZER_RELPATH  = "checkpoints/tokenizer"

# ── Model sizing ──────────────────────────────────────────────────────────────
VOCAB_SIZE     = 0      # 0 = derive from tokenizer
SEQ_LEN        = 2048
MAX_NEW_TOKENS = 2048
WARMUP_STEPS   = 1

# ── Compute precision ─────────────────────────────────────────────────────────
DTYPE = "bf16"          # fp32 | bf16 | fp16

# ── Sampling ──────────────────────────────────────────────────────────────────
TEMPERATURE   = 0.3
TOP_K         = 40
TOP_P         = 0.9
MIN_P         = 0.05
REP_PEN       = 1.3
PRES_PEN      = 0.3
FREQ_PEN      = 0.2
REPEAT_LAST_N = 64
SEED          = 0

# ── Middleware / CoT format guard ─────────────────────────────────────────────
REASONING          = True
REASONING_BUDGET   = 0      # 0 = no cap; model self-paces <think>
FORMAT_GUARD       = True
BAN_IM_START       = True
CLOSE_BIAS         = 4.0
CLOSE_BIAS_MAX     = 16.0
CLOSE_BIAS_START   = 0
FORCE_FINAL_INJECT = True
FINAL_MIN_TOKENS   = 16
