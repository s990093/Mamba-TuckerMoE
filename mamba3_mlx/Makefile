# mamba3_mlx — quick-launch shortcuts
# Run from this directory:  cd mamba3_mlx && make
# Or from repo root:        make -C mamba3_mlx

ROOT    := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO    := $(ROOT)/..
PYTHON  ?= $(if $(wildcard $(REPO)/.venv/bin/python3),$(REPO)/.venv/bin/python3,python3)
RUN     := $(PYTHON) $(ROOT)/run.py

CHECKPOINT ?= $(REPO)/checkpoints/latest_sft_cot_model.npz
TOKENIZER  ?= $(REPO)/cot_dataset/tokenizer.json

# ── Generation defaults ────────────────────────────────────────────────────────
PROMPT   ?= who are you?
MODE     ?= self_awareness
TEMP     ?= 0.1
MAX_TOK  ?= 512
SEED     ?= 0
STREAM   ?= 1
NO_THINK ?= 0

_STREAM_FLAG  = $(if $(filter 1,$(STREAM)),--stream,)
_THINK_FLAG   = $(if $(filter 1,$(NO_THINK)),--no-seed-think,)

_BASE = \
  --model_path     "$(CHECKPOINT)" \
  --tokenizer_path "$(TOKENIZER)" \
  --prompt         "$(PROMPT)" \
  --mode           $(MODE) \
  --temp           $(TEMP) \
  --max_tokens     $(MAX_TOK) \
  --seed           $(SEED) \
  $(_STREAM_FLAG) \
  $(_THINK_FLAG)

.PHONY: run self emotion email movie daily syscall deep default help

## run (default): self_awareness mode, "who are you?"
run:
	$(RUN) $(_BASE)

## self: same as run — self-awareness / identity
self:
	$(RUN) $(_BASE) --mode self_awareness

## emotion: emotion support
emotion:
	$(RUN) $(_BASE) --mode emotion

## email: summarise & email
email:
	$(RUN) $(_BASE) --mode summarize_email

## movie: movie intro / film analysis
movie:
	$(RUN) $(_BASE) --mode movie_intro

## daily: daily conversation
daily:
	$(RUN) $(_BASE) --mode daily_conversation

## syscall: system-call / tool invocation
syscall:
	$(RUN) $(_BASE) --mode system_call

## deep: deep dive / long-form analysis
deep:
	$(RUN) $(_BASE) --mode deep_dive

## default: plain assistant (no category system prompt)
default:
	$(PYTHON) $(ROOT)/run.py \
	  --model_path     "$(CHECKPOINT)" \
	  --tokenizer_path "$(TOKENIZER)" \
	  --prompt         "$(PROMPT)" \
	  --temp           $(TEMP) \
	  --max_tokens     $(MAX_TOK) \
	  --seed           $(SEED) \
	  $(_STREAM_FLAG) \
	  $(_THINK_FLAG)

help:
	@echo ""
	@echo "mamba3_mlx quick-launch"
	@echo ""
	@echo "  make              → self_awareness mode, PROMPT='who are you?'"
	@echo "  make self         → self-awareness / identity"
	@echo "  make emotion      → emotion support"
	@echo "  make email        → summarise & email"
	@echo "  make movie        → movie intro / analysis"
	@echo "  make daily        → daily conversation"
	@echo "  make syscall      → system-call / tool invocation"
	@echo "  make deep         → deep dive / long-form analysis"
	@echo "  make default      → plain assistant (no mode)"
	@echo ""
	@echo "Overrides (all optional):"
	@echo "  PROMPT='...'      user message  (default: 'who are you?')"
	@echo "  MODE=emotion      system prompt (default: self_awareness)"
	@echo "  TEMP=0.0          temperature   (default: 0.8;  0 = greedy)"
	@echo "  MAX_TOK=512       max tokens    (default: 256)"
	@echo "  SEED=42           random seed   (default: 0)"
	@echo "  STREAM=0          disable stream (default: 1)"
	@echo "  NO_THINK=1        skip <think> seed (default: 0)"
	@echo ""
	@echo "Examples:"
	@echo "  make                               # who are you? / self_awareness"
	@echo "  make PROMPT='Tell me about yourself' TEMP=0.0"
	@echo "  make emotion PROMPT='I feel stuck'"
	@echo "  make deep PROMPT='Explain transformers vs SSM' MAX_TOK=512"
