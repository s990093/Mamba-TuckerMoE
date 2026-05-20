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
PROMPT    ?= who are you?
MODE      ?= self_awareness
TEMP      ?= 0.1
MAX_TOK   ?= 512
SEED      ?= 0
STREAM    ?= 1
NO_THINK  ?= 0
# Stop control
NO_EOS    ?= 0          # set to 1 to ignore EOS/<|im_end|> tokens
STOP_STR  ?=            # e.g. STOP_STR='</final>'
# Output
SHOW_SPEC ?= 0          # set to 1 to show special tokens in stream
# Performance
COMPILE   ?= 0          # set to 1 to enable mx.compile on decode steps
WARMUP    ?= 3          # JIT warmup steps when COMPILE=1

_STREAM_FLAG  = $(if $(filter 1,$(STREAM)),--stream,)
_THINK_FLAG   = $(if $(filter 1,$(NO_THINK)),--no-seed-think,)
_NOEOS_FLAG   = $(if $(filter 1,$(NO_EOS)),--no-eos-stop,)
_SPEC_FLAG    = $(if $(filter 1,$(SHOW_SPEC)),--show-special,)
_COMPILE_FLAG = $(if $(filter 1,$(COMPILE)),--full-decode-compile --warmup $(WARMUP),)
_STOPSTR_FLAG = $(if $(strip $(STOP_STR)),--stop-string "$(STOP_STR)",)

_BASE = \
  --model_path     "$(CHECKPOINT)" \
  --tokenizer_path "$(TOKENIZER)" \
  --prompt         "$(PROMPT)" \
  --mode           $(MODE) \
  --temp           $(TEMP) \
  --max_tokens     $(MAX_TOK) \
  --seed           $(SEED) \
  $(_STREAM_FLAG) \
  $(_THINK_FLAG) \
  $(_NOEOS_FLAG) \
  $(_SPEC_FLAG) \
  $(_COMPILE_FLAG) \
  $(_STOPSTR_FLAG)

_REQUIRE_PROMPT = @test -n "$(PROMPT)" || \
  (echo 'Set PROMPT="your question"  (current default: who are you?)' && exit 1)

.PHONY: run self emotion email movie daily syscall deep default debug help

## run (default): self_awareness mode, "who are you?"
run:
	$(RUN) $(_BASE)

## self: self-awareness / identity
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
	$(RUN) \
	  --model_path     "$(CHECKPOINT)" \
	  --tokenizer_path "$(TOKENIZER)" \
	  --prompt         "$(PROMPT)" \
	  --temp           $(TEMP) \
	  --max_tokens     $(MAX_TOK) \
	  --seed           $(SEED) \
	  $(_STREAM_FLAG) \
	  $(_THINK_FLAG) \
	  $(_NOEOS_FLAG) \
	  $(_SPEC_FLAG) \
	  $(_COMPILE_FLAG) \
	  $(_STOPSTR_FLAG)

## debug: no-eos-stop + show-special — see raw model output past EOS boundary
debug:
	$(RUN) $(_BASE) --no-eos-stop --show-special

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
	@echo "  make debug        → no-eos-stop + show-special (raw model inspection)"
	@echo ""
	@echo "Variables (all optional):"
	@echo "  PROMPT='...'      user message         (default: 'who are you?')"
	@echo "  MODE=emotion      system prompt mode   (default: self_awareness)"
	@echo "  TEMP=0.0          temperature          (default: 0.1;  0 = greedy)"
	@echo "  MAX_TOK=512       max tokens           (default: 512)"
	@echo "  SEED=42           random seed          (default: 0)"
	@echo "  STREAM=0          0=batch 1=stream     (default: 1)"
	@echo "  NO_THINK=1        skip <think> seed    (default: 0)"
	@echo "  NO_EOS=1          ignore EOS tokens    (default: 0)"
	@echo "  STOP_STR='</x>'   extra stop string    (default: empty)"
	@echo "  SHOW_SPEC=1       show special tokens  (default: 0)"
	@echo "  COMPILE=1         mx.compile decode    (default: 0)"
	@echo "  WARMUP=5          JIT warmup steps     (default: 3, used when COMPILE=1)"
	@echo ""
	@echo "Examples:"
	@echo "  make                                      # default: who are you? / self_awareness"
	@echo "  make PROMPT='Tell me about yourself' TEMP=0.0"
	@echo "  make emotion PROMPT='I feel stuck and overwhelmed'"
	@echo "  make deep PROMPT='Explain SSM vs Transformer' MAX_TOK=800"
	@echo "  make debug PROMPT='Hello' MAX_TOK=50     # inspect past EOS boundary"
	@echo "  make self NO_EOS=1 STOP_STR='</final>'   # stop on </final> only"
	@echo "  make self COMPILE=1                       # faster decode after warmup"
