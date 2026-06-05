# mamba3_mlx — quick-launch shortcuts
# Run from this directory:  cd mamba3_mlx && make
# Or from repo root:        make -C mamba3_mlx

ROOT    := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO    := $(ROOT)/..
PYTHON  ?= $(if $(wildcard $(REPO)/.venv/bin/python3),$(REPO)/.venv/bin/python3,python3)
RUN     := $(PYTHON) $(ROOT)/run.py

CHECKPOINT ?= $(REPO)/checkpoints/v3/latest_sft_cot_model.npz
TOKENIZER  ?= $(REPO)/cot_dataset/tokenizer.json

# ── Auto-select fastest checkpoint ────────────────────────────────────────────
# Strips .npz / .pt to derive the MLX-native sidecar name.
# If <stem>.mlx_bf16.npz exists → pass it directly (mmap, <5 ms load).
# Otherwise fall back to CHECKPOINT (Python-side auto-converts on first run).
_CKPT_STEM := $(patsubst %.pt,%,$(patsubst %.npz,%,$(CHECKPOINT)))
_SIDECAR   := $(_CKPT_STEM).mlx_bf16.npz
_FAST_CKPT := $(if $(wildcard $(_SIDECAR)),$(_SIDECAR),$(CHECKPOINT))

# ── Generation defaults ────────────────────────────────────────────────────────
# PROMPT    ?= Who are you?
# PROMPT    ?= Write a email
MODE      ?= self_awareness
TEMP      ?= 0.2   # tune_self.py 96-trial best: 低溫穩定 CoT 結構
MAX_TOK   ?= 1024      # 保持：符合 v2 SFT 規範的預算
SEED      ?= 0
STREAM    ?= 1
NO_THINK  ?= 0

# Sampling knobs (tune_self.py 96-trial random search, self_awareness mode)
TOP_K          ?= 30      # 96-trial best for self-awareness CoT structure
TOP_P          ?= 0.959   # tune_self.py best
MIN_P          ?= 0.122   # tune_self.py best
REP_PEN        ?= 1.281   # tune_self.py best
PRES_PEN       ?= 0.298   # tune_self.py best
FREQ_PEN       ?= 0.168   # tune_self.py best
REPEAT_LAST_N  ?= 128     # 保持
# Stop control
NO_EOS    ?=0           # set to 1 to ignore EOS/<|im_end|> tokens
STOP_STR  ?=             # e.g. STOP_STR='</final>'
# Output
SHOW_SPEC ?= 0           # set to 1 to show special tokens in stream
# Performance
COMPILE   ?= 0           # set to 1 to enable mx.compile on decode steps
WARMUP    ?= 3           # JIT warmup steps when COMPILE=1

_STREAM_FLAG  = $(if $(filter 1,$(STREAM)),--stream,)
_THINK_FLAG   = $(if $(filter 1,$(NO_THINK)),--no-seed-think,)
_NOEOS_FLAG   = $(if $(filter 1,$(NO_EOS)),--no-eos-stop,)
_SPEC_FLAG    = $(if $(filter 1,$(SHOW_SPEC)),--show-special,)
_COMPILE_FLAG = $(if $(filter 1,$(COMPILE)),--full-decode-compile --warmup $(WARMUP),)
_STOPSTR_FLAG = $(if $(strip $(STOP_STR)),--stop-string "$(STOP_STR)",)

_BASE = \
  --model_path     "$(_FAST_CKPT)" \
  --tokenizer_path "$(TOKENIZER)" \
  --prompt         "$(PROMPT)" \
  --mode           $(MODE) \
  --temp           $(TEMP) \
  --top_k          $(TOP_K) \
  --top_p          $(TOP_P) \
  --min_p          $(MIN_P) \
  --rep_pen        $(REP_PEN) \
  --pres_pen       $(PRES_PEN) \
  --freq_pen       $(FREQ_PEN) \
  --repeat_last_n  $(REPEAT_LAST_N) \
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

# Print selected checkpoint path once at parse time (visible for every target).
$(info [ckpt] $(if $(wildcard $(_SIDECAR)),sidecar → $(_SIDECAR),source  → $(CHECKPOINT)  (sidecar not found — will convert on first run)))

.PHONY: run self emotion email movie daily syscall deep default debug help \
        cot-demo cot-verify chat chat-mock chat-smoke chat-kill \
        profiler profiler-kill \
        convert-sidecar \
        sjd sjd-self sjd-math sjd-daily sjd-emotion sjd-email \
        sjd-movie sjd-deep sjd-syscall \
        sjds sjds-self sjds-math sjds-daily sjds-emotion sjds-email \
        sjds-movie sjds-deep sjds-syscall

## convert-sidecar: one-time export of PyTorch-style .npz → MLX-native sidecar
## (run once; subsequent loads use mmap instant-path automatically)
convert-sidecar:
	cd $(REPO) && $(PYTHON) $(ROOT)/tools/convert_sidecar.py "$(CHECKPOINT)" --dtype bf16

## run (default): self_awareness mode, "who are you?"
run:
	$(RUN) $(_BASE)

## self: self-awareness / identity (standard single-pass)
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

## math: arithmetic drill (EN/ZH short answer)
math:
	$(RUN) $(_BASE) --mode math_drill --prompt "$(or $(PROMPT),What is 7 times 8?)"

## syscall: system-call / tool invocation
syscall:
	$(RUN) $(_BASE) --mode system_call

## deep: deep dive / long-form analysis
deep:
	$(RUN) $(_BASE) --mode deep_dive

## default: plain assistant (no category system prompt)
default:
	$(RUN) \
	  --model_path     "$(_FAST_CKPT)" \
	  --tokenizer_path "$(TOKENIZER)" \
	  --prompt         "$(PROMPT)" \
	  --temp           $(TEMP) \
	  --top_k          $(TOP_K) \
	  --top_p          $(TOP_P) \
	  --min_p          $(MIN_P) \
	  --rep_pen        $(REP_PEN) \
	  --pres_pen       $(PRES_PEN) \
	  --freq_pen       $(FREQ_PEN) \
	  --repeat_last_n  $(REPEAT_LAST_N) \
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

# ── CoT middleware (mamba3_mlx/mv/) ───────────────────────────────────────────
# These don't touch run.py / generator.py / the middleware modules — they
# wire the middleware into a separate decode loop in run_cot.py.

## cot-demo: single-shot run through cot_middleware (PROMPT='…' MAX_TOK=… MODE=…)
cot-demo:
	PROMPT="$(PROMPT)" MAX_TOKENS="$(MAX_TOK)" MODE="$(MODE)" \
	  bash $(ROOT)/run_cot.sh

## cot-verify: 5-trial sweep — verifies <think>…</think><final>…</final> framing
cot-verify:
	N=5 PROMPT="$(PROMPT)" MAX_TOKENS="$(MAX_TOK)" MODE="$(MODE)" \
	  bash $(ROOT)/run_cot_verify.sh

# ── Chat demo (WebSocket / FastAPI, mamba3_mlx native stack) ──────────────────
CHAT_PORT ?= 7860

## chat: boot chat_demo; also starts profiler on $PROFILER_PORT if not already running
chat:
	@cd $(REPO) && \
	  PROF_PID="" ; \
	  if ! lsof -Pi :$(PROFILER_PORT) -sTCP:LISTEN -t >/dev/null 2>&1 ; then \
	    $(PYTHON) -m mamba3_mlx.profiler.server --port $(PROFILER_PORT) & PROF_PID=$$! ; \
	    echo "[chat] profiler started (pid=$$PROF_PID) on :$(PROFILER_PORT)" ; \
	  else \
	    echo "[chat] profiler already running on :$(PROFILER_PORT) — reusing" ; \
	  fi ; \
	  trap '[ -n "$$PROF_PID" ] && kill $$PROF_PID 2>/dev/null; wait $$PROF_PID 2>/dev/null' EXIT INT TERM ; \
	  $(PYTHON) -m mamba3_mlx.chat_demo --port $(CHAT_PORT)

## chat-mock: boot chat_demo in mock mode (UI only, no MLX weights)
chat-mock:
	cd $(REPO) && $(PYTHON) -m mamba3_mlx.chat_demo --port $(CHAT_PORT) --mock

## chat-smoke: launch the server, fire one WS chat turn, assert <think>+<final>, shut down
chat-smoke:
	$(PYTHON) $(ROOT)/chat_smoke_test.py --port $(CHAT_PORT) \
	  --prompt "$(PROMPT)" --max-tokens $(MAX_TOK)

## chat-kill: terminate any leftover uvicorn process bound to $CHAT_PORT
chat-kill:
	@pid="$$(lsof -ti tcp:$(CHAT_PORT) 2>/dev/null)"; \
	if [ -n "$$pid" ]; then echo "killing pid $$pid on port $(CHAT_PORT)"; kill -9 $$pid; \
	else echo "no process on port $(CHAT_PORT)"; fi

# ── Profiler (standalone WebSocket metrics server) ─────────────────────────────
PROFILER_PORT ?= 8765

## profiler: boot inference profiler at http://localhost:$PROFILER_PORT
profiler:
	cd $(REPO) && $(PYTHON) -m mamba3_mlx.profiler.server --port $(PROFILER_PORT)

## profiler-kill: terminate profiler bound to $PROFILER_PORT
profiler-kill:
	@pid="$$(lsof -ti tcp:$(PROFILER_PORT) 2>/dev/null)"; \
	if [ -n "$$pid" ]; then echo "killing pid $$pid on port $(PROFILER_PORT)"; kill -9 $$pid; \
	else echo "no process on port $(PROFILER_PORT)"; fi

# ── Speculative Jacobi Decoding ───────────────────────────────────────────────
# Calls run_sjd_best.sh which uses greedy Jacobi + adaptive_K + cot_caches.
# All standard variables (PROMPT MODE MAX_TOK STREAM DTYPE) are forwarded.
# Extra SJD-specific overrides:
#   K=8          starting window size      (default: 12)
#   K_MIN=4      adaptive lower bound      (default: 4)
#   K_MAX=24     adaptive upper bound      (default: 24)
#   SJD_VERBOSE=1  per-round accept log    (default: 0)

SJD_SCRIPT  := $(ROOT)/speculative/run_sjd_best.sh
K           ?= 12
K_MIN       ?= 4
K_MAX       ?= 24
SJD_VERBOSE ?= 0

_SJD_VERBOSE_FLAG = $(if $(filter 1,$(SJD_VERBOSE)),--verbose,)

# Environment variables forwarded to run_sjd_best.sh.
# STREAM is read as an env var by the script (1=stream, 0=batch).
_SJD_ENV = \
  K="$(K)" \
  K_MIN="$(K_MIN)" \
  K_MAX="$(K_MAX)" \
  MAX_TOK="$(MAX_TOK)" \
  MODE="$(MODE)" \
  DTYPE="$(DTYPE)" \
  STREAM="$(STREAM)" \
  PYTHON="$(PYTHON)"

# Per-run flags passed as positional args through the script into run_jacobi.
_SJD_ARGS = \
  --model_path     "$(_FAST_CKPT)" \
  --tokenizer_path "$(TOKENIZER)" \
  --prompt         "$(PROMPT)" \
  $(_SJD_VERBOSE_FLAG) \
  $(_THINK_FLAG) \
  $(_NOEOS_FLAG) \
  $(_SPEC_FLAG) \
  $(_STOPSTR_FLAG)

## sjd: Jacobi speculative decode — best config (adaptive_K + cot_caches)
sjd:
	$(_SJD_ENV) bash $(SJD_SCRIPT) $(_SJD_ARGS)

## sjd-self: sjd in self_awareness mode
sjd-self:
	$(_SJD_ENV) MODE=self_awareness bash $(SJD_SCRIPT) $(_SJD_ARGS)

## sjd-math: sjd in math_drill mode
sjd-math:
	$(_SJD_ENV) MODE=math_drill bash $(SJD_SCRIPT) \
	  --model_path "$(_FAST_CKPT)" --tokenizer_path "$(TOKENIZER)" \
	  --prompt "$(or $(PROMPT),What is 7 times 8?)" \
	  $(_SJD_VERBOSE_FLAG) $(_THINK_FLAG) $(_NOEOS_FLAG) $(_SPEC_FLAG) $(_STOPSTR_FLAG)

## sjd-daily: sjd in daily_conversation mode
sjd-daily:
	$(_SJD_ENV) MODE=daily_conversation bash $(SJD_SCRIPT) $(_SJD_ARGS)

## sjd-emotion: sjd in emotion mode
sjd-emotion:
	$(_SJD_ENV) MODE=emotion bash $(SJD_SCRIPT) $(_SJD_ARGS)

## sjd-email: sjd in summarize_email mode
sjd-email:
	$(_SJD_ENV) MODE=summarize_email bash $(SJD_SCRIPT) $(_SJD_ARGS)

## sjd-movie: sjd in movie_intro mode
sjd-movie:
	$(_SJD_ENV) MODE=movie_intro bash $(SJD_SCRIPT) $(_SJD_ARGS)

## sjd-deep: sjd in deep_dive mode
sjd-deep:
	$(_SJD_ENV) MODE=deep_dive bash $(SJD_SCRIPT) $(_SJD_ARGS)

## sjd-syscall: sjd in system_call mode
sjd-syscall:
	$(_SJD_ENV) MODE=system_call bash $(SJD_SCRIPT) $(_SJD_ARGS)

# ── SJD Sampling (janobi_decode_sampling) ──  distribution-equivalent streaming
# Directly calls run_jacobi_sampling.py.  Same env vars (PROMPT MODE MAX_TOK
# STREAM DTYPE) plus SJD-specific K, COT_CACHES, COT_BUCKET.
# Defaults: stream=1, K=8, bf16, self_awareness, cot_caches_v2.pkl.

SJDS_SCRIPT := $(ROOT)/speculative/run_jacobi_sampling.py
COT_CACHES_V2 := $(ROOT)/speculative/cot_caches_n4.pkl
SJDS_K         ?= 8
COT_BUCKET     ?= $(MODE)
DTYPE          ?= bf16

_SJDS_BASE = \
  --model_path     "$(_FAST_CKPT)" \
  --tokenizer_path "$(TOKENIZER)" \
  --prompt         "$(PROMPT)" \
  --mode           $(MODE) \
  --max_tokens     $(MAX_TOK) \
  --K              $(SJDS_K) \
  --dtype          $(DTYPE) \
  --no-eos-stop \
  $(_STREAM_FLAG) \
  $(_THINK_FLAG) \
  $(_SPEC_FLAG)

_SJDS_COT = \
  --cot_caches "$(COT_CACHES_V2)" \
  --cot_bucket "$(COT_BUCKET)"

## sjds: SJD sampling stream — best config (distribution-equivalent, ~1.5× vs AR)
sjds:
	$(PYTHON) $(SJDS_SCRIPT) $(_SJDS_BASE) $(_SJDS_COT)

## sjds-self: sjds in self_awareness mode
sjds-self:
	$(PYTHON) $(SJDS_SCRIPT) $(_SJDS_BASE) $(_SJDS_COT) --mode self_awareness

## sjds-math: sjds in math_drill mode
sjds-math:
	$(PYTHON) $(SJDS_SCRIPT) $(_SJDS_BASE) $(_SJDS_COT) \
	  --mode math_drill --prompt "$(or $(PROMPT),What is 7 times 8?)"

## sjds-daily: sjds in daily_conversation mode
sjds-daily:
	$(PYTHON) $(SJDS_SCRIPT) $(_SJDS_BASE) $(_SJDS_COT) --mode daily_conversation

## sjds-emotion: sjds in emotion mode
sjds-emotion:
	$(PYTHON) $(SJDS_SCRIPT) $(_SJDS_BASE) $(_SJDS_COT) --mode emotion

## sjds-email: sjds in summarize_email mode
sjds-email:
	$(PYTHON) $(SJDS_SCRIPT) $(_SJDS_BASE) $(_SJDS_COT) --mode summarize_email

## sjds-movie: sjds in movie_intro mode
sjds-movie:
	$(PYTHON) $(SJDS_SCRIPT) $(_SJDS_BASE) $(_SJDS_COT) --mode movie_intro

## sjds-deep: sjds in deep_dive mode
sjds-deep:
	$(PYTHON) $(SJDS_SCRIPT) $(_SJDS_BASE) $(_SJDS_COT) --mode deep_dive

## sjds-syscall: sjds in system_call mode
sjds-syscall:
	$(PYTHON) $(SJDS_SCRIPT) $(_SJDS_BASE) $(_SJDS_COT) --mode system_call

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
	@echo "  make math         → math drill (PROMPT='What is 23 times 45?')"
	@echo "  make syscall      → system-call / tool invocation"
	@echo "  make deep         → deep dive / long-form analysis"
	@echo "  make default      → plain assistant (no mode)"
	@echo "  make debug        → no-eos-stop + show-special (raw model inspection)"
	@echo ""
	@echo "Variables (all optional):"
	@echo "  PROMPT='...'         user message              (default: 'who are you?')"
	@echo "  MODE=emotion         system prompt mode        (default: self_awareness)"
	@echo "  TEMP=0.0             temperature               (default: 0.1;  0 = greedy)"
	@echo "  TOP_K=50             top-k candidates          (default: 40;   0 = off)"
	@echo "  TOP_P=0.95           nucleus sampling p        (default: 0.9)"
	@echo "  MIN_P=0.02           min-p filter              (default: 0.05)"
	@echo "  REP_PEN=1.2          repetition penalty        (default: 1.1;  1.0 = off)"
	@echo "  PRES_PEN=0.1         presence penalty          (default: 0.0)"
	@echo "  FREQ_PEN=0.05        frequency penalty         (default: 0.02)"
	@echo "  REPEAT_LAST_N=128    penalty context window    (default: 64)"
	@echo "  MAX_TOK=512          max tokens to generate    (default: 512)"
	@echo "  SEED=42              random seed               (default: 0)"
	@echo "  STREAM=0             0=batch 1=stream          (default: 1)"
	@echo "  NO_THINK=1           skip <think> seed         (default: 0)"
	@echo "  NO_EOS=1             ignore EOS tokens         (default: 0)"
	@echo "  STOP_STR='</x>'      extra stop string         (default: empty)"
	@echo "  SHOW_SPEC=1          show special tokens       (default: 0)"
	@echo "  COMPILE=1            mx.compile decode         (default: 0)"
	@echo "  WARMUP=5             JIT warmup steps          (default: 3)"
	@echo ""
	@echo "Examples:"
	@echo "  make                                      # default: who are you? / self_awareness"
	@echo "  make PROMPT='Tell me about yourself' TEMP=0.0"
	@echo "  make emotion PROMPT='I feel stuck and overwhelmed'"
	@echo "  make deep PROMPT='Explain SSM vs Transformer' MAX_TOK=800"
	@echo "  make debug PROMPT='Hello' MAX_TOK=50     # inspect past EOS boundary"
	@echo "  make self NO_EOS=1 STOP_STR='</final>'   # stop on </final> only"
	@echo "  make self COMPILE=1                       # faster decode after warmup"
	@echo ""
	@echo "Speculative Jacobi Decoding (greedy + adaptive_K + cot_caches):"
	@echo "  make sjd                                   # default: self_awareness, stream"
	@echo "  make sjd-self                              # self_awareness mode"
	@echo "  make sjd-math                              # math_drill mode"
	@echo "  make sjd-daily                             # daily_conversation mode"
	@echo "  make sjd-emotion                           # emotion mode"
	@echo "  make sjd-email                             # summarize_email mode"
	@echo "  make sjd-movie                             # movie_intro mode"
	@echo "  make sjd-deep                              # deep_dive mode"
	@echo "  make sjd-syscall                           # system_call mode"
	@echo ""
	@echo "  SJD variables (on top of standard PROMPT/MODE/MAX_TOK/STREAM/DTYPE):"
	@echo "  K=8            starting Jacobi window       (default: 12)"
	@echo "  K_MIN=4        adaptive lower bound         (default: 4)"
	@echo "  K_MAX=24       adaptive upper bound         (default: 24)"
	@echo "  SJD_VERBOSE=1  per-round accept diagnostics (default: 0)"
	@echo ""
	@echo "  Examples:"
	@echo "  make sjd PROMPT='Explain SSMs' MODE=deep_dive"
	@echo "  make sjd-math PROMPT='Solve 3x+5=17' MAX_TOK=256"
	@echo "  make sjd K=8 STREAM=0 PROMPT='Hello'"
	@echo "  make sjd SJD_VERBOSE=1 MAX_TOK=100"
	@echo ""
	@echo "SJD Sampling (distribution-equivalent, best speed, default stream=1):"
	@echo "  make sjds                                  # SJD sampling, self_awareness"
	@echo "  make sjds-self                             # self_awareness mode"
	@echo "  make sjds-math                             # math_drill mode"
	@echo "  make sjds-daily                            # daily_conversation mode"
	@echo "  make sjds-emotion                          # emotion mode"
	@echo "  make sjds-email                            # summarize_email mode"
	@echo "  make sjds-movie                            # movie_intro mode"
	@echo "  make sjds-deep                             # deep_dive mode"
	@echo "  make sjds-syscall                          # system_call mode"
	@echo ""
	@echo "  SJD sampling variables:"
	@echo "  SJDS_K=8       Jacobi window               (default: 8)"
	@echo "  COT_BUCKET=    COT per-category bucket     (default: same as MODE)"
	@echo ""
	@echo "  Examples:"
	@echo "  make sjds PROMPT='Tell me about yourself'"
	@echo "  make sjds-math PROMPT='Solve 2x+3=11' MAX_TOK=256"
	@echo "  make sjds SJDS_K=12 STREAM=0               # batch mode, larger K"
	@echo ""
	@echo "CoT middleware (mamba3_mlx/mv/cot_*.py):"
	@echo "  make cot-demo                              # one shot through middleware"
	@echo "  make cot-verify                            # 5-trial sweep, asserts <think>+<final> blocks"
	@echo "  make cot-demo PROMPT='Explain X' MAX_TOK=512"
	@echo ""
	@echo "Chat WebSocket server (mamba3_mlx native stack):"
	@echo "  make chat                                  # boot server at http://localhost:7860"
	@echo "  make chat-mock                             # UI-only (no MLX weights)"
	@echo "  make chat-smoke                            # boot + one WS turn + verdict"
	@echo "  make chat-kill                             # kill leftover server on \$$CHAT_PORT"
	@echo "  CHAT_PORT=8080 make chat                   # custom port"
	@echo ""
	@echo "Inference profiler (standalone metrics WebSocket):"
	@echo "  make profiler                              # http://localhost:8765  ws://localhost:8765/ws"
	@echo "  make profiler-kill                         # kill profiler on \$$PROFILER_PORT"
	@echo "  PROFILER_PORT=9000 make profiler           # custom port"
