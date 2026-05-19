# Mamba3-XR — local dev shortcuts (run from repo root: `make mlx-bench`)
#
# Override examples:
#   make mlx-bench CHECKPOINT=checkpoint.pt
#   make mlx-bench SEQ_LEN=256 DECODE_TOK=64
#   make mlx-bench BENCH_EXTRA='--prompt "Hi"'   # no truncation unless SEQ_LEN is set
#   make mlx-export-npz CHECKPOINT=weights/checkpoint.pt

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Prefer repo .venv so `make mlx-bench` works without activating the venv
VENV_PY := $(ROOT)/.venv/bin/python3
PYTHON ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python3)
BENCH := $(ROOT)/inference/benchmark_mlx.py
STREAM := $(ROOT)/inference/stream_mlx.py
PROF := $(ROOT)/inference/tools/profile_mlx_infer.py
TOK ?= $(ROOT)/inference/tokenizer
PROFILE_DECODE_STEPS ?= 32
# Preset: throughput | safe | eager | sequential-ssm | custom
INFER_TYPE ?= throughput
# Model compute / weight dtype: fp32 | bf16 | fp16
DTYPE ?= bf16

# Leave empty to use resolve_mlx_checkpoint(): repo model.npz → checkpoint.pt sidecars
CHECKPOINT ?=

# Optional max prefill tokens. Empty = use prompt token length directly (no truncation).
SEQ_LEN ?=
DECODE_TOK ?= 128
WARMUP ?= 2
KV_DTYPE ?= bf16
ROUTER_TEMP ?= 0.5

# Non-empty CHECKPOINT → " --checkpoint path" (leading space; empty when unset)
CKPT_ARG = $(if $(strip $(CHECKPOINT)), --checkpoint $(CHECKPOINT),)
# Non-empty SEQ_LEN → " --seq-len N"
SEQ_ARG = $(if $(strip $(SEQ_LEN)), --seq-len $(SEQ_LEN),)

# Extra benchmark args, e.g. BENCH_EXTRA='--prompt "Hello" --decode-tokens 64' or --no-show-io
BENCH_EXTRA ?=
# 1 / true / yes → pass --lookahead-router to benchmark_mlx.py (do NOT run `make target --lookahead-router`; make eats --flags)
LOOKAHEAD_ROUTER ?= 0
LOOKAHEAD_ARG = $(if $(filter 1 true TRUE yes YES,$(LOOKAHEAD_ROUTER)),--lookahead-router,)

.PHONY: help mlx-bench mlx-bench-quick mlx-stream mlx-stream-spec mlx-stream-spec-debug mlx-stream-spec-sweep mlx-stream-spec-ab-quant mlx-spec-external mlx-spec-external-sweep mlx-profile mlx-export-npz mlx-force-pt deps-mlx frontend-dev frontend backend-dev backend up

help:
	@echo "Mamba3-XR Makefile"
	@echo ""
	@echo "  make mlx-bench          MLX prefill/decode benchmark (default tokenizer + seq)"
	@echo "  make mlx-bench-quick    Shorter run (SEQ_LEN=128, DECODE_TOK=32)"
	@echo "  make mlx-stream         Stream tokens to stdout (default includes full compile + 4-bit quant)"
	@echo "  make mlx-stream-spec    Stream with single-model speculative decode (draft/target split)"
	@echo "  make mlx-stream-spec-debug  Spec debug mode: no-eos-stop + show special tokens"
	@echo "  make mlx-stream-spec-sweep  Sweep speculative draft layers (6/8/10/12)"
	@echo "  make mlx-stream-spec-ab-quant  A/B speculative quality: quant=8 vs quant=0"
	@echo "  make mlx-export-npz     Load .pt, write .npz cache next to checkpoint (set CHECKPOINT=...)"
	@echo "  make mlx-force-pt       Same as mlx-bench but --force-pt"
	@echo "  make mlx-profile        Layer/host-GPU proxy profiler (see inference/tools/profile_mlx_infer.py)"
	@echo "  make backend-dev        Start FastAPI backend with --reload"
	@echo "  make backend            Start FastAPI backend (production mode)"
	@echo "  make frontend-dev       Start Next.js frontend with hot-reload"
	@echo "  make frontend           Start Next.js frontend (production mode)"
	@echo "  make up                Start backend-dev + frontend-dev together"
	@echo "  make deps-mlx           pip install mlx numpy transformers torch"
	@echo ""
	@echo "Chat server (mamba3_mlx):"
	@echo "  make dev                Load model → start server → open browser (Ctrl-C stops all)"
	@echo "  make dev-mock           Start mock server → open browser (no GPU)"
	@echo "  make serve              Start server only (real model, blocks)"
	@echo "  make serve-mock         Start server only (mock mode)"
	@echo "  make open               Open browser to running server"
	@echo "  make mlx-chat           PROMPT=\"...\"  single-turn CLI chat"
	@echo "  make mlx-chat-creative  PROMPT=\"...\"  creative preset (temp=1.1)"
	@echo "  make mlx-chat-precise   PROMPT=\"...\"  precise preset  (temp=0.3)"
	@echo "  make mlx-benchmark      Throughput benchmark"
	@echo "  make mlx-test           Run test suite"
	@echo "Variables: MLX_CHECKPOINT, MLX_TOKENIZER, SERVER_HOST, SERVER_PORT, SERVER_DTYPE, PROMPT"
	@echo ""
	@echo "Variables: CHECKPOINT, SEQ_LEN(optional), DECODE_TOK, MAX_NEW_TOK, WARMUP, DTYPE, KV_DTYPE, ROUTER_TEMP, INFER_TYPE, BENCH_EXTRA, LOOKAHEAD_ROUTER, STREAM_EXTRA, STREAM_QUANT, SPEC_DRAFT_LAYERS, SPEC_MAX_DRAFT, SPEC_NO_EOS_STOP, SPEC_EXTRA, SPEC_SWEEP_LAYERS, SPEC_DEBUG_PROMPT, BACKEND_HOST, BACKEND_PORT, BACKEND_EXTRA, FRONTEND_PORT, FRONTEND_EXTRA, FRONTEND_API_BASE, FRONTEND_WS_BASE, PYTHON"
	@echo "Example:   make mlx-bench CHECKPOINT=checkpoint.pt SEQ_LEN=1024"
	@echo "Example:   make mlx-bench BENCH_EXTRA='--prompt \"Hi\" --decode-tokens 32'"
	@echo "Benchmark flags: BENCH_EXTRA='--foo' or LOOKAHEAD_ROUTER=1. Note: trailing --foo after the target is parsed by GNU make, not benchmark_mlx.py."
	@echo "Example:   make mlx-bench-quick LOOKAHEAD_ROUTER=1"
	@echo "Example:   make mlx-bench-quick BENCH_EXTRA='--lookahead-router --tucker-einsum-fuse'"
	@echo "Example:   make mlx-stream LOOKAHEAD_ROUTER=1   # disables default outer full-decode-compile for decode"

# Core benchmark (auto model.npz / checkpoint.pt when CHECKPOINT is empty)
mlx-bench:
	$(PYTHON) $(BENCH)$(CKPT_ARG) --tokenizer $(TOK) --inference-type $(INFER_TYPE) --dtype $(DTYPE)$(SEQ_ARG) --decode-tokens $(DECODE_TOK) --warmup $(WARMUP) --kv-dtype $(KV_DTYPE) --router-temp $(ROUTER_TEMP) $(LOOKAHEAD_ARG) $(BENCH_EXTRA)

mlx-bench-quick:
	$(MAKE) mlx-bench SEQ_LEN=128 DECODE_TOK=512 WARMUP=2

# Streaming generation (same checkpoint/tokenizer vars as mlx-bench; MAX_NEW_TOK replaces decode length)
MAX_NEW_TOK ?= 512
# Default stream mode: full decode compile + continue on EOS + 4-bit quant for speed.
STREAM_QUANT ?= 8
STREAM_QUANT_ARG = $(if $(strip $(STREAM_QUANT)), --quantize $(STREAM_QUANT),)
STREAM_EXTRA ?= --full-decode-compile
SPEC_DRAFT_LAYERS ?= 8
SPEC_MAX_DRAFT ?= 4
SPEC_NO_EOS_STOP ?= 0
SPEC_EOS_ARG = $(if $(filter 1 true TRUE yes YES,$(SPEC_NO_EOS_STOP)), --no-eos-stop,)
# Speed flags for spec mode: greedy, Metal fusion (same as run_fast_stream.sh baseline)
SPEC_SPEED_FLAGS = --fast-sample --no-penalties --fused-mamba-mixer --tucker-einsum-fuse --tucker-scalar-fuse
SPEC_EXTRA ?=
SPEC_SWEEP_LAYERS ?= 5 8 10 15
SPEC_DEBUG_PROMPT ?= Hello! Write one short sentence about MLX on Apple Silicon.

mlx-stream:
	$(PYTHON) $(STREAM)$(CKPT_ARG) --tokenizer $(TOK) --inference-type $(INFER_TYPE) --dtype $(DTYPE)$(SEQ_ARG) --max-new-tokens $(MAX_NEW_TOK) --warmup $(WARMUP) --kv-dtype $(KV_DTYPE) --router-temp $(ROUTER_TEMP)$(STREAM_QUANT_ARG) $(LOOKAHEAD_ARG) $(STREAM_EXTRA) --no-eos-stop

mlx-stream-spec:
	$(PYTHON) $(STREAM)$(CKPT_ARG) --tokenizer $(TOK) --inference-type $(INFER_TYPE) --dtype $(DTYPE)$(SEQ_ARG) --max-new-tokens $(MAX_NEW_TOK) --warmup $(WARMUP) --kv-dtype $(KV_DTYPE) --router-temp $(ROUTER_TEMP)$(STREAM_QUANT_ARG) $(LOOKAHEAD_ARG) $(SPEC_SPEED_FLAGS) --speculative --spec-draft-layers $(SPEC_DRAFT_LAYERS) --spec-max-draft $(SPEC_MAX_DRAFT) $(SPEC_EXTRA)$(SPEC_EOS_ARG)

mlx-stream-spec-debug:
	$(MAKE) mlx-stream-spec SPEC_NO_EOS_STOP=1 SPEC_EXTRA='--prompt "$(SPEC_DEBUG_PROMPT)"'

mlx-stream-spec-sweep:
	@for L in $(SPEC_SWEEP_LAYERS); do \
		echo ""; \
		echo "===== speculative draft_layers=$$L ====="; \
		$(PYTHON) $(STREAM)$(CKPT_ARG) --tokenizer $(TOK) --inference-type $(INFER_TYPE) --dtype $(DTYPE)$(SEQ_ARG) --max-new-tokens $(MAX_NEW_TOK) --warmup $(WARMUP) --kv-dtype $(KV_DTYPE) --router-temp $(ROUTER_TEMP)$(STREAM_QUANT_ARG) $(LOOKAHEAD_ARG) $(SPEC_SPEED_FLAGS) --speculative --spec-draft-layers $$L --spec-max-draft $(SPEC_MAX_DRAFT) $(SPEC_EXTRA)$(SPEC_EOS_ARG); \
	done

mlx-stream-spec-ab-quant:
	@echo ""
	@echo "===== speculative quant=8 ====="
	@$(MAKE) mlx-stream-spec STREAM_QUANT=8
	@echo ""
	@echo "===== speculative quant=0 ====="
	@$(MAKE) mlx-stream-spec STREAM_QUANT=0

# External draft model speculative decode benchmark (13M darf-model vs 417M main)
SPEC_EXT_K ?= 1 2 4 6 8
SPEC_EXT_DRAFT ?= checkpoints/darf-model
SPEC_EXT_N ?= 200
SPEC_EXT_OUT ?= inference/results/spec_external_results.json
SPEC_EXT_EXTRA ?=

mlx-spec-external:
	$(PYTHON) inference/benchmark_spec_external.py \
		--checkpoint $(CHECKPOINT) \
		--draft-checkpoint $(SPEC_EXT_DRAFT) \
		--tokenizer $(TOK) \
		--n-tokens $(SPEC_EXT_N) \
		--k-values $(SPEC_EXT_K) \
		--warmup $(WARMUP) \
		--router-temp $(ROUTER_TEMP) \
		--output-json $(SPEC_EXT_OUT) \
		$(SPEC_EXT_EXTRA)

mlx-spec-external-sweep:
	@mkdir -p inference/results
	@$(MAKE) mlx-spec-external SPEC_EXT_K="1 2 4 6 8" SPEC_EXT_N=200

# Bottleneck report: wall vs thread CPU, MLX peak memory (does not modify inference/lib/mlx_hybrid_infer.py)
mlx-profile:
	$(PYTHON) $(PROF)$(CKPT_ARG) --tokenizer $(TOK) --dtype $(DTYPE) --kv-dtype $(KV_DTYPE)$(SEQ_ARG) --profile-decode-steps $(PROFILE_DECODE_STEPS) $(BENCH_EXTRA)

mlx-force-pt:
	$(PYTHON) $(BENCH)$(CKPT_ARG) --tokenizer $(TOK) --inference-type $(INFER_TYPE) --dtype $(DTYPE) --force-pt$(SEQ_ARG) --decode-tokens $(DECODE_TOK) --warmup $(WARMUP) --kv-dtype $(KV_DTYPE) --router-temp $(ROUTER_TEMP) $(LOOKAHEAD_ARG) $(BENCH_EXTRA)

# After success, next `make mlx-bench` can load the .npz without torch
mlx-export-npz:
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=path/to/model.pt" && exit 1)
	$(PYTHON) $(BENCH) \
		--checkpoint $(CHECKPOINT) \
		--tokenizer $(TOK) \
		--force-pt \
		$(SEQ_ARG) \
		--decode-tokens $(DECODE_TOK) \
		--save-npz

deps-mlx:
	$(PYTHON) -m pip install -U mlx numpy transformers torch

# Backend (FastAPI)
BACKEND_DIR := $(ROOT)/inference/backend
BACKEND_HOST ?= 0.0.0.0
BACKEND_PORT ?= 8000
BACKEND_EXTRA ?=
BACKEND_NO_EOS_STOP ?= 0
FRONTEND_DIR := $(ROOT)/inference/frontend
FRONTEND_PORT ?= 3000
FRONTEND_EXTRA ?=
FRONTEND_API_BASE ?= http://localhost:$(BACKEND_PORT)
FRONTEND_WS_BASE ?= ws://localhost:$(BACKEND_PORT)

backend-dev:
	@if [ ! -d "$(BACKEND_DIR)" ]; then echo "Backend dir not found: $(BACKEND_DIR)"; exit 1; fi
	@if [ ! -f "$(BACKEND_DIR)/.env" ] && [ -f "$(BACKEND_DIR)/.env.example" ]; then cp "$(BACKEND_DIR)/.env.example" "$(BACKEND_DIR)/.env"; fi
	cd "$(BACKEND_DIR)" && INFERENCE_NO_EOS_STOP="$(BACKEND_NO_EOS_STOP)" $(PYTHON) -m uvicorn app.main:app --host $(BACKEND_HOST) --port $(BACKEND_PORT) --reload $(BACKEND_EXTRA)

backend:
	@if [ ! -d "$(BACKEND_DIR)" ]; then echo "Backend dir not found: $(BACKEND_DIR)"; exit 1; fi
	@if [ ! -f "$(BACKEND_DIR)/.env" ] && [ -f "$(BACKEND_DIR)/.env.example" ]; then cp "$(BACKEND_DIR)/.env.example" "$(BACKEND_DIR)/.env"; fi
	cd "$(BACKEND_DIR)" && INFERENCE_NO_EOS_STOP="$(BACKEND_NO_EOS_STOP)" $(PYTHON) -m uvicorn app.main:app --host $(BACKEND_HOST) --port $(BACKEND_PORT) $(BACKEND_EXTRA)

frontend-dev:
	@if [ ! -d "$(FRONTEND_DIR)" ]; then echo "Frontend dir not found: $(FRONTEND_DIR)"; exit 1; fi
	cd "$(FRONTEND_DIR)" && NEXT_PUBLIC_API_BASE="$(FRONTEND_API_BASE)" NEXT_PUBLIC_WS_BASE="$(FRONTEND_WS_BASE)" npm run dev -- --port $(FRONTEND_PORT) $(FRONTEND_EXTRA)

frontend:
	@if [ ! -d "$(FRONTEND_DIR)" ]; then echo "Frontend dir not found: $(FRONTEND_DIR)"; exit 1; fi
	cd "$(FRONTEND_DIR)" && NEXT_PUBLIC_API_BASE="$(FRONTEND_API_BASE)" NEXT_PUBLIC_WS_BASE="$(FRONTEND_WS_BASE)" npm run start -- --port $(FRONTEND_PORT) $(FRONTEND_EXTRA)

up:
	@if [ ! -d "$(BACKEND_DIR)" ]; then echo "Backend dir not found: $(BACKEND_DIR)"; exit 1; fi
	@if [ ! -d "$(FRONTEND_DIR)" ]; then echo "Frontend dir not found: $(FRONTEND_DIR)"; exit 1; fi
	@if [ ! -f "$(BACKEND_DIR)/.env" ] && [ -f "$(BACKEND_DIR)/.env.example" ]; then cp "$(BACKEND_DIR)/.env.example" "$(BACKEND_DIR)/.env"; fi
	@bash -lc 'set -e; trap "kill 0" INT TERM EXIT; \
		cd "$(BACKEND_DIR)" && INFERENCE_NO_EOS_STOP="$(BACKEND_NO_EOS_STOP)" "$(PYTHON)" -m uvicorn app.main:app --host "$(BACKEND_HOST)" --port "$(BACKEND_PORT)" --reload $(BACKEND_EXTRA) & \
		cd "$(FRONTEND_DIR)" && NEXT_PUBLIC_API_BASE="$(FRONTEND_API_BASE)" NEXT_PUBLIC_WS_BASE="$(FRONTEND_WS_BASE)" npm run dev -- --port "$(FRONTEND_PORT)" $(FRONTEND_EXTRA)'

# ─── mamba3_mlx — chat server & tests ────────────────────────────────────────
MLX_SCRIPTS    := $(ROOT)/mamba3_mlx/scripts
MLX_CHECKPOINT ?= $(ROOT)/checkpoints/latest_sft_cot_model.npz
MLX_TOKENIZER  ?= $(ROOT)/checkpoints/tokenizer
SERVER_HOST    ?= 0.0.0.0
SERVER_PORT    ?= 8000
SERVER_DTYPE   ?= bf16
# Browser connect host: 0.0.0.0 is not a valid browser target → use localhost
CONNECT_HOST    = $(if $(filter 0.0.0.0,$(SERVER_HOST)),localhost,$(SERVER_HOST))
PROMPT         ?=

.PHONY: serve serve-mock dev dev-mock open \
        mlx-chat mlx-chat-creative mlx-chat-precise \
        mlx-benchmark mlx-test

## serve: load model then start server (blocks, Ctrl-C to stop)
serve:
	$(MLX_SCRIPTS)/serve.sh \
	  --checkpoint $(MLX_CHECKPOINT) \
	  --tokenizer  $(MLX_TOKENIZER) \
	  --host       $(SERVER_HOST) \
	  --port       $(SERVER_PORT) \
	  --dtype      $(SERVER_DTYPE)

## serve-mock: start server in mock mode (no GPU, instant)
serve-mock:
	MOCK=1 $(MLX_SCRIPTS)/serve.sh \
	  --host $(SERVER_HOST) \
	  --port $(SERVER_PORT)

## open: open the chat UI in the default browser (server must already be running)
open:
	@open "http://$(CONNECT_HOST):$(SERVER_PORT)" 2>/dev/null || \
	 xdg-open "http://$(CONNECT_HOST):$(SERVER_PORT)" 2>/dev/null || \
	 echo "Visit: http://$(CONNECT_HOST):$(SERVER_PORT)"

## dev: load model → start server → auto-open browser once ready (Ctrl-C stops all)
dev:
	@bash -c '\
	  trap "kill 0" INT TERM EXIT; \
	  $(MLX_SCRIPTS)/serve.sh \
	    --checkpoint "$(MLX_CHECKPOINT)" \
	    --tokenizer  "$(MLX_TOKENIZER)" \
	    --host       "$(SERVER_HOST)" \
	    --port       "$(SERVER_PORT)" \
	    --dtype      "$(SERVER_DTYPE)" & \
	  echo "Waiting for model to load on http://$(CONNECT_HOST):$(SERVER_PORT) …"; \
	  until curl -sf "http://$(CONNECT_HOST):$(SERVER_PORT)/api/status" >/dev/null 2>&1; \
	    do sleep 1; done; \
	  echo "Server ready — opening browser"; \
	  open "http://$(CONNECT_HOST):$(SERVER_PORT)" 2>/dev/null || \
	    xdg-open "http://$(CONNECT_HOST):$(SERVER_PORT)" 2>/dev/null || \
	    echo "  Visit: http://$(CONNECT_HOST):$(SERVER_PORT)"; \
	  wait'

## dev-mock: start mock server → auto-open browser (no GPU, fast boot)
dev-mock:
	@bash -c '\
	  trap "kill 0" INT TERM EXIT; \
	  MOCK=1 $(MLX_SCRIPTS)/serve.sh \
	    --host "$(SERVER_HOST)" \
	    --port "$(SERVER_PORT)" & \
	  echo "Waiting for mock server on http://$(CONNECT_HOST):$(SERVER_PORT) …"; \
	  until curl -sf "http://$(CONNECT_HOST):$(SERVER_PORT)/api/status" >/dev/null 2>&1; \
	    do sleep 1; done; \
	  echo "Server ready — opening browser"; \
	  open "http://$(CONNECT_HOST):$(SERVER_PORT)" 2>/dev/null || \
	    xdg-open "http://$(CONNECT_HOST):$(SERVER_PORT)" 2>/dev/null || \
	    echo "  Visit: http://$(CONNECT_HOST):$(SERVER_PORT)"; \
	  wait'

## mlx-chat: single-turn CLI chat with default sampling   (PROMPT="...")
mlx-chat:
	@test -n "$(PROMPT)" || (echo 'Usage: make mlx-chat PROMPT="your question"' && exit 1)
	$(MLX_SCRIPTS)/chat.sh "$(PROMPT)"

## mlx-chat-creative: single-turn CLI chat, high-temperature creative preset
mlx-chat-creative:
	@test -n "$(PROMPT)" || (echo 'Usage: make mlx-chat-creative PROMPT="your question"' && exit 1)
	$(MLX_SCRIPTS)/chat_creative.sh "$(PROMPT)"

## mlx-chat-precise: single-turn CLI chat, low-temperature factual preset
mlx-chat-precise:
	@test -n "$(PROMPT)" || (echo 'Usage: make mlx-chat-precise PROMPT="your question"' && exit 1)
	$(MLX_SCRIPTS)/chat_precise.sh "$(PROMPT)"

## mlx-benchmark: throughput benchmark (prefill + decode)
mlx-benchmark:
	$(MLX_SCRIPTS)/benchmark.sh

## mlx-test: run mamba3_mlx test suite
mlx-test:
	$(MLX_SCRIPTS)/run_tests.sh