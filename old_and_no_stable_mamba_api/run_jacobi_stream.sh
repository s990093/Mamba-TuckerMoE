#!/usr/bin/env bash
# Jacobi +ngram 最佳配置串流腳本
#
# 鎖定最佳設定: +ngram K=32, 2.32-2.36× universal speedup
# 所有超參數均可透過環境變數覆蓋
#
# ── 基本用法 ────────────────────────────────────────────────────────────────
#   ./inference/run_jacobi_stream.sh                        # 互動 REPL（預設）
#   ./inference/run_jacobi_stream.sh -p "你的問題"         # 單一 prompt
#   ./inference/run_jacobi_stream.sh -i                   # 互動 REPL（同上）
#   ./inference/run_jacobi_stream.sh -p "..." --no-progress  # 純文字輸出
#   NO_EOS_STOP=1 ./inference/run_jacobi_stream.sh -p "..."  # 不因 EOS/im_end 提早停止
#
# ── K 值實驗 ────────────────────────────────────────────────────────────────
#   K=48  ./inference/run_jacobi_stream.sh -p "..."   # 測試 K=48 ARL 是否 >32
#   K=96  ./inference/run_jacobi_stream.sh -p "..."   # 測試 K=96
#
# ── 品質模式 ────────────────────────────────────────────────────────────────
#   TEMP=0.3 QUANTIZE=8 ./inference/run_jacobi_stream.sh -p "..."
#
# ── 環境變數 ────────────────────────────────────────────────────────────────
#   CHECKPOINT  — 模型權重路徑  (預設: checkpoints/latest_sft_cot_model.npz)
#   K           — Jacobi 視窗大小 (預設: 32, 建議測試: 48, 64, 96)
#   NGRAM_N     — N-gram 最大階數 (預設: 4)
#   TEMP        — 採樣溫度, 0=greedy (預設: 0)
#   TOP_K       — top-k 採樣 (預設: 40)
#   TOP_P       — nucleus sampling (預設: 0.9)
#   REP_PEN     — repetition penalty (預設: 1.0)
#   QUANTIZE    — 量化位元 0/4/8 (預設: 4)
#   MAX_TOKENS  — 最大生成 token 數 (預設: 512)
#   WARMUP      — 編譯預熱次數 (預設: 2)
#   NO_EOS_STOP — 1/true 時傳 --no-eos-stop（預設 REPL 會在 im_end 停止）
#   STOP_ON_EOS — 1/true 強制 --stop-on-eos（單次 prompt 預設不開，REPL 預設開）
if [ -z "${BASH_VERSION-}" ] || [ -n "${POSIXLY_CORRECT-}" ]; then
  exec /usr/bin/env bash --noprofile --norc "$0" "$@"
fi
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── 預設值 ──────────────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-checkpoints/latest_sft_cot_model.npz}"
K="${K:-32}"
NGRAM_N="${NGRAM_N:-4}"
TEMP="${TEMP:-0.7}"
TOP_K="${TOP_K:-40}"
TOP_P="${TOP_P:-0.9}"
REP_PEN="${REP_PEN:-1.3}"
QUANTIZE="${QUANTIZE:-8}"
MAX_TOKENS="${MAX_TOKENS:-512}"
WARMUP="${WARMUP:-2}"
NO_EOS_STOP="${NO_EOS_STOP:-0}"
STOP_ON_EOS="${STOP_ON_EOS:-}"

# ── Banner ───────────────────────────────────────────────────────────────────
print_banner() {
  local eos_note="eos-stop=on (REPL)"
  if [[ "$(echo "${NO_EOS_STOP}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes)$ ]]; then
    eos_note="eos-stop=off"
  elif [[ -n "$STOP_ON_EOS" && "$(echo "${STOP_ON_EOS}" | tr '[:upper:]' '[:lower:]')" =~ ^(0|false|no)$ ]]; then
    eos_note="eos-stop=off"
  fi
  local mode="K=${K}  ngram-n=${NGRAM_N}  quant=${QUANTIZE}-bit  temp=${TEMP}  ${eos_note}"
  echo ""
  echo "╔═══════════════════════════════════════════════════════════════╗"
  echo "║          Mamba3-XR · Jacobi +ngram Streaming                 ║"
  echo "╠═══════════════════════════════════════════════════════════════╣"
  printf "║  Config    : %-48s ║\n" "$mode"
  printf "║  Checkpoint: %-48s ║\n" "$(basename "$CHECKPOINT")"
  printf "║  Expected  : %-48s ║\n" "2.32-2.36× (K=32)  or  ~2.56× (K=48)"
  echo "╚═══════════════════════════════════════════════════════════════╝"
  echo ""
}

# ── 解析額外參數 (pass-through to jacobi_stream.py) ───────────────────────
EXTRA=()
SILENT=0
HAS_PROMPT=0
HAS_INTERACTIVE=0
HAS_NO_EOS_STOP=0
HAS_STOP_ON_EOS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-banner|-q) SILENT=1; shift ;;
    --no-eos-stop)  HAS_NO_EOS_STOP=1; EXTRA+=("$1"); shift ;;
    --stop-on-eos)  HAS_STOP_ON_EOS=1; EXTRA+=("$1"); shift ;;
    -p|--prompt)
      HAS_PROMPT=1
      EXTRA+=("$1" "$2")
      shift 2
      ;;
    -i|--interactive)
      HAS_INTERACTIVE=1
      EXTRA+=("$1")
      shift
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

# 無 prompt / 無 -i 時預設進入 REPL（與 run_stable_stream 行為一致）
if [[ "$HAS_PROMPT" == "0" && "$HAS_INTERACTIVE" == "0" ]]; then
  EXTRA=(--interactive "${EXTRA[@]+"${EXTRA[@]}"}")
fi

[[ "$SILENT" == "0" ]] && print_banner

PY=(.venv/bin/python inference/jacobi_stream.py)

EOS_ARGS=()
if [[ "$HAS_NO_EOS_STOP" == "0" && "$HAS_STOP_ON_EOS" == "0" ]]; then
  if [[ "$(echo "${NO_EOS_STOP}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes)$ ]]; then
    EOS_ARGS=(--no-eos-stop)
  elif [[ -n "$STOP_ON_EOS" && "$(echo "${STOP_ON_EOS}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes)$ ]]; then
    EOS_ARGS=(--stop-on-eos)
  fi
fi
# REPL 預設由 jacobi_stream.py 在 --interactive 時開啟 stop-on-eos；單次 -p 不加旗標則不停止

ARGS=(
  --checkpoint     "$CHECKPOINT"
  --K              "$K"
  --ngram-n        "$NGRAM_N"
  --temp           "$TEMP"
  --top_k          "$TOP_K"
  --top_p          "$TOP_P"
  --rep_pen        "$REP_PEN"
  --quantize       "$QUANTIZE"
  --max-new-tokens "$MAX_TOKENS"
  --warmup         "$WARMUP"
)
((${#EOS_ARGS[@]})) && ARGS+=("${EOS_ARGS[@]}")
((${#EXTRA[@]})) && ARGS+=("${EXTRA[@]}")

exec "${PY[@]}" "${ARGS[@]}"
