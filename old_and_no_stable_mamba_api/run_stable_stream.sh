#!/usr/bin/env bash
# 勿用 `sh` 執行（POSIX sh 無陣列；且 bash 以 sh 名稱啟動會開 POSIXLY_CORRECT，易觸發 set -u 與 fwd[@] 錯誤）。
# 請：chmod +x 後 ./inference/run_stable_stream.sh …  或  bash inference/run_stable_stream.sh …
if [ -z "${BASH_VERSION-}" ] || [ -n "${POSIXLY_CORRECT-}" ]; then
  exec /usr/bin/env bash --noprofile --norc "$0" "$@"
fi

# 穩定優先（預設）：
#   STREAM_PRESET=stable    → 安全品質向（materialize caches / stochastic）
#   STREAM_PRESET=best      → 速度向（einsum fuse；不預設 AMX）
#   STREAM_PRESET=best_amx  → 速度實驗向（在 best 基礎上加 --tucker-amx-fuse）
#
# 單次：  ./inference/run_stable_stream.sh --prompt "你的問題"   # 預設遇到 tokenizer EOS 即停止（--stop-on-eos）
# 交互：  ./inference/run_stable_stream.sh --interactive       # 可加 -i；會附帶 --plain-output（與 REPL / input() 相容）；同樣預設 EOS 停止
# 預設權重：repo 根目錄的 checkpoint_sft_s27510_model_only.pt（若已有同名 .npz 會自動優先載入）。
# 覆寫：CHECKPOINT=/path/other.pt ./inference/run_stable_stream.sh … 或在參數最後加上 --checkpoint /path/other
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-checkpoints/checkpoint_sft_s27510_model_only.pt}"
STREAM_PRESET="${STREAM_PRESET:-stable}"
if [[ "$STREAM_PRESET" != "stable" && "$STREAM_PRESET" != "best" && "$STREAM_PRESET" != "best_amx" ]]; then
  echo "Invalid STREAM_PRESET=$STREAM_PRESET (expected: stable, best, or best_amx)" >&2
  exit 2
fi

print_launch_banner() {
  local w="${1:-72}" i
  # shellcheck disable=SC2046
  i="$(printf '─%.0s' $(seq 1 "$w"))"
  printf '\n╭%s╮\n' "$i"
  printf '│ %-*s │\n' "$((w - 2))" "Mamba3-XR · run_stable_stream (${STREAM_PRESET} preset)"
  printf '├%s┤\n' "$i"
  printf '│ %-*s │\n' "$((w - 2))" "Checkpoint  : ${CHECKPOINT}"
  printf '│ %-*s │\n' "$((w - 2))" "Repo root   : ${ROOT}"
  printf '│ %-*s │\n' "$((w - 2))" "Python      : inference/stream_mlx.py"
  if [[ "$STREAM_PRESET" == "stable" ]]; then
    printf '│ %-*s │\n' "$((w - 2))" "Core flags  : safe │ bf16 │ kv bf16 │ --materialize-caches │ stop-on-eos"
    printf '│ %-*s │\n' "$((w - 2))" "             quantize=8 · no Tucker einsum fuse"
    printf '│ %-*s │\n' "$((w - 2))" "Sampling    : stochastic (temp / top-k / top-p / min-p from script)"
  elif [[ "$STREAM_PRESET" == "best" ]]; then
    printf '│ %-*s │\n' "$((w - 2))" "Core flags  : throughput │ bf16 │ kv auto │ stop-on-eos │ full-decode-compile"
    printf '│ %-*s │\n' "$((w - 2))" "             quantize=4 · Tucker einsum fuse on · no cache materialize"
    printf '│ %-*s │\n' "$((w - 2))" "Sampling    : fast-sample (greedy) for max tok/s"
  else
    printf '│ %-*s │\n' "$((w - 2))" "Core flags  : throughput │ bf16 │ kv auto │ stop-on-eos │ full-decode-compile"
    printf '│ %-*s │\n' "$((w - 2))" "             quantize=4 · Tucker einsum+AMX fuse on · no cache materialize"
    printf '│ %-*s │\n' "$((w - 2))" "Sampling    : fast-sample (greedy) for max tok/s"
  fi
  printf '│ %-*s │\n' "$((w - 2))" "Override    : CHECKPOINT=path.pt │ args after script → python"
  printf '╰%s╯\n\n' "$i"
}

print_launch_banner 76

PY=(.venv/bin/python inference/stream_mlx.py)
if [[ "$STREAM_PRESET" == "best" || "$STREAM_PRESET" == "best_amx" ]]; then
  COMMON=(
    --checkpoint "$CHECKPOINT"
    --inference-type throughput
    --dtype bf16
    --kv-dtype auto
    --quantize 4
    --tucker-einsum-fuse
    --fast-sample
    --stop-on-eos
    --full-decode-compile
    --router-temp 0.5
  )
  if [[ "$STREAM_PRESET" == "best_amx" ]]; then
    COMMON+=(--tucker-amx-fuse)
  fi
else
  COMMON=(
    --checkpoint "$CHECKPOINT"
    --inference-type safe
    --materialize-caches
    --dtype bf16
    --kv-dtype bf16
    --quantize 8
    --no-tucker-einsum-fuse
    --no-fast-sample
    --stop-on-eos
    --full-decode-compile
    --temp 0.3
    --top_p 0.9
    --top_k 40
    --min_p 0.05
  )
fi

interactive_mode=false
extras=()
have_extra=false

for arg in "$@"; do
  case "$arg" in
    -i | --interactive) interactive_mode=true ;;
    *)
      extras+=("$arg")
      have_extra=true
      ;;
  esac
done

# With `set -u`, some Bash-as-sh setups treat empty "${array[@]}" on exec as unset — only expand extras when nonempty.
if [[ "$interactive_mode" == true ]]; then
  if [[ "$have_extra" == true ]]; then
    exec "${PY[@]}" "${COMMON[@]}" --interactive --plain-output "${extras[@]}"
  else
    exec "${PY[@]}" "${COMMON[@]}" --interactive --plain-output
  fi
else
  if [[ "$have_extra" == true ]]; then
    exec "${PY[@]}" "${COMMON[@]}" "${extras[@]}"
  else
    exec "${PY[@]}" "${COMMON[@]}"
  fi
fi
