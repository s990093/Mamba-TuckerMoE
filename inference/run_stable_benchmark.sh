#!/usr/bin/env bash
# 勿用 POSIX `sh` 跑（請用 bash 或 ./）。
if [ -z "${BASH_VERSION-}" ] || [ -n "${POSIXLY_CORRECT-}" ]; then
  exec /usr/bin/env bash --noprofile --norc "$0" "$@"
fi

# 穩定優先 decode 基準。b benchmark_mlx 預設物化 caches。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-checkpoints/checkpoint_sft_s27510_model_only.pt}"

w=76
# shellcheck disable=SC2046
bar="$(printf '─%.0s' $(seq 1 "$w"))"
printf '\n╭%s╮\n' "$bar"
printf '│ %-*s │\n' "$((w - 2))" "Mamba3-XR · run_stable_benchmark"
printf '├%s┤\n' "$bar"
printf '│ %-*s │\n' "$((w - 2))" "Checkpoint : ${CHECKPOINT}"
printf '│ %-*s │\n' "$((w - 2))" "Script      : inference/benchmark_mlx.py (safe bf16 KV-auto)"
printf '│ %-*s │\n' "$((w - 2))" "Default     : 128 decode tokens · quantize off · no tucker fuse"
printf '╰%s╯\n\n' "$bar"

exec .venv/bin/python inference/benchmark_mlx.py \
  --checkpoint "$CHECKPOINT" \
  --inference-type safe \
  --dtype bf16 \
  --kv-dtype auto \
  --quantize 0 \
  --decode-tokens 128 \
  "$@"
