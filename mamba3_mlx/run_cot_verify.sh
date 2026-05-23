#!/usr/bin/env bash
# run_cot_verify.sh — multi-seed sanity sweep over the CoT middleware.
# Runs run_cot.sh N times with different --seed values and grades each
# trial on (a) splitter reached `done`, (b) <final> was injected,
# (c) the budget watchdog never fired (budget disabled by default).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${LOG_DIR:-$HERE/.run_cot_verify_logs}"
mkdir -p "$LOG_DIR"

N="${N:-5}"
PROMPT="${PROMPT:-What is 2 + 2? Answer in one sentence.}"
MODE="${MODE:-self}"
MAX_TOKENS="${MAX_TOKENS:-256}"
# No reasoning-budget watchdog — model decides when to close <think>.
export MODE PROMPT MAX_TOKENS

echo "=== run_cot_verify.sh (N=$N) ==="
echo "REASONING_BUDGET = (disabled — model self-paces <think>)"
echo "PROMPT           = $PROMPT"
echo "MODE             = $MODE"
echo "LOG_DIR          = $LOG_DIR"
echo "----------------------------------"

pass=0
fail=0
for i in $(seq 1 "$N"); do
  log="$LOG_DIR/trial_${i}.log"
  echo
  echo "── trial $i / $N (seed=$i) ─────────────────────────────────"
  bash "$HERE/run_cot.sh" --seed "$i" > "$log" 2>&1 || true

  # Verdict from the health line emitted by render_health_line().
  health_line="$(grep -E '^\[mw\]   mode=' "$log" | tail -1 || true)"
  mode_done="$(echo "$health_line"   | grep -c 'mode=done'        || true)"
  final_inj="$(echo "$health_line"   | grep -c 'final_injected'   || true)"
  budget_hit="$(echo "$health_line"  | grep -c 'budget_hit'       || true)"

  stop_line="$(grep -oE 'stop=[a-z_]+' "$log" | tail -1 || true)"

  status="PASS"
  if [ "${mode_done:-0}" -lt 1 ] || [ "${final_inj:-0}" -lt 1 ] || [ "${budget_hit:-0}" -ge 1 ]; then
    status="FAIL"
  fi

  if [ "$status" = "PASS" ]; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
  fi

  printf "  status=%s  %s  %s\n" "$status" "$stop_line" "$health_line"
done

echo
echo "=================================="
echo " summary: PASS=$pass / $N   FAIL=$fail / $N"
echo " logs:    $LOG_DIR/trial_<i>.log"
echo "=================================="

exit $fail
