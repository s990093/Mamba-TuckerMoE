#!/bin/bash
# Usage: bash scripts/tools/progress.sh
# Displays training progress with a progress bar.
set -e
BUNDLE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOGDIR="$BUNDLE_ROOT/output/logs"
LOG=$(ls -t "$LOGDIR"/train_sft_cot_*.log 2>/dev/null | head -1)

if [ -z "$LOG" ]; then
  echo "No training log found in $LOGDIR"
  exit 1
fi

LINE=$(tail -5 "$LOG" | grep "step" | tail -1)

STEP=$(echo "$LINE" | grep -oP '累計 step \K\d+')
TOTAL=$(echo "$LINE" | grep -oP '/\K\d+' | head -1)
EP=$(echo "$LINE" | grep -oP '本輪 ep \K\d+')
TOTEP=$(echo "$LINE" | grep -oP '本輪 ep \d+/\K\d+')
PPL=$(echo "$LINE" | grep -oP 'ppl \K[\d.]+')
CE=$(echo "$LINE" | grep -oP ' CE \K[\d.]+')
LR=$(echo "$LINE" | grep -oP 'lr \K[\d.e\-]+')
SPD_RAW=$(echo "$LINE" | grep -oP '\d+\.\d+(?=s/step)' | head -1)
SPD="${SPD_RAW:-30}s/step"

PCT=$(python3 -c "print(f'{$STEP / $TOTAL * 100:.1f}')")
BAR_W=40
FILLED=$(python3 -c "print(int($BAR_W * $STEP / $TOTAL))")
BAR=$(python3 -c "print('█' * $FILLED + '░' * ($BAR_W - $FILLED))")
STEP_S=$(echo "$SPD" | grep -oP '[\d.]+')
STEP_S=${STEP_S:-30}
ETA_S=$(python3 -c "print(int(($TOTAL - $STEP) * $STEP_S))")
ETA_M=$(python3 -c "print(int($ETA_S / 60))")
ETA_H=$(python3 -c "print(f'{$ETA_M / 60:.1f}')")

echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  SFT-CoT Overfit Training                           ║"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  [%s] %5s%%  (%s/%s)             ║\n" "$BAR" "$PCT" "$STEP" "$TOTAL"
printf "  ║  Epoch: %s/%-3s   PPL: %-6s  CE: %-7s        ║\n" "$EP" "$TOTEP" "$PPL" "$CE"
printf "  ║  LR: %-10s  %-12s                    ║\n" "$LR" "$SPD"
echo "  ╠══════════════════════════════════════════════════════╣"
printf "  ║  ETA: ~%s min (~%s h)                         ║\n" "$ETA_M" "$ETA_H"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""
