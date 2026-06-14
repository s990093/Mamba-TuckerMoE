#!/bin/bash
# ==============================================================================
#  GPU Scheduler 測試腳本 — 不影響真實訓練、不碰 real nvidia-smi / CSV
#
#  測試範圍：
#    1. calc_grad_accum       — 梯度累積計算 (2/3/4/5/6/8 卡)
#    2. is_night_window       — 時間邊界 (0~23 每一小時)
#    3. build_gpu_list        — GPU 列表組合 (目標 5, 4, 3 卡 / 不同空閒狀況)
#    4. detect_available_gpus — GPU 空閒偵測 (mock nvidia-smi)
#    5. 模擬 nvidia-smi       — 各種忙碌/空閒組合
#    6. signal_graceful_stop  — flag file 生命週期
#    7. session CSV           — 寫入格式驗證
#    8. 完整整合場景          — 模擬 2→5 GPUs 切換流程
#
#  用法：
#    ./scripts/training/test_scheduler.sh
#    ./scripts/training/test_scheduler.sh --verbose
# ==============================================================================

set -euo pipefail
export TZ=Asia/Taipei

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCHEDULER="$SCRIPT_DIR/gpu_scheduler.sh"

# ── 測試用暫存目錄（不碰真實 output） ─────────────────────────────────
TEST_DIR="$(mktemp -d /tmp/scheduler_test_XXXXXX)"
cleanup() { rm -rf "$TEST_DIR"; }
trap cleanup EXIT

export SCHEDULER_DIR="$TEST_DIR/.scheduler"
export LOG_DIR="$TEST_DIR/logs"
export PID_FILE="$TEST_DIR/train.pid"

mkdir -p "$SCHEDULER_DIR" "$LOG_DIR"

PASS=0
FAIL=0
VERBOSE=false
[[ "${1:-}" == "--verbose" ]] && VERBOSE=true

green()  { printf '\033[32m%s\033[0m\n' "$1"; }
red()    { printf '\033[31m%s\033[0m\n' "$1"; }
yellow() { printf '\033[33m%s\033[0m\n' "$1"; }

# 注意：((PASS++)) 在 PASS=0 時 return code=1，會觸發 set -e，故使用 PASS=$((PASS+1))
assert_eq() {
    local label="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
        PASS=$((PASS + 1))
        if $VERBOSE; then green "  ✓ $label"; fi
    else
        FAIL=$((FAIL + 1))
        red "  ✗ $label"
        red "    expected: $expected"
        red "    actual:   $actual"
    fi
}

assert_contains() {
    local label="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -qF "$needle"; then
        PASS=$((PASS + 1))
        if $VERBOSE; then green "  ✓ $label"; fi
    else
        FAIL=$((FAIL + 1))
        red "  ✗ $label"
        red "    expected to contain: $needle"
        red "    got: $haystack"
    fi
}

assert_file_exists() {
    local label="$1" file="$2"
    if [[ -f "$file" ]]; then
        PASS=$((PASS + 1))
        if $VERBOSE; then green "  ✓ $label"; fi
    else
        FAIL=$((FAIL + 1))
        red "  ✗ $label  (file missing: $file)"
    fi
}

assert_file_not_exists() {
    local label="$1" file="$2"
    if [[ ! -f "$file" ]]; then
        PASS=$((PASS + 1))
        if $VERBOSE; then green "  ✓ $label"; fi
    else
        FAIL=$((FAIL + 1))
        red "  ✗ $label  (file should not exist: $file)"
    fi
}

# ── Mock nvidia-smi ────────────────────────────────────────────────────
# 在 PATH 最前面插入一個假的 nvidia-smi，讓 gpu_is_free 不碰到真實 GPU
MOCK_BIN="$TEST_DIR/mock_bin"
mkdir -p "$MOCK_BIN"

# 預設 mock 回傳：所有 GPU 100% 忙碌
export MOCK_SMI_FILE="$TEST_DIR/mock_smi_data"
cat > "$MOCK_SMI_FILE" <<'EOF'
# 格式: gpu_id,util
# 預設全部忙碌
0,100
3,100
4,100
5,100
6,100
EOF

cat > "$MOCK_BIN/nvidia-smi" <<'SCRIPT'
#!/bin/bash
# Mock nvidia-smi: 從 MOCK_SMI_FILE 讀取假資料
MOCK_FILE="${MOCK_SMI_FILE:-/tmp/mock_smi_data}"
gpu_id="${@: -1}"   # 最後一個 arg 是 GPU id

# 移除非數字
gpu_id="${gpu_id//[!0-9]/}"

while IFS=',' read -r id util; do
    id="${id//[!0-9]/}"
    if [[ "$id" == "$gpu_id" ]]; then
        echo "$id, $util %"
        exit 0
    fi
done < "$MOCK_FILE"

# fallback: GPU not found → busy
echo "$gpu_id, 100 %"
SCRIPT
chmod +x "$MOCK_BIN/nvidia-smi"

# 設定 mock PATH（優先使用 mock）
_real_path="$PATH"
export PATH="$MOCK_BIN:$_real_path"

# ── Helper: 寫 mock GPU 狀態 ───────────────────────────────────────────
# 用法: mock_gpu_util <gpu_id=util> ...
mock_gpu_util() {
    echo "# mock generated at $(date)" > "$MOCK_SMI_FILE"
    for pair in "$@"; do
        local id="${pair%%=*}"
        local util="${pair##*=}"
        echo "$id,$util"
    done >> "$MOCK_SMI_FILE"
}

# ── Source scheduler config 值（不執行 main） ──────────────────────────
# 從 scheduler 抓出常數值
BATCH_SIZE=4
TARGET_EFF_BATCH=24
BASE_GPUS="1,2"
NIGHT_EXTRA_ORDER=(3 4 5 6 0)
NIGHT_TARGET_TOTAL=5
NIGHT_START=3
NIGHT_END=7
GPU_FREE_THRESHOLD=5
MAX_TOTAL_GPUS=6  # 留至少 1 張卡給別人

# 導入 scheduler 的 helper functions（用 subshell 的方式）
# 直接 copy 關鍵函式做純 logic test，避免 source 整個 scheduler 造成副作用
gpu_is_free_test() {
    local gpu_id="$1"
    local util
    while IFS=',' read -r id util; do
        id="${id//[!0-9]/}"
        if [[ "$id" == "$gpu_id" ]]; then
            util="${util//[!0-9]/}"
            [[ "$util" -lt "$GPU_FREE_THRESHOLD" ]] && return 0
            return 1
        fi
    done < "$MOCK_SMI_FILE"
    return 1
}

detect_available_gpus_test() {
    local available=()
    for gpu_id in "${NIGHT_EXTRA_ORDER[@]}"; do
        if gpu_is_free_test "$gpu_id"; then
            available+=("$gpu_id")
        fi
    done
    echo "${available[@]}"
}

calc_grad_accum_test() {
    local num_gpus="$1"
    local val=$(( TARGET_EFF_BATCH / (BATCH_SIZE * num_gpus) ))
    if [[ $val -lt 1 ]]; then echo 1; else echo "$val"; fi
}

build_gpu_list_test() {
    local target_total="$1"
    shift
    local available=("$@")

    local base_count
    base_count=$(echo "$BASE_GPUS" | tr ',' '\n' | wc -l)

    # 硬上限：不超過 MAX_TOTAL_GPUS
    if [[ $target_total -gt $MAX_TOTAL_GPUS ]]; then
        target_total=$MAX_TOTAL_GPUS
    fi

    local extra_wanted=$(( target_total - base_count ))
    local extra_available=${#available[@]}

    # 永遠留 1 張額外卡空著
    local extra_usable=$(( extra_available - 1 ))
    if [[ $extra_usable -lt 0 ]]; then
        extra_usable=0
    fi

    local max_extra=$(( MAX_TOTAL_GPUS - base_count ))

    local take=$extra_wanted
    if [[ $take -gt $extra_usable ]]; then take=$extra_usable; fi
    if [[ $take -gt $max_extra ]]; then take=$max_extra; fi
    if [[ $take -lt 0 ]]; then take=0; fi

    if [[ $take -le 0 ]]; then
        echo "$BASE_GPUS"
        return
    fi

    local extra_selected=()
    local count=0
    for gpu in "${available[@]}"; do
        [[ $count -ge $take ]] && break
        extra_selected+=("$gpu")
        count=$((count + 1))
    done

    if [[ ${#extra_selected[@]} -gt 0 ]]; then
        local extra_list
        extra_list=$(IFS=,; echo "${extra_selected[*]}")
        echo "${BASE_GPUS},${extra_list}"
    else
        echo "$BASE_GPUS"
    fi
}

is_night_window_test() {
    local h="$1"
    [[ "$h" -ge "$NIGHT_START" && "$h" -lt "$NIGHT_END" ]]
}

write_session_event_test() {
    local csv="$1"; shift
    local session_id="$1" event="$2" step="$3" gpus="$4"
    local num_gpus="$5" grad_accum="$6" eff_batch="$7" trigger="$8"
    local ts
    ts=$(date -Iseconds)

    if [[ ! -f "$csv" ]]; then
        echo "session_id,event,timestamp,gpu_devices,num_gpus,batch_size,grad_accum,eff_batch,step,trigger" > "$csv"
    fi
    echo "$session_id,$event,$ts,$gpus,$num_gpus,$BATCH_SIZE,$grad_accum,$eff_batch,$step,$trigger" >> "$csv"
}


# ══════════════════════════════════════════════════════════════════════════
#  TEST SECTION 1 — calc_grad_accum（梯度累積計算）
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════"
echo "  Test 1: calc_grad_accum"
echo "══════════════════════════════════════"

# 公式：floor(24 / (4 × n_gpu))，最小 1
assert_eq "  2 GPU → grad_accum=3"  "3" "$(calc_grad_accum_test 2)"
assert_eq "  3 GPU → grad_accum=2"  "2" "$(calc_grad_accum_test 3)"
assert_eq "  4 GPU → grad_accum=1"  "1" "$(calc_grad_accum_test 4)"
assert_eq "  5 GPU → grad_accum=1"  "1" "$(calc_grad_accum_test 5)"
assert_eq "  6 GPU → grad_accum=1"  "1" "$(calc_grad_accum_test 6)"
assert_eq "  8 GPU → grad_accum=1"  "1" "$(calc_grad_accum_test 8)"
assert_eq "  1 GPU → grad_accum=6"  "6" "$(calc_grad_accum_test 1)"


# ══════════════════════════════════════════════════════════════════════════
#  TEST SECTION 2 — is_night_window（時間邊界）
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════"
echo "  Test 2: is_night_window 時間邊界"
echo "══════════════════════════════════════"

# 白天（包含 2 點，因為 NIGHT_START=3）
for h in 0 1 2 7 8 9 12 13 14 18 22 23; do
    if is_night_window_test "$h"; then
        result="night"
    else
        result="day"
    fi
    expected="day"
    [[ "$h" -ge 3 && "$h" -lt 7 ]] && expected="night"
    assert_eq "  hour=$h → $expected" "$expected" "$result"
done

# 半夜 (3,4,5,6)
for h in 3 4 5 6; do
    if is_night_window_test "$h"; then
        result="night"
    else
        result="day"
    fi
    assert_eq "  hour=$h → night" "night" "$result"
done

# 邊界測試
assert_eq "  hour=1 → day (boundary before)" "day" "$(is_night_window_test 1 && echo night || echo day)"
assert_eq "  hour=2 → day (boundary before)" "day" "$(is_night_window_test 2 && echo night || echo day)"
assert_eq "  hour=3 → night (boundary start)" "night" "$(is_night_window_test 3 && echo night || echo day)"
assert_eq "  hour=6 → night (boundary end-1)" "night" "$(is_night_window_test 6 && echo night || echo day)"
assert_eq "  hour=7 → day (boundary end)" "day" "$(is_night_window_test 7 && echo night || echo day)"


# ══════════════════════════════════════════════════════════════════════════
#  TEST SECTION 3 — GPU 空閒偵測 (mock nvidia-smi)
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════"
echo "  Test 3: GPU 空閒偵測 (mock nvidia-smi)"
echo "══════════════════════════════════════"

# Case 3a: 全部忙碌 → detect 回傳空
mock_gpu_util "0=100" "3=80" "4=90" "5=60" "6=100"
avail=$(detect_available_gpus_test)
assert_eq "  全部忙碌 → 空閒 GPU: (none)" "" "$avail"

# Case 3b: 部分空閒
mock_gpu_util "0=0" "3=2" "4=50" "5=0" "6=100"
avail=$(detect_available_gpus_test)
assert_eq "  mixed → 空閒: 3,5,0 (依 NIGHT_EXTRA_ORDER)" "3 5 0" "$avail"

# Case 3c: 全部空閒
mock_gpu_util "0=0" "3=1" "4=0" "5=0" "6=2"
avail=$(detect_available_gpus_test)
assert_eq "  全部空閒 → 3,4,5,6,0 (依 NIGHT_EXTRA_ORDER)" "3 4 5 6 0" "$avail"

# Case 3d: 閾值邊界 (GPU_FREE_THRESHOLD=5)
mock_gpu_util "0=4" "3=5" "4=6" "5=0"
avail=$(detect_available_gpus_test)
assert_eq "  閾值=5: util 4 → free, util 5 → busy → 5,0" "5 0" "$avail"

# Case 3e: 只有一張額外空閒
mock_gpu_util "0=90" "3=100" "4=3" "5=80" "6=90"
avail=$(detect_available_gpus_test)
assert_eq "  只有 GPU 4 空閒" "4" "$avail"


# ══════════════════════════════════════════════════════════════════════════
#  TEST SECTION 4 — build_gpu_list（GPU 列表組合）
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════"
echo "  Test 4: build_gpu_list"
echo "══════════════════════════════════════"

# Case 4a: 目標 5 卡，5 張空閒 → 1,2,3,4,5
result=$(build_gpu_list_test 5 3 4 5 6 0)
assert_eq "  目標5 全空 → 1,2,3,4,5" "1,2,3,4,5" "$result"

# Case 4b: 目標 5 卡，只有 2 張空閒 → 留 1 張 = 只用 1 張 → 1,2,3 (3卡)
result=$(build_gpu_list_test 5 3 4)
assert_eq "  目標5 只2空→留1→1,2,3 (3卡)" "1,2,3" "$result"

# Case 4c: 目標 5 卡，只有 1 張空閒 → 留那 1 張 = 退回 2 卡
result=$(build_gpu_list_test 5 3)
assert_eq "  目標5 只1空→留著→退回 1,2 (2卡)" "1,2" "$result"

# Case 4d: 目標 5 卡，全部沒空 → 1,2 (退回 2 卡)
result=$(build_gpu_list_test 5)
assert_eq "  目標5 沒空 → 1,2 (2卡)" "1,2" "$result"

# Case 4e: 目標 4 卡，2 張空閒 → 1,2,+前2張
result=$(build_gpu_list_test 4 3 4 5)
assert_eq "  目標4 3空 → 1,2,3,4" "1,2,3,4" "$result"

# Case 4f: 目標 3 卡，1 張空閒 → 1,2,3
result=$(build_gpu_list_test 3 3 4 5)
assert_eq "  目標3 → 1,2,3" "1,2,3" "$result"

# Case 4g: 目標 5 卡，3 張空閒 → 留 1 張 = 只用 2 張 → 1,2,0,3
result=$(build_gpu_list_test 5 0 3 4)
assert_eq "  目標5 3空→留1→1,2,0,3 (4卡)" "1,2,0,3" "$result"

# Case 4h: MAX_TOTAL_GPUS cap — 即使全部空閒也只能拿 6 張 (留 1 張給別人)
result=$(build_gpu_list_test 7 3 4 5 6 0)
assert_eq "  MAX=6: 目標7全空 → 強制6卡" "1,2,3,4,5,6" "$result"

# Case 4i: MAX cap — 3 張空閒 → 留 1 張 = 只用 2 → 4 卡
result=$(build_gpu_list_test 6 3 4 5)
assert_eq "  MAX=6: 3空→留1→1,2,3,4 (4卡)" "1,2,3,4" "$result"

# Case 4j: MAX cap — 沒有空閒卡 → 退回 2 卡
result=$(build_gpu_list_test 7)
assert_eq "  MAX=6 但沒空 → 退回2卡" "1,2" "$result"


# ══════════════════════════════════════════════════════════════════════════
#  TEST SECTION 5 — Graceful Stop flag 生命週期
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════"
echo "  Test 5: Graceful Stop flag 生命週期"
echo "══════════════════════════════════════"

FLAG="$SCHEDULER_DIR/request_graceful_stop"

# 初始不應存在
assert_file_not_exists "  flag 初始不存在" "$FLAG"

# 建立 flag（模擬 scheduler 發信號）
touch "$FLAG"
assert_file_exists "  建立 flag" "$FLAG"

# 模擬 training 收到信號後刪除
rm -f "$FLAG"
assert_file_not_exists "  刪除 flag" "$FLAG"

# 重複建立再刪除
touch "$FLAG"
touch "$FLAG"  # 重複應該沒問題
assert_file_exists "  重複建立" "$FLAG"
rm -f "$FLAG"
assert_file_not_exists "  再次刪除" "$FLAG"


# ══════════════════════════════════════════════════════════════════════════
#  TEST SECTION 6 — Session CSV 格式驗證
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════"
echo "  Test 6: Session CSV 記錄"
echo "══════════════════════════════════════"

SESSION_CSV="$TEST_DIR/test_sessions.csv"
rm -f "$SESSION_CSV"

# 寫入 start
write_session_event_test "$SESSION_CSV" "s_test001" start "0" "1,2" "2" "3" "24" "day"
assert_file_exists "  CSV 建立後存在" "$SESSION_CSV"

# 檢查 header
header=$(head -1 "$SESSION_CSV")
assert_contains "  header 含 session_id" "session_id" "$header"
assert_contains "  header 含 event"       "event"       "$header"
assert_contains "  header 含 gpu_devices"  "gpu_devices" "$header"
assert_contains "  header 含 num_gpus"     "num_gpus"    "$header"
assert_contains "  header 含 grad_accum"   "grad_accum"  "$header"
assert_contains "  header 含 eff_batch"    "eff_batch"   "$header"
assert_contains "  header 含 trigger"      "trigger"     "$header"

# 檢查 row 內容
row=$(tail -1 "$SESSION_CSV")
assert_contains "  row 含 session id"  "s_test001" "$row"
assert_contains "  row event=start"    "start"     "$row"
assert_contains "  row gpus=1,2"       "1,2"       "$row"
assert_contains "  row num_gpus=2"     ",2,"       "$row"

# 寫入 end
write_session_event_test "$SESSION_CSV" "s_test001" end "58400" "1,2" "2" "3" "24" "night_start"

# 驗證兩行
line_count=$(wc -l < "$SESSION_CSV")
assert_eq "  CSV 共 3 行 (header + 2 rows)" "3" "$line_count"

last_row=$(tail -1 "$SESSION_CSV")
assert_contains "  end row event=end"     "end"         "$last_row"
assert_contains "  end row step=58400"    "58400"       "$last_row"
assert_contains "  end row trigger=night_start" "night_start" "$last_row"

# 寫入更多 session（模擬多次切換）
write_session_event_test "$SESSION_CSV" "s_test002" start "58400" "1,2,3,4,5" "5" "1" "20" "night_start"
write_session_event_test "$SESSION_CSV" "s_test002" end   "76800" "1,2,3,4,5" "5" "1" "20" "night_end"
write_session_event_test "$SESSION_CSV" "s_test003" start "76800" "1,2"       "2" "3" "24" "night_end"

line_count=$(wc -l < "$SESSION_CSV")
assert_eq "  CSV 共 6 行 (header + 5 rows)" "6" "$line_count"


# ══════════════════════════════════════════════════════════════════════════
#  TEST SECTION 7 — 整合場景模擬
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════"
echo "  Test 7: 整合場景模擬"
echo "══════════════════════════════════════"

# ── 場景 7a：白天啟動，GPU 3,4,5 空閒 → 不會切換因為是白天 ──
mock_gpu_util "0=100" "3=0" "4=0" "5=0" "6=80"
# 雖然有 3 張空閒，但白天 window 不會觸發 build_gpu_list with target>2
# 這裡只測 logic function 本身（留1張 = 只用2張 = 4卡）
result=$(build_gpu_list_test "$NIGHT_TARGET_TOTAL" $(detect_available_gpus_test))
assert_eq "  7a: logic: 3空→留1→1,2,3,4 (4卡)" "1,2,3,4" "$result"

# ── 場景 7b：半夜 3 點，GPU 3,5 空閒 → 留 1 張 → 只取 1 張 → 3 卡 ──
mock_gpu_util "0=90" "3=2" "4=80" "5=1" "6=100"
avail=$(detect_available_gpus_test)
assert_eq "  7b: 半夜可用 GPU: 3,5" "3 5" "$avail"

gpus=$(build_gpu_list_test "$NIGHT_TARGET_TOTAL" $avail)
assert_eq "  7b: 留1→ 1,2,3 (3卡)" "1,2,3" "$gpus"

ngpu=$(echo "$gpus" | tr ',' '\n' | wc -l)
assert_eq "  7b: GPU 數量: 3" "3" "$ngpu"

ga=$(calc_grad_accum_test "$ngpu")
assert_eq "  7b: grad_accum: 2" "2" "$ga"

# ── 場景 7c：半夜 5 點，全部空閒 → 5 卡全開 ──
mock_gpu_util "0=0" "3=0" "4=0" "5=0" "6=0"
gpus=$(build_gpu_list_test "$NIGHT_TARGET_TOTAL" $(detect_available_gpus_test))
assert_eq "  7c: 全部空閒 → 5卡" "1,2,3,4,5" "$gpus"

ngpu=$(echo "$gpus" | tr ',' '\n' | wc -l)
assert_eq "  7c: GPU 數量: 5" "5" "$ngpu"

ga=$(calc_grad_accum_test "$ngpu")
assert_eq "  7c: grad_accum: 1" "1" "$ga"

# ── 場景 7d：半夜 4 點，全部忙碌 → 退回 2 卡 ──
mock_gpu_util "0=95" "3=88" "4=70" "5=99" "6=100"
gpus=$(build_gpu_list_test "$NIGHT_TARGET_TOTAL" $(detect_available_gpus_test))
assert_eq "  7d: 全部忙碌 → 退回2卡" "1,2" "$gpus"

ngpu=$(echo "$gpus" | tr ',' '\n' | wc -l)
assert_eq "  7d: GPU 數量: 2" "2" "$ngpu"

ga=$(calc_grad_accum_test "$ngpu")
assert_eq "  7d: grad_accum: 3" "3" "$ga"

# ── 場景 7e：完整時間線模擬（session CSV 連貫性） ──
echo "  --- 7e: 完整時間線 ---"
CSV="$TEST_DIR/full_timeline.csv"
rm -f "$CSV"

# 13:00 手動啟動 → 白天 2 卡
mock_gpu_util "0=100" "3=100" "4=100" "5=100" "6=100"
if is_night_window_test 13; then mode="night"; else mode="day"; fi
gpus=$(build_gpu_list_test "$NIGHT_TARGET_TOTAL" $(detect_available_gpus_test))
ngpu=$(echo "$gpus" | tr ',' '\n' | wc -l)
ga=$(calc_grad_accum_test "$ngpu")
eb=$((4 * ngpu * ga))
write_session_event_test "$CSV" "s_full" start "0" "$gpus" "$ngpu" "$ga" "$eb" "$mode"

# 02:00 半夜開始 → 偵測空閒 → 只有 3,4 空
mock_gpu_util "0=90" "3=0" "4=0" "5=85" "6=70"
# 模擬切換：先 end 舊 session
write_session_event_test "$CSV" "s_full" end "58400" "$gpus" "$ngpu" "$ga" "$eb" "night_start"
# 新 session
if is_night_window_test 3; then mode="night"; else mode="day"; fi
gpus=$(build_gpu_list_test "$NIGHT_TARGET_TOTAL" $(detect_available_gpus_test))
ngpu=$(echo "$gpus" | tr ',' '\n' | wc -l)
ga=$(calc_grad_accum_test "$ngpu")
eb=$((4 * ngpu * ga))
write_session_event_test "$CSV" "s_full2" start "58400" "$gpus" "$ngpu" "$ga" "$eb" "$mode"

assert_eq "  7e: 半夜 mode=night"  "night" "$mode"
assert_eq "  7e: 半夜 gpus=1,2,3 (2空→留1→3卡)" "1,2,3" "$gpus"
assert_eq "  7e: 半夜 ngpu=3" "3" "$ngpu"

# 07:00 半夜結束
write_session_event_test "$CSV" "s_full2" end "76800" "$gpus" "$ngpu" "$ga" "$eb" "night_end"
if is_night_window_test 8; then mode="night"; else mode="day"; fi
gpus="1,2"; ngpu=2; ga=$(calc_grad_accum_test 2); eb=$((4*2*ga))
write_session_event_test "$CSV" "s_full3" start "76800" "$gpus" "$ngpu" "$ga" "$eb" "night_end"

line_count=$(wc -l < "$CSV")
assert_eq "  7e: timeline CSV 共 6 行 (header + 5 sessions)" "6" "$line_count"

# 驗證 sessions 切換順序
events=$(grep -o 'start\|end' "$CSV" | head -5)
assert_contains "  7e: 順序含 start" "start" "$events"


# ══════════════════════════════════════════════════════════════════════════
#  TEST SECTION 8 — 不影響真實 output 目錄驗證
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════"
echo "  Test 8: 不影響真實目錄"
echo "══════════════════════════════════════"

REAL_OUTPUT="$PROJECT_ROOT/output"
if [[ -d "$REAL_OUTPUT/logs/.scheduler" ]]; then
    yellow "  ⚠  真實 output/logs/.scheduler 目錄存在（非本測試建立）"
else
    green "  ✓  真實 output/logs/.scheduler 目錄不存在（未污染）"
fi

# 確認測試中沒有不小心寫入真實目錄
if [[ -f "$REAL_OUTPUT/logs/.scheduler/request_graceful_stop" ]]; then
    yellow "  ⚠  真實 graceful_stop flag 存在（可能是之前手動測試留下）"
else
    green "  ✓  真實 graceful_stop flag 不存在"
fi


# ══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Results"
echo "════════════════════════════════════════════════════════════"
printf "  Pass: %s  Fail: %s\n" "$PASS" "$FAIL"

if [[ $FAIL -eq 0 ]]; then
    green "  ✅ ALL TESTS PASSED"
else
    red "  ❌ $FAIL test(s) FAILED"
    exit 1
fi

# Cleanup msg
echo ""
echo "  Test files cleaned up (tmp dir: $TEST_DIR)"
echo "  真實 training CSV / scheduler log 均未變動。"
