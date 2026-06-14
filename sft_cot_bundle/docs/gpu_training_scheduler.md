# GPU Training Scheduler — 動態 GPU 排程系統

## 概述

自動化管理 SFT-CoT 訓練的 GPU 資源，根據時間與 GPU 空閒狀態動態切換 GPU 數量，
自動調整梯度累積步數以維持訓練穩定性，並透過優雅停止（graceful stop）在 checkpoint 後安全切換。

**時區：台灣時間 (Asia/Taipei, UTC+8)**，透過 `export TZ=Asia/Taipei` 固定。

## 運作模式

```
日用 (03:00–07:00 以外)      半夜 (03:00 – 07:00 台灣時間)
┌─────────────────┐        ┌──────────────────────────────┐
│  GPU: 1,2       │        │  目標 5 張卡                   │
│  DDP: 2         │        │  上限 6 張 (留 1 張給別人)    │
│  grad_accum: 3  │        │  遞減順序: 6→5→4→3→2         │
│  eff_batch: 24  │        │  優先加 3,4,5,6,0            │
└─────────────────┘        └──────────────────────────────┘
```

### GPU 數量與梯度累積對照表

| 額外可用 | 留空 | 實際取用 | 總 GPU | grad_accum | eff_batch |
|---------|------|---------|--------|-----------|-----------|
| 5 張    | 1 張 | 4 張    | 6      | 1         | 24        |
| 3 張    | 1 張 | 2 張    | 4      | 1         | 16        |
| 2 張    | 1 張 | 1 張    | 3      | 2         | 24        |
| 1 張    | 1 張 | 0 張    | 2      | 3         | 24        |
| 0 張    | —    | 0 張    | 2      | 3         | 24        |

> **核心規則：永遠留至少 1 張額外空閒 GPU 給別人。** 只有當可用的額外卡 > 1 張時才多拿。

## 架構

```
start_training.sh
  └─ gpu_scheduler.sh (daemon, 背景常駐)
       ├─ GPU 探測 (nvidia-smi)
       ├─ 時間窗口判斷
       ├─ 優雅停止信號 → train_sft.py 收到後在下一個 checkpoint 退出
       ├─ Session 記錄 → output/logs/.scheduler/sessions.csv
       └─ 重啟訓練 (新 GPU config + auto resume)
```

### 檔案說明

| 檔案 | 用途 |
|------|------|
| `scripts/training/gpu_scheduler.sh` | 核心排程 daemon |
| `scripts/training/start_training.sh` | 啟動入口（預設走 scheduler，可用 env var 手動模式） |
| `scripts/training/stop_training.sh` | 優雅停止（先等 checkpoint，逾時再 force kill） |
| `scripts/train_sft.py` | 訓練程式（新增 graceful stop flag 檢查） |
| `output/logs/.scheduler/sessions.csv` | Session 記錄 |
| `output/logs/.scheduler/request_graceful_stop` | 優雅停止信號檔 |

## 使用方法

### 啟動（預設 scheduler 模式）

```bash
cd /home/hungwei/llm/sft_cot_bundle
./scripts/training/start_training.sh
```

會自動：
1. 判斷當前是否在半夜窗口
2. 半夜時偵測 GPU 空閒狀況
3. 以最佳 GPU 配置啟動訓練
4. 背景常駐監控，自動在 02:00 / 07:00 切換

### 查看狀態

```bash
./scripts/training/start_training.sh status
```

輸出範例：
```
✅ Scheduler running (PID=12345)
✅ Training running (PID=12346, last ckpt step=58400)
   Scheduler log: tail -f output/logs/.scheduler/scheduler.log
   Sessions:      cat output/logs/.scheduler/sessions.csv
```

### 停止

```bash
./scripts/training/start_training.sh stop
```

會先發優雅停止信號，訓練跑完下一個 checkpoint 後退出。

### 手動模式（不用 scheduler，直接指定 GPU config）

```bash
# 2 卡手動跑
GPU_DEVICES=1,2 NUM_GPUS=2 GRAD_ACCUM=3 ./scripts/training/start_training.sh

# 5 卡手動跑
GPU_DEVICES=1,2,3,4,5 NUM_GPUS=5 GRAD_ACCUM=2 ./scripts/training/start_training.sh
```

### 直接啟動 scheduler（前景除錯）

```bash
./scripts/training/gpu_scheduler.sh --foreground
```

## Session 記錄

`output/logs/.scheduler/sessions.csv` 每次啟動/停止/切換都會記錄：

| 欄位 | 說明 |
|------|------|
| `session_id` | 唯一 session ID (s_YYYYMMDD_HHMMSS) |
| `event` | `start` / `end` |
| `timestamp` | ISO 8601 |
| `gpu_devices` | e.g. `1,2,3,4,5` |
| `num_gpus` | GPU 數量 |
| `batch_size` | 固定 4 |
| `grad_accum` | 動態計算的梯度累積 |
| `eff_batch` | 有效 batch |
| `step` | checkpoint step |
| `trigger` | `day` / `night` / `night_start` / `night_end` / `auto_restart` / `crashed` |

範例：
```csv
session_id,event,timestamp,gpu_devices,num_gpus,batch_size,grad_accum,eff_batch,step,trigger
s_20260528_140000,start,2026-05-28T14:00:00+08:00,1,2,2,4,3,24,0,day
s_20260528_140000,end,2026-05-29T02:05:00+08:00,1,2,2,4,3,24,58400,night_start
s_20260529_020500,start,2026-05-29T02:05:00+08:00,1,2,3,4,5,5,2,40,58400,night_start
s_20260529_020500,end,2026-05-29T07:03:00+08:00,1,2,3,4,5,5,2,40,76800,night_end
s_20260529_070300,start,2026-05-29T07:03:00+08:00,1,2,2,4,3,24,76800,night_end
```

## 優雅停止流程 (Graceful Stop)

```
Scheduler 決定要切換
  │
  ├─ 寫入 output/logs/.scheduler/request_graceful_stop
  │
  ▼
train_sft.py 繼續訓練（不中斷）
  │
  ├─ 跑到下一個 checkpoint step（每 100 step）
  ├─ 儲存 checkpoint
  ├─ 檢查 graceful stop flag file → 存在
  ├─ 刪除 flag file
  ├─ print 優雅停止訊息
  ├─ break（退出訓練迴圈）
  ├─ 正常關閉 log file
  │
  ▼
Scheduler 偵測到 training process 已退出
  │
  ├─ 寫入 session end record
  ├─ 切換 GPU config
  ├─ 用 SFT_COT_AUTO_RESUME=1 重啟 training
  │
  ▼
訓練從上次 checkpoint 續跑（新 GPU 數 + 新 grad_accum）
```

逾時機制：
- Scheduler 最多等 30 分鐘（`GRACEFUL_TIMEOUT=1800`）
- `stop_training.sh` 手動停止最多等 20 分鐘（`STOP_TIMEOUT=1200`）
- 逾時後先 SIGTERM，再 SIGKILL

## 配置參數

所有參數在 `gpu_scheduler.sh` 頂端，可依需求修改：

```bash
BASE_GPUS="1,2"                    # 永遠使用的 GPU
NIGHT_EXTRA_ORDER=(3 4 5 6 0)      # 半夜額外 GPU 的嘗試順序
NIGHT_TARGET_TOTAL=5               # 半夜目標 GPU 總數
MAX_TOTAL_GPUS=6                   # 上限：留至少 1 張卡給別人
NIGHT_START=3                      # 半夜開始時間 (24 小時制)
NIGHT_END=7                        # 半夜結束時間
BATCH_SIZE=4
TARGET_EFF_BATCH=24                # 目標有效 batch size
GRACEFUL_TIMEOUT=1800              # 優雅停止逾時 (秒)
GPU_FREE_THRESHOLD=5               # GPU 使用率 < 此值視為空閒
POLL_INTERVAL=60                   # Scheduler 檢查間隔 (秒)
```

## 半夜 GPU 偵測邏輯

```
1. 在 02:00 進入半夜窗口
2. 用 nvidia-smi 查詢 GPU 0,3,4,5,6 的使用率
3. 篩選出使用率 < 5% 的 GPU（視為空閒）
4. 依照順序 3,4,5,6,0 從空閒 GPU 中挑選
5. **永遠留至少 1 張額外空閒 GPU**（如果只空 1 張就不拿）
6. 最多不超過 MAX_TOTAL_GPUS=6 張
7. 自動計算對應的 grad_accum
8. 如果一張額外 GPU 都沒有 → 維持 2 卡
```

## 監控與除錯

```bash
# 查看 scheduler log
tail -f output/logs/.scheduler/scheduler.log

# 查看當前訓練 log
tail -f $(ls -t output/logs/train_sft_cot_*.log | head -1)

# 查看 session 記錄
cat output/logs/.scheduler/sessions.csv

# 查看當前 GPU 使用狀況
nvidia-smi

# 查看 scheduler 狀態
./scripts/training/start_training.sh status
```

## 注意事項

1. **DDP 不支援 hot-swap**：切換 GPU 數量必須 restart training，但透過 checkpoint resume 無縫接續
2. **有效 batch 守恆**：grad_accum 會自動調整，但因為整數取整無法完全一致（24→32 或 24→40），影響輕微
3. **半夜 GPU 檢查只在切換時執行一次**：如果在半夜期間有人搶走 GPU，不會自動縮減（DDP 限制）
4. **Scheduler 本身 crash**：sessions.csv 會有 `crashed` 記錄，重新啟動 `start_training.sh` 即可恢復
5. **同時只能有一個 scheduler**：重複啟動會報錯並退出

## 測試（不需等到半夜）

### 1. 測試特定 GPU 配置（手動模式，不走 scheduler）

```bash
# 5 卡測試
GPU_DEVICES=1,2,3,4,5 NUM_GPUS=5 GRAD_ACCUM=1 ./scripts/training/start_training.sh

# 2 卡測試（等同日用）
GPU_DEVICES=1,2 NUM_GPUS=2 GRAD_ACCUM=3 ./scripts/training/start_training.sh
```

### 2. 測試 scheduler 邏輯（模擬半夜時間）

```bash
# 模擬凌晨 3 點啟動 scheduler（會自動偵測 GPU + 用半夜模式）
SIMULATE_HOUR=3 ./scripts/training/gpu_scheduler.sh --foreground

# 模擬白天 14 點（強制日用模式）
SIMULATE_HOUR=14 ./scripts/training/gpu_scheduler.sh --foreground

# 先模擬半夜 3 點跑，然後手動切換到 8 點測試切回白天
# (需要開兩個 terminal)
```

### 3. 測試 graceful stop 機制

```bash
# 先手動跑 2 卡訓練
GPU_DEVICES=1,2 NUM_GPUS=2 GRAD_ACCUM=3 ./scripts/training/start_training.sh

# 等訓練跑起來後，模擬 graceful stop：
# 手動建立 flag file，觀察訓練是否在下一個 checkpoint 後退出
mkdir -p output/logs/.scheduler
touch output/logs/.scheduler/request_graceful_stop

# 監看訓練 log 確認
tail -f $(ls -t output/logs/train_sft_cot_*.log | head -1)
```

### 4. 完整端到端測試（模擬半夜切換）

```bash
# Terminal 1: 啟動 scheduler（模擬白天 13 點，2 卡）
SIMULATE_HOUR=13 ./scripts/training/gpu_scheduler.sh --foreground

# 等訓練跑幾個 checkpoint 後...

# Terminal 2: 手動觸發夜間切換（停止 scheduler 後以半夜模式重啟）
# 這會觸發 graceful stop → checkpoint → 重啟
./scripts/training/start_training.sh stop
SIMULATE_HOUR=3 ./scripts/training/gpu_scheduler.sh --foreground

# 驗證 session CSV 記錄
cat output/logs/.scheduler/sessions.csv
```
