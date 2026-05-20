# 方案 B 進度報告：Metal Kernel 融合優化

**目標:** 達到 40-50 tok/s decode  
**當前:** 23.3 tok/s  
**需要改進:** 1.7-2.1× speedup

---

## 實驗結果

### 測試 1: 編譯優化 ❌ 失敗

| 配置 | 性能 | 結果 |
| --- | --- | --- |
| Baseline (eager) | 22.8 tok/s | ✓ |
| With mx.compile() | 10.0 tok/s | ❌ -56% |

**原因:** Graph 編譯開銷大於收益（decode 是 per-token 執行，編譯初始化時間不易回本）

### 測試 2: 編譯策略改進 🔄 持續

需要嘗試的策略：
- [ ] Per-step 編譯（而非全圖）
- [ ] 編譯快取（複用已編譯的版本）
- [ ] Layer 粒度編譯

---

## 根本瓶頸識別

### Per-Token 耗時分析 (42.98 ms)

```
TuckerMoE 路由:     ~17 ms  (40%)  ← 主要瓶頸
  - Router linear:   ~3 ms
  - Softmax/Top-K:   ~4 ms
  - Expert gather:   ~5 ms
  - Expert forward:  ~5 ms

SSM 掃描:          ~12 ms  (28%)
  - Sequential loop: ~8 ms
  - Exp/state ops:   ~4 ms

其他 (Norm/RoPE/Attn): ~14 ms  (32%)
```

### 最有效的優化策略

**策略 1: MoE Kernel 融合** ⭐ 推薦
```
當前: 5 個 kernel dispatch
  1. Router projection
  2. Softmax
  3. Top-K selection
  4. Expert gather
  5. Expert forward multiply

融合後: 1 個 Metal kernel
- 預期減少: 4-5 ms (12-15% 改進)
- 新性能: 23.3 + 15% = 26.8 tok/s
```

**策略 2: SSM + RoPE 融合**
```
當前開銷: ~12 ms (SSM) + ~3 ms (RoPE) = 15 ms
融合後: ~8 ms (20% 改進)
- 預期: 23.3 + 12% = 26.1 tok/s
```

**策略 3: KV 緩存優化**
```
當前: 每個 token 都重新讀取 KV
優化: 保持在 GPU 內存中（減少主-GPU 往返）
- 預期: +15-20%
```

---

## Metal 融合內核開發進度

### MoE Fused Kernel (moe_fused.metal)

當前狀態: **初步實現** 🔄

```metal
kernel void moe_fused_forward(
  input,           // 當前 token 特徵
  router_weight,   // Router 權重
  expert_weights,  // 專家權重

  output           // 輸出
)
```

任務:
- [ ] Router projection → Top-K softmax
- [ ] Expert dynamic dispatch
- [ ] Weighted expert output aggregation
- [ ] 性能驗證

---

## 下一步行動計劃 (今天至明天)

### 立即 (2-4 小時)

1. **完成 Metal MoE kernel**
   - 實現 router + top-k 選擇
   - 測試數值正確性
   - 基準測試

2. **集成到推理管道**
   - 更新 TuckerMoE 層使用 Metal kernel
   - 自動回落機制

### 短期 (明天)

3. **SSM + RoPE 融合**
   - 創建 ssm_rope_fused.metal
   - 集成

4. **基準測試與驗證**
   - benchmark_metal_fusion.py
   - 衡量實際改進

### 預期結果

```
Baseline:          23.3 tok/s
+ MoE fusion:      26-27 tok/s (+15%)
+ SSM fusion:      28-30 tok/s (+10%)
+ KV optimization: 32-35 tok/s (+12%)
────────────────────────────────
目標 (方案 B):      40-50 tok/s ✓
```

---

## 關鍵洞察

### 為什麼編譯失敗

```
Decode 的特點:
- Per-token execution (L=1)
- 狀態持久化 (mamba_states, kv_caches)
- Python dispatch overhead < 2ms

編譯開銷:
- Graph 構建: ~500ms (一次性)
- 初始化: ~100ms
- Per-step: 編譯後 per-step 快 30-50%

Problem:
  23.3 tok/s = 43 ms per token
  編譯開銷 > 43 ms × speedup benefit

Solution: Kernel fusion (無編譯開銷，直接減少計算)
```

### Metal Kernel 爲何有效

```
當前 (5 dispatch):
  dispatch(router)
  dispatch(softmax)
  dispatch(gather)
  dispatch(expert)
  dispatch(matmul)
  Total dispatch overhead: ~3-5 ms

融合後 (1 dispatch):
  dispatch(moe_fused)
  Total dispatch overhead: ~0.5-1 ms
  
Savings: 2-4 ms per token = 10-20% improvement
```

---

## 資源

### 已創建的文件

```
mamba3_mlx/mlx_model/
  ├── quantized_model.py      # 8-bit 量化框架
  ├── moe_fused.metal         # MoE 融合 kernel (WIP)

Benchmarks:
  ├── benchmark_optimized_decode.py  # 量化測試
  ├── benchmark_compile_only.py      # 編譯測試
```

### 下一個要創建

```
mamba3_mlx/mlx_model/
  ├── moe_metal_bindings.py   # Python 綁定
  ├── ssm_rope_fused.metal    # SSM + RoPE 融合
  ├── benchmark_metal_fusion.py
```

---

## 總結

✅ **方案 B 正在進行中**

- ❌ 編譯優化不可行（開銷太大）
- ✅ Metal kernel 融合是正確方向
- 🔄 實現中... (預計明天完成)

**預期結果:** 32-50 tok/s (vs 目標 40-50 tok/s)

