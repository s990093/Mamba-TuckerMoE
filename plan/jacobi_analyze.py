#!/usr/bin/env python3
"""
Jacobi 隔離測試結果分析器
讀取 plan/isolation_results/ 下的所有輸出，計算重複率、速度、ARL，輸出 Markdown 報告。
"""
from __future__ import annotations
import os
import re
import sys
from collections import Counter
from pathlib import Path


# ── 工具函數 ────────────────────────────────────────────────────────────────

def extract_assistant_text(raw: str) -> str:
    """從 jacobi_stream.py 輸出中取出 Assistant: 後的生成文字。"""
    # jacobi_stream 輸出格式:  \nAssistant:\n<text>\n──────...
    m = re.search(r'Assistant:\s*\n(.*?)(?:\n──|$)', raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    # stream_mlx.py 格式: Assistant:\n<text>
    m = re.search(r'Assistant:\s*(.*?)(?:\n─|Streamed|$)', raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()[:800]


def repetition_score(text: str, n: int = 5) -> float:
    """
    n-gram 重複率：重複 n-gram 數 / 總 n-gram 數。
    1.0 = 完全重複；0.0 = 無任何重複。
    """
    tokens = text.split()
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(v - 1 for v in counts.values() if v > 1)
    return repeated / len(ngrams)


def longest_repeated_substring(text: str, min_len: int = 15) -> tuple[str, int]:
    """找出最長重複子字串及其出現次數。"""
    best = ("", 1)
    words = text.split()
    for w_len in range(min(30, len(words) // 2), min_len - 1, -1):
        seen: dict[tuple, int] = {}
        for i in range(len(words) - w_len + 1):
            key = tuple(words[i:i + w_len])
            seen[key] = seen.get(key, 0) + 1
        for key, cnt in seen.items():
            if cnt > best[1] or (cnt == best[1] and len(key) > len(best[0].split())):
                best = (" ".join(key), cnt)
        if best[1] > 1:
            break
    return best


def extract_metrics(raw: str) -> dict:
    """從 jacobi_stream.py 或 compare 腳本輸出中提取速度/ARL 指標。"""
    m = {}
    # tok/s
    tok_s = re.search(r'([\d.]+)\s*tok/s', raw)
    if tok_s:
        m["tok_s"] = float(tok_s.group(1))
    # speedup
    sp = re.search(r'([\d.]+)×\s*speedup', raw)
    if sp:
        m["speedup"] = float(sp.group(1))
    # ARL
    arl = re.search(r'ARL\s*[=:]\s*([\d.]+)', raw)
    if not arl:
        arl = re.search(r'ARL.*?(\d+\.\d+)', raw)
    if arl:
        m["arl"] = float(arl.group(1))
    # ngram hit
    ng = re.search(r'ngram=([\d.]+)%', raw)
    if ng:
        m["ngram_pct"] = float(ng.group(1))
    # intra / replay
    intra = re.search(r'intra=(\d+)', raw)
    if intra:
        m["intra"] = int(intra.group(1))
    replay = re.search(r'replay=(\d+)', raw)
    if replay:
        m["replay"] = int(replay.group(1))
    # exit code
    ec = re.search(r'EXIT_CODE=(\d+)', raw)
    if ec:
        m["exit_code"] = int(ec.group(1))
    return m


STEP_LABELS = {
    "0":  "AR baseline (stream_mlx)",
    "0b": "AR compare_ar_jacobi",
    "1":  "純 Jacobi (no-ngram, no-dyn-K, no-intra)",
    "2":  "ngram only",
    "3":  "dynamic-K only",
    "4":  "intra only",
    "5a": "ngram + dynamic-K",
    "5b": "ngram + intra",
    "5c": "dynamic-K + intra",
    "6":  "全功能 default",
}

FEATURE_MAP = {
    "1":  {"ngram": False, "dyn_k": False, "intra": False},
    "2":  {"ngram": True,  "dyn_k": False, "intra": False},
    "3":  {"ngram": False, "dyn_k": True,  "intra": False},
    "4":  {"ngram": False, "dyn_k": False, "intra": True},
    "5a": {"ngram": True,  "dyn_k": True,  "intra": False},
    "5b": {"ngram": True,  "dyn_k": False, "intra": True},
    "5c": {"ngram": False, "dyn_k": True,  "intra": True},
    "6":  {"ngram": True,  "dyn_k": True,  "intra": True},
}


def main() -> None:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("plan/isolation_results")

    rows: list[dict] = []
    for step_id, label in STEP_LABELS.items():
        fname = outdir / f"step{step_id}.txt"
        if step_id == "0":
            fname = outdir / "step0_ar.txt"
        elif step_id == "0b":
            fname = outdir / "step0b_compare.txt"
        if not fname.exists():
            continue
        raw = fname.read_text(errors="replace")
        text = extract_assistant_text(raw)
        rep = repetition_score(text)
        lrs, lrs_cnt = longest_repeated_substring(text)
        metrics = extract_metrics(raw)
        rows.append({
            "step": step_id,
            "label": label,
            "features": FEATURE_MAP.get(step_id, {}),
            "text_preview": text[:200].replace("\n", "↵"),
            "rep_score": rep,
            "longest_rep": lrs[:60],
            "longest_rep_cnt": lrs_cnt,
            "metrics": metrics,
            "ok": metrics.get("exit_code", 0) == 0,
        })

    # ── 打印報告 ─────────────────────────────────────────────────────────────
    print("# Jacobi 隔離測試分析報告\n")
    print(f"Results directory: `{outdir}`\n")

    # 表格
    print("## 指標摘要\n")
    print(f"{'步驟':<6} {'功能組合':<38} {'重複率':>7} {'最長重複(次)':>14} {'tok/s':>7} {'ARL':>6} {'speedup':>8}")
    print("-" * 100)
    ar_toks = None
    for r in rows:
        m = r["metrics"]
        toks = f"{m['tok_s']:.1f}" if "tok_s" in m else "—"
        arl  = f"{m['arl']:.2f}"   if "arl"   in m else "—"
        spd  = f"{m['speedup']:.2f}×" if "speedup" in m else "—"
        rep  = f"{r['rep_score']*100:.1f}%"
        lrep = f"{r['longest_rep_cnt']}×" if r["longest_rep_cnt"] > 1 else "—"
        feat = r["features"]
        feat_str = (
            ("N" if feat.get("ngram") else "·") +
            ("D" if feat.get("dyn_k") else "·") +
            ("I" if feat.get("intra") else "·")
            if feat else "AR"
        )
        status = "✓" if r["ok"] else "✗"
        if r["step"] in ("0", "0b"):
            ar_toks = m.get("tok_s")
        print(f"Step {r['step']:<4} {status} [{feat_str}] {r['label']:<34} {rep:>7} {lrep:>14} {toks:>7} {arl:>6} {spd:>8}")

    # ── 文字預覽 ─────────────────────────────────────────────────────────────
    print("\n## 輸出預覽 (前200字元)\n")
    for r in rows:
        print(f"### Step {r['step']}: {r['label']}")
        print(f"```\n{r['text_preview']}\n```")
        if r["longest_rep_cnt"] > 1:
            print(f"⚠️  最長重複片段 ({r['longest_rep_cnt']} 次): `{r['longest_rep']}`")
        print()

    # ── 根因分析 ─────────────────────────────────────────────────────────────
    print("## 根因分析\n")

    # 取得各步驟重複率
    rep_by_step = {r["step"]: r["rep_score"] for r in rows}
    baseline_rep = rep_by_step.get("1", rep_by_step.get("0", 0.0))

    anomalies: list[str] = []
    for step_id in ("2", "3", "4"):
        if step_id not in rep_by_step:
            continue
        delta = rep_by_step[step_id] - baseline_rep
        feat = {"2": "ngram", "3": "dynamic-K", "4": "intra"}[step_id]
        if delta > 0.05:
            anomalies.append(f"- **{feat}** (Step {step_id}): 重複率 +{delta*100:.1f}pp ← 可疑")
        else:
            print(f"- {feat} (Step {step_id}): 重複率 delta={delta*100:+.1f}pp — 單獨無問題")

    combo_anomalies: list[str] = []
    for step_id in ("5a", "5b", "5c"):
        if step_id not in rep_by_step:
            continue
        delta = rep_by_step[step_id] - baseline_rep
        feat_map = {"5a": "ngram+dyn-K", "5b": "ngram+intra", "5c": "dyn-K+intra"}
        if delta > 0.05:
            combo_anomalies.append(f"- **{feat_map[step_id]}** (Step {step_id}): 重複率 +{delta*100:.1f}pp ← 組合問題")

    if not anomalies and not combo_anomalies:
        full_delta = rep_by_step.get("6", 0.0) - baseline_rep
        if full_delta > 0.05:
            print(f"\n⚠️  單一功能皆無問題，但全功能 (Step 6) 重複率 +{full_delta*100:.1f}pp")
            print("   → 三者交互作用導致問題，需要更細粒度測試。")
        else:
            print("\n✅ 所有步驟重複率無顯著差異。")
            print("   → 重複輸出可能來自模型本身（checkpoint 品質），Jacobi 加速功能無負面影響。")
            if ar_toks:
                print(f"   AR baseline: {ar_toks:.1f} tok/s")
    else:
        if anomalies:
            print("\n⚠️  **單一功能問題**：")
            for a in anomalies:
                print(a)
        if combo_anomalies:
            print("\n⚠️  **組合問題**：")
            for a in combo_anomalies:
                print(a)

    # ── 比對 Step1(純Jacobi) vs Step0(AR) 重複率 ────────────────────────────
    ar_rep = rep_by_step.get("0", None)
    jac_rep = rep_by_step.get("1", None)
    if ar_rep is not None and jac_rep is not None:
        print(f"\n### AR vs 純Jacobi 重複率")
        print(f"- AR : {ar_rep*100:.1f}%")
        print(f"- 純Jacobi: {jac_rep*100:.1f}%")
        if abs(jac_rep - ar_rep) < 0.05:
            print("→ 基礎 Jacobi 與 AR 重複率相近，問題來自 Jacobi 加速功能或 checkpoint。")
        elif jac_rep > ar_rep + 0.05:
            print("→ 純 Jacobi 本身即導致更多重複，基礎 verify/replay 可能有問題。")

    print("\n---\n_分析完成_")


if __name__ == "__main__":
    main()
