"""bench_efficiency.py
=========================

Measure **backend-vs-CLI efficiency** for the chat_demo streaming server.

Story
-----
The Mamba3-XR CLI runs a tight pure-compute loop (StaticDecoder with quantised
weights and Metal fusion).  The chat backend wraps that *same* compiled graph
in a streaming service that adds per-token overhead: tokenizer decode, CoT
format FSM, JSON-encoded WebSocket frames, asyncio scheduling.

This script quantifies that overhead per SFT mode and writes a single
artefact pair (Markdown table + CSV) ready to drop into the report.

What it measures (per mode, repeated R times):
    1. **CLI peak**:   the StaticDecoder bench printed by chat_demo at startup
                       (5 warmup rounds + 1 timed round, identical to the
                       ``make self-s`` path).  This is the same number now
                       exposed via ``/api/status.cli_peak_tps``.
    2. **Backend tps**: tok/s reported in the ``done`` event of a real chat
                       turn over WebSocket.
    3. **Efficiency**:  backend / CLI peak.

Usage
-----
    # Server already running on :7860
    python tools/bench_efficiency.py --rounds 3

    # Custom subset of modes
    python tools/bench_efficiency.py --modes self_awareness math_drill --rounds 5

Outputs
-------
    paper/data/efficiency_table.md   ← Markdown for the report
    paper/data/efficiency_table.csv  ← raw numbers for further plots
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import websockets

REPO_ROOT = Path(__file__).resolve().parents[2]

# Same prompts make_<mode> targets use; aligned with mode_configs.py.
MODES: dict[str, str] = {
    "self_awareness":     "Who are you?",
    "emotion":            "I had a really tough day at work today.",
    "summarize_email":    "Summarize: Hi team, the deployment is delayed by two days, please update the schedule.",
    "movie_intro":        "Tell me about Inception.",
    "daily_conversation": "How was your weekend?",
    "math_drill":         "What is 7 times 8?",
    "system_call":        "Schedule a meeting for tomorrow at 10 am.",
    "deep_dive":          "Briefly explain how Mamba state-space models work.",
}


def fetch_cli_peak(url: str) -> float | None:
    """Grab CLI peak tok/s from /api/status (measured at server boot)."""
    try:
        with urllib.request.urlopen(url, timeout=5.0) as r:
            data = json.loads(r.read().decode("utf-8"))
        peak = data.get("cli_peak_tps")
        return float(peak) if peak else None
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None


async def run_one(uri: str, mode: str, prompt: str, max_tokens: int) -> dict:
    """Drive one chat turn over WS, return decode tok/s + a few timings."""
    payload = {
        "action": "chat",
        "prompt": prompt,
        "category_key": mode,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    n_tokens = 0
    first_token_t: float | None = None
    async with websockets.connect(uri, max_size=None) as ws:
        # Drain "connected"
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            m = json.loads(raw)
            if m.get("type") == "connected":
                break
        await ws.send(json.dumps(payload))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=180.0)
            m = json.loads(raw)
            events = m.get("events", [m]) if m.get("type") == "batch" else [m]
            for ev in events:
                t = ev.get("type")
                if t == "token":
                    n_tokens += 1
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                elif t == "done":
                    return {
                        "mode":       mode,
                        "tok_s":      ev.get("tok_s") or ev.get("decode_tps"),
                        "prefill_ms": ev.get("prefill_ms"),
                        "ttft_ms":    ev.get("ttft_ms")
                                       or ((first_token_t - t0) * 1000.0
                                           if first_token_t else None),
                        "n_tokens":   n_tokens,
                        "wall_s":     time.perf_counter() - t0,
                    }
                elif t == "error":
                    return {"mode": mode, "error": ev.get("message", "unknown")}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri",        default="ws://localhost:7860/ws")
    ap.add_argument("--status-url", default="http://localhost:7860/api/status")
    ap.add_argument("--modes",      nargs="*", default=list(MODES.keys()))
    ap.add_argument("--rounds",     type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--out-dir",    default=str(REPO_ROOT / "paper" / "data"))
    args = ap.parse_args()

    cli_peak = fetch_cli_peak(args.status_url)
    if cli_peak is None:
        print(f"WARN: could not fetch CLI peak from {args.status_url} — "
              "is chat_demo running? continuing without efficiency column",
              file=sys.stderr)

    results: list[dict] = []
    for mode in args.modes:
        prompt = MODES.get(mode)
        if prompt is None:
            print(f"unknown mode: {mode}", file=sys.stderr)
            continue
        round_tps: list[float] = []
        round_pre: list[float] = []
        round_ttft: list[float] = []
        n_tok = None
        err: str | None = None
        for r in range(args.rounds):
            print(f"  [{mode}] round {r+1}/{args.rounds} …", end="", flush=True)
            res = await run_one(args.uri, mode, prompt, args.max_tokens)
            if "error" in res:
                err = res["error"]
                print(f"  ERR: {err}")
                break
            round_tps.append(res["tok_s"])
            if res.get("prefill_ms"): round_pre.append(res["prefill_ms"])
            if res.get("ttft_ms"):    round_ttft.append(res["ttft_ms"])
            n_tok = res["n_tokens"]
            print(f"  {res['tok_s']:.1f} tok/s "
                  f"({res['n_tokens']} tok, ttft={res.get('ttft_ms', 0):.0f}ms)")
        if err:
            results.append({"mode": mode, "error": err})
            continue
        mean_tps = statistics.mean(round_tps)
        results.append({
            "mode":       mode,
            "tok_s_mean": mean_tps,
            "tok_s_min":  min(round_tps),
            "tok_s_max":  max(round_tps),
            "tok_s_std":  statistics.stdev(round_tps) if len(round_tps) > 1 else 0.0,
            "prefill_ms": statistics.mean(round_pre) if round_pre else None,
            "ttft_ms":    statistics.mean(round_ttft) if round_ttft else None,
            "n_tokens":   n_tok,
            "rounds":     len(round_tps),
            "efficiency": (mean_tps / cli_peak) if cli_peak else None,
        })

    # ── Write artefacts ─────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path  = out_dir / "efficiency_table.md"
    csv_path = out_dir / "efficiency_table.csv"

    # Markdown
    md_lines = []
    md_lines.append("# Backend-vs-CLI Efficiency\n")
    md_lines.append("Measurements taken with the chat_demo WebSocket server "
                    "running locally; CLI peak refers to the StaticDecoder "
                    "pure-compute loop (5 warmup rounds + 1 timed round, "
                    "identical to `make self-s`).\n")
    if cli_peak:
        md_lines.append(f"**CLI peak (this machine):** {cli_peak:.1f} tok/s "
                        f"(measured once at server boot)\n")
    md_lines.append("")
    md_lines.append("| Mode | Backend tok/s (mean ± std) | min | max | "
                    "TTFT (ms) | Prefill (ms) | Tokens | Efficiency |")
    md_lines.append("|------|-----------------------------|-----|-----|"
                    "-----------|--------------|--------|------------|")
    for r in results:
        if "error" in r:
            md_lines.append(f"| `{r['mode']}` | ERROR: {r['error']} | | | | | | |")
            continue
        eff_cell = f"{r['efficiency']*100:.1f} %" if r.get("efficiency") else "—"
        ttft     = f"{r['ttft_ms']:.0f}"  if r.get("ttft_ms")    is not None else "—"
        prefill  = f"{r['prefill_ms']:.0f}" if r.get("prefill_ms") is not None else "—"
        md_lines.append(
            f"| `{r['mode']}` "
            f"| {r['tok_s_mean']:.2f} ± {r['tok_s_std']:.2f} "
            f"| {r['tok_s_min']:.2f} | {r['tok_s_max']:.2f} "
            f"| {ttft} | {prefill} "
            f"| {r['n_tokens']} | {eff_cell} |"
        )
    if cli_peak:
        ok_eff = [r["efficiency"] for r in results
                  if r.get("efficiency") is not None]
        if ok_eff:
            md_lines.append("")
            md_lines.append(
                f"**Average efficiency across modes:** "
                f"{statistics.mean(ok_eff)*100:.1f} % "
                f"(min {min(ok_eff)*100:.1f} %, max {max(ok_eff)*100:.1f} %)"
            )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {md_path.relative_to(REPO_ROOT)}")

    # CSV
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "mode", "tok_s_mean", "tok_s_min", "tok_s_max", "tok_s_std",
            "prefill_ms", "ttft_ms", "n_tokens", "rounds",
            "cli_peak_tps", "efficiency", "error",
        ])
        w.writeheader()
        for r in results:
            row = {**r, "cli_peak_tps": cli_peak}
            for k in list(row):
                if isinstance(row[k], float) and row[k] is not None:
                    row[k] = round(row[k], 4)
            w.writerow({k: row.get(k, "") for k in w.fieldnames})
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
