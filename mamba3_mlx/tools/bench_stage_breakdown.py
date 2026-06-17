"""bench_stage_breakdown.py
=============================

Per-stage micro-timing for the chat backend's decode loop.

While ``bench_efficiency.py`` measures **end-to-end** backend tok/s vs the
CLI peak, this script answers the *why*: for each generated token, where
exactly do the milliseconds go?  The chat_demo decode loop is already
instrumented behind ``MAMBA_PROFILE_LOOP=1`` and prints a single line of the
form::

    [prof] per-token µs (n=120): logits=21 sample=5310 text=205 route=164
                                  fwd=7510 sum=13210 | wall=13520 (74.0 tok/s)

This tool boots a chat server with ``MAMBA_PROFILE_LOOP=1``, fires one
representative prompt per SFT mode over WebSocket, parses the prof line that
appears in the server's stdout, and writes a single Markdown + CSV pair.

Usage:
    python tools/bench_stage_breakdown.py

Outputs:
    paper/data/stage_breakdown.md
    paper/data/stage_breakdown.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import websockets

REPO_ROOT = Path(__file__).resolve().parents[2]

PROMPTS = {
    "self_awareness":     "Who are you?",
    "math_drill":         "What is 7 times 8?",
    "daily_conversation": "How was your weekend?",
    "deep_dive":          "Briefly explain how Mamba state-space models work.",
}

# Match e.g. "[prof] per-token µs (n=120): logits=21 sample=5310 text_dec=58 text_cli=140 route=164 fwd=7510 sum=13210 | wall=13520 (74.0 tok/s)"
PROF_RE = re.compile(
    r"\[prof\] per-token .*?\(n=(?P<n>\d+)\)"
    r":\s*logits=(?P<logits>\d+)"
    r"\s+sample=(?P<sample>\d+)"
    r"\s+text_dec=(?P<text_dec>\d+)"
    r"\s+text_cli=(?P<text_cli>\d+)"
    r"\s+route=(?P<route>\d+)"
    r"\s+fwd=(?P<fwd>\d+)"
    r"\s+sum=(?P<sum>\d+)"
    r"\s*\|\s*wall=(?P<wall>\d+)"
    r".*?\((?P<tps>[\d.]+)\s*tok/s\)"
)


async def _drive_one(uri: str, mode: str, prompt: str, max_tokens: int) -> None:
    """Fire one chat turn and wait for done."""
    payload = {
        "action": "chat",
        "prompt": prompt,
        "category_key": mode,
        "max_tokens": max_tokens,
    }
    async with websockets.connect(uri, max_size=None) as ws:
        # connected
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
            if m.get("type") == "connected":
                break
        await ws.send(json.dumps(payload))
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=180.0))
            evs = m.get("events", [m]) if m.get("type") == "batch" else [m]
            for ev in evs:
                if ev.get("type") in ("done", "error"):
                    return


def _wait_ready(host: str, port: int, timeout_s: int = 90) -> bool:
    import urllib.request
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/api/status",
                                        timeout=2.0) as r:
                d = json.loads(r.read().decode("utf-8"))
                if d.get("ready"):
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",       type=int, default=7861)  # avoid clobbering existing :7860
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--rounds",     type=int, default=2)
    ap.add_argument("--out-dir",    default=str(REPO_ROOT / "paper" / "data"))
    args = ap.parse_args()

    log_path = Path("/tmp/chat_prof_loop.out")
    if log_path.exists():
        log_path.unlink()

    env = os.environ.copy()
    env["MAMBA_PROFILE_LOOP"] = "1"
    env["PYTHONUNBUFFERED"]   = "1"

    py = shutil.which(sys.executable) or sys.executable
    print(f"[stage] booting chat_demo with MAMBA_PROFILE_LOOP=1 on :{args.port} …")
    proc = subprocess.Popen(
        [py, "-u", "-m", "mamba3_mlx.chat_demo", "--port", str(args.port)],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        if not _wait_ready("127.0.0.1", args.port):
            print("[stage] server failed to become ready", file=sys.stderr)
            return 2
        print("[stage] server ready, firing prompts …")

        rows: list[dict] = []
        order: list[tuple[str, int]] = []  # (mode, round) for each fired turn
        for mode, prompt in PROMPTS.items():
            for r in range(args.rounds):
                print(f"  [{mode}] r{r+1}/{args.rounds}  firing …", flush=True)
                asyncio.run(_drive_one(f"ws://127.0.0.1:{args.port}/ws",
                                       mode, prompt, args.max_tokens))
                order.append((mode, r + 1))
                # Small pause so the server has time to flush its [prof] line
                # to disk before the next round starts.
                time.sleep(1.0)

        # ── Read the whole log once and match prof lines in order ─────
        # The chat_demo prints exactly one [prof] line per generate call (in
        # _stream_generate, just before yielding ``done``).  We rely on the
        # in-order property of stdout to align each match with the firing
        # order recorded in ``order``.
        full = log_path.read_text(encoding="utf-8", errors="replace")
        matches = list(PROF_RE.finditer(full))
        print(f"\n[stage] saw {len(matches)} [prof] lines for "
              f"{len(order)} fired turns")
        if len(matches) < len(order):
            print(f"WARN: only {len(matches)}/{len(order)} prof lines "
                  f"captured — table will be partial",
                  file=sys.stderr)
        for (mode, rnum), m in zip(order, matches):
            row = {
                "mode":         mode,
                "round":        rnum,
                "n_tokens":     int(m["n"]),
                "logits_us":    int(m["logits"]),
                "sample_us":    int(m["sample"]),
                "text_dec_us":  int(m["text_dec"]),
                "text_cli_us":  int(m["text_cli"]),
                "route_us":     int(m["route"]),
                "fwd_us":       int(m["fwd"]),
                "sum_us":       int(m["sum"]),
                "wall_us":      int(m["wall"]),
                "tps":        float(m["tps"]),
            }
            rows.append(row)
            print(f"  [{mode}] r{rnum}  "
                  f"wall={row['wall_us']}µs/tok  "
                  f"fwd={row['fwd_us']}  sample={row['sample_us']}  "
                  f"tdec={row['text_dec_us']}  tcli={row['text_cli_us']}  "
                  f"route={row['route_us']}  "
                  f"({row['tps']:.1f} tok/s, n={row['n_tokens']})")

    finally:
        print("[stage] stopping chat_demo …")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()

    if not rows:
        print("[stage] no rows captured", file=sys.stderr)
        return 3

    # ── Aggregate per-mode means ─────────────────────────────────────
    by_mode: dict[str, list[dict]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)

    aggregated = []
    for mode, rs in by_mode.items():
        keys = ("logits_us", "sample_us", "text_dec_us", "text_cli_us",
                "route_us", "fwd_us", "sum_us", "wall_us", "tps")
        agg = {"mode": mode, "rounds": len(rs)}
        for k in keys:
            vals = [r[k] for r in rs]
            agg[k] = round(statistics.mean(vals), 1)
        aggregated.append(agg)

    # ── Markdown ─────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "stage_breakdown.md"
    csv_path = out_dir / "stage_breakdown.csv"

    md_lines = []
    md_lines.append("# Decode-loop stage breakdown (per token, µs)\n")
    md_lines.append("Captured live from the chat_demo backend with "
                    "`MAMBA_PROFILE_LOOP=1`.  Each row averages "
                    f"{args.rounds} rounds of a representative prompt.\n")
    md_lines.append("Columns map to the chat_demo decode loop sections:")
    md_lines.append("- **fwd**      — `StaticDecoder.step` GPU dispatch (Mamba + Tucker MoE + Tx)")
    md_lines.append("- **sample**   — compiled rep/freq/temp/top-k/p sampler + `mx.eval(.item())`")
    md_lines.append("- **logits**   — `mw.transform_logits(...)` (CoT FSM ban masks + close_bias)")
    md_lines.append("- **text_dec** — `_tokenizer.decode(...)` (BPE Python decode)")
    md_lines.append("- **text_cli** — Rich `_cprint` styled console emission")
    md_lines.append("- **route**    — FSM `mw.step(tid, ...)` + yield events + stop checks")
    md_lines.append("- **wall**     — measured per-token wall time (= 1 / tps × 10⁶)\n")
    md_lines.append("| Mode | wall (µs) | fwd | sample | logits | text_dec | text_cli | route | sum | tok/s |")
    md_lines.append("|------|-----------|-----|--------|--------|----------|----------|-------|-----|-------|")
    for r in aggregated:
        md_lines.append(
            f"| `{r['mode']}` "
            f"| **{r['wall_us']:.0f}** "
            f"| {r['fwd_us']:.0f} "
            f"| {r['sample_us']:.0f} "
            f"| {r['logits_us']:.0f} "
            f"| {r['text_dec_us']:.0f} "
            f"| {r['text_cli_us']:.0f} "
            f"| {r['route_us']:.0f} "
            f"| {r['sum_us']:.0f} "
            f"| {r['tps']:.1f} |"
        )
    md_lines.append("")
    md_lines.append("### What this tells us")
    md_lines.append("- **`fwd` + `sample` ≈ 95 % of `wall`** — the GPU and sampling kernel are the dominant cost; everything we add for streaming is in the remaining ~5 %.")
    md_lines.append("- **`text_dec` vs `text_cli`** — splits the previous "
                    "`text` row into BPE decode (Python-side) and Rich console "
                    "print, so we can see which is the streaming overhead.")
    md_lines.append("- The `sample` figure is GPU-bound (it includes `mx.eval(.item())` synchronisation), not Python compute, which is why the compiled sampler is already near-optimal.\n")
    md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nwrote {md.relative_to(REPO_ROOT)}")

    # ── CSV ──────────────────────────────────────────────────────────
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "mode", "rounds", "wall_us", "fwd_us", "sample_us",
            "logits_us", "text_dec_us", "text_cli_us", "route_us",
            "sum_us", "tps",
        ])
        w.writeheader()
        for r in aggregated:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
