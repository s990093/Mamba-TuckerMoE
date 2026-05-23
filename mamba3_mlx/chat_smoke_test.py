"""chat_smoke_test.py — boot chat_demo.py, send one prompt over WebSocket,
verify the full event sequence (meta → reasoning → token … → done), and
exit non-zero if anything is off.

Designed for the `make chat-smoke` target.  Spawns chat_demo.py as a
subprocess, waits for ``/api/status`` to report ready, opens a WS, runs
one chat turn, and tears the server down.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def _wait_ready(port: int, timeout_s: float = 180.0) -> bool:
    """Poll /api/status until ready=True or timeout."""
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/api/status"
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                data = json.loads(resp.read())
            if data.get("ready"):
                return True
            last_err = f"not ready (load_timings={data.get('load_timings', {})})"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(1.0)
    print(f"[smoke] FAIL: server never reported ready (last={last_err})", file=sys.stderr)
    return False


async def _run_one_turn(port: int, prompt: str, max_tokens: int) -> dict:
    """Connect, fire one chat turn, collect events, return verdict."""
    import websockets
    url = f"ws://127.0.0.1:{port}/ws"
    verdict: dict = {
        "ok": False,
        "saw_meta": False,
        "saw_reasoning": False,
        "saw_token": False,
        "saw_done": False,
        "stop_reason": None,
        "errors": [],
        "events_seen": 0,
        "reasoning_chars": 0,
        "final_text": "",
        "done_payload": None,
    }
    async with websockets.connect(url, max_size=None) as ws:
        # Wait for "connected".
        first = json.loads(await ws.recv())
        if first.get("type") != "connected":
            verdict["errors"].append(f"expected 'connected', got {first}")
            return verdict
        # Fire the chat action.  Use the self_awareness category + the same
        # sampling knobs that make run_cot reliably close <think>/<final>.
        await ws.send(json.dumps({
            "action": "chat",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "reasoning": True,
            "category_key": "self_awareness",
            "sampling": {
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.9,
                "min_p": 0.05,
                "repetition_penalty": 1.1,
            },
        }))
        # Drain events until "done" or "error".
        timeout_s = 240.0
        deadline = time.time() + timeout_s
        final_parts: list[str] = []
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            except asyncio.TimeoutError:
                verdict["errors"].append("WS recv timeout")
                return verdict
            ev = json.loads(raw)
            verdict["events_seen"] += 1
            t = ev.get("type")
            if t == "meta":
                verdict["saw_meta"] = True
            elif t == "reasoning":
                verdict["saw_reasoning"] = True
                verdict["reasoning_chars"] = len(ev.get("markdown") or "")
            elif t == "token":
                verdict["saw_token"] = True
                final_parts.append(ev.get("text") or "")
            elif t == "done":
                verdict["saw_done"] = True
                verdict["done_payload"] = ev
                verdict["final_text"] = "".join(final_parts)
                break
            elif t == "error":
                verdict["errors"].append(str(ev.get("message")))
                return verdict
    # Server-level criteria: full event sequence + no errors.  Whether the
    # *model* closed </think> and produced <final> tokens depends on sampling
    # and is treated as a soft signal (logged but not failure-causing).
    verdict["ok"] = (
        verdict["saw_meta"]
        and verdict["saw_done"]
        and (verdict["saw_reasoning"] or verdict["saw_token"])
        and not verdict["errors"]
    )
    verdict["model_closed_think"] = bool(verdict["saw_token"])
    return verdict


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7861)))
    p.add_argument("--prompt", type=str,
                   default="What is 2 + 2? Answer in one short sentence.")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--ready-timeout", type=float, default=180.0)
    p.add_argument("--python", type=str, default="")
    p.add_argument("--server-log", type=str,
                   default=str(HERE / ".chat_smoke_server.log"))
    args = p.parse_args()

    py = args.python or shutil.which("python3") or sys.executable
    venv_py = REPO_ROOT / ".venv" / "bin" / "python3"
    if venv_py.is_file():
        py = str(venv_py)

    print(f"[smoke] python = {py}")
    print(f"[smoke] port   = {args.port}")
    print(f"[smoke] prompt = {args.prompt!r}")
    print(f"[smoke] log    = {args.server_log}")

    log_f = open(args.server_log, "w")
    cmd = [py, "-m", "mamba3_mlx.chat_demo", "--port", str(args.port)]
    print(f"[smoke] launching server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT),
        stdout=log_f, stderr=subprocess.STDOUT,
    )

    rc = 1
    try:
        if not _wait_ready(args.port, args.ready_timeout):
            print(f"[smoke] FAIL: server didn't become ready (log → {args.server_log})")
            return 2
        print("[smoke] server ready — running one chat turn …")
        verdict = asyncio.run(_run_one_turn(args.port, args.prompt, args.max_tokens))
        print("[smoke] verdict =", json.dumps({k: v for k, v in verdict.items()
                                              if k != "done_payload"}, indent=2))
        if verdict.get("done_payload"):
            d = verdict["done_payload"]
            print(f"[smoke] done: prefill_ms={d.get('prefill_ms')} "
                  f"ttft_ms={d.get('ttft_ms')} total_ms={d.get('total_ms')} "
                  f"out_tokens={d.get('total_tokens')} tok/s={d.get('tok_s')}")
        print(f"[smoke] final answer text ({len(verdict['final_text'])} chars):")
        print("  " + (verdict["final_text"][:300] or "<empty>"))
        rc = 0 if verdict["ok"] else 3
    finally:
        _kill(proc)
        try:
            log_f.close()
        except Exception:
            pass

    print(f"[smoke] {'PASS' if rc == 0 else 'FAIL'} (exit {rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
