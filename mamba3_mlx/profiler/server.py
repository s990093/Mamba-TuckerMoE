#!/usr/bin/env python3
"""
Mamba3-XR Profiler — standalone WebSocket metrics server.

Streams a fixed JSON envelope every second:
  { schema_version, snapshot, history }

Usage:
  python -m mamba3_mlx.profiler.server
  python -m mamba3_mlx.profiler.server --port 8765 --no-gpu

GPU metrics via ioreg AGXAccelerator (no sudo required). CPU/RAM/swap via psutil.
LLM fields are populated when chat_demo publishes via ``profiler.bridge``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_INF_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _INF_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from mamba3_mlx.profiler import config as _cfg
from mamba3_mlx.profiler.buffer import RollingBuffer
from mamba3_mlx.profiler.collectors.gpu import start_gpu_monitor, stop_gpu_monitor
from mamba3_mlx.profiler.llm_state import set_inference_state, update_llm_telemetry
from mamba3_mlx.profiler.metrics import collect_metrics
from mamba3_mlx.profiler.schema import SCHEMA_VERSION, make_envelope

_UI_DIR = _INF_DIR / "ui"
_buffer = RollingBuffer(_cfg.BUFFER_SECONDS)
_gpu_enabled = True


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if _gpu_enabled:
        ok = start_gpu_monitor(_cfg.GPU_SAMPLE_MS)
        if not ok:
            print("[profiler] GPU monitor unavailable (powermetrics/sudo). GPU fields stay null.")
    yield
    stop_gpu_monitor()


app = FastAPI(title="Mamba3-XR Profiler", version=str(SCHEMA_VERSION), lifespan=_lifespan)
if _UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")


@app.get("/")
async def index() -> HTMLResponse:
    html = (_UI_DIR / "profiler.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/schema")
async def schema() -> JSONResponse:
    return JSONResponse(
        {
            "schema_version": SCHEMA_VERSION,
            "interval_s": _cfg.INTERVAL_S,
            "buffer_seconds": _cfg.BUFFER_SECONDS,
            "snapshot_fields": [
                "timestamp",
                "gpu",
                "cpu",
                "memory",
                "swap",
                "thermal",
                "llm",
                "state",
            ],
        }
    )


class TelemetryIn(BaseModel):
    """Partial LLM + inference state push from chat_demo (separate process)."""

    llm: dict | None = None
    state: dict | None = None


@app.post("/telemetry")
async def ingest_telemetry(body: TelemetryIn) -> JSONResponse:
    if body.llm:
        update_llm_telemetry(**body.llm)
    if body.state:
        set_inference_state(**body.state)
    return JSONResponse({"ok": True})


@app.get("/snapshot")
async def snapshot_rest() -> JSONResponse:
    snap = collect_metrics()
    _buffer.push(snap)
    return JSONResponse(make_envelope(snap, _buffer.list()))


@app.websocket("/ws")
async def websocket_metrics(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_text(
        json.dumps(
            {
                "type": "connected",
                "schema_version": SCHEMA_VERSION,
                "interval_s": _cfg.INTERVAL_S,
                "buffer_seconds": _cfg.BUFFER_SECONDS,
            }
        )
    )
    try:
        while True:
            snap = collect_metrics()
            _buffer.push(snap)
            payload = make_envelope(snap, _buffer.list())
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(_cfg.INTERVAL_S)
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        raise


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Mamba3-XR inference profiler server")
    parser.add_argument("--host", default=_cfg.HOST)
    parser.add_argument("--port", type=int, default=_cfg.PORT)
    parser.add_argument("--interval", type=float, default=_cfg.INTERVAL_S,
                        help="Seconds between WebSocket ticks")
    parser.add_argument("--buffer", type=int, default=_cfg.BUFFER_SECONDS,
                        help="Rolling history length (samples)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Skip powermetrics GPU monitor")
    args = parser.parse_args()

    global _gpu_enabled, _buffer
    _cfg.HOST = args.host
    _cfg.PORT = args.port
    _cfg.INTERVAL_S = max(0.2, float(args.interval))
    _cfg.BUFFER_SECONDS = max(1, int(args.buffer))
    _gpu_enabled = not args.no_gpu
    _buffer = RollingBuffer(_cfg.BUFFER_SECONDS)

    print(f"[profiler] http://{args.host}:{args.port}/  ws://{args.host}:{args.port}/ws")
    if not _gpu_enabled:
        print("[profiler] GPU monitor disabled (--no-gpu)")
    else:
        print("[profiler] GPU monitor via sudo powermetrics (fields null until samples arrive)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
