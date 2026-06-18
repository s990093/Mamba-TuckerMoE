"""Lazy-loaded official Mamba-130M (HF) baseline.

Used by the /baselines comparison page to stream same-scale baseline output
alongside our 417M Hybrid Mamba-TuckerMoE.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_DIR = _PROJECT_ROOT / "baselines" / "mamba-130m-hf"

# Lazy globals
_lock = threading.Lock()
_model = None
_tok = None
_device = None
_loaded_path: Path | None = None


def _resolve_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_available(path: Path = DEFAULT_BASELINE_DIR) -> bool:
    return (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()


def info(path: Path = DEFAULT_BASELINE_DIR) -> dict:
    return {
        "path": str(path),
        "available_on_disk": is_available(path),
        "model_id": "state-spaces/mamba-130m-hf",
        "params_M": 129.1,
        "train_tokens_B": 300.0,  # The Pile, official Mamba release
        "device": _device,
        "loaded": _model is not None,
    }


def ensure_loaded(path: Path = DEFAULT_BASELINE_DIR) -> None:
    """Load Mamba-130M weights into memory if not already loaded."""
    global _model, _tok, _device, _loaded_path
    if _model is not None and _loaded_path == path:
        return
    if not is_available(path):
        raise FileNotFoundError(
            f"Mamba-130M weights not found at {path}. "
            f"Run: hf download state-spaces/mamba-130m-hf --local-dir {path}"
        )
    with _lock:
        if _model is not None and _loaded_path == path:
            return
        import torch
        from transformers import AutoTokenizer, MambaForCausalLM

        _device = _resolve_device()
        _tok = AutoTokenizer.from_pretrained(str(path))
        # fp32 keeps numerics stable on MPS; the model is small (~500MB).
        # low_cpu_mem_usage=False avoids the accelerate meta-device path that
        # otherwise breaks subsequent .to(device).
        mdl = MambaForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.float32, low_cpu_mem_usage=False,
        )
        mdl = mdl.to(_device).eval()
        _model = mdl
        _loaded_path = path


def stream(
    prompt: str,
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    seed: int | None = None,
) -> Iterator[dict]:
    """Yield events: meta → token(text, tok_s) → done(total_tokens, tok_s, ttft_ms)."""
    import torch
    from transformers import TextIteratorStreamer

    ensure_loaded()
    assert _model is not None and _tok is not None and _device is not None

    if seed is not None:
        torch.manual_seed(seed)

    ids = _tok(prompt, return_tensors="pt").input_ids.to(_device)
    n_prompt = int(ids.shape[1])
    yield {"type": "meta", "n_prompt": n_prompt, "device": _device}

    streamer = TextIteratorStreamer(
        _tok, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = dict(
        input_ids=ids,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature) if do_sample else 1.0,
        top_p=float(top_p) if do_sample else 1.0,
        pad_token_id=_tok.eos_token_id,
        streamer=streamer,
    )

    t0 = time.perf_counter()
    ttft_ms: float | None = None
    n_out = 0

    def _run() -> None:
        try:
            with torch.no_grad():
                _model.generate(**gen_kwargs)
        except Exception as exc:  # surface via streamer end
            streamer.text_queue.put(f"\n[error] {type(exc).__name__}: {exc}")
            streamer.text_queue.put(streamer.stop_signal)

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    for chunk in streamer:
        if not chunk:
            continue
        n_out += 1
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000.0
        elapsed = max(time.perf_counter() - t0, 1e-6)
        yield {
            "type": "token",
            "text": chunk,
            "n": n_out,
            "tok_s": round(n_out / elapsed, 1),
        }

    elapsed = max(time.perf_counter() - t0, 1e-6)
    yield {
        "type": "done",
        "total_tokens": n_out,
        "total_ms": round(elapsed * 1000.0, 2),
        "tok_s": round(n_out / elapsed, 1),
        "ttft_ms": round(ttft_ms or 0.0, 2),
    }
