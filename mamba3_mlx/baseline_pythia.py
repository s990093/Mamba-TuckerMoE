"""Lazy-loaded Pythia-160M @ step1000 baseline.

Same training token budget as our 417M model (~2.1B tokens), so the comparison
is purely about architecture and post-training — not data quantity.

Pythia step math: batch=1024, seq_len=2048 → 2,097,152 tok/step.
step1000 → 2.097B tokens ≈ our 2.1B training budget.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterator

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHIA_DIR = _PROJECT_ROOT / "baselines" / "pythia-160m-step1000"

_lock = threading.Lock()
_model = None
_tok = None
_device = None


def _resolve_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_available(path: Path = DEFAULT_PYTHIA_DIR) -> bool:
    return (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()


def info(path: Path = DEFAULT_PYTHIA_DIR) -> dict:
    return {
        "path": str(path),
        "available_on_disk": is_available(path),
        "model_id": "EleutherAI/pythia-160m",
        "revision": "step1000",
        "params_M": 162.0,
        "train_tokens_B": 2.097,  # 1000 × (1024 × 2048) / 1e9
        "device": _device,
        "loaded": _model is not None,
    }


def ensure_loaded(path: Path = DEFAULT_PYTHIA_DIR) -> None:
    global _model, _tok, _device
    if _model is not None:
        return
    if not is_available(path):
        raise FileNotFoundError(
            f"Pythia weights not found at {path}. "
            f"Run: hf download EleutherAI/pythia-160m --revision step1000 "
            f"--local-dir {path}"
        )
    with _lock:
        if _model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _device = _resolve_device()
        _tok = AutoTokenizer.from_pretrained(str(path))
        if _tok.pad_token_id is None:
            _tok.pad_token = _tok.eos_token
        # low_cpu_mem_usage=False forces eager materialisation on CPU so the
        # subsequent .to(device) doesn't trip over meta tensors. (Default True
        # uses the accelerate meta-device path which can't be .to()-moved.)
        mdl = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.float32, low_cpu_mem_usage=False,
        )
        _model = mdl.to(_device).eval()


def stream(
    prompt: str,
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    seed: int | None = None,
) -> Iterator[dict]:
    """Yield events: meta → token(text, tok_s) → done(...)."""
    import torch
    from transformers import TextIteratorStreamer

    ensure_loaded()
    assert _model is not None and _tok is not None and _device is not None

    if seed is not None:
        torch.manual_seed(seed)

    ids = _tok(prompt, return_tensors="pt").input_ids.to(_device)
    n_prompt = int(ids.shape[1])
    yield {"type": "meta", "n_prompt": n_prompt, "device": _device}

    streamer = TextIteratorStreamer(_tok, skip_prompt=True, skip_special_tokens=True)
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
        except Exception as exc:
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
