"""Side-by-side baseline comparison: official Mamba-130M / Mamba-370M.

Usage:
    .venv/bin/python3 baselines/compare_baselines.py
    .venv/bin/python3 baselines/compare_baselines.py --prompt "Who are you?"
    .venv/bin/python3 baselines/compare_baselines.py --max-new-tokens 80 --sample

The point: run the SAME prompt through Mamba-130M and Mamba-370M (both at the
same ~or larger~ parameter scale than our 417M model) and show that vanilla
Mamba baselines at this scale produce incoherent / repetitive output. Our
417M Hybrid Mamba-TuckerMoE produces grounded, in-character, CoT-aware text.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM

BASE = Path(__file__).resolve().parent
MODELS = {
    "Mamba-130M (official)": BASE / "mamba-130m-hf",
    "Mamba-370M (official)": BASE / "mamba-370m-hf",
}

DEFAULT_PROMPTS = [
    "Who are you?",
    "Are you Mamba?",
    "Explain why the sky is blue in one sentence.",
]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_model(path: Path, prompt: str, max_new_tokens: int, sample: bool, device: str):
    tok = AutoTokenizer.from_pretrained(str(path))
    model = MambaForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32).to(device).eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=0.7 if sample else 1.0,
            top_p=0.9 if sample else 1.0,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.time() - t0
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    tps = (out.shape[1] - ids.shape[1]) / max(elapsed, 1e-6)
    return text.strip(), n_params, tps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    device = pick_device()
    print(f"[device] {device}\n")

    for prompt in prompts:
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        print("=" * 80)
        for label, path in MODELS.items():
            if not (path / "model.safetensors").exists() and not (path / "pytorch_model.bin").exists():
                print(f"  [{label}] (weights not downloaded yet — skip)\n")
                continue
            text, n_params, tps = run_model(path, prompt, args.max_new_tokens, args.sample, device)
            print(f"\n  [{label}] params={n_params:.1f}M  speed={tps:.1f} tok/s")
            print(f"  ->  {text}\n")
        print()


if __name__ == "__main__":
    main()
