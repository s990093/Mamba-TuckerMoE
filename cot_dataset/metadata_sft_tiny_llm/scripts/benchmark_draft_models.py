#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import LlamaConfig, LlamaForCausalLM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "draft_model_benchmark_mps.json"


@dataclass
class ModelSpec:
    name: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
def get_candidate_specs() -> list[ModelSpec]:
    return [
        # (name, vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads)
        
        # 組別 A：試探底線 (淺層)
        ModelSpec("15M_baseline", 32007, 192, 768, 8, 3, 1),
        
        # 組別 B：20M 級距 (用 256 維度，但層數變淺，確保 Head=64)
        ModelSpec("25M_mid", 32007, 256, 1024, 10, 4, 1),
        
        # 組別 C：黃金比例 (深窄型，最容易學會 CoT)
        ModelSpec("33M_recommended", 32007, 256, 1024, 16, 4, 1),
        
        # 組別 D：加寬嘗試
        ModelSpec("44M_deeper", 32007, 320, 1280, 16, 5, 1),
        
        # 組別 E：效能天花板 (改回 16 層)
        ModelSpec("60M_upper", 32007, 384, 1536, 16, 6, 1),
    ]

def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device: torch.device, requested: str) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    # auto
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def create_model(spec: ModelSpec, device: torch.device, dtype: torch.dtype) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=spec.vocab_size,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.num_hidden_layers,
        num_attention_heads=spec.num_attention_heads,
        num_key_value_heads=spec.num_key_value_heads,
        max_position_embeddings=spec.max_position_embeddings,
        rms_norm_eps=spec.rms_norm_eps,
        tie_word_embeddings=spec.tie_word_embeddings,
    )
    model = LlamaForCausalLM(config)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def benchmark_model(
    model: LlamaForCausalLM,
    spec: ModelSpec,
    device: torch.device,
    prompt_tokens: int,
    max_new_tokens: int,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, Any]:
    input_ids = torch.randint(
        low=0,
        high=spec.vocab_size,
        size=(1, prompt_tokens),
        device=device,
        dtype=torch.long,
    )

    with torch.inference_mode():
        for _ in range(warmup_steps):
            _ = model(input_ids=input_ids)
        synchronize(device)

        t0 = time.perf_counter()
        for _ in range(measure_steps):
            _ = model(input_ids=input_ids)
        synchronize(device)
        forward_elapsed = time.perf_counter() - t0

        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": 0,
            "eos_token_id": 1,
        }
        for _ in range(max(1, warmup_steps // 2)):
            _ = model.generate(**generate_kwargs)
        synchronize(device)

        t1 = time.perf_counter()
        for _ in range(measure_steps):
            _ = model.generate(**generate_kwargs)
        synchronize(device)
        generate_elapsed = time.perf_counter() - t1

    avg_forward_s = forward_elapsed / measure_steps
    avg_generate_s = generate_elapsed / measure_steps
    generate_toks_per_s = max_new_tokens / avg_generate_s
    total_params = model.num_parameters()

    return {
        "name": spec.name,
        "params_m": round(total_params / 1e6, 2),
        "avg_forward_ms": round(avg_forward_s * 1000, 2),
        "avg_generate_ms": round(avg_generate_s * 1000, 2),
        "generate_tokens_per_s": round(generate_toks_per_s, 2),
        "tokens_per_million_params": round(generate_toks_per_s / (total_params / 1e6), 2),
        "config": asdict(spec),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'rank':<4} {'name':<18} {'params(M)':>9} {'fwd(ms)':>9} "
        f"{'gen(ms)':>9} {'tok/s':>10} {'tok/s/M':>10}"
    )
    print("-" * 74)
    for idx, row in enumerate(rows, start=1):
        print(
            f"{idx:<4} {row['name']:<18} {row['params_m']:>9.2f} {row['avg_forward_ms']:>9.2f} "
            f"{row['avg_generate_ms']:>9.2f} {row['generate_tokens_per_s']:>10.2f} "
            f"{row['tokens_per_million_params']:>10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 15M~60M Llama draft-model speed.")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)

    print(f"device={device} dtype={dtype} prompt_tokens={args.prompt_tokens} max_new_tokens={args.max_new_tokens}")
    if device.type == "mps":
        print("note: MPS 不支援 flash_attention_2，本測速使用 PyTorch 預設 attention 實作。")

    results: list[dict[str, Any]] = []
    specs = get_candidate_specs()

    for spec in specs:
        print(f"\n[benchmark] {spec.name} ...")
        model = create_model(spec=spec, device=device, dtype=dtype)
        row = benchmark_model(
            model=model,
            spec=spec,
            device=device,
            prompt_tokens=args.prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
        results.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ranked = sorted(results, key=lambda r: r["generate_tokens_per_s"], reverse=True)
    print("\n=== Ranked by generate tok/s ===")
    print_table(ranked)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "dtype": str(dtype),
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "ranked_results": ranked,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved JSON report to: {output_path}")


if __name__ == "__main__":
    main()
