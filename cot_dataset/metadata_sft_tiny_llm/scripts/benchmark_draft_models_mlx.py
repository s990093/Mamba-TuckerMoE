#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mu


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "draft_model_benchmark_mlx.json"


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
        ModelSpec("15M_baseline", 32007, 192, 768, 8, 3, 1),
        ModelSpec("25M_mid", 32007, 256, 1024, 10, 4, 1),
        ModelSpec("33M_recommended", 32007, 256, 1024, 16, 4, 1),
        ModelSpec("44M_deeper", 32007, 320, 1280, 16, 5, 1),
        ModelSpec("60M_upper", 32007, 384, 1536, 16, 6, 1),
    ]


class MlxTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, eps: float) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=eps)
        self.attn = nn.MultiHeadAttention(dims=hidden_size, num_heads=num_heads, bias=False)
        self.norm2 = nn.RMSNorm(hidden_size, eps=eps)
        self.ffn_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        h = self.norm1(x)
        h = self.attn(h, h, h, mask=mask)
        x = x + h
        h2 = self.norm2(x)
        h2 = nn.silu(self.ffn_gate(h2)) * self.ffn_up(h2)
        h2 = self.ffn_down(h2)
        return x + h2


class MlxDraftLlama(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.embed = nn.Embedding(spec.vocab_size, spec.hidden_size)
        self.blocks = [
            MlxTransformerBlock(
                hidden_size=spec.hidden_size,
                intermediate_size=spec.intermediate_size,
                num_heads=spec.num_attention_heads,
                eps=spec.rms_norm_eps,
            )
            for _ in range(spec.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(spec.hidden_size, eps=spec.rms_norm_eps)
        self.lm_head = nn.Linear(spec.hidden_size, spec.vocab_size, bias=False)
        self.causal_mask_cache: dict[tuple[int, str], mx.array] = {}

    def _causal_mask(self, seq_len: int, dtype: mx.Dtype) -> mx.array:
        key = (seq_len, str(dtype))
        if key not in self.causal_mask_cache:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            self.causal_mask_cache[key] = mask.astype(dtype)
        return self.causal_mask_cache[key]

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = self.embed(input_ids)
        mask = self._causal_mask(x.shape[1], x.dtype)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.lm_head(x)


def pick_dtype(dtype_name: str) -> mx.Dtype:
    if dtype_name == "float16":
        return mx.float16
    if dtype_name == "bfloat16":
        return mx.bfloat16
    return mx.float32


def count_parameters(model: nn.Module) -> int:
    leaves = mu.tree_flatten(model.parameters())
    return sum(int(v.size) for _, v in leaves)


def benchmark_model(
    model: MlxDraftLlama,
    spec: ModelSpec,
    prompt_tokens: int,
    max_new_tokens: int,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, Any]:
    input_ids = mx.random.randint(0, spec.vocab_size, (1, prompt_tokens))
    mx.eval(input_ids)

    for _ in range(warmup_steps):
        logits = model(input_ids)
        mx.eval(logits)

    t0 = time.perf_counter()
    for _ in range(measure_steps):
        logits = model(input_ids)
        mx.eval(logits)
    forward_elapsed = time.perf_counter() - t0

    # 非 KV-cache 版本的 decode loop：每步重新前向，作為不同尺寸模型的公平相對比較。
    for _ in range(max(1, warmup_steps // 2)):
        cur = input_ids
        for _ in range(max_new_tokens):
            logits = model(cur)
            next_id = mx.argmax(logits[:, -1, :], axis=-1)
            cur = mx.concatenate([cur, mx.expand_dims(next_id, axis=-1)], axis=1)
        mx.eval(cur)

    t1 = time.perf_counter()
    for _ in range(measure_steps):
        cur = input_ids
        for _ in range(max_new_tokens):
            logits = model(cur)
            next_id = mx.argmax(logits[:, -1, :], axis=-1)
            cur = mx.concatenate([cur, mx.expand_dims(next_id, axis=-1)], axis=1)
        mx.eval(cur)
    generate_elapsed = time.perf_counter() - t1

    avg_forward_s = forward_elapsed / measure_steps
    avg_generate_s = generate_elapsed / measure_steps
    generate_toks_per_s = max_new_tokens / avg_generate_s
    total_params = count_parameters(model)

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
    parser = argparse.ArgumentParser(description="Benchmark 15M~60M draft model speed with MLX.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    dtype = pick_dtype(args.dtype)
    print(
        f"backend=mlx dtype={dtype} prompt_tokens={args.prompt_tokens} "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print("note: 這是 MLX 的 Llama-style benchmark，GQA 與 HF generate/KV-cache 行為不完全相同。")

    results: list[dict[str, Any]] = []
    for spec in get_candidate_specs():
        print(f"\n[benchmark] {spec.name} ...")
        model = MlxDraftLlama(spec)
        model.set_dtype(dtype)
        row = benchmark_model(
            model=model,
            spec=spec,
            prompt_tokens=args.prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
        results.append(row)

    ranked = sorted(results, key=lambda r: r["generate_tokens_per_s"], reverse=True)
    print("\n=== Ranked by generate tok/s ===")
    print_table(ranked)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": "mlx",
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
