#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "arnir0__Tiny-LLM"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(model_path: str, tokenizer_path: str | None, resize_token_embeddings: bool, device: torch.device):
    tok_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if resize_token_embeddings and len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()
    return model, tokenizer


def generate(
    prompt: str,
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        )

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def decode_prompt_escapes(prompt: str) -> str:
    return prompt.replace("\\n", "\n").replace("\\t", "\t")


def print_model_summary(model, tokenizer, device: torch.device) -> None:
    cfg = model.config
    num_params = sum(p.numel() for p in model.parameters())
    attention_kind = "causal self-attention"
    if getattr(cfg, "num_key_value_heads", None) and cfg.num_key_value_heads < cfg.num_attention_heads:
        attention_kind += " with grouped-query / multi-query KV heads"

    print("=== Model Summary ===")
    print(f"architecture: {getattr(cfg, 'architectures', ['unknown'])[0]}")
    print(f"model_type: {cfg.model_type}")
    print(f"parameters: {num_params:,}")
    print(f"layers: {cfg.num_hidden_layers}")
    print(f"hidden_size: {cfg.hidden_size}")
    print(f"intermediate_size: {cfg.intermediate_size}")
    print(f"attention: {attention_kind}")
    print(f"attention_heads: {cfg.num_attention_heads}")
    print(f"kv_heads: {getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)}")
    print(f"position_encoding: RoPE")
    print(f"context_length: {cfg.max_position_embeddings}")
    print(f"model_vocab_size: {cfg.vocab_size}")
    print(f"tokenizer_len: {len(tokenizer)}")
    print(f"device: {device}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic PyTorch generation for arnir0/Tiny-LLM.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Local model directory or Hugging Face repo id.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer directory. Defaults to --model.")
    parser.add_argument("--prompt", default="According to all known laws of aviation, there is no way a bee should be able to fly.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8, help="Use 0 for greedy decoding.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, etc.")
    parser.add_argument("--no-decode-escapes", action="store_true", help="Keep literal \\n and \\t in --prompt.")
    parser.add_argument(
        "--resize-token-embeddings",
        action="store_true",
        help="Resize model embeddings when using a tokenizer whose length differs from the checkpoint vocab.",
    )
    parser.add_argument("--summary", action="store_true", help="Print model architecture summary before generation.")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, tokenizer = load(args.model, args.tokenizer, args.resize_token_embeddings, device)

    if args.summary:
        print_model_summary(model, tokenizer, device)

    prompt = args.prompt if args.no_decode_escapes else decode_prompt_escapes(args.prompt)

    text = generate(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()
