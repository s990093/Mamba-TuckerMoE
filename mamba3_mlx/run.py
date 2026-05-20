#!/usr/bin/env python3
"""
Main entry point for MLX Mamba3 inference.
Supports: generation, benchmarking, and speculative decoding.
"""

import json
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
from mlx_model.hybrid_model import Mamba3LanguageModel, Mamba3Config
from inference.generator import AutoregressiveGenerator
from inference.speculative import SpeculativeGenerator
from utils.args import get_generation_args, get_benchmark_args, args_to_generation_config
from utils.config import ModelConfig


class BPETokenizer:
    """BPE tokenizer using transformers/tokenizers library."""

    def __init__(self, tokenizer_path="cot_dataset/tokenizer.json"):
        try:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        except ImportError:
            print("Warning: tokenizers library not found, using fallback")
            self.tokenizer = None
            import json
            with open(tokenizer_path, 'r') as f:
                self.data = json.load(f)
            self.vocab = self.data.get('model', {}).get('vocab', {})
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        self.vocab_size = 32000  # Standard vocab size for this tokenizer

    def encode(self, text):
        """Encode text to token IDs."""
        if self.tokenizer:
            encoding = self.tokenizer.encode(text)
            return encoding.ids
        else:
            # Fallback: simple character encoding
            return [ord(c) % self.vocab_size for c in text]

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        if self.tokenizer:
            # Convert MLX arrays to Python ints
            clean_ids = []
            for tid in token_ids:
                if isinstance(tid, mx.array):
                    clean_ids.append(int(tid))
                else:
                    clean_ids.append(int(tid))
            return self.tokenizer.decode(clean_ids)
        else:
            # Fallback
            return "".join(chr(min(id, 127)) for id in token_ids if id < 127)


def load_model(model_path, dtype="bf16"):
    """
    Load MLX model from checkpoint.

    Args:
        model_path: Path to .npz or .pt checkpoint
        dtype: Data type for weights ("bf16", "fp32", "fp16")

    Returns:
        (model, weights) tuple
    """
    print(f"[Model] Loading from {model_path}...")
    start = time.time()

    # Create config (in practice, load from checkpoint metadata)
    config = Mamba3Config(d_model=768, num_layers=15, vocab_size=32768)

    # Create model
    model = Mamba3LanguageModel(config)

    # Load weights (TODO: implement proper weight loading)
    print(f"[Model] Created with {config.num_layers} layers, d_model={config.d_model}")
    print(f"[Model] Loading took {time.time() - start:.2f}s")

    return model, config


def main_generate():
    """Main generation pipeline."""
    parser = get_generation_args()
    args = parser.parse_args()

    # Load model
    model, config = load_model(args.model_path, args.dtype)

    # Create tokenizer
    tokenizer = BPETokenizer(tokenizer_path="cot_dataset/tokenizer.json")

    # Create generator
    generator = AutoregressiveGenerator(
        model=model,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
    )

    # Generate
    gen_config = args_to_generation_config(args)

    if args.verbose:
        print("\n[Generation Config]")
        print(f"  Temperature: {gen_config.temp}")
        print(f"  Top-k: {gen_config.top_k}")
        print(f"  Top-p: {gen_config.top_p}")
        print(f"  Min-p: {gen_config.min_p}")
        print()

    result = generator.generate(
        prompt_text=args.prompt,
        temperature=gen_config.temp,
        top_k=gen_config.top_k,
        top_p=gen_config.top_p,
        min_p=gen_config.min_p,
        repetition_penalty=gen_config.rep_pen,
        presence_penalty=gen_config.pres_pen,
        frequency_penalty=gen_config.freq_pen,
        repeat_last_n=gen_config.repeat_last_n,
        greedy=gen_config.greedy,
        eos_token_id=gen_config.eos_token_id,
        verbose=args.verbose,
    )

    # Output
    print("\n[Generated Text]")
    print(result["text"])

    if args.verbose:
        print("\n[Performance]")
        print(f"  Prefill latency: {result['prefill_latency']:.3f}s")
        print(f"  Decode latency: {result['decode_latency']:.3f}s")
        print(f"  Decode throughput: {result['decode_throughput']:.1f} tok/s")
        print(f"  Total tokens: {result['total_tokens']}")


def main_benchmark():
    """Main benchmarking pipeline."""
    parser = get_benchmark_args()
    args = parser.parse_args()

    # Load model
    model, config = load_model(args.model_path, args.dtype)

    # Create tokenizer
    tokenizer = DummyTokenizer(vocab_size=config.vocab_size)

    # Create generator
    generator = AutoregressiveGenerator(
        model=model,
        tokenizer=tokenizer,
        max_tokens=args.num_generate,
    )

    # Benchmark
    result = generator.benchmark(
        prompt_text=args.prompt,
        num_generate=args.num_generate,
        temperature=0.8,
        verbose=args.verbose,
    )

    # Output JSON
    print(json.dumps(result, indent=2))


def main():
    """Detect mode and dispatch."""
    if len(sys.argv) < 2:
        print("Usage: python run.py [generate|benchmark] [options]")
        print("\nExamples:")
        print("  python run.py generate --model_path checkpoint.npz --prompt 'Hello'")
        print("  python run.py benchmark --model_path checkpoint.npz --prompt 'Test'")
        sys.exit(1)

    mode = sys.argv[1]
    sys.argv.pop(1)  # Remove mode from argv

    if mode == "generate":
        main_generate()
    elif mode == "benchmark":
        main_benchmark()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
