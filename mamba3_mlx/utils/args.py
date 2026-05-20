"""
Command-line argument parsing for MLX inference.
All parameters from plan section VIII.
"""

import argparse
from .config import GenerationConfig, ModelConfig


def get_generation_args():
    """Parse generation-related CLI arguments."""
    parser = argparse.ArgumentParser(description="MLX Mamba3 Generation")

    # Model and checkpoint
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (.npz or .pt)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (auto-detect from model dir if not specified)")

    # Generation
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt text")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum tokens to generate")

    # Sampling parameters (section VIII table)
    parser.add_argument("--temp", type=float, default=0.8,
                        help="Temperature (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k filtering (default: 40)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus (top-p) filtering (default: 0.9)")
    parser.add_argument("--min_p", type=float, default=0.05,
                        help="Min-p relative probability threshold (default: 0.05)")
    parser.add_argument("--rep_pen", type=float, default=1.1,
                        help="Repetition penalty coefficient (default: 1.1)")
    parser.add_argument("--pres_pen", type=float, default=0.0,
                        help="Presence penalty (OpenAI style) (default: 0.0)")
    parser.add_argument("--freq_pen", type=float, default=0.02,
                        help="Frequency penalty (OpenAI style) (default: 0.02)")
    parser.add_argument("--repeat_last_n", type=int, default=64,
                        help="Penalty window size (default: 64)")

    # Data types
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "bf16", "fp16"],
                        help="Model weight/compute dtype (default: bf16)")
    parser.add_argument("--kv_dtype", type=str, default="auto",
                        choices=["auto", "bf16", "fp16", "fp32"],
                        help="KV cache dtype; 'auto' matches --dtype (default: auto)")

    # Optimization
    parser.add_argument("--full-decode-compile", action="store_true", default=True,
                        help="Use mx.compile for decode step (default: True)")
    parser.add_argument("--no-full-decode-compile", dest="full_decode_compile",
                        action="store_false",
                        help="Disable mx.compile for decode")

    # Speculative decoding
    parser.add_argument("--speculative", action="store_true",
                        help="Enable speculative decoding (requires draft model)")
    parser.add_argument("--speculative_k", type=int, default=5,
                        help="Number of draft tokens per speculative iteration (default: 5)")
    parser.add_argument("--draft_model_path", type=str, default=None,
                        help="Path to draft model for speculative decoding")

    # Other
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy sampling (highest probability token)")
    parser.add_argument("--eos_token_id", type=int, default=None,
                        help="End-of-sequence token ID")
    parser.add_argument("--verbose", action="store_true",
                        help="Print timing and progress information")

    return parser


def get_benchmark_args():
    """Parse benchmark-related CLI arguments."""
    parser = argparse.ArgumentParser(description="MLX Mamba3 Benchmark")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt for benchmarking")
    parser.add_argument("--num_generate", type=int, default=100,
                        help="Number of tokens to generate (default: 100)")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--verbose", action="store_true")

    return parser


def args_to_generation_config(args):
    """Convert parsed arguments to GenerationConfig."""
    kv_dtype = args.kv_dtype if args.kv_dtype != "auto" else args.dtype

    return GenerationConfig(
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        rep_pen=args.rep_pen,
        pres_pen=args.pres_pen,
        freq_pen=args.freq_pen,
        repeat_last_n=args.repeat_last_n,
        dtype=args.dtype,
        kv_dtype=kv_dtype,
        full_decode_compile=args.full_decode_compile,
        speculative=args.speculative,
        speculative_k=args.speculative_k,
        greedy=args.greedy,
        eos_token_id=args.eos_token_id,
    )


def infer_model_config_from_checkpoint(checkpoint_path):
    """
    Infer model configuration from checkpoint metadata.
    For now, returns default config; should be enhanced to read from checkpoint.
    """
    return ModelConfig()
