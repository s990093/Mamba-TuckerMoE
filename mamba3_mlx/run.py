"""CLI entry point for MLX Mamba3 inference."""
import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.utils.config import Mamba3Config, GenerationConfig
from mamba3_mlx.utils.system_prompts import SYSTEM_PROMPTS, MODE_ALIASES, resolve_system_prompt
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint
from mamba3_mlx.inference.generator import generate


DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Think step by step before answering."
)


def build_chatml_prompt(tokenizer: Tokenizer, system_prompt: str, user_msg: str,
                        seed_think: bool = True):
    """Build prompt ids ending with `<|im_start|>assistant\\n<think>\\n` if seed_think."""
    parts = []
    parts.append("<|im_start|>system\n" + system_prompt + "<|im_end|>\n")
    parts.append("<|im_start|>user\n" + user_msg + "<|im_end|>\n")
    suffix = "<|im_start|>assistant\n"
    if seed_think:
        suffix += "<think>\n"
    parts.append(suffix)
    text = "".join(parts)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    # Prepend BOS
    bos = tokenizer.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids
    return ids, text


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes (--mode shorthand → system prompt category):\n"
            + "\n".join(
                f"  {alias:<20s} → {key}"
                for alias, key in sorted(MODE_ALIASES.items())
                if alias == key or alias in ("self", "email", "movie", "daily", "syscall", "deep")
            )
        ),
    )
    p.add_argument("--model_path", type=str,
                   default=str(REPO_ROOT / "checkpoints" / "latest_sft_cot_model.npz"))
    p.add_argument("--tokenizer_path", type=str,
                   default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"))
    p.add_argument("--prompt", type=str, default="What is the capital of France?")
    p.add_argument("--mode", type=str, default=None,
                   choices=sorted(set(MODE_ALIASES.keys())),
                   help="Select a training-category system prompt. Overrides --system.")
    p.add_argument("--system", type=str, default=_DEFAULT_SYSTEM,
                   help="Custom system prompt (ignored when --mode is set).")

    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_p", type=float, default=0.05)
    p.add_argument("--rep_pen", type=float, default=1.1)
    p.add_argument("--pres_pen", type=float, default=0.0)
    p.add_argument("--freq_pen", type=float, default=0.02)
    p.add_argument("--repeat_last_n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--dtype", type=str, default="bf16", choices=list(DTYPE_MAP))
    p.add_argument("--kv_dtype", type=str, default="auto",
                   choices=["auto"] + list(DTYPE_MAP))
    p.add_argument("--full-decode-compile", action="store_true")
    p.add_argument("--speculative", action="store_true")
    p.add_argument("--speculative-k", type=int, default=5)

    p.add_argument("--no-seed-think", action="store_true",
                   help="Don't pre-seed '<think>\\n' after assistant tag.")
    p.add_argument("--raw-prompt", action="store_true",
                   help="Treat --prompt as a literal string, skip ChatML wrap.")
    p.add_argument("--stream", action="store_true",
                   help="Stream tokens to stdout as they are produced.")
    return p.parse_args()


def main():
    args = get_args()
    dtype = DTYPE_MAP[args.dtype]

    print(f"[load] tokenizer: {args.tokenizer_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)

    print(f"[load] model:     {args.model_path}", file=sys.stderr)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    t0 = time.time()
    load_checkpoint(model, args.model_path, dtype=dtype)
    mx.eval(model.parameters())
    print(f"[load] done in {time.time() - t0:.2f}s (dtype={args.dtype})", file=sys.stderr)

    # Resolve system prompt (--mode wins over --system)
    sys_prompt = resolve_system_prompt(args.mode, args.system)
    if args.mode:
        print(f"[mode] {args.mode} → {sys_prompt[:60]}…", file=sys.stderr)

    # Build prompt
    if args.raw_prompt:
        prompt_text = args.prompt
        ids = tok.encode(prompt_text, add_special_tokens=False).ids
        bos = tok.token_to_id("<s>")
        if bos is not None:
            ids = [bos] + ids
    else:
        ids, prompt_text = build_chatml_prompt(
            tok, sys_prompt, args.prompt, seed_think=not args.no_seed_think)

    print("===== PROMPT =====", file=sys.stderr)
    print(prompt_text, file=sys.stderr)
    print(f"===== PROMPT TOKENS: {len(ids)} =====", file=sys.stderr)

    gen_cfg = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        rep_pen=args.rep_pen,
        pres_pen=args.pres_pen,
        freq_pen=args.freq_pen,
        repeat_last_n=args.repeat_last_n,
        seed=args.seed,
    )

    stop_ids = []
    im_end = tok.token_to_id("<|im_end|>")
    eos = tok.token_to_id("</s>")
    if im_end is not None:
        stop_ids.append(im_end)
    if eos is not None:
        stop_ids.append(eos)

    on_token = None
    if args.stream:
        seen = []
        last_decoded_len = [0]
        def _on_token(tid):
            seen.append(tid)
            # Decode the running list so SentencePiece word boundaries render correctly.
            text_full = tok.decode(seen, skip_special_tokens=False)
            new = text_full[last_decoded_len[0]:]
            last_decoded_len[0] = len(text_full)
            print(new, end="", flush=True)
        on_token = _on_token

    t0 = time.time()
    out_ids = generate(model, ids, gen_cfg, stop_token_ids=stop_ids, on_token=on_token)
    dt = time.time() - t0
    if args.stream:
        print()
    print(f"===== GENERATED {len(out_ids)} tokens in {dt:.2f}s "
          f"({len(out_ids) / max(dt, 1e-6):.2f} tok/s) =====", file=sys.stderr)

    full_text = tok.decode(out_ids, skip_special_tokens=False)
    if not args.stream:
        print("===== ASSISTANT OUTPUT =====")
        if not args.no_seed_think:
            print("<think>")
        print(full_text)


if __name__ == "__main__":
    main()
