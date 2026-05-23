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
from mamba3_mlx.utils.system_prompts import MODE_ALIASES, resolve_system_prompt
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint, _sidecar_path
from mamba3_mlx.inference.generator import generate


DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Think step by step before answering."
)


def build_chatml_prompt(tokenizer: Tokenizer, system_prompt: str, user_msg: str,
                        seed_think: bool = True):
    """Wrap user_msg in ChatML, optionally seeding <think>\\n after assistant tag."""
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        + ("<think>\n" if seed_think else "")
    )
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    bos = tokenizer.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids
    return ids, text


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes (--mode) map to SFT-CoT training categories:\n"
            "  emotion, self / self_awareness, email / summarize_email,\n"
            "  movie / movie_intro, daily / daily_conversation,\n"
            "  syscall / system_call, deep / deep_dive\n"
        ),
    )

    # ── Paths ──────────────────────────────────────────────────────────────────
    p.add_argument("--model_path", default=str(REPO_ROOT / "checkpoints" / "latest_sft_cot_model.npz"))
    p.add_argument("--tokenizer_path", default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"))

    # ── Prompt / mode ─────────────────────────────────────────────────────────
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--mode", default=None, choices=sorted(set(MODE_ALIASES.keys())),
                   help="System prompt category. Overrides --system.")
    p.add_argument("--system", default=_DEFAULT_SYSTEM,
                   help="Custom system prompt (ignored when --mode is set).")

    # ── Generation ────────────────────────────────────────────────────────────
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temp",       type=float, default=0.8)
    p.add_argument("--top_k",      type=int,   default=40)
    p.add_argument("--top_p",      type=float, default=0.9)
    p.add_argument("--min_p",      type=float, default=0.05)
    p.add_argument("--rep_pen",    type=float, default=1.1)
    p.add_argument("--pres_pen",   type=float, default=0.0)
    p.add_argument("--freq_pen",   type=float, default=0.02)
    p.add_argument("--repeat_last_n", type=int, default=64)
    p.add_argument("--seed",       type=int,   default=0)

    # ── Stop conditions ───────────────────────────────────────────────────────
    p.add_argument("--no-eos-stop", action="store_true",
                   help="Ignore EOS / <|im_end|> tokens — run until max_tokens "
                        "or a --stop-string match. Useful for spec-decode debugging.")
    p.add_argument("--stop-string", action="append", dest="stop_strings",
                   metavar="TEXT", default=[],
                   help="Stop when TEXT appears in decoded output. "
                        "Repeatable, handles multi-token strings. "
                        "Example: --stop-string '</final>'")

    # ── Hardware / precision ──────────────────────────────────────────────────
    p.add_argument("--dtype",    default="bf16", choices=list(DTYPE_MAP))
    p.add_argument("--kv_dtype", default="auto", choices=["auto"] + list(DTYPE_MAP))
    p.add_argument("--full-decode-compile", action="store_true",
                   help="Compile the entire model graph with mx.compile before decoding. "
                        "Warmup steps run first to pay JIT cost; subsequent steps reuse the graph.")
    p.add_argument("--warmup", type=int, default=3,
                   help="Number of greedy decode steps for JIT warm-up when "
                        "--full-decode-compile is set (default: 3).")

    # ── Output / format ───────────────────────────────────────────────────────
    p.add_argument("--stream", action="store_true",
                   help="Stream tokens to stdout as they are produced.")
    p.add_argument("--show-special", action="store_true",
                   help="Include special tokens (<think>, <|im_end|> …) in stream output.")
    p.add_argument("--no-seed-think", action="store_true",
                   help="Skip pre-seeding <think>\\n after the assistant tag.")
    p.add_argument("--raw-prompt", action="store_true",
                   help="Use --prompt verbatim; skip ChatML wrapping.")

    # ── Speculative (scaffold) ────────────────────────────────────────────────
    p.add_argument("--speculative",   action="store_true")
    p.add_argument("--speculative-k", type=int, default=5)

    return p.parse_args()


def main():
    args = get_args()
    dtype = DTYPE_MAP[args.dtype]

    print(f"[load] tokenizer: {args.tokenizer_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)

    print(f"[load] model:     {args.model_path}", file=sys.stderr)
    cfg   = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)

    # ── checkpoint load (mmap fast-path when sidecar exists) ─────────────────
    t0 = time.time()
    sidecar = _sidecar_path(args.model_path, dtype)
    if sidecar.exists():
        print(f"[load] sidecar:   {sidecar.name}  (mmap instant)", file=sys.stderr)
    else:
        print(f"[load] converting → {sidecar.name}  (one-time, saved for next run)",
              file=sys.stderr)
    load_checkpoint(model, args.model_path, dtype=dtype)
    mx.eval(model.parameters())
    print(f"[load] weights done in {time.time() - t0:.2f}s  (dtype={args.dtype})",
          file=sys.stderr)

    # ── warmup: one dummy forward pass to pre-JIT the decode graph ───────────
    t_w = time.time()
    _dummy = mx.zeros((1, 1), dtype=mx.int32)
    _lo, _st = model(_dummy, states=None)
    mx.eval(_lo)
    del _dummy, _lo, _st
    print(f"[warmup] graph JIT done in {time.time() - t_w:.2f}s", file=sys.stderr)

    # ── System prompt ─────────────────────────────────────────────────────────
    sys_prompt = resolve_system_prompt(args.mode, args.system)
    if args.mode:
        print(f"[mode] {args.mode} → {sys_prompt[:70]}…", file=sys.stderr)

    # ── Prompt ids ────────────────────────────────────────────────────────────
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
    if args.no_eos_stop:
        print("[stop] --no-eos-stop active — EOS tokens will NOT halt generation",
              file=sys.stderr)
    if args.stop_strings:
        print(f"[stop] stop-strings: {args.stop_strings}", file=sys.stderr)
    if args.full_decode_compile:
        print(f"[compile] mx.compile enabled — warmup={args.warmup} steps", file=sys.stderr)

    # ── Generation config ─────────────────────────────────────────────────────
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

    # ── Stop token ids (EOS / im_end) ─────────────────────────────────────────
    stop_ids: list[int] = []
    if not args.no_eos_stop:
        for name in ("<|im_end|>", "</s>"):
            tid = tok.token_to_id(name)
            if tid is not None:
                stop_ids.append(tid)

    # ── Streaming callback ────────────────────────────────────────────────────
    on_token = None
    if args.stream:
        seen: list[int] = []
        prev_len = [0]
        skip_special = not args.show_special

        def _on_token(tid: int) -> None:
            seen.append(tid)
            text_full = tok.decode(seen, skip_special_tokens=skip_special)
            new = text_full[prev_len[0]:]
            prev_len[0] = len(text_full)
            if new:
                print(new, end="", flush=True)

        on_token = _on_token

    # ── Generate ──────────────────────────────────────────────────────────────
    result = generate(
        model, ids, gen_cfg,
        stop_token_ids=stop_ids,
        stop_strings=args.stop_strings or [],
        no_eos_stop=args.no_eos_stop,
        full_decode_compile=args.full_decode_compile,
        warmup_steps=args.warmup,
        tokenizer=tok if args.stop_strings else None,
        on_token=on_token,
    )

    if args.stream:
        print()   # newline after streamed content

    n_timed = len(result.tokens) - result.n_warmup
    warmup_note = f"  warmup={result.n_warmup}" if result.n_warmup else ""
    print(
        f"===== "
        f"prefill {result.prefill_tps:,.0f} tok/s ({result.n_prompt} tok, {result.elapsed_prefill*1000:.0f} ms)  |  "
        f"decode  {result.decode_tps:,.0f} tok/s ({n_timed} tok, {result.elapsed_decode*1000:.0f} ms)"
        f"{warmup_note}  |  stop={result.stop_reason}"
        f" =====",
        file=sys.stderr,
    )

    if not args.stream:
        full_text = tok.decode(result.tokens,
                               skip_special_tokens=not args.show_special)
        print("===== ASSISTANT OUTPUT =====")
        if not args.no_seed_think:
            print("<think>")
        print(full_text)


if __name__ == "__main__":
    main()
