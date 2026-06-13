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
from mamba3_mlx.utils.mode_configs import get_mode_gen_config
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint, _sidecar_path
from mamba3_mlx.inference.generator import generate
from mamba3_mlx.mlx_model.static_decode import StaticDecoder, StaticGenerateResult


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
    # Defaults are None so per-mode configs can fill in; ultimate fallbacks below.
    p.add_argument("--max_tokens", type=int,   default=256)
    p.add_argument("--temp",       type=float, default=None)
    p.add_argument("--top_k",      type=int,   default=None)
    p.add_argument("--top_p",      type=float, default=None)
    p.add_argument("--min_p",      type=float, default=None)
    p.add_argument("--rep_pen",    type=float, default=None)
    p.add_argument("--pres_pen",   type=float, default=None)
    p.add_argument("--freq_pen",   type=float, default=None)
    p.add_argument("--repeat_last_n", type=int, default=128)
    p.add_argument("--seed",       type=int,   default=None)

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

    # ── Static decoder (fast path: 1 dispatch/token, metal_fuse, q8) ─────
    p.add_argument("--static", action="store_true",
                   help="Use StaticDecoder (single compiled graph) instead of "
                        "the per-block reference path.  ~2-3x faster decode.")
    p.add_argument("--metal-fuse", action="store_true",
                   help="Fused Metal kernel for Mamba SSM inner chain "
                        "(value-identical, ~+50% over compiled-graph path).")
    p.add_argument("--quant-moe", type=int, default=0,
                   help="Quantize TuckerMoE U_in/U_out weight matrices to N bits "
                        "(recommended: 8).  Router stays bf16.  +25-30% throughput.")
    p.add_argument("--quant-proj", type=int, default=0,
                   help="Quantize in_proj/dense_proj/qkv/o_proj weights to N bits "
                        "(recommended: 8).  dt/A/lambda tail stays bf16.")
    p.add_argument("--quant-head", type=int, default=0,
                   help="Quantize head projection weight (49 MB) to N bits "
                        "(recommended: 8).")

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

    # ── Generation config — mode defaults then CLI overrides ─────────────────
    # Priority: CLI arg (not None) > mode config > GenerationConfig defaults.
    mc = get_mode_gen_config(args.mode)

    gen_cfg = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temp    if args.temp    is not None else mc.get("temperature", 0.426),
        top_k=     args.top_k   if args.top_k   is not None else mc.get("top_k",       20),
        top_p=     args.top_p   if args.top_p   is not None else mc.get("top_p",        0.981),
        min_p=     args.min_p   if args.min_p   is not None else mc.get("min_p",        0.067),
        rep_pen=   args.rep_pen  if args.rep_pen  is not None else mc.get("rep_pen",    1.146),
        pres_pen=  args.pres_pen if args.pres_pen is not None else mc.get("pres_pen",   0.143),
        freq_pen=  args.freq_pen if args.freq_pen is not None else mc.get("freq_pen",   0.133),
        repeat_last_n=args.repeat_last_n,
        seed=      args.seed    if args.seed    is not None else mc.get("seed",         0),
    )
    if args.mode:
        print(f"[mode-cfg] {args.mode}: temp={gen_cfg.temperature} top_k={gen_cfg.top_k} seed={gen_cfg.seed}", file=sys.stderr)

    # ── Stop token ids (EOS / im_end) ─────────────────────────────────────────
    stop_ids: list[int] = []
    if not args.no_eos_stop:
        for name in ("<|im_end|>", "</s>"):
            tid = tok.token_to_id(name)
            if tid is not None:
                stop_ids.append(tid)

    # ── CoT structural tags always rendered (even without --show-special) ─────
    _STRUCT_TAGS: dict[int, str] = {}
    for _n, _s in [("</think>", "\n</think>\n"), ("<final>", "<final>\n"), ("</final>", "\n</final>")]:
        _t = tok.token_to_id(_n)
        if _t is not None:
            _STRUCT_TAGS[_t] = _s

    # ── Streaming callback ────────────────────────────────────────────────────
    # Sliding-window decode: O(WINDOW) per token instead of O(n).
    # Each call decodes only the last WINDOW tokens, not the full history.
    _STREAM_WINDOW = 32

    on_token = None
    if args.stream:
        seen: list[int] = []
        skip_special = not args.show_special
        _ws   = [0]          # window start index into seen
        _prev = ['']         # decoded text of seen[_ws[0]:-1] (window without latest token)

        def _on_token(tid: int) -> None:
            seen.append(tid)
            n = len(seen)

            if skip_special and tid in _STRUCT_TAGS:
                # Flush pending text before structural tag
                ctx_prev = seen[_ws[0] : -1]
                cur_prev = tok.decode(ctx_prev, skip_special_tokens=True)
                pending  = cur_prev[len(_prev[0]):]
                if pending:
                    print(pending, end="", flush=True)
                print(_STRUCT_TAGS[tid], end="", flush=True)
                # Reset window to start fresh after the tag
                _ws[0]   = n
                _prev[0] = ''
                return

            # Slide window when it grows too large (amortised O(WINDOW))
            if n - _ws[0] > _STREAM_WINDOW * 2:
                new_ws = n - _STREAM_WINDOW
                _ws[0] = new_ws
                _prev[0] = tok.decode(seen[new_ws : -1], skip_special_tokens=skip_special)

            # Decode only the current window (O(WINDOW))
            cur = tok.decode(seen[_ws[0]:], skip_special_tokens=skip_special)
            new = cur[len(_prev[0]):]
            _prev[0] = cur
            if new:
                print(new, end="", flush=True)

        on_token = _on_token

    # ── Generate ──────────────────────────────────────────────────────────────
    if args.static:
        decoder = StaticDecoder(
            model,
            metal_fuse=args.metal_fuse,
            quant_moe_bits=args.quant_moe,
            quant_proj_bits=args.quant_proj,
            quant_head_bits=args.quant_head,
        )
        result = decoder.generate(
            ids, gen_cfg,
            stop_token_ids=stop_ids,
            on_token=on_token,
        )
    else:
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

    n_timed = len(result.tokens)
    compile_note = ""
    if isinstance(result, StaticGenerateResult):
        compile_note = f"  compile={result.compile_s:.1f}s" if result.compile_s else ""
        n_timed = len(result.tokens)
    else:
        n_timed = len(result.tokens) - result.n_warmup
        compile_note = f"  warmup={result.n_warmup}" if result.n_warmup else ""
    print(
        f"===== "
        f"prefill {result.prefill_tps:,.0f} tok/s ({result.n_prompt} tok, {result.elapsed_prefill*1000:.0f} ms)  |  "
        f"decode  {result.decode_tps:,.0f} tok/s ({n_timed} tok, {result.elapsed_decode*1000:.0f} ms)"
        f"{compile_note}  |  stop={result.stop_reason}"
        f" =====",
        file=sys.stderr,
    )

    if not args.stream:
        # Decode with structural CoT tags always visible
        skip_proto = not args.show_special
        parts, seg = [], []
        for _tid in result.tokens:
            if skip_proto and _tid in _STRUCT_TAGS:
                if seg:
                    parts.append(tok.decode(seg, skip_special_tokens=True))
                    seg = []
                parts.append(_STRUCT_TAGS[_tid])
            else:
                seg.append(_tid)
        if seg:
            parts.append(tok.decode(seg, skip_special_tokens=skip_proto))
        full_text = "".join(parts)

        print("===== ASSISTANT OUTPUT =====")
        if not args.no_seed_think:
            print("<think>")
        print(full_text)


if __name__ == "__main__":
    main()
