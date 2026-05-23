"""run_cot.py — CoT middleware demo entry point.

Wraps the existing Mamba3 MLX inference path with the format-guard + CoT
splitter + reasoning-budget watchdog + multi-stage `<final>` injection
middleware from `mamba3_mlx/mv/`. **Read-only** with respect to the
existing pipeline: `run.py`, `inference/generator.py`, and the modules
under `mv/` are not modified.

How it differs from `run.py`
----------------------------
* Logits are biased per step via `CotMiddleware.transform_logits` before
  sampling (ban + close-bias ramp + final-min guard).
* Each sampled token is fed to `CotMiddleware.step` so the splitter
  routes text into "reasoning" / "final" streams.
* When the splitter exits `<think>`, the middleware runs a one-shot
  `<final>\n` continuation prefill to commit the structural transition
  to the model's caches (`maybe_inject_final`).
* The reasoning budget watchdog can force a stop with a synthetic notice.

Usage
-----
    python -m mamba3_mlx.run_cot --prompt "..."  --mode self
    sh mamba3_mlx/run_cot.sh
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# `cot_middleware` imports `cot_format_fsm` as a top-level module; expose mv/
# on sys.path without modifying either file.
sys.path.insert(0, str(Path(__file__).resolve().parent / "mv"))

from mamba3_mlx.utils.config import Mamba3Config, GenerationConfig
from mamba3_mlx.utils.system_prompts import MODE_ALIASES, resolve_system_prompt
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint
from mamba3_mlx.inference.generator import prefill
from mamba3_mlx.inference.sampler import (
    sample_logits,
    apply_repetition_penalty,
    apply_freq_presence_penalty,
)

from cot_middleware import (  # noqa: E402  (sys.path tweak above)
    CotMiddleware,
    CotMiddlewareConfig,
    CotMiddlewareDeps,
    render_health_line,
)


DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Think step by step before answering."
)


# ---------------------------------------------------------------------------
# Tokenizer shim — adapt tokenizers.Tokenizer (Rust HF) to the
# transformers-style API the middleware expects.
# ---------------------------------------------------------------------------
class _RustTokenizerHFShim:
    """Expose a transformers-tokenizer-like surface on a Rust ``Tokenizer``.

    Only the methods the middleware actually calls are implemented; everything
    else raises ``AttributeError`` via normal Python lookup.
    """

    def __init__(self, tok: Tokenizer):
        self._tok = tok
        specials: set[int] = set()
        for name in (
            "<s>", "</s>", "<unk>", "<pad>",
            "<|im_start|>", "<|im_end|>",
            "<think>", "</think>", "<final>", "</final>",
        ):
            tid = tok.token_to_id(name)
            if tid is not None and tid >= 0:
                specials.add(tid)
        self.all_special_ids = sorted(specials)
        self.unk_token_id = tok.token_to_id("<unk>")
        self.eos_token_id = tok.token_to_id("</s>")
        self.bos_token_id = tok.token_to_id("<s>")

    # The middleware tries this first for single-token resolution.
    def convert_tokens_to_ids(self, text: str) -> int:
        tid = self._tok.token_to_id(text)
        return -1 if tid is None else int(tid)

    def convert_ids_to_tokens(self, tid: int) -> str:
        tok = self._tok.id_to_token(int(tid))
        return tok if tok is not None else ""

    def encode(self, text: str, add_special_tokens: bool = False):
        enc = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return list(enc.ids)

    def decode(
        self,
        ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="CoT middleware demo (does not modify existing pipeline).",
    )
    # Paths
    p.add_argument("--model_path",
                   default=str(REPO_ROOT / "checkpoints" / "latest_sft_cot_model.npz"))
    p.add_argument("--tokenizer_path",
                   default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"))
    # Prompt
    p.add_argument("--prompt", default="What is the capital of France?")
    p.add_argument("--mode", default=None, choices=sorted(set(MODE_ALIASES.keys())))
    p.add_argument("--system", default=_DEFAULT_SYSTEM)
    p.add_argument("--no-seed-think", action="store_true",
                   help="Don't pre-seed <think>\\n after the assistant tag.")
    p.add_argument("--raw-prompt", action="store_true")
    # Generation
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temp",       type=float, default=0.8)
    p.add_argument("--top_k",      type=int,   default=40)
    p.add_argument("--top_p",      type=float, default=0.9)
    p.add_argument("--min_p",      type=float, default=0.05)
    p.add_argument("--rep_pen",    type=float, default=1.1)
    p.add_argument("--pres_pen",   type=float, default=0.0)
    p.add_argument("--freq_pen",   type=float, default=0.02)
    p.add_argument("--repeat_last_n", type=int, default=64)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--dtype",      default="bf16", choices=list(DTYPE_MAP))
    # Middleware knobs (mirrors CotMiddlewareConfig.config_from_args)
    p.add_argument("--format-guard",        dest="format_guard",
                   action="store_true", default=True)
    p.add_argument("--no-format-guard",     dest="format_guard",
                   action="store_false")
    p.add_argument("--ban-im-start",        dest="ban_im_start",
                   action="store_true", default=True)
    p.add_argument("--close-bias",          dest="close_bias",
                   type=float, default=4.0)
    p.add_argument("--close-bias-max",      dest="close_bias_max",
                   type=float, default=16.0)
    p.add_argument("--close-bias-start",    dest="close_bias_start",
                   type=int, default=0,
                   help="0 = auto (reasoning_budget // 2).")
    p.add_argument("--reasoning-budget",    dest="reasoning_budget",
                   type=int, default=0,
                   help="0 = no cap; model decides when to close <think>.")
    p.add_argument("--force-final-inject",  dest="force_final_inject",
                   action="store_true", default=True)
    p.add_argument("--no-force-final-inject", dest="force_final_inject",
                   action="store_false")
    p.add_argument("--final-min-tokens",    dest="final_min_tokens",
                   type=int, default=16)
    return p.parse_args()


def build_chatml_prompt(tok: Tokenizer, system_prompt: str, user_msg: str,
                        seed_think: bool):
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        + ("<think>\n" if seed_think else "")
    )
    ids = tok.encode(text, add_special_tokens=False).ids
    bos = tok.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids
    return ids, text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()
    dtype = DTYPE_MAP[args.dtype]

    print(f"[load] tokenizer: {args.tokenizer_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)
    tok_shim = _RustTokenizerHFShim(tok)

    print(f"[load] model:     {args.model_path}", file=sys.stderr)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    t0 = time.time()
    load_checkpoint(model, args.model_path, dtype=dtype)
    mx.eval(model.parameters())
    print(f"[load] done in {time.time() - t0:.2f}s  (dtype={args.dtype})", file=sys.stderr)

    # System prompt + ChatML wrap
    sys_prompt = resolve_system_prompt(args.mode, args.system)
    if args.mode:
        print(f"[mode] {args.mode} → {sys_prompt[:70]}…", file=sys.stderr)

    seed_think = not args.no_seed_think
    if args.raw_prompt:
        prompt_text = args.prompt
        ids = tok.encode(prompt_text, add_special_tokens=False).ids
        bos = tok.token_to_id("<s>")
        if bos is not None:
            ids = [bos] + ids
    else:
        ids, prompt_text = build_chatml_prompt(tok, sys_prompt, args.prompt, seed_think)

    print("===== PROMPT =====", file=sys.stderr)
    print(prompt_text, file=sys.stderr)
    print(f"===== PROMPT TOKENS: {len(ids)} =====", file=sys.stderr)

    # Build middleware
    mw_cfg = CotMiddlewareConfig.config_from_args(args)

    # Existing stop ids (mirrors run.py logic — EOS + <|im_end|>).
    existing_stop_ids = []
    for name in ("<|im_end|>", "</s>"):
        tid = tok.token_to_id(name)
        if tid is not None:
            existing_stop_ids.append(tid)

    deps = CotMiddlewareDeps.build(
        tokenizer=tok_shim,
        vocab_size=cfg.vocab_size,
        existing_stop_ids=existing_stop_ids,
        cfg=mw_cfg,
    )
    print(f"[mw]   {deps.describe()}", file=sys.stderr)

    # Adapter: middleware expects (x_ids[1,N], caches, seq_pos) -> (logits[1,N,V], caches')
    # The Mamba3 model carries its own per-block state; seq_pos is unused.
    def model_apply(x_ids: mx.array, caches: Any, seq_pos: mx.array):
        logits, new_states = model(x_ids, states=caches)
        return logits, new_states

    mw = CotMiddleware(
        deps=deps,
        cfg=mw_cfg,
        reasoning=seed_think,
        model_apply=model_apply,
    )

    # The middleware's stop_ids include </final> (single-token close), but
    # it's only logit-banned in `final` mode.  If the model jumps to </final>
    # from inside <think>/<between>, the decode loop breaks before <final>
    # is ever entered — leaving final_injected=False.  Pre-build an extra
    # ban mask for non-final modes here so the script can keep the
    # middleware unmodified.
    def _build_mask(names: list[str]) -> mx.array | None:
        ids = [t for n in names
               if (t := tok.token_to_id(n)) is not None and t >= 0]
        if not ids or not mw_cfg.enabled:
            return None
        m = mx.zeros((cfg.vocab_size,), dtype=mx.float32)
        for t in ids:
            m[t] = -1e9
        mx.eval(m)
        return m

    # think/between: ban </final> so model can't exit loop before entering final mode.
    extra_ban_mask = _build_mask(["</final>"])
    # final: ban <think>/<final>/</think> so model can't restart CoT structure
    # after the middleware-injected <final> (prevents the **<think><final> re-entry loop).
    final_ban_mask = _build_mask(["<think>", "<final>", "</think>"])
    if extra_ban_mask is not None:
        print("[script] ban[think/between]: </final>", file=sys.stderr)
    if final_ban_mask is not None:
        print("[script] ban[final]:         <think> <final> </think>", file=sys.stderr)

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

    # Prefill
    last_logits, states, elapsed_prefill = prefill(model, ids)
    prefill_tps = len(ids) / max(elapsed_prefill, 1e-6)
    pos = len(ids)

    # Decode loop
    key = mx.random.key(gen_cfg.seed)
    window = max(1, gen_cfg.repeat_last_n)
    generated: list[int] = []
    t_decode = time.perf_counter()
    elapsed_s_fn = lambda: time.perf_counter() - t_decode  # noqa: E731

    reasoning_emitted = 0  # bytes of reasoning already echoed
    stop_reason = "max_tokens"

    print("===== STREAM =====")
    for step_i in range(gen_cfg.max_tokens):
        # 1) Middleware logit transform (+ script-level extra ban so the
        #    model can't sample </final> from think/between modes).
        row = mw.transform_logits(last_logits)
        if extra_ban_mask is not None and mw.mode in ("think", "between"):
            row = row + extra_ban_mask.astype(row.dtype)
        if final_ban_mask is not None and mw.mode == "final":
            row = row + final_ban_mask.astype(row.dtype)
        z = row.astype(mx.float32)
        recent = (list(ids) + generated)[-window:]
        z = apply_repetition_penalty(z, recent, gen_cfg.rep_pen)
        z = apply_freq_presence_penalty(z, recent, gen_cfg.pres_pen, gen_cfg.freq_pen)

        tok_arr, key = sample_logits(
            z, gen_cfg.temperature, gen_cfg.top_k, gen_cfg.top_p, gen_cfg.min_p, key,
        )
        mx.eval(tok_arr)
        tid = int(tok_arr.item())
        generated.append(tid)
        n_out = len(generated)

        # 2) Middleware step → reasoning / final / stop events.
        stop_now = False
        for evt in mw.step(tid, n_out=n_out, elapsed_s_fn=elapsed_s_fn):
            if evt.get("__stop__"):
                stop_now = True
                continue
            if evt["type"] == "reasoning":
                md = evt["markdown"]
                delta = md[reasoning_emitted:]
                reasoning_emitted = len(md)
                if delta:
                    sys.stdout.write(f"\033[2m{delta}\033[0m")
                    sys.stdout.flush()
            elif evt["type"] == "token":
                sys.stdout.write(evt["text"])
                sys.stdout.flush()

        # 3) Stop checks: EOS-ish id OR splitter signaled __stop__.
        if mw.should_break(tid):
            stop_reason = "eos"
            break
        if stop_now:
            stop_reason = "splitter_stop"
            break

        # 4) Advance model one token.
        x = mx.array([[tid]], dtype=mx.int32)
        logits_out, states = model(x, states=states)
        last_logits = logits_out[0, -1]
        pos += 1
        mx.eval(last_logits)

        # 5) Multi-stage <final> injection (at most once per turn).
        states, pos, injected_row, did_inject, _ms = mw.maybe_inject_final(
            caches=states, pos=pos,
        )
        if did_inject and injected_row is not None:
            last_logits = injected_row
            mx.eval(last_logits)

    # Flush splitter tail.
    for evt in mw.flush(n_out=len(generated), elapsed_s_fn=elapsed_s_fn):
        if evt["type"] == "reasoning":
            md = evt["markdown"]
            delta = md[reasoning_emitted:]
            reasoning_emitted = len(md)
            if delta:
                sys.stdout.write(f"\033[2m{delta}\033[0m")
                sys.stdout.flush()
        elif evt["type"] == "token":
            sys.stdout.write(evt["text"])
            sys.stdout.flush()
    print()  # newline after stream

    elapsed_decode = time.perf_counter() - t_decode
    decode_tps = len(generated) / max(elapsed_decode, 1e-6)

    print(
        f"===== prefill {prefill_tps:,.0f} tok/s ({len(ids)} tok, "
        f"{elapsed_prefill*1000:.0f} ms)  |  decode  {decode_tps:,.0f} tok/s "
        f"({len(generated)} tok, {elapsed_decode*1000:.0f} ms)  |  stop={stop_reason} =====",
        file=sys.stderr,
    )
    print(f"[mw]   {render_health_line(mw.health_report())}", file=sys.stderr)


if __name__ == "__main__":
    main()
