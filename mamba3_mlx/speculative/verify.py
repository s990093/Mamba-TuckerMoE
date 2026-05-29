"""Correctness harness: Jacobi greedy stream vs. autoregressive greedy stream.

Loads the model once, runs an AR-greedy reference, then runs jacobi_decode
with several K values (both with and without n-gram seeding) and compares
token streams.

* fp32:   Both paths are mathematically identical; the streams must be
          byte-equal.  Mismatches indicate a real bug.
* bf16:   The verify path uses ``chunk_parallel_scan`` (L>1), while the AR
          reference uses the single-step path (L=1).  These differ by ~1e-3
          relative error, so the streams may diverge after enough tokens.
          We report the longest matching prefix and warn rather than fail.

Run::

    python -m mamba3_mlx.speculative.verify --dtype fp32 \\
        --prompt "The quick brown fox" --max_tokens 64

    python -m mamba3_mlx.speculative.verify --dtype bf16 \\
        --prompt "Solve 2x+3=11 step by step" --use_ngram --max_tokens 96
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.inference.generator import _iter_state_arrays, prefill
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint
from mamba3_mlx.speculative.jacobi import jacobi_decode
from mamba3_mlx.utils.config import GenerationConfig, Mamba3Config
from mamba3_mlx.utils.system_prompts import MODE_ALIASES, resolve_system_prompt


DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Think step by step before answering."
)


def _build_chatml_prompt(tokenizer: Tokenizer, system_prompt: str,
                         user_msg: str, seed_think: bool = True):
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


def greedy_autoregressive(model, prompt_ids: list[int],
                          max_tokens: int,
                          stop_set: set[int] | None = None) -> list[int]:
    """Reference: pure greedy AR (L=1 every step) for ``max_tokens`` steps."""
    stop_set = stop_set or set()
    last_logits, state, _ = prefill(model, prompt_ids)
    tokens: list[int] = []
    for _ in range(max_tokens):
        tok_arr = mx.argmax(last_logits).astype(mx.int32)
        mx.eval(tok_arr)
        tok = int(tok_arr.item())
        tokens.append(tok)
        if tok in stop_set:
            break
        if len(tokens) >= max_tokens:
            break
        ids = mx.array([[tok]], dtype=mx.int32)
        logits_out, state = model(ids, states=state)
        last_logits = logits_out[0, -1]
        mx.eval(last_logits, *_iter_state_arrays(state))
    return tokens


def _longest_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        default=str(REPO_ROOT / "checkpoints" / "latest_sft_cot_model.npz"),
    )
    p.add_argument(
        "--tokenizer_path",
        default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"),
    )
    p.add_argument("--prompt", default="Who are you?",
                   help="(Default mirrors Makefile: 'Who are you?')")
    p.add_argument(
        "--mode", default="self_awareness",
        choices=sorted(set(MODE_ALIASES.keys())),
        help="System-prompt category (overrides --system). "
             "Mirrors Makefile default: self_awareness.",
    )
    p.add_argument("--system", default=_DEFAULT_SYSTEM,
                   help="Custom system prompt (ignored when --mode is set).")
    p.add_argument("--no-seed-think", action="store_true",
                   help="Skip pre-seeding <think> after the assistant tag.")
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument(
        "--K", action="append", type=int, default=None,
        help="Jacobi window sizes to test. Repeatable. Default: 4 8 16.",
    )
    p.add_argument("--use_ngram", action="store_true",
                   help="Also test +ngram variants.")
    p.add_argument("--ngram_n", type=int, default=3,
                   help="Total n-gram length; key = n-1 tokens.")
    p.add_argument("--tree_B", action="append", type=int, default=None,
                   help="Tree-of-guesses branch count. Repeatable. Default: 1.")
    p.add_argument("--use_retrieval", action="store_true",
                   help="Enable suffix-retrieval drafter (PLD / N-Grammys "
                        "context source). Composes with n-gram via the "
                        "hybrid multi-source draft builder.")
    p.add_argument("--adaptive_K", action="store_true",
                   help="Enable GammaTune-style adaptive K (EWMA over "
                        "ARL/K_cur ratio).")
    p.add_argument("--K_min", type=int, default=4)
    p.add_argument("--K_max", type=int, default=16)
    p.add_argument("--cot_caches", type=str, default=None,
                   help="Path to a pkl baked by bake_cot_caches: pre-warms "
                        "the lookahead-branch draft source with think + "
                        "final n-grams harvested from the training corpus. "
                        "Phase is auto-tracked via </think>/<final> "
                        "marker tokens.")
    p.add_argument("--cot_bucket", type=str, default=None,
                   help="(v2 cot_caches only) bucket key whose per-category "
                        "caches/retriever should drive the cot_ngram + "
                        "cot_retriever slots.  Defaults to the bundle's "
                        "baked default_bucket.")
    p.add_argument("--dtype", default="fp32", choices=list(DTYPE_MAP),
                   help="fp32 = strict byte-equal; bf16 = soft (warn on drift).")
    p.add_argument("--no-eos-stop", action="store_true",
                   help="Ignore EOS so we always generate max_tokens. "
                        "Strongly recommended for verification.")
    p.add_argument("--raw-prompt", action="store_true",
                   help="Use --prompt verbatim (skip ChatML wrapping).")
    args = p.parse_args()

    dtype = DTYPE_MAP[args.dtype]
    Ks = sorted(set(args.K or [4, 8, 16]))
    strict = (args.dtype == "fp32")

    print(f"[verify] dtype={args.dtype}  strict={strict}", file=sys.stderr)
    print(f"[verify] loading {args.model_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    load_checkpoint(model, args.model_path, dtype=dtype)
    mx.eval(model.parameters())

    sys_prompt = resolve_system_prompt(args.mode, args.system)
    print(f"[verify] mode={args.mode} sys[:60]={sys_prompt[:60]!r}",
          file=sys.stderr)

    if args.raw_prompt:
        ids = tok.encode(args.prompt, add_special_tokens=False).ids
        bos = tok.token_to_id("<s>")
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
    else:
        ids, _ = _build_chatml_prompt(
            tok, sys_prompt, args.prompt,
            seed_think=not args.no_seed_think,
        )

    # Stop tokens (default: respect EOS; --no-eos-stop disables)
    stop_set: set[int] = set()
    if not args.no_eos_stop:
        for name in ("<|im_end|>", "</s>"):
            tid = tok.token_to_id(name)
            if tid is not None:
                stop_set.add(tid)
    print(f"[verify] prompt_tokens={len(ids)}  max_tokens={args.max_tokens}  "
          f"stop_set={sorted(stop_set) or 'none'}",
          file=sys.stderr)

    # ── Reference: AR greedy ────────────────────────────────────────────────
    t0 = time.perf_counter()
    ref = greedy_autoregressive(model, ids, args.max_tokens, stop_set=stop_set)
    t_ref = time.perf_counter() - t0
    ar_tps = len(ref) / max(t_ref, 1e-6)
    print(f"[verify] AR greedy:  {len(ref):4d} tok  {t_ref*1000:.0f} ms  "
          f"{ar_tps:6.1f} tok/s", file=sys.stderr)

    # ── Test Jacobi configurations ──────────────────────────────────────────
    gen_cfg = GenerationConfig(max_tokens=args.max_tokens)
    tree_Bs = sorted(set(args.tree_B or [1]))
    configs: list[tuple[int, bool, int]] = []
    for K in Ks:
        for tb in tree_Bs:
            configs.append((K, False, tb))
            if args.use_ngram:
                configs.append((K, True, tb))

    all_ok = True
    for (K, ngram, tree_B) in configs:
        # Warmup the K-token graph
        dummy = mx.zeros((1, K), dtype=mx.int32)
        _l, _s = model(dummy, states=None)
        mx.eval(_l)
        del _l, _s, dummy

        t0 = time.perf_counter()
        r = jacobi_decode(
            model, ids, gen_cfg, K=K, use_ngram=ngram,
            ngram_n=args.ngram_n,
            tree_B=tree_B,
            use_retrieval=args.use_retrieval,
            adaptive_K=args.adaptive_K,
            K_min=args.K_min, K_max=args.K_max,
            cot_caches=args.cot_caches,
            cot_bucket=args.cot_bucket,
            stop_token_ids=sorted(stop_set),
            no_eos_stop=args.no_eos_stop,
        )
        dt = time.perf_counter() - t0

        prefix_len = _longest_prefix(ref, r.tokens)
        bytewise_equal = (ref == r.tokens)
        speedup = (len(r.tokens) / max(r.elapsed_decode, 1e-6)) / max(ar_tps, 1e-6)

        if bytewise_equal:
            mark = "OK "
        elif prefix_len == min(len(ref), len(r.tokens)):
            mark = "OK*"  # one is a prefix of the other (lengths differ due to stop)
        elif (not strict) and prefix_len >= max(16, len(ref) // 2):
            mark = "WARN"  # bf16 drift after a healthy prefix
        else:
            mark = "BAD"
            all_ok = False

        branch_tag = (
            f"branch_wins={r.branch_wins}" if r.tree_B > 1 else ""
        )
        K_tag = ""
        if args.adaptive_K and r.K_history:
            k_lo, k_hi = min(r.K_history), max(r.K_history)
            k_mean = sum(r.K_history) / len(r.K_history)
            K_tag = f"K_range=[{k_lo}-{k_hi}] mean={k_mean:.1f}"
        feat = []
        if ngram:
            feat.append("ng")
        if args.use_retrieval:
            feat.append("rt")
        if args.adaptive_K:
            feat.append("aK")
        if args.cot_caches:
            feat.append("cot")
        feat_tag = "+".join(feat) if feat else "-"
        print(
            f"[verify] K={K:3d} {feat_tag} tree_B={tree_B} "
            f"dtype={args.dtype}: "
            f"{mark}  prefix={prefix_len:3d}/{len(ref)} "
            f"got={len(r.tokens):3d} arl={r.arl:5.2f} rounds={r.n_rounds:3d} "
            f"hits={r.n_ngram_hits:3d} "
            f"decode={r.decode_tps:6.1f} tok/s  "
            f"speedup={speedup:4.2f}x  {branch_tag} {K_tag}",
            file=sys.stderr,
        )

        if mark in ("BAD", "WARN") and prefix_len < len(ref) and prefix_len < len(r.tokens):
            i = prefix_len
            ctx_a = max(0, i - 4)
            print(
                f"           first mismatch @ idx {i}: "
                f"ref={ref[i]} got={r.tokens[i]}",
                file=sys.stderr,
            )
            print(
                f"           ref[{ctx_a}:{i+4}] = {ref[ctx_a:i+4]}",
                file=sys.stderr,
            )
            print(
                f"           got[{ctx_a}:{i+4}] = {r.tokens[ctx_a:i+4]}",
                file=sys.stderr,
            )

    summary = "ALL OK" if all_ok else "FAILURES (see above)"
    print(f"[verify] {summary}", file=sys.stderr)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
