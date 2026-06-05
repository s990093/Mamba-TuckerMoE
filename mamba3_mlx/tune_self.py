#!/usr/bin/env python3
"""Hyperparameter tuner for mamba3_mlx --mode self_awareness (make self).

Random search over sampling / repetition parameters. Evaluates output quality
accounting for CoT structure (<think> / <final> blocks), English coherence,
and diversity. Reports best configs and individually verifies top-3.

Usage:
  cd mamba3_mlx && .venv/bin/python3 tune_self.py                  # 96 trials
  python tune_self.py --trials 128 --top-n 5 --prompt "Who are you?"
  python tune_self.py --verbose-top 3 --output results.json
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.utils.config import Mamba3Config, GenerationConfig
from mamba3_mlx.utils.system_prompts import resolve_system_prompt
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint, _sidecar_path
from mamba3_mlx.inference.generator import generate

DEFAULT_MODEL     = str(REPO_ROOT / "checkpoints" / "v2" / "latest_sft_cot_model.npz")
DEFAULT_TOKENIZER = str(REPO_ROOT / "cot_dataset" / "tokenizer.json")
DEFAULT_PROMPT    = "Who are you?"

PARAM_SPACE = {
    "temperature":   (0.20, 0.70),
    "top_k":         [20, 30, 40, 50, 60, 80],
    "top_p":         (0.70, 0.98),
    "min_p":         (0.02, 0.15),
    "rep_pen":       (1.0,  1.45),
    "pres_pen":      (0.0,  0.45),
    "freq_pen":      (0.0,  0.18),
}

FIXED_REPEAT_LAST_N = 128
DEFAULT_MAX_TOKENS  = 300
DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}

# ── Param sampler ─────────────────────────────────────────────────────────

def sample_params(rng: random.Random) -> dict:
    params = {}
    for name, space in PARAM_SPACE.items():
        if isinstance(space, list):
            params[name] = rng.choice(space)
        else:
            lo, hi = space
            params[name] = round(rng.uniform(lo, hi), 3)
    return params


# ── CoT block parser ───────────────────────────────────────────────────────

_COT_TAGS = ("<think>", "</think>", "<final>", "</final>", "<|im_end|>", "</s>")


def parse_cot_blocks(raw_text: str) -> tuple[str, str, bool, bool]:
    """Parse <think>...</think> and <final>...</final> from raw decoded text.

    Returns: (think_body, final_body, has_think, has_final)
    """
    has_think = False
    has_final = False
    think_body = ""
    final_body = ""

    m_think = re.search(r"<think>\s*(.*?)\s*</think>", raw_text, re.DOTALL)
    if m_think:
        has_think = True
        think_body = m_think.group(1).strip()

    m_final = re.search(r"<final>\s*(.*?)\s*</final>", raw_text, re.DOTALL)
    if m_final:
        has_final = True
        final_body = m_final.group(1).strip()
    else:
        # No <final> block: check for text after </think> (non-CoT output)
        tail = raw_text
        if m_think:
            tail = raw_text[m_think.end():]
        tail = tail.strip()
        for tag in _COT_TAGS:
            tail = tail.replace(tag, " ")
        tail = tail.strip()
        if tail:
            final_body = tail

    return think_body, final_body, has_think, has_final


# ── English-likeness ───────────────────────────────────────────────────────

_VOWELS: set[str] = set("aeiouyAEIOUY")
_CONSONANTS: set[str] = set("bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ")


def _is_likely_english_word(w: str) -> bool:
    """Heuristic: a word looks like English if it has vowels and no insane consonant runs."""
    if not w or len(w) == 0:
        return False
    # Pure punctuation / symbols
    if re.fullmatch(r"[^\w]+", w):
        return True  # standalone punctuation is fine
    # Must have at least one letter
    letters = [c for c in w if c.isalpha()]
    if not letters:
        return any(c.isdigit() for c in w)  # digits OK but suspicious
    # Check for at least one vowel
    has_vowel = any(c in _VOWELS for c in letters)
    if not has_vowel:
        return False
    # No run of >= 5 consonants without a vowel
    max_run = 0
    run = 0
    for c in letters:
        if c in _CONSONANTS:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    if max_run >= 5:
        return False
    # All-uppercase > 4 chars is suspicious
    if len(w) > 4 and w.isupper():
        return False
    return True


def english_word_ratio(text: str) -> float:
    """Fraction of words that look like plausible English."""
    words = [w.strip(",.!?;:\"'()[]{}*#") for w in text.split() if w.strip()]
    if not words:
        return 0.0
    ok = sum(1 for w in words if _is_likely_english_word(w))
    return ok / len(words)


# ── Text quality scoring ──────────────────────────────────────────────────

def _words(text: str) -> list[str]:
    return [w for w in text.split() if w.strip()]


def _n_grams(words: list[str], n: int) -> list[tuple]:
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def rep_rate(text: str, n: int = 3) -> float:
    words = _words(text)
    if len(words) < n:
        return 0.0
    counts = Counter(_n_grams(words, n))
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / max(len(counts), 1)


def distinct_n(text: str, n: int = 2) -> float:
    words = _words(text)
    if len(words) < n:
        return 1.0
    ngrams = _n_grams(words, n)
    return len(set(ngrams)) / max(len(ngrams), 1)


def max_dup_span(text: str, max_window: int = 16) -> int:
    words = _words(text)
    seen: dict[tuple, int] = {}
    best = 0
    for i in range(len(words)):
        for j in range(i + 2, min(i + max_window, len(words) + 1)):
            seg = tuple(words[i:j])
            prev = seen.get(seg)
            if prev is not None and prev <= i - len(seg):
                best = max(best, len(seg))
            seen[seg] = i
    return best


def score_cot_output(raw_text: str) -> dict:
    """Score CoT-formatted output with structure + coherence checks.

    Weights:
      25% structural validity  (<think> + <final> present)
      40% final-text quality   (coherence, English-likeness, non-repetitive)
      20% diversity            (n-gram variety in final block)
      15% length sanity

    Returns dict with component scores and composite.
    """
    think_body, final_body, has_think, has_final = parse_cot_blocks(raw_text)

    # ── Structural score ───────────────────────────────────────────────
    struct = 0.0
    if has_think and has_final:
        struct = 1.0
    elif has_final:
        struct = 0.8
    elif has_think:
        struct = 0.5
    else:
        struct = 0.2

    # ── Final-body quality ─────────────────────────────────────────────
    eval_text = final_body if final_body else raw_text
    words = _words(eval_text)
    nw = len(words)

    if nw < 5:
        return {
            "n_words_think": len(_words(think_body)),
            "n_words_final": nw,
            "has_think": has_think, "has_final": has_final,
            "struct_score": struct,
            "english_ratio": 0.0,
            "rep_rate_3": 0.0, "distinct_2": 0.0,
            "max_dup": 0,
            "composite": max(0.1, struct * 0.3),
        }

    r3 = rep_rate(eval_text, 3)
    r4 = rep_rate(eval_text, 4)
    d2 = distinct_n(eval_text, 2)
    dup = max_dup_span(eval_text)
    eng_ratio = english_word_ratio(eval_text)

    # Anti-repetition (0-1)
    anti_rep = 1.0 - min(r3 * 0.6 + r4 * 0.4, 1.0)
    if dup >= 4:
        anti_rep *= max(0.1, 1.0 - (dup - 3) * 0.25)

    # English coherence (0-1): heavy penalty for gibberish
    # Map eng_ratio [0,1] → strongly nonlinear: <0.5 is near-zero
    if eng_ratio < 0.3:
        eng_score = 0.0
    elif eng_ratio < 0.6:
        eng_score = (eng_ratio - 0.3) / 0.3 * 0.5
    elif eng_ratio < 0.85:
        eng_score = 0.5 + (eng_ratio - 0.6) / 0.25 * 0.3
    else:
        eng_score = 0.8 + (eng_ratio - 0.85) / 0.15 * 0.2

    # Diversity
    div_score = d2

    # Length sanity
    if nw >= 80:
        len_ok = 1.0
    elif nw >= 30:
        len_ok = 0.5 + 0.5 * (nw - 30) / 50.0
    else:
        len_ok = 0.5 * nw / 30.0

    quality = anti_rep * 0.30 + eng_score * 0.40 + div_score * 0.20 + len_ok * 0.10
    composite = struct * 0.25 + quality * 0.75

    return {
        "n_words_think": len(_words(think_body)),
        "n_words_final": nw,
        "has_think": has_think, "has_final": has_final,
        "struct_score":     round(struct, 4),
        "english_ratio":    round(eng_ratio, 4),
        "eng_score":        round(eng_score, 4),
        "rep_rate_3":       round(r3, 4),
        "distinct_2":       round(d2, 4),
        "max_dup":          dup,
        "composite":        round(composite, 4),
    }


# ── Model loading ─────────────────────────────────────────────────────────

def load_everything(model_path: str, tokenizer_path: str, dtype_str: str):
    dtype = DTYPE_MAP.get(dtype_str, mx.bfloat16)
    print(f"[load] tokenizer: {tokenizer_path}", file=sys.stderr)
    tok = Tokenizer.from_file(tokenizer_path)
    print(f"[load] model:     {model_path}", file=sys.stderr)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    t0 = time.time()
    load_checkpoint(model, model_path, dtype=dtype)
    mx.eval(model.parameters())
    print(f"[load] weights done in {time.time() - t0:.2f}s  (dtype={dtype_str})",
          file=sys.stderr)
    _dummy = mx.zeros((1, 1), dtype=mx.int32)
    _lo, _st = model(_dummy, states=None)
    mx.eval(_lo)
    del _dummy, _lo, _st
    return model, tok


def build_prompt_ids(tok: Tokenizer, system_prompt: str, user_msg: str,
                     seed_think: bool = True) -> tuple[list[int], str]:
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


# ── Single trial ──────────────────────────────────────────────────────────

def run_one(model, tok, params: dict, prompt_ids: list[int],
            max_tokens: int, stop_ids: list[int]) -> str:
    gen_cfg = GenerationConfig(
        max_tokens=max_tokens,
        temperature=params["temperature"],
        top_k=params["top_k"],
        top_p=params["top_p"],
        min_p=params["min_p"],
        rep_pen=params["rep_pen"],
        pres_pen=params["pres_pen"],
        freq_pen=params["freq_pen"],
        repeat_last_n=FIXED_REPEAT_LAST_N,
        seed=params["_seed"],
    )
    result = generate(model, prompt_ids, gen_cfg, stop_token_ids=stop_ids)
    return tok.decode(result.tokens, skip_special_tokens=False).strip()


# ── Verbose single-run display ────────────────────────────────────────────

def show_trial(model, tok, prompt_ids, params, max_tokens, stop_ids,
               label: str = ""):
    """Run one trial with streaming-like display, print full output."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  TEMP={params['temperature']:.3f}  TOP_K={params['top_k']}  "
          f"TOP_P={params['top_p']:.3f}  MIN_P={params['min_p']:.3f}")
    print(f"  REP_PEN={params['rep_pen']:.3f}  PRES_PEN={params['pres_pen']:.3f}  "
          f"FREQ_PEN={params['freq_pen']:.3f}")
    print(f"{'─'*60}")

    t0 = time.time()
    try:
        text = run_one(model, tok, params, prompt_ids, max_tokens, stop_ids)
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return
    elapsed = time.time() - t0

    scores = score_cot_output(text)
    print(text)
    print(f"\n  score={scores['composite']:.4f}  "
          f"struct={scores['struct_score']:.2f}  "
          f"eng={scores['english_ratio']:.2f}  "
          f"think={scores['has_think']}  final={scores['has_final']}  "
          f"{elapsed:.1f}s")
    return text, scores


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Hyperparameter tuner for make self (self_awareness mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-path",    default=DEFAULT_MODEL)
    p.add_argument("--tokenizer-path",default=DEFAULT_TOKENIZER)
    p.add_argument("--dtype",         default="bf16", choices=["bf16","fp16","fp32"])
    p.add_argument("--prompt",        default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens",    type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--trials",        type=int, default=96)
    p.add_argument("--top-n",         type=int, default=5)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--output",        type=str, default=None)
    p.add_argument("--verbose-top",   type=int, default=3,
                   help="Re-run top-N configs with full output display")
    p.add_argument("--no-seed-think", action="store_true")
    args = p.parse_args()

    # ── Checkpoint ──────────────────────────────────────────────────────
    model_path = args.model_path
    sidecar = _sidecar_path(model_path, DTYPE_MAP.get(args.dtype, mx.bfloat16))
    if sidecar.exists():
        model_path = str(sidecar)
    print(f"[ckpt] {model_path}", file=sys.stderr)

    # ── Load ────────────────────────────────────────────────────────────
    model, tok = load_everything(model_path, args.tokenizer_path, args.dtype)

    sys_prompt = resolve_system_prompt("self_awareness", "")
    prompt_ids, prompt_text = build_prompt_ids(
        tok, sys_prompt, args.prompt, seed_think=not args.no_seed_think)

    print(f"\n[mode] self_awareness", file=sys.stderr)
    print(f"[user] {args.prompt}", file=sys.stderr)
    print(f"[sys]  {sys_prompt[:80]}...", file=sys.stderr)
    print(f"[toks] prompt={len(prompt_ids)}  max_gen={args.max_tokens}\n",
          file=sys.stderr)

    stop_ids = []
    for name in ("<|im_end|>", "</s>"):
        tid = tok.token_to_id(name)
        if tid is not None:
            stop_ids.append(tid)

    # ── Random search ───────────────────────────────────────────────────
    rng = random.Random(args.seed)
    trials: list[dict] = []
    t_total = time.time()

    for i in range(args.trials):
        trial_seed = rng.randint(0, 2**31 - 1)
        params = sample_params(rng)
        params["_seed"] = trial_seed

        t0 = time.time()
        try:
            text = run_one(model, tok, params, prompt_ids, args.max_tokens, stop_ids)
        except Exception as exc:
            text = ""
            print(f"\n  [err] trial {i+1}: {exc}", file=sys.stderr)
        elapsed = time.time() - t0

        scores = score_cot_output(text)

        trials.append({
            "trial":  i + 1,
            "params": {k: v for k, v in params.items() if not k.startswith("_")},
            "scores": scores,
            "text":   text,
            "elapsed": round(elapsed, 2),
            "_seed":  params["_seed"],
        })

        # progress
        n_bars = 30
        done = int(n_bars * (i + 1) / args.trials)
        bar = "█" * done + "░" * (n_bars - done)
        sys.stderr.write(
            f"\r[{bar}] {i+1:3d}/{args.trials}  "
            f"comp={scores['composite']:.4f}  "
            f"struct={scores['struct_score']:.2f}  "
            f"eng={scores['english_ratio']:.2f}  "
            f"think={int(scores['has_think'])} fin={int(scores['has_final'])}  "
            f"{elapsed:.1f}s  "
        )
        sys.stderr.flush()

    total_s = time.time() - t_total
    print(f"\n\n[done] {args.trials} trials in {total_s:.0f}s  "
          f"({total_s/args.trials:.1f}s/trial)\n", file=sys.stderr)

    # ── Rank ────────────────────────────────────────────────────────────
    trials.sort(key=lambda r: r["scores"]["composite"], reverse=True)
    best_score = trials[0]["scores"]["composite"]

    W = 70
    print("=" * W)
    print(f"TOP {min(args.top_n, len(trials))} CONFIGS  |  "
          f"make self / self_awareness  |  best={best_score:.4f}")
    print("=" * W)

    for rank, t in enumerate(trials[:args.top_n], 1):
        pa = t["params"]
        sc = t["scores"]
        print(f"\n── #{rank}  composite={sc['composite']:.4f}  "
              f"struct={sc['struct_score']:.2f}  eng={sc['english_ratio']:.2f}  "
              f"final_words={sc['n_words_final']}  {t['elapsed']:.1f}s")
        print(f"   TEMP={pa['temperature']:.3f}  TOP_K={pa['top_k']}  "
              f"TOP_P={pa['top_p']:.3f}  MIN_P={pa['min_p']:.3f}")
        print(f"   REP_PEN={pa['rep_pen']:.3f}  PRES_PEN={pa['pres_pen']:.3f}  "
              f"FREQ_PEN={pa['freq_pen']:.3f}")
        print(f"   rep3={sc['rep_rate_3']:.3f}  dist2={sc['distinct_2']:.3f}  "
              f"dup={sc['max_dup']}  think={sc['has_think']}  final={sc['has_final']}")
        tx = t["text"]
        print(f"   text: {tx[:250]}{'...' if len(tx) > 250 else ''}")

    # ── Best config Makefile snippet ─────────────────────────────────────
    best_trial = trials[0]
    bp = best_trial["params"]
    print("\n" + "=" * W)
    print("BEST CONFIG — Makefile defaults (paste lines 25-37)")
    print("=" * W)
    print(f"TEMP      ?= {bp['temperature']:.3f}")
    print(f"TOP_K     ?= {bp['top_k']}")
    print(f"TOP_P     ?= {bp['top_p']:.3f}")
    print(f"MIN_P     ?= {bp['min_p']:.3f}")
    print(f"REP_PEN   ?= {bp['rep_pen']:.3f}")
    print(f"PRES_PEN  ?= {bp['pres_pen']:.3f}")
    print(f"FREQ_PEN  ?= {bp['freq_pen']:.3f}")
    print("=" * W)

    # ── Quality check ───────────────────────────────────────────────────
    if best_score < 0.55:
        print("\n[!] WARNING: best composite < 0.55 — "
              "quality may be poor, consider more trials", file=sys.stderr)

    # ── Verbose re-run of top-N ─────────────────────────────────────────
    num_verbose = min(args.verbose_top, len(trials))
    if num_verbose > 0:
        print("\n" + "=" * W)
        print(f"INDIVIDUAL VERIFICATION — re-running top {num_verbose} config(s)")
        print("=" * W)
        for rank, t in enumerate(trials[:num_verbose]):
            verb_params = dict(t["params"])
            verb_params["_seed"] = t["_seed"]
            show_trial(model, tok, prompt_ids, verb_params,
                       args.max_tokens, stop_ids,
                       label=f"Config #{rank+1}  (search rank #{rank+1})")

    # ── JSON output ─────────────────────────────────────────────────────
    if args.output:
        out = []
        for t in trials:
            out.append({
                "trial": t["trial"], "params": t["params"],
                "scores": t["scores"], "text": t["text"],
                "elapsed": t["elapsed"],
            })
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
