#!/usr/bin/env python3
"""Hyperparameter tuner for mamba3_mlx — self_awareness + email modes.

Random search over sampling / repetition parameters. Supports two scoring
modes: identity (self_awareness) and email (summarize_email). Can use the
StaticDecoder fast path (--static) and batch sampling (--batch-size B) to
score B independent samples per param config in one compiled graph, which is
~2.6x more efficient than sequential single-stream runs.

Usage:
  # Standard (non-static) — same as before
  cd mamba3_mlx && .venv/bin/python3 tune_self.py --trials 200

  # Static fast path, both modes, B=4 samples/config
  .venv/bin/python3 tune_self.py --static --metal-fuse \\
      --quant-moe 8 --quant-proj 8 --quant-head 8 \\
      --tune-mode both --batch-size 4 --trials 120

  # Email only
  .venv/bin/python3 tune_self.py --static --metal-fuse \\
      --quant-moe 8 --tune-mode email --batch-size 2

  # Identity only, exact-phrase filter
  .venv/bin/python3 tune_self.py --static --metal-fuse \\
      --quant-moe 8 --exact-only --trials 160
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
from mamba3_mlx.mlx_model.static_decode import StaticDecoder

DEFAULT_MODEL     = str(REPO_ROOT / "checkpoints" / "v3" / "latest_sft_cot_model.npz")
DEFAULT_TOKENIZER = str(REPO_ROOT / "cot_dataset" / "tokenizer.json")
DEFAULT_PROMPT    = "Who are you?"

PARAM_SPACE = {
    "temperature":   (0.02, 0.55),
    "top_k":         [5, 10, 15, 20, 30, 40, 50, 60],
    "top_p":         (0.80, 0.99),
    "min_p":         (0.01, 0.20),
    "rep_pen":       (1.05, 1.50),
    "pres_pen":      (0.05, 0.50),
    "freq_pen":      (0.02, 0.20),
}

# Match make self-s exactly so tuned seeds reproduce there:
#   repeat_last_n=256 (Makefile REPEAT_LAST_N), max_tokens=512 (SELF_MAX_TOK).
# ⚠️ StaticDecoder output is sensitive to max_tokens (KV buffer size → Metal
#    kernel tiling → bf16 rounding → sampled token). Tuning at a different
#    max_tokens will NOT reproduce in self-s. Keep these aligned.
FIXED_REPEAT_LAST_N = 256
DEFAULT_MAX_TOKENS  = 512
DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}

# ── Identity keywords for self_awareness scoring ──────────────────────────────
_IDENTITY_EXACT: list[str] = [
    "i am mamba", "i'm mamba", "i am a mamba",
    "my name is mamba", "called mamba", "named mamba",
    "i am the mamba", "this is mamba",
]
_IDENTITY_REQUIRED: list[str] = [
    "mamba", "i am", "i'm",
]
_IDENTITY_HIGH: list[str] = [
    "hybrid", "tuckermoe", "tucker", "moe", "mixture of experts",
    "edge", "offline", "iphone", "apple", "silicon",
    "state space", "ssm", "transformer",
    "architecture", "model", "language model",
]
_IDENTITY_MED: list[str] = [
    "assistant", "ai", "built", "designed", "trained",
    "capable", "run", "deployed", "compute",
]
_IDENTITY_NEG: list[str] = [
    "gpt", "chatgpt", "openai", "claude", "gemini", "llama",
    "anthropic", "google", "meta ai",
]

# ── Email keywords for summarize_email scoring ────────────────────────────────
_EMAIL_SUBJECT: list[str] = ["subject:", "re:", "fw:", "fwd:"]
_EMAIL_GREET:   list[str] = ["dear ", "hi ", "hello ", "greetings", "to whom"]
_EMAIL_CLOSE:   list[str] = [
    "best regards", "kind regards", "sincerely", "regards,",
    "best,", "yours truly", "thank you,", "thanks,",
]

# Prompts for each mode
IDENTITY_PROMPTS: list[str] = [
    "Who are you?",
    "What are you?",
    "Tell me about yourself.",
    "What can you do?",
    "Are you an AI?",
]

EMAIL_PROMPTS: list[str] = [
    "Write a short professional email to Professor Chen requesting a 30-minute meeting to discuss on-device AI inference research.",
    "Draft a polite email asking your supervisor for feedback on your monthly project report.",
]


# ── Param sampler ─────────────────────────────────────────────────────────────

def sample_params(rng: random.Random) -> dict:
    params = {}
    for name, space in PARAM_SPACE.items():
        if isinstance(space, list):
            params[name] = rng.choice(space)
        else:
            lo, hi = space
            params[name] = round(rng.uniform(lo, hi), 3)
    return params


# ── CoT block parser ──────────────────────────────────────────────────────────

_COT_TAGS = ("<think>", "</think>", "<final>", "</final>", "<|im_end|>", "</s>")


def parse_cot_blocks(raw_text: str) -> tuple[str, str, bool, bool]:
    """Return (think_body, final_body, has_think, has_final)."""
    m_think = re.search(r"<think>\s*(.*?)\s*</think>", raw_text, re.DOTALL)
    m_final = re.search(r"<final>\s*(.*?)\s*</final>", raw_text, re.DOTALL)

    has_think = bool(m_think)
    has_final = bool(m_final)
    think_body = m_think.group(1).strip() if m_think else ""
    final_body = m_final.group(1).strip() if m_final else ""

    if not has_final:
        tail = raw_text[m_think.end():] if m_think else raw_text
        for tag in _COT_TAGS:
            tail = tail.replace(tag, " ")
        final_body = tail.strip()

    return think_body, final_body, has_think, has_final


# ── English-likeness ──────────────────────────────────────────────────────────

_VOWELS:     set[str] = set("aeiouyAEIOUY")
_CONSONANTS: set[str] = set("bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ")


def _is_likely_english_word(w: str) -> bool:
    if not w:
        return False
    if re.fullmatch(r"[^\w]+", w):
        return True
    letters = [c for c in w if c.isalpha()]
    if not letters:
        return any(c.isdigit() for c in w)
    if not any(c in _VOWELS for c in letters):
        return False
    run = max_run = 0
    for c in letters:
        if c in _CONSONANTS:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    if max_run >= 5:
        return False
    if len(w) > 4 and w.isupper():
        return False
    return True


def english_word_ratio(text: str) -> float:
    words = [w.strip(",.!?;:\"'()[]{}*#") for w in text.split() if w.strip()]
    if not words:
        return 0.0
    return sum(1 for w in words if _is_likely_english_word(w)) / len(words)


# ── Text quality helpers ──────────────────────────────────────────────────────

def _words(text: str) -> list[str]:
    return [w for w in text.split() if w.strip()]


def _n_grams(words: list[str], n: int) -> list[tuple]:
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def rep_rate(text: str, n: int = 3) -> float:
    words = _words(text)
    if len(words) < n:
        return 0.0
    counts = Counter(_n_grams(words, n))
    return sum(1 for c in counts.values() if c > 1) / max(len(counts), 1)


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


def _body_quality(eval_text: str) -> tuple[float, float, float, float, float, int]:
    """Return (r3, d2, anti_rep, eng_ratio, eng_score, dup)."""
    r3  = rep_rate(eval_text, 3)
    r4  = rep_rate(eval_text, 4)
    d2  = distinct_n(eval_text, 2)
    dup = max_dup_span(eval_text)
    eng = english_word_ratio(eval_text)

    anti_rep = 1.0 - min(r3 * 0.6 + r4 * 0.4, 1.0)
    if dup >= 4:
        anti_rep *= max(0.1, 1.0 - (dup - 3) * 0.25)

    if eng < 0.3:
        eng_score = 0.0
    elif eng < 0.6:
        eng_score = (eng - 0.3) / 0.3 * 0.5
    elif eng < 0.85:
        eng_score = 0.5 + (eng - 0.6) / 0.25 * 0.3
    else:
        eng_score = 0.8 + (eng - 0.85) / 0.15 * 0.2

    return r3, d2, anti_rep, eng, eng_score, dup


# ── Identity accuracy scoring (self_awareness) ────────────────────────────────

def score_identity_accuracy(text: str) -> float:
    lo = text.lower()
    has_exact    = any(kw in lo for kw in _IDENTITY_EXACT)
    has_required = has_exact or any(kw in lo for kw in _IDENTITY_REQUIRED)
    if not has_required:
        return 0.0
    score = 0.70 if has_exact else 0.35
    score += min(sum(1 for kw in _IDENTITY_HIGH if kw in lo), 5) / 5.0 * 0.25
    score += min(sum(1 for kw in _IDENTITY_MED  if kw in lo), 3) / 3.0 * 0.05
    score -= sum(1 for kw in _IDENTITY_NEG if kw in lo) * 0.30
    return max(0.0, min(1.0, score))


def score_cot_output(raw_text: str, check_identity: bool = True) -> dict:
    """Score CoT output for self_awareness mode.

    Weights: struct 10%, identity 50%, quality 30%, diversity 10%.
    """
    think_body, final_body, has_think, has_final = parse_cot_blocks(raw_text)

    struct = 1.0 if (has_think and has_final) else (0.8 if has_final else (0.5 if has_think else 0.2))

    eval_text = final_body if final_body else raw_text
    words = _words(eval_text)
    nw = len(words)
    id_score = score_identity_accuracy(eval_text) if check_identity else 0.0

    if nw < 5:
        return {
            "n_words_think": len(_words(think_body)),
            "n_words_final": nw,
            "has_think": has_think, "has_final": has_final,
            "struct_score": struct,
            "identity_score": round(id_score, 4),
            "english_ratio": 0.0,
            "rep_rate_3": 0.0, "distinct_2": 0.0,
            "max_dup": 0,
            "composite": max(0.1, struct * 0.10 + id_score * 0.50),
        }

    r3, d2, anti_rep, eng, eng_score, dup = _body_quality(eval_text)

    nw = len(words)
    if nw >= 80:
        len_ok = 1.0
    elif nw >= 30:
        len_ok = 0.5 + 0.5 * (nw - 30) / 50.0
    else:
        len_ok = 0.5 * nw / 30.0

    quality = anti_rep * 0.35 + eng_score * 0.45 + len_ok * 0.20
    composite = struct * 0.10 + id_score * 0.50 + quality * 0.30 + d2 * 0.10

    return {
        "n_words_think": len(_words(think_body)),
        "n_words_final": nw,
        "has_think": has_think, "has_final": has_final,
        "struct_score":   round(struct, 4),
        "identity_score": round(id_score, 4),
        "english_ratio":  round(eng, 4),
        "eng_score":      round(eng_score, 4),
        "rep_rate_3":     round(r3, 4),
        "distinct_2":     round(d2, 4),
        "max_dup":        dup,
        "composite":      round(composite, 4),
    }


# ── Email quality scoring (summarize_email) ───────────────────────────────────

def score_email_output(raw_text: str) -> dict:
    """Score CoT output for summarize_email mode.

    Weights: struct 10%, email structure 45%, body quality 35%, diversity 10%.
    Email structure: subject(35%) + greeting(25%) + closing(25%) + length(15%).
    """
    think_body, final_body, has_think, has_final = parse_cot_blocks(raw_text)

    struct = 1.0 if (has_think and has_final) else (0.8 if has_final else (0.5 if has_think else 0.2))

    eval_text = final_body if final_body else raw_text
    lo = eval_text.lower()
    words = _words(eval_text)
    nw = len(words)

    has_subject = any(kw in lo for kw in _EMAIL_SUBJECT)
    has_greet   = any(kw in lo for kw in _EMAIL_GREET)
    has_close   = any(kw in lo for kw in _EMAIL_CLOSE)

    if nw < 5:
        return {
            "n_words_think": len(_words(think_body)),
            "n_words_final": nw,
            "has_think": has_think, "has_final": has_final,
            "struct_score":  round(struct, 4),
            "email_score":   0.0,
            "has_subject":   has_subject,
            "has_greet":     has_greet,
            "has_close":     has_close,
            "english_ratio": 0.0,
            "rep_rate_3":    0.0,
            "distinct_2":    0.0,
            "max_dup":       0,
            "composite":     round(struct * 0.10, 4),
        }

    r3, d2, anti_rep, eng, eng_score, dup = _body_quality(eval_text)

    # Email structural components
    email_score = (
        has_subject * 0.35
        + has_greet * 0.25
        + has_close * 0.25
        + (1.0 if nw >= 40 else nw / 40.0) * 0.15
    )

    # Body quality
    if nw >= 100:
        len_ok = 1.0
    elif nw >= 40:
        len_ok = 0.5 + 0.5 * (nw - 40) / 60.0
    else:
        len_ok = 0.5 * nw / 40.0

    body_qual = anti_rep * 0.35 + eng_score * 0.50 + len_ok * 0.15

    composite = struct * 0.10 + email_score * 0.45 + body_qual * 0.35 + d2 * 0.10

    return {
        "n_words_think": len(_words(think_body)),
        "n_words_final": nw,
        "has_think":  has_think, "has_final": has_final,
        "struct_score":  round(struct, 4),
        "email_score":   round(email_score, 4),
        "has_subject":   has_subject,
        "has_greet":     has_greet,
        "has_close":     has_close,
        "english_ratio": round(eng, 4),
        "eng_score":     round(eng_score, 4),
        "rep_rate_3":    round(r3, 4),
        "distinct_2":    round(d2, 4),
        "max_dup":       dup,
        "composite":     round(composite, 4),
    }


def score_text(text: str, mode: str) -> dict:
    """Dispatch to the right scorer for *mode* (self | email)."""
    if mode == "email":
        return score_email_output(text)
    return score_cot_output(text, check_identity=True)


# ── Model / decoder loading ───────────────────────────────────────────────────

def load_everything(model_path: str, tokenizer_path: str, dtype_str: str,
                    skip_warmup: bool = False):
    dtype = DTYPE_MAP.get(dtype_str, mx.bfloat16)
    print(f"[load] tokenizer: {tokenizer_path}", file=sys.stderr)
    tok = Tokenizer.from_file(tokenizer_path)
    print(f"[load] model:     {model_path}", file=sys.stderr)
    cfg   = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    t0 = time.time()
    load_checkpoint(model, model_path, dtype=dtype)
    mx.eval(model.parameters())
    print(f"[load] weights done in {time.time() - t0:.2f}s  (dtype={dtype_str})",
          file=sys.stderr)
    # Reference-path dummy forward warms JIT but perturbs Metal numeric state,
    # changing StaticDecoder's sampled output. run.py skips it under --static,
    # so tuning skips it too (skip_warmup=True) to match make self-s exactly.
    if not skip_warmup:
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


def _make_gen_cfg(params: dict, max_tokens: int) -> GenerationConfig:
    return GenerationConfig(
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


# ── Single-stream trial ───────────────────────────────────────────────────────

def run_one(runner, tok, params: dict, prompt_ids: list[int],
            max_tokens: int, stop_ids: list[int]) -> str:
    """Run one generation; supports both StaticDecoder and plain model."""
    gen_cfg = _make_gen_cfg(params, max_tokens)
    try:
        if isinstance(runner, StaticDecoder):
            result = runner.generate(prompt_ids, gen_cfg, stop_token_ids=stop_ids)
            tokens = result.tokens
        else:
            result = generate(runner, prompt_ids, gen_cfg, stop_token_ids=stop_ids)
            tokens = result.tokens
        return tok.decode(tokens, skip_special_tokens=False).strip()
    except Exception as exc:
        print(f"\n  [run_one err] {exc}", file=sys.stderr)
        return ""


def run_batch(decoder: StaticDecoder, tok, params: dict, prompt_ids: list[int],
              max_tokens: int, stop_ids: list[int], B: int) -> list[str]:
    """Run B independent sampled streams from the same prompt in one graph.

    Uses StaticDecoder.generate_batch — ~2.6x more efficient than B sequential
    calls.  Returns a list of B decoded strings.
    """
    gen_cfg = _make_gen_cfg(params, max_tokens)
    try:
        all_tokens, _ = decoder.generate_batch(
            prompt_ids, gen_cfg, batch_size=B,
            stop_token_ids=stop_ids, unroll=1,
        )
        return [tok.decode(toks, skip_special_tokens=False).strip()
                for toks in all_tokens]
    except Exception as exc:
        print(f"\n  [run_batch err] {exc}", file=sys.stderr)
        return [""] * B


# ── Verbose display ───────────────────────────────────────────────────────────

def show_trial(runner, tok, prompt_ids, params, max_tokens, stop_ids,
               label: str = "", mode: str = "self"):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  TEMP={params['temperature']:.3f}  TOP_K={params['top_k']}  "
          f"TOP_P={params['top_p']:.3f}  MIN_P={params['min_p']:.3f}")
    print(f"  REP_PEN={params['rep_pen']:.3f}  PRES_PEN={params['pres_pen']:.3f}  "
          f"FREQ_PEN={params['freq_pen']:.3f}")
    print(f"{'─'*60}")
    t0 = time.time()
    try:
        text = run_one(runner, tok, params, prompt_ids, max_tokens, stop_ids)
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return
    elapsed = time.time() - t0
    scores = score_text(text, mode)
    print(text)
    print(f"\n  score={scores['composite']:.4f}  "
          f"struct={scores['struct_score']:.2f}  "
          f"eng={scores['english_ratio']:.2f}  "
          f"think={scores['has_think']}  final={scores['has_final']}  "
          f"{elapsed:.1f}s")
    return text, scores


# ── Makefile patch ────────────────────────────────────────────────────────────

def _patch_makefile(makefile_path: Path, bp: dict) -> bool:
    """Patch sampling knobs in-place.  Returns True if file was written.

    Uses ^...$  anchors (MULTILINE) + \\S* so empty-value lines like
    ``TEMP     ?=`` are matched correctly without consuming the next line.
    """
    if not makefile_path.exists():
        return False
    text = makefile_path.read_text()
    replacements = {
        r"(?m)^(TEMP\s+\?=\s*)\S*$":     f"\\g<1>{bp['temperature']:.3f}",
        r"(?m)^(TOP_K\s+\?=\s*)\S*$":    f"\\g<1>{bp['top_k']}",
        r"(?m)^(TOP_P\s+\?=\s*)\S*$":    f"\\g<1>{bp['top_p']:.3f}",
        r"(?m)^(MIN_P\s+\?=\s*)\S*$":    f"\\g<1>{bp['min_p']:.3f}",
        r"(?m)^(REP_PEN\s+\?=\s*)\S*$":  f"\\g<1>{bp['rep_pen']:.3f}",
        r"(?m)^(PRES_PEN\s+\?=\s*)\S*$": f"\\g<1>{bp['pres_pen']:.3f}",
        r"(?m)^(FREQ_PEN\s+\?=\s*)\S*$": f"\\g<1>{bp['freq_pen']:.3f}",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    makefile_path.write_text(text)
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Hyperparameter tuner — self_awareness + email modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Paths
    p.add_argument("--model-path",     default=DEFAULT_MODEL)
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    p.add_argument("--dtype",          default="bf16", choices=["bf16", "fp16", "fp32"])
    # Prompt
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Primary prompt override (single-mode only)")
    p.add_argument("--multi-prompt", action="store_true",
                   help="Average score over all IDENTITY_PROMPTS (self mode)")
    # Mode
    p.add_argument("--tune-mode", default="self",
                   choices=["self", "email", "both"],
                   help="self=identity scoring, email=email scoring, "
                        "both=average of the two (default: self)")
    # Generation
    p.add_argument("--max-tokens",  type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--trials",      type=int, default=200)
    p.add_argument("--top-n",       type=int, default=5)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output",      type=str, default=None)
    p.add_argument("--verbose-top", type=int, default=3)
    p.add_argument("--no-seed-think", action="store_true")
    p.add_argument("--patch-makefile", action="store_true")
    p.add_argument("--exact-only",   action="store_true",
                   help="Only show/rank configs with exact 'I am Mamba' phrase")
    # ── Static decoder flags ────────────────────────────────────────────────
    p.add_argument("--static",      action="store_true",
                   help="Use StaticDecoder (single compiled graph, ~130-144 tok/s with q8).")
    p.add_argument("--metal-fuse",  action="store_true",
                   help="Fused Metal SSM kernel (value-identical, +~50% over compiled path).")
    p.add_argument("--quant-moe",   type=int, default=0,
                   help="Quantize TuckerMoE weights to N bits (recommended: 8).")
    p.add_argument("--quant-proj",  type=int, default=0,
                   help="Quantize in_proj/dense/qkv weights to N bits (recommended: 8).")
    p.add_argument("--quant-head",  type=int, default=0,
                   help="Quantize head projection to N bits (recommended: 8).")
    # ── Batch sampling ──────────────────────────────────────────────────────
    p.add_argument("--batch-size",  type=int, default=1,
                   help="B independent samples per param config via generate_batch. "
                        "Requires --static. B=2 safe, B=4 recommended (M2 Pro 16GB). "
                        "Score = max(B scores). Default: 1 (sequential).")
    # ── Memory management ───────────────────────────────────────────────────
    p.add_argument("--clear-cache-every", type=int, default=25,
                   help="Call mx.metal.clear_cache() every N trials to free "
                        "intermediate buffers (default: 25; 0 = never).")
    args = p.parse_args()

    B = max(1, args.batch_size)
    use_batch = args.static and B > 1

    # ── Checkpoint ──────────────────────────────────────────────────────────
    model_path = args.model_path
    sidecar = _sidecar_path(model_path, DTYPE_MAP.get(args.dtype, mx.bfloat16))
    if sidecar.exists():
        model_path = str(sidecar)
    print(f"[ckpt] {model_path}", file=sys.stderr)

    # ── Load model ──────────────────────────────────────────────────────────
    # skip_warmup under --static to match run.py / make self-s numeric path.
    model, tok = load_everything(model_path, args.tokenizer_path, args.dtype,
                                 skip_warmup=args.static)

    # ── Static decoder (created once; compile cost amortised across all trials)
    runner = model
    if args.static:
        print(f"[static] building StaticDecoder  metal_fuse={args.metal_fuse}  "
              f"quant_moe={args.quant_moe}  quant_proj={args.quant_proj}  "
              f"quant_head={args.quant_head}", file=sys.stderr)
        t_sd = time.time()
        runner = StaticDecoder(
            model,
            metal_fuse=args.metal_fuse,
            quant_moe_bits=args.quant_moe,
            quant_proj_bits=args.quant_proj,
            quant_head_bits=args.quant_head,
        )
        print(f"[static] ready in {time.time() - t_sd:.1f}s", file=sys.stderr)
        if use_batch:
            print(f"[batch]  B={B} samples/config via generate_batch", file=sys.stderr)
            print("[batch]  ⚠️ generate_batch is a DIFFERENT numeric path "
                  "(metal=False); seeds found here may NOT reproduce in "
                  "single-stream `make self-s`. Use --batch-size 1 for "
                  "reproducible self-s tuning.", file=sys.stderr)

    # ── Build prompt lists per mode ──────────────────────────────────────────
    seed_think = not args.no_seed_think

    def _build_ids(mode_name: str, prompts: list[str]) -> list[list[int]]:
        sys_p = resolve_system_prompt(mode_name, "")
        return [build_prompt_ids(tok, sys_p, pr, seed_think=seed_think)[0]
                for pr in prompts]

    self_prompts  = IDENTITY_PROMPTS if args.multi_prompt else [args.prompt]
    email_prompts = EMAIL_PROMPTS

    prompt_sets: dict[str, list[list[int]]] = {}
    if args.tune_mode in ("self", "both"):
        prompt_sets["self"]  = _build_ids("self_awareness",  self_prompts)
    if args.tune_mode in ("email", "both"):
        prompt_sets["email"] = _build_ids("summarize_email", email_prompts)

    # Stop tokens
    stop_ids: list[int] = []
    for name in ("<|im_end|>", "</s>"):
        tid = tok.token_to_id(name)
        if tid is not None:
            stop_ids.append(tid)

    # ── Progress banner ──────────────────────────────────────────────────────
    print(f"\n[tune-mode] {args.tune_mode}  trials={args.trials}  "
          f"static={args.static}  batch_size={B}", file=sys.stderr)
    for mode_name, psets in prompt_sets.items():
        print(f"  [{mode_name}] {len(psets)} prompts", file=sys.stderr)
    print(f"[toks] max_gen={args.max_tokens}\n", file=sys.stderr)

    # ── Random search ────────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    trials: list[dict] = []
    t_total = time.time()

    for i in range(args.trials):
        trial_seed = rng.randint(0, 2**31 - 1)
        params = sample_params(rng)
        params["_seed"] = trial_seed

        # ── Periodic cache clear ──────────────────────────────────────────
        if args.clear_cache_every > 0 and i > 0 and i % args.clear_cache_every == 0:
            mx.clear_cache()

        t0 = time.time()
        all_mode_scores: list[dict] = []
        primary_text = ""
        has_exact_any = False

        for mode_name, pids_list in prompt_sets.items():
            mode_texts: list[str] = []
            for pids in pids_list:
                if use_batch:
                    texts = run_batch(runner, tok, params, pids,
                                      args.max_tokens, stop_ids, B)
                else:
                    t = run_one(runner, tok, params, pids,
                                args.max_tokens, stop_ids)
                    texts = [t]
                mode_texts.extend(texts)

            # Score all texts for this mode; take the best composite
            mode_sc_list = [score_text(tx, mode_name) for tx in mode_texts]
            best_sc = max(mode_sc_list, key=lambda s: s["composite"])
            all_mode_scores.append(best_sc)

            # Primary text = first prompt, first (or best) sample
            if not primary_text:
                best_idx = mode_sc_list.index(best_sc)
                primary_text = mode_texts[best_idx]

            if mode_name == "self":
                has_exact_any = any(
                    any(kw in tx.lower() for kw in _IDENTITY_EXACT)
                    for tx in mode_texts
                )

        elapsed = time.time() - t0

        # Average composite across modes
        avg_scores: dict = {}
        for key_name in all_mode_scores[0]:
            vals = [s.get(key_name, 0.0) for s in all_mode_scores]
            if isinstance(vals[0], float):
                avg_scores[key_name] = round(sum(vals) / len(vals), 4)
            elif isinstance(vals[0], bool):
                avg_scores[key_name] = all(vals)
            else:
                avg_scores[key_name] = vals[0]

        trials.append({
            "trial":     i + 1,
            "params":    {k: v for k, v in params.items() if not k.startswith("_")},
            "scores":    avg_scores,
            "text":      primary_text,
            "elapsed":   round(elapsed, 2),
            "_seed":     params["_seed"],
            "has_exact": has_exact_any,
        })

        n_bars = 28
        done = int(n_bars * (i + 1) / args.trials)
        bar  = "█" * done + "░" * (n_bars - done)
        sys.stderr.write(
            f"\r[{bar}] {i+1:3d}/{args.trials}  "
            f"comp={avg_scores['composite']:.4f}  "
            f"struct={avg_scores['struct_score']:.2f}  "
            f"eng={avg_scores['english_ratio']:.2f}  "
            f"{elapsed:.1f}s  "
        )
        sys.stderr.flush()

    total_s = time.time() - t_total
    print(f"\n\n[done] {args.trials} trials in {total_s:.0f}s  "
          f"({total_s/args.trials:.1f}s/trial)\n", file=sys.stderr)

    # ── Rank ─────────────────────────────────────────────────────────────────
    trials.sort(key=lambda r: r["scores"]["composite"], reverse=True)
    best_score = trials[0]["scores"]["composite"]
    exact_trials = [t for t in trials if t.get("has_exact")]
    n_exact = len(exact_trials)

    W = 72
    print("=" * W)
    mode_tag = f"{args.tune_mode} mode"
    print(f"TOP {min(args.top_n, len(trials))} CONFIGS  |  {mode_tag}  |  "
          f"best={best_score:.4f}")
    if args.tune_mode in ("self", "both"):
        print(f"  'I am Mamba' exact phrase found in: {n_exact}/{len(trials)} trials")
    print("=" * W)

    ranked_list = exact_trials if (args.exact_only and exact_trials) else trials
    if args.exact_only and exact_trials:
        print(f"\n[--exact-only] showing only {len(exact_trials)} exact-phrase configs\n")

    for rank, t in enumerate(ranked_list[:args.top_n], 1):
        pa = t["params"]
        sc = t["scores"]
        exact_flag = "  *** EXACT PHRASE ***" if t.get("has_exact") else ""
        print(f"\n── #{rank}  composite={sc['composite']:.4f}  "
              f"struct={sc['struct_score']:.2f}  eng={sc['english_ratio']:.2f}  "
              f"words={sc['n_words_final']}  {t['elapsed']:.1f}s{exact_flag}")
        print(f"   TEMP={pa['temperature']:.3f}  TOP_K={pa['top_k']}  "
              f"TOP_P={pa['top_p']:.3f}  MIN_P={pa['min_p']:.3f}")
        print(f"   REP_PEN={pa['rep_pen']:.3f}  PRES_PEN={pa['pres_pen']:.3f}  "
              f"FREQ_PEN={pa['freq_pen']:.3f}")
        if "identity_score" in sc:
            print(f"   id={sc['identity_score']:.2f}", end="")
        if "email_score" in sc:
            print(f"   email={sc['email_score']:.2f}", end="")
        print(f"   rep3={sc['rep_rate_3']:.3f}  dist2={sc['distinct_2']:.3f}  "
              f"dup={sc['max_dup']}  think={sc['has_think']}  final={sc['has_final']}")
        tx = t["text"]
        print(f"   text: {tx[:260]}{'...' if len(tx) > 260 else ''}")

    # ── Best config block ─────────────────────────────────────────────────────
    best_trial = exact_trials[0] if (args.exact_only and exact_trials) else trials[0]
    bp = best_trial["params"]
    print("\n" + "=" * W)
    print("BEST CONFIG — Makefile defaults")
    print("=" * W)
    print(f"TEMP      ?= {bp['temperature']:.3f}")
    print(f"TOP_K     ?= {bp['top_k']}")
    print(f"TOP_P     ?= {bp['top_p']:.3f}")
    print(f"MIN_P     ?= {bp['min_p']:.3f}")
    print(f"REP_PEN   ?= {bp['rep_pen']:.3f}")
    print(f"PRES_PEN  ?= {bp['pres_pen']:.3f}")
    print(f"FREQ_PEN  ?= {bp['freq_pen']:.3f}")
    print("=" * W)

    if best_score < 0.55:
        print("\n[!] WARNING: best composite < 0.55 — consider more trials",
              file=sys.stderr)

    if args.patch_makefile:
        mk = Path(__file__).resolve().parent / "Makefile"
        patched = _patch_makefile(mk, bp)
        print(f"\n[{'patched' if patched else '! Makefile not found'}] {mk}",
              file=sys.stderr)

    # ── Verbose re-run ────────────────────────────────────────────────────────
    num_verbose = min(args.verbose_top, len(trials))
    primary_mode = "email" if args.tune_mode == "email" else "self"
    primary_pids = list(prompt_sets.values())[0][0]

    if num_verbose > 0:
        print("\n" + "=" * W)
        print(f"INDIVIDUAL VERIFICATION — re-running top {num_verbose} config(s)")
        print("=" * W)
        for rank, t in enumerate(ranked_list[:num_verbose]):
            vp = dict(t["params"])
            vp["_seed"] = t["_seed"]
            show_trial(runner, tok, primary_pids, vp,
                       args.max_tokens, stop_ids,
                       label=f"Config #{rank+1}  (rank #{rank+1})",
                       mode=primary_mode)

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.output:
        out = [{"trial": t["trial"], "params": t["params"],
                "scores": t["scores"], "text": t["text"],
                "elapsed": t["elapsed"], "_seed": t["_seed"]} for t in trials]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
