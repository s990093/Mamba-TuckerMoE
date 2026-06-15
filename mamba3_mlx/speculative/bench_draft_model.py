"""Speculative-decoding A/B: trained draft transformer vs. n-gram drafter.

Answers the question "does a trained draft model accept more tokens per round
(higher ARL / acceptance rate) than the training-free n-gram drafter?" using
the *same* target verifier (``model_verify_forward``) for both, so the ARL
numbers are directly comparable.

Pipeline per prompt::

  1. AR greedy (target only)                 → reference token stream + tok/s
  2. Model-drafted spec decode (this file)   → ARL, accept-rate, decode tok/s,
                                               longest-prefix match vs AR
  3. n-gram Jacobi (speculative/jacobi.py)   → ARL for comparison

Greedy only — speculative acceptance needs a deterministic verifier.  On bf16
the verify path (chunk-scan, L>1) drifts slightly from AR single-step decode,
so we report the longest matching prefix rather than requiring byte-equality
(use --dtype fp32 for strict equality).

Run::

    python -m mamba3_mlx.speculative.bench_draft_model \\
        --prompt "Who are you?" --K 6 --max_tokens 96
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
from mamba3_mlx.speculative.draft_transformer import DraftGuesser, load_draft_model
from mamba3_mlx.speculative.forward import extract_state_at, model_verify_forward
from mamba3_mlx.speculative.jacobi import jacobi_decode
from mamba3_mlx.utils.config import GenerationConfig, Mamba3Config
from mamba3_mlx.utils.system_prompts import MODE_ALIASES, resolve_system_prompt

DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}

_DEFAULT_SYSTEM = "You are a helpful assistant. Think step by step before answering."


def _build_chatml_prompt(tok: Tokenizer, system_prompt: str, user_msg: str,
                         seed_think: bool = True) -> list[int]:
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n" + ("<think>\n" if seed_think else "")
    )
    ids = tok.encode(text, add_special_tokens=False).ids
    bos = tok.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids
    return ids


def greedy_autoregressive(model, prompt_ids, max_tokens, stop_set):
    last_logits, state, _ = prefill(model, prompt_ids)
    tokens: list[int] = []
    for _ in range(max_tokens):
        tok = int(mx.argmax(last_logits).item())
        tokens.append(tok)
        if tok in stop_set or len(tokens) >= max_tokens:
            break
        logits_out, state = model(mx.array([[tok]], dtype=mx.int32), states=state)
        last_logits = logits_out[0, -1]
        mx.eval(last_logits, *_iter_state_arrays(state))
    return tokens


def _longest_prefix(a, b) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def model_drafted_decode(model, guesser: DraftGuesser, prompt_ids, max_tokens,
                         K, stop_set):
    """Classic speculative decode: draft proposes K-1, target verifies.

    Returns dict with tokens, arl, accept_rate, n_rounds, n_proposed,
    n_accept_guesses, decode_tps.
    """
    last_logits, state, _ = prefill(model, prompt_ids)
    first_token = int(mx.argmax(last_logits).item())
    generated: list[int] = [first_token]

    guesser.reset()
    guesser.prefill(list(prompt_ids))   # KV covers up-to-but-excluding first_token

    prev_token = first_token
    n_rounds = 0
    n_accepted = 0            # tokens emitted by verify rounds (incl. bonus)
    n_proposed = 0            # draft guesses issued
    n_accept_guesses = 0      # draft guesses that matched the target
    stop_hit = first_token in stop_set

    t0 = time.perf_counter()
    while len(generated) < max_tokens and not stop_hit:
        guesses = guesser.draft(prev_token, K - 1)
        n_proposed += len(guesses)

        verify_ids = mx.array([[prev_token] + guesses], dtype=mx.int32)
        logits, payload = model_verify_forward(model, verify_ids, state)
        preds = mx.argmax(logits, axis=-1).astype(mx.int32)[0]
        mx.eval(preds)
        pred_list = [int(t) for t in preds.tolist()]

        accepted = [pred_list[0]]
        for i in range(K - 1):
            if pred_list[i] == int(guesses[i]):
                accepted.append(pred_list[i + 1])
            else:
                break
        m = len(accepted)
        n_accept_guesses += (m - 1)

        state = extract_state_at(payload, m, branch=0)
        mx.eval(*_iter_state_arrays(state))
        guesser.commit([prev_token] + accepted[:-1])

        for tok in accepted:
            generated.append(tok)
            if tok in stop_set:
                stop_hit = True
                break
            if len(generated) >= max_tokens:
                break
        prev_token = accepted[-1]
        n_rounds += 1
        n_accepted += m
    elapsed = time.perf_counter() - t0

    timed = max(len(generated) - 1, 0)
    return {
        "tokens": generated,
        "n_rounds": n_rounds,
        "n_accepted": n_accepted,
        "arl": n_accepted / max(n_rounds, 1),
        "n_proposed": n_proposed,
        "n_accept_guesses": n_accept_guesses,
        "accept_rate": n_accept_guesses / max(n_proposed, 1),
        "decode_tps": timed / max(elapsed, 1e-6),
        "elapsed": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",
                   default=str(REPO_ROOT / "checkpoints" / "v6" / "latest_sft_cot_model.npz"))
    p.add_argument("--draft_path",
                   default=str(REPO_ROOT / "checkpoints" / "draft_model" / "draft_distill_s107000.pt"))
    p.add_argument("--tokenizer_path",
                   default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"))
    p.add_argument("--prompt", default="Who are you?")
    p.add_argument("--mode", default="self_awareness",
                   choices=sorted(set(MODE_ALIASES.keys())))
    p.add_argument("--system", default=_DEFAULT_SYSTEM)
    p.add_argument("--no-seed-think", action="store_true")
    p.add_argument("--raw-prompt", action="store_true")
    p.add_argument("--max_tokens", type=int, default=96)
    p.add_argument("--K", action="append", type=int, default=None,
                   help="Speculative window(s). Repeatable. Default: 4 6 8.")
    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP))
    p.add_argument("--ngram_n", type=int, default=4)
    p.add_argument("--no-ngram-compare", action="store_true",
                   help="Skip the n-gram Jacobi comparison rows.")
    p.add_argument("--no-eos-stop", action="store_true",
                   help="Ignore EOS so all paths run the full max_tokens.")
    p.add_argument("--static-spec", action="store_true",
                   help="Also run StaticSpecDecoder (compiled draft + quantized "
                        "single-graph verify) — the optimized path.")
    p.add_argument("--quant-moe", type=int, default=8,
                   help="static-spec: 8-bit quant on TuckerMoE U-factors (0=off).")
    p.add_argument("--quant-head", type=int, default=8,
                   help="static-spec: 8-bit quant on the LM head (0=off).")
    p.add_argument("--eager-ar", action="store_true",
                   help="Use the slow eager AR loop as baseline instead of the "
                        "production `make self-s` StaticDecoder (default).")
    args = p.parse_args()

    dtype = DTYPE_MAP[args.dtype]
    Ks = sorted(set(args.K or [4, 6, 8]))

    print(f"[bench] dtype={args.dtype}  loading target {args.model_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    load_checkpoint(model, args.model_path, dtype=dtype)
    mx.eval(model.parameters())

    print(f"[bench] loading draft {args.draft_path}", file=sys.stderr)
    draft = load_draft_model(args.draft_path, dtype=dtype)
    guesser = DraftGuesser(draft)

    sys_prompt = resolve_system_prompt(args.mode, args.system)
    if args.raw_prompt:
        ids = tok.encode(args.prompt, add_special_tokens=False).ids
        bos = tok.token_to_id("<s>")
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
    else:
        ids = _build_chatml_prompt(tok, sys_prompt, args.prompt,
                                   seed_think=not args.no_seed_think)

    stop_set: set[int] = set()
    if not args.no_eos_stop:
        for name in ("<|im_end|>", "</s>"):
            tid = tok.token_to_id(name)
            if tid is not None:
                stop_set.add(tid)

    print(f"[bench] mode={args.mode} prompt_tokens={len(ids)} "
          f"max_tokens={args.max_tokens} stop={sorted(stop_set) or 'none'}",
          file=sys.stderr)

    # ── AR baseline = production `make self-s` (StaticDecoder metal_fuse+q8) ──
    # This is the REAL AR speed (~95-99 tok/s), not the ~2x-slower eager loop.
    # Speedups are measured against THIS unless --eager-ar is given.
    self_s_tps = None
    if not args.eager_ar:
        from mamba3_mlx.mlx_model.static_decode import StaticDecoder
        model.precompute()
        sd = StaticDecoder(model, metal_fuse=True, quant_moe_bits=8,
                           quant_proj_bits=8, quant_head_bits=8)
        sd.generate(ids, GenerationConfig(max_tokens=16, temperature=0.0))  # warm
        sr = sd.generate(ids, GenerationConfig(max_tokens=args.max_tokens,
                                               temperature=0.0),
                         stop_token_ids=sorted(stop_set))
        ref = sr.tokens
        ar_tps = self_s_tps = sr.decode_tps
        print(f"\n[AR baseline = make self-s] {len(ref):4d} tok  {ar_tps:6.1f} tok/s  "
              f"(StaticDecoder metal_fuse+q8 — the real target)\n", file=sys.stderr)
    else:
        t0 = time.perf_counter()
        ref = greedy_autoregressive(model, ids, args.max_tokens, stop_set)
        ar_tps = len(ref) / max(time.perf_counter() - t0, 1e-6)
        print(f"\n[AR baseline = eager]  {len(ref):4d} tok  {ar_tps:6.1f} tok/s  "
              f"(slow eager loop, drifts with thermal)\n", file=sys.stderr)

    # ── StaticSpecDecoder (optimized: compiled draft + quantized verify) ─────
    sspec = None
    if args.static_spec:
        from mamba3_mlx.speculative.static_spec import StaticSpecDecoder
        sspec = StaticSpecDecoder(model, draft, quant_moe_bits=args.quant_moe,
                                  quant_head_bits=args.quant_head)

    gen_cfg = GenerationConfig(max_tokens=args.max_tokens)
    hdr = (f"{'method':<16} {'K':>3} {'ARL':>5} {'accept%':>8} "
           f"{'tok/s':>7} {'speedup':>8} {'prefix':>9} {'got':>4}")
    print(hdr, file=sys.stderr)
    print("-" * len(hdr), file=sys.stderr)

    for K in Ks:
        # ── Draft-model speculative ──────────────────────────────────────────
        # warm the K-token verify graph once
        _l, _ = model(mx.zeros((1, K), dtype=mx.int32), states=None)
        mx.eval(_l)
        r = model_drafted_decode(model, guesser, ids, args.max_tokens, K, stop_set)
        pref = _longest_prefix(ref, r["tokens"])
        speedup = r["decode_tps"] / max(ar_tps, 1e-6)
        print(f"{'draft-model':<16} {K:>3d} {r['arl']:>5.2f} "
              f"{100*r['accept_rate']:>7.1f}% {r['decode_tps']:>7.1f} "
              f"{speedup:>7.2f}x {pref:>4d}/{len(ref):<4d} {len(r['tokens']):>4d}",
              file=sys.stderr)

        # ── n-gram Jacobi comparison ─────────────────────────────────────────
        if not args.no_ngram_compare:
            jr = jacobi_decode(
                model, ids, gen_cfg, K=K, use_ngram=True, ngram_n=args.ngram_n,
                stop_token_ids=sorted(stop_set), no_eos_stop=args.no_eos_stop,
            )
            jpref = _longest_prefix(ref, jr.tokens)
            jaccept = jr.n_ngram_hits / max(jr.n_rounds * (K - 1), 1)
            jspeed = jr.decode_tps / max(ar_tps, 1e-6)
            print(f"{'ngram-jacobi':<16} {K:>3d} {jr.arl:>5.2f} "
                  f"{100*jaccept:>7.1f}% {jr.decode_tps:>7.1f} "
                  f"{jspeed:>7.2f}x {jpref:>4d}/{len(ref):<4d} {len(jr.tokens):>4d}",
                  file=sys.stderr)

        # ── StaticSpecDecoder (optimized path) ───────────────────────────────
        if sspec is not None:
            sspec.generate(ids, 16, K=K)   # warm compile
            sr = sspec.generate(ids, args.max_tokens, K=K,
                                stop_token_ids=sorted(stop_set))
            spref = _longest_prefix(ref, sr.tokens)
            base = self_s_tps if self_s_tps else ar_tps
            sspeed = sr.decode_tps / max(base, 1e-6)
            tag = "vs self-s" if self_s_tps else "vs AR"
            print(f"{'static-spec(q8)':<16} {K:>3d} {sr.arl:>5.2f} "
                  f"{'—':>8} {sr.decode_tps:>7.1f} "
                  f"{sspeed:>7.2f}x {spref:>4d}/{len(ref):<4d} {len(sr.tokens):>4d}"
                  f"  [{tag}]", file=sys.stderr)
        print("", file=sys.stderr)


if __name__ == "__main__":
    main()
