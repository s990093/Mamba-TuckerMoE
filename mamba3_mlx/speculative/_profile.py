"""Quick per-round timing of jacobi_decode to identify bottleneck.

Pass ``--sjd`` to profile the SJD sampling path instead of greedy.
"""
from __future__ import annotations

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
from mamba3_mlx.utils.config import Mamba3Config
from mamba3_mlx.utils.system_prompts import resolve_system_prompt
from mamba3_mlx.speculative.forward import extract_state_at, model_verify_forward
from mamba3_mlx.speculative.ngram_cache import NGramCache
from mamba3_mlx.speculative.jacobi import _build_guesses


def _chatml_ids(tok, sys_prompt, user_msg):
    text = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    ids = tok.encode(text, add_special_tokens=False).ids
    bos = tok.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids
    return ids


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        default=str(REPO_ROOT / "checkpoints" / "latest_sft_cot_model.npz"),
    )
    p.add_argument(
        "--tokenizer_path",
        default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"),
    )
    p.add_argument("--prompt", default="Who are you?")
    p.add_argument("--mode", default="self_awareness")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--use_ngram", action="store_true")
    args = p.parse_args()

    DTYPE_MAP = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    dtype = DTYPE_MAP[args.dtype]

    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    load_checkpoint(model, args.model_path, dtype=dtype)
    mx.eval(model.parameters())

    sys_prompt = resolve_system_prompt(args.mode, None)
    ids = _chatml_ids(tok, sys_prompt, args.prompt)
    print(f"prompt_tokens={len(ids)}")

    last_logits, state, t_pf = prefill(model, ids)
    print(f"prefill: {t_pf*1000:.1f} ms")

    # Warm up a few decode shapes
    for L in (args.K, 1, 2):
        dummy = mx.zeros((1, L), dtype=mx.int32)
        _l, _s = model(dummy, states=state)
        mx.eval(_l)

    # First token
    first = int(mx.argmax(last_logits).item())
    generated = [first]
    prev_token = first
    fallback = first
    ngram = NGramCache(n=3) if args.use_ngram else None
    if ngram:
        ngram.update_sequence(ids)

    # Reference AR step timing (compiled-free, like verify.py)
    print("\n--- AR baseline timing (uncompiled) ---")
    ar_state = state
    ar_logits = last_logits
    ar_tok = first
    ar_times = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        a = mx.array([[ar_tok]], dtype=mx.int32)
        lo, ar_state = model(a, states=ar_state)
        ar_logits = lo[0, -1]
        ar_pred = mx.argmax(ar_logits).astype(mx.int32)
        mx.eval(ar_pred, *_iter_state_arrays(ar_state))
        ar_tok = int(ar_pred.item())
        dt = time.perf_counter() - t0
        ar_times.append(dt)
    print(f"AR median: {1000*sorted(ar_times)[len(ar_times)//2]:.1f} ms/tok "
          f"({1.0/(sorted(ar_times)[len(ar_times)//2]):.1f} tok/s)")

    # Jacobi round-by-round timing
    print("\n--- Jacobi round timing ---")
    for r in range(args.rounds):
        t0 = time.perf_counter()
        history = ids + generated[:-1]
        guesses, _ = _build_guesses(args.K, prev_token, history, ngram, fallback)
        t_build = time.perf_counter() - t0

        t1 = time.perf_counter()
        verify_ids = mx.array([[prev_token] + guesses], dtype=mx.int32)
        logits, perpos = model_verify_forward(model, verify_ids, state)
        preds = mx.argmax(logits[0], axis=-1).astype(mx.int32)
        mx.eval(preds)
        t_verify = time.perf_counter() - t1

        t2 = time.perf_counter()
        pred_list = preds.tolist()
        accepted = [pred_list[0]]
        for i in range(args.K - 1):
            if pred_list[i] == guesses[i]:
                accepted.append(pred_list[i + 1])
            else:
                break
        m = len(accepted)
        t_accept = time.perf_counter() - t2

        t3 = time.perf_counter()
        # No replay forward — extract state at position m-1 from the
        # per-position payload that the verify forward already recorded.
        state = extract_state_at(perpos, m)
        mx.eval(*_iter_state_arrays(state))
        t_replay = time.perf_counter() - t3

        total = time.perf_counter() - t0
        generated.extend(accepted)
        prev_token = accepted[-1]
        fallback = prev_token
        if ngram:
            full = ids + generated
            ngram.update_sequence(full, start_idx=max(0, len(full) - m - 2))

        print(
            f"round={r:2d} m={m:2d}/K={args.K} "
            f"build={1000*t_build:5.2f}ms "
            f"verify(L={args.K})={1000*t_verify:6.2f}ms "
            f"accept={1000*t_accept:5.2f}ms "
            f"state_extract={1000*t_replay:6.2f}ms "
            f"total={1000*total:6.2f}ms tok/round={m}"
        )


if __name__ == "__main__":
    main()
