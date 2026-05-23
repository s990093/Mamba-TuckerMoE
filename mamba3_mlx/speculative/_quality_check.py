"""Quality check: print AR-sampling output vs SJD output side-by-side.

The two are *distribution-equivalent* (by the spec-sampling proof) but
not byte-equal because different RNG paths.  We verify quality by
visual inspection: SJD output should be just as coherent as AR.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.inference.generator import generate
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint
from mamba3_mlx.speculative.jacobi_sampling import jacobi_decode_sampling
from mamba3_mlx.utils.config import GenerationConfig, Mamba3Config
from mamba3_mlx.utils.system_prompts import resolve_system_prompt


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",
                   default=str(REPO_ROOT / "checkpoints" / "latest_sft_cot_model.npz"))
    p.add_argument("--tokenizer_path",
                   default=str(REPO_ROOT / "cot_dataset" / "tokenizer.json"))
    p.add_argument("--prompt", default="Who are you?")
    p.add_argument("--mode", default="self_awareness")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--temp", type=float, default=0.15)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_p", type=float, default=0.85)
    p.add_argument("--min_p", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer_path)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    load_checkpoint(model, args.model_path, dtype=mx.bfloat16)
    mx.eval(model.parameters())

    sys_prompt = resolve_system_prompt(args.mode, None)
    text = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    ids = tok.encode(text, add_special_tokens=False).ids
    bos = tok.token_to_id("<s>")
    if bos is not None:
        ids = [bos] + ids

    gen_cfg = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temp, top_k=args.top_k,
        top_p=args.top_p, min_p=args.min_p,
        seed=args.seed,
    )
    stop_set = set()
    for name in ("<|im_end|>", "</s>"):
        tid = tok.token_to_id(name)
        if tid is not None:
            stop_set.add(tid)

    # ── AR-sampling reference ──
    t0 = time.perf_counter()
    ar = generate(model, ids, gen_cfg,
                  stop_token_ids=sorted(stop_set),
                  no_eos_stop=False)
    t_ar = time.perf_counter() - t0
    ar_text = tok.decode(ar.tokens, skip_special_tokens=False)
    print("=" * 30, "AR-sampling (baseline)", "=" * 30)
    print(f"[{len(ar.tokens)} tok | {t_ar*1000:.0f} ms | "
          f"{len(ar.tokens)/t_ar:.1f} tok/s | stop={ar.stop_reason}]")
    print(ar_text)
    print()

    # ── SJD ──
    # Warm up K-token graph.
    dummy = mx.zeros((1, args.K), dtype=mx.int32)
    _l, _s = model(dummy, states=None); mx.eval(_l); del _l, _s, dummy

    t0 = time.perf_counter()
    sjd = jacobi_decode_sampling(
        model, ids, gen_cfg,
        K=args.K,
        use_ngram=True, ngram_n=4,
        use_retrieval=True,
        seed=args.seed,
        stop_token_ids=sorted(stop_set),
        no_eos_stop=False,
    )
    t_sjd = time.perf_counter() - t0
    sjd_text = tok.decode(sjd.tokens, skip_special_tokens=False)
    speedup = (len(sjd.tokens) / max(sjd.elapsed_decode, 1e-6)) / (
        len(ar.tokens) / max(t_ar, 1e-6)
    )
    print("=" * 30, f"SJD K={args.K} (ours)", "=" * 30)
    print(f"[{len(sjd.tokens)} tok | {t_sjd*1000:.0f} ms | "
          f"{sjd.decode_tps:.1f} tok/s | ARL={sjd.arl:.2f} "
          f"full={100*sjd.n_full_accepts/max(sjd.n_rounds,1):.1f}% | "
          f"speedup={speedup:.2f}x | stop={sjd.stop_reason}]")
    print(sjd_text)


if __name__ == "__main__":
    main()
