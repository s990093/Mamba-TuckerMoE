"""Quality sweep for a single prompt — loads model once, runs many sampling configs.

Usage:
    .venv/bin/python3 mamba3_mlx/tools/sweep_params.py \
        --prompt "Who are you?" --mode self_awareness --max_tokens 220
"""
import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mamba3_mlx.utils.config import Mamba3Config, GenerationConfig
from mamba3_mlx.utils.system_prompts import resolve_system_prompt
from mamba3_mlx.mlx_model.hybrid_model import Mamba3LanguageModel
from mamba3_mlx.mlx_model.weights import load_checkpoint
from mamba3_mlx.inference.generator import generate


# ── Configs to sweep (name → GenerationConfig overrides) ─────────────────────────
SWEEP = [
    ("A_T0.2_k40_rp1.2",  dict(temperature=0.2, top_k=40, top_p=0.85, min_p=0.08, rep_pen=1.20, pres_pen=0.2,  freq_pen=0.05)),
    ("B_T0.2_k30_rp1.18", dict(temperature=0.2, top_k=30, top_p=0.9,  min_p=0.06, rep_pen=1.18, pres_pen=0.15, freq_pen=0.03)),
    ("C_T0.18_k40_rp1.15",dict(temperature=0.18,top_k=40, top_p=0.9,  min_p=0.05, rep_pen=1.15, pres_pen=0.1,  freq_pen=0.02)),
    ("D_T0.25_k40_rp1.2", dict(temperature=0.25,top_k=40, top_p=0.88, min_p=0.06, rep_pen=1.20, pres_pen=0.2,  freq_pen=0.04)),
]


def build_prompt(tok, system_prompt, user_msg, seed_think=True):
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
    return ids


def render(tok, token_ids, struct_tags):
    """Decode keeping <think>/</think>/<final>/</final> visible."""
    parts, seg = [], []
    for tid in token_ids:
        if tid in struct_tags:
            if seg:
                parts.append(tok.decode(seg, skip_special_tokens=True))
                seg = []
            parts.append(struct_tags[tid])
        else:
            seg.append(tid)
    if seg:
        parts.append(tok.decode(seg, skip_special_tokens=True))
    return "".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=str(REPO_ROOT / "checkpoints/v2/latest_sft_cot_model.mlx_bf16.npz"))
    p.add_argument("--tokenizer_path", default=str(REPO_ROOT / "cot_dataset/tokenizer.json"))
    p.add_argument("--prompt", default="Who are you?")
    p.add_argument("--mode", default="self_awareness")
    p.add_argument("--max_tokens", type=int, default=220)
    p.add_argument("--repeat_last_n", type=int, default=128)
    p.add_argument("--seed", default="0", help="comma-separated seeds, e.g. 0,1,7")
    args = p.parse_args()

    print(f"[load] tokenizer: {args.tokenizer_path}", file=sys.stderr)
    tok = Tokenizer.from_file(args.tokenizer_path)

    struct_tags = {}
    for n, s in [("</think>", "\n</think>\n"), ("<final>", "<final>\n"), ("</final>", "\n</final>")]:
        t = tok.token_to_id(n)
        if t is not None:
            struct_tags[t] = s

    print(f"[load] model: {args.model_path}", file=sys.stderr)
    cfg = Mamba3Config(vocab_size=tok.get_vocab_size())
    model = Mamba3LanguageModel(cfg)
    t0 = time.time()
    load_checkpoint(model, args.model_path, dtype=mx.bfloat16)
    mx.eval(model.parameters())
    print(f"[load] done in {time.time()-t0:.2f}s", file=sys.stderr)

    # warmup
    _l, _ = model(mx.zeros((1, 1), dtype=mx.int32), states=None)
    mx.eval(_l)

    sys_prompt = resolve_system_prompt(args.mode, "")
    ids = build_prompt(tok, sys_prompt, args.prompt)

    stop_ids = [t for t in (tok.token_to_id("<|im_end|>"), tok.token_to_id("</s>")) if t is not None]

    seeds = [int(s) for s in str(args.seed).split(",")]
    for name, overrides in SWEEP:
        knobs = "  ".join(f"{k}={v}" for k, v in overrides.items())
        print("\n" + "=" * 78)
        print(f"### {name}")
        print(f"# {knobs}")
        for sd in seeds:
            gen_cfg = GenerationConfig(
                max_tokens=args.max_tokens,
                repeat_last_n=args.repeat_last_n,
                seed=sd,
                **overrides,
            )
            res = generate(model, ids, gen_cfg, stop_token_ids=stop_ids)
            text = render(tok, res.tokens, struct_tags)
            print("-" * 78)
            print(f"[seed={sd}] stop={res.stop_reason}  tokens={len(res.tokens)}  decode={res.decode_tps:.0f} tok/s")
            print(text)


if __name__ == "__main__":
    main()
