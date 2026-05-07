#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


EXPECTED_ADDED = [
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<final>",
    "</final>",
    "[PAD]",
]


def tokenizer_json_vocab(path: Path) -> dict[str, int]:
    data = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
    vocab: dict[str, int] = {}
    for token in data.get("added_tokens", []):
        vocab[token["content"]] = int(token["id"])
    model_vocab = data.get("model", {}).get("vocab")
    if isinstance(model_vocab, dict):
        for token, idx in model_vocab.items():
            vocab[token] = int(idx)
    elif isinstance(model_vocab, list):
        for idx, item in enumerate(model_vocab):
            if isinstance(item, list):
                token = item[0]
            else:
                token = item
            vocab[str(token)] = idx
    return vocab


def hf_vocab(path_or_repo: str) -> dict[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(path_or_repo, trust_remote_code=True, use_fast=True)
    return {str(token): int(idx) for token, idx in tokenizer.get_vocab().items()}


def load_vocab(path_or_repo: str) -> dict[str, int]:
    path = Path(path_or_repo)
    if path.exists() and (path / "tokenizer.json").exists():
        return tokenizer_json_vocab(path)
    return hf_vocab(path_or_repo)


def ordered_added(vocab: dict[str, int], tokens: list[str]) -> list[dict[str, Any]]:
    return [{"token": token, "id": vocab.get(token)} for token in tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare vocabularies token-by-token.")
    parser.add_argument("--base", required=True, help="Tiny-LLM repo id or local snapshot path.")
    parser.add_argument("--target", required=True, help="Local tokenizer directory to compare against.")
    parser.add_argument("--out", type=Path, default=Path("reports/vocab_diff.json"))
    args = parser.parse_args()

    base_vocab = load_vocab(args.base)
    target_vocab = load_vocab(args.target)

    base_tokens = set(base_vocab)
    target_tokens = set(target_vocab)
    only_in_target = sorted(target_tokens - base_tokens, key=lambda t: target_vocab[t])
    only_in_base = sorted(base_tokens - target_tokens, key=lambda t: base_vocab[t])
    shared_with_different_ids = sorted(
        [
            {
                "token": token,
                "base_id": base_vocab[token],
                "target_id": target_vocab[token],
            }
            for token in base_tokens & target_tokens
            if base_vocab[token] != target_vocab[token]
        ],
        key=lambda row: (row["target_id"], row["base_id"], row["token"]),
    )

    target_expected_added = [token for token in only_in_target if token in EXPECTED_ADDED]
    verdict = (
        len(only_in_target) == len(EXPECTED_ADDED)
        and set(only_in_target) == set(EXPECTED_ADDED)
        and not only_in_base
        and not shared_with_different_ids
    )

    report = {
        "base": args.base,
        "target": args.target,
        "base_vocab_len": len(base_vocab),
        "target_vocab_len": len(target_vocab),
        "expected_added_tokens": ordered_added(target_vocab, EXPECTED_ADDED),
        "only_in_target_count": len(only_in_target),
        "only_in_base_count": len(only_in_base),
        "shared_with_different_ids_count": len(shared_with_different_ids),
        "only_in_target": only_in_target,
        "only_in_base_preview": only_in_base[:200],
        "shared_with_different_ids_preview": shared_with_different_ids[:200],
        "target_expected_added": target_expected_added,
        "is_exactly_expected_7_token_delta": verdict,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
