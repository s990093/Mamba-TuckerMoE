# -*- coding: utf-8 -*-
"""
將 GAIR/lima 寫入目前目錄下 dataset/，並轉成與 train.py 相容的單一 uint16 串流 train.bin（mmap）。

格式（僅 instruction，不含 CoT；與 7 個擴充 token 相容）:
  <s> + 多輪:
    <|im_start|>user\n{u}<|im_end|>
    <|im_start|>assistant\n{a}<|im_end|>
  + </s>
末輪僅有 user 時，仍補一輪空 assistant 以維持結構。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
LIMA_HF = DATASET_DIR / "lima_hf"
TOK_OUT = DATASET_DIR / "tokenizer"
BIN_PATH = DATASET_DIR / "train.bin"
META_PATH = DATASET_DIR / "lima_bin_meta.json"

# 7 個擴充（id 32000..32006）：需與「專案內」llama2_tokenizer 的 tokenizer.json 一致。
# 目前為 llama2_tokenizer 內建詞表（非舊版 <|redacted_thinking|> 帶雙層 | 的寫法）。
SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<final>",
    "</final>",
    "[PAD]",
]


def _format_conversation(turns: list[str]) -> str:
    """ChatML-only format (no <s>/<\/s>). Training data and inference are identical.

    完整對話（有 assistant 回覆）：
        <|im_start|>user
        {u}<|im_end|>
        <|im_start|>assistant
        {a}<|im_end|>

    僅 user 最後一輪（推理前綴，等待模型續寫）：
        <|im_start|>user
        {u}<|im_end|>
        <|im_start|>assistant
    """
    if not turns:
        return ""
    parts: list[str] = []
    n = len(turns)
    i = 0
    while i < n:
        u = str(turns[i]).strip()
        parts.append(f"<|im_start|>user\n{u}<|im_end|>\n")
        if i + 1 < n:
            a = str(turns[i + 1]).strip()
            parts.append(f"<|im_start|>assistant\n{a}<|im_end|>\n")
        else:
            # 最後一輪只有 user：開放 assistant 前綴供推理續寫
            parts.append("<|im_start|>assistant\n")
        i += 2
    return "".join(parts)


def expand_conversation_turns_to_single_turn_chatml(turns: list[str]) -> list[str]:
    """
    將多輪對話序列 [u1,a1,u2,a2,...] 拆成多個**單輪** supervised 樣本：
    每則為 `user→assistant` 的完整 ChatML（與 `_format_conversation([u,a])` 一致）。

    - 末尾僅剩 user、尚無對應 assistant 的輪次**略過**（不產生硬標）。
    """
    out: list[str] = []
    i = 0
    while i + 1 < len(turns):
        u = str(turns[i]).strip()
        a = str(turns[i + 1]).strip()
        piece = _format_conversation([u, a])
        if piece.strip():
            out.append(piece)
        i += 2
    return out


_CHATML_ONE_TURN_PAIR_RE = re.compile(
    r"<\|im_start\|>user\n([\s\S]*?)(?:<\|redacted_im_end\|>|<\|im_end\|>)\s*"
    r"<\|im_start\|>assistant\n([\s\S]*?)(?:<\|redacted_im_end\|>|<\|im_end\|>)",
)


def split_multiturn_chatml_text_into_single_turns(full: str) -> list[str]:
    """
    從一串已序列化的 ChatML（可含多輪）抽出每個完整 user→assistant 片段；
    對不上 `_format_conversation` 模板時退回整串，避免資料被靜默丟棄。
    """
    full = full.strip()
    if not full:
        return []
    pieces: list[str] = []
    for m in _CHATML_ONE_TURN_PAIR_RE.finditer(full):
        u, a = m.group(1).strip(), m.group(2).strip()
        txt = _format_conversation([u, a])
        if txt.strip():
            pieces.append(txt)
    if pieces:
        return pieces
    return [full]


def build_tokenizer() -> PreTrainedTokenizerFast:
    """載入專案內 llama2_tokenizer：已含 7 則用校驗 id；僅 32000 基底時才 add_tokens。"""
    tok = AutoTokenizer.from_pretrained(str(ROOT / "llama2_tokenizer"), local_files_only=True)
    n0 = len(tok)
    if n0 == 32007:
        for i, s in enumerate(SPECIAL_TOKENS):
            tid = tok.convert_tokens_to_ids(s)
            if tid != 32000 + i:
                raise SystemExit(
                    f"llama2_tokenizer 與 SPECIAL_TOKENS 第 {i} 項不一致：{s!r} -> id {tid}，預期 {32000 + i}。"
                    f"請對齊 llama2_tokenizer/ 與本檔 SPECIAL_TOKENS 順序與字串。"
                )
    elif n0 == 32000:
        added = tok.add_tokens(SPECIAL_TOKENS)
        if added != 7 or len(tok) != 32007:
            raise SystemExit(
                f"基底 32000 上應 add 7 個 token 得 32007；目前 new_len={len(tok)} added={added}"
                f"（字串衝突時 added 可能 <7）"
            )
    else:
        raise SystemExit(
            f"預期 llama2_tokenizer len 為 32000（再 +7）或已內建 32007，目前 len={n0}"
        )
    if tok.pad_token is None:
        tok.pad_token = SPECIAL_TOKENS[6]  # [PAD]
    # 避免長對話 encode 時噴「超過 model_max_length」警告（實際切片由 train seq_len 決定）
    tok.model_max_length = 1_000_000
    return tok  # type: ignore[return-value]


def main() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("Loading GAIR/lima …")
    ds = load_dataset("GAIR/lima")
    print(f"Saving HF snapshot -> {LIMA_HF}")
    ds.save_to_disk(str(LIMA_HF))

    print("Tokenizing (train split only, 有完整 user/assistant) …")
    tok = build_tokenizer()
    TOK_OUT.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(TOK_OUT))

    all_ids: list[int] = []
    train = ds["train"]
    n_skipped = 0
    for row in train:
        conv = row.get("conversations")
        if not isinstance(conv, list) or len(conv) == 0:
            n_skipped += 1
            continue
        text = _format_conversation([str(x) for x in conv])
        if not text.strip():
            n_skipped += 1
            continue
        ids = tok.encode(text, add_special_tokens=False)
        for wid in ids:
            if not (0 <= wid < 65_535):
                raise SystemExit(f"token id 超出 uint16: {wid}")
        all_ids.extend(ids)

    n_tokens = len(all_ids)
    arr = np.asarray(all_ids, dtype=np.uint16)
    print(f"Writing {n_tokens} tokens -> {BIN_PATH}")
    arr.tofile(str(BIN_PATH))

    meta = {
        "format": "lima_im_chatml_v1",
        "hf_snapshot": str(LIMA_HF.resolve()),
        "train_bin": str(BIN_PATH.resolve()),
        "tokenizer_dir": str(TOK_OUT.resolve()),
        "num_tokens": n_tokens,
        "num_examples": len(train) - n_skipped,
        "num_skipped": n_skipped,
        "vocab_size": len(tok),
        "special_tokens": SPECIAL_TOKENS,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Meta -> {META_PATH}")
    print("Done. train.py: DATA_PATH 指到", BIN_PATH, "，並讓模型用 32007 embedding（resize_token_embeddings）。")


if __name__ == "__main__":
    main()
