#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


DEFAULT_MODEL = "arnir0/Tiny-LLM"
DEFAULT_DATASET = "commonsense_qa"


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_commonsense_qa(example: dict) -> dict:
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    choices = [{"label": label, "text": text} for label, text in zip(labels, texts, strict=True)]
    answer_key = example.get("answerKey")
    answer_text = next((c["text"] for c in choices if c["label"] == answer_key), "")
    return {
        "id": example.get("id"),
        "question": example["question"],
        "choices": choices,
        "answer_key": answer_key,
        "answer": answer_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Tiny-LLM and a commonsense reasoning dataset.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    model_dir = args.root / "models" / args.model.replace("/", "__")
    data_dir = args.root / "datasets" / args.dataset
    report_path = args.root / "reports" / "download_manifest.json"

    snapshot_path = snapshot_download(
        repo_id=args.model,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
    )

    ds = load_dataset(args.dataset)
    exported: dict[str, dict[str, int | str]] = {}
    for split, split_ds in ds.items():
        rows = [normalize_commonsense_qa(row) for row in split_ds]
        out_path = data_dir / f"{split}.jsonl"
        write_jsonl(rows, out_path)
        exported[split] = {"path": str(out_path), "num_rows": len(rows)}

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "model_path": str(snapshot_path),
        "dataset": args.dataset,
        "dataset_path": str(data_dir),
        "splits": exported,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
