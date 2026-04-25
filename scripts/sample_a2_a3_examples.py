#!/usr/bin/env python3
"""Generate sample outputs from the local A2 and A3 LoRA checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DATASET_DIR = Path("/arxiv/huggingface/paper_text_1m_dedup_v1")
BASE_MODEL = "google/flan-t5-base"
CHECKPOINTS = {
    "A2": Path("models/checkpoints/A2/checkpoint-1000"),
    "A3": Path("models/checkpoints/A3/checkpoint-1000"),
}


def _sample_rows(limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for parquet_path in sorted(DATASET_DIR.glob("train_*.parquet")):
        table = pq.read_table(
            parquet_path,
            columns=["paper_id", "canonical_paper_id", "title", "abstract", "categories"],
        )
        for row in table.to_pylist():
            title = str(row.get("title") or "").strip()
            abstract = str(row.get("abstract") or "").strip()
            if len(title) < 8 or len(abstract) < 300:
                continue
            rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


def _load_model(checkpoint: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        cache_dir="/data/checkpoints",
        local_files_only=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, checkpoint, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def _generate(tokenizer, model, prompts: List[str], *, device: str, max_new_tokens: int) -> List[str]:
    outputs: List[str] = []
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
            )
            outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True).strip())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = _sample_rows(args.limit)
    prompts = [str(row.get("abstract") or "").strip() for row in rows]

    a2_tokenizer, a2_model = _load_model(CHECKPOINTS["A2"], args.device)
    a2_outputs = _generate(a2_tokenizer, a2_model, prompts, device=args.device, max_new_tokens=160)
    del a2_model

    a3_tokenizer, a3_model = _load_model(CHECKPOINTS["A3"], args.device)
    a3_outputs = _generate(a3_tokenizer, a3_model, prompts, device=args.device, max_new_tokens=48)

    examples = []
    for idx, row in enumerate(rows):
        examples.append(
            {
                "paper_id": row.get("canonical_paper_id") or row.get("paper_id"),
                "title": row.get("title"),
                "categories": row.get("categories"),
                "abstract_preview": str(row.get("abstract") or "").strip()[:420],
                "a2_method_summary": a2_outputs[idx],
                "a3_keywords": a3_outputs[idx],
            }
        )

    if args.json:
        print(json.dumps(examples, indent=2, ensure_ascii=False))
        return

    for idx, example in enumerate(examples, 1):
        print(f"\n## Example {idx}: {example['title']}")
        print(f"paper_id: {example['paper_id']} | categories: {example['categories']}")
        print(f"abstract: {example['abstract_preview']}...")
        print(f"A2 method summary: {example['a2_method_summary']}")
        print(f"A3 keywords: {example['a3_keywords']}")


if __name__ == "__main__":
    main()
