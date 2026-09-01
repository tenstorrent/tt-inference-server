#!/usr/bin/env python3
"""Predeclare exact bounded GSM8K rows and verify their rendered token envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-output", type=int, default=768)
    parser.add_argument("--max-context", type=int, default=4096)
    args = parser.parse_args()

    rows = [
        dict(row)
        for row in load_dataset("openai/gsm8k", "main", split="test")
    ][: args.samples]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_root.resolve())
    counts = []
    for row in rows:
        prompt = f"Question: {row['question']}\nAnswer:"
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]
        elif hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
        if token_ids and isinstance(token_ids[0], (list, tuple)):
            if len(token_ids) != 1:
                raise RuntimeError("tokenizer returned an unexpected batch")
            token_ids = token_ids[0]
        counts.append(len(token_ids))
    canonical_rows = json.dumps(
        rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    document = {
        "schema": "ttis.gsm8k-context-precheck/v1",
        "dataset": "openai/gsm8k",
        "sample_indices": list(range(args.samples)),
        "selected_rows_sha256": hashlib.sha256(canonical_rows).hexdigest(),
        "rendered_input_counts": counts,
        "max_rendered_input_tokens": max(counts),
        "max_output_tokens": args.max_output,
        "max_total_tokens": max(counts) + args.max_output,
        "server_context": args.max_context,
        "fits": max(counts) + args.max_output <= args.max_context,
        "tokenizer_root": str(args.tokenizer_root.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, sort_keys=True))
    return 0 if document["fits"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
