#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""OpenAI MRCR (v2) runner against an OpenAI-compatible chat endpoint.

The "MRCR v2 8 needle 128k (average)" row of the google/gemma-4-31B-it model card:
8-needle conversations, every token bin up to 128K, scored with the official grader
(SequenceMatcher ratio, 0 unless the required random prefix is present), averaged.
Multi-turn prompts cannot be expressed as an lm-eval doc_to_text, hence this runner.

Bins follow the dataset README: prompt+answer tokens (o200k_base) in
[4096, 8192], (8192, 16384], (16384, 32768], (32768, 65536], (65536, 131072], ...

Example:
  mrcr_runner.py --base-url http://127.0.0.1:8000/v1 --model google/gemma-4-31B-it \
      --needles 8 --max-bin 131072 --concurrency 8 --out mrcr_8needle_128k.json
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from difflib import SequenceMatcher

import pandas as pd
import tiktoken
from huggingface_hub import hf_hub_download
from openai import OpenAI

BIN_EDGES = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]


def grade(response: str, answer: str, prefix: str) -> float:
    """Official grader from the openai/mrcr README."""
    if not (response or "").startswith(prefix):
        return 0.0
    response = response.removeprefix(prefix)
    answer = answer.removeprefix(prefix)
    return float(SequenceMatcher(None, response, answer).ratio())


def bin_of(n_tokens: int):
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        if lo <= n_tokens <= hi if lo == BIN_EDGES[0] else lo < n_tokens <= hi:
            return f"{lo // 1024}k-{hi // 1024}k"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "none"))
    ap.add_argument("--needles", type=int, default=8, choices=(2, 4, 8))
    ap.add_argument("--max-bin", type=int, default=131072, help="keep samples whose prompt+answer tokens <= this")
    ap.add_argument("--min-bin", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap samples per bin (debug)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for i in (0, 1):
        path = hf_hub_download("openai/mrcr", f"{args.needles}needle/{args.needles}needle_{i}.parquet", repo_type="dataset")
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    enc = tiktoken.get_encoding("o200k_base")

    rows = []
    for idx, row in df.iterrows():
        messages = json.loads(row["prompt"])
        n_tok = sum(len(enc.encode(m["content"])) for m in messages) + len(enc.encode(row["answer"]))
        if n_tok > args.max_bin or n_tok < args.min_bin:
            continue
        rows.append({"idx": int(idx), "messages": messages, "answer": row["answer"],
                     "prefix": row["random_string_to_prepend"], "n_tokens": n_tok, "bin": bin_of(n_tok)})
    if args.limit:
        kept, per_bin = [], {}
        for r in rows:
            if per_bin.get(r["bin"], 0) < args.limit:
                kept.append(r); per_bin[r["bin"]] = per_bin.get(r["bin"], 0) + 1
        rows = kept
    print(f"{len(rows)} samples, bins: {sorted(set(r['bin'] for r in rows), key=lambda b: int(b.split('k')[0]))}", flush=True)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout, max_retries=2)

    def run_one(r):
        t0 = time.time()
        try:
            c = client.chat.completions.create(
                model=args.model, messages=r["messages"], max_tokens=args.max_tokens,
                temperature=args.temperature, top_p=args.top_p,
                extra_body={"top_k": args.top_k, "seed": 42},
            )
            text = c.choices[0].message.content or ""
            usage = c.usage.model_dump() if c.usage else None
            err = None
        except Exception as e:  # noqa: BLE001
            text, usage, err = "", None, repr(e)
        return {**{k: r[k] for k in ("idx", "n_tokens", "bin", "prefix")}, "score": grade(text, r["answer"], r["prefix"]),
                "response_head": text[:200], "usage": usage, "error": err, "latency_s": round(time.time() - t0, 1)}

    results, done = [], 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for res in ex.map(run_one, rows):
            results.append(res); done += 1
            if done % 10 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)} mean={sum(x['score'] for x in results)/len(results):.4f}", flush=True)

    by_bin = {}
    for x in results:
        by_bin.setdefault(x["bin"], []).append(x["score"])
    summary = {
        "model": args.model, "needles": args.needles, "max_bin": args.max_bin, "n": len(results),
        "errors": sum(1 for x in results if x["error"]),
        "mean_score": sum(x["score"] for x in results) / len(results) if results else None,
        "per_bin": {b: {"n": len(v), "mean": sum(v) / len(v)} for b, v in sorted(by_bin.items(), key=lambda kv: int(kv[0].split('k')[0]))},
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "max_tokens": args.max_tokens},
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
