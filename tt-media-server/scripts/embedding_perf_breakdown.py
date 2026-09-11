# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Measure the per-batch time breakdown of the BGE embedding runner.

Answers two questions the benchmark numbers cannot:
  1. Is device forward time flat in batch size (trace-padded), or does it grow?
  2. How much do the host-side phases (tokenize, tolist) cost per batch?

Run from the tt-media-server directory with the pinned tt-metal venv:

    source /localdev/jzivanovic/tt-metal-65718bb/python_env/bin/activate
    TT_METAL_HOME=/localdev/jzivanovic/tt-metal-65718bb \
    PYTHONPATH=/localdev/jzivanovic/tt-metal-65718bb \
    python scripts/embedding_perf_breakdown.py
"""

import os
import sys
import time

# Must happen before any tt-media-server import (Settings reads env at import).
os.environ.setdefault("MODEL", "bge-large-en-v1.5")
os.environ.setdefault("DEVICE", "n150")
os.environ.setdefault("DEVICE_IDS", "(0)")
os.environ.setdefault("MAX_BATCH_SIZE", "8")

import asyncio  # noqa: E402

from domain.text_embedding_request import TextEmbeddingRequest  # noqa: E402
from tt_model_runners.embedding_runner import BGELargeENRunner  # noqa: E402

MODEL_ID = "BAAI/bge-large-en-v1.5"
WARM_ITERS = 5
MEASURE_ITERS = 30

# Long enough to truncate to the full 384 tokens, matching the benchmark's ISL.
LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog while the curious cat "
    "watches from the windowsill and contemplates the meaning of life. "
) * 40


def percentile(values, p):
    values = sorted(values)
    k = min(len(values) - 1, max(0, int(round(p / 100 * (len(values) - 1)))))
    return values[k]


def measure_batch(runner, batch_size):
    requests = [
        TextEmbeddingRequest(model=MODEL_ID, input=LONG_TEXT) for _ in range(batch_size)
    ]
    text_inputs = [req.input for req in requests]

    import ttnn

    tok_times, fwd_times, post_times = [], [], []
    token_len = None
    for i in range(WARM_ITERS + MEASURE_ITERS):
        t0 = time.perf_counter()
        tokenized = runner.tokenizer.tokenize(text_inputs, runner.max_model_len)
        token_counts = runner.tokenizer.calculate_token_counts(tokenized, len(requests))
        t1 = time.perf_counter()
        result = runner.model.forward(
            tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask"),
        )
        ttnn.synchronize_device(runner.ttnn_device)
        t2 = time.perf_counter()
        responses = runner._process_result(result, requests, token_counts)
        t3 = time.perf_counter()

        if i >= WARM_ITERS:
            tok_times.append((t1 - t0) * 1000)
            fwd_times.append((t2 - t1) * 1000)
            post_times.append((t3 - t2) * 1000)
        token_len = tokenized["input_ids"].shape[1]
        assert len(responses) == batch_size
        assert len(responses[0].embedding) == 1024

    def stats(ts):
        return (
            sum(ts) / len(ts),
            percentile(ts, 50),
            percentile(ts, 95),
        )

    return {
        "batch_size": batch_size,
        "token_len": token_len,
        "tokenize": stats(tok_times),
        "forward": stats(fwd_times),
        "process_result": stats(post_times),
    }


def main() -> int:
    print("[breakdown] constructing BGELargeENRunner('0')", flush=True)
    runner = BGELargeENRunner("0")

    print("[breakdown] set_device() ...", flush=True)
    runner.set_device()

    print("[breakdown] warmup() ...", flush=True)
    t0 = time.time()
    ok = asyncio.run(runner.warmup())
    print(f"[breakdown] warmup() -> {ok} in {time.time() - t0:.1f}s", flush=True)
    if not ok:
        return 1

    results = []
    for bs in (1, 2, 4, 8):
        print(f"[breakdown] measuring batch_size={bs} ...", flush=True)
        results.append(measure_batch(runner, bs))

    print()
    header = (
        f"{'batch':>5} {'tokens':>6} "
        f"{'tokenize avg/p50/p95':>24} "
        f"{'forward avg/p50/p95':>24} "
        f"{'tolist avg/p50/p95':>24}"
    )
    print(header)
    for r in results:

        def fmt(s):
            return f"{s[0]:7.2f}/{s[1]:6.2f}/{s[2]:6.2f}"

        print(
            f"{r['batch_size']:>5} {r['token_len']:>6} "
            f"{fmt(r['tokenize']):>24} "
            f"{fmt(r['forward']):>24} "
            f"{fmt(r['process_result']):>24}"
        )

    print()
    b1 = next(r for r in results if r["batch_size"] == 1)
    b8 = next(r for r in results if r["batch_size"] == 8)
    fwd1, fwd8 = b1["forward"][0], b8["forward"][0]
    total8 = b8["tokenize"][0] + fwd8 + b8["process_result"][0]
    print(
        f"[breakdown] forward batch1={fwd1:.2f}ms batch8={fwd8:.2f}ms "
        f"(delta {fwd8 - fwd1:+.2f}ms)"
    )
    print(
        f"[breakdown] python-side total for batch8: {total8:.2f}ms "
        f"-> per-worker ceiling {8000 / total8:.1f} req/s, "
        f"32 workers: {32 * 8000 / total8:.0f} req/s, "
        f"{32 * 8000 / total8 * 384 / 1e6:.2f}M tok/s (excl. IPC/JSON/C++)"
    )
    print(
        f"[breakdown] device-only ceiling (forward8): "
        f"{32 * 8000 / fwd8:.0f} req/s, "
        f"{32 * 8000 / fwd8 * 384 / 1e6:.2f}M tok/s"
    )

    runner.close_device()
    print("[breakdown] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
