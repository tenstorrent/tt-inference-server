#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Analyze the EmbeddingService 'Dispatch summary' log line.

The service logs one parseable summary line at graceful shutdown: per-phase
dispatch timings (mean/p99) and a batch-size histogram ("batches: 506x8 2x7").
This script renders both as tables — aligned text in the step log and markdown
in GITHUB_STEP_SUMMARY — and gates on batching quality: the largest observed
batch must equal the configured maximum, and at least --min-full-batch-share
of all dispatched batches must be that size. A drop below the share threshold
means batching degraded (e.g. collapsed toward singles) even if latency
thresholds still pass.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PHASE_LABELS = {
    "queue_wait": "Queue wait (request sat in queue)",
    "batch_collect": "Batch collect (forming the batch)",
    "batch_json_encode": "JSON encode (batch -> worker payload)",
    "pipe_round_trip": "Pipe round trip (IPC + worker compute)",
    "completion": "Completion (decode + callbacks)",
}


def fail(message: str) -> int:
    print(f"::error::{message}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate batching quality from the Dispatch summary line."
    )
    parser.add_argument("log", type=Path, help="Server log file")
    parser.add_argument(
        "--expected-max-batch-size",
        type=int,
        default=8,
        help="Configured max batch size the dispatcher must reach",
    )
    parser.add_argument(
        "--min-full-batch-share",
        type=float,
        default=0.93,
        help="Minimum share of dispatched batches that must be full-size",
    )
    args = parser.parse_args()

    log = args.log.read_text(encoding="utf-8", errors="replace")
    lines = re.findall(r"Dispatch summary:.*", log)
    if not lines:
        return fail(
            "No 'Dispatch summary' line in server log "
            "(server did not shut down gracefully?)"
        )
    summary = lines[-1]

    phases = re.findall(r"(\w+)_ms=([0-9.]+)/([0-9.]+)\(mean/p99\)", summary)
    histogram = [
        (int(n), int(s))
        for n, s in re.findall(r"(\d+)x(\d+)", summary.split("batches:")[-1])
    ]
    if not histogram:
        return fail("Dispatch summary has no batch histogram")

    batches = sum(n for n, _ in histogram)
    requests = sum(n * s for n, s in histogram)
    max_size = max(s for _, s in histogram)
    full_batches = sum(n for n, s in histogram if s == max_size)
    batch_share = full_batches / batches
    request_share = sum(n * s for n, s in histogram if s == max_size) / requests

    # ── Step log: aligned tables ─────────────────────────────────────────
    print(f"Raw summary line:\n  {summary}\n")
    print(f"Batches dispatched: {batches}   Requests served: {requests}\n")
    print(f"{'Dispatch phase':<42}{'Mean (ms)':>12}{'P99 (ms)':>12}")
    print("-" * 66)
    for name, mean, p99 in phases:
        label = PHASE_LABELS.get(name, name)
        print(f"{label:<42}{float(mean):>12.3f}{float(p99):>12.3f}")
    print()
    print(f"{'Batch size':<14}{'Batches':>10}{'Requests':>10}{'Share':>10}")
    print("-" * 44)
    for n, s in sorted(histogram, key=lambda x: -x[1]):
        print(f"{s:<14}{n:>10}{n * s:>10}{n * s / requests:>10.1%}")
    print()
    verdict = (
        f"{full_batches}/{batches} batches ({batch_share:.1%}) at max size "
        f"{max_size}; {request_share:.1%} of requests rode in them"
    )
    print(verdict)

    # ── GitHub run summary: markdown tables ──────────────────────────────
    ok = (
        max_size >= args.expected_max_batch_size
        and batch_share >= args.min_full_batch_share
    )
    md = [
        "## Custom pipelining analysis (embedding_mock, server-side)",
        "",
        f"Batches dispatched: **{batches}** — requests served: **{requests}**",
        "",
        "| Dispatch phase | Mean (ms) | P99 (ms) |",
        "|---|---:|---:|",
    ]
    for name, mean, p99 in phases:
        md.append(
            f"| {PHASE_LABELS.get(name, name)} | {float(mean):.3f} | {float(p99):.3f} |"
        )
    md += [
        "",
        "| Batch size | Batches | Requests | Share |",
        "|---:|---:|---:|---:|",
    ]
    for n, s in sorted(histogram, key=lambda x: -x[1]):
        md.append(f"| {s} | {n} | {n * s} | {n * s / requests:.1%} |")
    md += [
        "",
        ("✅ " if ok else "❌ ")
        + verdict
        + (
            f" (gate: max batch == {args.expected_max_batch_size}, "
            f"full-batch share >= {args.min_full_batch_share:.0%})"
        ),
        "",
    ]
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")

    # ── Gate ─────────────────────────────────────────────────────────────
    if max_size < args.expected_max_batch_size:
        return fail(
            f"Largest batch was {max_size}, expected the configured max of "
            f"{args.expected_max_batch_size} — batching is not filling up"
        )
    if batch_share < args.min_full_batch_share:
        return fail(
            f"Only {batch_share:.1%} of batches reached max size "
            f"(need >= {args.min_full_batch_share:.0%}) — batching degraded "
            "under load"
        )
    print("Batching check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
