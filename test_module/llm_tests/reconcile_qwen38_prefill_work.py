# SPDX-License-Identifier: Apache-2.0
"""Reconcile HTTP uncached input with worker-completed logical prefill chunks.

Run after a fully successful isolated probe, using the complete worker log.
Clock offset is worker wall clock minus client wall clock, measured externally.
Warmup and unrelated worker events outside the HTTP interval are excluded.
"""

import argparse
import json
from pathlib import Path


def reconcile(calls, events, worker_clock_offset):
    if not calls or any(
        "error" in row or "uncached_prompt_tokens" not in row for row in calls
    ):
        raise ValueError(
            "All HTTP requests must complete successfully with token accounting"
        )
    start = min(row["arrival_wall"] for row in calls) + worker_clock_offset
    end = max(row["end_wall"] for row in calls) + worker_clock_offset
    chunks = [
        row
        for row in events
        if row.get("event") == "prefill_chunk" and start <= row["time"] <= end
    ]
    if not chunks:
        raise ValueError("No worker prefill completion events in the matched interval")
    identities = [
        (r["pid"], r["time"], r["start"], r["length"], tuple(r["slots"]))
        for r in chunks
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "Duplicate worker events; do not merge JSONL and stdout copies"
        )
    executed = sum(row["length"] * len(row["slots"]) for row in chunks)
    uncached = sum(row["uncached_prompt_tokens"] for row in calls)
    if executed < uncached:
        raise ValueError(
            "Worker log is incomplete, clocks mismatch, or accounting semantics need investigation"
        )
    return dict(
        status="reconciled_completed_logical_prefill_not_physical_padding",
        worker_clock_offset_seconds=worker_clock_offset,
        worker_interval=[start, end],
        worker_completed_logical_prefill_tokens=executed,
        HTTP_uncached_prompt_tokens=uncached,
        additional_executed_logical_prefill_tokens=executed - uncached,
        eligible_prefix_missed_tokens=sum(
            row["eligible_prefix_missed_tokens"] for row in calls
        ),
        completed_chunk_events=len(chunks),
        scope="Includes repeated logical chunks; excludes padding and partial failed chunks. Isolated server and complete worker log required.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calls", type=Path, required=True)
    p.add_argument("--worker-log", type=Path, required=True)
    p.add_argument("--worker-clock-offset", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    calls = [
        json.loads(line) for line in a.calls.read_text().splitlines() if line.strip()
    ]
    events = []
    for line in a.worker_log.read_text().splitlines():
        marker = "QWEN38_MEMORY_CHECKPOINT "
        payload = line.split(marker, 1)[1] if marker in line else line.strip()
        try:
            row = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("event") == "prefill_chunk":
            events.append(row)
    result = reconcile(calls, events, a.worker_clock_offset)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
