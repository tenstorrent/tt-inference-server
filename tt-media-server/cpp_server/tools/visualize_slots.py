#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Visualize KV cache slot allocation events from JSONL event log.

Usage:
    python3 visualize_slots.py slot_events.jsonl [--output report.html]

The event log is produced by running the server with:
    SLOT_EVENT_LOG=slot_events.jsonl LLM_DEVICE_BACKEND=mock_pipeline ./build/tt_media_server_cpp

Events are emitted by SlotPoolMemoryManager (source "MM"):
    POOL_INITIALIZED, SLOT_ALLOCATED, SLOT_DEALLOCATED, ALLOC_EXHAUSTED, DEALLOC_UNKNOWN_SLOT
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def load_events(path: str) -> list[dict]:
    events = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {line_no}: {e}", file=sys.stderr)
    return events


def compute_stats(events: list[dict]) -> dict:
    pool_size = None
    free_over_time: list[tuple[float, int]] = []
    alloc_count = 0
    dealloc_count = 0
    exhaustion_count = 0
    unknown_dealloc_count = 0

    slot_intervals: defaultdict[int, list] = defaultdict(list)

    for ev in events:
        t_ms = ev["t_us"] / 1000.0
        event = ev.get("event", "")

        if event == "POOL_INITIALIZED":
            pool_size = ev.get("pool_size", 0)
            free_over_time.append((t_ms, pool_size))

        elif event == "SLOT_ALLOCATED":
            slot_id = ev.get("slot_id", -1)
            free_count = ev.get("free", 0)
            alloc_count += 1
            free_over_time.append((t_ms, free_count))
            slot_intervals[slot_id].append({"start": t_ms, "end": None})

        elif event == "SLOT_DEALLOCATED":
            slot_id = ev.get("slot_id", -1)
            free_count = ev.get("free", 0)
            dealloc_count += 1
            free_over_time.append((t_ms, free_count))
            for interval in reversed(slot_intervals.get(slot_id, [])):
                if interval["end"] is None:
                    interval["end"] = t_ms
                    break

        elif event == "ALLOC_EXHAUSTED":
            exhaustion_count += 1
            free_over_time.append((t_ms, 0))

        elif event == "DEALLOC_UNKNOWN_SLOT":
            unknown_dealloc_count += 1

    hold_times = []
    for intervals in slot_intervals.values():
        for iv in intervals:
            if iv["end"] is not None:
                hold_times.append(iv["end"] - iv["start"])

    return {
        "pool_size": pool_size,
        "total_events": len(events),
        "alloc_count": alloc_count,
        "dealloc_count": dealloc_count,
        "exhaustion_count": exhaustion_count,
        "unknown_dealloc_count": unknown_dealloc_count,
        "free_over_time": free_over_time,
        "slot_intervals": dict(slot_intervals),
        "hold_times": hold_times,
    }


def print_text_report(stats: dict):
    pool = stats["pool_size"] or "unknown"
    print(f"\n{'='*60}")
    print(f"  KV Cache Slot Pool Report (observer: MemoryManager)")
    print(f"{'='*60}")
    print(f"  Pool size:               {pool}")
    print(f"  Total events:            {stats['total_events']}")
    print(f"  Allocations:             {stats['alloc_count']}")
    print(f"  Deallocations:           {stats['dealloc_count']}")
    print(f"  Pool exhaustions:        {stats['exhaustion_count']}")
    print(f"  Unknown deallocs:        {stats['unknown_dealloc_count']}")

    hold = stats["hold_times"]
    if hold:
        hold_sorted = sorted(hold)
        p50 = hold_sorted[len(hold_sorted) // 2]
        p99 = hold_sorted[int(len(hold_sorted) * 0.99)]
        print(f"\n  Slot hold time (alloc → dealloc):")
        print(f"    min:  {min(hold):.2f} ms")
        print(f"    p50:  {p50:.2f} ms")
        print(f"    p99:  {p99:.2f} ms")
        print(f"    max:  {max(hold):.2f} ms")

    free_ts = stats["free_over_time"]
    if free_ts:
        free_vals = [f for _, f in free_ts]
        print(f"\n  Free slot count range:   [{min(free_vals)}, {max(free_vals)}]")
        duration_s = (free_ts[-1][0] - free_ts[0][0]) / 1000.0
        print(f"  Recording duration:      {duration_s:.1f} s")

    print(f"{'='*60}\n")


def generate_html_report(stats: dict, output_path: str):
    if not HAS_PLOTLY:
        print("Warning: plotly not installed, skipping HTML report. Install with: pip install plotly",
              file=sys.stderr)
        return

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Free Slots Over Time",
            "Slot Hold Time Distribution (ms)",
            "Slot Occupancy Timeline",
        ),
        row_heights=[0.25, 0.25, 0.5],
        vertical_spacing=0.08,
    )

    free_ts = stats["free_over_time"]
    if free_ts:
        times = [t / 1000.0 for t, _ in free_ts]
        frees = [f for _, f in free_ts]
        fig.add_trace(
            go.Scatter(x=times, y=frees, mode="lines", name="Free slots",
                       line=dict(color="steelblue")),
            row=1, col=1,
        )
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Free slots", row=1, col=1)

    hold = stats["hold_times"]
    if hold:
        fig.add_trace(
            go.Histogram(x=hold, nbinsx=50, name="Slot hold time",
                         marker_color="coral"),
            row=2, col=1,
        )
        fig.update_xaxes(title_text="Hold time (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)

    intervals = stats["slot_intervals"]
    if intervals:
        max_slot = max(intervals.keys())
        last_t = free_ts[-1][0] / 1000.0 if free_ts else 0
        for slot_id in range(max_slot + 1):
            for iv in intervals.get(slot_id, []):
                start_s = iv["start"] / 1000.0
                end_s = (iv["end"] / 1000.0) if iv["end"] else last_t
                fig.add_trace(
                    go.Bar(
                        x=[end_s - start_s], y=[slot_id],
                        base=[start_s],
                        orientation="h",
                        marker_color="steelblue",
                        opacity=0.7,
                        showlegend=False,
                        hovertext=f"Slot {slot_id}<br>"
                                  f"Start: {start_s:.2f}s<br>End: {end_s:.2f}s<br>"
                                  f"Hold: {(end_s - start_s)*1000:.0f}ms",
                        hoverinfo="text",
                    ),
                    row=3, col=1,
                )
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Slot ID", row=3, col=1)

    pool = stats["pool_size"] or "?"
    title = (
        f"KV Cache Slot Report — Pool: {pool} | "
        f"Allocs: {stats['alloc_count']} | "
        f"Deallocs: {stats['dealloc_count']} | "
        f"Exhaustions: {stats['exhaustion_count']}"
    )
    fig.update_layout(
        title_text=title,
        height=1200,
        showlegend=True,
        barmode="overlay",
    )

    fig.write_html(output_path)
    print(f"HTML report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize KV cache slot allocation events")
    parser.add_argument("input", help="Path to JSONL event log file")
    parser.add_argument("--output", "-o", default=None, help="Output HTML report path")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    events = load_events(args.input)
    if not events:
        print("No events found in file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(events)} events from {args.input}")

    stats = compute_stats(events)
    print_text_report(stats)

    if args.output:
        generate_html_report(stats, args.output)
    elif HAS_PLOTLY:
        output = Path(args.input).stem + "_report.html"
        generate_html_report(stats, output)


if __name__ == "__main__":
    main()
