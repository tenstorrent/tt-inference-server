#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Side by side table of two video benchmark reports (for example the Python media
server and the C++ server).

Usage:
    python3 scripts/compare_video_benchmark.py report_python.json report_cpp.json \\
        [--labels python cpp] [--specs model_spec_python.json model_spec_cpp.json]

Each report is ``report_*.json`` of a ``benchmarks`` run. ``--specs`` are the matching
``model_spec_*.json`` files; they only add the host name, which matters because two
runs on different machines are not a clean comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (label, path in the "Benchmarks" block, lower is better)
ROWS = [
    ("requests ok", ("num_successful",), None),
    ("steps per second", ("inference_steps_per_second",), False),
    # older reports call it ttft
    ("whole request s", (("end_to_end_latency_s",), ("ttft",)), True),
    ("video size bytes", ("video_bytes",), None),
    ("generation s (submit to completed)", ("generation_seconds",), True),
    ("submit (POST) s", ("phases", "submit_s"), True),
    ("queue wait s", ("phases", "queue_wait_s"), True),
    ("processing s (device + encode)", ("phases", "processing_s"), True),
    ("download s", ("phases", "download_s"), True),
    ("server: whole video s", ("server_side", "e2e_ok_s", "mean"), True),
    ("server: denoise s", ("server_side", "denoise_s", "mean"), True),
    ("server: vae decode s", ("server_side", "vae_decode_s", "mean"), True),
    ("server: queue wait s", ("server_side", "queue_wait_s", "mean"), True),
    ("server: runner time s", ("server_side", "generation_s", "mean"), True),
    ("server: frames per second", ("server_side", "frames_per_second", "mean"), False),
    ("server: frames made", ("server_side", "totals", "frames"), None),
    ("server: steps made", ("server_side", "totals", "denoise_steps"), None),
    ("server: mp4 bytes made", ("server_side", "totals", "output_bytes"), None),
    ("server: seconds of video", ("server_side", "totals", "content_seconds"), None),
    ("videos per second (tput_user)", ("tput_user",), False),
    ("burst videos per minute", ("burst_videos_per_minute",), False),
    ("burst latency p95 s", ("burst_latency_p95",), True),
]


def benchmarks_of(report: dict) -> dict:
    for section in report.get("sections", []):
        if section.get("kind") == "benchmarks":
            return section.get("data", {}).get("Benchmarks", {})
    return {}


def pick(block: dict, path: tuple):
    """The number at ``path``; ``path`` may also be a tuple of alternative paths
    (the first one present wins)."""
    if path and isinstance(path[0], tuple):
        for alternative in path:
            found = pick(block, alternative)
            if found is not None:
                return found
        return None
    value = block
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if isinstance(value, (int, float)) else None


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4g}"


def change(a, b) -> str:
    """b relative to a, in percent; '-' when it cannot be told."""
    if a is None or b is None or a == 0:
        return "-"
    return f"{(b - a) / abs(a) * 100:+.1f} %"


def host_of(spec_path: str | None) -> str:
    if not spec_path:
        return "?"
    info = json.loads(Path(spec_path).read_text()).get("system_info", {})
    host = info.get("host_info", {}).get("Hostname", "?")
    devices = info.get("devices") or [{}]
    dram = devices[0].get("board_info", {}).get("dram_speed", "?")
    return f"{host} (GDDR {dram})"


def table(a: dict, b: dict, labels: list[str], hosts: list[str]) -> str:
    first, second = labels
    lines = [
        f"| | {first} | {second} | {second} vs {first} |",
        "|---|---|---|---|",
        f"| host | {hosts[0]} | {hosts[1]} | |",
    ]
    for label, path, _lower in ROWS:
        x, y = pick(a, path), pick(b, path)
        if x is None and y is None:
            continue
        lines.append(f"| {label} | {fmt(x)} | {fmt(y)} | {change(x, y)} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", nargs=2, help="two report_*.json files")
    parser.add_argument("--labels", nargs=2, default=["first", "second"])
    parser.add_argument("--specs", nargs=2, default=[None, None])
    args = parser.parse_args(argv)
    blocks = [benchmarks_of(json.loads(Path(p).read_text())) for p in args.reports]
    if not all(blocks):
        print("a report has no benchmarks block", file=sys.stderr)
        return 1
    hosts = [host_of(spec) for spec in args.specs]
    print(table(blocks[0], blocks[1], args.labels, hosts))
    if (
        "?" not in hosts[0] + hosts[1]
        and hosts[0].split(" ")[0] != hosts[1].split(" ")[0]
    ):
        print("\nThe runs were on different hosts: read the difference with care.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
