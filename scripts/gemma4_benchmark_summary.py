#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Extract numeric Gemma throughput points from downloaded CI artifacts."""

import argparse
import json
import re
from pathlib import Path

SHAPE = re.compile(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)\.json$")
KV_SAMPLE = re.compile(
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%"
)


def summarize(root):
    points = []
    for path in root.rglob("benchmark_*.json"):
        match = SHAPE.search(path.name)
        if match is None:
            continue
        data = json.loads(path.read_text())
        points.append(
            {
                "input_tokens": int(match[1]),
                "output_tokens": int(match[2]),
                "concurrency": int(match[3]),
                "requested": int(match[4]),
                "completed": data["completed"],
                "failed": data["failed"],
                "duration_seconds": data["duration"],
                "output_tokens_per_second": data["output_throughput"],
                "requests_per_second": data["request_throughput"],
                "mean_ttft_ms": data["mean_ttft_ms"],
                "mean_tpot_ms": data["mean_tpot_ms"],
                "mean_e2el_ms": data["mean_e2el_ms"],
                "individual_ttft_ms": [
                    round(value * 1000, 3) for value in data["ttfts"]
                ],
            }
        )
    points.sort(key=lambda p: (p["input_tokens"], p["output_tokens"], p["concurrency"]))
    samples = []
    for path in root.rglob("docker_server/*.log"):
        for line in path.open(errors="replace"):
            match = KV_SAMPLE.search(line)
            if match:
                samples.append((int(match[1]), int(match[2]), float(match[3])))
    return {
        "points": points,
        "server_samples": len(samples),
        "waiting_samples": sum(sample[1] > 0 for sample in samples),
        "waiting_sample_fraction": (
            sum(sample[1] > 0 for sample in samples) / len(samples) if samples else None
        ),
        "peak_running_requests": max((s[0] for s in samples), default=None),
        "peak_waiting_requests": max((s[1] for s in samples), default=None),
        "peak_kv_usage_percent": max((s[2] for s in samples), default=None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.artifact_dir)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
