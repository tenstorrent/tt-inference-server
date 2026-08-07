# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""SDXL inference-server perf benchmark.

Sends a fixed sequence of requests against a running media server and prints
client-side timing stats (min / median / p90 / max / mean). Intended for
side-by-side comparison of:
  - trace runner (full-on-device)
  - Forge runner with TTXLA_SDXL_FULL_ON_DEVICE=false (UNet-only)
  - Forge runner with TTXLA_SDXL_FULL_ON_DEVICE=true  (full-on-device)

Per-stage timings (text encoding, UNet diffusion, VAE decode) are emitted by
the runners to the server log; grep hints are printed at the end.

Usage:
    python sdxl_perf_benchmark.py --host localhost --port 8000 \
        --config-label forge-full-on-device --runs 5
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict

import requests

DEFAULT_PROMPT = (
    "A beautiful sunset over a mountain landscape with vibrant colors"
)
DEFAULT_NEGATIVE = "blurry, low quality, distorted"
DEFAULT_SEED = 42
DEFAULT_GUIDANCE = 7.5
DEFAULT_TOKEN = "your-secret-key"
ENDPOINT = "/v1/images/generations"


@dataclass
class RunStats:
    config_label: str
    num_inference_steps: int
    warmup_runs: int
    measured_runs: int
    min_s: float
    median_s: float
    p90_s: float
    max_s: float
    mean_s: float
    per_run_s: list[float]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def send_request(
    session: requests.Session,
    url: str,
    headers: dict,
    payload: dict,
    timeout: int,
) -> float:
    start = time.perf_counter()
    response = session.post(url, json=payload, headers=headers, timeout=timeout)
    duration = time.perf_counter() - start
    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code}: {response.text[:500]}"
        )
    return duration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--config-label",
        required=True,
        help="Free-form label identifying the runner config "
        "(e.g. trace-full-on-device, forge-unet-only, forge-full-on-device)",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument(
        "--timeout", type=int, default=600, help="Per-request timeout in seconds"
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}{ENDPOINT}"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.token}",
    }
    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "guidance_scale": DEFAULT_GUIDANCE,
        "number_of_images": 1,
    }

    print(
        f"[{args.config_label}] Benchmarking {url} "
        f"(steps={args.steps}, warmup={args.warmup}, runs={args.runs})",
        file=sys.stderr,
    )

    with requests.Session() as session:
        for i in range(args.warmup):
            d = send_request(session, url, headers, payload, args.timeout)
            print(f"  warmup {i + 1}/{args.warmup}: {d:.3f}s", file=sys.stderr)

        per_run: list[float] = []
        for i in range(args.runs):
            d = send_request(session, url, headers, payload, args.timeout)
            per_run.append(d)
            print(f"  run {i + 1}/{args.runs}: {d:.3f}s", file=sys.stderr)

    stats = RunStats(
        config_label=args.config_label,
        num_inference_steps=args.steps,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
        min_s=min(per_run),
        median_s=statistics.median(per_run),
        p90_s=_percentile(per_run, 0.9),
        max_s=max(per_run),
        mean_s=statistics.fmean(per_run),
        per_run_s=[round(v, 3) for v in per_run],
    )

    if args.output == "json":
        print(json.dumps(asdict(stats), indent=2))
    else:
        print(f"\n=== {stats.config_label} (steps={stats.num_inference_steps}) ===")
        print(f"  runs:    {stats.per_run_s}")
        print(f"  min:     {stats.min_s:.3f}s")
        print(f"  median:  {stats.median_s:.3f}s")
        print(f"  p90:     {stats.p90_s:.3f}s")
        print(f"  max:     {stats.max_s:.3f}s")
        print(f"  mean:    {stats.mean_s:.3f}s")
        print(
            "\nFor per-stage breakdown, grep the server log for "
            "'Text encoding took', 'UNet', and 'VAE' lines."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
