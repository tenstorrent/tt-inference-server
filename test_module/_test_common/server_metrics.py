# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""What the video service says about itself while a benchmark runs.

The C++ server exposes its video histograms on ``/metrics`` (``tt_video_*``): the
wait in the queue, the time the worker spent, the whole task and the cost of the
server around the device run. Two scrapes, one before and one after a phase, give
the observations of that phase. A server without these series (or without
``/metrics``) gives an empty result, so the benchmark works against any server.
"""

from __future__ import annotations

import logging
import math
import re

import requests

logger = logging.getLogger(__name__)

SERIES_PREFIX = "tt_video_"
_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf)$"
)
OVERHEAD_PHASES = ("request_write", "shm_enqueue", "response_handling")

Snapshot = dict[tuple[str, str], float]


def parse_video_metrics(text: str) -> Snapshot:
    """``{(name, labels): value}`` of the tt_video_* samples in a /metrics body."""
    out: Snapshot = {}
    for line in text.splitlines():
        if not line.startswith(SERIES_PREFIX):
            continue
        match = _LINE.match(line.strip())
        if not match:
            continue
        try:
            out[(match.group(1), match.group(2) or "")] = float(match.group(3))
        except ValueError:
            pass
    return out


def scrape_video_metrics(base_url: str, timeout: float = 10) -> Snapshot:
    """The server's tt_video_* samples, or {} when it has no /metrics."""
    try:
        response = requests.get(f"{base_url}/metrics", timeout=timeout)
    except Exception as e:  # noqa: BLE001 - metrics must never fail a benchmark
        logger.info(f"No /metrics to read: {e}")
        return {}
    if response.status_code != 200:
        logger.info(f"/metrics answered {response.status_code}; skipping")
        return {}
    return parse_video_metrics(response.text)


def histogram_delta(
    before: Snapshot, after: Snapshot, name: str, label_filter: str = ""
) -> dict:
    """Count, mean and bucket-interpolated p50 and p95 of the observations made
    between two scrapes; ``{"n": 0}`` when there were none."""

    def pick(snapshot: Snapshot, suffix: str) -> dict[str, float]:
        return {
            labels: value
            for (series, labels), value in snapshot.items()
            if series == name + suffix and label_filter in labels
        }

    count = sum(pick(after, "_count").values()) - sum(pick(before, "_count").values())
    total = sum(pick(after, "_sum").values()) - sum(pick(before, "_sum").values())
    if count <= 0:
        return {"n": 0}

    def buckets(snapshot: Snapshot) -> dict[float, float]:
        summed: dict[float, float] = {}
        for labels, value in pick(snapshot, "_bucket").items():
            match = re.search(r'le="([^"]+)"', labels)
            if match:
                edge = math.inf if match.group(1) == "+Inf" else float(match.group(1))
                summed[edge] = summed.get(edge, 0.0) + value
        return summed

    first, last = buckets(before), buckets(after)
    cumulative = [(edge, last[edge] - first.get(edge, 0.0)) for edge in sorted(last)]

    def quantile(q: float) -> float:
        target = q * count
        previous_edge, previous_count = 0.0, 0.0
        for edge, seen in cumulative:
            if seen >= target:
                if math.isinf(edge):
                    return previous_edge
                span = seen - previous_count
                fraction = (target - previous_count) / span if span > 0 else 1.0
                return previous_edge + (edge - previous_edge) * fraction
            previous_edge, previous_count = edge, seen
        return previous_edge

    return {
        "n": int(count),
        "mean": total / count,
        "p50": quantile(0.5),
        "p95": quantile(0.95),
    }


def server_side_video_metrics(before: Snapshot, after: Snapshot) -> dict:
    """The video histograms of one phase, keyed by what they measure; {} when the
    server exposes none of them."""
    if not before or not after:
        return {}
    result = {
        "queue_wait_s": histogram_delta(before, after, "tt_video_queue_wait_seconds"),
        "generation_s": histogram_delta(before, after, "tt_video_generation_seconds"),
        "e2e_ok_s": histogram_delta(
            before, after, "tt_video_e2e_seconds", 'outcome="ok"'
        ),
    }
    for phase in OVERHEAD_PHASES:
        result[f"overhead_{phase}_s"] = histogram_delta(
            before, after, "tt_video_server_overhead_seconds", f'phase="{phase}"'
        )
    return {key: value for key, value in result.items() if value["n"] > 0}
