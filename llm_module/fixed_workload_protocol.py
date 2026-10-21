# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Evidence checks for opt-in fixed token-timing references."""

from __future__ import annotations

import math
import statistics

from .config import LLMRunConfig


def validate_fixed_workload(raw: dict, config: LLMRunConfig) -> None:
    """Reject partial/short workloads, even when the benchmark process exits zero.

    Recompute the three graded metrics from detailed results using the same
    first-to-last-content and whole-window definitions as the frozen baseline.
    The raw file is retained on failure; there is no retry or reference update.
    """
    n, isl, osl = config.num_prompts, config.isl, config.osl
    if (
        n < 1
        or osl < 2
        or (raw["completed"], raw["failed"], raw["num_prompts"], raw["max_concurrency"])
        != (n, 0, n, config.max_concurrency)
    ):
        raise ValueError("Incomplete workload or wrong concurrency/request count")
    if raw["input_lens"] != [isl] * n or raw["output_lens"] != [osl] * n:
        raise ValueError("Actual token lengths differ from the fixed workload")
    if len(raw["errors"]) != n or any(raw["errors"]):
        raise ValueError("Missing request outcomes or failed requests")
    if raw["total_input_tokens"] != isl * n or raw["total_output_tokens"] != osl * n:
        raise ValueError("Total token counts disagree with detailed results")
    if (
        len(raw["ttfts"]) != n
        or len(raw["itls"]) != n
        or any(not row for row in raw["itls"])
    ):
        raise ValueError("Missing detailed timing")
    timings = raw["ttfts"] + [t for row in raw["itls"] for t in row] + [raw["duration"]]
    if not all(
        isinstance(t, (int, float))
        and not isinstance(t, bool)
        and math.isfinite(t)
        and t > 0
        for t in timings
    ):
        raise ValueError("Nonpositive or nonfinite timing")
    measured = {
        "mean_ttft_ms": 1000 * statistics.mean(raw["ttfts"]),
        "mean_tpot_ms": 1000
        * statistics.mean(sum(row) / (osl - 1) for row in raw["itls"]),
        "output_throughput": osl * n / raw["duration"],
    }
    for name, actual in measured.items():
        if not math.isclose(raw[name], actual, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"Detailed timing disagrees with {name}")
