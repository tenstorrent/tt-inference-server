# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Threshold checks against `BenchResult`s, with markdown summaries.

Replaces the duplicated `check_thresholds` shell function from test-gate.yml
(once in cpp-server-benchmarks, once in cpp-server-prefill-decode-split).
"""

from __future__ import annotations

from typing import Optional

from _bench import BenchResult
from _step_summary import write_step_summary


def assert_bench_thresholds(
    result: BenchResult,
    *,
    label: Optional[str] = None,
    mean_tpot_ms_max: Optional[float] = None,
    mean_ttft_ms_max: Optional[float] = None,
) -> None:
    """Assert thresholds and emit a markdown summary section.

    The summary always renders, even when assertions pass — so green CI runs
    keep showing the metrics in the job summary.
    """
    label = label or result.label

    write_step_summary(_summary_section(label, result, mean_tpot_ms_max, mean_ttft_ms_max))

    failures: list[str] = []
    if result.failed > 0:
        failures.append(
            f"{result.failed} request(s) failed (completed={result.completed}, failed={result.failed})"
        )
    if mean_tpot_ms_max is not None and result.mean_tpot_ms > mean_tpot_ms_max:
        failures.append(
            f"mean_tpot_ms {result.mean_tpot_ms} exceeds threshold {mean_tpot_ms_max}ms"
        )
    if mean_ttft_ms_max is not None and result.mean_ttft_ms > mean_ttft_ms_max:
        failures.append(
            f"mean_ttft_ms {result.mean_ttft_ms} exceeds threshold {mean_ttft_ms_max}ms"
        )

    assert not failures, f"[{label}] " + "; ".join(failures)


def _summary_section(
    label: str,
    result: BenchResult,
    tpot_limit: Optional[float],
    ttft_limit: Optional[float],
) -> str:
    tpot_threshold = f"≤ {tpot_limit}ms" if tpot_limit is not None else "—"
    ttft_threshold = f"≤ {ttft_limit}ms" if ttft_limit is not None else "—"
    return (
        f"## {label}\n\n"
        "| Metric | Value | Threshold |\n"
        "|--------|-------|-----------|\n"
        f"| **mean_tpot_ms** | {result.mean_tpot_ms} | {tpot_threshold} |\n"
        f"| **mean_ttft_ms** | {result.mean_ttft_ms} | {ttft_threshold} |\n"
        f"| **completed** | {result.completed} | - |\n"
        f"| **failed** | {result.failed} | 0 |\n\n"
    )
