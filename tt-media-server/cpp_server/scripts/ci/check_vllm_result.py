#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def as_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value)


def load_scenario(config_path: Path, scenario: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scenarios = config.get("scenarios", {})
    if scenario not in scenarios:
        known = ", ".join(sorted(scenarios))
        raise SystemExit(f"Unknown scenario {scenario!r}. Known scenarios: {known}")
    return scenarios[scenario]


def github_step_summary_path() -> Path | None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return None

    summary = Path(summary_path).resolve(strict=False)
    runner_temp = Path(os.environ.get("RUNNER_TEMP", tempfile.gettempdir())).resolve(
        strict=False
    )
    if not summary.is_relative_to(runner_temp):
        raise SystemExit(f"GITHUB_STEP_SUMMARY must be under RUNNER_TEMP: {summary}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a vLLM bench serve result JSON."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()

    scenario = load_scenario(args.config, args.scenario)
    label = scenario.get("label", args.scenario)
    if not args.result.exists():
        print(f"::error::[{label}] Result file not found: {args.result}")
        return 1

    result = json.loads(args.result.read_text(encoding="utf-8"))
    completed = int(as_number(result.get("completed"), 0))
    failed = int(as_number(result.get("failed"), 0))
    mean_tpot_ms = result.get("mean_tpot_ms")
    mean_ttft_ms = result.get("mean_ttft_ms")

    failures: list[str] = []
    min_completed = int(scenario.get("min_completed", 0))
    max_failed = int(scenario.get("max_failed", 0))
    max_mean_tpot_ms = scenario.get("max_mean_tpot_ms")
    max_mean_ttft_ms = scenario.get("max_mean_ttft_ms")

    if completed < min_completed:
        failures.append(f"completed {completed} is below minimum {min_completed}")
    if failed > max_failed:
        failures.append(f"failed {failed} exceeds maximum {max_failed}")
    if max_mean_tpot_ms is not None and as_number(mean_tpot_ms, float("inf")) > float(
        max_mean_tpot_ms
    ):
        failures.append(
            f"mean_tpot_ms {mean_tpot_ms} exceeds threshold {max_mean_tpot_ms}ms"
        )
    if max_mean_ttft_ms is not None and as_number(mean_ttft_ms, float("inf")) > float(
        max_mean_ttft_ms
    ):
        failures.append(
            f"mean_ttft_ms {mean_ttft_ms} exceeds threshold {max_mean_ttft_ms}ms"
        )

    percentile_thresholds = scenario.get("percentile_thresholds", {})
    for metric, threshold in percentile_thresholds.items():
        if metric in result and as_number(result.get(metric), float("inf")) > float(
            threshold
        ):
            failures.append(
                f"{metric} {result.get(metric)} exceeds threshold {threshold}ms"
            )

    summary = github_step_summary_path()
    lines = [
        f"## {label}",
        "",
        "| Metric | Value | Threshold |",
        "|--------|-------|-----------|",
        f"| **completed** | {completed} | >= {min_completed} |",
        f"| **failed** | {failed} | <= {max_failed} |",
        f"| **mean_tpot_ms** | {format_value(mean_tpot_ms)} | <= {format_value(max_mean_tpot_ms)}ms |",
        f"| **mean_ttft_ms** | {format_value(mean_ttft_ms)} | <= {format_value(max_mean_ttft_ms)}ms |",
    ]
    for metric, threshold in percentile_thresholds.items():
        lines.append(
            f"| **{metric}** | {format_value(result.get(metric))} | <= {threshold}ms |"
        )
    lines.append("")
    if summary:
        with summary.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")

    if failures:
        for failure in failures:
            print(f"::error::[{label}] {failure}")
        return 1

    print(f"[{label}] result passed thresholds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
