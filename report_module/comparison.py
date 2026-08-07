# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Compare two sets of benchmark reports within a relative margin.

Built for the Milestone-0 reproduction check: a Partner reports numbers, we
re-run their submission on our own hardware, and every metric must agree within
a stated margin. The governing rule is that a number we cannot reproduce is not
a result, so this errs toward failing.

Two behaviours follow from that and are not negotiable:

* **Medians across runs, not single runs.** Each side supplies several reports of
  the same configuration (three, per the requirements' run discipline) and the
  median of each metric is compared. A single run would let noise decide a
  commercial outcome.
* **A value missing on one side fails.** A metric present on one side and absent
  on the other, or an operating point only one side ran, fails. Treating a
  one-sided absence as agreement would let an incomplete submission reproduce
  perfectly.

  A metric absent from *both* sides is a different thing: there is nothing to
  disagree about, so it does not fail the reproduction. It is still recorded as a
  gap, because the requirements ask for every metric to be captured — but that is
  an acceptance question, not a reproduction one. Keeping the two separate is
  also what lets a report compare cleanly against itself, which is the sanity
  check this whole tool rests on.

Usage::

    python -m report_module.comparison \\
        --reported partner_run{1,2,3}.json \\
        --measured our_run{1,2,3}.json \\
        --margin 0.05
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# An operating point is identified by these three; everything else in the record
# is a measurement to be compared.
POINT_KEYS = ("concurrency", "input_sequence_length", "output_sequence_length")

# Fields that identify or annotate rather than measure.
NON_METRIC_FIELDS = frozenset(POINT_KEYS) | frozenset(
    {"tool", "model", "device", "timestamp", "target_check", "target_checks"}
)

PointKey = Tuple[Any, ...]


@dataclass
class MetricComparison:
    metric: str
    reported: Optional[float]
    measured: Optional[float]
    relative_difference: Optional[float]
    within_margin: bool
    note: str = ""


@dataclass
class PointComparison:
    point: PointKey
    metrics: List[MetricComparison] = field(default_factory=list)
    #: Metrics neither side measured. Not reproduction failures, but reported so
    #: an incomplete submission is still visible.
    unmeasured: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.metrics) and all(m.within_margin for m in self.metrics)


@dataclass
class ComparisonResult:
    margin: float
    points: List[PointComparison] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only when every point reproduced and nothing was missing."""
        return (
            bool(self.points) and not self.notes and all(p.passed for p in self.points)
        )

    @property
    def failures(self) -> List[Tuple[PointKey, MetricComparison]]:
        return [
            (p.point, m) for p in self.points for m in p.metrics if not m.within_margin
        ]


def _is_benchmark_section(section: Mapping[str, Any]) -> bool:
    """Identify a benchmark section by its content, not its ``kind``.

    ``kind`` is not stable across versions: current parsers emit ``benchmarks``
    while reports produced by earlier builds carry the tool name (``vllm``,
    ``aiperf``). Both must compare, and a future tool should too, so key off the
    operating-point fields that any benchmark record has to have.
    """
    data = section.get("data")
    if not isinstance(data, Mapping):
        return False
    return all(k in data for k in POINT_KEYS)


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def collect_points(
    reports: Sequence[Mapping[str, Any]],
) -> Dict[PointKey, Dict[str, List[float]]]:
    """Gather every numeric metric per operating point across several reports.

    Values from repeated runs of the same point accumulate into one list, which
    is what gets reduced to a median.
    """
    points: Dict[PointKey, Dict[str, List[float]]] = {}
    for report in reports:
        for section in report.get("sections") or []:
            if not isinstance(section, Mapping) or not _is_benchmark_section(section):
                continue
            data = section["data"]
            key = tuple(data.get(k) for k in POINT_KEYS)
            bucket = points.setdefault(key, {})
            for name, value in data.items():
                if name in NON_METRIC_FIELDS:
                    continue
                number = _as_number(value)
                if number is None:
                    # A metric explicitly reported as null is recorded as seen
                    # but unusable, so it fails rather than silently vanishing.
                    bucket.setdefault(name, [])
                    continue
                bucket.setdefault(name, []).append(number)
    return points


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _relative_difference(reported: float, measured: float) -> Optional[float]:
    if reported == 0:
        return 0.0 if measured == 0 else None
    return abs(measured - reported) / abs(reported)


def compare(
    reported_reports: Sequence[Mapping[str, Any]],
    measured_reports: Sequence[Mapping[str, Any]],
    margin: float = 0.05,
) -> ComparisonResult:
    """Compare two sets of reports, one set per side, within ``margin``."""
    result = ComparisonResult(margin=margin)

    reported = collect_points(reported_reports)
    measured = collect_points(measured_reports)

    if not reported:
        result.notes.append("No benchmark points found in the reported set.")
    if not measured:
        result.notes.append("No benchmark points found in the measured set.")

    only_reported = sorted(set(reported) - set(measured), key=str)
    only_measured = sorted(set(measured) - set(reported), key=str)
    for key in only_reported:
        result.notes.append(f"Operating point {key} was reported but never measured.")
    for key in only_measured:
        result.notes.append(f"Operating point {key} was measured but never reported.")

    for key in sorted(set(reported) & set(measured), key=str):
        point = PointComparison(point=key)
        lhs, rhs = reported[key], measured[key]

        for metric in sorted(set(lhs) | set(rhs)):
            a, b = _median(lhs.get(metric, [])), _median(rhs.get(metric, []))

            if a is None and b is None:
                # Neither side measured it. Nothing to disagree about, so this is
                # not a reproduction failure — but record it, since the
                # requirements do ask for every metric to be captured.
                point.unmeasured.append(metric)
                continue

            if a is None or b is None:
                missing = "reported" if a is None else "measured"
                point.metrics.append(
                    MetricComparison(
                        metric=metric,
                        reported=a,
                        measured=b,
                        relative_difference=None,
                        within_margin=False,
                        note=f"no usable value on the {missing} side",
                    )
                )
                continue

            diff = _relative_difference(a, b)
            if diff is None:
                point.metrics.append(
                    MetricComparison(
                        metric=metric,
                        reported=a,
                        measured=b,
                        relative_difference=None,
                        within_margin=False,
                        note="reported 0 but measured non-zero",
                    )
                )
                continue

            point.metrics.append(
                MetricComparison(
                    metric=metric,
                    reported=a,
                    measured=b,
                    relative_difference=diff,
                    within_margin=diff <= margin,
                )
            )

        result.points.append(point)

    return result


def render_markdown(result: ComparisonResult) -> str:
    """Human-readable comparison, suitable for pasting into a submission review."""
    verdict = "PASS" if result.passed else "FAIL"
    lines = [
        "# Reproduction comparison",
        "",
        f"- Margin: ±{result.margin * 100:.3g} %",
        f"- Operating points compared: {len(result.points)}",
        f"- Verdict: **{verdict}**",
        "",
    ]

    if result.notes:
        lines += ["## Blocking issues", ""]
        lines += [f"- {n}" for n in result.notes]
        lines.append("")

    failures = result.failures
    if failures:
        lines += [
            "## Metrics outside the margin",
            "",
            "| Point (conc, isl, osl) | Metric | Reported | Measured | Diff | Note |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for key, m in failures:
            diff = (
                f"{m.relative_difference * 100:.2f} %"
                if m.relative_difference is not None
                else "—"
            )
            rep = "—" if m.reported is None else f"{m.reported:,.4g}"
            mea = "—" if m.measured is None else f"{m.measured:,.4g}"
            lines.append(
                f"| {key} | `{m.metric}` | {rep} | {mea} | {diff} | {m.note} |"
            )
        lines.append("")

    unmeasured = sorted({m for p in result.points for m in p.unmeasured})
    if unmeasured:
        lines += [
            "## Captured by neither side",
            "",
            "Not reproduction failures — there is nothing to disagree about. Listed "
            "because the requirements ask for every metric to be captured, so these "
            "are gaps in the submission rather than mismatches.",
            "",
        ]
        lines += [f"- `{m}`" for m in unmeasured]
        lines.append("")

    lines += [
        "## All points",
        "",
        "| Point (conc, isl, osl) | Compared | Unmeasured | Result |",
        "| --- | --- | --- | --- |",
    ]
    for p in result.points:
        lines.append(
            f"| {p.point} | {len(p.metrics)} | {len(p.unmeasured)} | "
            f"{'PASS' if p.passed else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


def _load(paths: Sequence[str]) -> List[Mapping[str, Any]]:
    return [json.loads(Path(p).read_text()) for p in paths]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--reported", nargs="+", required=True, help="report JSON file(s) as submitted"
    )
    parser.add_argument(
        "--measured",
        nargs="+",
        required=True,
        help="report JSON file(s) from our reproduction",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="relative margin, default 0.05 (±5 %%)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable output instead of markdown",
    )
    args = parser.parse_args(argv)

    result = compare(_load(args.reported), _load(args.measured), args.margin)

    if args.json:
        print(
            json.dumps(
                {
                    "margin": result.margin,
                    "passed": result.passed,
                    "notes": result.notes,
                    "points": [
                        {
                            "point": list(p.point),
                            "passed": p.passed,
                            "metrics": [vars(m) for m in p.metrics],
                            "unmeasured": p.unmeasured,
                        }
                        for p in result.points
                    ],
                },
                indent=2,
            )
        )
    else:
        print(render_markdown(result), end="")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
