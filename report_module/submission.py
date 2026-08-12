# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Assemble a Milestone-0 ``submission.json`` from completed workflow runs.

Run after the workflows finish; it reads their reports and writes the document
:mod:`report_module.scorecard` scores. Nothing here computes a score, and nothing
here is a workflow change — it is transcription, done by machine because doing it
by hand is the one step of an auditable process that has no audit trail.

    python -m report_module.submission \\
      --partner "Acme Inc" \\
      --run workflow_logs \\
      --out submission.json

What is derived, and what must be supplied
------------------------------------------

**Derived** from the benchmark reports, because the mapping is exact: every graded
point's targets (from its own ``target_checks["target"]``) and measurements, the
fitted scaling exponent per concurrency level, and — when a prefix-cache report is
present — the two bonus figures named in
:data:`report_module.prefix_cache_uplift.SCORED_FIELDS`.

**Supplied** on the command line: the eval margins, contribution quality,
assistance units and the reproduction outcome. These are deliberately not guessed.
The rubric fixes each line's qualifying and excellence values, but how a set of
per-task eval ratios becomes one margin is a scoring policy that is not expressed
anywhere in this repository, and inventing an aggregation would invent published
terms. The per-task ratios found in the reports are written to ``_evidence`` so
whoever sets the margin can see the inputs they are setting it from.

``run_to_run_cov`` is the one middle case: it is computed here, from repeat runs,
under the definition in :data:`COV_DEFINITION`. That definition is this module's
choice rather than a published one, so it is stated in the output and can be
overridden with ``--run-to-run-cov``.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from report_module.prefix_cache_uplift import SCORED_FIELDS

#: How ``run_to_run_cov`` is derived when it is not supplied explicitly. Stated in
#: the output because it is a choice this module makes, not a published rule.
COV_DEFINITION = (
    "max over graded points of (stdev / mean) of mean_ttft_ms across runs; "
    "the worst point governs, so a single unstable operating point cannot be "
    "averaged away by stable ones"
)

#: Report fields a graded point contributes, mapped to the submission's names.
_MEASURED = {
    "p50_ttft": "p50_ttft",
    "p90_ttft": "p90_ttft",
    "p99_ttft": "p99_ttft",
    "prefill_throughput_tok_s": "prefill_throughput",
    "ttft_tail_ratio": "tail_ratio",
    "tput_user": "tput_user",
    "tps_decode_throughput": "decode_throughput",
}

#: Targets live on the point's own strictest tier, so they travel with the
#: measurement they grade and cannot drift from it.
_TARGETS = {
    "ttft": "target_ttft_ms",
    "tput_user": "target_tput_user",
    "tput": "target_decode_throughput",
}


class SubmissionError(RuntimeError):
    """A submission that cannot be assembled correctly, with the reason why."""


def find_report(run_dir: Path) -> Path:
    """Newest ``report_data_*.json`` under ``run_dir``.

    Accepts either a ``workflow_logs`` tree or any directory above one, so a
    Partner can point this at whatever the workflow left behind without having to
    know its layout.
    """
    matches = [
        Path(p)
        for p in glob.glob(str(run_dir / "**" / "report_data_*.json"), recursive=True)
    ]
    if not matches:
        raise SubmissionError(
            f"No report_data_*.json under {run_dir}. Point --run at the directory "
            f"a workflow wrote (its workflow_logs tree, or any parent of it)."
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def _blocks(report: Mapping[str, Any], kind: str) -> List[Mapping[str, Any]]:
    return [s for s in report.get("sections", []) if s.get("kind") == kind]


def graded_points(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every benchmark point carrying targets, as scorecard ``GradedPoint`` dicts.

    Points without targets are skipped rather than defaulted: an ungraded point
    has no qualifying value to sit against, and inventing one would put a score on
    something nobody agreed to grade.
    """
    points: List[Dict[str, Any]] = []
    for block in _blocks(report, "benchmarks"):
        data = block.get("data") or {}
        checks = data.get("target_checks")
        if not isinstance(checks, Mapping):
            continue
        target = checks.get("target")
        if not isinstance(target, Mapping):
            continue
        point: Dict[str, Any] = {
            "concurrency": data.get("concurrency"),
            "input_length": data.get("input_sequence_length"),
        }
        for src, dst in _TARGETS.items():
            point[dst] = target.get(src)
        for src, dst in _MEASURED.items():
            point[dst] = data.get(src)
        if point["concurrency"] is None or point["input_length"] is None:
            raise SubmissionError(
                f"Benchmark block {block.get('title')!r} carries targets but no "
                f"concurrency/input length, so it cannot be placed in the sweep."
            )
        points.append(point)
    return points


def scaling_exponents(report: Mapping[str, Any]) -> Dict[int, Optional[float]]:
    """Fitted TTFT growth exponent per concurrency level."""
    out: Dict[int, Optional[float]] = {}
    for block in _blocks(report, "benchmarks"):
        data = block.get("data") or {}
        exponent = data.get("ttft_scaling_exponent")
        concurrency = data.get("concurrency")
        if exponent is not None and concurrency is not None:
            out[int(concurrency)] = exponent
    return out


def bonus_figures(report: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Prefix-cache hit rate and TTFT uplift, when a prefix-cache run is present.

    Reads only the fields ``prefix_cache_uplift`` declares as scored, so this
    cannot drift from what that module attaches.
    """
    hit_rate_field, uplift_field = SCORED_FIELDS[0], SCORED_FIELDS[1]
    found: Dict[str, Optional[float]] = {}
    for block in report.get("sections", []):
        data = block.get("data") or {}
        if hit_rate_field in data and "prefix_cache_hit_rate" not in found:
            found["prefix_cache_hit_rate"] = data[hit_rate_field]
        if uplift_field in data and "ttft_uplift" not in found:
            found["ttft_uplift"] = data[uplift_field]
    return found


def eval_evidence(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Per-task eval ratios, recorded so a supplied margin can be checked."""
    rows = []
    for block in _blocks(report, "evals"):
        data = block.get("data") or {}
        if "task_name" not in data:
            continue
        rows.append(
            {
                "task_name": data.get("task_name"),
                "score": data.get("score"),
                "ratio_to_reference": data.get("ratio_to_reference"),
                "ratio_to_published": data.get("ratio_to_published"),
            }
        )
    return rows


def merge_points(reports: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Graded points across every run, one per ``(concurrency, input_length)``.

    Merging rather than reading only the first run covers both shapes a Partner
    can legitimately produce: one run holding the whole sweep, and one run per
    concurrency corner. Where a point appears in several runs — which is what
    repeat runs for :func:`run_to_run_cov` are — the earliest run wins, so the
    scored figures come from one consistent run rather than a mix.
    """
    merged: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for report in reports:
        for point in graded_points(report):
            merged.setdefault((point["concurrency"], point["input_length"]), point)
    return [merged[k] for k in sorted(merged)]


def run_to_run_cov(runs: Sequence[Mapping[str, Any]]) -> Tuple[Optional[float], Dict]:
    """Coefficient of variation across repeat runs, per :data:`COV_DEFINITION`.

    Returns ``(value, per_point_evidence)``. ``None`` when no point was measured
    twice — a single measurement is not a variation, and reporting 0.0 for it would
    award full marks for having done nothing. Points measured once are skipped
    rather than treated as perfectly stable, so splitting a sweep across runs
    cannot manufacture a good score.
    """
    series: Dict[Tuple[int, int], List[float]] = {}
    for report in runs:
        for block in _blocks(report, "benchmarks"):
            data = block.get("data") or {}
            mean = data.get("mean_ttft_ms")
            key = (data.get("concurrency"), data.get("input_sequence_length"))
            if (
                mean is None
                or None in key
                or not isinstance(data.get("target_checks"), Mapping)
            ):
                continue
            series.setdefault(key, []).append(float(mean))

    evidence, worst = {}, None
    for key, values in sorted(series.items()):
        if len(values) < 2:
            continue
        mean = statistics.fmean(values)
        if mean <= 0:
            continue
        cov = statistics.stdev(values) / mean
        evidence[f"conc{key[0]}_isl{key[1]}"] = round(cov, 6)
        worst = cov if worst is None else max(worst, cov)
    return (round(worst, 6) if worst is not None else None), evidence


def validate(points: Sequence[Mapping[str, Any]]) -> List[str]:
    """Problems that would make the scorecard wrong or unbuildable.

    Checked here, where the message can name the workflow argument that caused it,
    rather than surfacing later as a weighting error from deep inside scoring.
    """
    problems: List[str] = []
    if not points:
        problems.append(
            "No graded points found. The sweep registered no targets for this "
            "model/device, so nothing can be scored."
        )
        return problems

    levels = sorted({p["concurrency"] for p in points})
    if len(levels) != 2:
        problems.append(
            f"Found {len(levels)} concurrency level(s) {levels}; the rubric weights "
            f"exactly two (an idle corner and a loaded one, Appendix B.5). If extra "
            f"levels appeared, set ONLY_BENCHMARK_TARGETS=1 so the run keeps to the "
            f"registered points."
        )

    for point in points:
        missing = [k for k, v in point.items() if v is None]
        if missing:
            problems.append(
                f"Point (concurrency {point['concurrency']}, ISL "
                f"{point['input_length']}) is missing {', '.join(sorted(missing))}."
            )
    return problems


def build(
    run_dirs: Sequence[Path],
    *,
    partner: str,
    model: str = "",
    once: Optional[Mapping[str, Optional[float]]] = None,
    reproduced_first_attempt: Optional[bool] = None,
    cov_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble the submission document from one or more completed runs."""
    if not run_dirs:
        raise SubmissionError("At least one --run directory is required.")

    reports = [json.loads(find_report(Path(d)).read_text()) for d in run_dirs]
    primary = reports[0]

    points = merge_points(reports)
    problems = validate(points)
    if problems:
        raise SubmissionError(
            "Cannot assemble a scoreable submission:\n  - " + "\n  - ".join(problems)
        )

    merged: Dict[str, Optional[float]] = dict(once or {})
    for report in reports:
        for key, value in bonus_figures(report).items():
            merged.setdefault(key, value)

    cov, cov_evidence = run_to_run_cov(reports)
    if cov_override is not None:
        cov = cov_override
    if cov is not None:
        merged["run_to_run_cov"] = cov

    exponents: Dict[int, Optional[float]] = {}
    for report in reports:
        for level, value in scaling_exponents(report).items():
            exponents.setdefault(level, value)

    return {
        "partner": partner,
        "model": model or str(primary.get("metadata", {}).get("model_name", "")),
        "points": points,
        "scaling_exponents": {str(k): exponents[k] for k in sorted(exponents)},
        "once": merged,
        "reproduced_first_attempt": reproduced_first_attempt,
        "_evidence": {
            "reports": [str(find_report(Path(d))) for d in run_dirs],
            "runs_used_for_cov": len(reports),
            "cov_definition": COV_DEFINITION if cov_override is None else "supplied",
            "cov_per_point": cov_evidence,
            "evals": eval_evidence(primary),
            "error_request_count_per_point": [
                (b.get("data") or {}).get("error_request_count")
                for b in _blocks(primary, "benchmarks")
            ],
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a Milestone-0 submission.json from completed runs.",
    )
    parser.add_argument("--partner", required=True, help="Partner name")
    parser.add_argument("--model", default="", help="Model, defaults to the report's")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="DIR",
        help="A completed run's output directory. Repeat for each run; the first "
        "supplies the graded points, all of them supply run-to-run variation.",
    )
    parser.add_argument("--out", default="submission.json", help="Output path")
    for line in ("agentic-eval", "standard-eval", "contribution-quality"):
        parser.add_argument(f"--{line}", type=float, default=None)
    parser.add_argument("--technical-assistance", type=float, default=None)
    parser.add_argument(
        "--run-to-run-cov",
        type=float,
        default=None,
        help="Override the computed coefficient of variation.",
    )
    reproduced = parser.add_mutually_exclusive_group()
    reproduced.add_argument(
        "--reproduced-first-attempt", dest="reproduced", action="store_true"
    )
    reproduced.add_argument(
        "--not-reproduced-first-attempt", dest="reproduced", action="store_false"
    )
    parser.set_defaults(reproduced=None)
    args = parser.parse_args(argv)

    once = {
        "agentic_eval": args.agentic_eval,
        "standard_eval": args.standard_eval,
        "contribution_quality": args.contribution_quality,
        "technical_assistance": args.technical_assistance,
    }
    try:
        doc = build(
            [Path(r) for r in args.run],
            partner=args.partner,
            model=args.model,
            once={k: v for k, v in once.items() if v is not None},
            reproduced_first_attempt=args.reproduced,
            cov_override=args.run_to_run_cov,
        )
    except SubmissionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    levels = sorted({p["concurrency"] for p in doc["points"]})
    unset = sorted(k for k, v in doc["once"].items() if v is None)
    print(f"wrote {args.out}: {len(doc['points'])} graded points, concurrency {levels}")
    if doc["reproduced_first_attempt"] is None:
        unset.append("reproduced_first_attempt")
    if unset:
        print(
            "note: these lines have no value and will score zero as unscoreable: "
            + ", ".join(unset)
        )
    return 0


__all__ = [
    "COV_DEFINITION",
    "SubmissionError",
    "build",
    "eval_evidence",
    "find_report",
    "graded_points",
    "run_to_run_cov",
    "scaling_exponents",
    "validate",
]


if __name__ == "__main__":
    import sys

    sys.exit(main())
