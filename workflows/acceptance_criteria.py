# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Any, Dict, Tuple

from workflows.workflow_types import ReportCheckTypes

FAIL_CHECK = int(ReportCheckTypes.FAIL)
TARGET_CHECK_LEVELS = ("functional", "complete", "target")
CHECK_SUFFIX = "_check"
BENCHMARK_ACTUAL_FIELDS = {
    "ttft": ("ttft",),
    "tput_user": ("tput_user",),
    "tput_prefill": ("tput_prefill",),
    "e2el_ms": ("e2el_ms",),
    "rtr": ("rtr",),
    "concurrency": ("concurrency",),
    "tput": ("tput", "tput_user", "inference_steps_per_second"),
}


def acceptance_criteria_check(report_data: dict) -> Tuple[bool, Dict[str, str]]:
    """Return acceptance status and blocker details for release report data."""
    acceptance_blockers = {}
    acceptance_blockers.update(
        _benchmarks_acceptance(report_data.get("benchmarks_summary"))
    )
    acceptance_blockers.update(_evals_acceptance(report_data.get("evals")))
    acceptance_blockers.update(
        _parameter_support_acceptance(report_data.get("parameter_support_tests"))
    )
    return len(acceptance_blockers) == 0, acceptance_blockers


def format_acceptance_summary_markdown(
    accepted: bool, acceptance_blockers: Dict[str, str]
) -> str:
    """Format acceptance status and blockers as markdown."""
    lines = [
        "### Acceptance Criteria",
        "",
        f"- Acceptance status: `{'PASS' if accepted else 'FAIL'}`",
    ]
    if accepted:
        lines.append("- All acceptance criteria passed.")
        return "\n".join(lines)

    for blocker_key, blocker_message in acceptance_blockers.items():
        lines.append(f"- `{blocker_key}`: {blocker_message}")

    return "\n".join(lines)


def _benchmarks_acceptance(benchmarks_summary: Any) -> Dict[str, str]:
    acceptance_blockers = {}
    if not isinstance(benchmarks_summary, list) or not benchmarks_summary:
        acceptance_blockers["benchmarks_summary"] = (
            "Missing benchmarks summary entries in report data."
        )
        return acceptance_blockers

    first_summary = benchmarks_summary[0]
    if not isinstance(first_summary, dict):
        acceptance_blockers["benchmarks_summary.0"] = (
            "Benchmark summary entry is not a dictionary."
        )
        return acceptance_blockers

    target_checks = first_summary.get("target_checks")
    if not isinstance(target_checks, dict):
        acceptance_blockers["benchmarks_summary.0.target_checks"] = (
            "Missing target_checks in benchmark summary."
        )
        return acceptance_blockers

    found_check = False
    level_failures = {}
    for level_name in TARGET_CHECK_LEVELS:
        level_checks = target_checks.get(level_name)
        if not isinstance(level_checks, dict):
            acceptance_blockers[f"benchmarks.{level_name}"] = (
                f"Missing target_checks.{level_name} in benchmark summary."
            )
            continue

        level_found_check = False
        for check_name, check_value in level_checks.items():
            if not check_name.endswith(CHECK_SUFFIX):
                continue
            level_found_check = True
            found_check = True
            if not _passes_report_check(check_value):
                blocker_key = f"benchmarks.{level_name}.{check_name}"
                level_failures[blocker_key] = _format_benchmark_failure(
                    level_name, check_name, first_summary, level_checks
                )

        if not level_found_check:
            acceptance_blockers[f"benchmarks.{level_name}"] = (
                f"No *_check fields found in target_checks.{level_name}."
            )
            continue

        if _level_passes(level_checks):
            return acceptance_blockers

    if not found_check:
        acceptance_blockers["benchmarks_summary.0.target_checks"] = (
            "No benchmark *_check fields found in target_checks."
        )
        return acceptance_blockers

    acceptance_blockers.update(level_failures)

    return acceptance_blockers


def _evals_acceptance(evals_data: Any) -> Dict[str, str]:
    acceptance_blockers = {}
    if not isinstance(evals_data, list) or not evals_data:
        acceptance_blockers["evals"] = (
            "Missing normalized eval rows with accuracy_check in report data."
        )
        return acceptance_blockers

    for index, eval_row in enumerate(evals_data):
        blocker_key = _eval_blocker_key(eval_row, index)
        if not isinstance(eval_row, dict):
            acceptance_blockers[blocker_key] = "Eval row is not a dictionary."
            continue

        accuracy_check = eval_row.get("accuracy_check")
        if accuracy_check is None:
            acceptance_blockers[blocker_key] = "Missing accuracy_check in eval row."
            continue
        if not _passes_report_check(accuracy_check):
            acceptance_blockers[blocker_key] = _format_eval_failure(eval_row)

    return acceptance_blockers


def _parameter_support_acceptance(parameter_support_tests: Any) -> Dict[str, str]:
    acceptance_blockers = {}
    if not isinstance(parameter_support_tests, dict):
        return acceptance_blockers

    results = parameter_support_tests.get("results")
    if not isinstance(results, dict) or not results:
        return acceptance_blockers

    for test_name, test_results in results.items():
        if not isinstance(test_results, list):
            continue

        for index, test_result in enumerate(test_results):
            if not isinstance(test_result, dict):
                continue

            if test_result.get("status") != "passed":
                blocker_key = f"parameter_support_tests.{test_name}.{index}"
                acceptance_blockers[blocker_key] = _format_parameter_support_failure(
                    test_result
                )

    return acceptance_blockers


def _passes_report_check(check_value: Any) -> bool:
    if check_value is None:
        return False

    try:
        return int(check_value) != FAIL_CHECK
    except (TypeError, ValueError):
        return False


def _level_passes(level_checks: dict) -> bool:
    check_values = [
        check_value
        for check_name, check_value in level_checks.items()
        if check_name.endswith(CHECK_SUFFIX)
    ]
    return bool(check_values) and all(
        _passes_report_check(check_value) for check_value in check_values
    )


def _format_benchmark_failure(
    level_name: str, check_name: str, benchmark_summary: dict, level_checks: dict
) -> str:
    metric_name = check_name[: -len(CHECK_SUFFIX)]
    actual_field_name, actual_value = _resolve_benchmark_actual(
        metric_name, benchmark_summary
    )
    threshold_value = level_checks.get(metric_name)
    ratio_value = level_checks.get(f"{metric_name}_ratio")

    detail_parts = [f"{level_name} {check_name} failed"]
    if actual_field_name is not None:
        detail_parts.append(f"actual {actual_field_name}={_format_value(actual_value)}")
    else:
        detail_parts.append("actual value unavailable in report data")

    if threshold_value is not None:
        detail_parts.append(f"threshold {metric_name}={_format_value(threshold_value)}")
    else:
        detail_parts.append("threshold unavailable in report data")

    if ratio_value not in (None, "Undefined"):
        detail_parts.append(f"ratio={_format_value(ratio_value)}")

    return "; ".join(detail_parts) + "."


def _format_eval_failure(eval_row: dict) -> str:
    reference_field_name = None
    reference_value = None
    ratio_value = None

    if eval_row.get("gpu_reference_score") not in (None, "", "N/A"):
        reference_field_name = "gpu_reference_score"
        reference_value = eval_row.get("gpu_reference_score")
        ratio_value = eval_row.get("ratio_to_reference")
    elif eval_row.get("published_score") not in (None, "", "N/A"):
        reference_field_name = "published_score"
        reference_value = eval_row.get("published_score")
        ratio_value = eval_row.get("ratio_to_published")

    detail_parts = ["Accuracy check failed"]
    if eval_row.get("score") not in (None, "", "N/A"):
        detail_parts.append(f"actual score={_format_value(eval_row.get('score'))}")
    else:
        detail_parts.append("actual score unavailable in report data")

    if reference_field_name is not None:
        detail_parts.append(
            f"reference {reference_field_name}={_format_value(reference_value)}"
        )
    else:
        detail_parts.append("reference value unavailable in report data")

    if ratio_value not in (None, "", "N/A"):
        detail_parts.append(f"ratio={_format_value(ratio_value)}")

    return "; ".join(detail_parts) + "."


def _format_parameter_support_failure(test_result: dict) -> str:
    detail_parts = [
        f"status={_format_value(test_result.get('status'))} vs expected passed"
    ]
    message = test_result.get("message")
    if message:
        detail_parts.append(f"message={message}")
    return "; ".join(detail_parts) + "."


def _eval_blocker_key(eval_row: Any, index: int) -> str:
    if isinstance(eval_row, dict):
        task_name = eval_row.get("task_name")
        if task_name:
            return f"evals.{task_name}"
    return f"evals.index_{index}"


def _resolve_benchmark_actual(
    metric_name: str, benchmark_summary: dict
) -> Tuple[Any, Any]:
    field_names = BENCHMARK_ACTUAL_FIELDS.get(metric_name, (metric_name,))
    for field_name in field_names:
        if field_name in benchmark_summary:
            return field_name, benchmark_summary.get(field_name)
    return None, None


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, ReportCheckTypes):
        return str(int(value))
    return str(value)
