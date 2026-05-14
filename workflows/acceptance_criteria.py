# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from typing import Any, Dict, Iterable, List, Optional, Tuple

from workflows.workflow_types import (
    ModelStatusTypes,
    ReportCheckTypes,
    WorkflowType,
)

FAIL_CHECK = int(ReportCheckTypes.FAIL)
TARGET_CHECK_LEVELS = ("functional", "complete", "target")
CHECK_SUFFIX = "_check"
BENCHMARK_ACTUAL_FIELDS = {
    "ttft": ("ttft",),
    "latency": ("latency",),
    "tput_user": ("tput_user",),
    "tput_prefill": ("tput_prefill",),
    "e2el_ms": ("e2el_ms",),
    "rtr": ("rtr",),
    "concurrency": ("concurrency",),
    "tput": ("tput", "tput_user", "inference_steps_per_second"),
}


def acceptance_criteria_check(
    report_data: dict,
    model_status: Optional[ModelStatusTypes] = None,
    known_issues: Optional[Iterable[Any]] = None,
) -> Tuple[bool, Dict[str, str], Dict[str, Any]]:
    raw_blockers: Dict[str, str] = {}
    raw_blockers.update(_benchmarks_acceptance(report_data.get("benchmarks_summary")))
    raw_blockers.update(_evals_acceptance(report_data.get("evals")))
    raw_blockers.update(
        _parameter_support_acceptance(report_data.get("parameter_support_tests"))
    )
    raw_blockers.update(_server_tests_acceptance(report_data.get("server_tests")))

    target_checks = _extract_first_target_checks(report_data.get("benchmarks_summary"))
    if model_status is None:
        enforced_tiers: List[str] = list(TARGET_CHECK_LEVELS)
        informational_tiers: List[str] = []
        status_name = "UNSPECIFIED"
    else:
        enforced_tiers = list(model_status.required_target_tiers)
        informational_tiers = [
            tier for tier in TARGET_CHECK_LEVELS if tier not in enforced_tiers
        ]
        status_name = model_status.name

    informational_blockers: Dict[str, str] = {}
    for blocker_key in list(raw_blockers.keys()):
        tier = _benchmark_tier_from_key(blocker_key)
        if tier is None or tier in enforced_tiers:
            continue
        informational_blockers[blocker_key] = raw_blockers.pop(blocker_key)

    masked_blockers: Dict[str, Dict[str, str]] = {}
    known_issue_records = _serialize_known_issues(known_issues)
    if known_issues:
        for blocker_key in list(raw_blockers.keys()):
            workflow, task_name = _classify_blocker(blocker_key)
            if workflow is None:
                continue
            matching = _find_matching_known_issue(known_issues, workflow, task_name)
            if matching is None:
                continue
            masked_blockers[blocker_key] = {
                "message": raw_blockers.pop(blocker_key),
                "reason": matching.reason,
                "workflow_type": matching.workflow_type.name,
                "task_name": matching.task_name,
            }

    failed_enforced_tiers: List[str] = []
    if target_checks:
        failed_enforced_tiers = _failed_tiers(target_checks, enforced_tiers)

    accepted = len(raw_blockers) == 0
    waivers_applied_count = len(masked_blockers) + len(informational_blockers)
    acceptance_before_masking = accepted and waivers_applied_count == 0

    enforcement_metadata: Dict[str, Any] = {
        "enforcement_result": "PASS" if accepted else "FAIL",
        "model_status": status_name,
        "enforced_tiers": enforced_tiers,
        "informational_tiers": informational_tiers,
        "failed_enforced_tiers": failed_enforced_tiers,
        "informational_blockers": informational_blockers,
        "masked_blockers": masked_blockers,
        "known_issues_declared": known_issue_records,
        "waivers_applied_count": waivers_applied_count,
        "acceptance_before_masking": acceptance_before_masking,
    }

    return accepted, raw_blockers, enforcement_metadata


def format_acceptance_summary_markdown(
    accepted: bool,
    acceptance_blockers: Dict[str, str],
    enforcement_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    informational_blockers = (
        enforcement_metadata.get("informational_blockers")
        if enforcement_metadata
        else None
    ) or {}
    masked_blockers = (
        enforcement_metadata.get("masked_blockers") if enforcement_metadata else None
    ) or {}
    waivers_applied_count = len(masked_blockers) + len(informational_blockers)

    status_label = "PASS" if accepted else "FAIL"
    status_line = f"- Acceptance status: `{status_label}`"
    if accepted and waivers_applied_count > 0:
        status_line = (
            f"- Acceptance status: `PASS` "
            f"(with {waivers_applied_count} waiver(s) applied; "
            f"would have FAILED without masking)"
        )

    lines = [
        "### Acceptance Criteria",
        "",
        status_line,
    ]

    if enforcement_metadata is not None:
        lines.append(
            f"- Model status: `{enforcement_metadata.get('model_status', 'UNSPECIFIED')}`"
        )
        enforced = enforcement_metadata.get("enforced_tiers") or []
        informational = enforcement_metadata.get("informational_tiers") or []
        lines.append(
            f"- Enforced tiers: `{enforced or 'none'}` | Informational tiers: `{informational or 'none'}`"
        )

    if accepted and not acceptance_blockers:
        if waivers_applied_count == 0:
            lines.append("- All acceptance criteria passed (no waivers applied).")
        else:
            lines.append(
                f"- Acceptance criteria passed after applying "
                f"{waivers_applied_count} waiver(s); see sections below."
            )
    else:
        for blocker_key, blocker_message in acceptance_blockers.items():
            lines.append(f"- `{blocker_key}`: {blocker_message}")

    if informational_blockers:
        lines.append("")
        lines.append("#### Informational (tier above model status)")
        for blocker_key, blocker_message in informational_blockers.items():
            lines.append(f"- `{blocker_key}`: {blocker_message}")

    if masked_blockers:
        lines.append("")
        lines.append("#### Masked (known issues)")
        for blocker_key, mask_record in masked_blockers.items():
            reason = mask_record.get("reason", "")
            message = mask_record.get("message", "")
            lines.append(f"- `{blocker_key}`: {message} (waived: {reason})")

    return "\n".join(lines)


def enforce_acceptance_criteria(
    target_checks: Dict[str, Dict[str, Any]], model_status: ModelStatusTypes
) -> Dict[str, Any]:
    required_tiers = list(model_status.required_target_tiers)
    informational_tiers = [
        tier for tier in target_checks.keys() if tier not in required_tiers
    ]
    failed_enforced_tiers = _failed_tiers(target_checks, required_tiers)
    enforcement_result = "FAIL" if failed_enforced_tiers else "PASS"
    return {
        "enforcement_result": enforcement_result,
        "model_status": model_status.name,
        "enforced_tiers": required_tiers,
        "informational_tiers": informational_tiers,
        "failed_enforced_tiers": failed_enforced_tiers,
    }


def _benchmarks_acceptance(benchmarks_summary: Any) -> Dict[str, str]:
    acceptance_blockers: Dict[str, str] = {}
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
                acceptance_blockers[blocker_key] = _format_benchmark_failure(
                    level_name, check_name, first_summary, level_checks
                )

        if not level_found_check:
            acceptance_blockers[f"benchmarks.{level_name}"] = (
                f"No *_check fields found in target_checks.{level_name}."
            )

    if not found_check:
        acceptance_blockers["benchmarks_summary.0.target_checks"] = (
            "No benchmark *_check fields found in target_checks."
        )

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


def _server_tests_acceptance(server_tests: Any) -> Dict[str, str]:
    """Collect blockers from the server_tests / spec_tests workflow output.

    `output_data["server_tests"]` is a list of per-file report dicts produced
    by `server_tests/run_spec_tests.py` (registered as WorkflowType.SPEC_TESTS).
    Each dict carries `tests: [{test_name, success: bool, error, ...}]`. A
    test with `success` falsy (or non-bool) is considered failing.
    """
    acceptance_blockers: Dict[str, str] = {}
    if not isinstance(server_tests, list) or not server_tests:
        return acceptance_blockers

    counters_by_name: Dict[str, int] = {}
    for report in server_tests:
        if not isinstance(report, dict):
            continue
        tests = report.get("tests")
        if not isinstance(tests, list):
            continue
        for test_entry in tests:
            if not isinstance(test_entry, dict):
                continue
            if test_entry.get("success") is True:
                continue
            test_name = test_entry.get("test_name") or "unknown_test"
            index = counters_by_name.get(test_name, 0)
            counters_by_name[test_name] = index + 1
            blocker_key = f"server_tests.{test_name}.{index}"
            acceptance_blockers[blocker_key] = _format_server_test_failure(test_entry)

    return acceptance_blockers


def _passes_report_check(check_value: Any) -> bool:
    if check_value is None:
        return False

    try:
        return int(check_value) != FAIL_CHECK
    except (TypeError, ValueError):
        return False


def _is_check_failing(value: Any) -> bool:
    if isinstance(value, (ReportCheckTypes, int)):
        return int(value) == FAIL_CHECK
    return False


def _failed_tiers(
    target_checks: Dict[str, Dict[str, Any]], tiers: Iterable[str]
) -> List[str]:
    failed: List[str] = []
    for tier in tiers:
        tier_checks = target_checks.get(tier, {})
        if not isinstance(tier_checks, dict):
            continue
        for key, value in tier_checks.items():
            if key.endswith(CHECK_SUFFIX) and _is_check_failing(value):
                failed.append(tier)
                break
    return failed


def _extract_first_target_checks(
    benchmarks_summary: Any,
) -> Optional[Dict[str, Dict[str, Any]]]:
    if not isinstance(benchmarks_summary, list) or not benchmarks_summary:
        return None
    first_summary = benchmarks_summary[0]
    if not isinstance(first_summary, dict):
        return None
    target_checks = first_summary.get("target_checks")
    if not isinstance(target_checks, dict):
        return None
    return target_checks


def _benchmark_tier_from_key(blocker_key: str) -> Optional[str]:
    parts = blocker_key.split(".")
    if len(parts) < 2 or parts[0] != "benchmarks":
        return None
    candidate = parts[1]
    if candidate in TARGET_CHECK_LEVELS:
        return candidate
    return None


def _classify_blocker(
    blocker_key: str,
) -> Tuple[Optional[WorkflowType], Optional[str]]:
    parts = blocker_key.split(".")
    if not parts:
        return None, None
    head = parts[0]

    if head in ("benchmarks", "benchmarks_summary"):
        return WorkflowType.BENCHMARKS, None
    if head == "evals":
        if len(parts) >= 2 and not parts[1].startswith("index_"):
            return WorkflowType.EVALS, parts[1]
        return WorkflowType.EVALS, None
    if head == "parameter_support_tests":
        if len(parts) >= 2:
            return WorkflowType.TESTS, parts[1]
        return WorkflowType.TESTS, None
    if head == "server_tests":
        if len(parts) >= 2:
            return WorkflowType.SPEC_TESTS, parts[1]
        return WorkflowType.SPEC_TESTS, None
    return None, None


def _find_matching_known_issue(
    known_issues: Iterable[Any],
    workflow_type: WorkflowType,
    task_name: Optional[str],
) -> Optional[Any]:
    for issue in known_issues:
        if issue.matches(workflow_type, task_name):
            return issue
    return None


def _serialize_known_issues(
    known_issues: Optional[Iterable[Any]],
) -> List[Dict[str, Optional[str]]]:
    if not known_issues:
        return []
    serialized = []
    for issue in known_issues:
        workflow_name = (
            issue.workflow_type.name
            if isinstance(issue.workflow_type, WorkflowType)
            else str(issue.workflow_type)
        )
        serialized.append(
            {
                "workflow_type": workflow_name,
                "task_name": issue.task_name,
                "reason": issue.reason,
            }
        )
    return serialized


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


def _format_server_test_failure(test_entry: dict) -> str:
    detail_parts = [
        f"success={_format_value(test_entry.get('success'))} vs expected True"
    ]
    error = test_entry.get("error")
    if error:
        detail_parts.append(f"error={error}")
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
