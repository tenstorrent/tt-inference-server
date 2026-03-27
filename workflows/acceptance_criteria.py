# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from workflows.model_spec import (
    AcceptanceBenchmarkCheck,
    AcceptanceCheckGroup,
    AcceptanceCriteria,
    AcceptancePerfTarget,
    ModelSpec,
    resolve_acceptance_criteria,
)
from workflows.perf_targets import (
    DEFAULT_PERF_TARGETS_MAP,
    REGRESSION_TARGET_NAME,
    BenchmarkTaskParams,
    PerfTarget,
    find_matching_benchmark_row,
    get_named_perf_reference,
    perf_target_from_benchmark_task,
)
from workflows.workflow_types import ReportCheckTypes

FAIL_CHECK = int(ReportCheckTypes.FAIL)
SUPPORT_TARGET_LEVELS = tuple(
    target_name.value
    for target_name in (
        AcceptancePerfTarget.FUNCTIONAL,
        AcceptancePerfTarget.COMPLETE,
        AcceptancePerfTarget.TARGET,
    )
)
BENCHMARK_LATENCY_METRICS = ("ttft_ms", "ttft_streaming_ms", "e2el_ms")
BENCHMARK_THROUGHPUT_METRICS = ("tput_user", "tput_prefill", "tput", "rtr")
BENCHMARK_ACTUAL_FIELDS = {
    "ttft_ms": ("mean_ttft_ms", "ttft"),
    "ttft_streaming_ms": ("ttft_streaming_ms", "mean_ttft_ms", "ttft"),
    "tput_user": ("mean_tps", "tput_user", "t/s/u", "inference_steps_per_second"),
    "tput_prefill": ("tps_prefill_throughput", "tput_prefill"),
    "e2el_ms": ("mean_e2el_ms", "e2el_ms", "e2el", "end_to_end_latency_ms"),
    "tput": ("tps_decode_throughput", "tput", "inference_steps_per_second"),
    "rtr": ("rtr",),
}
REPORT_METRIC_FALLBACK_FIELDS = {
    "tput_user": ("tput_user",),
    "tput_prefill": ("tput_prefill",),
    "e2el_ms": ("e2el_ms", "e2el"),
    "tput": ("tput", "inference_steps_per_second"),
    "rtr": ("rtr",),
}


def acceptance_criteria_check(
    report_data: dict, benchmark_target_evaluation: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, str]]:
    """Return acceptance status and blocker details for release report data."""
    acceptance_criteria = _load_acceptance_criteria_from_report(report_data)
    acceptance_blockers = {}
    benchmark_target_evaluation = (
        benchmark_target_evaluation or evaluate_benchmark_targets(report_data)
    )
    acceptance_blockers.update(
        _benchmarks_acceptance(benchmark_target_evaluation, acceptance_criteria)
    )
    acceptance_blockers.update(
        _evals_acceptance(
            report_data.get("evals"),
            required=_is_check_group_required(
                acceptance_criteria, AcceptanceCheckGroup.EVALS
            ),
        )
    )
    acceptance_blockers.update(
        _parameter_support_acceptance(
            report_data.get("parameter_support_tests"),
            required=_is_check_group_required(
                acceptance_criteria, AcceptanceCheckGroup.TESTS
            ),
        )
    )
    acceptance_blockers.update(
        _spec_tests_acceptance(
            report_data.get("spec_tests"),
            required=_is_check_group_required(
                acceptance_criteria, AcceptanceCheckGroup.SPEC_TESTS
            ),
        )
    )
    return len(acceptance_blockers) == 0, acceptance_blockers


def format_acceptance_summary_markdown(
    accepted: bool,
    acceptance_blockers: Dict[str, str],
    benchmark_target_evaluation: Optional[Dict[str, Any]] = None,
) -> str:
    """Format acceptance status and blockers as markdown."""
    lines = [
        "### Acceptance Criteria",
        "",
        f"- Acceptance status: `{'PASS' if accepted else 'FAIL'}`",
    ]
    if isinstance(benchmark_target_evaluation, dict):
        support_status = benchmark_target_evaluation.get("status", "experimental")
        lines.append(f"- Benchmark perf status: `{support_status}`")
        regression = benchmark_target_evaluation.get("regression", {})
        if regression.get("checked"):
            regression_status = "PASS" if regression.get("passed") else "FAIL"
            lines.append(f"- Regression status: `{regression_status}`")

        next_status = benchmark_target_evaluation.get("next_status")
        next_failures = benchmark_target_evaluation.get("next_status_failures", [])
        if next_status and next_failures:
            lines.append(f"- Next benchmark status blocked at `{next_status}`.")

    if accepted and not acceptance_blockers:
        lines.append("- All acceptance criteria passed.")
        return "\n".join(lines)

    for blocker_key, blocker_message in acceptance_blockers.items():
        lines.append(f"- `{blocker_key}`: {blocker_message}")

    return "\n".join(lines)


def evaluate_benchmark_targets(report_data: dict) -> Dict[str, Any]:
    acceptance_criteria = _load_acceptance_criteria_from_report(report_data)
    benchmarks_summary = report_data.get("benchmarks_summary")
    analysis = _empty_benchmark_analysis()
    if not isinstance(benchmarks_summary, list) or not benchmarks_summary:
        analysis["errors"].append("Missing benchmarks summary entries in report data.")
        return analysis

    perf_references = _load_perf_references_from_report(
        report_data, acceptance_criteria
    )
    if not perf_references:
        analysis["errors"].append(
            "Missing benchmark performance references for report data."
        )
        return analysis

    analysis["reference_available"] = True
    support_failures = {level_name: [] for level_name in SUPPORT_TARGET_LEVELS}
    support_checked = {level_name: False for level_name in SUPPORT_TARGET_LEVELS}
    regression_failures = []
    regression_checked = False

    for benchmark_task in perf_references:
        target_plan = _resolve_perf_target_plan(acceptance_criteria, benchmark_task)
        if not target_plan["support_targets"] and not target_plan["check_regression"]:
            continue

        matched_row = _find_matching_benchmark_row(benchmarks_summary, benchmark_task)
        for target_name, perf_target in benchmark_task.targets.items():
            if target_name == REGRESSION_TARGET_NAME:
                if not target_plan["check_regression"]:
                    continue
                regression_checked = True
            elif target_name in SUPPORT_TARGET_LEVELS:
                if target_name not in target_plan["support_targets"]:
                    continue
                support_checked[target_name] = True
            else:
                continue

            failure = _evaluate_target_failure(
                benchmark_task=benchmark_task,
                perf_target=perf_target,
                target_name=target_name,
                matched_row=matched_row,
                report_data=report_data,
                tolerance_override=target_plan["tolerance_overrides"].get(target_name),
            )
            if failure is None:
                continue
            if target_name == REGRESSION_TARGET_NAME:
                regression_failures.append(failure)
            elif target_name in SUPPORT_TARGET_LEVELS:
                support_failures[target_name].append(failure)

    for level_name in SUPPORT_TARGET_LEVELS:
        failures = support_failures[level_name]
        analysis["support_levels"][level_name] = {
            "checked": support_checked[level_name],
            "passed": support_checked[level_name] and not failures,
            "failures": failures,
        }

    for level_name in reversed(SUPPORT_TARGET_LEVELS):
        if analysis["support_levels"][level_name]["passed"]:
            analysis["status"] = level_name
            break

    analysis["regression"] = {
        "checked": regression_checked,
        "passed": regression_checked and not regression_failures
        if regression_checked
        else True,
        "failures": regression_failures,
    }

    next_status_map = {
        "experimental": "functional",
        "functional": "complete",
        "complete": "target",
        "target": None,
    }
    next_status = next_status_map[analysis["status"]]
    analysis["next_status"] = next_status
    if next_status:
        analysis["next_status_failures"] = analysis["support_levels"][next_status][
            "failures"
        ]

    return analysis


def _benchmarks_acceptance(
    benchmark_target_evaluation: Dict[str, Any],
    acceptance_criteria: AcceptanceCriteria,
) -> Dict[str, str]:
    acceptance_blockers = {}
    if not isinstance(benchmark_target_evaluation, dict):
        acceptance_blockers["benchmarks_summary"] = (
            "Benchmark target evaluation was not generated."
        )
        return acceptance_blockers

    for index, error_message in enumerate(
        benchmark_target_evaluation.get("errors", [])
    ):
        acceptance_blockers[f"benchmarks_summary.error_{index}"] = error_message
    if acceptance_blockers:
        return acceptance_blockers

    regression = benchmark_target_evaluation.get("regression", {})
    for failure in regression.get("failures", []):
        acceptance_blockers[failure["key"]] = failure["message"]

    required_support_targets = set(acceptance_criteria.resolved_required_perf_targets())
    for level_name, level_data in (
        benchmark_target_evaluation.get("support_levels", {}) or {}
    ).items():
        if level_name not in required_support_targets:
            continue
        if not isinstance(level_data, dict) or not level_data.get("checked"):
            continue
        for failure in level_data.get("failures", []):
            acceptance_blockers[failure["key"]] = failure["message"]

    return acceptance_blockers


def _evals_acceptance(evals_data: Any, required: bool) -> Dict[str, str]:
    acceptance_blockers = {}
    if not required:
        return acceptance_blockers
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


def _parameter_support_acceptance(
    parameter_support_tests: Any, required: bool
) -> Dict[str, str]:
    acceptance_blockers = {}
    if not isinstance(parameter_support_tests, dict):
        if required:
            acceptance_blockers["parameter_support_tests"] = (
                "Missing normalized parameter support test results in report data."
            )
        return acceptance_blockers

    results = parameter_support_tests.get("results")
    if not isinstance(results, dict) or not results:
        if required:
            acceptance_blockers["parameter_support_tests"] = (
                "Missing normalized parameter support test results in report data."
            )
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


def _spec_tests_acceptance(spec_tests: Any, required: bool) -> Dict[str, str]:
    acceptance_blockers = {}
    test_results = list(_iter_spec_test_results(spec_tests))
    if not test_results:
        if required:
            acceptance_blockers["spec_tests"] = (
                "Missing normalized spec test results in report data."
            )
        return acceptance_blockers

    for index, test_result in enumerate(test_results):
        if test_result.get("success") is True:
            continue

        test_name = test_result.get("test_name") or f"index_{index}"
        blocker_key = f"spec_tests.{test_name}.{index}"
        acceptance_blockers[blocker_key] = _format_spec_test_failure(test_result)

    return acceptance_blockers


def _load_perf_references_from_report(
    report_data: Dict[str, Any],
    acceptance_criteria: Optional[AcceptanceCriteria] = None,
) -> List[BenchmarkTaskParams]:
    model_spec = _load_runtime_model_spec_from_report(report_data)
    if model_spec is not None:
        perf_reference = getattr(model_spec.device_model_spec, "perf_reference", [])
        if perf_reference:
            return perf_reference

    model_name, device = _resolve_report_identity(report_data)
    if not model_name or not device:
        return []
    return get_named_perf_reference(
        model_name=model_name,
        device=device,
        perf_targets_map=DEFAULT_PERF_TARGETS_MAP,
        acceptance_criteria=acceptance_criteria,
    )


def _load_acceptance_criteria_from_report(
    report_data: Dict[str, Any],
) -> AcceptanceCriteria:
    model_spec = _load_runtime_model_spec_from_report(report_data)
    if model_spec is not None:
        if getattr(model_spec, "acceptance_criteria", None) is not None:
            return resolve_acceptance_criteria(
                model_spec.acceptance_criteria,
                perf_targets_map=getattr(
                    model_spec.device_model_spec, "perf_targets_map", None
                ),
            )
        if (
            getattr(model_spec.device_model_spec, "acceptance_criteria", None)
            is not None
        ):
            return resolve_acceptance_criteria(
                model_spec.device_model_spec.acceptance_criteria,
                perf_targets_map=getattr(
                    model_spec.device_model_spec, "perf_targets_map", None
                ),
            )
    return resolve_acceptance_criteria(perf_targets_map=DEFAULT_PERF_TARGETS_MAP)


def _load_runtime_model_spec_from_report(
    report_data: Dict[str, Any],
) -> Optional[ModelSpec]:
    metadata = report_data.get("metadata", {})
    runtime_model_spec_json = metadata.get("runtime_model_spec_json")
    if not runtime_model_spec_json:
        return None

    runtime_model_spec_path = Path(runtime_model_spec_json)
    if not runtime_model_spec_path.exists():
        return None

    try:
        return ModelSpec.from_json(str(runtime_model_spec_path))
    except Exception:
        return None


def _empty_benchmark_analysis() -> Dict[str, Any]:
    return {
        "reference_available": False,
        "status": "experimental",
        "support_levels": {
            level_name: {"checked": False, "passed": False, "failures": []}
            for level_name in SUPPORT_TARGET_LEVELS
        },
        "next_status": None,
        "next_status_failures": [],
        "regression": {"checked": False, "passed": True, "failures": []},
        "errors": [],
    }


def _is_check_group_required(
    acceptance_criteria: AcceptanceCriteria, check_group: AcceptanceCheckGroup
) -> bool:
    if check_group == AcceptanceCheckGroup.EVALS:
        return acceptance_criteria.eval_checks.required
    if check_group == AcceptanceCheckGroup.TESTS:
        return acceptance_criteria.tests_checks.required
    if check_group == AcceptanceCheckGroup.SPEC_TESTS:
        return acceptance_criteria.spec_tests_checks.required
    return True


def _resolve_perf_target_plan(
    acceptance_criteria: AcceptanceCriteria, benchmark_task: BenchmarkTaskParams
) -> Dict[str, Any]:
    matching_checks = _matching_perf_checks(acceptance_criteria, benchmark_task)
    tolerance_overrides: Dict[str, float] = {}
    support_targets: List[str] = []

    if matching_checks:
        for check in matching_checks:
            if check.override.required is False:
                continue
            for target_name in check.resolved_target_names(
                acceptance_criteria.required_perf_targets
            ):
                if target_name == REGRESSION_TARGET_NAME:
                    continue
                if target_name not in support_targets:
                    support_targets.append(target_name)
                if check.override.tolerance is not None:
                    tolerance_overrides[target_name] = check.override.tolerance
        check_regression = bool(
            support_targets and acceptance_criteria.regression_check.required
        )
    elif acceptance_criteria.perf_checks:
        check_regression = False
    else:
        support_targets = list(SUPPORT_TARGET_LEVELS)
        check_regression = bool(acceptance_criteria.regression_check.required)

    return {
        "support_targets": support_targets,
        "check_regression": check_regression,
        "tolerance_overrides": tolerance_overrides,
    }


def _matching_perf_checks(
    acceptance_criteria: AcceptanceCriteria, benchmark_task: BenchmarkTaskParams
) -> List[AcceptanceBenchmarkCheck]:
    if not acceptance_criteria.perf_checks:
        return []

    benchmark_identity = perf_target_from_benchmark_task(benchmark_task)
    return [
        perf_check
        for perf_check in acceptance_criteria.perf_checks
        if perf_check.matches_perf_target(benchmark_identity)
    ]


def _iter_spec_test_results(spec_tests: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(spec_tests, dict):
        results = spec_tests.get("results")
        if isinstance(results, list):
            return [result for result in results if isinstance(result, dict)]
        reports = spec_tests.get("reports")
        if isinstance(reports, list):
            return _flatten_spec_test_reports(reports)

    if isinstance(spec_tests, list):
        return _flatten_spec_test_reports(spec_tests)

    return []


def _flatten_spec_test_reports(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for report_index, report in enumerate(reports):
        if not isinstance(report, dict):
            continue
        for test_index, test_result in enumerate(report.get("tests", [])):
            if not isinstance(test_result, dict):
                continue
            results.append(
                {
                    **test_result,
                    "report_index": report_index,
                    "test_index": test_index,
                }
            )
    return results


def _resolve_report_identity(
    report_data: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    metadata = report_data.get("metadata", {})
    model_name = metadata.get("model_name")
    device = metadata.get("device")
    if model_name and device:
        return model_name, str(device).lower()

    benchmarks_summary = report_data.get("benchmarks_summary", [])
    if not isinstance(benchmarks_summary, list) or not benchmarks_summary:
        return None, None

    for row in benchmarks_summary:
        if not isinstance(row, dict):
            continue
        model_name = model_name or row.get("model_name") or row.get("model")
        device = device or row.get("device")
        if model_name and device:
            return model_name, str(device).lower()

    return model_name, str(device).lower() if device else None


def _find_matching_benchmark_row(
    benchmarks_summary: Iterable[Dict[str, Any]], benchmark_task: BenchmarkTaskParams
) -> Optional[Dict[str, Any]]:
    return find_matching_benchmark_row(
        benchmarks_summary, perf_target_from_benchmark_task(benchmark_task)
    )


def _evaluate_target_failure(
    benchmark_task: BenchmarkTaskParams,
    perf_target: PerfTarget,
    target_name: str,
    matched_row: Optional[Dict[str, Any]],
    report_data: Dict[str, Any],
    tolerance_override: Optional[float] = None,
) -> Optional[Dict[str, str]]:
    metric_failures = []
    metric_names = _metric_names_for_perf_target(perf_target)
    if not metric_names:
        return None

    for metric_name in metric_names:
        threshold_value = getattr(perf_target, metric_name)
        actual_value = _resolve_actual_metric(metric_name, matched_row, report_data)
        tolerance = (
            tolerance_override
            if tolerance_override is not None
            else perf_target.tolerance
        )
        ratio_value = (
            actual_value / threshold_value
            if actual_value is not None and threshold_value not in (None, 0)
            else None
        )
        if actual_value is None:
            metric_failures.append(
                _format_metric_failure(
                    metric_name=metric_name,
                    actual_value=None,
                    threshold_value=threshold_value,
                    ratio_value=ratio_value,
                )
            )
            continue

        if metric_name in BENCHMARK_LATENCY_METRICS:
            passed = actual_value <= threshold_value * (1 + tolerance)
        else:
            passed = actual_value >= threshold_value * (1 - tolerance)
        if not passed:
            metric_failures.append(
                _format_metric_failure(
                    metric_name=metric_name,
                    actual_value=actual_value,
                    threshold_value=threshold_value,
                    ratio_value=ratio_value,
                )
            )

    if not metric_failures:
        return None

    config_label = _format_perf_target_label(benchmark_task)
    config_key = _format_perf_target_key(benchmark_task)
    return {
        "key": f"benchmarks.{target_name}.{config_key}",
        "label": f"{target_name} {config_label}",
        "message": "; ".join(metric_failures) + ".",
    }


def _metric_names_for_perf_target(perf_target: PerfTarget) -> List[str]:
    metric_names = []
    for metric_name in BENCHMARK_LATENCY_METRICS + BENCHMARK_THROUGHPUT_METRICS:
        if getattr(perf_target, metric_name) is not None:
            metric_names.append(metric_name)
    return metric_names


def _resolve_actual_metric(
    metric_name: str,
    matched_row: Optional[Dict[str, Any]],
    report_data: Dict[str, Any],
) -> Optional[float]:
    actual_value = _resolve_metric_from_row(metric_name, matched_row)
    if actual_value is not None:
        return actual_value
    return _resolve_metric_from_report(metric_name, report_data)


def _resolve_metric_from_row(
    metric_name: str, row: Optional[Dict[str, Any]]
) -> Optional[float]:
    if not isinstance(row, dict):
        return None
    for field_name in BENCHMARK_ACTUAL_FIELDS.get(metric_name, (metric_name,)):
        if field_name in row:
            value = _coerce_optional_float(row.get(field_name))
            if value is not None:
                return value
    return None


def _resolve_metric_from_report(
    metric_name: str, report_data: Dict[str, Any]
) -> Optional[float]:
    for eval_row in report_data.get("evals", []):
        if not isinstance(eval_row, dict):
            continue
        for field_name in REPORT_METRIC_FALLBACK_FIELDS.get(
            metric_name, (metric_name,)
        ):
            value = _coerce_optional_float(eval_row.get(field_name))
            if value is not None:
                return value
    return None


def _format_metric_failure(
    metric_name: str,
    actual_value: Optional[float],
    threshold_value: Optional[float],
    ratio_value: Optional[float],
) -> str:
    detail_parts = [f"{metric_name} failed"]
    if actual_value is not None:
        detail_parts.append(f"actual={_format_value(actual_value)}")
    else:
        detail_parts.append("actual unavailable")
    if threshold_value is not None:
        detail_parts.append(f"threshold={_format_value(threshold_value)}")
    if ratio_value is not None:
        detail_parts.append(f"ratio={_format_value(ratio_value)}")
    return "; ".join(detail_parts)


def _format_perf_target_label(benchmark_task: BenchmarkTaskParams) -> str:
    parts = [benchmark_task.task_type]
    for field_name in (
        "isl",
        "osl",
        "max_concurrency",
        "num_eval_runs",
        "image_height",
        "image_width",
        "images_per_prompt",
        "num_inference_steps",
    ):
        value = getattr(benchmark_task, field_name)
        if value is not None and not (field_name == "images_per_prompt" and value == 0):
            parts.append(f"{field_name}={value}")
    return " ".join(parts)


def _format_perf_target_key(benchmark_task: BenchmarkTaskParams) -> str:
    return (
        _format_perf_target_label(benchmark_task)
        .replace("=", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value in (None, "", "N/A", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _passes_report_check(check_value: Any) -> bool:
    if check_value is None:
        return False

    try:
        return int(check_value) != FAIL_CHECK
    except (TypeError, ValueError):
        return False


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


def _format_spec_test_failure(test_result: Dict[str, Any]) -> str:
    detail_parts = ["status=failed vs expected passed"]
    if "success" in test_result:
        detail_parts[0] = (
            f"success={_format_value(test_result.get('success'))} vs expected True"
        )
    error_message = test_result.get("error")
    if error_message:
        detail_parts.append(f"error={error_message}")
    return "; ".join(detail_parts) + "."


def _eval_blocker_key(eval_row: Any, index: int) -> str:
    if isinstance(eval_row, dict):
        task_name = eval_row.get("task_name")
        if task_name:
            return f"evals.{task_name}"
    return f"evals.index_{index}"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, ReportCheckTypes):
        return str(int(value))
    return str(value)
