#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import copy

import pytest

from workflows.model_spec import (
    AcceptanceBenchmarkCheck,
    AcceptanceCheckOverride,
    AcceptanceCriteria,
    AcceptancePerfTarget,
    AcceptanceSectionCriteria,
)
from workflows.acceptance_criteria import (
    acceptance_criteria_check,
    evaluate_benchmark_targets,
    format_acceptance_summary_markdown,
)
from workflows.perf_targets import (
    DEFAULT_PERF_TARGETS_MAP,
    BenchmarkTaskParams,
    DeviceTypes,
    PerfTarget,
    get_perf_target,
)


def make_perf_reference(include_regression=False):
    task_targets = [
        BenchmarkTaskParams(
            isl=128,
            osl=128,
            max_concurrency=1,
            task_type="text",
            targets={
                "functional": PerfTarget(ttft_ms=100.0, tput_user=10.0),
                "complete": PerfTarget(ttft_ms=50.0, tput_user=20.0),
                "target": PerfTarget(ttft_ms=25.0, tput_user=30.0),
            },
        ),
        BenchmarkTaskParams(
            isl=2048,
            osl=128,
            max_concurrency=1,
            task_type="text",
            targets={
                "functional": PerfTarget(ttft_ms=100.0, tput_user=5.0),
                "complete": PerfTarget(ttft_ms=80.0, tput_user=12.0),
                "target": PerfTarget(ttft_ms=45.0, tput_user=15.0),
            },
        ),
    ]
    if include_regression:
        task_targets[0].targets["regression"] = PerfTarget(
            ttft_ms=55.0,
            tput_user=16.0,
            tolerance=0.05,
        )
    return task_targets


def make_non_text_perf_reference():
    return [
        BenchmarkTaskParams(
            max_concurrency=1,
            task_type="image",
            num_inference_steps=20,
            targets={
                "functional": PerfTarget(ttft_ms=12500.0, tput_user=0.08),
            },
        ),
        BenchmarkTaskParams(
            max_concurrency=1,
            task_type="audio",
            num_eval_runs=2,
            targets={
                "functional": PerfTarget(
                    ttft_ms=400.0,
                    tput_user=100.0,
                    rtr=10.0,
                ),
            },
        ),
        BenchmarkTaskParams(
            max_concurrency=2,
            task_type="embedding",
            targets={
                "functional": PerfTarget(
                    tput_user=120.0,
                    tput_prefill=400.0,
                    e2el_ms=20.0,
                ),
            },
        ),
    ]


@pytest.fixture
def report_data():
    return {
        "metadata": {
            "model_name": "DemoModel",
            "device": "N150",
        },
        "benchmarks_summary": [
            {
                "task_type": "text",
                "input_sequence_length": 128,
                "output_sequence_length": 128,
                "max_con": 1,
                "mean_ttft_ms": 60.0,
                "mean_tps": 15.0,
            },
            {
                "task_type": "text",
                "input_sequence_length": 2048,
                "output_sequence_length": 128,
                "max_con": 1,
                "mean_ttft_ms": 90.0,
                "mean_tps": 8.0,
            },
        ],
        "evals": [
            {
                "task_name": "hellaswag",
                "accuracy_check": 2,
                "score": 0.77,
                "gpu_reference_score": 0.80,
                "ratio_to_reference": 0.9625,
            }
        ],
        "parameter_support_tests": {
            "results": {
                "test_temperature": [{"status": "passed", "message": ""}],
            }
        },
    }


def test_acceptance_criteria_check_reports_next_status_failures(
    report_data, monkeypatch
):
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: make_perf_reference(),
    )

    benchmark_target_evaluation = evaluate_benchmark_targets(report_data)
    accepted, blockers = acceptance_criteria_check(
        report_data, benchmark_target_evaluation
    )
    summary_markdown = format_acceptance_summary_markdown(
        accepted, blockers, benchmark_target_evaluation
    )

    assert accepted is True
    assert blockers == {}
    assert benchmark_target_evaluation["status"] == "functional"
    assert benchmark_target_evaluation["next_status"] == "complete"
    assert len(benchmark_target_evaluation["next_status_failures"]) == 2
    assert "### Acceptance Criteria" in summary_markdown
    assert "- Acceptance status: `PASS`" in summary_markdown
    assert "- Benchmark perf status: `functional`" in summary_markdown
    assert "- Next benchmark status blocked at `complete`." in summary_markdown
    assert "complete text isl=128 osl=128 max_concurrency=1" not in summary_markdown
    assert "complete text isl=2048 osl=128 max_concurrency=1" not in summary_markdown


def test_acceptance_summary_omits_duplicate_next_status_failure_details(
    report_data, monkeypatch
):
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: make_perf_reference(),
    )
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["benchmarks_summary"][0]["mean_ttft_ms"] = 338.5

    benchmark_target_evaluation = evaluate_benchmark_targets(failed_report_data)
    accepted, blockers = acceptance_criteria_check(
        failed_report_data, benchmark_target_evaluation
    )
    summary_markdown = format_acceptance_summary_markdown(
        accepted, blockers, benchmark_target_evaluation
    )

    blocker_key = "benchmarks.functional.text_isl_128_osl_128_max_concurrency_1"

    assert accepted is False
    assert benchmark_target_evaluation["status"] == "experimental"
    assert benchmark_target_evaluation["next_status"] == "functional"
    assert len(benchmark_target_evaluation["next_status_failures"]) == 1
    assert blocker_key in blockers
    assert "- Next benchmark status blocked at `functional`." in summary_markdown
    assert "functional text isl=128 osl=128 max_concurrency=1" not in summary_markdown
    assert f"- `{blocker_key}`: {blockers[blocker_key]}" in summary_markdown


def test_acceptance_criteria_check_regression_failure_blocks_acceptance(
    report_data, monkeypatch
):
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: make_perf_reference(include_regression=True),
    )

    benchmark_target_evaluation = evaluate_benchmark_targets(report_data)
    accepted, blockers = acceptance_criteria_check(
        report_data, benchmark_target_evaluation
    )
    summary_markdown = format_acceptance_summary_markdown(
        accepted, blockers, benchmark_target_evaluation
    )

    assert accepted is False
    assert benchmark_target_evaluation["regression"]["checked"] is True
    assert benchmark_target_evaluation["regression"]["passed"] is False
    regression_blocker_keys = [
        blocker_key
        for blocker_key in blockers
        if blocker_key.startswith("benchmarks.regression.")
    ]
    assert regression_blocker_keys
    assert "- Regression status: `FAIL`" in summary_markdown


def test_acceptance_criteria_matches_non_text_rows_without_overmatching(monkeypatch):
    report_data = {
        "metadata": {
            "model_name": "DemoModel",
            "device": "N150",
        },
        "benchmarks_summary": [
            {
                "task_type": "image",
                "num_inference_steps": 28,
                "mean_ttft_ms": 60000.0,
                "inference_steps_per_second": 0.02,
            },
            {
                "task_type": "image",
                "num_inference_steps": 20,
                "mean_ttft_ms": 11000.0,
                "inference_steps_per_second": 0.09,
            },
            {
                "task_type": "audio",
                "num_eval_runs": 4,
                "mean_ttft_ms": 250.0,
                "t/s/u": 10.0,
                "rtr": 1.0,
            },
            {
                "task_type": "audio",
                "num_eval_runs": 2,
                "mean_ttft_ms": 250.0,
                "t/s/u": 112.62,
                "rtr": 15.61,
            },
            {
                "task_type": "embedding",
                "input_sequence_length": 256,
                "max_con": 2,
                "mean_tps": 123.45,
                "tps_prefill_throughput": 456.7,
                "mean_e2el_ms": 12.34,
            },
        ],
        "evals": [
            {
                "task_name": "dummy",
                "accuracy_check": 2,
                "score": 1.0,
                "gpu_reference_score": 1.0,
                "ratio_to_reference": 1.0,
            }
        ],
    }
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: make_non_text_perf_reference(),
    )

    benchmark_target_evaluation = evaluate_benchmark_targets(report_data)
    accepted, blockers = acceptance_criteria_check(
        report_data, benchmark_target_evaluation
    )

    assert benchmark_target_evaluation["status"] == "functional"
    assert benchmark_target_evaluation["support_levels"]["functional"]["passed"] is True
    assert benchmark_target_evaluation["support_levels"]["functional"]["failures"] == []
    assert accepted is True
    assert blockers == {}


def test_acceptance_criteria_supports_real_image_perf_reference(monkeypatch):
    perf_target_set = get_perf_target("stable-diffusion-xl-base-1.0", DeviceTypes.N150)
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: perf_target_set.to_benchmark_task_params(
            DEFAULT_PERF_TARGETS_MAP
        ),
    )

    report_data = {
        "metadata": {
            "model_name": "stable-diffusion-xl-base-1.0",
            "device": "N150",
        },
        "benchmarks_summary": [
            {
                "task_type": "image",
                "num_inference_steps": 20,
                "mean_ttft_ms": 12000.0,
                "inference_steps_per_second": 0.08,
            }
        ],
        "evals": [
            {
                "task_name": "dummy",
                "accuracy_check": 2,
                "score": 1.0,
                "gpu_reference_score": 1.0,
                "ratio_to_reference": 1.0,
            }
        ],
    }

    benchmark_target_evaluation = evaluate_benchmark_targets(report_data)
    accepted, blockers = acceptance_criteria_check(
        report_data, benchmark_target_evaluation
    )

    assert benchmark_target_evaluation["reference_available"] is True
    assert benchmark_target_evaluation["status"] == "target"
    assert benchmark_target_evaluation["support_levels"]["functional"]["failures"] == []
    assert benchmark_target_evaluation["support_levels"]["complete"]["failures"] == []
    assert benchmark_target_evaluation["support_levels"]["target"]["failures"] == []
    assert accepted is True
    assert blockers == {}


@pytest.mark.parametrize(
    "eval_row,expected_blocker_key,expected_message",
    [
        (
            {
                "task_name": "hellaswag",
                "accuracy_check": 3,
                "score": 0.72,
                "gpu_reference_score": 0.80,
                "ratio_to_reference": 0.90,
            },
            "evals.hellaswag",
            "reference gpu_reference_score=0.8000",
        ),
        (
            {"task_name": "hellaswag"},
            "evals.hellaswag",
            "Missing accuracy_check",
        ),
    ],
)
def test_acceptance_criteria_check_returns_blocker_for_invalid_eval_checks(
    report_data, eval_row, expected_blocker_key, expected_message, monkeypatch
):
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: make_perf_reference(),
    )
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["evals"] = [eval_row]

    accepted, blockers = acceptance_criteria_check(failed_report_data)

    assert accepted is False
    assert expected_blocker_key in blockers
    assert expected_message in blockers[expected_blocker_key]


def test_acceptance_criteria_check_returns_blocker_for_failed_parameter_support_test(
    report_data, monkeypatch
):
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: make_perf_reference(),
    )
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["parameter_support_tests"]["results"]["test_temperature"][0][
        "status"
    ] = "failed"
    failed_report_data["parameter_support_tests"]["results"]["test_temperature"][0][
        "message"
    ] = "temperature argument rejected"

    accepted, blockers = acceptance_criteria_check(failed_report_data)

    assert accepted is False
    assert "parameter_support_tests.test_temperature.0" in blockers
    assert (
        "status=failed vs expected passed"
        in blockers["parameter_support_tests.test_temperature.0"]
    )
    assert (
        "message=temperature argument rejected"
        in blockers["parameter_support_tests.test_temperature.0"]
    )


@pytest.mark.parametrize(
    "field_value", [None, [], {}, {"results": {}}, {"configs": {}}]
)
def test_acceptance_criteria_check_allows_missing_parameter_support_test_results(
    report_data, field_value, monkeypatch
):
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: make_perf_reference(),
    )
    updated_report_data = copy.deepcopy(report_data)
    if field_value is None:
        updated_report_data.pop("parameter_support_tests")
    else:
        updated_report_data["parameter_support_tests"] = field_value

    accepted, blockers = acceptance_criteria_check(updated_report_data)

    assert accepted is True
    assert blockers == {}


def test_acceptance_criteria_check_returns_false_for_missing_benchmarks_summary(
    report_data, monkeypatch
):
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: make_perf_reference(),
    )
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["benchmarks_summary"] = [{"model": "placeholder"}]

    accepted, blockers = acceptance_criteria_check(failed_report_data)
    summary_markdown = format_acceptance_summary_markdown(accepted, blockers)

    assert accepted is False
    functional_blocker_keys = [
        blocker_key
        for blocker_key in blockers
        if blocker_key.startswith("benchmarks.functional.")
    ]
    assert functional_blocker_keys
    assert "actual unavailable" in blockers[functional_blocker_keys[0]]
    assert "benchmarks.functional." in summary_markdown


def test_acceptance_criteria_check_supports_perf_subset_and_tolerance_override(
    report_data, monkeypatch
):
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: make_perf_reference(),
    )
    monkeypatch.setattr(
        "workflows.acceptance_criteria._load_acceptance_criteria_from_report",
        lambda *_, **__: AcceptanceCriteria(
            perf_checks=(
                AcceptanceBenchmarkCheck(
                    isl=128,
                    osl=128,
                    max_concurrency=1,
                    target_names=(AcceptancePerfTarget.FUNCTIONAL,),
                    override=AcceptanceCheckOverride(tolerance=0.20),
                ),
            )
        ),
    )

    adjusted_report_data = copy.deepcopy(report_data)
    adjusted_report_data["benchmarks_summary"][0]["mean_ttft_ms"] = 115.0
    adjusted_report_data["benchmarks_summary"][0]["mean_tps"] = 8.5
    adjusted_report_data["benchmarks_summary"][1]["mean_ttft_ms"] = 999.0
    adjusted_report_data["benchmarks_summary"][1]["mean_tps"] = 1.0

    benchmark_target_evaluation = evaluate_benchmark_targets(adjusted_report_data)
    accepted, blockers = acceptance_criteria_check(
        adjusted_report_data, benchmark_target_evaluation
    )

    assert accepted is True
    assert blockers == {}
    assert (
        benchmark_target_evaluation["support_levels"]["functional"]["checked"] is True
    )
    assert benchmark_target_evaluation["support_levels"]["functional"]["passed"] is True
    assert benchmark_target_evaluation["support_levels"]["complete"]["checked"] is False


def test_acceptance_criteria_check_skips_optional_sections_when_disabled(
    report_data, monkeypatch
):
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: make_perf_reference(),
    )
    monkeypatch.setattr(
        "workflows.acceptance_criteria._load_acceptance_criteria_from_report",
        lambda *_, **__: AcceptanceCriteria(
            eval_checks=AcceptanceSectionCriteria(required=False),
            tests_checks=AcceptanceSectionCriteria(required=False),
            spec_tests_checks=AcceptanceSectionCriteria(required=False),
        ),
    )

    updated_report_data = copy.deepcopy(report_data)
    updated_report_data.pop("evals")
    updated_report_data.pop("parameter_support_tests")
    updated_report_data["spec_tests"] = {"results": []}

    accepted, blockers = acceptance_criteria_check(updated_report_data)

    assert accepted is True
    assert blockers == {}


def test_acceptance_criteria_check_blocks_failed_spec_tests_when_present(
    report_data, monkeypatch
):
    monkeypatch.setattr(
        "workflows.acceptance_criteria.get_named_perf_reference",
        lambda *args, **kwargs: make_perf_reference(),
    )
    updated_report_data = copy.deepcopy(report_data)
    updated_report_data["spec_tests"] = {
        "results": [
            {
                "test_name": "device_liveness",
                "success": False,
                "error": "worker timed out",
            }
        ]
    }

    accepted, blockers = acceptance_criteria_check(updated_report_data)

    assert accepted is False
    assert "spec_tests.device_liveness.0" in blockers
    assert "worker timed out" in blockers["spec_tests.device_liveness.0"]
