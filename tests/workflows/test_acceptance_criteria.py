#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import copy

import pytest

from workflows.acceptance_criteria import (
    acceptance_criteria_check,
    format_acceptance_summary_markdown,
)


@pytest.fixture
def report_data():
    return {
        "benchmarks_summary": [
            {
                "ttft": 1.2,
                "tput_user": 42.0,
                "target_checks": {
                    "functional": {
                        "ttft": 10.0,
                        "ttft_ratio": 0.12,
                        "ttft_check": 2,
                        "tput_check": 1,
                    },
                    "complete": {
                        "ttft": 2.0,
                        "ttft_ratio": 0.6,
                        "ttft_check": 2,
                        "tput_check": 2,
                    },
                    "target": {
                        "ttft": 1.0,
                        "ttft_ratio": 1.2,
                        "ttft_check": 2,
                        "tput_check": 2,
                    },
                },
            }
        ],
        "evals": [
            {
                "task_name": "hellaswag",
                "accuracy_check": 2,
                "score": 0.77,
                "gpu_reference_score": 0.80,
                "ratio_to_reference": 0.9625,
            },
            {
                "task_name": "mmlu",
                "accuracy_check": 1,
                "score": 0.71,
                "published_score": 0.72,
                "ratio_to_published": 0.9861,
            },
        ],
        "parameter_support_tests": {
            "results": {
                "test_temperature": [{"status": "passed", "message": ""}],
                "test_top_p": [
                    {"status": "passed", "message": ""},
                    {"status": "passed", "message": ""},
                ],
            }
        },
    }


def test_acceptance_criteria_check_returns_tuple_when_all_checks_pass(report_data):
    accepted, blockers = acceptance_criteria_check(report_data)
    summary_markdown = format_acceptance_summary_markdown(accepted, blockers)

    assert accepted is True
    assert blockers == {}
    assert "### Acceptance Criteria" in summary_markdown
    assert "- Acceptance status: `PASS`" in summary_markdown
    assert "- All acceptance criteria passed." in summary_markdown


def test_acceptance_criteria_check_allows_failed_higher_benchmark_levels_when_functional_passes(
    report_data,
):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["benchmarks_summary"][0]["target_checks"]["target"][
        "ttft_check"
    ] = 3
    failed_report_data["benchmarks_summary"][0]["target_checks"]["complete"][
        "ttft_check"
    ] = 3

    accepted, blockers = acceptance_criteria_check(failed_report_data)

    assert accepted is True
    assert blockers == {}


def test_acceptance_criteria_check_returns_blockers_when_all_benchmark_levels_fail(
    report_data,
):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["benchmarks_summary"][0]["target_checks"]["functional"][
        "ttft_check"
    ] = 3
    failed_report_data["benchmarks_summary"][0]["target_checks"]["complete"][
        "ttft_check"
    ] = 3
    failed_report_data["benchmarks_summary"][0]["target_checks"]["target"][
        "ttft_check"
    ] = 3

    accepted, blockers = acceptance_criteria_check(failed_report_data)

    assert accepted is False
    assert "benchmarks.functional.ttft_check" in blockers
    assert "benchmarks.complete.ttft_check" in blockers
    assert "benchmarks.target.ttft_check" in blockers
    assert "actual ttft=1.2000" in blockers["benchmarks.functional.ttft_check"]
    assert "threshold ttft=10.0000" in blockers["benchmarks.functional.ttft_check"]
    assert "ratio=0.1200" in blockers["benchmarks.functional.ttft_check"]


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
    report_data, eval_row, expected_blocker_key, expected_message
):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["evals"] = [eval_row]

    accepted, blockers = acceptance_criteria_check(failed_report_data)

    assert accepted is False
    assert expected_blocker_key in blockers
    assert expected_message in blockers[expected_blocker_key]


def test_acceptance_criteria_check_returns_blocker_for_failed_parameter_support_test(
    report_data,
):
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
    report_data, field_value
):
    updated_report_data = copy.deepcopy(report_data)
    if field_value is None:
        updated_report_data.pop("parameter_support_tests")
    else:
        updated_report_data["parameter_support_tests"] = field_value

    accepted, blockers = acceptance_criteria_check(updated_report_data)

    assert accepted is True
    assert blockers == {}


@pytest.mark.parametrize(
    "field_name,field_value,expected_blocker_key",
    [
        (
            "benchmarks_summary",
            [{"model": "placeholder"}],
            "benchmarks_summary.0.target_checks",
        ),
        ("evals", [{"model": "placeholder"}], "evals.index_0"),
        (
            "parameter_support_tests",
            [{"model": "placeholder"}],
            None,
        ),
    ],
)
def test_acceptance_criteria_check_returns_false_for_placeholder_shapes(
    report_data, field_name, field_value, expected_blocker_key
):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data[field_name] = field_value

    accepted, blockers = acceptance_criteria_check(failed_report_data)
    summary_markdown = format_acceptance_summary_markdown(accepted, blockers)

    if expected_blocker_key is not None:
        assert accepted is False
        assert expected_blocker_key in blockers
        assert expected_blocker_key in summary_markdown
    else:
        assert accepted is True
        assert blockers == {}
