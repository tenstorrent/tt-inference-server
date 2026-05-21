#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import copy
import json

import pytest

from workflows.acceptance_criteria import (
    acceptance_criteria_check,
    format_acceptance_summary_markdown,
)
from workflows.model_spec import KnownIssue
from workflows.workflow_types import ModelStatusTypes, WorkflowType


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
    accepted, blockers, metadata = acceptance_criteria_check(report_data)
    summary_markdown = format_acceptance_summary_markdown(accepted, blockers, metadata)

    assert accepted is True
    assert blockers == {}
    assert metadata["enforcement_result"] == "PASS"
    assert metadata["failed_enforced_tiers"] == []
    assert metadata["informational_blockers"] == {}
    assert metadata["masked_blockers"] == {}
    assert metadata["waivers_applied_count"] == 0
    assert metadata["acceptance_before_masking"] is True
    assert "### Acceptance Criteria" in summary_markdown
    assert "- Acceptance status: `PASS`" in summary_markdown
    assert "- All acceptance criteria passed (no waivers applied)." in summary_markdown


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

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.FUNCTIONAL
    )

    assert accepted is True
    assert blockers == {}
    assert "benchmarks.complete.ttft_check" in metadata["informational_blockers"]
    assert "benchmarks.target.ttft_check" in metadata["informational_blockers"]


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

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.TOP_PERF
    )

    assert accepted is False
    assert "benchmarks.functional.ttft_check" in blockers
    assert "benchmarks.complete.ttft_check" in blockers
    assert "benchmarks.target.ttft_check" in blockers
    assert "actual ttft=1.2000" in blockers["benchmarks.functional.ttft_check"]
    assert "threshold ttft=10.0000" in blockers["benchmarks.functional.ttft_check"]
    assert "ratio=0.1200" in blockers["benchmarks.functional.ttft_check"]
    assert set(metadata["failed_enforced_tiers"]) == {
        "functional",
        "complete",
        "target",
    }


def test_experimental_status_treats_all_tier_failures_as_informational(report_data):
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

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.EXPERIMENTAL
    )

    assert accepted is True
    assert blockers == {}
    assert metadata["enforcement_result"] == "PASS"
    assert set(metadata["informational_tiers"]) == {"functional", "complete", "target"}
    assert "benchmarks.functional.ttft_check" in metadata["informational_blockers"]
    assert "benchmarks.complete.ttft_check" in metadata["informational_blockers"]
    assert "benchmarks.target.ttft_check" in metadata["informational_blockers"]


def test_complete_status_gates_functional_and_complete_tiers(report_data):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["benchmarks_summary"][0]["target_checks"]["complete"][
        "ttft_check"
    ] = 3
    failed_report_data["benchmarks_summary"][0]["target_checks"]["target"][
        "ttft_check"
    ] = 3

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.COMPLETE
    )

    assert accepted is False
    assert "benchmarks.complete.ttft_check" in blockers
    assert "benchmarks.target.ttft_check" in metadata["informational_blockers"]
    assert metadata["failed_enforced_tiers"] == ["complete"]


def test_known_issue_masks_eval_blocker(report_data):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["evals"][0]["accuracy_check"] = 3

    known_issues = [
        KnownIssue(
            workflow_type=WorkflowType.EVALS,
            reason="GH#1234 - hellaswag flaky on N150",
            task_name="hellaswag",
        ),
    ]

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.TOP_PERF, known_issues
    )

    assert accepted is True
    assert blockers == {}
    assert "evals.hellaswag" in metadata["masked_blockers"]
    masked = metadata["masked_blockers"]["evals.hellaswag"]
    assert masked["reason"] == "GH#1234 - hellaswag flaky on N150"
    assert masked["workflow_type"] == "EVALS"
    assert masked["task_name"] == "hellaswag"


def test_known_issue_masks_parameter_support_blocker(report_data):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["parameter_support_tests"]["results"]["test_temperature"][0][
        "status"
    ] = "failed"

    known_issues = [
        KnownIssue(
            workflow_type=WorkflowType.TESTS,
            reason="GH#2200 - temperature param support broken",
            task_name="test_temperature",
        ),
    ]

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.TOP_PERF, known_issues
    )

    assert accepted is True
    assert blockers == {}
    assert "parameter_support_tests.test_temperature.0" in metadata["masked_blockers"]


def test_known_issue_workflow_level_mask_covers_all_tasks(report_data):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["evals"][0]["accuracy_check"] = 3
    failed_report_data["evals"][1]["accuracy_check"] = 3

    known_issues = [
        KnownIssue(
            workflow_type=WorkflowType.EVALS,
            reason="GH#9999 - whole evals workflow flaky",
        ),
    ]

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.TOP_PERF, known_issues
    )

    assert accepted is True
    assert blockers == {}
    assert "evals.hellaswag" in metadata["masked_blockers"]
    assert "evals.mmlu" in metadata["masked_blockers"]


def test_combined_tier_and_known_issue_masking(report_data):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["benchmarks_summary"][0]["target_checks"]["functional"][
        "ttft_check"
    ] = 3
    failed_report_data["benchmarks_summary"][0]["target_checks"]["complete"][
        "ttft_check"
    ] = 3
    failed_report_data["evals"][0]["accuracy_check"] = 3

    known_issues = [
        KnownIssue(
            workflow_type=WorkflowType.EVALS,
            reason="GH#321 - hellaswag flaky",
            task_name="hellaswag",
        ),
    ]

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.FUNCTIONAL, known_issues
    )

    assert accepted is False
    assert "benchmarks.functional.ttft_check" in blockers
    assert "benchmarks.complete.ttft_check" in metadata["informational_blockers"]
    assert "evals.hellaswag" in metadata["masked_blockers"]


def test_known_issues_declared_is_serialized_in_metadata(report_data):
    known_issues = [
        KnownIssue(
            workflow_type=WorkflowType.BENCHMARKS,
            reason="GH#100 - benchmark mask",
        ),
    ]

    _, _, metadata = acceptance_criteria_check(
        report_data, ModelStatusTypes.TOP_PERF, known_issues
    )

    declared = metadata["known_issues_declared"]
    assert len(declared) == 1
    assert declared[0]["workflow_type"] == "BENCHMARKS"
    assert declared[0]["task_name"] is None
    assert declared[0]["reason"] == "GH#100 - benchmark mask"


def test_final_report_json_shape_contains_bool_and_metadata(report_data, tmp_path):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["benchmarks_summary"][0]["target_checks"]["complete"][
        "ttft_check"
    ] = 3
    failed_report_data["benchmarks_summary"][0]["target_checks"]["target"][
        "ttft_check"
    ] = 3

    known_issues = [
        KnownIssue(
            workflow_type=WorkflowType.EVALS,
            reason="GH#42 - mask hellaswag",
            task_name="hellaswag",
        ),
    ]

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.FUNCTIONAL, known_issues
    )
    output_data = dict(failed_report_data)
    output_data["acceptance_criteria"] = accepted
    output_data["acceptance_blockers"] = blockers
    output_data["acceptance_criteria_metadata"] = metadata
    output_data["acceptance_summary_markdown"] = format_acceptance_summary_markdown(
        accepted, blockers, metadata
    )

    json_path = tmp_path / "release_report.json"
    json_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))

    assert reloaded["acceptance_criteria"] is True
    metadata_payload = reloaded["acceptance_criteria_metadata"]
    assert isinstance(metadata_payload, dict)
    expected_metadata_keys = {
        "enforcement_result",
        "model_status",
        "enforced_tiers",
        "informational_tiers",
        "failed_enforced_tiers",
        "informational_blockers",
        "masked_blockers",
        "known_issues_declared",
        "waivers_applied_count",
        "acceptance_before_masking",
    }
    assert expected_metadata_keys.issubset(metadata_payload.keys())
    assert metadata_payload["model_status"] == "FUNCTIONAL"
    assert metadata_payload["enforced_tiers"] == ["functional"]
    assert (
        "benchmarks.complete.ttft_check" in metadata_payload["informational_blockers"]
    )
    assert metadata_payload["known_issues_declared"][0]["workflow_type"] == "EVALS"
    assert metadata_payload["waivers_applied_count"] >= 1
    assert metadata_payload["acceptance_before_masking"] is False


def test_format_acceptance_summary_markdown_renders_masked_sections(report_data):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["benchmarks_summary"][0]["target_checks"]["complete"][
        "ttft_check"
    ] = 3
    failed_report_data["evals"][0]["accuracy_check"] = 3

    known_issues = [
        KnownIssue(
            workflow_type=WorkflowType.EVALS,
            reason="GH#321 - hellaswag flaky",
            task_name="hellaswag",
        ),
    ]

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.FUNCTIONAL, known_issues
    )
    markdown = format_acceptance_summary_markdown(accepted, blockers, metadata)

    assert "Model status: `FUNCTIONAL`" in markdown
    assert "#### Informational (tier above model status)" in markdown
    assert "#### Masked (known issues)" in markdown
    assert "GH#321 - hellaswag flaky" in markdown


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

    accepted, blockers, _ = acceptance_criteria_check(failed_report_data)

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

    accepted, blockers, _ = acceptance_criteria_check(failed_report_data)

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

    accepted, blockers, _ = acceptance_criteria_check(updated_report_data)

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

    accepted, blockers, metadata = acceptance_criteria_check(failed_report_data)
    summary_markdown = format_acceptance_summary_markdown(accepted, blockers, metadata)

    if expected_blocker_key is not None:
        assert accepted is False
        assert expected_blocker_key in blockers
        assert expected_blocker_key in summary_markdown
    else:
        assert accepted is True
        assert blockers == {}


def _make_server_tests_payload(failed_test_name="DeviceLivenessTest", error="OOM"):
    return [
        {
            "summary": {
                "total_tests": 2,
                "passed": 1,
                "failed": 1,
            },
            "tests": [
                {
                    "test_name": "ImageGenerationLoadTest",
                    "success": True,
                    "error": None,
                },
                {
                    "test_name": failed_test_name,
                    "success": False,
                    "error": error,
                },
            ],
        }
    ]


def test_server_tests_blocker_added_for_failed_test(report_data):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["server_tests"] = _make_server_tests_payload(
        failed_test_name="DeviceLivenessTest", error="device unreachable"
    )

    accepted, blockers, _ = acceptance_criteria_check(failed_report_data)

    assert accepted is False
    assert "server_tests.DeviceLivenessTest.0" in blockers
    assert "device unreachable" in blockers["server_tests.DeviceLivenessTest.0"]


def test_known_issue_masks_server_tests_blocker(report_data):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["server_tests"] = _make_server_tests_payload(
        failed_test_name="DeviceLivenessTest", error="device unreachable"
    )

    known_issues = [
        KnownIssue(
            workflow_type=WorkflowType.SPEC_TESTS,
            reason="GH#5500 - liveness test flaky on T3K",
            task_name="DeviceLivenessTest",
        ),
    ]

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.TOP_PERF, known_issues
    )

    assert accepted is True
    assert blockers == {}
    masked = metadata["masked_blockers"]
    assert "server_tests.DeviceLivenessTest.0" in masked
    assert masked["server_tests.DeviceLivenessTest.0"]["workflow_type"] == "SPEC_TESTS"
    assert (
        masked["server_tests.DeviceLivenessTest.0"]["reason"]
        == "GH#5500 - liveness test flaky on T3K"
    )


def test_known_issue_workflow_level_mask_covers_all_server_tests(report_data):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["server_tests"] = [
        {
            "tests": [
                {"test_name": "TestA", "success": False, "error": "boom"},
                {"test_name": "TestB", "success": False, "error": "kaboom"},
            ]
        }
    ]

    known_issues = [
        KnownIssue(
            workflow_type=WorkflowType.SPEC_TESTS,
            reason="GH#9999 - whole spec_tests workflow flaky",
        ),
    ]

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.TOP_PERF, known_issues
    )

    assert accepted is True
    assert blockers == {}
    assert "server_tests.TestA.0" in metadata["masked_blockers"]
    assert "server_tests.TestB.0" in metadata["masked_blockers"]


def test_acceptance_summary_announces_waivers_in_header(report_data):
    failed_report_data = copy.deepcopy(report_data)
    failed_report_data["evals"][0]["accuracy_check"] = 3

    known_issues = [
        KnownIssue(
            workflow_type=WorkflowType.EVALS,
            reason="GH#321 - hellaswag flaky",
            task_name="hellaswag",
        ),
    ]

    accepted, blockers, metadata = acceptance_criteria_check(
        failed_report_data, ModelStatusTypes.TOP_PERF, known_issues
    )
    markdown = format_acceptance_summary_markdown(accepted, blockers, metadata)

    assert accepted is True
    assert metadata["waivers_applied_count"] == 1
    assert metadata["acceptance_before_masking"] is False
    assert (
        "- Acceptance status: `PASS` (with 1 waiver(s) applied; "
        "would have FAILED without masking)"
    ) in markdown
    assert "- All acceptance criteria passed (no waivers applied)." not in markdown
    assert "Acceptance criteria passed after applying 1 waiver(s)" in markdown
