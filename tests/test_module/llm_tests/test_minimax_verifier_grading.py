# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tests for the MiniMax-Provider-Verifier grading (metrics, JUnit, pass rate)."""

from __future__ import annotations

import json

import pytest

from test_module.llm_tests import minimax_verifier_grading as grading


def _row(index, expected, reason, valid=True, status="success", message=None, **extra):
    """A verify.py results row; ``reason`` is the server's finish_reason."""
    row = {
        "data_index": index,
        "status": status,
        "expected_tool_call": expected,
        "response": {
            "choices": [
                {"finish_reason": reason, "message": message or {"content": "ok"}}
            ]
        },
        **extra,
    }
    if expected is not None:
        row["tool_calls_finish_reason"] = reason
        row["tool_calls_valid"] = valid
    return row


def _metric(metrics, key):
    return metrics[key]["value"]


def test_tool_call_metrics_follow_the_verifier_formulas():
    rows = [
        _row(1, True, "tool_calls"),
        _row(2, True, "tool_calls", valid=False),
        _row(3, True, "stop"),
        _row(4, False, "stop"),
        _row(5, False, "tool_calls"),
    ]

    metrics = grading.verify_metrics(rows, {"all_count": 5}, baseline=None)

    assert _metric(metrics, "query_success_rate") == 1.0
    # TP = 2 (cases 1, 2), TN = 1 (case 4), 5 labelled.
    assert _metric(metrics, "tool_calls_match_rate") == pytest.approx(3 / 5)
    assert _metric(metrics, "tool_calls_schema_accuracy") == pytest.approx(1 / 2)
    assert _metric(metrics, "tool_calls_trigger_similarity") is None


def test_retries_count_against_query_success():
    rows = [_row(1, True, "tool_calls"), _row(2, True, "stop", status="failed")]

    metrics = grading.verify_metrics(rows, {"all_count": 4}, baseline=None)

    assert _metric(metrics, "query_success_rate") == pytest.approx(1 / 4)


def test_a_case_routed_to_another_check_never_matches():
    row = _row(1, True, "tool_calls")
    del row["tool_calls_finish_reason"]

    metrics = grading.verify_metrics([row], None, baseline=None)

    assert _metric(metrics, "tool_calls_match_rate") == 0.0


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_reasoning_only_responses_are_counted(field):
    rows = [
        _row(1, None, "stop", message={field: "thinking...", "content": ""}),
        _row(2, None, "stop", message={field: "thinking...", "content": "answer"}),
    ]

    metrics = grading.verify_metrics(rows, None, baseline=None)

    assert _metric(metrics, "error_only_reasoning_rate") == 0.5


def test_language_and_scenario_checks_count_only_checked_cases():
    rows = [
        _row(
            1,
            None,
            "stop",
            language_following_checked=True,
            language_following_valid=False,
        ),
        _row(
            2,
            None,
            "stop",
            language_following_checked=True,
            language_following_valid=True,
        ),
        _row(3, None, "stop", scenario_check_checked=True, scenario_check_valid=True),
        _row(4, None, "stop"),
    ]

    metrics = grading.verify_metrics(rows, None, baseline=None)

    assert _metric(metrics, "language_following_success_rate") == 0.5
    assert _metric(metrics, "scenario_check_pass_rate") == 1.0


def test_trigger_similarity_is_the_f1_against_the_baseline():
    rows = [
        _row(1, True, "tool_calls"),  # TP
        _row(2, True, "tool_calls"),  # FP: the official deployment answered
        _row(3, True, "stop"),  # FN
        _row(4, False, "stop"),  # TN
        _row(9, True, "tool_calls"),  # not in the baseline: ignored
    ]
    baseline = {1: True, 2: False, 3: True, 4: False}

    sim = grading.verify_metrics(rows, None, baseline)["tool_calls_trigger_similarity"]

    # precision 1/2, recall 1/2 -> F1 1/2
    assert sim["value"] == pytest.approx(0.5)
    assert "TP=1, FP=1, FN=1, TN=1" in sim["detail"]


def test_baseline_is_read_from_the_official_results(tmp_path):
    path = tmp_path / "official_results.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r) for r in (_row(1, True, "tool_calls"), _row(2, False, "stop"))
        )
    )

    assert grading.baseline_tool_calls(path) == {1: True, 2: False}


def test_metrics_are_graded_against_their_thresholds():
    metrics = {key: {"value": None, "detail": ""} for key in grading.VERIFY_METRICS}
    metrics["tool_calls_schema_accuracy"] = {"value": 0.975, "detail": "81/83"}
    metrics["error_only_reasoning_rate"] = {"value": 0.0, "detail": ""}

    graded = grading.grade_verify(metrics, thresholds={})
    by_key = {m["key"]: m for m in graded["metrics"]}

    assert graded["success"] is False
    assert graded["failed_metrics"] == ["ToolCalls-Schema-Accuracy"]
    assert by_key["tool_calls_schema_accuracy"]["threshold"] == ">= 98.00%"
    assert by_key["error_only_reasoning_rate"]["status"] == grading.PASS
    # No value is NA, never a FAIL.
    assert by_key["tool_calls_trigger_similarity"]["status"] == grading.NA

    lowered = grading.grade_verify(metrics, {"tool_calls_schema_accuracy": 0.97})
    assert lowered["success"] is True


def test_upper_bound_metrics_fail_above_their_threshold():
    metrics = {key: {"value": 1.0, "detail": ""} for key in grading.VERIFY_METRICS}
    metrics["error_only_reasoning_rate"] = {"value": 0.01, "detail": ""}

    assert grading.grade_verify(metrics, {})["failed_metrics"] == [
        "Error-Only-Reasoning-Rate"
    ]


def test_several_runs_are_graded_on_their_mean():
    def run(value):
        return {key: {"value": value, "detail": ""} for key in grading.VERIFY_METRICS}

    mean = grading.mean_metrics([run(0.96), run(1.0)])

    assert mean["tool_calls_match_rate"]["value"] == pytest.approx(0.98)
    assert "mean of 2/2 runs" in mean["tool_calls_match_rate"]["detail"]


def test_case_failures_name_each_lost_case():
    rows = [
        _row(1, True, "stop"),
        _row(2, False, "tool_calls"),
        _row(3, True, "tool_calls"),
        _row(4, True, "tool_calls", status="failed"),
    ]
    rows[3]["response"] = {"error": "boom"}

    failures = grading.verify_case_failures(rows)

    assert [(f["case"], f["cause"]) for f in failures] == [
        (1, "tool call expected, finish_reason=stop"),
        (2, "plain answer expected, finish_reason=tool_calls"),
        (4, "request failed after all retries"),
    ]


def test_schema_failure_explains_the_violation():
    row = _row(
        1,
        True,
        "tool_calls",
        valid=False,
        message={
            "tool_calls": [
                {"function": {"name": "f", "arguments": json.dumps({"n": "3"})}}
            ]
        },
        request={
            "tools": [
                {
                    "function": {
                        "name": "f",
                        "parameters": {
                            "type": "object",
                            "properties": {"n": {"type": "integer"}},
                        },
                    }
                }
            ]
        },
    )

    assert grading.schema_failure(row).startswith("f at n: '3' is not of type")


_JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
  <testcase classname="m3_text_tests.TestBasicText" name="test_01_ok" time="1"/>
  <testcase classname="m3_text_tests.TestErrorCodes" name="test_20_05_no_key" time="1">
    <failure message="AssertionError: Expected 401">def test():
&gt;   assert r.status == 401
E   AssertionError: Expected 401, got 200</failure>
  </testcase>
  <testcase classname="m3_text_tests.TestRoleRoot" name="test_14_01" time="1">
    <failure message="role">E   assert 400 == 200</failure>
  </testcase>
  <testcase classname="m3_text_tests.TestResponseFormat" name="test_09" time="0">
    <skipped type="pytest.skip" message="unsupported"/>
  </testcase>
  <testcase classname="m3_text_tests.TestModelCompat" name="test_15" time="0">
    <skipped type="pytest.xfail" message="known"/>
  </testcase>
</testsuite></testsuites>
"""


def test_junit_outcomes_and_assertion_messages(tmp_path):
    path = tmp_path / "text.xml"
    path.write_text(_JUNIT)

    results, collection_error = grading.parse_junit(path)

    assert collection_error is None
    assert [r["outcome"] for r in results] == [
        "passed",
        "failed",
        "failed",
        "skipped",
        "xfailed",
    ]
    assert results[1]["test"] == "m3_text_tests.TestErrorCodes::test_20_05_no_key"
    assert results[1]["message"] == "AssertionError: Expected 401, got 200"


def test_a_collection_error_is_reported(tmp_path):
    path = tmp_path / "text.xml"
    path.write_text(
        '<testsuites><testsuite><testcase classname="" name="m3_text_tests">'
        "<error>ImportError\nE   ModuleNotFoundError: httpx</error>"
        "</testcase></testsuite></testsuites>"
    )

    results, collection_error = grading.parse_junit(path)

    assert results == []
    assert collection_error == "ModuleNotFoundError: httpx"


def test_pass_rate_excludes_skips_and_waived_failures(tmp_path):
    path = tmp_path / "text.xml"
    path.write_text(_JUNIT)
    results, _ = grading.parse_junit(path)

    strict = grading.grade_pytest(results, threshold=1.0)
    waived = grading.grade_pytest(
        results, threshold=0.5, waivers=[(r"TestErrorCodes::test_20_0[57]_", "no auth")]
    )

    assert strict["pass_rate"] == pytest.approx(1 / 3)
    assert strict["success"] is False
    assert (strict["skipped"], strict["xfailed"]) == (1, 1)
    assert waived["waived"] == 1
    assert waived["waived_failures"][0]["waived"] == "no auth"
    assert [f["test"] for f in waived["failures"]] == [
        "m3_text_tests.TestRoleRoot::test_14_01"
    ]
    assert waived["pass_rate"] == pytest.approx(1 / 2)
    assert waived["success"] is True
