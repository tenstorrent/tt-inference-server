# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tests for the MiniMax provider-verifier spec tests: grading, report parsing,
child commands, and a full run against a stub verifier checkout."""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from test_module._test_common import TestConfig
from test_module.llm_tests import minimax_provider_verifier_test as mpv
from test_module.llm_tests.minimax_provider_verifier_test import (
    ENV_PREFIX,
    METRICS,
    PROVIDER_NAME,
    VERIFIER_DIR_ENV,
    VERIFIER_PYTHON_ENV,
    MiniMaxM3FormatCheckTest,
    MiniMaxProviderVerifierTest,
    grade_verifier_loops,
    loop_metrics,
    parse_junit_results,
    trigger_similarity,
)

DEFAULT_THRESHOLDS = {key: default for key, _name, _dir, default in METRICS}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep a caller's verifier overrides and API keys out of these tests."""
    for key in ("loops", "max_workers", "stream", "extra_body", "workers", "suites"):
        monkeypatch.delenv(ENV_PREFIX + key.upper(), raising=False)
    for name in (VERIFIER_DIR_ENV, VERIFIER_PYTHON_ENV, "OPENAI_API_KEY", "API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _result(index, finish, expected=None, **extra):
    return {
        "data_index": index,
        "status": "success",
        "expected_tool_call": expected,
        "response": {"choices": [{"finish_reason": finish}]},
        **extra,
    }


def _summary(**counts):
    base = {
        "success_count": 10,
        "all_count": 10,
        "tool_calls_finish_tool_calls": 8,
        "stop_finish_stop": 2,
        "tool_calls_successful_count": 8,
        "error_only_reasoning_count": 0,
        "error_only_reasoning_checked_count": 10,
        "language_following_valid_count": 1,
        "language_following_checked_count": 2,
        "scenario_check_valid_count": 1,
        "scenario_check_checked_count": 1,
    }
    return {**base, **counts}


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------


def test_trigger_similarity_pairs_runs_by_data_index():
    baseline = [_result(1, "tool_calls"), _result(2, "stop"), _result(3, "tool_calls")]
    # Out of order and missing row 2: pairing by index keeps 1 and 3 aligned.
    results = [_result(3, "tool_calls"), _result(1, "tool_calls")]

    assert trigger_similarity(baseline, results) == 1.0


def test_trigger_similarity_is_f1_of_tool_call_finishes():
    baseline = [_result(i, "tool_calls") for i in range(1, 5)] + [_result(5, "stop")]
    # tp=3 (1-3), fn=1 (4), fp=1 (5): precision = recall = 3/4.
    results = [_result(i, "tool_calls") for i in (1, 2, 3, 5)] + [_result(4, "stop")]

    assert trigger_similarity(baseline, results) == pytest.approx(0.75)
    assert trigger_similarity(baseline, []) is None


def test_loop_metrics_use_the_upstream_formulas():
    metrics = loop_metrics(
        _summary(all_count=12, tool_calls_successful_count=6),
        [],
        [],
        expected_total=10,
    )

    assert metrics["query_success_rate"] == pytest.approx(10 / 12)
    assert metrics["tool_calls_match_rate"] == pytest.approx(1.0)
    assert metrics["tool_calls_schema_accuracy"] == pytest.approx(6 / 8)
    assert metrics["error_only_reasoning_rate"] == 0.0
    assert metrics["language_following_success_rate"] == 0.5
    assert metrics["scenario_check_pass_rate"] == 1.0
    assert metrics["tool_calls_trigger_similarity"] is None


def test_unchecked_scenario_is_not_measured():
    metrics = loop_metrics(_summary(scenario_check_checked_count=0), [], [], 10)

    assert metrics["scenario_check_pass_rate"] is None


def _loop(summary=None, results=None):
    baseline = [_result(1, "tool_calls"), _result(2, "stop")]
    results = results or [
        _result(1, "tool_calls", expected=True, tool_calls_valid=True),
        _result(2, "stop", expected=False),
    ]
    return (summary or _summary()), results, baseline


def test_passing_loops_pass_every_metric():
    summary, results, baseline = _loop()

    graded = grade_verifier_loops(
        [(summary, results)] * 2, baseline, 10, DEFAULT_THRESHOLDS
    )

    assert graded["success"]
    assert graded["loops"] == 2
    assert [r["status"] for r in graded["parameter_conformance_summary"]] == [
        "✅ PASS"
    ] * len(METRICS)
    assert graded["failing_cases"] == []


def test_a_metric_below_its_threshold_fails_only_its_row():
    summary, results, baseline = _loop(_summary(tool_calls_successful_count=7))

    graded = grade_verifier_loops(
        [(summary, results)], baseline, 10, DEFAULT_THRESHOLDS
    )

    failing = [
        r["test_case"]
        for r in graded["parameter_conformance_summary"]
        if "FAIL" in r["status"]
    ]
    assert not graded["success"]
    assert failing == ["tool_calls_schema_accuracy"]


def test_error_only_reasoning_is_an_upper_bound():
    summary, results, baseline = _loop(_summary(error_only_reasoning_count=1))

    graded = grade_verifier_loops(
        [(summary, results)], baseline, 10, DEFAULT_THRESHOLDS
    )
    row = next(
        r
        for r in graded["parameter_conformance_summary"]
        if r["test_case"] == "error_only_reasoning_rate"
    )

    assert "FAIL" in row["status"]
    assert "<=" in row["summary"]


def test_metric_no_loop_measured_fails():
    summary, results, baseline = _loop(_summary(scenario_check_checked_count=0))

    graded = grade_verifier_loops(
        [(summary, results)], baseline, 10, DEFAULT_THRESHOLDS
    )
    row = graded["parameter_conformance_summary"][-1]

    assert row["test_case"] == "scenario_check_pass_rate"
    assert row["summary"] == "not measured"
    assert not graded["success"]


def test_unmeasured_loops_are_left_out_of_the_mean():
    summary, results, baseline = _loop()
    unchecked = _summary(scenario_check_checked_count=0)

    graded = grade_verifier_loops(
        [(summary, results), (unchecked, results)], baseline, 10, DEFAULT_THRESHOLDS
    )

    assert graded["metrics"]["scenario_check_pass_rate"] == 1.0


def test_failing_cases_are_counted_across_loops():
    _summary_, _results, baseline = _loop()
    bad = [
        _result(1, "stop", expected=True),
        _result(2, "tool_calls", expected=False),
        {"data_index": 3, "status": "failed", "response": {"error": "boom"}},
    ]
    good = [_result(1, "tool_calls", expected=True), _result(2, "stop", expected=False)]

    graded = grade_verifier_loops(
        [(_summary(), bad), (_summary(), good)], baseline, 10, DEFAULT_THRESHOLDS
    )
    rows = {(r["data_index"], r["check"]): r for r in graded["failing_cases"]}

    assert set(rows) == {
        (1, "missed_tool_call"),
        (2, "unexpected_tool_call"),
        (3, "request_failed"),
    }
    assert rows[(1, "missed_tool_call")]["loops_failed"] == "1/2"
    assert rows[(3, "request_failed")]["detail"] == "boom"


def test_targets_override_a_threshold():
    test = MiniMaxProviderVerifierTest(TestConfig({}), {"tool_calls_match_rate": 0.9})

    thresholds = test._resolve_thresholds()

    assert thresholds["tool_calls_match_rate"] == 0.9
    assert thresholds["query_success_rate"] == 1.0


@pytest.mark.parametrize("value", ["high", 1.5])
def test_invalid_threshold_is_rejected(value):
    test = MiniMaxProviderVerifierTest(TestConfig({}), {"query_success_rate": value})

    with pytest.raises(ValueError):
        test._resolve_thresholds()


# --------------------------------------------------------------------------
# JUnit parsing
# --------------------------------------------------------------------------


def test_junit_statuses_are_grouped_by_class():
    xml = textwrap.dedent(
        """\
        <testsuites><testsuite>
          <testcase classname="m3_text_tests.TestUsageField" name="test_a"/>
          <testcase classname="m3_text_tests.TestUsageField" name="test_b[stream]">
            <failure message="usage missing">trace</failure>
          </testcase>
          <testcase classname="m3_text_tests.TestErrorCodes" name="test_c">
            <error message="setup broke"/>
          </testcase>
          <testcase classname="m3_stream_tests.TestSSEStream" name="test_d">
            <skipped type="pytest.xfail" message="known"/>
          </testcase>
        </testsuite></testsuites>
        """
    )

    results = parse_junit_results(xml)

    assert [t["status"] for t in results["m3_text_tests.TestUsageField"]] == [
        "passed",
        "failed",
    ]
    assert results["m3_text_tests.TestUsageField"][1]["message"] == "usage missing"
    assert results["m3_text_tests.TestErrorCodes"][0]["status"] == "failed"
    assert results["m3_stream_tests.TestSSEStream"][0]["status"] == "skipped"


# --------------------------------------------------------------------------
# child commands
# --------------------------------------------------------------------------


def test_batch_verify_command_carries_the_settings(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "MAX_WORKERS", "4")
    test = MiniMaxProviderVerifierTest(
        TestConfig({"stream": True, "extra_body": {"top_p": 0.95}, "retries": 5}), {}
    )

    command = test._batch_verify_command(
        "py", tmp_path, tmp_path / "s.jsonl", tmp_path / "p.json", tmp_path / "out"
    )

    assert command[:3] == [
        "py",
        str(tmp_path / "scripts" / "batch_verify.py"),
        str(tmp_path / "s.jsonl"),
    ]
    assert command[command.index("--max-workers") + 1] == "4"
    assert command[command.index("--retries") + 1] == "5"
    assert "--stream" in command
    assert json.loads(command[command.index("--extra-body") + 1]) == {"top_p": 0.95}


def test_environment_overrides_the_loop_count(monkeypatch):
    monkeypatch.setenv(ENV_PREFIX + "LOOPS", "1")

    assert mpv._int_setting({"loops": 10}, "loops", 10) == 1


def test_format_check_command_selects_suites_and_workers(monkeypatch, tmp_path):
    test = MiniMaxM3FormatCheckTest(
        TestConfig({"workers": 4, "deselect": ["m3_text_tests.py::TestX::test_y"]}), {}
    )

    command = test._pytest_command("py", tmp_path / "junit.xml")

    assert command[:5] == [
        "py",
        "-m",
        "pytest",
        "m3_text_tests.py",
        "m3_stream_tests.py",
    ]
    assert command[command.index("-n") + 1] == "4"
    assert command[command.index("--deselect") + 1] == "m3_text_tests.py::TestX::test_y"


def test_format_check_env_without_a_key_disables_auth(tmp_path):
    test = MiniMaxM3FormatCheckTest(TestConfig({"skip_reasoning_split": True}), {})

    env = test._suite_env("MiniMaxAI/MiniMax-M3", tmp_path)

    assert env["M3_AUTH_TYPE"] == "none"
    assert env["M3_MODEL"] == "MiniMaxAI/MiniMax-M3"
    assert env["M3_SKIP_REASONING_SPLIT"] == "1"
    assert env["M3_BASE_URL"] == test.base_url


def test_format_check_env_passes_the_workflow_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    test = MiniMaxM3FormatCheckTest(TestConfig({}), {})

    env = test._suite_env("m", tmp_path)

    assert env["M3_AUTH_TYPE"] == "bearer"
    assert env["M3_API_KEY"] == "secret"


# --------------------------------------------------------------------------
# full runs against a stub checkout
# --------------------------------------------------------------------------

_STUB_BATCH_VERIFY = """
import argparse, json, os
parser = argparse.ArgumentParser()
parser.add_argument("test_file")
parser.add_argument("--provider-file")
parser.add_argument("--output-dir")
parser.add_argument("--max-workers")
parser.add_argument("--timeout")
parser.add_argument("--retries")
args = parser.parse_args()
provider = json.load(open(args.provider_file))[0]
ok = os.environ.get("STUB_SUCCESS", "1") == "1"
rows = []
for i, line in enumerate(open(args.test_file), 1):
    sample = json.loads(line)
    finish = "tool_calls" if sample.get("expected_tool_call") else "stop"
    rows.append({
        "data_index": i,
        "status": "success" if ok else "failed",
        "expected_tool_call": sample.get("expected_tool_call"),
        "tool_calls_valid": True,
        "response": {"choices": [{"finish_reason": finish}]} if ok else {"error": "refused"},
    })
name = provider["name"]
with open(os.path.join(args.output_dir, name + "_results.jsonl"), "w") as f:
    f.writelines(json.dumps(r) + "\\n" for r in rows)
summary = {
    "model": provider["model"],
    "success_count": len(rows) if ok else 0,
    "all_count": len(rows),
    "tool_calls_finish_tool_calls": 1,
    "stop_finish_stop": 1,
    "tool_calls_successful_count": 1,
    "error_only_reasoning_count": 0,
    "error_only_reasoning_checked_count": len(rows),
    "language_following_valid_count": 1,
    "language_following_checked_count": 1,
    "scenario_check_valid_count": 1,
    "scenario_check_checked_count": 1,
}
json.dump(summary, open(os.path.join(args.output_dir, name + "_summary.json"), "w"))
print("stub base_url", provider["base_url"], "model", provider["model"])
"""


@pytest.fixture
def stub_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "verifier"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "scripts" / "batch_verify.py").write_text(_STUB_BATCH_VERIFY)
    samples = [{"expected_tool_call": True}, {"expected_tool_call": False}]
    (checkout / "sample.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in samples)
    )
    baseline = checkout / mpv.DEFAULT_BASELINE_RESULTS
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps(_result(1, "tool_calls"))
        + "\n"
        + json.dumps(_result(2, "stop"))
        + "\n"
    )
    monkeypatch.setenv(VERIFIER_DIR_ENV, str(checkout))
    monkeypatch.setenv(VERIFIER_PYTHON_ENV, sys.executable)
    monkeypatch.setattr(
        MiniMaxProviderVerifierTest, "_assert_hardware_ready", lambda self: None
    )
    monkeypatch.setattr(
        MiniMaxM3FormatCheckTest, "_assert_hardware_ready", lambda self: None
    )
    monkeypatch.setattr(
        mpv._VerifierMixin, "output_root", str(tmp_path / "out"), raising=False
    )
    return checkout


def _run(cls, config):
    base = {
        "timeout": 120,
        "retry_attempts": 0,
        "retry_delay": 0,
        "break_on_failure": False,
    }
    return cls(TestConfig({**base, **config}), {}).run_tests()


def test_verifier_run_grades_every_loop(stub_checkout):
    block = _run(
        MiniMaxProviderVerifierTest, {"loops": 2, "model": "MiniMaxAI/MiniMax-M3"}
    )

    assert block.data["status"] == "pass", block.data.get("error")
    assert block.task_type == "functional"
    assert block.data["loops"] == 2
    assert block.data["endpoint_url"].endswith("/v1")
    assert block.targets["tool_calls_match_rate"] == 0.97
    loop_dir = stub_checkout.parent / "out" / "minimax_provider_verifier"
    assert len(list(loop_dir.glob("*/loop_*/" + PROVIDER_NAME + "_summary.json"))) == 2
    # The API key lives only in a temporary provider.json, not the artifacts.
    assert not list(loop_dir.glob("**/provider.json"))


def test_verifier_run_with_no_successful_request_is_an_error(
    stub_checkout, monkeypatch
):
    monkeypatch.setenv("STUB_SUCCESS", "0")

    block = _run(MiniMaxProviderVerifierTest, {"loops": 1})

    assert block.data["status"] == "error"
    assert "server unreachable" in block.data["error"]["message"]
    assert "refused" in block.data["error"]["message"]


_STUB_FORMAT_SUITE = """
import pytest

class TestUsageField:
    def test_usage_present(self):
        pass

    @pytest.mark.parametrize("stream", [False, True])
    def test_usage_last_chunk(self, stream):
        assert not stream, "usage missing in last chunk"

class TestResponseFormat:
    @pytest.mark.skip(reason="not supported")
    def test_json_object(self):
        pass
"""


def test_format_check_run_reports_classes_and_cases(stub_checkout):
    suite_dir = stub_checkout / mpv.FORMAT_CHECK_DIR
    suite_dir.mkdir()
    (suite_dir / "m3_text_tests.py").write_text(_STUB_FORMAT_SUITE)

    block = _run(
        MiniMaxM3FormatCheckTest, {"workers": 1, "suites": ["m3_text_tests.py"]}
    )

    data = block.data
    assert data["status"] == "fail", data.get("error")
    assert (data["passed"], data["failed"], data["skipped"]) == (2, 1, 1)
    summary = {
        r["test_case"]: r["status"] for r in data["parameter_conformance_summary"]
    }
    assert summary == {
        "m3_text_tests.TestResponseFormat": "⚠️ SKIP",
        "m3_text_tests.TestUsageField": "❌ FAIL",
    }
    failed = [r for r in data["detailed_test_results"] if "FAILED" in r["status"]]
    assert [r["parametrization"] for r in failed] == ["test_usage_last_chunk[True]"]


# --------------------------------------------------------------------------
# acceptance waivers
# --------------------------------------------------------------------------


def _failing_verifier_block():
    summary, results, baseline = _loop(_summary(tool_calls_successful_count=7))
    graded = grade_verifier_loops(
        [(summary, results)], baseline, 10, DEFAULT_THRESHOLDS
    )
    return MiniMaxProviderVerifierTest(TestConfig({}), {})._block(
        {**graded, "status": "fail"}
    )


def test_known_issue_naming_the_failing_metric_waives_the_block():
    from report_module.acceptance_criteria import _spec_waiver

    block = _failing_verifier_block()
    waiver = [
        {
            "workflow_type": "SPEC_TESTS",
            "task_name": "tool_calls_schema_accuracy",
            "reason": "tracked in #1",
        }
    ]

    assert _spec_waiver(block, waiver) == "tool_calls_schema_accuracy: tracked in #1"


def test_known_issue_for_another_metric_does_not_waive():
    from report_module.acceptance_criteria import _spec_waiver

    block = _failing_verifier_block()
    waiver = [
        {
            "workflow_type": "SPEC_TESTS",
            "task_name": "query_success_rate",
            "reason": "x",
        }
    ]

    assert _spec_waiver(block, waiver) is None


def test_standalone_entry_point_exits_on_the_verdict(stub_checkout, tmp_path, capsys):
    rc = mpv.main(
        [
            "--server-url",
            "http://127.0.0.1:9/",
            "--tests",
            "verifier",
            "--set",
            "loops=1",
            "--output-dir",
            str(tmp_path / "cli"),
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "MiniMaxProviderVerifierTest: PASS" in out
    assert list((tmp_path / "cli" / "minimax_provider_verifier").glob("*/loop_01"))
