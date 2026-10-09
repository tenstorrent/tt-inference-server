# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tests for the MiniMax-Provider-Verifier spec-test wrapper.

The verifier's processes are replaced by a fake ``_run_child`` that writes the
files ``verify.py`` / pytest would, so nothing here touches git or the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_module._test_common import TestConfig, base_test
from test_module.llm_tests.minimax_provider_verifier_test import (
    ENV_PREFIX,
    MiniMaxProviderVerifierTest,
)

MODEL = "MiniMaxAI/MiniMax-M3"
_KEYS = (
    "suites",
    "workers",
    "include_slow",
    "verify_loops",
    "verify_limit",
    "verify_baseline",
    "pytest_args",
    "waived_tests",
)


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch):
    """Keep a caller's MINIMAX_VERIFIER_* environment out of these tests."""
    for key in _KEYS:
        monkeypatch.delenv(ENV_PREFIX + key.upper(), raising=False)


def _row(index, reason="tool_calls", **extra):
    return {
        "data_index": index,
        "status": "success",
        "expected_tool_call": reason == "tool_calls",
        "tool_calls_finish_reason": reason,
        "tool_calls_valid": True,
        "response": {"choices": [{"finish_reason": reason, "message": {}}]},
        **extra,
    }


_JUNIT_PASS = (
    '<testsuites><testsuite><testcase classname="m3_text_tests.TestBasicText" '
    'name="test_01"/></testsuite></testsuites>'
)
_JUNIT_401_FAIL = (
    '<testsuites><testsuite><testcase classname="m3_text_tests.TestBasicText" '
    'name="test_01"/><testcase classname="m3_text_tests.TestErrorCodes" '
    'name="test_20_05_no_key"><failure message="m">E   Expected 401, got 200'
    "</failure></testcase></testsuite></testsuites>"
)


class _FakeVerifier:
    """Stands in for the venv, the checkout and the verifier processes."""

    def __init__(self, tmp_path: Path):
        self.dir = tmp_path / "MiniMax-Provider-Verifier"
        (self.dir / "m3_format_check" / "logs").mkdir(parents=True)
        (self.dir / "sample.jsonl").write_text(
            "".join(json.dumps({"case": i}) + "\n" for i in range(5))
        )
        baseline = self.dir / "output-dir" / "MiniMax-M3" / "loop_01" / "official"
        baseline.mkdir(parents=True)
        (baseline / "official_results.jsonl").write_text(
            "".join(json.dumps(_row(i)) + "\n" for i in range(3))
        )
        (self.dir / ".source_commit").write_text("c0ffee\n")
        self.calls = []
        self.rows = [_row(0), _row(1), _row(2)]
        self.junit = _JUNIT_PASS
        self.token = "tok"

    async def run_child(self, test, command, cwd, env, log_path):
        self.calls.append({"command": list(command), "cwd": cwd, "env": env})
        log_path.write_text("ran\n")
        if "verify.py" in command:
            output = Path(command[command.index("--output") + 1])
            output.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
            summary = Path(command[command.index("--summary") + 1])
            summary.write_text(json.dumps({"all_count": len(self.rows)}))
        else:
            junit = next(a for a in command if a.startswith("--junitxml="))
            Path(junit.split("=", 1)[1]).write_text(self.junit)
            (cwd / "logs" / "run_x_gw0.jsonl").write_text("{}\n")
        return 0


@pytest.fixture
def fake(tmp_path, monkeypatch):
    verifier = _FakeVerifier(tmp_path)
    cls = MiniMaxProviderVerifierTest
    monkeypatch.setattr(
        cls, "_provision_verifier", lambda self: ("venv-python", verifier.dir)
    )
    monkeypatch.setattr(cls, "_auth_token", staticmethod(lambda: verifier.token))
    monkeypatch.setattr(
        cls, "_run_child", lambda self, *args: verifier.run_child(self, *args)
    )
    monkeypatch.setattr(cls, "_assert_hardware_ready", lambda self: None)
    monkeypatch.setattr(base_test, "block_id", lambda ctx: "minimax_m3_super_cluster")
    return verifier


def _run(tmp_path, config=None, targets=None):
    ctx = SimpleNamespace(
        service_port=8000,
        base_url="http://server:8000",
        output_path=str(tmp_path / "output"),
        model_spec=SimpleNamespace(hf_model_repo=MODEL),
    )
    base = {"timeout": 60, "retry_attempts": 0, "retry_delay": 0}
    test = MiniMaxProviderVerifierTest(
        TestConfig({**base, **(config or {})}), dict(targets or {}), ctx=ctx
    )
    return test.run_tests()


_VERIFY = {"suite": "verify", "verify_baseline": "output-dir/MiniMax-M3/loop_01"}


def test_verify_runs_verify_py_against_the_v1_url(fake, tmp_path):
    block = _run(tmp_path, {**_VERIFY, "workers": 8})

    (call,) = fake.calls
    command = call["command"]
    assert command[:3] == ["venv-python", "verify.py", str(fake.dir / "sample.jsonl")]
    assert command[command.index("--base-url") + 1] == "http://server:8000/v1"
    assert command[command.index("--model") + 1] == MODEL
    assert command[command.index("--concurrency") + 1] == "8"
    assert call["cwd"] == fake.dir
    assert call["env"]["OPENAI_API_KEY"] == "tok"
    assert block.data["status"] == "pass"
    assert block.data["verifier_commit"] == "c0ffee"
    assert block.data["verify_baseline"].startswith("output-dir/MiniMax-M3/loop_01/")
    metrics = {m["key"]: m for m in block.data["verify_metrics"]}
    assert metrics["tool_calls_trigger_similarity"]["value"] == 1.0
    # Each gate is surfaced on the Block's targets.
    assert block.targets["tool_calls_schema_accuracy"] == 0.98


def test_verify_fails_on_a_metric_below_its_threshold(fake, tmp_path):
    fake.rows = [_row(i) for i in range(3)] + [_row(3, tool_calls_valid=False)]

    block = _run(tmp_path, _VERIFY)

    assert block.data["status"] == "fail"
    assert block.data["failed_metrics"] == ["ToolCalls-Schema-Accuracy"]
    assert block.data["case_failures"][0]["case"] == 3


def test_targets_override_a_metric_threshold(fake, tmp_path):
    fake.rows = [_row(i) for i in range(3)] + [_row(3, tool_calls_valid=False)]

    block = _run(tmp_path, _VERIFY, targets={"tool_calls_schema_accuracy": 0.7})

    assert block.data["status"] == "pass"
    assert block.targets["tool_calls_schema_accuracy"] == 0.7


def test_verify_limit_and_loops(fake, tmp_path):
    block = _run(tmp_path, {**_VERIFY, "verify_limit": 2, "verify_loops": 2})

    assert len(fake.calls) == 2
    sample = Path(fake.calls[0]["command"][2])
    assert len(sample.read_text().splitlines()) == 2
    assert block.data["verify_runs"] == 2
    assert "mean of 2/2 runs" in block.data["verify_metrics"][0]["detail"]


def test_without_a_baseline_trigger_similarity_is_not_graded(fake, tmp_path):
    block = _run(tmp_path, {"suite": "verify"})

    metrics = {m["key"]: m for m in block.data["verify_metrics"]}
    assert metrics["tool_calls_trigger_similarity"]["status"] == "NA"
    assert block.data["status"] == "pass"


def test_an_unreachable_server_is_an_error(fake, tmp_path):
    fake.rows = [{**_row(0), "status": "failed", "response": {"error": "refused"}}]

    block = _run(tmp_path, _VERIFY)

    assert block.data["status"] == "error"
    assert "server unreachable" in block.data["error"]["message"]


def test_a_pytest_suite_runs_in_the_format_check_dir(fake, tmp_path):
    block = _run(
        tmp_path,
        {
            "suite": "text",
            "workers": 4,
            "include_slow": False,
            "pytest_args": "-k basic",
        },
    )

    (call,) = fake.calls
    command = call["command"]
    assert command[:4] == ["venv-python", "-m", "pytest", "m3_text_tests.py"]
    assert command[command.index("-n") + 1] == "4"
    assert ["-m", "not slow"] == command[command.index("not slow") - 1 :][:2]
    assert command[-2:] == ["-k", "basic"]
    assert call["cwd"] == fake.dir / "m3_format_check"
    assert call["env"]["M3_BASE_URL"] == "http://server:8000"
    assert call["env"]["M3_AUTH_TYPE"] == "bearer"
    assert call["env"]["M3_MODEL"] == MODEL
    assert block.data["status"] == "pass"
    assert block.targets["pass_rate_threshold"] == 1.0
    # The request logs are moved out of the checkout into the artifacts.
    artifacts = Path(block.data["artifacts_dir"])
    assert artifacts == tmp_path / "output" / "minimax_verifier" / "text"
    assert (artifacts / "logs" / "run_x_gw0.jsonl").exists()
    assert not list((fake.dir / "m3_format_check" / "logs").iterdir())


def test_slow_cases_run_by_default(fake, tmp_path):
    _run(tmp_path, {"suite": "text"})

    assert "not slow" not in fake.calls[0]["command"]


def test_without_a_key_the_401_checks_are_waived(fake, tmp_path):
    fake.token = None
    fake.junit = _JUNIT_401_FAIL

    block = _run(tmp_path, {"suite": "text"})

    assert fake.calls[0]["env"]["M3_AUTH_TYPE"] == "none"
    assert block.data["status"] == "pass"
    assert block.data["waived"] == 1


def test_a_failing_pytest_suite_fails(fake, tmp_path):
    fake.junit = _JUNIT_401_FAIL

    block = _run(tmp_path, {"suite": "text", "non_blocking": True})

    assert block.data["status"] == "fail"
    assert block.data["non_blocking"] is True
    assert block.title == "MiniMax Provider Verifier — text (non-blocking)"
    assert block.data["failures"][0]["message"] == "Expected 401, got 200"


def test_configured_waivers_and_threshold(fake, tmp_path):
    fake.junit = _JUNIT_401_FAIL

    waived = _run(
        tmp_path,
        {"suite": "text", "waived_tests": {"TestErrorCodes": "known server bug"}},
    )
    lowered = _run(tmp_path, {"suite": "text"}, targets={"pass_rate_threshold": 0.5})

    assert waived.data["status"] == "pass"
    assert lowered.data["status"] == "pass"


def test_a_missing_junit_report_is_an_error(fake, tmp_path, monkeypatch):
    async def no_report(test, command, cwd, env, log_path):
        return 4

    monkeypatch.setattr(MiniMaxProviderVerifierTest, "_run_child", no_report)

    block = _run(tmp_path, {"suite": "text"})

    assert block.data["status"] == "error"
    assert "no JUnit report" in block.data["error"]["message"]


def test_environment_overrides_test_config(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_VERIFIER_VERIFY_LIMIT", "1")
    monkeypatch.setenv("MINIMAX_VERIFIER_INCLUDE_SLOW", "0")

    _run(tmp_path, _VERIFY)
    _run(tmp_path, {"suite": "text", "include_slow": True})

    assert len(Path(fake.calls[0]["command"][2]).read_text().splitlines()) == 1
    assert "not slow" in fake.calls[1]["command"]


def test_unknown_suite_is_rejected(fake, tmp_path):
    block = _run(tmp_path, {"suite": "nope"})

    assert block.data["status"] == "error"


def test_runs_are_blocking_by_default(fake, tmp_path):
    block = _run(tmp_path, _VERIFY)

    assert "non_blocking" not in block.data
    assert block.title == "MiniMax Provider Verifier — verify"


def test_the_verifier_venv_is_provisioned_and_its_checkout_used(monkeypatch, tmp_path):
    calls = []

    class _Provisioner:
        def provision(self, venv_type, model_spec):
            calls.append((venv_type.name, model_spec))
            return True

        def venv_path(self, venv_type):
            return tmp_path

        def venv_python(self, venv_type):
            return str(tmp_path / "bin" / "python")

    import workflow_module.venv_provisioner as venv_provisioner

    monkeypatch.setattr(venv_provisioner, "get_venv_provisioner", _Provisioner)
    spec = SimpleNamespace(hf_model_repo=MODEL)
    ctx = SimpleNamespace(service_port=8000, base_url="x", model_spec=spec)
    test = MiniMaxProviderVerifierTest(TestConfig({}), {}, ctx=ctx)

    python, verifier_dir = test._provision_verifier()

    assert calls == [("MINIMAX_VERIFIER", spec)]
    assert python == str(tmp_path / "bin" / "python")
    assert verifier_dir == tmp_path / "MiniMax-Provider-Verifier"


def test_suites_selection_skips_the_other_cases(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_VERIFIER_SUITES", "verify")

    verify = _run(tmp_path, _VERIFY)
    text = _run(tmp_path, {"suite": "text", "non_blocking": True})

    assert verify.data["status"] == "pass"
    assert text.data["status"] == "skip"
    assert [c["command"][1] for c in fake.calls] == ["verify.py"]


def test_an_unknown_selected_suite_is_an_error(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_VERIFIER_SUITES", "verify,nope")

    assert _run(tmp_path, _VERIFY).data["status"] == "error"


def test_suite_cannot_be_overridden_from_the_environment(fake, tmp_path, monkeypatch):
    """It would turn every enrolled case into the same suite."""
    monkeypatch.setenv("MINIMAX_VERIFIER_SUITE", "verify")

    _run(tmp_path, {"suite": "text"})

    assert fake.calls[0]["command"][3] == "m3_text_tests.py"
