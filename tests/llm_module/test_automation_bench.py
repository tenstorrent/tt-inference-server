# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""zapier/AutomationBench: the auto-bench command, its export rewritten as a
Harbor-shaped result.json, driver selection, and the MiniMax-M3 task."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from llm_module.agentic import automation_bench
from llm_module.drivers.agentic import (
    AutomationBenchAgenticDriver,
    HarborAgenticDriver,
    make_agentic_driver,
)
from llm_module.parsers.agentic import extract_harbor_metrics
from reference_config.evals.eval_config import (
    ALL_EVAL_CONFIGS,
    AUTOMATION_BENCH_DOMAINS,
    AutomationBenchEvalConfig,
    HarborEvalConfig,
)
from workflow_module.engine_types import EvalLimitMode
from workflows.workflow_venvs import AUTOMATION_BENCH_COMMIT

REQUIREMENTS = (
    Path(__file__).resolve().parents[2] / "requirements" / "automation-bench.txt"
)


def _run_config(tmp_path, **kw):
    defaults = dict(
        task_name="automation_bench",
        model_name="MiniMaxAI/MiniMax-M3",
        api_base="http://localhost:8000/v1",
        job_dir=tmp_path / "job",
        venv_python=Path("/opt/venvs/automation_bench/bin/python"),
        checkout_dir=Path("/opt/venvs/automation_bench/AutomationBench"),
        domains=list(AUTOMATION_BENCH_DOMAINS),
    )
    defaults.update(kw)
    return automation_bench.AutomationBenchRunConfig(**defaults)


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_command_runs_module_from_venv_with_pinned_api(tmp_path):
    cmd = automation_bench.build_command(_run_config(tmp_path))
    assert cmd[:3] == [
        "/opt/venvs/automation_bench/bin/python",
        "-m",
        "automationbench.scripts.eval",
    ]
    assert _flag(cmd, "--model") == "MiniMaxAI/MiniMax-M3"
    assert _flag(cmd, "--base-url") == "http://localhost:8000/v1"
    assert _flag(cmd, "--api") == "chat_completions"
    assert _flag(cmd, "--domains") == "sales,marketing,operations,support,finance,hr"
    assert _flag(cmd, "--toolset") == "api"
    assert _flag(cmd, "--max-steps") == "50"
    assert _flag(cmd, "--export-json") == str(
        tmp_path / "job" / "automation_bench_results.json"
    )
    for absent in ("--num-examples", "--tasks", "--reasoning-effort", "--extra-body"):
        assert absent not in cmd


def test_command_optional_flags(tmp_path):
    cmd = automation_bench.build_command(
        _run_config(
            tmp_path,
            n_tasks=5,
            task_names=["sales.multi_hop_lookup", "hr.training_compliance"],
            reasoning_effort="high",
            extra_body={"top_p": 0.95, "temperature": 1.0},
            max_concurrent=62,
        )
    )
    assert _flag(cmd, "--num-examples") == "5"
    assert _flag(cmd, "--tasks") == "sales.multi_hop_lookup,hr.training_compliance"
    assert _flag(cmd, "--reasoning-effort") == "high"
    assert json.loads(_flag(cmd, "--extra-body")) == {"temperature": 1.0, "top_p": 0.95}
    assert _flag(cmd, "--max-concurrent") == "62"


def _task(name, *, passed, score, errors=None):
    task = {"name": name, "passed": passed, "score": score}
    if errors:
        task["errors"] = errors
    return task


def _export():
    return {
        "meta": {"benchmark_version": "1.0.6", "max_steps": 50},
        "summary": {"aborted_tasks": ["hr.b"]},
        "tasks": [
            _task("sales.a", passed=True, score=1.0),
            _task("sales.b", passed=False, score=0.5),
            _task("hr.a", passed=True, score=1.0),
            _task("hr.b", passed=False, score=0.0, errors=["timeout"]),
            # Unscored sanity-baseline domain: reported, not counted.
            _task("simple.a", passed=False, score=0.0),
        ],
    }


def _result():
    start = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
    return automation_bench.to_harbor_result(
        _export(),
        eval_name="automation_bench",
        started_at=start,
        finished_at=start + timedelta(seconds=400),
    )


def test_result_scores_pass_rate_over_scored_domains():
    summary = _result()["automation_bench"]
    assert summary["pass_rate"] == 0.5
    assert summary["partial_credit"] == 0.625
    assert summary["n_scored_tasks"] == 4
    assert summary["aborted_tasks"] == ["hr.b"]
    assert summary["per_domain"]["sales"] == {
        "n_tasks": 2,
        "pass_rate": 0.5,
        "partial_credit": 0.75,
    }
    assert summary["per_domain"]["simple"]["n_tasks"] == 1


def test_result_reads_as_harbor_through_the_agentic_parser():
    metrics = extract_harbor_metrics(_result())
    assert metrics["accuracy"] == 0.5
    assert metrics["n_trials"] == 4
    assert metrics["n_total_trials"] == 4
    assert metrics["n_errors"] == 1
    assert metrics["n_resolved"] == 2
    assert metrics["mean_seconds_per_task"] == 100.0


def _eval_task(config):
    return SimpleNamespace(
        task_name="t", agentic_eval_config=config, score=None, priority="must"
    )


def test_driver_selected_by_config_type():
    driver = make_agentic_driver(_eval_task(AutomationBenchEvalConfig()))
    assert isinstance(driver, AutomationBenchAgenticDriver)
    harbor_cfg = HarborEvalConfig(dataset="d", agent="terminus-2")
    assert isinstance(make_agentic_driver(_eval_task(harbor_cfg)), HarborAgenticDriver)


def _m3_task():
    (task,) = [
        t
        for t in ALL_EVAL_CONFIGS["MiniMaxAI/MiniMax-M3"].tasks
        if t.task_name == "automation_bench"
    ]
    return task


def test_m3_task_runs_official_settings_with_m3_sampling():
    config = _m3_task().agentic_eval_config
    assert config.domains == AUTOMATION_BENCH_DOMAINS
    assert config.toolset == "api"
    assert config.max_steps == 50
    assert config.agent_kwargs["extra_body"] == {"temperature": 1.0, "top_p": 0.95}
    assert "reasoning_effort" not in config.agent_kwargs


def test_m3_nightly_subset_covers_every_scored_domain():
    nightly = _m3_task().agentic_eval_config.task_names_map[EvalLimitMode.CI_NIGHTLY]
    assert {automation_bench.task_domain(n) for n in nightly} == set(
        AUTOMATION_BENCH_DOMAINS
    )


def test_requirements_pin_the_lock_verifiers():
    # 0.4.x is what an unpinned install resolves, and AutomationBench 1.0.6
    # does not import against it.
    lines = REQUIREMENTS.read_text().splitlines()
    assert "verifiers==0.2.1" in lines
    assert "datasets==4.4.2" in lines
    assert len(AUTOMATION_BENCH_COMMIT) == 40
