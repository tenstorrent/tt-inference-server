#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import types
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.eval_config import (
    TerminalBenchEvalConfig,
    EvalConfig,
    EvalTask,
    SWEbenchEvalConfig,
)
from workflows.workflow_types import WorkflowVenvType


def _import_run_evals(monkeypatch):
    base_strategy_module = types.ModuleType(
        "utils.media_clients.base_strategy_interface"
    )

    class _BaseMediaStrategy:
        pass

    base_strategy_module.BaseMediaStrategy = _BaseMediaStrategy
    monkeypatch.setitem(
        sys.modules,
        "utils.media_clients.base_strategy_interface",
        base_strategy_module,
    )

    media_factory_module = types.ModuleType("utils.media_clients.media_client_factory")

    class _MediaTaskType:
        EVALUATION = "evaluation"
        BENCHMARK = "benchmark"

    class _MediaClientFactory:
        @staticmethod
        def run_media_task(*args, **kwargs):
            return 0

    media_factory_module.MediaClientFactory = _MediaClientFactory
    media_factory_module.MediaTaskType = _MediaTaskType
    media_factory_module.STRATEGY_MAP = {}
    monkeypatch.setitem(
        sys.modules,
        "utils.media_clients.media_client_factory",
        media_factory_module,
    )
    monkeypatch.delitem(sys.modules, "evals.run_evals", raising=False)
    return importlib.import_module("evals.run_evals")


def test_select_eval_config_smoke_test_keeps_only_first_task(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    eval_config = EvalConfig(
        hf_model_repo="test/repo",
        tasks=[EvalTask(task_name="first"), EvalTask(task_name="second")],
    )
    runtime_config = SimpleNamespace(limit_samples_mode="smoke-test")

    selected_config = run_evals._select_eval_config(eval_config, runtime_config)

    assert [task.task_name for task in selected_config.tasks] == ["first"]


def test_build_eval_command_smoke_test_uses_limit_three(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    task = EvalTask(task_name="first")
    model_spec = SimpleNamespace(hf_model_repo="test/repo", model_id="test-model")
    runtime_config = SimpleNamespace(limit_samples_mode="smoke-test")

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="n150",
        output_path="/tmp/evals",
        service_port="8000",
        runtime_config=runtime_config,
    )

    limit_index = cmd.index("--limit")
    assert cmd[limit_index + 1] == str(run_evals.SMOKE_TEST_EVAL_LIMIT)


def test_build_agentic_eval_command_uses_harbor_and_vllm_base(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    task = EvalTask(
        task_name="terminal_bench_2",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        agentic_eval_config=TerminalBenchEvalConfig(
            dataset="terminal-bench/terminal-bench-2",
            agent="terminus-2",
            n_tasks=5,
            override_cpus=32,
            override_memory_mb=49152,
            agent_kwargs={
                "temperature": 0.0,
                "llm_kwargs": {"top_p": 0.95, "extra_body": {"top_k": 20}},
            },
        ),
        limit_samples_map={},
    )
    model_spec = SimpleNamespace(
        hf_model_repo="Qwen/Qwen3.6-27B",
        model_id="test-model",
    )
    runtime_config = SimpleNamespace(limit_samples_mode=None)

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="gpu",
        output_path="/tmp/evals",
        service_port="8000",
        runtime_config=runtime_config,
    )

    assert cmd[1].endswith("evals/agentic/run_agentic_eval.py")
    assert cmd[2] == "terminal-bench"
    assert cmd[cmd.index("--dataset") + 1] == "terminal-bench/terminal-bench-2"
    assert cmd[cmd.index("--agent") + 1] == "terminus-2"
    assert cmd[cmd.index("--model-name") + 1] == "openai/Qwen/Qwen3.6-27B"
    assert cmd[cmd.index("--n-tasks") + 1] == "5"
    assert cmd[cmd.index("--override-cpus") + 1] == "32"
    assert cmd[cmd.index("--override-memory-mb") + 1] == "49152"
    agent_kwargs_json = cmd[cmd.index("--agent-kwargs-json") + 1]
    agent_kwargs = json.loads(agent_kwargs_json)
    assert agent_kwargs["temperature"] == 0.0
    assert agent_kwargs["llm_kwargs"] == {"top_p": 0.95, "extra_body": {"top_k": 20}}
    assert cmd[cmd.index("--api-base") + 1] == "http://127.0.0.1:8000/v1"


def test_build_agentic_eval_command_uses_task_smoke_limit(monkeypatch):
    run_evals = _import_run_evals(monkeypatch)
    task = EvalTask(
        task_name="terminal_bench_2",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        agentic_eval_config=TerminalBenchEvalConfig(
            dataset="terminal-bench/terminal-bench-2",
            agent="terminus-2",
            n_tasks=5,
        ),
        limit_samples_map={run_evals.EvalLimitMode.SMOKE_TEST: 1},
    )
    model_spec = SimpleNamespace(
        hf_model_repo="Qwen/Qwen3.6-27B",
        model_id="test-model",
    )
    runtime_config = SimpleNamespace(limit_samples_mode="smoke-test")

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="gpu",
        output_path="/tmp/evals",
        service_port="8000",
        runtime_config=runtime_config,
    )

    assert cmd[cmd.index("--n-tasks") + 1] == "1"


def test_build_agentic_eval_command_writes_harbor_config_for_agent_timeout(
    monkeypatch, tmp_path
):
    run_evals = _import_run_evals(monkeypatch)
    task = EvalTask(
        task_name="terminal_bench_2",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        agentic_eval_config=TerminalBenchEvalConfig(
            dataset="terminal-bench/terminal-bench-2",
            agent="terminus-2",
            n_attempts=5,
            n_tasks=10,
            override_cpus=32,
            override_memory_mb=49152,
            timeout_multiplier=2.0,
            agent_timeout_sec=3 * 60 * 60,
            agent_kwargs={
                "temperature": 1.0,
            },
        ),
        limit_samples_map={},
    )
    model_spec = SimpleNamespace(
        hf_model_repo="Qwen/Qwen3.6-27B",
        model_id="test-model",
    )
    runtime_config = SimpleNamespace(limit_samples_mode=None)

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="gpu",
        output_path=tmp_path,
        service_port="8000",
        runtime_config=runtime_config,
    )

    assert cmd[1].endswith("evals/agentic/run_agentic_eval.py")
    assert cmd[2] == "terminal-bench"
    assert cmd[cmd.index("--agent-timeout-sec") + 1] == str(3 * 60 * 60)
    assert cmd[cmd.index("--n-attempts") + 1] == "5"
    assert cmd[cmd.index("--timeout-multiplier") + 1] == "2.0"
    assert cmd[cmd.index("--n-tasks") + 1] == "10"
    assert cmd[cmd.index("--override-cpus") + 1] == "32"
    assert cmd[cmd.index("--override-memory-mb") + 1] == "49152"
    agent_kwargs = json.loads(cmd[cmd.index("--agent-kwargs-json") + 1])
    assert agent_kwargs["temperature"] == 1.0


def test_build_swebench_eval_command_uses_wrapper_and_task_limit(monkeypatch, tmp_path):
    run_evals = _import_run_evals(monkeypatch)
    task = EvalTask(
        task_name="swe_bench_verified",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        swebench_eval_config=SWEbenchEvalConfig(
            dataset_name="SWE-bench/SWE-bench_Verified",
            sweagent_subset="verified",
            dataset_split="test",
            n_concurrent_trials=2,
            max_workers=3,
            n_tasks=10,
            temperature=1.0,
            top_p=0.95,
            max_input_tokens=200 * 1024,
            max_output_tokens=32 * 1024,
            completion_kwargs={"extra_body": {"top_k": 20}},
        ),
        limit_samples_map={run_evals.EvalLimitMode.SMOKE_TEST: 2},
    )
    model_spec = SimpleNamespace(
        hf_model_repo="Qwen/Qwen3.6-27B",
        model_id="test-model",
    )
    runtime_config = SimpleNamespace(limit_samples_mode="smoke-test")

    cmd = run_evals.build_eval_command(
        task=task,
        model_spec=model_spec,
        device="gpu",
        output_path=tmp_path,
        service_port="8000",
        runtime_config=runtime_config,
    )

    assert cmd[0].endswith(".venv_evals_agentic/bin/python")
    assert cmd[1].endswith("evals/agentic/run_agentic_eval.py")
    assert cmd[2] == "swebench"
    assert cmd[cmd.index("--task-name") + 1] == "swe_bench_verified"
    assert cmd[cmd.index("--dataset-name") + 1] == "SWE-bench/SWE-bench_Verified"
    assert cmd[cmd.index("--sweagent-subset") + 1] == "verified"
    assert cmd[cmd.index("--agent-backend") + 1] == "mini-swe-agent"
    assert cmd[cmd.index("--model-name") + 1] == "openai/Qwen/Qwen3.6-27B"
    assert cmd[cmd.index("--api-base") + 1] == "http://127.0.0.1:8000/v1"
    assert cmd[cmd.index("--mini-config") + 1] == "swebench.yaml"
    assert cmd[cmd.index("--mini-model-class") + 1] == "litellm"
    assert cmd[cmd.index("--mini-environment-class") + 1] == "docker"
    assert cmd[cmd.index("--n-concurrent-trials") + 1] == "2"
    assert cmd[cmd.index("--max-workers") + 1] == "3"
    assert cmd[cmd.index("--n-tasks") + 1] == "2"
    assert cmd[cmd.index("--max-input-tokens") + 1] == str(200 * 1024)
    assert cmd[cmd.index("--max-output-tokens") + 1] == str(32 * 1024)
    assert (
        cmd[cmd.index("--completion-kwargs-json") + 1]
        == '{"extra_body": {"top_k": 20}}'
    )
