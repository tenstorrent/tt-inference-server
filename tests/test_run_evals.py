#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import importlib
from pathlib import Path
from types import SimpleNamespace

import types
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.eval_config import EvalConfig, EvalTask


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
