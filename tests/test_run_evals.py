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


class TestClampMaxGenToks:
    """#3533 Problem 6: clamp eval-client max_gen_toks to fit within the
    server's max_context. Tasks tuned for a model's full context (e.g. Qwen3
    with max_gen_toks=32768 assuming 65K) otherwise over-subscribe a forge
    entry with smaller max_context and trigger 100% server-side rejection."""

    def test_clamps_when_max_gen_toks_exceeds_ceiling(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        # max_context=4096 -> ceiling = max(256, 4096 - 1024) = 3072.
        out = run_evals._clamp_max_gen_toks(
            {"max_gen_toks": 32768, "stream": "true"}, 4096, "task_x"
        )
        assert out["max_gen_toks"] == 3072
        assert out["stream"] == "true"

    def test_pass_through_when_within_ceiling(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        gen_kwargs = {"max_gen_toks": 256, "stream": "False"}
        out = run_evals._clamp_max_gen_toks(gen_kwargs, 4096, "task_x")
        # Returns original dict unchanged (no copy needed).
        assert out is gen_kwargs

    def test_floor_protects_tiny_max_context(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        # max_context=512 -> 512 - 1024 < 0, floor of 256 kicks in.
        out = run_evals._clamp_max_gen_toks({"max_gen_toks": 32768}, 512, "task_x")
        assert out["max_gen_toks"] == 256

    def test_no_clamp_when_max_context_unset(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        gen_kwargs = {"max_gen_toks": 32768}
        out = run_evals._clamp_max_gen_toks(gen_kwargs, None, "task_x")
        assert out is gen_kwargs

    def test_no_clamp_when_max_gen_toks_absent(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        gen_kwargs = {"stream": "False"}
        out = run_evals._clamp_max_gen_toks(gen_kwargs, 4096, "task_x")
        assert out is gen_kwargs

    def test_non_numeric_max_gen_toks_passes_through(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        gen_kwargs = {"max_gen_toks": "not-a-number"}
        out = run_evals._clamp_max_gen_toks(gen_kwargs, 4096, "task_x")
        assert out is gen_kwargs

    def test_string_numeric_max_gen_toks_clamps(self, monkeypatch):
        run_evals = _import_run_evals(monkeypatch)
        # lm-eval task defs sometimes serialize max_gen_toks as a string.
        out = run_evals._clamp_max_gen_toks({"max_gen_toks": "32768"}, 4096, "task_x")
        assert out["max_gen_toks"] == 3072
