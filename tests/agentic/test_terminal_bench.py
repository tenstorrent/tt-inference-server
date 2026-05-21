#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.agentic.terminal_bench import TerminalBenchRunConfig, _write_harbor_config  # noqa: E402


def _terminal_bench_config(tmp_path, **overrides):
    defaults = dict(
        task_name="terminal_bench_2",
        dataset="terminal-bench/terminal-bench-2",
        agent="terminus-2",
        model_name="openai/Qwen/Qwen3.6-27B",
        jobs_dir=tmp_path / "agentic",
        api_base="http://127.0.0.1:8000/v1",
        n_concurrent_trials=10,
        n_attempts=1,
        environment_type="docker",
        agent_kwargs={"temperature": 1.0},
        n_tasks=None,
        override_cpus=None,
        override_memory_mb=None,
        timeout_multiplier=None,
        agent_timeout_sec=None,
        task_names=[],
        exclude_task_names=[],
        quiet=True,
        yes=True,
    )
    defaults.update(overrides)
    return TerminalBenchRunConfig(**defaults)


def test_write_harbor_config_with_agent_timeout(tmp_path):
    config = _terminal_bench_config(
        tmp_path,
        n_attempts=5,
        n_tasks=10,
        override_cpus=32,
        override_memory_mb=49152,
        timeout_multiplier=2.0,
        agent_timeout_sec=3 * 60 * 60,
    )

    config_path = _write_harbor_config(config)
    harbor_config = json.loads(config_path.read_text(encoding="utf-8"))

    assert harbor_config["job_name"] == "terminal_bench_2"
    assert harbor_config["n_attempts"] == 5
    assert harbor_config["timeout_multiplier"] == 2.0
    assert harbor_config["agent_timeout_multiplier"] == 1.0
    assert harbor_config["datasets"][0]["name"] == "terminal-bench/terminal-bench-2"
    assert harbor_config["datasets"][0]["n_tasks"] == 10
    assert harbor_config["environment"]["override_cpus"] == 32
    assert harbor_config["environment"]["override_memory_mb"] == 49152
    assert harbor_config["agents"][0]["name"] == "terminus-2"
    assert harbor_config["agents"][0]["model_name"] == "openai/Qwen/Qwen3.6-27B"
    assert harbor_config["agents"][0]["override_timeout_sec"] == 10800
    assert harbor_config["agents"][0]["kwargs"]["temperature"] == 1.0
    assert (
        harbor_config["agents"][0]["kwargs"]["api_base"] == "http://127.0.0.1:8000/v1"
    )


def test_write_harbor_config_without_agent_timeout(tmp_path):
    config = _terminal_bench_config(tmp_path, n_tasks=5)

    config_path = _write_harbor_config(config)
    harbor_config = json.loads(config_path.read_text(encoding="utf-8"))

    assert harbor_config["job_name"] == "terminal_bench_2"
    assert "agent_timeout_multiplier" not in harbor_config
    assert "timeout_multiplier" not in harbor_config
    assert harbor_config["datasets"][0]["n_tasks"] == 5


def test_write_harbor_config_sets_api_base_from_config(tmp_path):
    config = _terminal_bench_config(
        tmp_path, agent_kwargs={}, api_base="http://127.0.0.1:9000/v1"
    )

    config_path = _write_harbor_config(config)
    harbor_config = json.loads(config_path.read_text(encoding="utf-8"))

    assert (
        harbor_config["agents"][0]["kwargs"]["api_base"] == "http://127.0.0.1:9000/v1"
    )
