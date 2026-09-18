# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""tau3-bench runs from a locally generated task directory, not the registry.

The published ``sierra-research/tau3-bench`` package builds its image from a
Dockerfile that clones tau2-bench at an unpinned ``main``, so the scorer a
runner ends up with depends on when its docker layer cache was first warmed.
The EVALS_AGENTIC venv generates those tasks from the pinned Harbor adapter
instead, and ``dataset_path`` is how that directory reaches ``harbor run``.

A local directory is only expressible as ``datasets[].path`` in the config
file -- ``harbor run -d`` takes a dataset *name* -- so setting it must also
force the config-file route.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from llm_module.agentic import harbor


def _harbor_config(tmp_path, **overrides):
    kwargs = dict(
        task_name="tau3_bench_banking",
        dataset="sierra-research/tau3-bench",
        agent="tau3_llm_agent",
        model_name="m",
        jobs_dir=tmp_path / "jobs",
        api_base="http://localhost:8000/v1",
        n_concurrent_trials=1,
        n_attempts=1,
        environment_type="docker",
        agent_kwargs={},
        n_tasks=None,
        override_cpus=None,
        override_memory_mb=None,
        timeout_multiplier=None,
        agent_timeout_sec=None,
    )
    kwargs.update(overrides)
    return harbor.HarborRunConfig(**kwargs)


def _written_datasets(config):
    return json.loads(harbor._write_harbor_config(config).read_text())["datasets"]


def test_local_path_replaces_the_registry_name(tmp_path):
    tasks = tmp_path / "tau3-tasks"
    config = _harbor_config(tmp_path, dataset_path=tasks)

    (dataset,) = _written_datasets(config)

    assert dataset["path"] == str(tasks)
    # Harbor rejects a dataset carrying both a path and a name.
    assert "name" not in dataset


def test_registry_name_is_used_without_a_local_path(tmp_path):
    (dataset,) = _written_datasets(_harbor_config(tmp_path))

    assert dataset["name"] == "sierra-research/tau3-bench"
    assert "path" not in dataset


def test_local_path_forces_the_config_file_route(tmp_path):
    plain = _harbor_config(tmp_path)
    local = _harbor_config(tmp_path, dataset_path=tmp_path / "tau3-tasks")

    assert not harbor._needs_config_file(plain)
    assert harbor._needs_config_file(local)


def test_local_path_reaches_the_harbor_command(tmp_path):
    config = _harbor_config(tmp_path, dataset_path=tmp_path / "tau3-tasks")
    captured = {}

    def fake_run_with_progress(cmd, *a, **k):
        captured["cmd"] = cmd
        return 0

    with patch.object(
        harbor, "run_with_progress", fake_run_with_progress
    ), patch.object(harbor, "_annotate_result_file", lambda *_a, **_k: None):
        assert harbor.run(config) == 0

    cmd = captured["cmd"]
    assert "--dataset" not in cmd
    config_path = Path(cmd[cmd.index("--config") + 1])
    written = json.loads(config_path.read_text())
    assert written["datasets"] == [{"path": str(tmp_path / "tau3-tasks")}]
