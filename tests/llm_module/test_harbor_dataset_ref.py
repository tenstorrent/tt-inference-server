# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""HarborEvalConfig.dataset_ref pins a Hub package dataset; it reaches the
harbor job config as ``datasets[0].ref`` and is absent when unset (``@latest``)."""

import json
from pathlib import Path

from llm_module.agentic import harbor
from reference_config.evals.eval_config import (
    TERMINAL_BENCH_2_DATASET_REF,
    _eval_config_map,
)


def _run_config(tmp_path, **kw):
    return harbor.HarborRunConfig(
        task_name="tb",
        dataset="terminal-bench/terminal-bench-2",
        agent="terminus-2",
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
        venv_python=Path("/opt/venvs/evals_agentic/bin/python"),
        **kw,
    )


def _datasets(path):
    return json.loads(Path(path).read_text())["datasets"]


def test_dataset_ref_is_written_to_the_harbor_config(tmp_path):
    path = harbor._write_harbor_config(_run_config(tmp_path, dataset_ref="sha256:abc"))
    assert _datasets(path) == [
        {"name": "terminal-bench/terminal-bench-2", "ref": "sha256:abc"}
    ]


def test_no_dataset_ref_leaves_harbor_on_latest(tmp_path):
    path = harbor._write_harbor_config(_run_config(tmp_path))
    assert _datasets(path) == [{"name": "terminal-bench/terminal-bench-2"}]


def test_gemma4_terminal_bench_2_is_pinned_to_the_hub_digest():
    assert (
        TERMINAL_BENCH_2_DATASET_REF.startswith("sha256:")
        and len(TERMINAL_BENCH_2_DATASET_REF) == 71
    )
    for model in ("google/gemma-4-31B-it", "google/gemma-4-12B-it"):
        task = next(
            t
            for t in _eval_config_map[model].tasks
            if t.task_name == "terminal_bench_2"
        )
        assert task.agentic_eval_config.dataset_ref == TERMINAL_BENCH_2_DATASET_REF


def test_dataset_ref_forces_config_file_mode(tmp_path):
    # Without the ref nothing else in this config needs the file; with it the
    # pin has to travel in the datasets block, so run() must use the file.
    assert harbor._needs_config_file(_run_config(tmp_path)) is False
    assert (
        harbor._needs_config_file(_run_config(tmp_path, dataset_ref="sha256:abc"))
        is True
    )
