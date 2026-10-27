# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pathlib import Path
from unittest.mock import patch

from llm_module.agentic import deepswe, local_datasets
from llm_module.agentic.deepswe import DEEPSWE_COMMIT, DEEPSWE_DATASET
from llm_module.agentic.local_datasets import prepare_local_dataset
from reference_config.evals.eval_config import (
    ALL_EVAL_CONFIGS,
    DEEPSWE_TEST_WORKER_ENV,
    HarborEvalConfig,
)


def _harbor_configs():
    for eval_config in ALL_EVAL_CONFIGS.values():
        for task in eval_config.tasks:
            if isinstance(task.agentic_eval_config, HarborEvalConfig):
                yield task.agentic_eval_config


def _fake_git(calls):
    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            (Path(cmd[-1]) / ".git").mkdir(parents=True)
        elif "checkout" in cmd:
            repo = Path(cmd[cmd.index("-C") + 1])
            (repo / "tasks" / "abs-module-cache-flags").mkdir(parents=True)

    return run


def test_deepswe_is_a_local_dataset():
    assert local_datasets._GENERATORS[DEEPSWE_DATASET] is deepswe.generate_deepswe


def test_checks_out_the_pinned_commit_once_then_reuses_it(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    calls = []

    with patch.object(deepswe.subprocess, "run", side_effect=_fake_git(calls)):
        first = prepare_local_dataset(DEEPSWE_DATASET, python)
        n_calls = len(calls)
        second = prepare_local_dataset(DEEPSWE_DATASET, python)

    assert first == second
    assert (
        first
        == (tmp_path / "venv" / "harbor-datasets" / f"deep-swe-{DEEPSWE_COMMIT[:12]}")
        / "tasks"
    )
    assert (first / "abs-module-cache-flags").is_dir()
    assert ["--quiet", DEEPSWE_COMMIT] == calls[1][-2:]
    assert len(calls) == n_calls
    assert not list(first.parent.parent.glob("*.tmp-*"))


def test_test_worker_cap_is_deepswe_only():
    for config in _harbor_configs():
        if config.dataset == DEEPSWE_DATASET:
            continue
        env = {**config.environment_env, **config.verifier_env}
        assert not env.keys() & DEEPSWE_TEST_WORKER_ENV.keys(), config.dataset
