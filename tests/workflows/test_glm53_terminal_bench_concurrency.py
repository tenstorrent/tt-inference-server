# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""GLM-5.3 Terminal-Bench 2.1 runs 60 Harbor trials at once against pdg-g53-a9."""

import pytest

pytest.importorskip("yaml")


def test_glm53_terminal_bench_2_1_runs_60_concurrent_trials():
    from reference_config.evals.eval_config import EVAL_CONFIGS

    if "zai-org/GLM-5.3" not in EVAL_CONFIGS:
        pytest.skip("GLM-5.3 eval config not loaded")
    (task,) = [
        t
        for t in EVAL_CONFIGS["zai-org/GLM-5.3"].tasks
        if t.task_name == "terminal_bench_2_1"
    ]
    assert task.agentic_eval_config.n_concurrent_trials == 60


def test_glm53_tau3_bench_runs_60_concurrent_trials():
    from reference_config.evals.eval_config import EVAL_CONFIGS

    if "zai-org/GLM-5.3" not in EVAL_CONFIGS:
        pytest.skip("GLM-5.3 eval config not loaded")
    tau3 = [
        t
        for t in EVAL_CONFIGS["zai-org/GLM-5.3"].tasks
        if t.task_name.startswith("tau3_bench_")
    ]
    assert tau3
    assert {t.agentic_eval_config.n_concurrent_trials for t in tau3} == {60}


def test_glm53_gpqa_diamond_runs_40_concurrent_requests():
    from reference_config.evals.eval_config import EVAL_CONFIGS

    if "zai-org/GLM-5.3" not in EVAL_CONFIGS:
        pytest.skip("GLM-5.3 eval config not loaded")
    (task,) = [
        t
        for t in EVAL_CONFIGS["zai-org/GLM-5.3"].tasks
        if t.task_name == "gpqa_diamond_cot_zeroshot"
    ]
    assert task.max_concurrent == 40


def test_glm53_longbench2_runs_20_concurrent_requests():
    from reference_config.evals.eval_config import EVAL_CONFIGS

    if "zai-org/GLM-5.3" not in EVAL_CONFIGS:
        pytest.skip("GLM-5.3 eval config not loaded")
    (task,) = [
        t
        for t in EVAL_CONFIGS["zai-org/GLM-5.3"].tasks
        if t.task_name == "longbench2_generate"
    ]
    assert task.max_concurrent == 20


def test_glm53_swe_bench_verified_runs_all_tasks_at_60_concurrent_trials():
    from reference_config.evals.eval_config import EVAL_CONFIGS
    from workflows.workflow_types import WorkflowVenvType

    if "zai-org/GLM-5.3" not in EVAL_CONFIGS:
        pytest.skip("GLM-5.3 eval config not loaded")
    (task,) = [
        t
        for t in EVAL_CONFIGS["zai-org/GLM-5.3"].tasks
        if t.task_name == "swe_bench_verified"
    ]
    assert task.workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC
    cfg = task.agentic_eval_config
    assert (cfg.dataset, cfg.agent) == ("swebench-verified", "mini-swe-agent")
    assert cfg.n_concurrent_trials == 60
    assert cfg.n_tasks is None  # the whole dataset
    # GLM-5.2's GPU reference does not apply to GLM-5.3.
    assert task.score.gpu_reference_score is None
