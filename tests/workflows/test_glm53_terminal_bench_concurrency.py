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
