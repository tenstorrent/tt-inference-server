# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""The combined release must not silently inherit a suite-only dispatch config."""

from reference_config.evals.eval_config import _eval_config_map
from workflows.workflow_types import EvalLimitMode


def test_qwen38_matplotlib_validation_selects_only_that_instance():
    tasks = _eval_config_map["Qwen/Qwen3.8-27B"].tasks
    assert [task.task_name for task in tasks] == ["swe_bench_verified"]
    assert tasks[0].swebench_eval_config.instance_ids_map[
        EvalLimitMode.CI_NIGHTLY
    ] == [
        "matplotlib__matplotlib-25332",
    ]
    assert tasks[0].swebench_eval_config.n_concurrent_trials == 1
