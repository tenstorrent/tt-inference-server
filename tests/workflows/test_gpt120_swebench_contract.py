# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from llm_module import DriverContext, ServerConnection
from llm_module.drivers.agentic import (
    build_swebench_config,
    resolve_instance_ids,
    resolve_n_tasks,
)
from llm_module.parsers.agentic import compute_accuracy_check
from reference_config.evals.eval_config import _eval_config_map
from workflows.workflow_types import EvalLimitMode, ReportCheckTypes, WorkflowVenvType


EXPECTED_INSTANCES = [
    "django__django-11299",
    "astropy__astropy-14096",
    "matplotlib__matplotlib-25332",
    "sympy__sympy-13551",
    "scikit-learn__scikit-learn-14629",
]


class Runtime:
    limit_samples_mode = "ci-nightly"


def _task():
    config = _eval_config_map["openai/gpt-oss-120b"]
    return next(task for task in config.tasks if task.task_name == "swe_bench_verified")


def test_gpt120_swebench_is_exact_c1_124k_five_instance_contract(tmp_path):
    task = _task()
    cfg = task.swebench_eval_config
    assert task.workflow_venv_type is WorkflowVenvType.EVALS_AGENTIC
    assert cfg.dataset_name == "SWE-bench/SWE-bench_Verified"
    assert cfg.dataset_split == "test"
    assert cfg.agent_backend == "mini-swe-agent"
    assert cfg.n_concurrent_trials == 1
    assert cfg.max_input_tokens == 92 * 1024
    assert cfg.max_output_tokens == 32 * 1024
    assert cfg.max_input_tokens + cfg.max_output_tokens == 124 * 1024
    assert cfg.agent_generation_timeout_sec == 6 * 60 * 60
    assert cfg.swebench_timeout_sec == 30 * 60
    assert resolve_instance_ids(task, Runtime()) == EXPECTED_INSTANCES
    assert resolve_n_tasks(task, Runtime()) is None

    run = build_swebench_config(
        task,
        ServerConnection(
            base_url="http://127.0.0.1",
            service_port=18091,
            model="openai/gpt-oss-120b",
        ),
        DriverContext(output_dir=tmp_path, device="P300X2"),
        runtime_config=Runtime(),
    )
    assert run.instance_ids == EXPECTED_INSTANCES
    assert run.api_base == "http://127.0.0.1:18091/v1"
    assert run.n_concurrent_trials == 1


def test_gpt120_swebench_accuracy_is_report_only_until_cs_sets_reference():
    score = _task().score
    assert score.published_score is None
    assert score.gpu_reference_score is None
    assert (
        compute_accuracy_check({"accuracy": 100.0}, score, EvalLimitMode.CI_NIGHTLY)
        == ReportCheckTypes.NA
    )
    assert (
        compute_accuracy_check({"accuracy": 0.0}, score, EvalLimitMode.CI_NIGHTLY)
        == ReportCheckTypes.NA
    )
