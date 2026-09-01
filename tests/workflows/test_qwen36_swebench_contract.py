# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from llm_module import DriverContext, ServerConnection
from llm_module.drivers.agentic import (
    build_swebench_config,
    resolve_instance_ids,
)
from llm_module.parsers.agentic import compute_accuracy_check
from reference_config.evals.eval_config import _eval_config_map
from workflows.workflow_types import EvalLimitMode, ReportCheckTypes, WorkflowVenvType

INSTANCE = "django__django-11299"


def _task():
    config = _eval_config_map["Qwen/Qwen3.6-27B"]
    return next(task for task in config.tasks if task.task_name == "swe_bench_verified")


def test_qwen36_swebench_is_bounded_to_c1_s8192(tmp_path):
    task = _task()
    cfg = task.swebench_eval_config
    assert task.workflow_venv_type is WorkflowVenvType.EVALS_AGENTIC
    assert task.min_context_required == 8 * 1024
    assert cfg.agent_backend == "mini-swe-agent"
    assert cfg.n_concurrent_trials == 1
    assert cfg.max_workers == 1
    assert cfg.max_input_tokens == 5 * 1024
    assert cfg.max_output_tokens == 2 * 1024
    assert cfg.max_input_tokens + cfg.max_output_tokens == 7 * 1024
    assert cfg.mini_agent_kwargs == {"step_limit": 8}
    assert cfg.completion_kwargs == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }
    assert cfg.agent_generation_timeout_sec == 6 * 60 * 60
    assert cfg.swebench_timeout_sec == 30 * 60

    class Runtime:
        limit_samples_mode = "smoke-test"

    assert resolve_instance_ids(task, Runtime()) == [INSTANCE]
    run = build_swebench_config(
        task,
        ServerConnection(
            base_url="http://127.0.0.1",
            service_port=18000,
            model="Qwen/Qwen3.6-27B",
        ),
        DriverContext(output_dir=tmp_path, device="P300X2"),
        runtime_config=Runtime(),
    )
    assert run.instance_ids == [INSTANCE]
    assert run.api_base == "http://127.0.0.1:18000/v1"
    assert run.mini_agent_kwargs == {"step_limit": 8}


def test_qwen36_bounded_swe_is_report_only_until_reference_exists():
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
