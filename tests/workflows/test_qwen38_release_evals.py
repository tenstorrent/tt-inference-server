# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""The combined release must not silently inherit a suite-only dispatch config."""

from reference_config.evals.eval_config import _eval_config_map
from workflows.workflow_types import EvalLimitMode


def test_qwen38_release_contains_all_three_original_suites():
    tasks = _eval_config_map["Qwen/Qwen3.8-27B"].tasks
    assert [task.task_name for task in tasks] == [
        "r1_gpqa_diamond",
        "terminal_bench_2_1",
        "swe_bench_verified",
    ]
    assert tasks[0].gen_kwargs["max_gen_toks"] == 80 * 1024
    assert tasks[0].limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 0.05


def test_qwen38_release_keeps_original_agentic_cohorts_and_budgets():
    _, terminal, swe = _eval_config_map["Qwen/Qwen3.8-27B"].tasks
    terminal_config = terminal.agentic_eval_config
    swe_config = swe.swebench_eval_config
    assert terminal_config.task_names_map[EvalLimitMode.CI_NIGHTLY] == [
        "terminal-bench/break-filter-js-from-html",
        "terminal-bench/cobol-modernization",
        "terminal-bench/compile-compcert",
        "terminal-bench/feal-differential-cryptanalysis",
        "terminal-bench/qemu-startup",
    ]
    assert swe_config.instance_ids_map[EvalLimitMode.CI_NIGHTLY] == [
        "django__django-11299",
        "astropy__astropy-14096",
        "matplotlib__matplotlib-25332",
        "sympy__sympy-13551",
        "scikit-learn__scikit-learn-14629",
    ]
    assert terminal_config.agent_timeout_sec == 3 * 60 * 60
    assert terminal_config.agent_kwargs["model_info"]["max_output_tokens"] == 80 * 1024
    assert swe_config.max_output_tokens == 32 * 1024
    assert terminal_config.n_concurrent_trials == swe_config.n_concurrent_trials == 5
