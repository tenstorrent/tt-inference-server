# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""A limit of 0 for a preset drops the task from that preset only."""

from types import SimpleNamespace

from llm_module.eval_command import _resolve_eval_limit
from llm_module.eval_configs import _select_tasks, tasks_for_limit_mode
from reference_config.evals.eval_config import EvalTask
from workflow_module.engine_types import EvalLimitMode


def _task(name, **limits):
    return SimpleNamespace(
        task_name=name,
        limit_samples_map={EvalLimitMode[k]: v for k, v in limits.items()},
    )


GPQA = _task("r1_gpqa_diamond", CI_NIGHTLY=0.2, SMOKE_TEST=0.01)
MMLU = _task("mmlu_pro", CI_NIGHTLY=0, CI_LONG=0, SMOKE_TEST=1)
LCB = _task("livecodebench")  # no entry = full set in every preset


def _runtime(mode):
    return SimpleNamespace(limit_samples_mode=mode, eval_samples=None)


def test_zero_drops_the_task_from_that_preset():
    assert tasks_for_limit_mode([GPQA, MMLU, LCB], EvalLimitMode.CI_NIGHTLY) == [
        GPQA,
        LCB,
    ]
    assert tasks_for_limit_mode([GPQA, MMLU, LCB], EvalLimitMode.CI_LONG) == [GPQA, LCB]


def test_no_preset_and_smoke_keep_every_task():
    assert tasks_for_limit_mode([GPQA, MMLU, LCB], None) == [GPQA, MMLU, LCB]
    assert tasks_for_limit_mode([GPQA, MMLU, LCB], EvalLimitMode.SMOKE_TEST) == [
        GPQA,
        MMLU,
        LCB,
    ]


def test_a_fraction_or_count_is_not_a_skip():
    kept = _task("kept", CI_NIGHTLY=0.05)
    counted = _task("counted", CI_NIGHTLY=5)
    assert tasks_for_limit_mode([kept, counted], EvalLimitMode.CI_NIGHTLY) == [
        kept,
        counted,
    ]


def test_task_without_the_attribute_is_kept():
    legacy = SimpleNamespace(task_name="ifeval")
    assert tasks_for_limit_mode([legacy], EvalLimitMode.CI_LONG) == [legacy]


def test_select_tasks_applies_the_skip_before_smoke_picks_the_first():
    assert _select_tasks([GPQA, MMLU, LCB], _runtime("ci-nightly")) == [GPQA, LCB]
    assert _select_tasks([GPQA, MMLU, LCB], _runtime("ci-long")) == [GPQA, LCB]
    assert _select_tasks([MMLU, GPQA], _runtime("smoke-test")) == [MMLU]
    assert _select_tasks([GPQA, MMLU, LCB], _runtime(None)) == [GPQA, MMLU, LCB]


def test_an_explicitly_selected_task_runs_in_full_despite_the_preset_skip():
    """--eval-tasks never caps a task: a preset's 0 is dropped from the copy that
    is handed on, so the eval command renders no --limit for it."""
    mmlu = EvalTask(
        task_name="mmlu_pro",
        limit_samples_map={EvalLimitMode.CI_NIGHTLY: 0, EvalLimitMode.SMOKE_TEST: 1},
    )
    gpqa = EvalTask(
        task_name="r1_gpqa_diamond",
        limit_samples_map={EvalLimitMode.SMOKE_TEST: 0.01},
    )
    runtime = SimpleNamespace(
        limit_samples_mode="ci-nightly", eval_samples=None, eval_tasks="mmlu_pro"
    )
    (selected,) = _select_tasks([gpqa, mmlu], runtime)
    assert selected.task_name == "mmlu_pro"
    assert EvalLimitMode.CI_NIGHTLY not in selected.limit_samples_map
    assert selected.limit_samples_map[EvalLimitMode.SMOKE_TEST] == 1
    assert _resolve_eval_limit(selected, runtime) is None
    # The catalog object is untouched: without the explicit selection the preset
    # still skips it.
    assert mmlu.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 0
    assert _resolve_eval_limit(mmlu, runtime) == 0
    assert _select_tasks([gpqa, mmlu], _runtime("ci-nightly")) == [gpqa]
