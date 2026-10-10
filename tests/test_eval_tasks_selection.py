#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""--eval-tasks splits a model's eval set across runs without capping any task."""

from types import SimpleNamespace

import pytest

from llm_module.eval_configs import _select_tasks, eval_task_names
from llm_module.eval_command import _resolve_eval_samples


def _task(name):
    return SimpleNamespace(task_name=name, workflow_venv_type=None)


TASKS = [_task("aime25"), _task("gpqa_diamond_cot_zeroshot"), _task("mmlu_generative")]


def _rc(eval_tasks):
    return SimpleNamespace(
        eval_tasks=eval_tasks, eval_samples=None, limit_samples_mode=None
    )


def test_unset_keeps_every_task():
    assert _select_tasks(TASKS, _rc(None)) == TASKS
    assert eval_task_names(_rc("")) is None


def test_selects_named_tasks_in_configured_order():
    sel = _select_tasks(TASKS, _rc("mmlu_generative, aime25"))
    assert [t.task_name for t in sel] == ["aime25", "mmlu_generative"]


def test_an_unknown_task_name_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError, match="does not configure"):
        _select_tasks(TASKS, _rc("aime25,aime26"))


def test_selected_tasks_keep_every_sample():
    """No --samples filter is produced: the split never caps a task."""
    task = SimpleNamespace(task_name="aime25", workflow_venv_type=None)
    assert _resolve_eval_samples(task, _rc("aime25")) is None
