# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from copy import deepcopy
from types import SimpleNamespace

import pytest

from llm_module.eval_configs import _select_tasks
from test_module.llm_tests.agentic_eval_tests import _select_agentic_tasks
from workflows.eval_selection import select_full_eval_tasks
from workflows.runtime_config import RuntimeConfig
from workflows.workflow_dispatch import _selected_eval_tasks
from workflows.workflow_types import WorkflowVenvType


def task(name, agentic=False):
    return SimpleNamespace(
        task_name=name,
        workflow_venv_type=(
            WorkflowVenvType.EVALS_AGENTIC if agentic else WorkflowVenvType.EVALS_COMMON
        ),
        gen_kwargs={"max_gen_toks": 98304},
        agentic_eval_config=SimpleNamespace(n_tasks=89, n_attempts=3),
    )


def runtime(selection=None, **kwargs):
    return SimpleNamespace(eval_tasks=selection, **kwargs)


@pytest.mark.parametrize(
    "selector", [select_full_eval_tasks, _select_tasks, _selected_eval_tasks]
)
def test_full_task_selection_preserves_task_objects_and_budgets(selector):
    tasks = [task("gpqa"), task("mmlu"), task("longbench")]
    before = deepcopy(tasks)
    selected = selector(tasks, runtime("longbench,gpqa"))
    assert selected == [tasks[0], tasks[2]]
    assert selected[0] is tasks[0]
    assert selected[1] is tasks[2]
    assert tasks == before


@pytest.mark.parametrize(
    "selection", ["", "gpqa,", "gpqa,gpqa", "missing", "gpqa,missing"]
)
def test_invalid_selection_never_falls_back_to_all(selection):
    with pytest.raises(ValueError, match="--eval-tasks"):
        select_full_eval_tasks([task("gpqa")], runtime(selection))


@pytest.mark.parametrize(
    "field,value",
    [("eval_samples", "{}"), ("limit_samples_mode", "smoke-test"), ("ci_mode", True)],
)
def test_full_selection_rejects_sample_limits(field, value):
    with pytest.raises(ValueError, match="sample-limit"):
        select_full_eval_tasks([task("gpqa")], runtime("gpqa", **{field: value}))


def test_omitting_selection_preserves_default_tasks():
    tasks = [task("gpqa")]
    assert select_full_eval_tasks(tasks, None) is tasks
    assert select_full_eval_tasks(tasks, runtime()) is tasks


def test_agentic_selection_preserves_full_repeat_configuration():
    tb, tau, standard = task("terminal", True), task("tau", True), task("gpqa")
    ctx = SimpleNamespace(
        all_params=SimpleNamespace(tasks=[tb, tau, standard]),
        runtime_config=runtime("terminal"),
    )
    assert _select_agentic_tasks(ctx) == [tb]
    assert tb.agentic_eval_config.n_tasks == 89
    assert tb.agentic_eval_config.n_attempts == 3
    ctx.runtime_config = runtime("gpqa")
    with pytest.raises(ValueError, match="unavailable in this workflow"):
        _select_agentic_tasks(ctx)


def test_selection_survives_runtime_serialization():
    config = RuntimeConfig(
        model="model", workflow="evals", device="gpu", eval_tasks="longbench"
    )
    restored = RuntimeConfig.from_dict(config.to_dict())
    assert restored.eval_tasks == "longbench"
    assert restored.eval_samples is None
    assert restored.limit_samples_mode is None
    assert not restored.ci_mode
