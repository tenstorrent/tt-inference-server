# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""--eval-samples selects tasks by key; a null value runs that whole task."""

import json
from types import SimpleNamespace

from llm_module.eval_command import _resolve_eval_samples
from llm_module.eval_configs import _select_tasks
from workflows.workflow_types import WorkflowVenvType


def _task(name):
    return SimpleNamespace(
        task_name=name, workflow_venv_type=WorkflowVenvType.EVALS_COMMON
    )


TASKS = [_task("longbench2_generate"), _task("gpqa_diamond_cot_zeroshot")]


def _config(mapping):
    return SimpleNamespace(eval_samples=json.dumps(mapping), limit_samples_mode=None)


def test_null_entry_selects_only_that_task_and_passes_no_samples_filter():
    config = _config({"longbench2_generate": None})

    selected = _select_tasks(TASKS, config)

    assert [t.task_name for t in selected] == ["longbench2_generate"]
    assert _resolve_eval_samples(selected[0], config) is None


def test_index_entry_still_filters_doc_ids():
    config = _config({"gpqa_diamond_cot_zeroshot": [0, 1, 2]})

    selected = _select_tasks(TASKS, config)

    assert [t.task_name for t in selected] == ["gpqa_diamond_cot_zeroshot"]
    assert json.loads(_resolve_eval_samples(selected[0], config)) == {
        "gpqa_diamond_cot_zeroshot": [0, 1, 2]
    }
