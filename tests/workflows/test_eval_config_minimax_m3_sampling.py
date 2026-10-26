#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""MiniMax-M3 eval tasks sample with MiniMax's recommended settings.

Every M3 task, lm-eval and agentic alike, sends top_p=0.95. Each harness takes
it in a different place (lm-eval ``gen_kwargs``, terminus-2 ``llm_kwargs``,
mini-swe-agent ``model_kwargs``, the tau3 adapter's ``llm_args_json`` string),
so the check walks every nested dict and decodes JSON strings. No task
forces a thinking mode (``thinking`` or a ``thinking_mode`` template
kwarg): M3 runs with its chat template's default behaviour.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reference_config.evals.eval_config import ALL_EVAL_CONFIGS

M3_REPO = "MiniMaxAI/MiniMax-M3"


def _walk(value, key=None):
    """Yield every (key, value) pair, decoding JSON object strings on the way."""
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return
    if isinstance(value, dict):
        for k, v in value.items():
            yield k, v
            yield from _walk(v, k)


def _sampling_settings(task):
    settings = dict(task.gen_kwargs or {})
    settings["request_body"] = task.request_body
    agentic = getattr(task, "agentic_eval_config", None)
    if agentic is not None:
        settings["agent_kwargs"] = agentic.agent_kwargs
    return list(_walk(settings))


def _m3_tasks():
    tasks = ALL_EVAL_CONFIGS[M3_REPO].tasks
    assert tasks, "MiniMax-M3 has no eval tasks"
    return tasks


def test_every_m3_task_sends_top_p_0_95():
    missing, wrong = [], {}
    for task in _m3_tasks():
        values = {float(v) for k, v in _sampling_settings(task) if k == "top_p"}
        if not values:
            missing.append(task.task_name)
        elif values != {0.95}:
            wrong[task.task_name] = values
    assert not missing, f"no top_p sent: {missing}"
    assert not wrong, f"top_p other than 0.95: {wrong}"


def test_no_m3_task_overrides_thinking_mode():
    overriding = [
        task.task_name
        for task in _m3_tasks()
        if task.thinking is not None
        or any(k == "thinking_mode" for k, _ in _sampling_settings(task))
    ]
    assert not overriding, f"thinking_mode overridden: {overriding}"
