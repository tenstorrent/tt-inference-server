# SPDX-License-Identifier: Apache-2.0
"""Host-only request sizing: real native constructor, no inference or task execution."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from reference_config.evals import qwen38_goal12
from reference_config.evals.eval_config import _eval_config_map
from workflows.workflow_types import EvalLimitMode


@pytest.fixture
def arm(monkeypatch, tmp_path):
    c = qwen38_goal12.controls()
    c.update(dataset='swebench-verified', dataset_ref=None, evals=['swe_bench_verified'],
             task_names=['django__django-11299', 'astropy__astropy-14096'],
             request_timeout_sec=7200)
    path = tmp_path / 'controls.py'
    path.with_suffix('.json').write_text(json.dumps(c))
    monkeypatch.setattr(qwen38_goal12, '__file__', str(path))
    return c, path


def select():
    return qwen38_goal12.select_tasks(_eval_config_map['Qwen/Qwen3.8-27B'].tasks, EvalLimitMode)[0]


def test_swe_override_reaches_native_constructor_and_keeps_release(arm, tmp_path):
    from llm_module.agentic.qwen38_isolated_agent import IsolatedMiniSweAgent
    original = _eval_config_map['Qwen/Qwen3.8-27B']
    before = copy.deepcopy(original)
    cfg = select().agentic_eval_config
    assert cfg.llm_timeout_sec == 7200
    agent = IsolatedMiniSweAgent(logs_dir=tmp_path / 'logs', model_name='openai/Qwen/Qwen3.8-27B', **cfg.agent_kwargs)
    assert yaml.safe_load(agent._config_yaml)['model']['model_kwargs']['timeout'] == 7200
    assert agent.options.max_tokens == 32768
    assert cfg.agent_timeout_sec == 21600
    assert cfg.dataset == 'swebench-verified' and cfg.agent == 'mini-swe-agent'
    cfg.agent_kwargs['config']['model']['model_kwargs']['timeout'] = 9000
    assert original == before
    assert select().agentic_eval_config.llm_timeout_sec == 7200


@pytest.mark.parametrize('timeout', [0, 3599, -1, True, '7200', None, float('nan'), float('inf')])
def test_invalid_or_below_release_override_fails(arm, timeout):
    c, path = arm
    c['request_timeout_sec'] = timeout
    path.with_suffix('.json').write_text(json.dumps(c))
    with pytest.raises(ValueError, match='request_timeout_sec'):
        select()


def test_no_override_preserves_swe_default(arm):
    c, path = arm
    del c['request_timeout_sec']
    path.with_suffix('.json').write_text(json.dumps(c))
    base = next(t.agentic_eval_config for t in _eval_config_map['Qwen/Qwen3.8-27B'].tasks if t.task_name == 'swe_bench_verified')
    cfg = select().agentic_eval_config
    assert cfg.llm_timeout_sec == base.llm_timeout_sec == 3600
    assert cfg.agent_kwargs == base.agent_kwargs


def test_terminal_override_rejected_and_default_unchanged(arm):
    c, path = arm
    c.update(dataset='terminal-bench/terminal-bench-2-1', evals=['terminal_bench_2_1'])
    path.with_suffix('.json').write_text(json.dumps(c))
    with pytest.raises(ValueError, match='SWE'):
        select()
    del c['request_timeout_sec']
    path.with_suffix('.json').write_text(json.dumps(c))
    base = next(t.agentic_eval_config for t in _eval_config_map['Qwen/Qwen3.8-27B'].tasks if t.task_name == 'terminal_bench_2_1')
    cfg = select().agentic_eval_config
    assert cfg.agent_kwargs == base.agent_kwargs
    assert cfg.agent_kwargs['llm_kwargs']['timeout'] == 3600


def test_release_floor_is_accepted(arm):
    c, path = arm
    c['request_timeout_sec'] = 3600
    path.with_suffix('.json').write_text(json.dumps(c))
    cfg = select().agentic_eval_config
    assert cfg.llm_timeout_sec == cfg.agent_kwargs['config']['model']['model_kwargs']['timeout'] == 3600


def test_explicit_native_timeout_is_not_lowered(arm):
    from dataclasses import replace
    tasks = copy.deepcopy(_eval_config_map['Qwen/Qwen3.8-27B'].tasks)
    swe = next(t for t in tasks if t.task_name == 'swe_bench_verified')
    kwargs = copy.deepcopy(swe.agentic_eval_config.agent_kwargs)
    kwargs['config'] = {'model': {'model_kwargs': {'timeout': 12000}}}
    modified = replace(swe, agentic_eval_config=replace(swe.agentic_eval_config, agent_kwargs=kwargs))
    with pytest.raises(ValueError, match='explicit native timeout'):
        qwen38_goal12.select_tasks([modified], EvalLimitMode)
    assert kwargs['config']['model']['model_kwargs']['timeout'] == 12000
