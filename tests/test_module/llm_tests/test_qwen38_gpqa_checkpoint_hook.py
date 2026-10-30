# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
from types import SimpleNamespace as NS

import pytest

import test_module.llm_tests.llm_eval_tests as evals
import workflows.qwen38_checkpoint_preflight as preflight


def context(impl='qwen38_autoport',model='Qwen/Qwen3.8-27B'):
    return NS(model_spec=NS(hf_model_repo=model,impl=NS(impl_id=impl)),
              runtime_config=None,server_port=8000,server_host='http://localhost')


def task(name='r1_gpqa_diamond'):
    return NS(task_name=name,wall_clock_timeout_seconds=None)


def setup(monkeypatch,events):
    monkeypatch.setattr(evals,'_prepare_eval_tokenizer',lambda *a:None)
    monkeypatch.setattr(evals,'_device_label',lambda *a:'p300x2')
    monkeypatch.setattr(evals,'_limit_mode',lambda *a:None)
    monkeypatch.setattr(evals,'build_eval_command',lambda *a,**kw:['lm-eval'])
    def run(**kwargs):events.append('launch');return 0
    monkeypatch.setattr(evals,'run_command',run)


def test_current_server_checked_once_before_each_attempt(monkeypatch,tmp_path):
    events=[];calls=[];setup(monkeypatch,events)
    def check(model,jobs):
        events.append('preflight');calls.append((model,jobs))
        return {'ok':True}
    monkeypatch.setattr(preflight,'ensure_checkpoint_preflight',check)
    for attempt in ['eval-attempt-first','eval-attempt-second']:
        path=tmp_path/attempt;path.mkdir()
        (path/'checkpoint-preflight.json').write_text('{"ok":true,"owner":"stale"}')
        assert evals._run_eval_task(context(),task(),'',output_path=path)==0
    assert events==['preflight','launch','preflight','launch']
    assert calls==[('Qwen/Qwen3.8-27B',tmp_path/'eval-attempt-first'),
                   ('Qwen/Qwen3.8-27B',tmp_path/'eval-attempt-second')]


def test_failed_preflight_never_launches_eval(monkeypatch,tmp_path):
    events=[];setup(monkeypatch,events)
    def fail(*args):events.append('preflight');raise RuntimeError('checkpoint mismatch')
    monkeypatch.setattr(preflight,'ensure_checkpoint_preflight',fail)
    with pytest.raises(RuntimeError,match='checkpoint mismatch'):
        evals._run_eval_task(context(),task(),'',output_path=tmp_path)
    assert events==['preflight']


@pytest.mark.parametrize('ctx,t',[(context('default'),task()),
    (context(model='another/model'),task()),(context(),task('other_task'))])
def test_other_models_and_tasks_unchanged(monkeypatch,tmp_path,ctx,t):
    events=[];setup(monkeypatch,events)
    def forbidden(*a):raise AssertionError('unrelated preflight')
    monkeypatch.setattr(preflight,'ensure_checkpoint_preflight',forbidden)
    assert evals._run_eval_task(ctx,t,'',output_path=tmp_path)==0
    assert events==['launch']
