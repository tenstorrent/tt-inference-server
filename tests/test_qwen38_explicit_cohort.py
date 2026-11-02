# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
import pytest
from reference_config.evals import qwen38_goal12 as arms
from reference_config.evals.eval_config import _eval_config_map
from workflows.workflow_types import EvalLimitMode
from llm_module.agentic.qwen38_isolation import frozen_swe_tasks, SWE_POPULATION_SHA256


def arm(count=32):
    c=copy.deepcopy(arms.controls())
    population=json.loads(Path('reference_config/evals/qwen38_swe_population.json').read_text())
    c.update(intent='full_suite', scored=True, sample_count=count,
             task_names=[t['name'] for t in population['tasks'][:count]],
             population_manifest_sha256=SWE_POPULATION_SHA256)
    c['cohort']={'selection_reason':'Distinct agents to measure real cache pressure',
                 'scope':'local_measurement','operator_grant_sha256':'a'*64}
    c['cohort']=arms.compute_cohort_binding(c)
    return c


@pytest.mark.parametrize('count',[10,17,32,100,500])
def test_explicit_population_selection_preserves_native_limits(monkeypatch,count):
    c=arm(count);arms.validate_arm(c)
    monkeypatch.setattr(arms,'controls',lambda:c)
    source=_eval_config_map['Qwen/Qwen3.8-27B'].tasks
    before=copy.deepcopy(source)
    selected=arms.select_tasks(source,EvalLimitMode)
    assert len(selected)==1
    t=selected[0];cfg=t.agentic_eval_config
    assert cfg.n_tasks==count and cfg.n_concurrent_trials==32
    assert cfg.task_names==c['task_names']
    assert cfg.agent_timeout_sec==21600 and cfg.llm_timeout_sec==7200
    assert cfg.agent_kwargs['max_tokens']==32768
    assert t.evaluation_metadata['cohort']==c['cohort']
    tasks,provenance=frozen_swe_tasks(c['task_names'],SWE_POPULATION_SHA256)
    assert len(tasks)==count and provenance['population']==500
    assert [t.get_task_id().get_name() for t in tasks]==c['task_names']
    assert source==before


@pytest.mark.parametrize('mutation',[
    lambda c:c['task_names'].reverse(),
    lambda c:c.update(max_concurrent=16),
    lambda c:c.update(request_timeout_sec=8000),
    lambda c:c['cohort'].update(count=31),
    lambda c:c['cohort'].update(operator_grant_sha256='bad'),
    lambda c:c['cohort'].update(scope='anything'),
    lambda c:c['cohort'].update(selection_reason=''),
    lambda c:c.update(sample_count=True),
    lambda c:c['task_names'].__setitem__(1,c['task_names'][0]),
])
def test_explicit_cohort_rejects_unbound_or_invalid_mutations(mutation):
    c=arm();mutation(c)
    with pytest.raises(ValueError):arms.validate_arm(c)


@pytest.mark.parametrize('concurrency',[0,33,True,1.5])
def test_invalid_concurrency_denied_even_with_fresh_binding(concurrency):
    c=arm();c['max_concurrent']=concurrency;c['cohort']=arms.compute_cohort_binding(c)
    with pytest.raises(ValueError,match='concurrency'):arms.validate_arm(c)


def test_other_counts_require_binding_and_smoke_remains_two():
    c=arm();c.pop('cohort')
    with pytest.raises(ValueError,match='explicit cohort'):arms.validate_arm(c)
    c=arm();c.update(intent='isolated_smoke',scored=False);c['cohort']=arms.compute_cohort_binding(c)
    with pytest.raises(ValueError,match='exactly two'):arms.validate_arm(c)


def test_legacy_frozen_source_unchanged_and_population_explicit():
    old=json.loads(Path('reference_config/evals/qwen38_swe_tasks.json').read_text())
    names=[r['name'] for r in old['tasks']]
    tasks,_=frozen_swe_tasks(names)
    assert len(tasks)==10
    extra=arm()['task_names'][-1]
    with pytest.raises(ValueError,match='selected frozen'):frozen_swe_tasks([extra])
    with pytest.raises(ValueError,match='Unapproved'):frozen_swe_tasks(names,'f'*64)


def test_receipt_paths_do_not_change_config_binding():
    c=arm();binding=c['cohort'];c.update(preflight_json='/new/preflight',go_json='/new/go')
    assert arms.compute_cohort_binding(c)==binding
