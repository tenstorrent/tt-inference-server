# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
import pytest
from reference_config.evals.qwen38_goal12 import controls, validate_arm, select_tasks
from reference_config.evals.eval_config import _eval_config_map
from workflows.workflow_types import EvalLimitMode
from llm_module.parsers.agentic import AgenticEvalParser
from report_module.acceptance_criteria import _check_evals
from report_module.schema import ReportSchema


def test_explicit_intent_controls_scored_and_sample_count():
    c=controls()
    assert c['intent']=='isolated_smoke' and c['scored'] is False
    for intent in ['cost_sample','paired_confirmation','full_suite']:
        arm=copy.deepcopy(c); arm['intent']=intent; arm['scored']=intent!='cost_sample'
        if intent=='full_suite':
            arm['task_names']=['task-'+str(i) for i in range(10)];arm['sample_count']=10
        validate_arm(arm)
        arm['scored']=not arm['scored']
        with pytest.raises(ValueError,match='scored flag'):validate_arm(arm)
    arm=copy.deepcopy(c);arm['intent']='full_suite';arm['scored']=True
    with pytest.raises(ValueError,match='sample_count'):validate_arm(arm)


@pytest.mark.parametrize("timeout", [None, 7200], ids=["release-default", "selected-7200"])
def test_swe_two_task_selection_keeps_release_limits(monkeypatch, timeout):
    from reference_config.evals import qwen38_goal12
    c=controls(); c.update(intent='cost_sample',dataset='swebench-verified',dataset_ref=None,
        evals=['swe_bench_verified'],task_names=[t['name'] for t in json.loads(Path('reference_config/evals/qwen38_swe_tasks.json').read_text())['tasks'][:2]])
    c.pop("request_timeout_sec", None)
    if timeout is not None:
        c["request_timeout_sec"] = timeout
    validate_arm(c)
    monkeypatch.setattr(qwen38_goal12,'controls',lambda:c)
    original=_eval_config_map['Qwen/Qwen3.8-27B'].tasks
    snapshot=copy.deepcopy(original)
    base=next(t.agentic_eval_config for t in original if t.task_name=="swe_bench_verified")
    task=select_tasks(original,EvalLimitMode)[0]
    cfg=task.agentic_eval_config
    assert cfg.n_tasks==2 and cfg.task_names==c['task_names']
    assert cfg.agent_timeout_sec==base.agent_timeout_sec==21600
    assert base.llm_timeout_sec==3600
    assert cfg.llm_timeout_sec==(3600 if timeout is None else timeout)
    assert cfg.llm_timeout_sec>=base.llm_timeout_sec
    expected_kwargs=copy.deepcopy(base.agent_kwargs)
    if timeout is not None:
        expected_kwargs.setdefault("config",{}).setdefault("model",{}).setdefault("model_kwargs",{})["timeout"]=timeout
    assert cfg.agent_kwargs==expected_kwargs
    assert original==snapshot
    assert cfg.agent_kwargs['max_tokens']==32768
    assert task.priority=='must'
    assert task.reference_threshold_informational is True
    assert task.evaluation_metadata['intent']=='cost_sample'
    assert all(not t.reference_threshold_informational for t in original)


def test_informational_reference_keeps_score_but_not_execution_failure():
    task=_eval_config_map['Qwen/Qwen3.8-27B'].tasks[1]
    parser=AgenticEvalParser(task_name=task.task_name,score=task.score,
        reference_threshold_informational=True,evaluation_metadata={'intent':'paired_confirmation','scored':True})
    # Native result metric extraction yields zero accuracy; comparison remains FAIL.
    block=parser.parse({'stats':{'evals':{'suite':{'metrics':[{'mean':0.0}], 'n_trials':2,'n_errors':0}}}})
    assert block.data['score']==0.0
    assert block.data['accuracy_check']==3
    assert block.data['evaluation_metadata']['scored'] is True
    schema=lambda b:ReportSchema(metadata={'report_id':'test'},sections=[b])
    category=_check_evals(schema(block))
    assert not category.blockers and category.waived
    failure=parser.failure_block(return_code=123)
    assert failure.data['success'] is False and failure.data['subprocess_rc']==123
    assert _check_evals(schema(failure)).blockers
    block.data['incomplete']=True
    assert _check_evals(schema(block)).blockers
