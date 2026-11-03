from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from workflows.workflow_types import WorkflowVenvType
from workflows import workflow_venvs as v
from tests.test_qwen38_gpqa_arm import spec

def test_actual_setup_selects_original_requirements_and_native_source_probe(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(v,'install_requirements',lambda config,path,overrides:(calls.append(('requirements',path)) or True))
    monkeypatch.setattr(v,'run_command',lambda cmd,**kw:(calls.append(('command',cmd)) or 0))
    cfg=v.VenvConfig(venv_type=WorkflowVenvType.EVALS_COMMON,venv_path=tmp_path,requirements_file='evals-common.txt',setup_function=v.setup_evals_common)
    cfg.setup(spec())
    assert ('requirements','evals-qwen38-original.txt') in calls
    assert any('verify_sources' in str(c) and str(cfg.venv_python) in str(c) for c in calls)
    calls.clear();cfg.setup(spec('other'))
    assert ('requirements','evals-common.txt') in calls
    assert any(v.EVALS_COMMON_LM_EVAL_COMMIT in str(c) for c in calls)

def test_original_rc_failure_nonwaivable_before_any_scoring(monkeypatch,tmp_path):
    from test_module.llm_tests import llm_eval_tests as e
    task=NS(evaluation_metadata={'intent':'original_cutoff_repair'})
    monkeypatch.setattr(e,'_fail_block',lambda *a:NS(data={'status':'fail','reason':a[-1]}))
    monkeypatch.setattr(e,'discover_eval_results',lambda *a:(_ for _ in ()).throw(AssertionError('Must not score nonzero exit')))
    result=e._score_task(NS(),task,output_path=tmp_path,rc=1,elapsed_seconds=1)
    assert result[0].data['incomplete'] is True and result[0].data['subprocess_rc']==1

def test_original_execution_errors_cannot_be_waived_but_zero_score_can_report(monkeypatch,tmp_path):
    from test_module.llm_tests import llm_eval_tests as e
    from report_module.schema import Block,ReportSchema
    from report_module.acceptance_criteria import acceptance_criteria_check
    task=NS(evaluation_metadata={'intent':'original_cutoff_repair'},task_name='gpqa_diamond_cot_zeroshot',doc_ids=[12,15],workflow_venv_type=WorkflowVenvType.EVALS_COMMON)
    ctx=NS(model_spec=NS(),runtime_config=NS(eval_samples=None,eval_shard=None))
    monkeypatch.setattr(e,'_fail_block',lambda *a:Block(kind='evals',data={'status':'fail','reason':a[-1]}))
    monkeypatch.setattr(e,'discover_eval_results',lambda *a:[])
    monkeypatch.setattr(e,'load_eval_results',lambda *a:({},{}))
    monkeypatch.setattr(e,'blocks_for_task',lambda *a,**kw:[])
    for rc in (1,0):
        block=e._score_task(ctx,task,output_path=tmp_path,rc=rc,elapsed_seconds=1)[0]
        assert block.data['incomplete'] is True
        assert not acceptance_criteria_check(ReportSchema(metadata={},sections=[block]),model_status='EXPERIMENTAL')[0]
    complete=Block(kind='evals',data={'task_name':task.task_name,'status':'pass','score':0.0,'accuracy_check':1,'reference_threshold_informational':True})
    monkeypatch.setattr(e,'blocks_for_task',lambda *a,**kw:[complete])
    (tmp_path/'samples_fixture.jsonl').write_text('{"doc_id":12}\n{"doc_id":15}\n')
    monkeypatch.setattr(e,'discover_eval_results',lambda *a:[str(tmp_path/'results_fixture.json')])
    result=e._score_task(ctx,task,output_path=tmp_path,rc=0,elapsed_seconds=1)
    assert result[0].data['score']==0.0 and not result[0].data.get('incomplete')
    assert acceptance_criteria_check(ReportSchema(metadata={},sections=result),model_status='EXPERIMENTAL')[0]

    for partial in ('{"doc_id":12}\n','{"doc_id":12}\n{"doc_id":'):
        (tmp_path/'samples_fixture.jsonl').write_text(partial)
        result=e._score_task(ctx,task,output_path=tmp_path,rc=0,elapsed_seconds=1)
        assert result[0].data['incomplete'] is True
        assert not acceptance_criteria_check(ReportSchema(metadata={},sections=result),model_status='EXPERIMENTAL')[0]

def test_original_native_environment_is_provisioned_before_server_only_for_selected_model(monkeypatch):
    import workflows.qwen38_original_admission as admission
    import workflow_module.venv_provisioner as provisioners
    calls=[]
    monkeypatch.setattr(provisioners,'get_venv_provisioner',lambda:NS(provision=lambda *a,**kw:(calls.append((a,kw)) or True)))
    admission.prepare_environment_before_server(spec('other'));assert calls==[]
    admission.prepare_environment_before_server(spec());assert calls[0][0]==(WorkflowVenvType.EVALS_COMMON,)
    source=Path('run.py').read_text();main=source[source.index('def main():'):]
    assert main.index('prepare_environment_before_server(model_spec)')<main.index('commands.append(ServerCommand(server_launch))')
