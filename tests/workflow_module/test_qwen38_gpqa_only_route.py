"""Actual GPQA-only controls must finish the owning workflow, not invoke Harbor."""
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from reference_config.evals.eval_config import EVAL_CONFIGS
from reference_config.evals.qwen38_goal12 import MODEL, controls
from report_module.schema import Block
from workflow_module.workflows import AgenticWorkflow


def run(*, selected=None, tasks=None, blocks=None, error=None, benchmark=None):
    c=copy.deepcopy(controls()) if selected is None else selected
    ctx=MagicMock();ctx.model_spec.hf_model_repo=MODEL
    ctx.all_params.tasks=EVAL_CONFIGS[MODEL].tasks if tasks is None else tasks
    ctx.runtime_config=SimpleNamespace(agentic_benchmark=benchmark)
    if blocks is None:blocks=[Block(kind='evals',task_type='llm',title='GPQA',data={'success':True,'accuracy':0.0})]
    with patch('reference_config.evals.qwen38_goal12.controls',return_value=c), \
         patch('test_module.llm_tests.agentic_eval_tests._configure_openai_env'), \
         patch('workflow_module.venv_provisioner.get_venv_provisioner') as provision, \
         patch('test_module.llm_tests.llm_eval_tests.run_llm_eval',return_value=blocks,side_effect=error) as standard, \
         patch('test_module.llm_tests.agentic_eval_tests.run_llm_agentic_eval',side_effect=RuntimeError('No EVALS_AGENTIC tasks configured')) as harbor:
        provision.return_value.provision.return_value=True
        outcome=AgenticWorkflow(ctx).run_tasks()
    return outcome,standard,harbor


def test_real_frozen32_gpqa_only_workflow_finishes_without_harbor():
    c=controls();assert c['evals']==['r1_gpqa_diamond'] and c['sample_count']==8
    assert len(EVAL_CONFIGS[MODEL].tasks[0].doc_ids)==8
    out,standard,harbor=run()
    assert out[0].exit_code==0
    standard.assert_called_once();harbor.assert_not_called()


@pytest.mark.parametrize('mode',['exception','failure_block','empty_blocks'])
def test_standard_failure_is_not_swallowed(mode):
    kw={'error':RuntimeError('actual standard failure')} if mode=='exception' else {'blocks':[] if mode=='empty_blocks' else [Block(kind='evals',task_type='llm',title='GPQA failure',data={'success':False,'subprocess_rc':124})]}
    out,standard,harbor=run(**kw)
    assert out[0].exit_code!=0
    standard.assert_called_once();harbor.assert_not_called()


@pytest.mark.parametrize('mode',['missing_actual','empty_controls','wrong_actual','agentic_filter','standard_disabled'])
def test_invalid_or_empty_configuration_still_fails(mode):
    c=copy.deepcopy(controls());kw={'selected':c}
    if mode=='missing_actual':kw['tasks']=[]
    if mode=='empty_controls':c['evals']=[]
    if mode=='wrong_actual':kw['tasks']=[SimpleNamespace(task_name='other',workflow_venv_type=None)]
    if mode=='agentic_filter':kw['benchmark']='terminal_bench_2_1'
    if mode=='standard_disabled':c['include_standard_evals']=False
    out,_,_=run(**kw)
    assert out[0].exit_code!=0


def test_mixed_selection_keeps_harbor_failure_visible():
    c=copy.deepcopy(controls());c['evals']=['r1_gpqa_diamond','swe_bench_verified']
    out,standard,harbor=run(selected=c)
    assert out[0].exit_code!=0
    standard.assert_called_once();harbor.assert_called_once()
