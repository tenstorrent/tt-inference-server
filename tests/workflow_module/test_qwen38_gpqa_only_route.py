"""Actual GPQA-only controls must finish the owning workflow, not invoke Harbor."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from reference_config.evals.qwen38_goal12 import MODEL, controls
from report_module.schema import Block
from workflow_module.workflows import AgenticWorkflow
from workflow_module.engine_types import WorkflowVenvType


def gpqa_fixture():
    """Freeze routing inputs independently of the active release arm selection.

    Identities are the original published occupancy32 documents, not synthetic
    IDs or a newly selected sample. This tests routing; harness identity and
    per-eval configuration have separate production-path tests.
    """
    manifest = Path(__file__).parents[2] / "reference_config/evals/qwen38_goal12_occupancy32.json"
    payload = manifest.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == "2faa7e48e18bad53b6de2443c9ae53fde4a4fe0032e31e956d1ba4b3f8e3d313"
    g = json.loads(payload)["evals"]["r1_gpqa_diamond"]
    c = copy.deepcopy(controls())
    c.update(include_standard_evals=True, evals=["r1_gpqa_diamond"],
             sample_count=32, full_suite=False, per_eval={},
             selection_manifest=str(manifest),
             selection_manifest_sha256=hashlib.sha256(payload).hexdigest())
    c["gpqa_handoff"].update(manifest=str(manifest),
        manifest_sha256=c["selection_manifest_sha256"], doc_ids=g["doc_ids"],
        document_sha256=g["document_sha256"], population_sha256=g["population_sha256"])
    task = SimpleNamespace(task_name="r1_gpqa_diamond",
        workflow_venv_type=WorkflowVenvType.EVALS_COMMON, doc_ids=g["doc_ids"])
    return c, [task]


def run(*, selected=None, tasks=None, blocks=None, error=None, benchmark=None):
    frozen, frozen_tasks = gpqa_fixture()
    c=frozen if selected is None else selected
    ctx=MagicMock();ctx.model_spec.hf_model_repo=MODEL
    ctx.all_params.tasks=frozen_tasks if tasks is None else tasks
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
    c,tasks=gpqa_fixture()
    assert c['evals']==['r1_gpqa_diamond'] and c['sample_count']==32
    assert len(set(tasks[0].doc_ids))==32
    h=c['gpqa_handoff']
    assert len(h['document_sha256'])==32
    assert all(len(v)==64 for v in h['document_sha256'])
    assert hashlib.sha256(Path(h['manifest']).read_bytes()).hexdigest()==h['manifest_sha256']
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
    c,_=gpqa_fixture();kw={'selected':c}
    if mode=='missing_actual':kw['tasks']=[]
    if mode=='empty_controls':c['evals']=[]
    if mode=='wrong_actual':kw['tasks']=[SimpleNamespace(task_name='other',workflow_venv_type=None)]
    if mode=='agentic_filter':kw['benchmark']='terminal_bench_2_1'
    if mode=='standard_disabled':c['include_standard_evals']=False
    out,_,_=run(**kw)
    assert out[0].exit_code!=0


def test_mixed_selection_keeps_harbor_failure_visible():
    c,_=gpqa_fixture();c['evals']=['r1_gpqa_diamond','swe_bench_verified']
    out,standard,harbor=run(selected=c)
    assert out[0].exit_code!=0
    standard.assert_called_once();harbor.assert_called_once()
