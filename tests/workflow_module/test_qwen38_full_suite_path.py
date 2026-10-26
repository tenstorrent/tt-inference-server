"""Host-only full/subset selector checks through actual command/config builders."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from llm_module import DriverContext, ServerConnection
from llm_module.agentic.harbor import _write_harbor_config
from llm_module.drivers.agentic import build_harbor_config, resolve_n_tasks, resolve_task_names
from llm_module.eval_command import build_eval_command
from llm_module.eval_configs import get_llm_eval_tasks
from reference_config.evals.eval_config import EVAL_CONFIGS
from reference_config.evals.qwen38_goal12 import MODEL, configure, controls, eval_controls
from test_module.llm_tests.agentic_eval_tests import _select_agentic_tasks
from workflows.model_spec import MODEL_SPECS
from workflows.workflow_types import EvalLimitMode


def spec():
    return next(s for s in MODEL_SPECS.values() if s.hf_model_repo == MODEL
                and s.device_type.name == 'P300X2' and s.device_model_spec.default_impl)


def configured(full, selected=None):
    c = copy.deepcopy(controls())
    c['full_suite'] = full
    if selected:
        c['evals'] = [selected]
    with patch('reference_config.evals.qwen38_goal12.controls',return_value=c):
        return configure(EVAL_CONFIGS[MODEL],EvalLimitMode), c


def runtime(ci=False):
    return SimpleNamespace(limit_samples_mode='ci_nightly' if ci else None,
                           eval_samples=None,eval_shard=None,eval_tasks=None,
                           agentic_benchmark=None)


@pytest.mark.parametrize('full,ci',[(False,True),(True,False)])
def test_gpqa_full198_vs_frozen32_actual_command(full,ci,tmp_path):
    config,c=configured(full)
    task=next(t for t in config.tasks if t.task_name=='r1_gpqa_diamond')
    e=eval_controls(c,task.task_name)
    with patch('huggingface_hub.snapshot_download',side_effect=AssertionError('network forbidden')):
        # Command construction only; no harness process or model call.
        with patch('llm_module.eval_command.get_venv_provisioner') as venv:
            venv.return_value.venv_path.return_value=tmp_path
            cmd=build_eval_command(task,spec(),None,tmp_path,8000,runtime_config=runtime(ci),tokenizer_path='/verified/local/tokenizer')
    assert '--limit' not in cmd
    if full:
        assert task.doc_ids is None
        assert '--samples' not in cmd
    else:
        assert task.doc_ids==c['gpqa_handoff']['doc_ids'][:e['sample_count']]
        assert len(task.doc_ids)==e['sample_count']
        assert json.loads(cmd[cmd.index('--samples')+1])=={'r1_gpqa_diamond':task.doc_ids}
    assert task.model_kwargs['timeout']==e['llm_timeout_sec']
    assert task.wall_clock_timeout_seconds==e['suite_timeout_sec']
    assert c['gpqa_handoff']['population']==198
    assert len(c['gpqa_handoff']['doc_ids'])==len(c['gpqa_handoff']['document_sha256'])


@pytest.mark.parametrize('name,population',[('swe_bench_verified',500),('terminal_bench_2_1',89)])
@pytest.mark.parametrize('full,ci',[(False,True),(True,False)])
def test_harbor_full_population_vs_frozen32_config(name,population,full,ci,tmp_path):
    config,c=configured(full)
    task=next(t for t in config.tasks if t.task_name==name)
    e=eval_controls(c,name)
    rc=runtime(ci)
    h=build_harbor_config(task,ServerConnection(base_url='http://127.0.0.1',service_port=8000,model=MODEL),
                         DriverContext(output_dir=tmp_path,device='P300X2'),runtime_config=rc,
                         n_tasks=resolve_n_tasks(task,rc),run_stamp='host-proof')
    body=json.loads(_write_harbor_config(h).read_text())
    dataset=body['datasets'][0]
    if full:
        assert h.task_names==[] and h.n_tasks is None
        assert 'task_names' not in dataset and 'n_tasks' not in dataset
    else:
        assert h.task_names==c['task_names'][name] and h.n_tasks==e['sample_count']
        assert dataset['task_names']==c['task_names'][name] and dataset['n_tasks']==e['sample_count']
    assert h.agent_timeout_sec==e['agent_timeout_sec']
    assert h.llm_timeout_sec==e['llm_timeout_sec']
    assert h.harbor_timeout_sec==e['suite_timeout_sec']
    manifest=json.loads(Path(c['selection_manifest']).read_text())
    assert manifest['evals'][name]['population']==population
    if name=='swe_bench_verified':
        receipt=json.loads((h.jobs_dir/f'{h.task_name}_dataset_identity.json').read_text())
        assert receipt['population']==500
        assert len(receipt['selected_task_names'])==(500 if full else e['sample_count'])
    else:
        assert dataset['ref']==c['terminal_dataset_ref']


@pytest.mark.parametrize('selected',['r1_gpqa_diamond','swe_bench_verified','terminal_bench_2_1'])
def test_full_one_eval_controls_route_actual_selectors(selected):
    config,c=configured(True,selected)
    assert [t.task_name for t in config.tasks]==[selected]
    ctx=SimpleNamespace(all_params=config,runtime_config=runtime())
    with patch('workflow_module.target_pack.get_target_pack') as pack:
        pack.return_value.eval_config.return_value=config
        standard=get_llm_eval_tasks(spec(),runtime())
    agentic=_select_agentic_tasks(ctx)
    assert [t.task_name for t in standard+agentic]==[selected]
    assert bool(standard)==(selected=='r1_gpqa_diamond')
    assert bool(agentic)==(selected!='r1_gpqa_diamond')
