# SPDX-License-Identifier: Apache-2.0
"""Resolve actual model, engine and native GPQA command without launching."""
import json
import pytest
from pathlib import Path
from types import SimpleNamespace as NS
from workflows.model_spec import load_templates_from_yaml,resolve_model_spec
from workflows.utils import get_repo_root_path
from workflows.runtime_config import RuntimeConfig
from workflows.workflow_dispatch import build_engine_commands
from llm_module.eval_configs import get_llm_eval_tasks
from llm_module.eval_command import build_eval_command

@pytest.mark.parametrize("original",[False,True])
def test_evals_routes_only_gpqa_despite_separate_agentic_selection(tmp_path,monkeypatch,original):
    import llm_module.eval_command as builder
    from reference_config.evals.qwen38_goal12 import controls
    assert controls()['evals']==['terminal_bench_2_1']
    if not original:
        monkeypatch.setattr('reference_config.evals.qwen38_gpqa_original.enabled',lambda:False)
    specs=[s for t in load_templates_from_yaml(get_repo_root_path()/'workflows/model_specs/dev/llm.yaml') for s in t.expand_to_specs() if s.hf_model_repo=='Qwen/Qwen3.8-27B']
    spec=resolve_model_spec(specs,model='Qwen/Qwen3.8-27B',device='p300x2',impl=None,catalog_name='Shield')
    rc=RuntimeConfig(model='Qwen/Qwen3.8-27B',workflow='evals',device='p300x2',service_port='8000',ci_mode=True)
    monkeypatch.setattr('workflows.workflow_dispatch.get_default_workflow_root_log_dir',lambda:tmp_path)
    commands=build_engine_commands(spec,rc,tmp_path/'spec.json')
    assert len(commands)==1 and Path(commands[0].argv[0]).name=='run_workflows.py'
    assert commands[0].argv[commands[0].argv.index('--workflow')+1]=='evals'
    assert [x.name for x in commands[0].dependency_venvs]==['EVALS_COMMON']
    tasks=get_llm_eval_tasks(spec,rc,device=spec.device_type)
    task_name='gpqa_diamond_cot_zeroshot' if original else 'r1_gpqa_diamond'
    assert [x.task_name for x in tasks]==[task_name]
    monkeypatch.setattr(builder,'get_venv_provisioner',lambda:NS(venv_path=lambda _:Path('/fake/venv')))
    command=build_eval_command(tasks[0],spec,None,str(tmp_path),8000,runtime_config=rc)
    if original:
        assert Path(command[1]).name=='qwen38_gpqa_original_guard.py'
        assert 'timeout=18000' in command[command.index('--model_args')+1]
        assert 'max_retries=0' in command[command.index('--model_args')+1]
        assert 'max_gen_toks=98304' in command[command.index('--gen_kwargs')+1]
        assert command[command.index('--seed')+1]=='0,1234,1234,1234'
    else:
        assert command[command.index('--intent')+1]=='isolated_smoke'
    assert json.loads(command[command.index('--samples')+1])=={task_name:[12,15] if original else [1,2]}
    assert '--limit' not in command
    assert 'num_concurrent=32' in command[command.index('--model_args')+1]
    assert command[command.index('--batch_size')+1]=='1'
    assert command.count('--apply_chat_template')==1
    assert tasks[0].evaluation_metadata['scored'] is original
    assert int(spec.device_model_spec.vllm_args['long_prefill_token_threshold'])==2048
