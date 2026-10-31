# SPDX-License-Identifier: Apache-2.0
"""Resolve actual model, engine and native GPQA command without launching."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from workflows.model_spec import load_templates_from_yaml,resolve_model_spec
from workflows.utils import get_repo_root_path
from workflows.runtime_config import RuntimeConfig
from workflows.workflow_dispatch import build_engine_commands
from llm_module.eval_configs import get_llm_eval_tasks
from llm_module.eval_command import build_eval_command

def test_evals_routes_only_gpqa_even_when_agentic_arm_selects_swe(tmp_path,monkeypatch):
    import llm_module.eval_command as builder
    from reference_config.evals.qwen38_goal12 import controls
    assert controls()['evals']==['swe_bench_verified']
    specs=[s for t in load_templates_from_yaml(get_repo_root_path()/'workflows/model_specs/dev/llm.yaml') for s in t.expand_to_specs() if s.hf_model_repo=='Qwen/Qwen3.8-27B']
    spec=resolve_model_spec(specs,model='Qwen/Qwen3.8-27B',device='p300x2',impl=None,catalog_name='Shield')
    rc=RuntimeConfig(model='Qwen/Qwen3.8-27B',workflow='evals',device='p300x2',service_port='8000',ci_mode=True)
    monkeypatch.setattr('workflows.workflow_dispatch.get_default_workflow_root_log_dir',lambda:tmp_path)
    commands=build_engine_commands(spec,rc,tmp_path/'spec.json')
    assert len(commands)==1 and Path(commands[0].argv[0]).name=='run_workflows.py'
    assert commands[0].argv[commands[0].argv.index('--workflow')+1]=='evals'
    assert [x.name for x in commands[0].dependency_venvs]==['EVALS_COMMON']
    tasks=get_llm_eval_tasks(spec,rc,device=spec.device_type)
    assert [x.task_name for x in tasks]==['r1_gpqa_diamond']
    monkeypatch.setattr(builder,'get_venv_provisioner',lambda:NS(venv_path=lambda _:Path('/fake/venv')))
    command=build_eval_command(tasks[0],spec,None,str(tmp_path),8000,runtime_config=rc)
    assert command[command.index('--intent')+1]=='isolated_smoke'
    assert json.loads(command[command.index('--samples')+1])=={'r1_gpqa_diamond':[1,2]}
    assert '--limit' not in command
    assert 'num_concurrent=32' in command[command.index('--model_args')+1]
    assert command[command.index('--batch_size')+1]=='1'
    assert command.count('--apply_chat_template')==1
    assert tasks[0].evaluation_metadata['scored'] is False
    assert int(spec.device_model_spec.vllm_args['long_prefill_token_threshold'])==2048
