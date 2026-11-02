# SPDX-License-Identifier: Apache-2.0
import copy,json
from dataclasses import replace
from unittest.mock import Mock
import pytest
from reference_config.evals import qwen38_goal12 as arm
from reference_config.evals.eval_config import _eval_config_map
from workflows.workflow_types import EvalLimitMode

@pytest.fixture
def controls(monkeypatch,tmp_path):
 c=arm.controls();c.pop('cohort',None)
 c.update(intent='isolated_smoke',scored=False,sample_count=2,evals=['terminal_bench_2_1'],dataset='terminal-bench/terminal-bench-2-1',
          task_names=['terminal-bench/break-filter-js-from-html','terminal-bench/cobol-modernization'],request_timeout_sec=7200,output_token_cap=32768)
 module=tmp_path/'controls.py';module.with_suffix('.json').write_text(json.dumps(c));monkeypatch.setattr(arm,'__file__',str(module))
 return c,module.with_suffix('.json')

def selected(tasks=None):
 return arm.select_tasks(tasks or _eval_config_map['Qwen/Qwen3.8-27B'].tasks,EvalLimitMode)[0]

def test_actual_selector_serializer_and_native_constructor(controls,tmp_path,monkeypatch):
 from llm_module.agentic.harbor import HarborRunConfig,_write_harbor_config
 from llm_module.agentic.qwen38_isolated_agent import IsolatedTerminus2
 import harbor.agents.terminus_2.terminus_2 as native
 before=copy.deepcopy(_eval_config_map['Qwen/Qwen3.8-27B']);task=selected();cfg=task.agentic_eval_config
 assert cfg.agent_timeout_sec==21600 and cfg.llm_timeout_sec==7200
 assert cfg.agent_kwargs['llm_kwargs']['timeout']==7200 and cfg.agent_kwargs['llm_kwargs']['max_tokens']==32768
 run=HarborRunConfig(task_name=task.task_name,dataset=cfg.dataset,agent=cfg.agent,model_name='openai/Qwen/Qwen3.8-27B',jobs_dir=tmp_path/'job',
 api_base='http://127.0.0.1:8000',n_concurrent_trials=32,n_attempts=1,environment_type='docker',agent_kwargs=cfg.agent_kwargs,n_tasks=2,
 override_cpus=None,override_memory_mb=None,timeout_multiplier=None,agent_timeout_sec=21600,llm_timeout_sec=7200)
 serialized=json.loads(_write_harbor_config(run).read_text())['agents'][0]
 assert serialized['override_timeout_sec']==21600
 factory=Mock(return_value=Mock());monkeypatch.setattr(native,'LiteLLM',factory)
 agent=IsolatedTerminus2(logs_dir=tmp_path/'logs',model_name='openai/Qwen/Qwen3.8-27B',**serialized['kwargs'])
 assert agent.options.llm_kwargs['max_tokens']==32768
 assert factory.call_args.kwargs['timeout']==7200 and factory.call_args.kwargs['max_tokens']==32768
 assert task.evaluation_metadata['output_token_cap']==32768
 assert _eval_config_map['Qwen/Qwen3.8-27B']==before

@pytest.mark.parametrize('field,value',[('request_timeout_sec',x) for x in (0,-1,3599,True,'7200',None,float('nan'),float('inf'))]+[('output_token_cap',x) for x in (0,-1,16383,True,'32768',None,32768.0,float('nan'),float('inf'))])
def test_bad_and_below_native_floor_denied(controls,field,value):
 c,p=controls;c[field]=value;p.write_text(json.dumps(c))
 with pytest.raises(ValueError,match=field):selected()

def test_defaults_unchanged_and_exact_floors_valid(controls):
 c,p=controls
 c.pop('request_timeout_sec');c.pop('output_token_cap');p.write_text(json.dumps(c))
 base=next(t.agentic_eval_config for t in _eval_config_map['Qwen/Qwen3.8-27B'].tasks if t.task_name=='terminal_bench_2_1')
 assert selected().agentic_eval_config.agent_kwargs==base.agent_kwargs
 c.update(request_timeout_sec=3600,output_token_cap=16384);p.write_text(json.dumps(c))
 cfg=selected().agentic_eval_config
 assert cfg.agent_kwargs['llm_kwargs']['timeout']==3600 and cfg.agent_kwargs['llm_kwargs']['max_tokens']==16384

def test_existing_higher_native_limits_cannot_be_lowered(controls):
 task=next(t for t in _eval_config_map['Qwen/Qwen3.8-27B'].tasks if t.task_name=='terminal_bench_2_1')
 for key,value in [('timeout',12000),('max_tokens',65536)]:
  kwargs=copy.deepcopy(task.agentic_eval_config.agent_kwargs);kwargs['llm_kwargs'][key]=value
  modified=replace(task,agentic_eval_config=replace(task.agentic_eval_config,agent_kwargs=kwargs))
  with pytest.raises(ValueError,match='release floor'):selected([modified])

def test_output_override_is_not_swe_route(controls):
 c,p=controls;c.update(evals=['swe_bench_verified'],dataset='swebench-verified');p.write_text(json.dumps(c))
 with pytest.raises(ValueError,match='Terminal Bench-only'):selected()
