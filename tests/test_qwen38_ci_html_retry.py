import copy,json,subprocess
from pathlib import Path
import pytest
from reference_config.evals import qwen38_goal12 as arm
from reference_config.evals.eval_config import _eval_config_map
from workflows.workflow_types import EvalLimitMode
from workflows.qwen38_harbor_diagnostics_patch import install,PATCHED

def test_actual_retry_limits_and_release_unchanged():
 original=copy.deepcopy(_eval_config_map['Qwen/Qwen3.8-27B']);c=arm.controls()
 assert c['intent']=='isolated_smoke' and c['scored'] is False
 assert c['task_names']==['terminal-bench/break-filter-js-from-html']
 selected=arm.select_tasks(original.tasks,EvalLimitMode)[0].agentic_eval_config
 assert selected.agent_timeout_sec==27000 and selected.llm_timeout_sec==14400
 assert selected.n_concurrent_trials==1 and selected.override_memory_mb==49152
 assert selected.agent_kwargs['llm_kwargs']['max_tokens']==65536
 assert selected.agent_kwargs['llm_kwargs']['timeout']==14400
 assert selected.agent_kwargs['model_info']['max_output_tokens']==65536
 assert selected.agent_kwargs['model_info']['max_input_tokens']==163840
 assert _eval_config_map['Qwen/Qwen3.8-27B']==original

@pytest.mark.parametrize('mutation',['other_task','scored','scope','concurrency','hash'])
def test_wrong_smoke_contract_rejected(mutation):
 c=arm.controls()
 if mutation=='other_task':c['task_names']=['terminal-bench/cobol-modernization']
 if mutation=='scored':c['scored']=True
 if mutation=='scope':c['cohort']['scope']='local_measurement'
 if mutation=='concurrency':c['max_concurrent']=2
 c['cohort']=arm.compute_cohort_binding(c)
 if mutation=='hash':c['cohort']['config_sha256']='0'*64
 with pytest.raises(ValueError):arm.validate_arm(c)

def test_exact_pinned_patch_install_idempotent(tmp_path):
 import hashlib
 root=tmp_path/'harbor'
 subprocess.run(['git','clone','--shared','--no-checkout','/home/mvasiljev/exp26-stage12/efforts/nov02-ci-html-harbor',str(root)],check=True,capture_output=True)
 subprocess.run(['git','checkout','--detach','e76c762fca5267dfb36e1f900f0a7724ed72e735'],cwd=root,check=True,capture_output=True)
 install(root);install(root)
 assert hashlib.sha256((root/'src/harbor/llms/lite_llm.py').read_bytes()).hexdigest()==PATCHED
 files=subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).splitlines()
 assert files==['src/harbor/llms/lite_llm.py']
 (root/'src/harbor/llms/lite_llm.py').write_text('altered')
 with pytest.raises(RuntimeError,match='target differs'):install(root)

def test_native_serialization_and_original_route(tmp_path):
 from llm_module.agentic.harbor import HarborRunConfig,_write_harbor_config
 from harbor.models.job.config import JobConfig
 cfg=arm.select_tasks(_eval_config_map['Qwen/Qwen3.8-27B'].tasks,EvalLimitMode)[0].agentic_eval_config
 run=HarborRunConfig(task_name='terminal_bench_2_1',dataset=cfg.dataset,agent=cfg.agent,model_name='openai/Qwen/Qwen3.8-27B',jobs_dir=tmp_path/'job',api_base='http://server:8000',n_concurrent_trials=1,n_attempts=1,environment_type='docker',agent_kwargs=cfg.agent_kwargs,n_tasks=1,override_cpus=cfg.override_cpus,override_memory_mb=cfg.override_memory_mb,timeout_multiplier=None,agent_timeout_sec=cfg.agent_timeout_sec,llm_timeout_sec=cfg.llm_timeout_sec,agent_import_path='llm_module.agentic.qwen38_isolated_agent:IsolatedTerminus2')
 raw=json.loads(_write_harbor_config(run).read_text());JobConfig.model_validate(raw)
 assert raw['agents'][0]['kwargs']['llm_kwargs']['max_tokens']==65536
 assert raw['agents'][0]['kwargs']['llm_kwargs']['timeout']==14400
 assert raw['agents'][0]['override_timeout_sec']==27000
 assert raw['environment']['override_memory_mb']==49152
 root=Path(__file__).parents[1]
 # Shared receipt helper adds an original-GPQA-only identity branch; its existing
 # non-original rejection paths remain covered by test_qwen38_gpqa_samples.py.
 for name in ['workflows/model_specs/dev/llm.yaml','workflows/run_docker_server.py','workflows/qwen38_serving_events.py','llm_module/agentic/qwen38_isolation.py','reference_config/evals/eval_config.py']:
  assert (root/name).read_bytes()==subprocess.check_output(['git','show','44b4aa1dcb25b83a4ff7e6b8b1e209aaf41f8d2c:'+name],cwd=root)

def test_agent_timeout_rejected_outside_exact_retry(monkeypatch):
 c=arm.controls();c['phase']='wrong_phase';c['cohort']=arm.compute_cohort_binding(c)
 monkeypatch.setattr(arm,'controls',lambda:c)
 with pytest.raises(ValueError,match='scoped to exact'):
  arm.select_tasks(_eval_config_map['Qwen/Qwen3.8-27B'].tasks,EvalLimitMode)
