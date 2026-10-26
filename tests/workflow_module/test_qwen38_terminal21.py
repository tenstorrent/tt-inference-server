"""Regression: an explicitly requested Terminal version must never be downgraded."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from reference_config.evals.eval_config import EVAL_CONFIGS, EvalLimitMode
from reference_config.evals.qwen38_goal12 import MODEL, configure, controls

# Arm selection filters imported catalog tasks. Generic Terminal regressions need
# the original model definition, independently of this GPQA-only arm selection.
import runpy
with patch('reference_config.evals.qwen38_goal12.configure', side_effect=lambda config, mode: config):
    RAW_MODEL_CONFIG = runpy.run_path('reference_config/evals/eval_config.py')['EVAL_CONFIGS'][MODEL]

@pytest.mark.parametrize('version,dataset', [('terminal_bench_2_1','terminal-bench/terminal-bench-2-1'),('terminal_bench_2','terminal-bench/terminal-bench-2')])
def test_explicit_terminal_version_and_digest(version,dataset):
    c=copy.deepcopy(controls()); c['evals']=[version]
    c['task_names'][version]=c['task_names']['terminal_bench_2_1'][:2]
    with patch('reference_config.evals.qwen38_goal12.controls', return_value=c):
        result=configure(RAW_MODEL_CONFIG,EvalLimitMode)
    assert len(result.tasks)==1
    task=result.tasks[0]
    assert task.task_name==version
    assert task.agentic_eval_config.dataset==dataset
    assert task.agentic_eval_config.dataset_ref==c['terminal_dataset_ref']
    assert task.agentic_eval_config.task_names_map[EvalLimitMode.CI_NIGHTLY]==c['task_names'][version]
    assert task.agentic_eval_config.agent_timeout_sec==c["agent_timeout_sec"]
    assert task.agentic_eval_config.llm_timeout_sec==c["llm_timeout_sec"]
    assert task.score.published_score is None

def test_frozen21_sources_and_population():
    c=controls(); manifest=Path(c['selection_manifest']); payload=manifest.read_bytes()
    assert hashlib.sha256(payload).hexdigest()==c['selection_manifest_sha256']
    t=json.loads(payload)['evals']['terminal_bench_2_1']
    p=json.loads(Path(t['population_manifest']).read_text())
    assert p['population']==len(p['tasks'])==89
    assert p['ref']==t['ref']==c['terminal_dataset_ref']
    assert t['dataset']=='terminal-bench/terminal-bench-2-1'
    assert set(t['selected_sources'])==set(t['task_names'])
    assert all(t['selected_sources'][n]==p['tasks'][n] for n in t['task_names'])

@pytest.mark.parametrize('agent_seconds,request_seconds', [(900,600),(1800,1200),(2700,1800)])
def test_calibration_effective_bounds(agent_seconds,request_seconds):
    c=copy.deepcopy(controls());c.update(evals=['terminal_bench_2_1'],agent_timeout_sec=agent_seconds,llm_timeout_sec=request_seconds,suite_timeout_sec=4500)
    with patch('reference_config.evals.qwen38_goal12.controls',return_value=c):
        result=configure(RAW_MODEL_CONFIG,EvalLimitMode)
    task=next(t for t in result.tasks if t.task_name=='terminal_bench_2_1');a=task.agentic_eval_config
    assert a.agent_timeout_sec==agent_seconds
    assert a.llm_timeout_sec==request_seconds
    assert a.harbor_timeout_sec==4500
    assert a.agent_kwargs['llm_kwargs']['timeout']==request_seconds
    assert a.agent_kwargs['llm_kwargs']['max_tokens']==16384
    assert a.agent_kwargs['model_info']['max_output_tokens']==16384
    assert a.n_concurrent_trials==c['max_concurrent']
    assert a.task_names_map[EvalLimitMode.CI_NIGHTLY]==c['task_names']['terminal_bench_2_1']


def test_gpqa32_effective_identity_limits_and_policy():
    c=controls()
    assert c['evals']==['r1_gpqa_diamond']
    assert c['sample_count']==32 and c['max_concurrent']==8
    assert c['seed']==1234 and c['output_tokens']==16384
    h=c['gpqa_handoff']; assert len(h['doc_ids'])==len(h['document_sha256'])==32
    assert len(set(h['doc_ids']))==32 and h['concurrency']==8
    manifest=Path(h['manifest']); assert hashlib.sha256(manifest.read_bytes()).hexdigest()==h['manifest_sha256']
    task=configure(EVAL_CONFIGS[MODEL],EvalLimitMode).tasks[0]
    assert task.task_name=='r1_gpqa_diamond' and task.agentic_eval_config is None
    assert task.doc_ids==h['doc_ids'] and task.max_concurrent==8
    assert task.wall_clock_timeout_seconds==10000
    assert task.model_kwargs['timeout']==3600
    assert task.gen_kwargs['max_gen_toks']==16384
    assert task.request_body['chat_template_kwargs']==h['generation']['chat_template_kwargs']
    assert task.request_body['chat_template_kwargs']==dict(enable_thinking=True,reasoning_effort='xhigh',preserve_thinking=True)
    for k,v in h['generation'].items():
        if k!='chat_template_kwargs': assert task.gen_kwargs[k]==v
    assert task.score.published_score is None and task.score.gpu_reference_score is None


def test_b8_pilot_serving_geometry():
    import yaml
    for path in ['workflows/model_specs/prod/llm.yaml','workflows/model_specs/dev/llm.yaml']:
        spec=next(s for s in yaml.safe_load(Path(path).read_text())['templates'] if s.get('weights')==['Qwen/Qwen3.8-27B'])
        device=spec['device_model_specs'][0]; args=device['vllm_args']; env=device['env_vars']
        assert args['max_num_seqs']==32
        assert device['max_concurrency']==args['max_num_seqs']
        assert device['max_context']==262144
        assert args['block_size']==args['mamba_block_size']==args['prefix_cache_retention_interval']==256
        assert env['QWEN38_SHARED_RECURRENT_SNAPSHOTS']=='1'
        assert env['QWEN38_CACHE_BUDGET_BYTES_PER_DEVICE']=='23661297664'
