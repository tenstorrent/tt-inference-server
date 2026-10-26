"""B16 serving capacity is independent of the frozen32 client workload."""
from reference_config.evals.eval_config import EVAL_CONFIGS
from reference_config.evals.qwen38_goal12 import MODEL, controls
from workflows.model_spec import MODEL_SPECS


def test_b16_server_retains_32_clients_and_protocol():
    spec=next(s for s in MODEL_SPECS.values() if s.hf_model_repo==MODEL and s.device_type.name=='P300X2' and s.device_model_spec.default_impl)
    d=spec.device_model_spec
    assert d.max_concurrency==16
    assert d.vllm_args['max_num_seqs']==16
    assert d.max_context==262144
    assert d.vllm_args['block_size']==d.vllm_args['mamba_block_size']==d.vllm_args['prefix_cache_retention_interval']==256
    assert str(d.env_vars['QWEN38_SHARED_RECURRENT_SNAPSHOTS'])=='1'
    assert str(d.env_vars['QWEN38_CACHE_BUDGET_BYTES_PER_DEVICE'])=='23661297664'
    c=controls();assert c['max_concurrent']==c['sample_count']==32
    assert 'terminal_bench_2_1' in c['evals']
    task=next(t for t in EVAL_CONFIGS[MODEL].tasks if t.task_name=='terminal_bench_2_1')
    assert task.agentic_eval_config.n_concurrent_trials==32
    assert task.agentic_eval_config.agent_timeout_sec==900
    assert task.agentic_eval_config.llm_timeout_sec==600
    assert c['output_tokens']==16384
    assert len(c['task_names']['terminal_bench_2_1'])==32
