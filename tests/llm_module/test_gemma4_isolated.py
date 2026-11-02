try:
    import tomllib
except ImportError:
    import tomli as tomllib
from llm_module.agentic.gemma4_isolated import isolate_manifest
from reference_config.evals.eval_config import EVAL_CONFIGS


def test_policy_preserves_task_and_verifier():
    text = '''schema_version = "1.4"
[task]
name = "test"
[agent]
timeout_sec = 18000
network_mode = "allowlist"
allowed_hosts = [
  "github.com",
  "pypi.org",
]
[verifier]
timeout_sec = 900
network_mode = "public"
'''
    before = tomllib.loads(text)
    after = tomllib.loads(isolate_manifest(text))
    assert after['task'] == before['task']
    assert after['verifier'] == before['verifier']
    assert after['agent'] == {'timeout_sec': 18000, 'network_mode': 'no-network'}
    assert isolate_manifest(isolate_manifest(text)).count('network_mode = "no-network"') == 1


def test_release_cohort_limits():
    tasks = {t.task_name: t for t in EVAL_CONFIGS['google/gemma-4-31B-it'].tasks}
    gpqa = tasks['r1_gpqa_diamond']
    assert gpqa.doc_ids == list(range(10))
    assert gpqa.model_kwargs['timeout'] == 14400
    assert gpqa.gen_kwargs['max_gen_toks'] == 126976
    for name in ['terminal_bench_2', 'swe_bench_verified']:
        cfg = tasks[name].agentic_eval_config
        assert cfg.n_tasks == 10 and cfg.n_concurrent_trials == 32
        assert cfg.agent_timeout_sec == 18000 and cfg.allow_agent_endpoint
