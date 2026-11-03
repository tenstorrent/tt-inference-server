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


def test_unscored_recovery_cohort_limits():
    from types import SimpleNamespace
    from llm_module.eval_command import _resolve_eval_samples
    tasks = EVAL_CONFIGS['google/gemma-4-31B-it'].tasks
    assert [t.task_name for t in tasks] == ['gemma4_preflight_unscored']
    task = tasks[0]
    assert task.limit_samples_map == {}
    cfg = task.agentic_eval_config
    assert cfg.n_tasks == 2 and cfg.n_concurrent_trials == 2
    assert cfg.agent_timeout_sec == 18000 and cfg.allow_agent_endpoint
    assert task.score.score_func_kwargs['result_keys'] == ['accuracy']


def test_recovery_prerequisites_do_not_reduce_release_limits():
    cfg=EVAL_CONFIGS['google/gemma-4-31B-it'].tasks[0].agentic_eval_config
    assert cfg.dataset=='gemma4-preflight'
    assert cfg.offline_task_prerequisites and cfg.offline_apt_fixtures
    assert cfg.offline_apt_fixture_inputs is None
    assert cfg.offline_apt_fixture_deadline=='2026-11-03T06:00:00Z'
    assert cfg.agent_timeout_sec==18000 and cfg.llm_timeout_sec==3600
    assert cfg.agent_kwargs['llm_kwargs']['timeout']==3600
    assert cfg.agent_kwargs['llm_kwargs']['max_tokens']==65536
    assert cfg.agent_kwargs['model_info']['max_output_tokens']==65536
    assert cfg.task_names==[] and cfg.exclude_task_names==[] and cfg.task_names_map=={}
