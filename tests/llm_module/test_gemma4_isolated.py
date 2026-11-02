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
    from types import SimpleNamespace
    from llm_module.eval_command import _resolve_eval_samples
    tasks = EVAL_CONFIGS['google/gemma-4-31B-it'].tasks
    assert [t.task_name for t in tasks] == ['swe_bench_verified']
    task = tasks[0]
    assert task.limit_samples_map == {}
    cfg = task.agentic_eval_config
    assert cfg.n_tasks == 5 and cfg.n_concurrent_trials == 5
    assert cfg.exclude_task_names == ['astropy__astropy-14508', 'django__django-14034', 'pylint-dev__pylint-7080', 'scikit-learn__scikit-learn-25973', 'scikit-learn__scikit-learn-26323']
    assert cfg.llm_timeout_sec == 3600
    assert cfg.agent_kwargs["max_tokens"] == 65536
    assert cfg.agent_timeout_sec == 18000 and cfg.allow_agent_endpoint
    assert task.score.score_func_kwargs['result_keys'] == ['accuracy']


def test_remaining_partition_preserves_frozen_cohort():
    import json
    from fnmatch import fnmatch
    from llm_module.agentic.gemma4_isolated import COHORTS
    from llm_module.agentic.local_datasets import local_task_pattern
    cohort = json.loads(COHORTS.read_text())["swe"]["tasks"]
    cfg = EVAL_CONFIGS["google/gemma-4-31B-it"].tasks[0].agentic_eval_config
    assert len(cohort) == len(set(cohort)) == 10
    patterns = [local_task_pattern(cfg.dataset, p) for p in cfg.exclude_task_names]
    selected = [name for name in sorted(cohort) if not any(fnmatch(name, p) for p in patterns)][:cfg.n_tasks]
    assert len(selected) == len(set(selected)) == 5
    assert selected == ['django__django-11532', 'django__django-11734', 'django__django-12125', 'django__django-13417', 'django__django-13741']
