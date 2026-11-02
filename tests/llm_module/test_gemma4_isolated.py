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
    assert cfg.n_tasks == 2 and cfg.n_concurrent_trials == 2
    assert cfg.exclude_task_names == []
    assert cfg.llm_timeout_sec == 3600
    assert cfg.agent_kwargs["max_tokens"] == 65536
    assert cfg.agent_timeout_sec == 18000 and cfg.allow_agent_endpoint
    assert task.score.score_func_kwargs['result_keys'] == ['accuracy']


def test_held_out_pair_preserves_frozen_cohort():
    import json
    from fnmatch import fnmatch
    from llm_module.agentic.gemma4_isolated import COHORTS
    from llm_module.agentic.local_datasets import local_task_pattern
    cohort = json.loads(COHORTS.read_text())["swe"]["tasks"]
    cfg = EVAL_CONFIGS["google/gemma-4-31B-it"].tasks[0].agentic_eval_config
    assert len(cohort) == len(set(cohort)) == 2
    patterns = [local_task_pattern(cfg.dataset, p) for p in cfg.exclude_task_names]
    selected = [name for name in sorted(cohort) if not any(fnmatch(name, p) for p in patterns)][:cfg.n_tasks]
    assert len(selected) == len(set(selected)) == 2
    assert selected == ['django__django-14608', 'sphinx-doc__sphinx-10466']


def test_pair_seed_survives_release_defaults():
    import copy
    from llm_module.agentic.harbor import _apply_mini_swe_agent_defaults
    cfg = EVAL_CONFIGS["google/gemma-4-31B-it"].tasks[0].agentic_eval_config
    kwargs = copy.deepcopy(cfg.agent_kwargs)
    _apply_mini_swe_agent_defaults(kwargs, cfg)
    model = kwargs["config"]["model"]["model_kwargs"]
    assert model["seed"] == 9472
    assert model["timeout"] == 3600
    assert model["max_tokens"] == 65536
    assert model["temperature"] == 1.0 and model["top_p"] == 0.95
