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
    assert cfg.n_tasks == 32 and cfg.n_concurrent_trials == 29
    assert cfg.exclude_task_names == []
    assert cfg.llm_timeout_sec == 3600
    assert cfg.agent_kwargs["max_tokens"] == 65536
    assert cfg.agent_timeout_sec == 18000 and cfg.allow_agent_endpoint
    assert task.score.score_func_kwargs['result_keys'] == ['accuracy']


# main's CI_NIGHTLY swe_bench_verified subset for the gemma/qwen family. The
# goal-12 cohort is built on top of it so the nightly numbers stay comparable.
MAIN_NIGHTLY_SWE5 = [
    'astropy__astropy-14096',
    'django__django-11299',
    'matplotlib__matplotlib-25332',
    'scikit-learn__scikit-learn-14629',
    'sympy__sympy-13551',
]


def test_cohort_is_repo_stratified_and_contains_main_subset():
    import json
    from collections import Counter
    from llm_module.agentic.gemma4_isolated import COHORTS
    cohort = json.loads(COHORTS.read_text())["swe"]["tasks"]
    assert len(cohort) == len(set(cohort)) == 32
    assert cohort == sorted(cohort), "cohort is stored in sorted order"
    # Built like main's five-task subset: distinct repositories, not a prefix of
    # one. All twelve repos in the pinned 500 are represented, and no repo takes
    # more than a round-robin share.
    counts = Counter(name.split('__')[0] for name in cohort)
    assert len(counts) == 12, counts
    assert max(counts.values()) <= 4, counts
    assert set(MAIN_NIGHTLY_SWE5) <= set(cohort)


def test_remaining_partition_preserves_frozen_cohort():
    import json
    from fnmatch import fnmatch
    from llm_module.agentic.gemma4_isolated import COHORTS
    from llm_module.agentic.local_datasets import local_task_pattern
    cohort = json.loads(COHORTS.read_text())["swe"]["tasks"]
    cfg = EVAL_CONFIGS["google/gemma-4-31B-it"].tasks[0].agentic_eval_config
    assert len(cohort) == len(set(cohort)) == 32
    patterns = [local_task_pattern(cfg.dataset, p) for p in cfg.exclude_task_names]
    selected = [name for name in sorted(cohort) if not any(fnmatch(name, p) for p in patterns)][:cfg.n_tasks]
    # Nothing is excluded and nothing is truncated: the run is the whole cohort.
    assert selected == sorted(cohort)
    assert len(selected) == len(set(selected)) == 32
    # Concurrency is below the cohort size on purpose; see eval_config.py.
    assert cfg.n_concurrent_trials == 29 < cfg.n_tasks
