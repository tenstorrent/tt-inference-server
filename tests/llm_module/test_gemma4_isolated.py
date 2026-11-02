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
    assert [t.task_name for t in tasks] == ['r1_gpqa_diamond']
    task = tasks[0]
    assert task.limit_samples_map == {}
    import json
    assert task.doc_ids == [0]
    assert task.model_kwargs['timeout'] == 14400
    assert task.gen_kwargs['max_gen_toks'] == 126976
    assert task.score.score_func_kwargs['result_keys'] == ['exact_match,none']
    assert task.thinking and task.use_chat_api
    for mode in ['ci-nightly', 'ci-long', None]:
        runtime = SimpleNamespace(limit_samples_mode=mode, eval_samples=None)
        assert json.loads(_resolve_eval_samples(task, runtime)) == {'r1_gpqa_diamond': [0]}
