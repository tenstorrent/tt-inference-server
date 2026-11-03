"""Conditional probe selects only the original frozen Cython task after external smoke admission."""
from pathlib import Path
import hashlib
from reference_config.evals.eval_config import EVAL_CONFIGS


def test_only_original_frozen_cython_probe_selected():
    tasks = EVAL_CONFIGS['google/gemma-4-31B-it'].tasks
    assert [task.task_name for task in tasks] == ['terminal_bench_2']
    cfg = tasks[0].agentic_eval_config
    assert cfg.n_tasks == 32 and cfg.n_concurrent_trials == 32 and cfg.n_attempts == 1
    assert cfg.agent_timeout_sec == 18000 and cfg.llm_timeout_sec == 3600
    assert cfg.agent_kwargs['model_info']['max_output_tokens'] == 65536
    assert cfg.agent_kwargs['llm_kwargs']['max_tokens'] == 65536
    assert not cfg.offline_task_prerequisites and not cfg.offline_apt_fixtures
    assert cfg.collect_preflight_evidence and cfg.allow_agent_endpoint
    assert cfg.preflight_min_free_bytes == 512 * 1024**3
    assert cfg.offline_apt_fixture_deadline is None
    assert cfg.environment_env == {} and cfg.agent_env == {}
    assert cfg.task_names_map == {} and tasks[0].limit_samples_map == {}


def test_original_smoke_file_hashes():
    root = Path('reference_config/evals/gemma4_preflight')
    expected = {
        'check-egress/environment/Dockerfile': '082dba0d5dd12e8a4dfb6b6d126de2d34eb75c789a72947d87da0721ea9b5b41',
        'check-egress/instruction.md': '4c82bd964fcba48514d8a55ad0f4bb2758b134e48aaf3c4a28e76b9f0b8055eb',
        'check-egress/task.toml': '2cc04c2d15a66bf51859d5d2a94d17d1946569b282102158deb6426ca6c6808f',
        'check-egress/tests/test.sh': '43bc3c980d8caf1f579106bac15aeaf6ca3e0b5d295b0c6eda4edba9b9f2b8ae',
        'write-result/environment/Dockerfile': '082dba0d5dd12e8a4dfb6b6d126de2d34eb75c789a72947d87da0721ea9b5b41',
        'write-result/instruction.md': '0462be33b75ebb32b36f7fe522bd9771ca7d3007ce4ad8a62e28987abb5a07d3',
        'write-result/task.toml': '4a8935776197c3ea7e84d4c0593b3943d152a3f2caa5eb995ff3b6814575abde',
        'write-result/tests/test.sh': '45620b88df74485ebd92cb22292f3dbb6481abf3a269df411d088e92d2b49ed8',
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_frozen_original_task_order():
    import json
    cohort = json.loads(Path('reference_config/evals/gemma4_cohorts.json').read_text())['terminal']
    assert cohort['tasks'][0] == 'build-cython-ext'
    assert cohort['tasks'] == ['build-cython-ext']
