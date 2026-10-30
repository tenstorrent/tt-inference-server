# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
import copy
from llm_module.agentic.qwen38_isolation import isolate_config


def test_isolation_changes_only_agent_network_policy():
    original = {'agent': {'timeout_sec': 21600, 'network_mode': 'public'},
                'verifier': {'timeout_sec': 1200}, 'environment': {'cpus': 16},
                'steps': [{'agent': {'timeout_sec': 30}, 'instruction': 'keep this'}]}
    source = copy.deepcopy(original)
    result = isolate_config(original)
    assert original == source
    assert result['agent'] == {'timeout_sec': 21600, 'network_mode': 'no-network', 'allowed_hosts': []}
    assert result['steps'][0]['agent']['network_mode'] == 'no-network'
    assert result['verifier'] == source['verifier']
    assert result['environment'] == source['environment']
    assert result['steps'][0]['instruction'] == 'keep this'


def test_network_receipt_does_not_accept_missing_probe_tools(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from llm_module.agentic.qwen38_isolated_agent import IsolatedTerminus2
    from unittest.mock import AsyncMock
    import pytest
    agent = object.__new__(IsolatedTerminus2)
    agent.logs_dir = tmp_path
    env = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=1)))
    with pytest.raises(RuntimeError, match='requires timeout and bash'):
        asyncio.run(agent.run('unchanged', env, None))
    assert env.exec.await_count == 1
    assert not (tmp_path / 'network-isolation.json').exists()


def test_network_receipt_rejects_open_egress(tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace
    from llm_module.agentic.qwen38_isolated_agent import IsolatedTerminus2
    from unittest.mock import AsyncMock
    import pytest
    agent = object.__new__(IsolatedTerminus2)
    agent.logs_dir = tmp_path
    env = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0)))
    with pytest.raises(RuntimeError, match='isolation probe failed'):
        asyncio.run(agent.run('unchanged', env, None))
    assert env.exec.await_count == 3
    assert json.loads((tmp_path / 'network-isolation.json').read_text())['ok'] is False


def test_owned_overlay_labels_main_and_policy_sidecar(tmp_path, monkeypatch):
    import json
    from llm_module.agentic.harbor import HarborRunConfig, _write_harbor_config
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER', 'run123-attempt1-job5')
    cfg = HarborRunConfig(task_name='smoke', dataset='terminal-bench/terminal-bench-2-1',
        agent='terminus-2', model_name='openai/Qwen/Qwen3.8-27B', jobs_dir=tmp_path,
        api_base='http://127.0.0.1:8000/v1', n_concurrent_trials=32, n_attempts=1,
        environment_type='docker', agent_kwargs={}, n_tasks=2, override_cpus=None,
        override_memory_mb=None, timeout_multiplier=None, agent_timeout_sec=21600,
        agent_import_path='llm_module.agentic.qwen38_isolated_agent:IsolatedTerminus2')
    result = json.loads(_write_harbor_config(cfg).read_text())
    overlay = json.loads(__import__('pathlib').Path(result['environment']['extra_docker_compose'][0]).read_text())
    assert set(overlay['services']) == {'main', 'harbor-docker-egress-control-sidecar'}
    assert all(v['labels']['com.tenstorrent.agentic-owner'] == 'run123-attempt1-job5' for v in overlay['services'].values())
    assert not any('extra_hosts' in v for v in overlay['services'].values())
