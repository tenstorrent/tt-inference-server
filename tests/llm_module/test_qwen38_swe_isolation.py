# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest

from llm_module.agentic.qwen38_isolation import frozen_swe_tasks, materialize, isolate_config


def manifest():
    return json.loads(Path('reference_config/evals/qwen38_swe_tasks.json').read_text())


def test_frozen_legacy_identities_and_invalid_selection():
    m = manifest()
    names = [t['name'] for t in m['tasks']]
    tasks, provenance = frozen_swe_tasks(names)
    assert len(tasks) == 10
    assert provenance['registry_git_blob'] == '0861ad066f7213927b380aa24af42b9a874441a1'
    assert provenance['registry_sha256'] == 'da1446bce05eabbd72a25eb9eef5a2f5db94645ce88c28e2497581433b3d2e60'
    for name, task in zip(names, tasks):
        assert task.name is None and task.ref is None
        assert task.get_task_id().get_name() == name
        assert task.git_url == 'https://github.com/laude-institute/harbor-datasets.git'
        assert task.git_commit_id == '86723674f04e4209ac479d0fb75d9d9f44b4377e'
        assert str(task.path) == 'datasets/swebench-verified/' + name
    for invalid in [[], [names[0], names[0]], ['unregistered-task']]:
        with pytest.raises(ValueError, match='unique subset'):
            frozen_swe_tasks(invalid)


@pytest.mark.parametrize('wrong_commit', [False, True])
def test_materialize_legacy_preserves_native_bytes(tmp_path, monkeypatch, wrong_commit):
    import toml
    from harbor.models.job.config import DatasetConfig
    from harbor.tasks.client import TaskClient
    names = [t['name'] for t in manifest()['tasks']][:2]
    c = dict(dataset='swebench-verified', dataset_ref=None, task_names=names,
             isolate_agent_network=True, scored=False, intent='isolated_smoke', evals=['swe_bench_verified'])
    monkeypatch.setattr('reference_config.evals.qwen38_goal12.controls', lambda: c)
    registry = AsyncMock(side_effect=AssertionError('Mutable registry lookup prohibited'))
    monkeypatch.setattr(DatasetConfig, 'get_task_configs', registry)
    original = {'version': '1.0', 'agent': {'timeout_sec': 21600},
                'verifier': {'timeout_sec': 1200}, 'environment': {'cpus': 2, 'memory_mb': 4096}}
    source = tmp_path / 'source'; source.mkdir()
    (source/'task.toml').write_text(toml.dumps(original))
    (source/'instruction.md').write_bytes(b'native instruction\n')
    (source/'tests').mkdir(); (source/'tests/test.sh').write_bytes(b'#!/bin/sh\nexit 0\n')
    async def download(self, ids):
        assert ids[0].git_commit_id == manifest()['tasks'][0]['git_commit_id']
        return SimpleNamespace(results=[SimpleNamespace(path=source,content_hash=None,
            resolved_git_commit_id='wrong' if wrong_commit else ids[0].git_commit_id)])
    monkeypatch.setattr(TaskClient,'download_tasks',download)
    output = tmp_path/'prepared'
    if wrong_commit:
        with pytest.raises(ValueError,match='Resolved Git commit mismatch'):
            asyncio.run(materialize(output))
        assert not (output/names[0]).exists()
        return
    asyncio.run(materialize(output))
    receipt=json.loads((tmp_path/'isolation-dataset.json').read_text())
    assert [t['name'] for t in receipt['tasks']] == names
    for name, entry in zip(names, receipt['tasks']):
        assert toml.loads((output/name/'task.toml').read_text()) == isolate_config(original)
        assert entry['changed_files'] == ['task.toml']
        assert entry['before']['instruction.md'] == entry['after']['instruction.md']
        assert entry['before']['tests/test.sh'] == entry['after']['tests/test.sh']
        assert entry['source']['git_commit_id'] == entry['resolved_git_commit_id']
    registry.assert_not_awaited()


@pytest.mark.parametrize('codes, passes', [([0,1,124], True),([0,0,1],False),([0,127,1],False),([1],False)])
def test_mini_probe_precedes_unchanged_parent(tmp_path, monkeypatch, codes, passes):
    from harbor.agents.installed.mini_swe_agent import MiniSweAgent
    from llm_module.agentic.qwen38_isolated_agent import IsolatedMiniSweAgent
    parent = AsyncMock()
    monkeypatch.setattr(MiniSweAgent,'run',parent)
    agent=object.__new__(IsolatedMiniSweAgent);agent.logs_dir=tmp_path
    environment=SimpleNamespace(exec=AsyncMock(side_effect=[SimpleNamespace(return_code=c) for c in codes]))
    context=object()
    if passes:
        asyncio.run(agent.run('native instruction',environment,context))
        parent.assert_awaited_once_with('native instruction',environment,context)
        assert json.loads((tmp_path/'network-isolation.json').read_text())['effective_policy']=='inference-host allowlist'
    else:
        with pytest.raises(RuntimeError):asyncio.run(agent.run('native instruction',environment,context))
        parent.assert_not_awaited()


def test_mini_config_keeps_timeout_endpoint_and_owned_sidecar(tmp_path, monkeypatch):
    from llm_module.agentic.harbor import HarborRunConfig, _write_harbor_config, _get_agent_kwargs
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER','owned-test-123')
    monkeypatch.setattr('llm_module.agentic.harbor._docker_host_address',lambda:'172.17.0.1')
    cfg=HarborRunConfig(task_name='swe',dataset='swebench-verified',agent='mini-swe-agent',
        model_name='openai/Qwen/Qwen3.8-27B',jobs_dir=tmp_path,api_base='http://127.0.0.1:8000/v1',
        n_concurrent_trials=32,n_attempts=1,environment_type='docker',
        agent_kwargs={'max_tokens':32768},n_tasks=2,override_cpus=None,override_memory_mb=None,
        timeout_multiplier=None,agent_timeout_sec=21600,llm_timeout_sec=3600,allow_agent_endpoint=True,
        agent_import_path='llm_module.agentic.qwen38_isolated_agent:IsolatedMiniSweAgent')
    before=copy.deepcopy(cfg.agent_kwargs)
    result=json.loads(_write_harbor_config(cfg).read_text())
    assert cfg.agent_kwargs==before
    assert result['agents'][0]['extra_allowed_hosts']==['172.17.0.1']
    kw=result['agents'][0]['kwargs']
    assert kw['max_tokens']==32768
    assert kw['config']['model']['model_kwargs']['timeout']==3600
    assert kw['config']['model']['model_kwargs']['drop_params'] is True
    assert kw['config']['model']['cost_tracking']=='ignore_errors'
    assert result['agents'][0]['override_timeout_sec']==21600
    overlay=json.loads(Path(result['environment']['extra_docker_compose'][0]).read_text())
    assert set(overlay['services'])=={'main','harbor-docker-egress-control-sidecar'}
    assert all(s['labels']['com.tenstorrent.agentic-owner']=='owned-test-123' for s in overlay['services'].values())
    override=replace(cfg,agent_kwargs={'config':{'model':{'model_kwargs':{'timeout':12000}}}})
    assert _get_agent_kwargs(override)['config']['model']['model_kwargs']['timeout']==12000
    from harbor.models.job.config import JobConfig
    JobConfig.model_validate(result)


def test_package_materialization_preserves_digest_and_task_bytes(tmp_path, monkeypatch):
    from harbor.models.job.config import DatasetConfig
    from harbor.models.trial.config import TaskConfig
    from harbor.tasks.client import TaskClient
    digest='sha256:'+'a'*64
    c=dict(dataset='terminal-bench/terminal-bench-2-1',dataset_ref=digest,
           task_names=['terminal-bench/test-task'],isolate_agent_network=True,scored=False, intent='isolated_smoke',evals=['terminal_bench_2_1'])
    monkeypatch.setattr('reference_config.evals.qwen38_goal12.controls',lambda:c)
    task=TaskConfig(name=c['task_names'][0],ref=digest)
    monkeypatch.setattr(DatasetConfig,'get_task_configs',AsyncMock(return_value=[task]))
    source=tmp_path/'source';source.mkdir()
    (source/'task.toml').write_text('version = "1.0"\n[agent]\ntimeout_sec = 21600\n')
    (source/'instruction.md').write_text('original native task\n')
    monkeypatch.setattr(TaskClient,'download_tasks',AsyncMock(return_value=SimpleNamespace(results=[
        SimpleNamespace(path=source,content_hash='a'*64,resolved_git_commit_id=None)])))
    asyncio.run(materialize(tmp_path/'prepared'))
    receipt=json.loads((tmp_path/'isolation-dataset.json').read_text())
    assert receipt['tasks'][0]['name']==c['task_names'][0]
    assert receipt['tasks'][0]['ref']==digest
    assert receipt['tasks'][0]['download_hash']=='a'*64
    assert (tmp_path/'prepared/test-task/instruction.md').read_bytes()==(source/'instruction.md').read_bytes()


def test_cost_sample_requires_receipt_before_download(tmp_path, monkeypatch):
    from harbor.tasks.client import TaskClient
    names=[t['name'] for t in manifest()['tasks']][:2]
    c=dict(dataset='swebench-verified',dataset_ref=None,task_names=names,
        isolate_agent_network=True,scored=False,intent='cost_sample',evals=['swe_bench_verified'])
    monkeypatch.setattr('reference_config.evals.qwen38_goal12.controls',lambda:c)
    monkeypatch.delenv('TT_QWEN38_PREFLIGHT_JSON',raising=False)
    download=AsyncMock(side_effect=AssertionError('Must fail before acquisition'))
    monkeypatch.setattr(TaskClient,'download_tasks',download)
    with pytest.raises(ValueError,match='Missing TT_QWEN38_PREFLIGHT_JSON'):
        asyncio.run(materialize(tmp_path/'prepared'))
    download.assert_not_awaited()
    assert not (tmp_path/'prepared').exists()
