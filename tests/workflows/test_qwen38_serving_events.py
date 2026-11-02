# SPDX-License-Identifier: Apache-2.0
import ast
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from workflows.qwen38_serving_events import events_mount, ENV_KEY, CONTAINER_FILE, CONTAINER_DIR


def model():
    from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
    from workflows.utils import get_repo_root_path
    specs = [s for t in load_templates_from_yaml(get_repo_root_path() / 'workflows/model_specs/dev/llm.yaml')
             for s in t.expand_to_specs() if s.hf_model_repo == 'Qwen/Qwen3.8-27B']
    return resolve_model_spec(specs, model='Qwen/Qwen3.8-27B', device='p300x2', impl=None, catalog_name='test')


def test_actual_registry_runtime_consumer_and_docker_artifact_transport(tmp_path, monkeypatch):
    import workflows.run_docker_server as server
    from workflows.runtime_config import RuntimeConfig
    from workflows.utils import get_repo_root_path
    m = model()
    source_root = get_repo_root_path()
    tree = ast.parse((source_root / 'vllm-tt-metal/src/run_vllm_api_server.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['_model_spec_env_vars', 'set_runtime_env_vars']]
    ns = {'os': os, 'logger': logging.getLogger(__name__), 'QUETZAL_IMPL_ID': 'quetzal', 'QUETZAL_EXTERNAL_PACKAGE_ENV': set()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-wrapper-env>', 'exec'), ns)
    monkeypatch.setenv(ENV_KEY, '/ambient/uncollected')
    ns['set_runtime_env_vars'](json.loads(json.dumps({'env_vars': m.env_vars, 'impl': {'impl_id': 'qwen38_autoport'}})))
    assert os.environ[ENV_KEY] == CONTAINER_FILE
    monkeypatch.setattr(server, 'get_repo_root_path', lambda: tmp_path)
    # CACHE_ROOT must not redirect this artifact into a persistent cache volume.
    monkeypatch.setenv('CACHE_ROOT', str(tmp_path / 'uncollected-cache'))
    runtime = RuntimeConfig(model='Qwen3.8-27B', workflow='server', device='p300x2', service_port='8000')
    command, name = server.generate_docker_run_command(m, runtime, json_fpath=tmp_path / 'runtime.json')
    host = tmp_path / 'workflow_logs' / 'serving_events' / name
    assert f'type=bind,src={host},dst={CONTAINER_DIR}' in command
    assert ENV_KEY + '=' + CONTAINER_FILE in command
    assert host.stat().st_mode & 0o777 == 0o777
    receipt = json.loads((host / 'transport.json').read_text())
    assert receipt['artifact_path'] == f'serving_events/{name}/events.jsonl'
    assert receipt['runtime_verified'] is False
    _, second = server.generate_docker_run_command(m, runtime)
    assert second != name
    assert (tmp_path / 'workflow_logs' / 'serving_events' / second).is_dir()


def test_existing_directory_is_not_reused_or_modified(tmp_path):
    existing = tmp_path / 'serving_events' / 'tt-inference-server-existing'
    existing.mkdir(parents=True, mode=0o700)
    (existing / 'events.jsonl').write_text('peer evidence')
    with pytest.raises(FileExistsError):
        events_mount(model(), tmp_path, existing.name)
    assert (existing / 'events.jsonl').read_text() == 'peer evidence'
    assert existing.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize('wrong', [None, '/tmp/not-collected'])
def test_runtime_path_mismatch_fails_before_directory_creation(tmp_path, wrong):
    m = model()
    m.env_vars[ENV_KEY] = wrong
    with pytest.raises(RuntimeError, match='collected event path'):
        events_mount(m, tmp_path, 'tt-inference-server-test')
    assert list(tmp_path.iterdir()) == []


def test_scope_and_name_guard(tmp_path):
    assert events_mount(NS(hf_model_repo='other'), tmp_path, 'irrelevant') == ([], {})
    with pytest.raises(ValueError):
        events_mount(model(), tmp_path, '../peer')
    assert list(tmp_path.iterdir()) == []
