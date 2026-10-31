# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import pytest
from workflows.qwen38_run_identity import current_run_owner
from workflows.qwen38_checkpoint_preflight import ensure_checkpoint_preflight
from tests.workflows.test_qwen38_checkpoint_preflight import runner_for, inspected, CID
MODEL='Qwen/Qwen3.8-27B'

@pytest.fixture
def native_ci(monkeypatch):
    monkeypatch.delenv('TT_AGENTIC_CONTAINER_OWNER', raising=False)
    for k,v in {'GITHUB_ACTIONS':'true','GITHUB_RUN_ID':'41130000000','GITHUB_RUN_ATTEMPT':'1','GITHUB_JOB':'run-tests','GITHUB_REPOSITORY':'tenstorrent/tt-agentic-bringup-qb2'}.items():
        monkeypatch.setenv(k,v)
    return current_run_owner(MODEL)

def test_default_owner_is_stable_scoped_and_legacy_preserved(native_ci,monkeypatch):
    assert native_ci.startswith('goal12-41130000000-1-')
    assert current_run_owner('openai/'+MODEL)==native_ci
    assert current_run_owner('another/model')==''
    monkeypatch.setenv('GITHUB_RUN_ATTEMPT','2');assert current_run_owner(MODEL)!=native_ci
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER','explicit-owned');assert current_run_owner(MODEL)=='explicit-owned'

@pytest.mark.parametrize('missing',['GITHUB_ACTIONS','GITHUB_RUN_ID','GITHUB_RUN_ATTEMPT','GITHUB_JOB','GITHUB_REPOSITORY'])
def test_incomplete_identity_never_invents_owner(native_ci,monkeypatch,missing):
    monkeypatch.delenv(missing);assert current_run_owner(MODEL)==''

def test_native_owner_reaches_exact_checkpoint_lookup(native_ci,tmp_path):
    run,calls=runner_for([CID],inspect=inspected(native_ci))
    receipt=ensure_checkpoint_preflight(MODEL,tmp_path,runner=run)
    assert receipt['ok'] and receipt['owner']==native_ci
    assert calls[0][0]==['docker','ps','--no-trunc','-q','--filter',f'label=com.tenstorrent.agentic-owner={native_ci}']

def test_native_owner_labels_server_and_isolated_tasks(native_ci,tmp_path,monkeypatch):
    from workflows.model_spec import load_templates_from_yaml
    from workflows.utils import get_repo_root_path
    from workflows.runtime_config import RuntimeConfig
    from workflows.run_docker_server import generate_docker_run_command
    from llm_module.agentic.harbor import HarborRunConfig,_write_harbor_config
    templates=load_templates_from_yaml(get_repo_root_path()/'workflows/model_specs/dev/llm.yaml')
    spec=next(t for t in templates if t.weights==[MODEL]).expand_to_specs()[0]
    runtime=RuntimeConfig(model=MODEL,workflow='agentic',device='p300x2',service_port='8000')
    monkeypatch.setattr('workflows.qwen38_selected_precision.policy_mount',lambda *a,**k:([],{}))
    command,_=generate_docker_run_command(spec,runtime)
    assert f'com.tenstorrent.agentic-owner={native_ci}' in command
    cfg=HarborRunConfig(task_name='smoke',dataset='terminal-bench/terminal-bench-2-1',agent='terminus-2',model_name='openai/'+MODEL,jobs_dir=tmp_path,api_base='http://127.0.0.1:8000/v1',n_concurrent_trials=32,n_attempts=1,environment_type='docker',agent_kwargs={},n_tasks=2,override_cpus=None,override_memory_mb=None,timeout_multiplier=None,agent_timeout_sec=21600,agent_import_path='llm_module.agentic.qwen38_isolated_agent:IsolatedTerminus2')
    config=json.loads(_write_harbor_config(cfg).read_text());overlay=json.loads(Path(config['environment']['extra_docker_compose'][0]).read_text())
    assert set(overlay['services'])=={'main','harbor-docker-egress-control-sidecar'}
    assert all(x['labels']['com.tenstorrent.agentic-owner']==native_ci for x in overlay['services'].values())
