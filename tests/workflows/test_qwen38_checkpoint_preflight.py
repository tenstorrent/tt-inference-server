# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
import json
import subprocess
from types import SimpleNamespace
import pytest
from workflows.qwen38_checkpoint_preflight import ensure_checkpoint_preflight

CID = 'a' * 64
OWNER = 'run12-attempt1-job5'


def inspected(owner=OWNER):
    return {'Id': CID, 'Image': 'sha256:'+'b'*64, 'State': {'Running': True, 'StartedAt':'2026-10-30T18:00:00Z'},
            'Config': {'Env':['QWEN38_PRECISION_CONFIG=/home/container_app_user/qwen38_selected_precision.json'], 'Labels': {'com.tenstorrent.agentic-owner': owner},
                       'Entrypoint': ['bash', '-c', 'exec python run_vllm_api_server.py'], 'Cmd': []}}


def runner_for(items, inspect=None, receipt=None):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        if args[1] == 'ps':
            return SimpleNamespace(returncode=0, stdout='\n'.join(items), stderr='')
        if args[1] == 'inspect':
            return SimpleNamespace(returncode=0, stdout=json.dumps([inspect or inspected()]), stderr='')
        if args[1] == 'logs':
            from workflows.qwen38_selected_precision import selected_policy
            marker = '2026-10-30T18:01:00Z (EngineCore pid=5) AUTOPORT_ADAPTER /x/models/autoports/qwen_qwen3_8_27b/tt/generator_vllm.py:Qwen38ForCausalLM layers='+repr(list(range(64)))+' precision='+repr(selected_policy())
            return SimpleNamespace(returncode=0, stdout=marker, stderr='')
        if args[1] == 'exec':
            payload = json.loads(kwargs['input'])
            manifest = payload['manifest']
            measured = {'ok': True, 'precision_artifact':{'ok':True,'sha256':payload['precision_sha256']}, 'model':manifest['model'], 'expected_revision':manifest['revision'], 'weights':'/snapshot',
                        'files':[{'file':name,'matches':True} for name in manifest['files']], 'serving_processes':[{'pid':5,'start_epoch':0}], 'disk':{'free_bytes':1234}}
            return SimpleNamespace(returncode=0, stdout=json.dumps(receipt or measured), stderr='')
        raise AssertionError(args)
    return run, calls


def test_missing_owner_fails_without_docker(tmp_path, monkeypatch):
    monkeypatch.delenv('TT_AGENTIC_CONTAINER_OWNER', raising=False)
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    run, calls = runner_for([CID])
    with pytest.raises(RuntimeError, match='ownership'):
        ensure_checkpoint_preflight('Qwen/Qwen3.8-27B', tmp_path, runner=run)
    assert calls == []


def test_foreign_inspect_identity_never_executes(tmp_path, monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER', OWNER)
    run, calls = runner_for([CID], inspect=inspected('peer'))
    with pytest.raises(RuntimeError):
        ensure_checkpoint_preflight('Qwen/Qwen3.8-27B', tmp_path, runner=run)
    assert not any(a[1]=='exec' for a,k in calls)


def test_ambiguous_server_fails_without_exec(tmp_path, monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER', OWNER)
    run, calls = runner_for([CID,CID])
    with pytest.raises(RuntimeError):
        ensure_checkpoint_preflight('Qwen/Qwen3.8-27B', tmp_path, runner=run)
    assert not any(a[1]=='exec' for a,k in calls)


def test_verified_exact_container_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER', OWNER)
    run, calls = runner_for([CID])
    result = ensure_checkpoint_preflight('Qwen/Qwen3.8-27B', tmp_path, runner=run)
    assert result['container_id'] == CID
    assert result['owner'] == OWNER
    assert result['image_id'] == 'sha256:'+'b'*64
    execute = [a for a,k in calls if a[1]=='exec']
    assert len(execute)==1 and execute[0][3] == CID
    assert json.loads((tmp_path/'checkpoint-preflight.json').read_text())['ok']


def test_hash_failure_preserves_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER', OWNER)
    run, calls = runner_for([CID], receipt={'ok':False,'weights':'/snapshot','files':[{'matches':False}]})
    with pytest.raises(RuntimeError):
        ensure_checkpoint_preflight('Qwen/Qwen3.8-27B', tmp_path, runner=run)
    assert json.loads((tmp_path/'checkpoint-preflight.json').read_text())['ok'] is False


@pytest.mark.parametrize("hf_identity", ["path", "id", "invalid"])
def test_container_probe_binds_process_path_and_disk(tmp_path, monkeypatch, hf_identity):
    import hashlib
    import io
    import pathlib
    import sys
    from workflows.qwen38_checkpoint_preflight import _CONTAINER_PROBE
    root = tmp_path/'proc'; proc = root/'123'; proc.mkdir(parents=True)
    weights = tmp_path/'weights'; weights.mkdir()
    config = b'{"text_config":{"num_hidden_layers":64}}'
    (weights/'config.json').write_bytes(config)
    (proc/'cmdline').write_bytes(b'python\0-m\0vllm.entrypoints.openai.api_server\0')
    hf = {'path':str(weights), 'id':'Qwen/Qwen3.8-27B', 'invalid':'other/model'}[hf_identity]
    (proc/'environ').write_bytes(b'QWEN38_CHECKPOINT='+str(weights).encode()+b'\0HF_MODEL='+hf.encode()+b'\0SECRET=must_not_emit\0')
    (proc/'stat').write_text('123 (vllm) '+' '.join(['S']+['0']*18+['999']))
    manifest = {'model':'test','revision':'a'*40,'files':{'config.json':{'sha256':hashlib.sha256(config).hexdigest()}}}
    source = pathlib.Path('workflows/qwen38_checkpoint_identity.py').read_text()
    stdin = io.StringIO(json.dumps({'manifest':manifest,'verifier':source}))
    stdout = io.StringIO()
    real_path = pathlib.Path
    monkeypatch.setattr(pathlib, 'Path', lambda path: root if str(path)=='/proc' else real_path(path))
    monkeypatch.setattr(sys, 'stdin', stdin); monkeypatch.setattr(sys, 'stdout', stdout)
    if hf_identity == 'invalid':
        with pytest.raises(RuntimeError, match='disagree'):
            exec(compile(_CONTAINER_PROBE, '<probe-test>', 'exec'), {})
        return
    exec(compile(_CONTAINER_PROBE, '<probe-test>', 'exec'), {})
    result = json.loads(stdout.getvalue())
    assert result['ok'] and result['checkpoint_layer_count']==64
    assert result['weights']==str(weights)
    assert result['disk']['free_bytes'] > 0
    assert len(result['serving_processes'])==1
    process=result['serving_processes'][0]
    assert process['pid']==123 and process['start_ticks']=='999' and process['weights']==str(weights)
    assert process['start_epoch']>0
    assert 'must_not_emit' not in stdout.getvalue()


def test_container_probe_refuses_missing_checkpoint_path(tmp_path, monkeypatch):
    import io
    import pathlib
    import sys
    from workflows.qwen38_checkpoint_preflight import _CONTAINER_PROBE
    root=tmp_path/'proc'; root.mkdir()
    source = pathlib.Path('workflows/qwen38_checkpoint_identity.py').read_text()
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({'manifest':{},'verifier':source})))
    real_path = pathlib.Path
    monkeypatch.setattr(pathlib, 'Path', lambda path: root if str(path)=='/proc' else real_path(path))
    with pytest.raises(RuntimeError, match='unique serving checkpoint'):
        exec(compile(_CONTAINER_PROBE, '<probe-test>', 'exec'), {})


def test_agentic_materialization_never_precedes_guard(tmp_path, monkeypatch):
    from llm_module.agentic import qwen38_isolation
    import workflows.qwen38_checkpoint_preflight as guard
    calls = []
    def fail(*args, **kwargs):
        calls.append('guard')
        raise RuntimeError('checkpoint rejected')
    monkeypatch.setattr(guard, 'ensure_checkpoint_preflight', fail)
    monkeypatch.setattr(qwen38_isolation.subprocess, 'run', lambda *a,**k: calls.append('materialize'))
    cfg=SimpleNamespace(model_name='openai/Qwen/Qwen3.8-27B', jobs_dir=tmp_path, dataset='terminal-bench/terminal-bench-2-1')
    with pytest.raises(RuntimeError, match='checkpoint rejected'):
        qwen38_isolation.prepare(cfg)
    assert calls == ['guard']


def test_non_qwen_model_does_not_inspect_containers(tmp_path):
    run, calls = runner_for([CID])
    assert ensure_checkpoint_preflight('other/model', tmp_path, runner=run) is None
    assert calls == []
