# SPDX-License-Identifier: Apache-2.0
"""Execute the complete container probe against tiny files and synthetic /proc."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pytest
from workflows.qwen38_checkpoint_preflight import _CONTAINER_PROBE, ensure_checkpoint_preflight
from workflows.qwen38_selected_precision import selected_policy
from tests.workflows.test_qwen38_checkpoint_preflight import runner_for, CID, OWNER


def fixture_probe(tmp_path, *, bad_policy=False, no_checkpoint=False, stale_precision=False):
    proc_root=tmp_path/'proc'; proc=proc_root/'142'; proc.mkdir(parents=True)
    (proc_root/'stat').write_text('btime 1000\n')
    weights=tmp_path/'weights'; weights.mkdir()
    config=b'{"text_config":{"num_hidden_layers":64}}'
    (weights/'config.json').write_bytes(config)
    alias=tmp_path/'hf-alias'; alias.symlink_to(weights, target_is_directory=True)
    policy=tmp_path/'precision.json'; policy.write_text(json.dumps(selected_policy()))
    (proc/'cmdline').write_bytes(b'EngineCore\0vllm\0')
    env={'QWEN38_CHECKPOINT':str(weights), 'HF_MODEL':str(alias), 'QWEN38_PRECISION_CONFIG':str(policy), 'SECRET':'do-not-emit-credential'}
    if no_checkpoint: env.pop('QWEN38_CHECKPOINT')
    if stale_precision: env['QWEN38_PRECISION_CONFIG']='/stale-exec-time-policy.json'
    (proc/'environ').write_bytes(b'\0'.join((k+'='+v).encode() for k,v in env.items())+b'\0')
    (proc/'stat').write_text('142 (EngineCore) '+' '.join(['S']+['0']*18+['100']))
    payload={'manifest':{'model':'test', 'revision':'test', 'files':{'config.json':{'sha256':hashlib.sha256(config).hexdigest()}}},
             'verifier':Path('workflows/qwen38_checkpoint_identity.py').read_text(),
             'selected_precision':selected_policy(), 'precision_path':str(policy),
             'precision_sha256':'0'*64 if bad_policy else hashlib.sha256(policy.read_bytes()).hexdigest()}
    bootstrap="""import pathlib
real_path=pathlib.Path
root=real_path(ROOT)
pathlib.Path=lambda path: root if str(path)=='/proc' else root/'stat' if str(path)=='/proc/stat' else real_path(path)
exec(compile(PROBE,'<owned-checkpoint-probe>','exec'),{})
""".replace('ROOT',repr(str(proc_root))).replace('PROBE',repr(_CONTAINER_PROBE))
    return bootstrap,payload


@pytest.mark.parametrize('failure',[None,'bad_policy','no_checkpoint'])
def test_entire_probe_subprocess_with_selected_policy(tmp_path,failure):
    script,payload=fixture_probe(tmp_path,**({failure:True} if failure else {}))
    r=subprocess.run([sys.executable,'-c',script],input=json.dumps(payload),capture_output=True,text=True,timeout=20)
    assert 'do-not-emit-credential' not in r.stdout+r.stderr
    if failure:
        assert r.returncode!=0 and not r.stdout
        diagnostic=json.loads(r.stderr.removeprefix('QWEN38_CHECKPOINT_DIAGNOSTIC '))
        assert diagnostic['stage']==('precision_policy' if failure=='bad_policy' else 'process_scan')
        assert diagnostic['code']==('precision_artifact' if failure=='bad_policy' else 'checkpoint_not_unique')
        assert 'Traceback' not in r.stderr and str(tmp_path) not in r.stderr
    else:
        assert r.returncode==0,r.stderr
        result=json.loads(r.stdout)
        assert result['ok'] and result['checkpoint_layer_count']==64
        assert result['serving_processes'][0]['pid']==142
        assert result['serving_processes'][0]['start_epoch']>1000
        assert len(result['files'])==1 and result['files'][0]['matches']


def test_docker_failure_retains_only_bounded_allowlisted_diagnostic(tmp_path,monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER',OWNER)
    good,_=runner_for([CID])
    def runner(args,**kwargs):
        if args[1]=='exec':
            safe={'stage':'process_scan','code':'checkpoint_not_unique','exception':'RuntimeError','counts':{'matching_commands':1,'checkpoint_environments':0,'SECRET':123},'secret':'do-not-emit-credential'}
            return subprocess.CompletedProcess(args,1,'', 'do-not-emit-credential\nQWEN38_CHECKPOINT_DIAGNOSTIC '+json.dumps(safe)+'\n')
        return good(args,**kwargs)
    with pytest.raises(RuntimeError,match='checkpoint_not_unique') as error:
        ensure_checkpoint_preflight('Qwen/Qwen3.8-27B',tmp_path,runner=runner)
    text=(tmp_path/'checkpoint-preflight.json').read_text()
    assert 'do-not-emit-credential' not in text+str(error.value)
    result=json.loads(text)
    assert not result['ok']
    assert result['docker_failure']['probe']['counts']=={'matching_commands':1,'checkpoint_environments':0}


def test_stale_proc_precision_path_is_not_live_policy_authority(tmp_path):
    script,payload=fixture_probe(tmp_path,stale_precision=True)
    r=subprocess.run([sys.executable,'-c',script],input=json.dumps(payload),capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr
    result=json.loads(r.stdout)
    assert result['precision_artifact']['ok'] is True
    assert result['precision_artifact']['sha256']==payload['precision_sha256']
    assert result['serving_processes'][0]['proc_policy_path_matches_fixed_artifact'] is False


def test_missing_artifact_proof_fails_host_guard(tmp_path,monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER',OWNER)
    good,_=runner_for([CID])
    def runner(args,**kwargs):
        result=good(args,**kwargs)
        if args[1]=='exec':
            value=json.loads(result.stdout);value.pop('precision_artifact');result.stdout=json.dumps(value)
        return result
    with pytest.raises(RuntimeError,match='artifact proof missing'):
        ensure_checkpoint_preflight('Qwen/Qwen3.8-27B',tmp_path,runner=runner)
    assert json.loads((tmp_path/'checkpoint-preflight.json').read_text())['ok'] is False


def test_policy_artifact_changed_during_hashing_fails(tmp_path):
    script,payload=fixture_probe(tmp_path)
    payload['verifier'] += "\noriginal_verify = verify\ndef verify(weights, expected):\n    receipt = original_verify(weights, expected)\n    Path("+repr(payload['precision_path'])+").write_text('{}')\n    return receipt\n"
    r=subprocess.run([sys.executable,'-c',script],input=json.dumps(payload),capture_output=True,text=True,timeout=20)
    assert r.returncode!=0 and not r.stdout
    assert json.loads(r.stderr.removeprefix('QWEN38_CHECKPOINT_DIAGNOSTIC '))['code']=='precision_artifact'
