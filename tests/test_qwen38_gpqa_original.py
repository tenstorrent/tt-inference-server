import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from llm_module.qwen38_gpqa_guard import digest
from llm_module.qwen38_gpqa_original_guard import verify_payload,validate_command
from reference_config.evals.qwen38_gpqa_original import configure,MANIFEST
from tests.test_qwen38_gpqa_arm import release_task,spec
from llm_module.eval_command import build_eval_command

def test_original_command(monkeypatch):
    import llm_module.eval_command as builder
    monkeypatch.setenv('TT_QWEN38_GPQA_ORIGINAL','1')
    monkeypatch.setattr(builder,'get_venv_provisioner',lambda:NS(venv_path=lambda _:Path('/fake/venv')))
    rc=NS(eval_samples=None,eval_shard=None,limit_samples_mode=None)
    task=configure([release_task()],rc)[0]
    assert task.score.published_score is None and task.score.gpu_reference_score is None
    assert task.doc_ids==[12,15] and task.max_attempts==1 and task.max_concurrent==32
    assert 'stream' not in task.gen_kwargs and task.gen_kwargs['max_gen_toks']==98304
    cmd=build_eval_command(task,spec(),None,'/tmp/results',8000,runtime_config=rc)
    assert Path(cmd[1]).name=='qwen38_gpqa_original_guard.py'
    target=cmd[cmd.index('--')+1:];validate_command(target)
    assert json.loads(target[target.index('--samples')+1])=={'gpqa_diamond_cot_zeroshot':[12,15]}
    for key,value in [('timeout=18000','timeout=7200'),('max_retries=0','max_retries=1')]:
        bad=[v.replace(key,value) for v in target]
        with pytest.raises(ValueError):validate_command(bad)

def test_payload_exactness_and_full_time_reserve():
    payload={'messages':[{'role':'user','content':'original'}],'max_tokens':98304,'seed':1234}
    m={'messages_sha256':{'12':digest(payload['messages'])},'payload_sha256':{'12':digest(payload)}}
    assert verify_payload(payload,m,20000,now=0)==12
    for field,value in [('seed',0),('max_tokens',16384),('stream',False),('messages',[])]:
        bad=copy.deepcopy(payload);bad[field]=value
        with pytest.raises(ValueError):verify_payload(bad,m,20000,now=0)
    with pytest.raises(ValueError):verify_payload(payload,m,18299,now=0)
    with pytest.raises(ValueError):verify_payload(payload,m,20000,cleanup=0,now=0)

def test_manifest_and_no_selection_override():
    m=json.loads(MANIFEST.read_text())
    assert m['doc_ids']==[12,15] and len(m['payload_sha256'])==2
    for rc in (NS(eval_samples='2'),NS(eval_shard='0/2'),NS(limit_samples_mode='smoke-test')):
        with pytest.raises(ValueError):configure([release_task()],rc)

def test_candidate_default_catalog_selects_original_without_environment(monkeypatch):
    from reference_config.evals.qwen38_gpqa_original import enabled
    monkeypatch.delenv('TT_QWEN38_GPQA_ORIGINAL',raising=False)
    assert enabled()
    assert configure([release_task()])[0].task_name=='gpqa_diamond_cot_zeroshot'
    # Actual default run.py invokes admission before host bootstrap/server startup.
    source=Path('run.py').read_text()
    main=source[source.index('def main():'):]
    assert main.index('prepare_original_gpqa(args)')<main.index('bootstrap_uv()')
    source=Path('llm_module/eval_configs.py').read_text()
    assert 'original_configure(standard, runtime_config) if enabled()' in source

def test_allocation_anchor_and_clock_never_reset(tmp_path):
    from workflows.qwen38_original_admission import allocation_from_jobs,bind_clock
    env={'RUNNER_NAME':'owned','GITHUB_RUN_ID':'12','GITHUB_RUN_ATTEMPT':'1','GITHUB_REPOSITORY':'tenstorrent/tt-shield'}
    jobs=[{'id':42,'runner_name':'owned','status':'in_progress','started_at':'2026-11-03T04:00:00Z'}]
    clock=allocation_from_jobs(jobs,env)
    assert clock['deadline_epoch']-clock['allocation_started_epoch']==19800
    path=tmp_path/'clock.json';bind_clock(clock,path);bind_clock(clock,path)
    with pytest.raises(ValueError):bind_clock(dict(clock,deadline_epoch=clock['deadline_epoch']+1),path)
    with pytest.raises(ValueError):allocation_from_jobs(jobs*2,env)

def test_actual_entrypoint_refuses_missing_admission_before_any_fetch(monkeypatch):
    from workflows.qwen38_original_admission import prepare
    with pytest.raises(ValueError,match='coordinator'):
        prepare(NS(model='Qwen3.8-27B'),{'GITHUB_ACTIONS':'true'},fetch=lambda _:pytest.fail('No remote call before admission'))

def test_runtime_identity_observes_actual_container_not_dispatch_refs(tmp_path,monkeypatch):
    import workflows.qwen38_original_admission as a
    c={'runtime_repo_paths':{'metal':'/actual/metal','plugin':'/actual/plugin'}}
    monkeypatch.setattr(a,'controls',lambda:c)
    observed={'host_tis_sha':'1'*40,'metal_sha':'2'*40,'plugin_sha':'3'*40,'image_digest':'sha256:actual'}
    import hashlib
    approved={'run.py':'approved-hash'}
    monkeypatch.setattr(a,'source_files',lambda:approved)
    controls_hash=hashlib.sha256((a.ROOT/'reference_config/evals/qwen38_gpqa_original.json').read_bytes()).hexdigest()
    pre={'runtime_sources':dict(observed,host_tis_sha=None),'host_source_files':approved,'selected_controls_sha256':controls_hash}
    p=tmp_path/'pre.json';p.write_text(json.dumps(pre))
    monkeypatch.setenv('TT_QWEN38_PREFLIGHT_JSON',str(p))
    seen=[]
    def run(cmd,**kwargs):
        seen.append(cmd)
        result=observed['host_tis_sha'] if cmd[0]=='git' else observed['metal_sha' if '/actual/metal' in cmd else 'plugin_sha']
        return NS(stdout=result)
    receipt=a.runtime_identity({'container_id':'a'*64,'image_id':'sha256:actual','ok':True},tmp_path,run)
    assert receipt['observed']==observed
    assert all(cmd[2]=='a'*64 for cmd in seen if cmd[0]=='docker')
    assert json.loads((tmp_path/'original-observed-preflight.json').read_text())['runtime_sources']['host_tis_sha']==observed['host_tis_sha']
    monkeypatch.setenv('TT_QWEN38_PREFLIGHT_JSON',str(p))
    p.write_text(json.dumps({**pre,'runtime_sources':dict(observed,metal_sha='dispatch-wish-ref')}))
    with pytest.raises(ValueError,match='Actual runtime'):a.runtime_identity({'container_id':'a'*64,'image_id':'sha256:actual','ok':True},tmp_path,run)

def test_approved_source_manifest_detects_edits_and_added_tracked_code(tmp_path):
    from workflows.qwen38_original_admission import source_files
    (tmp_path/'one.py').write_text('original\n')
    def ls(*a,**kw):return NS(stdout=b'one.py\0')
    first=source_files(tmp_path,ls)
    (tmp_path/'one.py').write_text('changed\n')
    assert source_files(tmp_path,ls)!=first
    (tmp_path/'two.py').write_text('new\n')
    assert source_files(tmp_path,lambda *a,**kw:NS(stdout=b'one.py\0two.py\0')).keys()!=first.keys()
