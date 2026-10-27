import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
from workflows import qwen38_final_audits as m
from workflows.run_docker_server import _write_qwen38_audit_source


def plan():
    return dict(source_run_id=123,source_owner='goal12-123-1-0',host_aliases=['120-qb2-p05t01','qb2-120-p05t01'],runner_name='120-qb2-p05t01',volume='volume_id_qwen38_autoport-Qwen3.8-27B',image=m.IMAGE,filenames=['qwen38-http-123.jsonl','qwen38-goal12-456.jsonl'])


def collect(tmp_path,edit=None):
    logs=tmp_path/'logs';logs.mkdir()
    for name in plan()['filenames']:(logs/name).write_bytes(b'{"event":"fixture"}\n')
    if edit:edit(logs)
    code=m.COLLECTOR.replace("'/evidence'",repr(str(tmp_path)))
    return subprocess.run([sys.executable,'-c',code,json.dumps(plan())],capture_output=True,timeout=5)


def test_real_stdlib_collector_exact_stable_zip(tmp_path):
    r=collect(tmp_path);assert r.returncode==0,r.stderr
    receipt=m.validate_archive(r.stdout,plan())
    assert receipt['required_paths']==[r['source_path'] for r in receipt['files']]
    assert receipt['serving_interval_complete'] is None
    assert all(r['stable_two_reads'] for r in receipt['files'])


@pytest.mark.parametrize('mode',['missing','partial','invalid','symlink','fifo'])
def test_bad_source_never_returns_stable_zip(tmp_path,mode):
    def edit(logs):
        p=logs/plan()['filenames'][0]
        if mode=='partial':p.write_bytes(b'{"event":')
        elif mode=='invalid':p.write_bytes(b'not-json\n')
        else:
            p.unlink()
            if mode=='symlink':p.symlink_to(logs/plan()['filenames'][1])
            if mode=='fifo':os.mkfifo(p)
    r=collect(tmp_path,edit);assert r.returncode!=0;assert not r.stdout


def test_metadata_or_member_tamper_rejected(tmp_path):
    raw=collect(tmp_path).stdout
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        files={n:z.read(n) for n in z.namelist()}
    files['tt_triage/'+plan()['filenames'][0]]=b'{"different":1}\n'
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for n,data in files.items():z.writestr(n,data)
    with pytest.raises(ValueError,match='integrity'):m.validate_archive(out.getvalue(),plan())


@pytest.mark.parametrize('key,value',[('source_owner','goal12-999-1-0'),('volume','other'),('image','mutable:latest'),('filenames',['../x','qwen38-http-1.jsonl'])])
def test_plan_attribution_rejected(key,value):
    p=plan();p[key]=value
    with pytest.raises(ValueError):m.validate_plan(p,p['runner_name'],expected_owner='goal12-123-1-0')


def test_wrong_host_rejected():
    with pytest.raises(ValueError):m.validate_plan(plan(),'120-qb2-p04t05')


@pytest.mark.parametrize('image',[m.OVERLAY_IMAGE,m.MASKED_OVERLAY_IMAGE])
def test_published_exact_overlay_attribution_accepted(image):
    p={**plan(),'image':image}
    assert m.validate_plan(p,p['runner_name'],expected_owner=p['source_owner'])==p


@pytest.mark.parametrize('image',[m.OVERLAY_IMAGE,m.MASKED_OVERLAY_IMAGE])
def test_other_overlay_digest_still_rejected(image):
    p={**plan(),'image':image[:-1]+'0'}
    with pytest.raises(ValueError,match='unexpected immutable image'):
        m.validate_plan(p,p['runner_name'],expected_owner=p['source_owner'])


def test_launch_receipt_only_generated_paths_no_env(tmp_path,monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER','goal12-123-1-0');monkeypatch.setenv('GITHUB_RUN_ID','123');monkeypatch.setenv('RUNNER_NAME','120-qb2-p05t01')
    monkeypatch.setattr('workflows.run_docker_server.get_default_workflow_root_log_dir',lambda:tmp_path)
    setup=SimpleNamespace(host_model_volume_root=None,docker_volume_name=plan()['volume'])
    spec=SimpleNamespace(impl=SimpleNamespace(impl_id='qwen38_autoport'),docker_image=m.IMAGE)
    env={'QWEN38_HTTP_AUDIT':'/logs/qwen38-http-123.jsonl','QWEN38_SERVING_EVENTS':'/logs/qwen38-goal12-456.jsonl','SECRET':'never-copy'}
    _write_qwen38_audit_source(setup,spec,'owned-server',env)
    p=next((tmp_path/'qwen38_audit_sources').glob('*.json'));body=json.loads(p.read_text());assert 'SECRET' not in p.read_text()
    m.validate_plan(body,'120-qb2-p05t01',expected_owner='goal12-123-1-0')


def test_streams_more_than_old_limit_under_memory_bound(tmp_path):
    # Exercise the real collector on 136 MiB, with a 96 MiB address-space cap.
    # Whole-file reads/joins cannot fit; only the compressed archive is retained.
    import hashlib
    import resource
    logs=tmp_path/'logs';logs.mkdir()
    line=json.dumps({'event':'fixture','payload':'x'*1000}).encode()+b'\n'
    chunk=line*1024
    first=logs/plan()['filenames'][0]
    digest=hashlib.sha256()
    with first.open('wb') as f:
        for _ in range(136):f.write(chunk);digest.update(chunk)
    (logs/plan()['filenames'][1]).write_bytes(b'{}\n')
    assert first.stat().st_size>128*1024*1024
    code=m.COLLECTOR.replace("'/evidence'",repr(str(tmp_path)))
    def memory_limit():resource.setrlimit(resource.RLIMIT_AS,(96*1024*1024,96*1024*1024))
    r=subprocess.run([sys.executable,'-c',code,json.dumps(plan())],capture_output=True,timeout=45,preexec_fn=memory_limit)
    assert r.returncode==0,r.stderr
    receipt=m.validate_archive(r.stdout,plan())
    assert receipt['files'][0]['sha256']==digest.hexdigest()
    assert receipt['files'][0]['bytes']==first.stat().st_size


def test_input_bound_before_reading_sparse_file(tmp_path):
    def edit(logs):
        with (logs/plan()['filenames'][0]).open('wb') as f:f.truncate(m.MAX_BYTES+1)
    r=collect(tmp_path,edit)
    assert r.returncode!=0 and b'input size bound' in r.stderr and not r.stdout


def test_line_bound(tmp_path):
    def edit(logs):(logs/plan()['filenames'][0]).write_bytes(b' '*m.MAX_LINE+b'\n')
    r=collect(tmp_path,edit)
    assert r.returncode!=0 and b'line size bound' in r.stderr and not r.stdout


@pytest.mark.parametrize('mutation',["os.write(os.open(name,os.O_WRONLY|os.O_APPEND,dir_fd=logsfd),b'{}\\n')", "os.truncate('/evidence/logs/'+name,0)"])
def test_change_between_reads_never_emits_archive(tmp_path,mutation):
    logs=tmp_path/'logs';logs.mkdir()
    for name in plan()['filenames']:(logs/name).write_bytes(b'{}\n')
    code=m.COLLECTOR.replace('   digest=h.hexdigest()', '   '+mutation+'\n   digest=h.hexdigest()')
    code=code.replace("'/evidence'",repr(str(tmp_path))).replace("'/evidence/logs/'",repr(str(tmp_path/'logs')+'/'))
    r=subprocess.run([sys.executable,'-c',code,json.dumps(plan())],capture_output=True,timeout=5)
    assert r.returncode!=0 and not r.stdout


def test_compressed_bound_emits_nothing(tmp_path):
    logs=tmp_path/'logs';logs.mkdir()
    for name in plan()['filenames']:(logs/name).write_bytes(b'{}\n')
    code=m.COLLECTOR.replace('MAX_ZIP=64*1024*1024','MAX_ZIP=1').replace("'/evidence'",repr(str(tmp_path)))
    r=subprocess.run([sys.executable,'-c',code,json.dumps(plan())],capture_output=True,timeout=5)
    assert r.returncode!=0 and b'compressed size bound' in r.stderr and not r.stdout


def test_collector_and_validator_bounds_agree():
    for name in ('MAX_BYTES','MAX_ZIP','MAX_LINE'):
        assignments=[line for line in m.COLLECTOR.splitlines() if line.startswith(name+'=')]
        assert len(assignments)==1
        assert eval(assignments[0].split('=',1)[1],{'__builtins__':{}})==getattr(m,name)
