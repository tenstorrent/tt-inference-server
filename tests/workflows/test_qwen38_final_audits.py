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


def test_published_exact_overlay_attribution_accepted():
    p={**plan(),'image':m.OVERLAY_IMAGE}
    assert m.validate_plan(p,p['runner_name'],expected_owner=p['source_owner'])==p


def test_other_overlay_digest_still_rejected():
    p={**plan(),'image':m.OVERLAY_IMAGE[:-1]+'0'}
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
