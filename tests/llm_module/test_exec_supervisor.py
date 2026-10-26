import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

SCRIPT = Path(__file__).parents[2] / 'agentic_owned_networks/exec_supervisor.py'


def launch(tmp_path, code):
    state=tmp_path/'state.json';token=uuid.uuid4().hex
    p=subprocess.Popen([sys.executable,str(SCRIPT),'run','--token',token,'--state',str(state),'--term-grace','.15','--kill-grace','2','--',sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    deadline=time.monotonic()+5
    while not state.exists() and p.poll() is None and time.monotonic()<deadline:time.sleep(.01)
    return p,state,token


def settle(state,token):
    return subprocess.run([sys.executable,str(SCRIPT),'settle','--token',token,'--state',str(state),'--timeout','3'],capture_output=True,text=True,timeout=5)


def test_normal_exit_and_no_command_leak(tmp_path):
    p,state,token=launch(tmp_path,"print('private-output');raise SystemExit(7)")
    stdout,stderr=p.communicate(timeout=5)
    assert p.returncode==7 and stdout.strip()=='private-output'
    data=json.loads(state.read_text());assert data['settled'] is True and 'private-output' not in state.read_text()
    assert state.stat().st_mode & 0o777 == 0o600
    assert settle(state,token).returncode==0


def test_normal_parent_exit_cleans_doublefork_setsid(tmp_path):
    code="import os,time,signal; pid=os.fork(); (os._exit(0) if pid else None); os.setsid(); pid=os.fork(); (os._exit(0) if pid else None); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
    p,state,token=launch(tmp_path,code)
    p.communicate(timeout=5)
    assert p.returncode==0
    data=json.loads(state.read_text());assert data['settled'] and data['reaped_descendants']>=1
    assert settle(state,token).returncode==0


def test_cancel_persistent_children_and_repeat(tmp_path):
    code="import os,time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); pid=os.fork(); (os.setsid() if pid==0 else None); time.sleep(30)"
    p,state,token=launch(tmp_path,code)
    time.sleep(.15)
    try:
        assert settle(state,token).returncode==0
        p.communicate(timeout=5);assert p.returncode==143
        assert settle(state,token).returncode==0
        assert json.loads(state.read_text())['settled'] is True
    finally:
        if p.poll() is None:p.send_signal(signal.SIGTERM);p.communicate(timeout=5)


def test_wrong_token_missing_and_pid_reuse(tmp_path):
    p,state,token=launch(tmp_path,'pass');p.communicate(timeout=5)
    assert settle(state,uuid.uuid4().hex).returncode==125
    assert settle(tmp_path/'missing',token).returncode==125
    data=json.loads(state.read_text());data['supervisor']['pid']=os.getpid();data['supervisor']['starttime']=0
    state.write_text(json.dumps(data));assert settle(state,token).returncode==125


def test_unsettled_missing_supervisor_not_absence(tmp_path):
    p,state,token=launch(tmp_path,'pass');p.communicate(timeout=5)
    data=json.loads(state.read_text());data['settled']=False;data['status']='running';state.write_text(json.dumps(data))
    assert settle(state,token).returncode==125


def test_no_overwrite_state(tmp_path):
    p,state,token=launch(tmp_path,'pass');p.communicate(timeout=5);original=state.read_bytes()
    retry=subprocess.run([sys.executable,str(SCRIPT),'run','--token',token,'--state',str(state),'--',sys.executable,'-c','raise SystemExit(99)'],capture_output=True,timeout=5)
    assert retry.returncode==125 and state.read_bytes()==original


def test_pidfd_unsupported_fails_before_launch(tmp_path, monkeypatch):
    from agentic_owned_networks import exec_supervisor as supervisor
    from types import SimpleNamespace
    import pytest
    def unsupported(pid): raise OSError('unsupported')
    monkeypatch.setattr(supervisor, 'pidfd_open', unsupported)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **k: pytest.fail('workload launched'))
    with pytest.raises(OSError):
        supervisor.run(SimpleNamespace(token=uuid.uuid4().hex,state=tmp_path/'state',command=['false']))
    assert not (tmp_path/'state').exists()
