"""Explicit opt-in real Docker CPU qualification; never collected by pytest.

Uses one exact uniquely labeled container, no device/mount/network access.
No adapter integration or hostile workload security claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'agentic_owned_networks/exec_supervisor.py'
LABEL = 'qwen38.goal12.supervisor-qualification'


def command(argv, check=True, timeout=20):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=check)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, help='Already cached image containing python3')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    owner = uuid.uuid4().hex
    receipt = {'status': 'running', 'owner': owner, 'cases': [],
               'supervisor_sha256': hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
               'limits': ['CPU Docker helper qualification only; Harbor adapter not invoked.',
                          'Cooperative workloads; not hostile-code isolation proof.',
                          'HTTP cancellation is blocked loopback urllib request terminated through supervised process settlement, not vLLM or external API cancellation.']}
    cid = None
    def save():
        a.output.write_text(json.dumps(receipt, indent=2) + '\n')
    def dex(*args, **kw):
        return command(['docker', 'exec', cid, 'python3', *args], **kw)
    def inspect():
        return json.loads(command(['docker', 'inspect', cid]).stdout)[0]
    try:
        image = json.loads(command(['docker', 'image', 'inspect', a.image]).stdout)[0]['Id']
        receipt['image_id'] = image
        cid = command(['docker', 'run', '-d', '--pull=never', '--name', 'g12-supervisor-' + owner,
                       '--user', '0:0', '--label', LABEL + '=' + owner, '--network', 'none', '--cap-drop', 'ALL',
                       '--security-opt', 'no-new-privileges', '--memory', '128m', '--cpus', '.5',
                       '--pids-limit', '64', '--entrypoint', 'python3', image,
                       '-c', 'import time; time.sleep(300)']).stdout.strip()
        info = inspect()
        assert info['Config']['Labels'][LABEL] == owner
        assert info['HostConfig']['Devices'] in (None, []) and not info['Mounts']
        assert info['HostConfig']['Privileged'] is False and info['HostConfig']['NetworkMode'] == 'none'
        receipt['container_id'] = cid
        receipt['authority'] = {'devices': [], 'mounts': [], 'privileged': False, 'network': 'none',
                                'cap_drop': info['HostConfig']['CapDrop'], 'memory_bytes': info['HostConfig']['Memory']}
        subprocess.run(['docker', 'exec', '-i', cid, 'python3', '-c', "import sys;open('/tmp/exec_supervisor.py','wb').write(sys.stdin.buffer.read())"], input=SCRIPT.read_bytes(), check=True, timeout=20)
        copied = dex('-c', "import hashlib;print(hashlib.sha256(open('/tmp/exec_supervisor.py','rb').read()).hexdigest())").stdout.strip()
        assert copied == receipt['supervisor_sha256']
        receipt['copied_source_hash_verified'] = True
        for mode in ('normal_doublefork', 'cancel_persistent_doublefork', 'cancel_blocked_http'):
            token = uuid.uuid4().hex
            prefix = '/tmp/' + token
            state = prefix + '.json'
            common = f'''import os,sys,time,signal,json,socket,urllib.request
sys.path.insert(0,'/tmp')
from exec_supervisor import identity
prefix={prefix!r}
def record():
    with open(prefix+'.pids','a') as f: f.write(json.dumps(identity(os.getpid()))+'\\n')
def ready():
    open(prefix+'.ready','w').write('ready')
signal.signal(signal.SIGTERM,signal.SIG_IGN)
record()
'''
            if mode == 'cancel_blocked_http':
                code = common + '''pid=os.fork()
if pid==0:
    record()
    s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',18099));s.listen(1)
    open(prefix+'.listening','w').write('1')
    c,addr=s.accept();c.recv(4096);ready();time.sleep(120);os._exit(0)
while not os.path.exists(prefix+'.listening'):time.sleep(.01)
urllib.request.urlopen('http://127.0.0.1:18099/',timeout=120).read()
'''
            else:
                code = common + '''pid=os.fork()
if pid==0:
    os.setsid();record();pid=os.fork()
    if pid:os._exit(0)
    record();ready();time.sleep(120);os._exit(0)
while not os.path.exists(prefix+'.ready'):time.sleep(.01)
'''
                if mode != 'normal_doublefork': code += 'time.sleep(120)\n'
            proc = subprocess.Popen(['docker','exec',cid,'python3','/tmp/exec_supervisor.py','run',
                                     '--token',token,'--state',state,'--term-grace','.3','--kill-grace','3',
                                     '--','python3','-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            case = {'name': mode, 'token': token}
            receipt['cases'].append(case)
            try:
                deadline = time.monotonic() + 15
                while True:
                    ready = dex('-c', f'import os;print(os.path.exists({prefix + ".ready"!r}))').stdout.strip() == 'True'
                    if ready: break
                    assert time.monotonic() < deadline, 'Workload readiness timeout'
                    assert proc.poll() is None, 'Supervisor exited before workload readiness'
                    time.sleep(.05)
                case['workload_ready_before_cancel'] = True
                if mode.startswith('cancel_'):
                    proc.terminate()
                    proc.communicate(timeout=5)
                    case['host_docker_cli_exit_code'] = proc.returncode
                    before = json.loads(dex('-c',f'print(open({state!r}).read())').stdout)
                    case['after_cli_cancel_before_settle'] = {'status':before['status'],'settled':before['settled']}
                    assert before['settled'] is False, 'Discriminator did not retain daemon workload'
                else:
                    proc.communicate(timeout=10)
                    case['host_docker_cli_exit_code'] = proc.returncode
                    assert proc.returncode == 0
                settle = dex('/tmp/exec_supervisor.py','settle','--token',token,'--state',state,'--timeout','8')
                case['settle_receipt'] = json.loads(settle.stdout)
                case['state'] = json.loads(dex('-c',f'print(open({state!r}).read())').stdout)
                assert case['state']['settled'] is True
                verify_code = f'''import sys,json,socket
sys.path.insert(0,'/tmp')
from exec_supervisor import identity,same
records=[json.loads(s) for s in open({prefix + '.pids'!r})]
remaining=[r for r in records if same(r,identity(r['pid']))]
assert not remaining,remaining
print(json.dumps({{'recorded_workload_processes':len(records),'remaining_matching_processes':len(remaining),'verifier_started_after_settle':True}}))
'''
                if mode == 'cancel_blocked_http':
                    verify_code += "s=socket.socket();assert s.connect_ex(('127.0.0.1',18099)) != 0;s.close()\n"
                case['verifier'] = json.loads(dex('-c', verify_code).stdout)
                case['repeat_settle_receipt'] = json.loads(dex('/tmp/exec_supervisor.py','settle','--token',token,'--state',state,'--timeout','3').stdout)
                case['status'] = 'pass'
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.communicate(timeout=5)
            save()
        receipt['status'] = 'pass'
    except Exception as exc:
        receipt['status'] = 'failed'
        receipt['error_class'] = type(exc).__name__
        receipt['error'] = str(exc)[:600]
        if isinstance(exc, subprocess.CalledProcessError): receipt['stderr'] = exc.stderr[:600]
        raise
    finally:
        if cid:
            info = inspect()
            assert info['Id'] == cid and info['Config']['Labels'][LABEL] == owner
            command(['docker','rm','-f',cid])
            absent = command(['docker','inspect',cid],check=False)
            receipt['cleanup'] = {'exact_id_label_verified_before_remove': True,
                                  'container_absent': absent.returncode != 0,
                                  'foreign_resources_mutated': False}
        save()
    print(json.dumps({'status': receipt['status'], 'output': str(a.output)}))


if __name__ == '__main__': main()
