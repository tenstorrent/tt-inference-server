"""CPU-only opt-in actual adapter qualification with exact pinned Harbor methods.

No dependency downloads. Full Harbor runtime and allocator network provisioning
are outside this fixture. Actual adapter exec/inspect/settle/teardown run unchanged.
"""
import argparse
import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import types
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HARBOR_SHA = '9973a7edc2d9950a03de6aeebafebb8f7b0a5f1e'


def cmd(args, check=True):
    return subprocess.run(args, capture_output=True, text=True, check=check, timeout=30)


def load_base(repo):
    source = subprocess.run(['git','-C',str(repo),'show',HARBOR_SHA+':src/harbor/environments/docker/docker.py'],
        env={**os.environ,'GIT_NO_LAZY_FETCH':'1'},capture_output=True,text=True,check=True,timeout=10).stdout
    klass = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='DockerEnvironment')
    selected = ['_compose_exec','_run_docker_compose_command','_collect_buffered_output','_terminate_process']
    methods = [n for n in klass.body if isinstance(n,ast.AsyncFunctionDef) and n.name in selected]
    fixture = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
        ast.ClassDef(name='DockerEnvironment',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    ns={'asyncio':asyncio,'ExecResult':types.SimpleNamespace,'MAIN_SERVICE_NAME':'main',
        '_sanitize_docker_compose_project_name':lambda x:x}
    exec(compile(ast.fix_missing_locations(fixture),str(repo)+'/exact-method-fixture','exec'),ns)
    base=ns['DockerEnvironment']
    base.runtime=classmethod(lambda cls:types.SimpleNamespace(compose=['docker','compose'],supports_compose_project_directory=True))
    base._docker_compose_paths=property(lambda self:[self.fixture_compose])
    base._compose_env_vars=lambda self,**kwargs:dict(os.environ)
    base._output_callback=lambda self:None
    for name in ['harbor','harbor.environments','harbor.environments.docker','harbor.environments.docker.docker','harbor.environments.base']:
        sys.modules[name]=types.ModuleType(name)
    sys.modules['harbor.environments.docker.docker'].DockerEnvironment=base
    sys.modules['harbor.environments.base'].ExecResult=types.SimpleNamespace
    return {'commit':HARBOR_SHA,'methods':selected,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'full_runtime_installed':False,'base_fixture_scaffolding':['runtime docker compose','public fixture fields','nonstream callback','Linux bash -lc platform','compose env os.environ']}


async def qualify(a, report):
    report['harbor']=load_base(a.harbor)
    from agentic_owned_networks.docker_environment import OwnedDockerEnvironment
    from agentic_owned_networks.allocator import Allocator,OWNER,MANAGED,ATTEMPT,JOB,POLICY,container_absent
    from agentic_owned_networks.execution_scope import installed_agent_scope
    from agentic_owned_networks.exec_authority import validate_exec_authority
    helper=ROOT/'agentic_owned_networks/exec_supervisor.py'
    report['sources']={name:hashlib.sha256((ROOT/'agentic_owned_networks'/name).read_bytes()).hexdigest()
                       for name in ['docker_environment.py','exec_supervisor.py','exec_authority.py']}
    owner=uuid.uuid4().hex;project='g12-adapter-'+owner
    labels={OWNER:owner,MANAGED:POLICY,ATTEMPT:owner,JOB:'cpu-adapter-qualification'}
    report['owner']=owner
    image=json.loads(cmd(['docker','image','inspect',a.image]).stdout)[0]['Id']
    report['image_id']=image
    with tempfile.TemporaryDirectory(prefix='g12-adapter-') as td:
        directory=Path(td); compose=directory/'compose.json'
        compose.write_text(json.dumps({'services':{'main':{
            'image':image,'pull_policy':'never','user':'0:0','network_mode':'none','labels':labels,
            'entrypoint':['python3','-c','import time;time.sleep(300)'],
            'mem_limit':'128m','cpus':.5,'pids_limit':64,'security_opt':['no-new-privileges'],
            'volumes':[{'type':'bind','source':str(helper),'target':'/opt/tt-agentic-owned/exec_supervisor.py','read_only':True}]}}}))
        env=OwnedDockerEnvironment.__new__(OwnedDockerEnvironment)
        env.fixture_compose=compose;env.environment_dir=directory;env.environment_name='cpu-qualification'
        env._platform=types.SimpleNamespace(exec_shell_args=lambda x:['bash','-lc',x])
        env.session_id=project;env._owned_labels=labels;env._owned_ready=True;env._owned_started=True
        env._owned_overlay=None;env._owned_uncertain=False;env._owned_workload_unsettled=False
        env._owned_allocator=Allocator(labels,project,directory/'journal.json','localhost')
        prefix=['docker','compose','--project-name',project,'-f',str(compose)]
        cid=None
        try:
            cmd(prefix+['up','-d','--no-build','--pull','never'])
            ids=await env._assert_project_owned();assert len(ids)==1;cid=ids[0]
            info=await env._container(cid)
            report['authority']=validate_exec_authority(info)
            report['inspected_ipc_mode']=info['HostConfig']['IpcMode']
            report['helper_mount']={k:info['Mounts'][0].get(k) for k in ['Type','Source','Destination','RW','Propagation']}
            assert report['helper_mount']['RW'] is False
            report['container_id']=cid
            async def execute(command):
                return await env._compose_exec(command,service='main',cwd='/tmp',env=None,timeout_sec=40,user='0')
            readcheck="import hashlib,sys;print(sys.version.split()[0]);print(hashlib.sha256(open('/opt/tt-agentic-owned/exec_supervisor.py','rb').read()).hexdigest())"
            result=await execute('python3 -c '+shlex.quote(readcheck))
            assert result.return_code==0,result.stdout
            version,copied=result.stdout.strip().splitlines()
            assert copied==report['sources']['exec_supervisor.py']
            report['python_version']=version;report['helper_source_visible_hash_verified']=True
            readonly=await execute("python3 -c \"import os;f=os.open('/opt/tt-agentic-owned/exec_supervisor.py',os.O_WRONLY)\"")
            assert readonly.return_code!=0 and 'Read-only file system' in readonly.stdout
            report['helper_write_denied_readonly']=True
            token=uuid.uuid4().hex;ready='/tmp/ready-'+token;pids='/tmp/pids-'+token
            code=f'''import os,time,signal,json,sys
sys.path.insert(0,'/opt/tt-agentic-owned')
from exec_supervisor import identity
signal.signal(signal.SIGTERM,signal.SIG_IGN)
def record():
 with open({pids!r},'a') as f:f.write(json.dumps(identity(os.getpid()))+'\\n')
record()
pid=os.fork()
if pid==0:
 os.setsid();record();pid=os.fork()
 if pid:os._exit(0)
 record();open({ready!r},'w').write('ready');time.sleep(120);os._exit(0)
while not os.path.exists({ready!r}):time.sleep(.01)
time.sleep(120)
'''
            async def scoped():
                scope=installed_agent_scope.set(True)
                try:return await execute('python3 -c '+shlex.quote(code))
                finally:installed_agent_scope.reset(scope)
            task=asyncio.create_task(scoped())
            try:
                for i in range(100):
                    r=await asyncio.to_thread(cmd,['docker','exec',cid,'python3','-c',f'import os;print(os.path.exists({ready!r}))'])
                    if r.stdout.strip()=='True':break
                    if task.done():await task;raise AssertionError('Agent exited before ready')
                    await asyncio.sleep(.05)
                else:raise TimeoutError('readiness')
                task.cancel();await asyncio.sleep(.15);task.cancel();await asyncio.sleep(.15);task.cancel()
                try:await asyncio.wait_for(task,35)
                except asyncio.CancelledError:report['original_cancellation_preserved']=True
                else:raise AssertionError('Cancellation lost')
                records=env._owned_allocator.record.get('workload_execs',[])
                assert len(records)==1 and records[0]['state']=='settled',records
                assert not env._owned_uncertain and not env._owned_workload_unsettled
                report['workload_records']=records;report['repeat_cancel_calls']=3
                report['structural_uncertainty']=env._owned_uncertain
                check=f'''import sys,json
sys.path.insert(0,'/opt/tt-agentic-owned')
from exec_supervisor import identity,same
rows=[json.loads(s) for s in open({pids!r})]
left=[r for r in rows if same(r,identity(r['pid']))]
assert not left,left
print(json.dumps({{'recorded_processes':len(rows),'remaining':len(left),'verifier_after_settle':True}}))
'''
                verifier=await execute('python3 -c '+shlex.quote(check))
                assert verifier.return_code==0,verifier.stdout
                report['verifier']=json.loads(verifier.stdout)
            finally:
                if not task.done():
                    task.cancel()
                    try:await task
                    except BaseException:pass
            await env._teardown()
            assert not await env._project_containers()
            absent=cmd(['docker','inspect',cid],check=False)
            assert container_absent(absent,cid)
            report['adapter_exact_owned_teardown']={'container_absent':True,'network_created':False}
            report['status']='pass'
        finally:
            # Only exact IDs whose current labels are verified by the adapter.
            for ident in await env._assert_project_owned():
                await env._container(ident)
                cmd(['docker','rm','-f',ident])
            report['final_owned_project_empty']=not bool(await env._project_containers())
            report['journal']=env._owned_allocator.record


def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--harbor',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    report={'status':'running','limits':['Exact pinned Harbor method fixture, not full installed Harbor runtime.',
        'Adapter __init__/start/_prepare_owned and allocated-bridge network creation are outside fixture; one owned network:none Compose service is provisioned directly.',
        'Actual adapter exec, real Docker authority inspection, repeated cancellation settlement, verifier order and exact-ID teardown tested.',
        'No model/device or external API calls.']}
    try:asyncio.run(qualify(a,report))
    except Exception as exc:
        report.update(status='failed',error_class=type(exc).__name__,error=str(exc)[:800]);raise
    finally:a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'output':str(a.output)}))


if __name__=='__main__':main()
