"""Explicit unscored32-network runtime proof; never imported as an automatic test.

Run on granted daemon host, after source tests, with an already-present image
containing python3+CA roots. No image pull, daemon change or pool exhaustion.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid
from urllib.parse import urlparse

from .allocator import Allocator, OWNER, MANAGED, ATTEMPT, JOB, POLICY, OwnershipError, checked_owner

CHECK = r'''
import json,socket,sys,urllib.request
own,foreign,foreign_alias,endpoint=sys.argv[1:]
assert socket.gethostbyname('main')==own,'within-trial main DNS mismatch'
assert urllib.request.urlopen('http://main:8080',timeout=5).status==200
try:socket.gethostbyname(foreign_alias)
except socket.gaierror:pass
else:raise AssertionError('cross-trial alias unexpectedly resolved')
try:
 c=socket.create_connection((foreign,8080),timeout=2);c.close()
except OSError:pass
else:raise AssertionError('cross-trial direct IP unexpectedly reachable')
assert urllib.request.urlopen('https://example.com',timeout=10).status==200
assert urllib.request.urlopen(endpoint,timeout=10).status==200
print(json.dumps({'dns':True,'same_trial_http':True,'cross_alias_blocked':True,'cross_ip_blocked':True,'public_https':True,'model_endpoint':True}))
'''


def run_once(config):
    """Opt-in CI admission proof, before the measured Harbor attempt begins."""
    import fcntl
    import os
    import sys
    from urllib.parse import urlsplit, urlunsplit
    from llm_module.agentic.harbor import _get_agent_endpoint
    if os.environ.get('TT_AGENTIC_NETWORK_PREFLIGHT') != '1':
        return
    if config.environment_type != 'docker':
        raise ValueError('Network proof requires Docker trials')
    owner=checked_owner(os.environ.get('TT_AGENTIC_CONTAINER_OWNER'))
    root=config.jobs_dir.resolve()/'network-preflight'
    root.parent.mkdir(parents=True,exist_ok=True)
    with (root.parent/'network-preflight.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        existing=root/'preflight.json'
        if existing.exists():
            result=json.loads(existing.read_text())
            if result.get('owner')==owner and result.get('status')=='pass' and len(result.get('checks',[]))==32 and not result.get('cleanup_errors'):
                return
            raise RuntimeError('Existing network proof is incomplete/failed or belongs to another owner')
        def command(args):
            return subprocess.run(args,capture_output=True,text=True,timeout=30,check=True).stdout
        ids=command(['docker','ps','-q','--no-trunc','--filter','label='+OWNER+'='+owner,'--filter','name=tt-inference-server-']).split()
        servers=[]
        fmt='{"id":{{json .Id}},"name":{{json .Name}},"image":{{json .Image}},"labels":{{json .Config.Labels}},"running":{{json .State.Running}}}'
        for ident in ids:
            item=json.loads(command(['docker','inspect','--format',fmt,ident]))
            if item.get('id')!=ident or item.get('labels',{}).get(OWNER)!=owner:
                raise OwnershipError('Owner inventory changed during preflight')
            if item.get('running') and item.get('name','').startswith('/tt-inference-server-'):
                servers.append(item)
        if len(servers)!=1:
            raise OwnershipError('Need exactly one running exact-owner inference server image; no fallback')
        endpoint=urlsplit(_get_agent_endpoint(config))
        host=endpoint.hostname
        if not host or endpoint.username or endpoint.password or endpoint.query:
            raise ValueError('Credential-free endpoint required')
        translated='host.docker.internal'if host in ('127.0.0.1','localhost','::1')else host
        if ':' in translated:translated='['+translated+']'
        health=urlunsplit((endpoint.scheme,translated+(':'+str(endpoint.port)if endpoint.port else ''),'/health','',''))
        env=os.environ.copy();source=str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH']=os.pathsep.join([source]+([env['PYTHONPATH']]if env.get('PYTHONPATH')else []))
        interpreter=str(config.venv_python or sys.executable)
        subprocess.run([interpreter,'-c',
            'from harbor.utils.import_path import import_class; import_class("agentic_owned_networks.docker_environment:OwnedDockerEnvironment")'],
            env=env,timeout=30,check=True)
        # No outer kill: the bounded module must finish exact-ID cleanup/receipt.
        result=subprocess.run([interpreter,'-m','agentic_owned_networks.preflight',
            '--owner',owner,'--image',servers[0]['image'],'--model-endpoint',health,
            '--inventory-model-host',host,'--output',str(root)],env=env,check=False)
        if result.returncode:
            raise RuntimeError('Network admission proof failed; agents were not started')
        receipt=json.loads(existing.read_text())
        if receipt.get('status')!='pass' or len(receipt.get('checks',[]))!=32 or receipt.get('cleanup_errors'):
            raise RuntimeError('Network admission proof did not produce a complete pass')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner',required=True)
    p.add_argument('--image',required=True,help='Existing immutable image with python3; never pulled')
    p.add_argument('--model-endpoint',required=True,help='Health/models URL reachable from trial containers, no credentials/query')
    p.add_argument('--inventory-model-host',required=True,help='Model hostname/IP as resolved on daemon host (before host-gateway translation)')
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();checked_owner(args.owner)
    started=time.monotonic()
    deadline=started+900
    def check_deadline():
        if time.monotonic()>deadline:raise TimeoutError('900-second preflight work budget exceeded; attempting owned cleanup')
    parsed=urlparse(args.model_endpoint)
    if parsed.scheme not in ('http','https')or not parsed.hostname or parsed.username or parsed.password or parsed.query:
        raise ValueError('Plain credential-free model health endpoint required')
    if args.output.exists():raise FileExistsError('Fresh output directory required')
    args.output.mkdir(parents=True)
    job=hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:24]
    journal=args.output/'owned_networks';records=[];allocators=[];containers=[]
    receipt={'scored':False,'status':'running','checks':[],'errors':[],'concurrent_networks_required':32,'owner':args.owner,'job':job,
             'limits':['Connectivity observed for this snapshot only; no model score or device throughput measurement.',
                       'Uses existing image only. Does not exhaust daemon default pools.','No cleanup of foreign resources.']}
    def save():(args.output/'preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    try:
        image=subprocess.run(['docker','image','inspect','--format','{{json .Id}}',args.image],capture_output=True,text=True,timeout=30,check=True)
        image_id=json.loads(image.stdout)
        if not isinstance(image_id,str) or not image_id.startswith('sha256:'):raise ValueError('Existing immutable image identity unavailable')
        receipt['image_id']=image_id
        for i in range(32):
            attempt=uuid.uuid4().hex;labels={OWNER:args.owner,MANAGED:POLICY,ATTEMPT:attempt,JOB:job}
            allocators.append(Allocator(labels,'g12-proof-'+attempt,journal/(attempt+'.json'),args.inventory_model_host))
        with concurrent.futures.ThreadPoolExecutor(max_workers=32)as pool:
            records=list(pool.map(lambda a:a.acquire(),allocators))
        receipt['networks']=[{k:r[k]for k in ('network_id','subnet','gateway','project','attempt')}for r in records]
        if len({r['network_id']for r in records})!=32:raise AssertionError('Need32distinct simultaneous network IDs')
        # All32networks are held until every connectivity check and cleanup.
        for a in allocators:
            check_deadline()
            command=['docker','run','--detach','--pull','never','--network',a.record['network_name'],
                     '--network-alias','main','--network-alias','proof-'+a.record['attempt'],
                     '--add-host','host.docker.internal:host-gateway','--entrypoint','python3']
            for k,v in a.labels.items():command+=['--label',k+'='+v]
            response=a.command(command+[image_id,'-m','http.server','8080','--bind','0.0.0.0'])
            cid=response.stdout.strip();containers.append((a,cid))
            a.record['container_ids']=[cid];a.record['state']='attached';a.save()
        ips=[]
        for a,cid in containers:
            a.verify(a.inspect(a.record['network_id']))
            net=a.inspect(a.record['network_id']);endpoints=net.get('Containers')or{}
            if list(endpoints)!=[cid]:raise OwnershipError('Unexpected network endpoints')
            ips.append(endpoints[cid]['IPv4Address'].split('/')[0])
        def verify_one(index_pair):
            check_deadline()
            i,(a,cid)=index_pair
            other=(i+1)%32
            result=a.command(['docker','exec',cid,'python3','-c',CHECK,ips[i],ips[other],
                              'proof-'+allocators[other].record['attempt'],args.model_endpoint])
            return {'attempt':a.record['attempt'],**json.loads(result.stdout)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=32)as pool:
            for checked in pool.map(verify_one,enumerate(containers)):
                receipt['checks'].append(checked);save()
        receipt['status']='connectivity_pass'
    except BaseException as exc:
        receipt['status']='failed';receipt['errors'].append(type(exc).__name__)
    finally:
        cleanup=[]
        for a,cid in containers:
            try:
                fmt='{"id":{{json .Id}},"labels":{{json .Config.Labels}}}'
                item=json.loads(a.command(['docker','inspect','--format',fmt,cid]).stdout)
                if item['id']!=cid or any(item['labels'].get(k)!=v for k,v in a.labels.items()):raise OwnershipError('Container ownership mismatch')
                a.command(['docker','stop','--time','10',cid]);a.command(['docker','rm',cid])
            except BaseException as exc:cleanup.append({'container':cid,'error':type(exc).__name__})
        for a in allocators:
            try:
                if a.record['state']=='needs_reconciliation':raise OwnershipError('Uncertain create retained')
                a.remove_empty()
            except BaseException as exc:cleanup.append({'attempt':a.record['attempt'],'error':type(exc).__name__})
        receipt['cleanup_errors']=cleanup
        receipt['wall_seconds']=time.monotonic()-started
        receipt['work_budget_seconds']=900
        receipt['command_timeout_seconds']=30
        receipt['cleanup_basis']='Exact IDs only; cleanup can extend work budget, each command bounded to30seconds'
        if cleanup or len(receipt['checks'])!=32:receipt['status']='failed'
        elif receipt['status']=='connectivity_pass':receipt['status']='pass'
        # The explicit synchronous executor has joined all creator workers here.
        journal.mkdir(exist_ok=True)
        (journal/('closed-'+job+'.json')).write_text(json.dumps({'schema_version':1,'owner':args.owner,'job':job,
            'journal_root':str(journal.resolve()),'process_group_settled':True,'closed_at':datetime.now(timezone.utc).isoformat(),
            'basis':'Explicit synchronous preflight; all executor workers joined; no allocator background process'},indent=2)+'\n')
        save()
    raise SystemExit(0 if receipt['status']=='pass'else 1)


if __name__=='__main__':main()
