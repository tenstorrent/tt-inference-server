import concurrent.futures
import ipaddress as ip
import json
import threading
from types import SimpleNamespace

import pytest
from agentic_owned_networks.allocator import Allocator, OWNER, MANAGED, ATTEMPT, JOB, POLICY, OwnershipError, candidates, topology
from tests.llm_module.test_harbor_dataset_ref import _run_config
from llm_module.agentic import harbor


def test_candidates_reject_broad_routes_and_containment():
    exclusions=[ip.ip_network(x)for x in ('10.0.0.0/8','172.16.0.0/12','192.168.0.0/17','192.168.128.4/32')]
    chosen=list(candidates(exclusions,'fixture'))
    assert chosen and all(n.subnet_of(ip.ip_network('192.168.0.0/16')) for n in chosen)
    assert all(not n.overlaps(x)for n in chosen for x in exclusions)
    assert list(candidates([ip.ip_network('0.0.0.0/0')],'fixture'))==[]


def test_rendered_topology_rejects_isolation_escapes():
    default={'services':{'main':{}},'networks':{'default':{'name':'p_default'}}}
    assert topology(default,'p')==['main']
    for escape in ({'ports':['80:80']},{'network_mode':'host'},{'privileged':True},{'networks':{'default':{'ipv4_address':'10.0.0.2'}}},{'deploy':{'replicas':14}}):
        with pytest.raises(ValueError):topology({'services':{'main':escape}},'p')
    with pytest.raises(ValueError):topology({'services':{'main':{}},'networks':{'default':{'external':True}}},'p')


class FakeDaemon:
    def __init__(self):self.lock=threading.Lock();self.networks={};self.created=0;self.calls=[]
    def run(self,args,**kwargs):
        with self.lock:
            self.calls.append(args);rc=0;stdout='';stderr=''
            if args[:3]==['docker','network','create']:
                subnet=args[args.index('--subnet')+1]
                if any(ip.ip_network(subnet).overlaps(ip.ip_network(n['IPAM']['Config'][0]['Subnet']))for n in self.networks.values()):rc=1;stderr='Pool overlaps with other one on this address space'
                else:
                    self.created+=1;ident=f'{self.created:064x}';labels={args[i+1].split('=',1)[0]:args[i+1].split('=',1)[1]for i,a in enumerate(args)if a=='--label'}
                    self.networks[ident]={'Id':ident,'Name':args[-1],'Driver':'bridge','Scope':'local','Labels':labels,'Internal':False,'Options':{},'IPAM':{'Config':[{'Subnet':subnet,'Gateway':args[args.index('--gateway')+1]}]},'Containers':{}}
                    stdout=ident
            elif args[:3]==['docker','network','inspect']:
                n=self.networks.get(args[-1]) or next((n for n in self.networks.values()if n['Name']==args[-1]),None)
                if n:stdout=json.dumps([n])
                else:rc=1;stderr='No such network'
            elif args[:3]==['docker','network','rm']:self.networks.pop(args[-1])
            else:raise AssertionError(args)
            return SimpleNamespace(returncode=rc,stdout=stdout,stderr=stderr)


class TestAllocator(Allocator):
    __test__=False
    def local_daemon(self):self.record['daemon_id']='fixture';self.save()
    def inventory(self,skip=None):
        with self.run.__self__.lock:
            return [ip.ip_network(n['IPAM']['Config'][0]['Subnet'])for key,n in self.run.__self__.networks.items()if key!=skip]


def allocator(tmp_path,daemon,index):
    labels={OWNER:'owner-'+str(index%2),MANAGED:POLICY,ATTEMPT:str(index),JOB:'job'}
    return TestAllocator(labels,'fixture-'+str(index),tmp_path/(str(index)+'.json'),'localhost',daemon.run)


def test_32_concurrent_creates_isolated_and_exact_id_cleanup(tmp_path):
    daemon=FakeDaemon();allocators=[allocator(tmp_path,daemon,i)for i in range(32)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=32)as pool:list(pool.map(lambda a:a.acquire(),allocators))
    assert len(daemon.networks)==32
    nets=[ip.ip_network(a.record['subnet'])for a in allocators]
    assert all(not a.overlaps(b)for i,a in enumerate(nets)for b in nets[i+1:])
    assert all('--ip-range'not in call for call in daemon.calls)
    for a in allocators:a.remove_empty();a.remove_empty()
    assert not daemon.networks
    assert all(call[:3]in (['docker','network','create'],['docker','network','inspect'],['docker','network','rm'])for call in daemon.calls)


def test_network_foreign_labels_or_endpoints_block_removal(tmp_path):
    daemon=FakeDaemon();a=allocator(tmp_path,daemon,0);a.acquire();n=daemon.networks[a.record['network_id']]
    n['Containers']={'foreign':{}}
    with pytest.raises(OwnershipError):a.remove_empty()
    n['Containers']={};n['Labels'][OWNER]='foreign'
    with pytest.raises(OwnershipError):a.remove_empty()
    assert len(daemon.networks)==1


def test_config_owner_forces_import_path_without_changing_host_gateway(tmp_path,monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER','goal12-fixture')
    config=_run_config(tmp_path)
    assert harbor._needs_config_file(config)
    data=json.loads(harbor._write_harbor_config(config).read_text())
    env=data['environment'];assert env['import_path'].endswith(':OwnedDockerEnvironment')
    assert env['kwargs']['owned_owner']=='goal12-fixture'
    assert str(tmp_path)in env['kwargs']['owned_journal_dir']
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER','bad owner!')
    with pytest.raises(ValueError):harbor._write_harbor_config(config)


def test_child_import_path_and_closed_marker_only_after_settlement(tmp_path,monkeypatch):
    monkeypatch.setenv('TT_AGENTIC_CONTAINER_OWNER','goal12-fixture')
    monkeypatch.setenv('PYTHONPATH','/existing/path')
    config=_run_config(tmp_path)
    captured={}
    def run(*args,**kw):captured.update(kw);return 0
    monkeypatch.setattr(harbor,'run_with_progress',run)
    monkeypatch.setattr(harbor,'_annotate_result_file',lambda path:None)
    assert harbor.run(config)==0
    assert captured['env']['PYTHONPATH'].endswith(':/existing/path')
    root=config.jobs_dir/'owned_networks'
    assert not list(root.glob('closed-*.json'))
    captured['on_process_settled']()
    marker=json.loads(next(root.glob('closed-*.json')).read_text())
    assert marker['owner']=='goal12-fixture' and marker['process_group_settled']
    with pytest.raises(ValueError):harbor._owned_job_closed_callback(config)


@pytest.mark.parametrize('alive',[True,False])
def test_progress_closure_requires_absent_group(monkeypatch,alive):
    from llm_module.agentic import progress
    process=SimpleNamespace(wait=lambda **kw:0,poll=lambda:0)
    monkeypatch.setattr(progress.subprocess,'Popen',lambda *a,**k:process)
    monkeypatch.setattr(progress,'_process_group_exists',lambda p:alive)
    callbacks=[]
    result=progress.run_with_progress(['fixture'],cwd=None,env={},probe=lambda:None,label='fixture',
        per_task_budget_s=10,concurrency=1,startup_grace_s=10,stall_grace_s=10,log_interval_s=.1,
        on_process_settled=lambda:callbacks.append('settled'))
    assert result==0 and bool(callbacks)is(not alive)


def test_observed_p04t05_network_and_route_exclusions():
    from pathlib import Path
    fixture=json.loads((Path(__file__).parent/'fixtures/owned_network_p04t05_exclusions.json').read_text())
    excluded=[ip.ip_network(x,strict=False)for x in fixture['excluded']]
    selected=list(candidates(excluded,'observed-inventory-fixture'))
    assert len(selected)>=32
    assert all(not candidate.overlaps(used)for candidate in selected for used in excluded)
    assert any(str(n)=='172.22.0.0/16'for n in excluded)
    assert any(n.subnet_of(ip.ip_network('10.42.0.0/16'))for n in excluded)


def test_lost_create_response_reconciles_exact_owned_identity(tmp_path):
    import subprocess
    daemon=FakeDaemon()
    class LostResponseAllocator(TestAllocator):
        def command(self,args,check=True):
            result=super().command(args,check)
            if args[:3]==['docker','network','create']:
                raise subprocess.TimeoutExpired(args,30)
            return result
    original=allocator(tmp_path,daemon,0)
    a=LostResponseAllocator(original.labels,original.project,tmp_path/'lost.json','localhost',daemon.run)
    with pytest.raises(subprocess.TimeoutExpired):a.acquire()
    assert a.record['state']=='created' and len(a.record['network_id'])==64
    a.remove_empty()
    assert not daemon.networks


def test_unknown_create_and_foreign_same_name_never_claim_removed(tmp_path):
    daemon=FakeDaemon();a=allocator(tmp_path,daemon,0)
    a.record.update(state='needs_reconciliation',subnet='10.0.0.0/28',gateway='10.0.0.1')
    with pytest.raises(OwnershipError):a.remove_empty()
    assert a.record['state']=='needs_reconciliation' and a.record['network_id']is None
    foreign=allocator(tmp_path,daemon,1);foreign.project=a.project;foreign.record['network_name']=a.record['network_name'];foreign.acquire()
    with pytest.raises(OwnershipError):a.reconcile_created_identity()
    assert a.record['network_id']is None and len(daemon.networks)==1


def test_disappeared_network_rescans_whole_snapshot_bounded(tmp_path,monkeypatch):
    from agentic_owned_networks.allocator import InventoryChanged
    a=Allocator({OWNER:'owner',MANAGED:POLICY,ATTEMPT:'a',JOB:'j'},'p',tmp_path/'a.json','localhost')
    calls=[]
    def snapshot(skip):
        calls.append(skip)
        if len(calls)==1:raise InventoryChanged('Confirmed disappearance')
        return [ip.ip_network('10.0.0.0/8')]
    monkeypatch.setattr(a,'_inventory_snapshot',snapshot)
    assert a.inventory()==[ip.ip_network('10.0.0.0/8')]and len(calls)==2
    calls.clear()
    def unstable(skip):calls.append(skip);raise InventoryChanged('Confirmed disappearance')
    monkeypatch.setattr(a,'_inventory_snapshot',unstable)
    with pytest.raises(InventoryChanged):a.inventory()
    assert len(calls)==3
    calls.clear()
    def invalid(skip):calls.append(skip);raise ValueError('Incomplete routes')
    monkeypatch.setattr(a,'_inventory_snapshot',invalid)
    with pytest.raises(ValueError,match='Incomplete routes'):a.inventory()
    assert len(calls)==1


def test_missing_network_is_distinguished_from_inspection_failure(tmp_path):
    from agentic_owned_networks.allocator import InventoryChanged
    a=Allocator({OWNER:'owner',MANAGED:POLICY,ATTEMPT:'a',JOB:'j'},'p',tmp_path/'a.json','localhost')
    a.command=lambda args:SimpleNamespace(stdout='a'*64)
    a.inspect=lambda ident:None
    with pytest.raises(InventoryChanged):a._inventory_snapshot()
    def unavailable(ident):raise RuntimeError('inspection unavailable')
    a.inspect=unavailable
    with pytest.raises(RuntimeError):a.inventory()
