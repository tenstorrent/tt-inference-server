"""Synthetic read-only inventory recovery. Never touches Docker or netlink."""
import copy
import errno
import json
import socket
import struct
from types import SimpleNamespace

import pytest
from agentic_owned_networks import netlink_inventory as nl, host_inventory as provider
from agentic_owned_networks.allocator import Allocator, InventoryChanged, OwnershipError, inventory_result, error_record, OWNER, MANAGED, ATTEMPT, JOB
from tests.llm_module.test_host_inventory import FakeAllocator, attr


def route(index=2,extra=b''):
    return struct.pack('=BBBBBBBBI',2,24,0,0,0,0,0,1,0)+attr(1,socket.inet_aton('10.9.0.0'))+attr(4,struct.pack('=I',index))+extra


def address(index=2):
    return struct.pack('=BBBBI',2,24,0,0,index)+attr(2,socket.inet_aton('10.9.0.2'))


def multipath(index=2,extra=b''):
    hop=struct.pack('=HBBi',8+len(extra),0,0,index)+extra
    return struct.pack('=BBBBBBBBI',2,0,0,0,0,0,0,1,0)+attr(9,hop)


def envelope(reason='dump_interrupted'):
    return {'schema_version':1,'status':'invalidated','diagnostic':{'phase':'netlink_dump'if reason=='dump_interrupted'else'interface_lookup','reason':reason,'exception_type':'InventoryInvalidated','errno':None if reason=='dump_interrupted'else errno.ENXIO}}


def response(value,rc=0):return SimpleNamespace(stdout=json.dumps(value),stderr='secret stderr never copied',returncode=rc)


def inventory(prefix='10.1.0.0/16'):
    return {'routes':[{'dst':prefix}],'addresses':[{'ifname':'eth0','addr_info':[]}],'dns':[],'model_addresses':['127.0.0.1']}


def success(prefix='10.1.0.0/16'):return {'schema_version':1,'status':'ok','inventory':inventory(prefix)}


def message(kind,payload,flags=2,seq=1,pid=123):
    raw=struct.pack('=IHHII',16+len(payload),kind,flags,seq,pid)+payload
    return raw+b'\0'*((-len(raw))%4)


def channel(monkeypatch,packets,sender=0,recv_flags=0):
    class Channel:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def bind(self,address):pass
        def getsockname(self):return (123,0)
        def sendto(self,*args):pass
        def settimeout(self,*args):pass
        def recvmsg(self,size):return packets.pop(0),[],recv_flags,(sender,0)
    monkeypatch.setattr(nl.socket,'socket',lambda *args:Channel())


@pytest.mark.parametrize('where',['data','done'])
def test_valid_dump_interruption_is_explicit_and_discards_data(monkeypatch,where):
    packets=[message(24,route(),18 if where=='data'else 2),message(3,struct.pack('=i',0),18 if where=='done'else 2)]
    channel(monkeypatch,packets)
    with pytest.raises(nl.InventoryInvalidated,match='dump_interrupted'):
        nl.dump(26,24,b'',lambda p:nl.route(p,False))
    assert not packets


@pytest.mark.parametrize('fault',['badpayload','unsupported','sequence','sender','truncated','wrongtype','nonmultipart','negative_done','short_done'])
def test_malformed_or_unsupported_flagged_dump_is_fatal(monkeypatch,fault):
    payload=route();kind=24;flags=18;seq=1;sender=0;recv=0;done=struct.pack('=i',0)
    if fault=='badpayload':payload=b'bad'
    if fault=='unsupported':payload=route(extra=attr(18,b'unsupported'))
    if fault=='sequence':seq=2
    if fault=='sender':sender=12
    if fault=='truncated':recv=socket.MSG_TRUNC
    if fault=='wrongtype':kind=20
    if fault=='nonmultipart':flags=16
    if fault=='negative_done':done=struct.pack('=i',-errno.EINTR)
    if fault=='short_done':done=b''
    channel(monkeypatch,[message(kind,payload,flags,seq),message(3,done)],sender,recv)
    with pytest.raises(ValueError)as raised:nl.dump(26,24,b'',lambda p:nl.route(p,False))
    assert not isinstance(raised.value,nl.InventoryInvalidated)


@pytest.mark.parametrize('payload,parser',[(route(),nl.route),(multipath(),nl.route),(address(),nl.interface_address)])
@pytest.mark.parametrize('number',[errno.ENXIO,errno.ENODEV])
def test_only_disappeared_valid_interfaces_are_retryable(monkeypatch,payload,parser,number):
    def gone(index):raise OSError(number,'private detail')
    monkeypatch.setattr(nl.socket,'if_indextoname',gone)
    with pytest.raises(nl.InventoryInvalidated)as raised:parser(payload)
    assert raised.value.reason=='interface_disappeared'and raised.value.errno==number


@pytest.mark.parametrize('number',[None,errno.EPERM,errno.EACCES,errno.EINVAL,errno.EMFILE,errno.ENOBUFS])
def test_other_lookup_errors_remain_fatal(monkeypatch,number):
    def failure(index):raise OSError(number,'private detail')
    monkeypatch.setattr(nl.socket,'if_indextoname',failure)
    with pytest.raises(OSError):nl.route(route())


@pytest.mark.parametrize('payload,parser',[(route(0),nl.route),(route(0xffffffff),nl.route),(multipath(-1),nl.route),(address(0),nl.interface_address),(route(extra=attr(18,b'bad')),nl.route),(multipath(2,attr(18,b'bad')),nl.route)])
def test_static_invalidity_precedes_lookup(monkeypatch,payload,parser):
    def forbidden(index):raise AssertionError('must validate before lookup')
    monkeypatch.setattr(nl.socket,'if_indextoname',forbidden)
    with pytest.raises(ValueError):parser(payload)
    assert nl.route(multipath(0))['nexthops']==[{}]


@pytest.mark.parametrize('reason',['dump_interrupted','interface_disappeared'])
def test_strict_invalidated_protocol(reason):
    with pytest.raises(InventoryChanged)as raised:inventory_result(response(envelope(reason)),'local_inventory')
    assert error_record(raised.value)['reason']==reason
    assert 'secret'not in json.dumps(error_record(raised.value))


@pytest.mark.parametrize('change',['version','status','reason','phase','errno','extra','mixed','nonzero','wrongtype'])
def test_unrecognized_protocol_never_requests_retry(change):
    value=envelope();rc=0
    if change=='version':value['schema_version']=2
    if change=='status':value['status']='unknown'
    if change=='reason':value['diagnostic']['reason']='unknown'
    if change=='phase':value['diagnostic']['phase']='unknown'
    if change=='errno':value['diagnostic']['errno']=errno.EPERM
    if change=='extra':value['diagnostic']['stderr']='private'
    if change=='mixed':value['inventory']=inventory()
    if change=='nonzero':rc=1
    if change=='wrongtype':value['diagnostic']['exception_type']='ValueError'
    with pytest.raises(RuntimeError):inventory_result(response(value,rc),'local_inventory')


def test_fatal_diagnostics_survive_nonzero_without_raw_details(monkeypatch):
    def fail(host):raise PermissionError(errno.EACCES,'credential-hidden')
    monkeypatch.setattr(nl,'collect',fail)
    value,code=nl.result('private-host')
    assert code==1 and value['status']=='error'
    with pytest.raises(RuntimeError)as raised:inventory_result(response(value,code),'local_inventory')
    record=error_record(raised.value)
    assert record['returncode']==1 and record['errno']==errno.EACCES and record['collector_exception_type']=='PermissionError'
    assert all(token not in json.dumps(record)for token in ('credential','private','secret'))


def test_full_combined_snapshot_restarts_all_sources_and_preserves_skip(tmp_path,monkeypatch):
    labels={OWNER:'owner',MANAGED:'isolated-harbor-v1',ATTEMPT:'attempt',JOB:'job'}
    a=Allocator(labels,'p',tmp_path/'journal.json','localhost');calls=[];attempt=[0]
    def command(args,check=True):
        if args[:3]==['docker','network','ls']:
            attempt[0]+=1;calls.append(('docker',attempt[0]));return SimpleNamespace(stdout='skip other')
        calls.append(('local',attempt[0]))
        return response(envelope())if attempt[0]==1 else response(success('10.2.0.0/16'))
    a.command=command
    a.inspect=lambda ident:{'IPAM':{'Config':[{'Subnet':f'172.{attempt[0]}.0.0/16'}]}}
    def host(allocator):calls.append(('provider',attempt[0]));return inventory(f'192.168.{attempt[0]}.0/24')
    monkeypatch.setattr(provider,'collect',host)
    output={str(n)for n in a.inventory(skip='skip')}
    assert calls==[(source,attempt)for attempt in (1,2)for source in ('docker','provider','local')]
    assert {'172.2.0.0/16','192.168.2.0/24','10.2.0.0/16'}<=output
    assert not {'172.1.0.0/16','192.168.1.0/24'}&output
    assert a.record['inventory_invalidations'][0]['skip_id']=='skip'


@pytest.mark.parametrize('cleanup_failure',[False,True])
def test_helper_invalidation_waits_for_verified_cleanup(tmp_path,cleanup_failure):
    class Invalidated(FakeAllocator):
        def command(self,args,check=True):
            result=super().command(args,check)
            if args[:2]==['docker','start']:return response(envelope())
            if cleanup_failure and args[:2]==['docker','inspect']and result.returncode:
                result.stderr='permission denied'
            return result
    a=Invalidated(tmp_path)
    with pytest.raises(OwnershipError if cleanup_failure else InventoryChanged):provider.collect(a)
    assert a.helper is None
    journal=json.loads(next((tmp_path/'inventory_helpers').glob('*.json')).read_text())
    assert journal['state']==('cleanup_unconfirmed'if cleanup_failure else'removed')
    assert journal['failure']['reason']=='dump_interrupted'


def test_address_dump_interruption_discards_prior_routes_and_reopens_all_sockets(monkeypatch):
    batches=[
        [message(24,route()),message(3,struct.pack('=i',0))],
        [message(20,address()),message(3,struct.pack('=i',0),18)],
        [message(24,route().replace(socket.inet_aton('10.9.0.0'),socket.inet_aton('10.10.0.0'))),message(3,struct.pack('=i',0))],
        [message(20,address()),message(3,struct.pack('=i',0))],
    ]; opened=[]
    class Channel:
        def __init__(self):self.packets=batches[len(opened)];opened.append(self)
        def __enter__(self):return self
        def __exit__(self,*args):self.closed=True
        def bind(self,address):pass
        def getsockname(self):return (123,0)
        def sendto(self,*args):pass
        def settimeout(self,*args):pass
        def recvmsg(self,size):return self.packets.pop(0),[],0,(0,0)
    monkeypatch.setattr(nl.socket,'socket',lambda *args:Channel())
    monkeypatch.setattr(nl.socket,'if_indextoname',lambda index:'eth'+str(index))
    monkeypatch.setattr(nl.socket,'getaddrinfo',lambda *args:[(None,None,None,None,('127.0.0.1',0))])
    first,code=nl.result('localhost')
    assert code==0 and first['status']=='invalidated' and 'inventory'not in first
    second,code=nl.result('localhost')
    assert code==0 and second['status']=='ok'
    assert second['inventory']['routes'][0]['dst']=='10.10.0.0/24'
    assert len(opened)==4 and all(c.closed and not c.packets for c in opened)


def test_duplicate_envelope_keys_are_not_accepted():
    r=response(envelope());r.stdout=r.stdout.replace('"status": "invalidated"','"status": "error", "status": "invalidated"')
    with pytest.raises(RuntimeError,match='invalid_json'):inventory_result(r,'local_inventory')


@pytest.mark.parametrize('reason',['dump_interrupted','interface_disappeared'])
def test_invalidation_exhausts_three_whole_snapshots_with_original_skip(tmp_path,monkeypatch,reason):
    labels={OWNER:'owner',MANAGED:'isolated-harbor-v1',ATTEMPT:'attempt',JOB:'job'}
    a=Allocator(labels,'p',tmp_path/'journal.json','localhost');calls=[]
    def snapshot(skip):
        calls.append(skip)
        return inventory_result(response(envelope(reason)),'local_inventory')
    a._inventory_snapshot=snapshot
    with pytest.raises(InventoryChanged):a.inventory(skip='known-created-id')
    assert calls==['known-created-id']*3
    assert len(a.record['inventory_invalidations'])==3


def test_fatal_collector_result_is_not_retried(tmp_path):
    labels={OWNER:'owner',MANAGED:'isolated-harbor-v1',ATTEMPT:'attempt',JOB:'job'}
    a=Allocator(labels,'p',tmp_path/'journal.json','localhost');calls=[]
    value=envelope();value['status']='error';value['diagnostic'].update(reason='collector_failure',exception_type='PermissionError',errno=errno.EACCES)
    def snapshot(skip):
        calls.append(skip);return inventory_result(response(value,1),'local_inventory')
    a._inventory_snapshot=snapshot
    with pytest.raises(RuntimeError):a.inventory()
    assert calls==[None] and not a.record.get('inventory_invalidations')
