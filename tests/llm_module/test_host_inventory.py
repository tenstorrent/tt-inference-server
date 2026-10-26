import json
from pathlib import Path
import socket
import struct
from types import SimpleNamespace

import pytest
from agentic_owned_networks import host_inventory as provider, netlink_inventory as nl
from agentic_owned_networks.allocator import OWNER, MANAGED, JOB, ATTEMPT, OwnershipError


def attr(kind, data):
    raw=struct.pack('=HH',len(data)+4,kind)+data
    return raw+b'\0'*((-len(raw))%4)


def test_routes_all_tables_multipath_and_peer_addresses(monkeypatch):
    monkeypatch.setattr(nl.socket,'if_indextoname',lambda index:'eth'+str(index))
    hop=struct.pack('=HBBi',16,0,0,2)+attr(5,socket.inet_aton('10.4.0.1'))
    route=struct.pack('=BBBBBBBBI',2,24,0,0,0,0,0,1,0)+attr(1,socket.inet_aton('10.9.0.0'))+attr(15,struct.pack('=I',1234))+attr(9,hop)
    parsed=nl.route(route)
    assert parsed['table']==1234 and parsed['dst']=='10.9.0.0/24'
    assert parsed['nexthops']==[{'dev':'eth2','gateway':'10.4.0.1'}]
    address=struct.pack('=BBBBI',2,24,0,0,2)+attr(2,socket.inet_aton('10.9.0.2'))+attr(1,socket.inet_aton('10.9.0.3'))
    assert len(nl.interface_address(address)['addr_info'])==2
    with pytest.raises(ValueError):nl.attributes(b'\xff\xff\x01\x00')


@pytest.mark.parametrize('fault',[None,'interrupted','wrong_sequence','truncated','error'])
def test_netlink_dump_requires_complete_matching_kernel_reply(monkeypatch,fault):
    body=b'payload'
    flags=2|(16 if fault=='interrupted'else 0)
    data=struct.pack('=IHHII',16+len(body),2 if fault=='error'else 24,flags,2 if fault=='wrong_sequence'else 1,123)+body+b'\0'
    done=struct.pack('=IHHIIi',20,3,2,1,123,0)
    class Channel:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def bind(self,address):pass
        def getsockname(self):return (123,0)
        def sendto(self,*args):pass
        def settimeout(self,*args):pass
        def recvmsg(self,size):
            response=self.responses.pop(0)
            return response,[],socket.MSG_TRUNC if fault=='truncated'else 0,(0,0)
        responses=[data,done]
    monkeypatch.setattr(nl.socket,'socket',lambda *args:Channel())
    if fault:
        with pytest.raises(ValueError):nl.dump(26,24,b'')
    else:assert nl.dump(26,24,b'')==[body]


class FakeAllocator:
    def __init__(self,path,escape=None):
        self.labels={OWNER:'fixture',MANAGED:'isolated-harbor-v1',JOB:'job',ATTEMPT:'outer'}
        self.record={'daemon_id':'daemon'};self.journal=path/'outer.json';self.endpoint_host='127.0.0.1'
        self.calls=[];self.helper=None;self.escape=escape
        self.server={'id':'a'*64,'image':'sha256:'+'c'*64,'name':'/tt-inference-server-owned','running':True,'labels':{OWNER:'fixture'}}
    def local_daemon(self):pass
    def save(self):pass
    def data(self,args):return None
    def command(self,args,check=True):
        self.calls.append(args);rc=0;out='';err=''
        if args[:2]==['docker','ps']:out=self.server['id']
        elif args[:2]==['docker','inspect']:
            item=self.server if args[-1]==self.server['id']else self.helper
            if item:out=json.dumps(item)
            else:rc=1;err='No such object'
        elif args[:2]==['docker','create']:
            labels={args[i+1].split('=',1)[0]:args[i+1].split('=',1)[1]for i,x in enumerate(args)if x=='--label'}
            assert '--read-only'in args and '--cap-drop'in args and '--security-opt'in args and '--device'not in args and '--mount'not in args
            self.helper={'id':'b'*64,'name':'/'+args[args.index('--name')+1],'image':self.server['image'],'labels':labels,'running':False,'network_mode':'host','privileged':False,'readonly':True,'cap_drop':['ALL'],'security_opt':['no-new-privileges'],'devices':[],'device_requests':[],'mounts':[]}
            if self.escape:self.helper[self.escape]=True if self.escape=='privileged'else'bridge'
            out=self.helper['id']
        elif args[:2]==['docker','start']:
            out=json.dumps({'routes':[{'dst':'10.1.0.0/16'}],'addresses':[{'ifname':'eth0','addr_info':[]}],'dns':['127.0.0.53'],'model_addresses':['127.0.0.1']})
        elif args[:2]==['docker','rm']:self.helper=None
        else:raise AssertionError(args)
        return SimpleNamespace(stdout=out,stderr=err,returncode=rc)


def test_owned_host_helper_returns_inventory_only_after_cleanup(tmp_path):
    a=FakeAllocator(tmp_path)
    result=provider.collect(a)
    assert result['routes'][0]['dst']=='10.1.0.0/16' and a.helper is None
    journal=json.loads(next((tmp_path/'inventory_helpers').glob('*.json')).read_text())
    assert journal['state']=='removed' and journal['container_id']=='b'*64
    assert a.record['inventory_provider']['cleanup_verified']


@pytest.mark.parametrize('escape',['privileged','network_mode'])
def test_changed_host_helper_rejected_without_unsafe_cleanup(tmp_path,escape):
    a=FakeAllocator(tmp_path,escape)
    with pytest.raises(OwnershipError):provider.collect(a)
    assert not any(c[:2]in (['docker','start'],['docker','rm'])for c in a.calls)
    assert a.helper is not None


def test_allocator_unions_host_provider_and_collector_without_pid1_assumption(tmp_path,monkeypatch):
    from agentic_owned_networks.allocator import Allocator
    labels={OWNER:'fixture',MANAGED:'isolated-harbor-v1',JOB:'job',ATTEMPT:'outer'}
    a=Allocator(labels,'test',tmp_path/'allocation.json','model')
    a.command=lambda args:SimpleNamespace(stdout='')
    a.data=lambda args:{'routes':[{'dst':'10.1.0.0/16'}],'addresses':[{'ifname':'eth0','addr_info':[{'family':'inet','local':'10.2.0.1','prefixlen':24}]}],'dns':['10.3.0.1'],'model_addresses':['10.4.0.1']}
    monkeypatch.setattr(provider,'collect',lambda a:{'routes':[{'dst':'172.20.0.0/16'}],'addresses':[{'ifname':'ens1','addr_info':[{'family':'inet','local':'192.168.2.1','prefixlen':24}]}],'dns':['192.168.3.1'],'model_addresses':['192.168.4.1']})
    excluded={str(n)for n in a.inventory()}
    assert {'10.1.0.0/16','10.2.0.0/24','10.3.0.1/32','10.4.0.1/32','172.20.0.0/16','192.168.2.0/24','192.168.3.1/32','192.168.4.1/32'}<=excluded


def test_server_inventory_filters_out_disappearing_helpers(tmp_path):
    class Interleaved(FakeAllocator):
        def command(self,args,check=True):
            if args[:2]==['docker','ps']:
                # A helper is listed only if the caller omitted the server filter.
                if 'name=tt-inference-server-'not in args:
                    return SimpleNamespace(stdout=self.server['id']+'\n'+'d'*64,stderr='',returncode=0)
            if args[:2]==['docker','inspect']and args[-1]=='d'*64:
                return SimpleNamespace(stdout='',stderr='No such object',returncode=1)
            return super().command(args,check)
    a=Interleaved(tmp_path)
    assert provider.collect(a)['routes']
    assert a.helper is None
