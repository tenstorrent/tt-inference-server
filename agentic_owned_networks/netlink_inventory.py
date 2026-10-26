"""Read-only IPv4 NETLINK_ROUTE inventory, usable as an embedded stdlib script."""
import errno
import ipaddress
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time


class InventoryInvalidated(ValueError):
    """Only known snapshot invalidation; never usable partial inventory."""
    def __init__(self, reason, phase, error_number=None):
        super().__init__('Network inventory invalidated: ' + reason)
        self.reason=reason;self.collector_phase=phase;self.errno=error_number


def interface_index(index, allow_zero=False):
    if type(index)is not int or not (0 if allow_zero else 1)<=index<=0x7fffffff:
        raise ValueError('Invalid interface index')
    return index


def interface_name(index):
    interface_index(index)
    try:return socket.if_indextoname(index)
    except OSError as error:
        if error.errno in (errno.ENXIO,errno.ENODEV):
            raise InventoryInvalidated('interface_disappeared','interface_lookup',error.errno)from error
        raise


def resolve_route(item):
    if '_ifindex'in item:item['dev']=interface_name(item.pop('_ifindex'))
    for hop in item.get('nexthops',[]):
        if '_ifindex'in hop:hop['dev']=interface_name(hop.pop('_ifindex'))
    return item


def attributes(data):
    result={};offset=0
    while offset<len(data):
        if len(data)-offset<4:raise ValueError('Truncated route attribute header')
        size,kind=struct.unpack_from('=HH',data,offset)
        if size<4 or offset+size>len(data):raise ValueError('Invalid route attribute length')
        kind &= 0x3fff
        if kind in result:raise ValueError('Duplicate route attribute')
        result[kind]=data[offset+4:offset+size]
        offset+=(size+3)&~3
    return result


def uint(data):
    if len(data)!=4:raise ValueError('Invalid uint route attribute')
    return struct.unpack('=I',data)[0]


def address(data):
    if len(data)!=4:raise ValueError('Invalid IPv4 route attribute')
    return str(ipaddress.IPv4Address(data))


def route(payload, resolve_names=True):
    if len(payload)<12:raise ValueError('Truncated route message')
    family,prefix,src_prefix,tos,table,protocol,scope,kind,flags=struct.unpack_from('=BBBBBBBBI',payload)
    if family!=socket.AF_INET or prefix>32:raise ValueError('Unexpected route family/prefix')
    attrs=attributes(payload[12:])
    dest=address(attrs[1])if 1 in attrs else '0.0.0.0'
    if prefix and 1 not in attrs:raise ValueError('Missing route destination')
    item={'dst':str(ipaddress.ip_network(dest+'/'+str(prefix),strict=False))if prefix else'default',
          'table':uint(attrs[15])if 15 in attrs else table,'type':kind}
    if 4 in attrs:item['_ifindex']=interface_index(uint(attrs[4]))
    if 5 in attrs:item['gateway']=address(attrs[5])
    if 18 in attrs:
        raise ValueError('Cross-family RTA_VIA routes unsupported; inventory incomplete')
    if 9 in attrs:
        hops=[];data=attrs[9];offset=0
        while offset<len(data):
            if len(data)-offset<8:raise ValueError('Truncated multipath hop')
            size,hop_flags,weight,index=struct.unpack_from('=HBBi',data,offset)
            if size<8 or offset+size>len(data):raise ValueError('Invalid multipath hop length')
            hopattrs=attributes(data[offset+8:offset+size]);hop={}
            interface_index(index,allow_zero=True)
            if index:hop['_ifindex']=index
            if 5 in hopattrs:hop['gateway']=address(hopattrs[5])
            if 18 in hopattrs:raise ValueError('Cross-family multipath hop unsupported')
            hops.append(hop);offset+=(size+3)&~3
        item['nexthops']=hops
    return resolve_route(item)if resolve_names else item


def interface_address(payload, resolve_names=True):
    if len(payload)<8:raise ValueError('Truncated address message')
    family,prefix,flags,scope,index=struct.unpack_from('=BBBBI',payload)
    if family!=socket.AF_INET or prefix>32:raise ValueError('Unexpected address family/prefix')
    attrs=attributes(payload[8:]);local=attrs.get(2,attrs.get(1))
    if local is None:raise ValueError('Missing interface address')
    infos=[{'family':'inet','local':address(local),'prefixlen':prefix}]
    if 1 in attrs and attrs[1]!=local:infos.append({'family':'inet','local':address(attrs[1]),'prefixlen':32})
    interface_index(index)
    return {'ifname':interface_name(index),'addr_info':infos}if resolve_names else {'_ifindex':index,'addr_info':infos}


def dump(request_type,response_type,body,validate=None):
    deadline=time.monotonic()+5;sequence=1;total=0;items=[];interrupted=False
    with socket.socket(socket.AF_NETLINK,socket.SOCK_RAW,socket.NETLINK_ROUTE)as channel:
        channel.bind((0,0));port=channel.getsockname()[0]
        packet=struct.pack('=IHHII',16+len(body),request_type,0x301,sequence,port)+body
        channel.sendto(packet,(0,0))
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Netlink dump deadline')
            channel.settimeout(remaining)
            data,ancillary,recv_flags,sender=channel.recvmsg(65536)
            total+=len(data)
            if sender[0]!=0 or recv_flags&socket.MSG_TRUNC or total>1048576:
                raise ValueError('Untrusted/truncated/oversize netlink dump')
            offset=0
            while offset<len(data):
                if len(data)-offset<16:raise ValueError('Truncated netlink header')
                length,kind,flags,seq,pid=struct.unpack_from('=IHHII',data,offset)
                if length<16 or offset+length>len(data)or seq!=sequence or pid!=port:
                    raise ValueError('Invalid netlink message identity/length')
                payload=data[offset+16:offset+length]
                if kind==2:raise ValueError('Netlink error response')
                if kind==3:
                    if len(payload)<4 or struct.unpack_from('=i',payload)[0]!=0 or offset+((length+3)&~3)!=len(data):raise ValueError('Netlink dump completion error')
                    if interrupted or flags&0x10:
                        raise InventoryInvalidated('dump_interrupted','netlink_dump')
                    return items
                if kind!=response_type or not(flags&2):raise ValueError('Unexpected nonmultipart netlink response')
                if validate is not None:validate(payload)
                interrupted=interrupted or bool(flags&0x10)
                items.append(payload)
                if len(items)>32768:raise ValueError('Too many netlink records')
                offset+=(length+3)&~3


def phased(phase,function,*args,**kwargs):
    try:return function(*args,**kwargs)
    except Exception as error:
        if not hasattr(error,'collector_phase'):error.collector_phase=phase
        raise


def collect(model_host):
    raw_routes=phased('routes',dump,26,24,struct.pack('=BBBBBBBBI',socket.AF_INET,0,0,0,0,0,0,0,0),
                      validate=lambda p:route(p,False))
    raw_addresses=phased('addresses',dump,22,20,struct.pack('=BBBBI',socket.AF_INET,0,0,0,0),
                         validate=lambda p:interface_address(p,False))
    # Complete static validation before any live interface-name lookup. A known
    # unsupported/malformed record must not be hidden by a vanished interface.
    routes=[phased('routes',route,p,False)for p in raw_routes]
    addresses=[phased('addresses',interface_address,p,False)for p in raw_addresses]
    for item in routes:phased('interface_lookup',resolve_route,item)
    for item in addresses:item['ifname']=phased('interface_lookup',interface_name,item.pop('_ifindex'))
    dns=[]
    for line in phased('resolver',Path('/etc/resolv.conf').read_text).splitlines():
        parts=line.split()
        if parts and parts[0]=='nameserver':dns.append(parts[1])
    resolved=sorted({row[4][0]for row in phased('resolver',socket.getaddrinfo,model_host,None,socket.AF_INET)})
    if not routes or not addresses or not resolved:raise ValueError('Incomplete host network inventory')
    namespace=phased('namespace',os.stat,'/proc/self/ns/net')
    return {'routes':routes,'addresses':addresses,'dns':dns,'model_addresses':resolved,
            'network_namespace':{'device':namespace.st_dev,'inode':namespace.st_ino}}


def result(model_host):
    try:return {'schema_version':1,'status':'ok','inventory':collect(model_host)},0
    except Exception as error:
        invalidated=isinstance(error,InventoryInvalidated)
        # No raw exception text, stderr, environment, endpoint or script body.
        diagnostic={'phase':getattr(error,'collector_phase','collect'),
                    'reason':error.reason if invalidated else 'collector_failure',
                    'exception_type':type(error).__name__,
                    'errno':getattr(error,'errno',None)}
        return {'schema_version':1,'status':'invalidated'if invalidated else 'error','diagnostic':diagnostic},0 if invalidated else 1


if __name__=='__main__':
    value,code=result(sys.argv[1])
    print(json.dumps(value));sys.exit(code)
