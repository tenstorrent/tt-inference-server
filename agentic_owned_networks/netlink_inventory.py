"""Read-only IPv4 NETLINK_ROUTE inventory, usable as an embedded stdlib script."""
import ipaddress
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time


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


def route(payload):
    if len(payload)<12:raise ValueError('Truncated route message')
    family,prefix,src_prefix,tos,table,protocol,scope,kind,flags=struct.unpack_from('=BBBBBBBBI',payload)
    if family!=socket.AF_INET or prefix>32:raise ValueError('Unexpected route family/prefix')
    attrs=attributes(payload[12:])
    dest=address(attrs[1])if 1 in attrs else '0.0.0.0'
    if prefix and 1 not in attrs:raise ValueError('Missing route destination')
    item={'dst':str(ipaddress.ip_network(dest+'/'+str(prefix),strict=False))if prefix else'default',
          'table':uint(attrs[15])if 15 in attrs else table,'type':kind}
    if 4 in attrs:item['dev']=socket.if_indextoname(uint(attrs[4]))
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
            if index:hop['dev']=socket.if_indextoname(index)
            if 5 in hopattrs:hop['gateway']=address(hopattrs[5])
            if 18 in hopattrs:raise ValueError('Cross-family multipath hop unsupported')
            hops.append(hop);offset+=(size+3)&~3
        item['nexthops']=hops
    return item


def interface_address(payload):
    if len(payload)<8:raise ValueError('Truncated address message')
    family,prefix,flags,scope,index=struct.unpack_from('=BBBBI',payload)
    if family!=socket.AF_INET or prefix>32:raise ValueError('Unexpected address family/prefix')
    attrs=attributes(payload[8:]);local=attrs.get(2,attrs.get(1))
    if local is None:raise ValueError('Missing interface address')
    infos=[{'family':'inet','local':address(local),'prefixlen':prefix}]
    if 1 in attrs and attrs[1]!=local:infos.append({'family':'inet','local':address(attrs[1]),'prefixlen':32})
    return {'ifname':socket.if_indextoname(index),'addr_info':infos}


def dump(request_type,response_type,body):
    deadline=time.monotonic()+5;sequence=1;total=0;items=[]
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
                if flags&0x10:raise ValueError('Interrupted netlink dump')
                payload=data[offset+16:offset+length]
                if kind==2:raise ValueError('Netlink error response')
                if kind==3:
                    if len(payload)<4 or struct.unpack_from('=i',payload)[0]!=0 or offset+((length+3)&~3)!=len(data):raise ValueError('Netlink dump completion error')
                    return items
                if kind!=response_type or not(flags&2):raise ValueError('Unexpected nonmultipart netlink response')
                items.append(payload)
                if len(items)>32768:raise ValueError('Too many netlink records')
                offset+=(length+3)&~3


def collect(model_host):
    routes=[route(p)for p in dump(26,24,struct.pack('=BBBBBBBBI',socket.AF_INET,0,0,0,0,0,0,0,0))]
    addresses=[interface_address(p)for p in dump(22,20,struct.pack('=BBBBI',socket.AF_INET,0,0,0,0))]
    dns=[]
    for line in Path('/etc/resolv.conf').read_text().splitlines():
        parts=line.split()
        if parts and parts[0]=='nameserver':dns.append(parts[1])
    resolved=sorted({row[4][0]for row in socket.getaddrinfo(model_host,None,socket.AF_INET)})
    if not routes or not addresses or not resolved:raise ValueError('Incomplete host network inventory')
    namespace=os.stat('/proc/self/ns/net')
    return {'routes':routes,'addresses':addresses,'dns':dns,'model_addresses':resolved,
            'network_namespace':{'device':namespace.st_dev,'inode':namespace.st_ino}}


if __name__=='__main__':
    print(json.dumps(collect(sys.argv[1])))
