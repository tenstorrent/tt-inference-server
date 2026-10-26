"""Explicit per-attempt Docker bridges. No daemon mutation or global cleanup."""
import hashlib
import ipaddress as ip
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import time

OWNER = 'com.tenstorrent.agentic-owner'
MANAGED = 'com.tenstorrent.agentic-managed'
ATTEMPT = 'com.tenstorrent.agentic-attempt'
JOB = 'com.tenstorrent.agentic-job'
POLICY = 'isolated-harbor-v1'
RFC1918 = tuple(ip.ip_network(n) for n in ('10.0.0.0/8','172.16.0.0/12','192.168.0.0/16'))


class OwnershipError(ValueError):
    """Not RuntimeError: Harbor must not swallow failed ownership checks."""


def resource_absent(response, ident, resource):
    """Only a resource/identity-specific Docker not-found response proves absence."""
    prefix=r'(?:Error(?: response from daemon)?:\s*)?'
    identity=re.escape(ident)
    if resource=='container':body=r'No such (?:object|container):\s*'+identity
    elif resource=='network':body=r'(?:No such network:\s*'+identity+r'|network\s+'+identity+r'\s+not found)'
    else:raise ValueError('Unsupported absence resource')
    return bool(response.returncode and re.fullmatch(prefix+body,response.stderr.strip(),re.IGNORECASE))


def container_absent(response, ident):
    return resource_absent(response,ident,'container')


def inspection_failure(response, operation="docker.inspect"):
    error=OwnershipError('Container inspection unavailable; absence not established')
    error.operation=operation
    error.returncode=response.returncode
    message=response.stderr.lower()
    error.reason=('permission_denied'if 'permission denied'in message else
                  'daemon_unavailable'if 'cannot connect'in message or 'connection refused'in message else
                  'unrecognized_inspection_error')
    return error


def error_record(error):
    """Compact cause without stderr, argv, environment or arbitrary raw logs."""
    result={'type':type(error).__name__}
    if isinstance(error,(OwnershipError,ValueError,RuntimeError)):
        result['message']=str(error).replace('\n',' ')[:320]
    for key in ('operation','reason','returncode'):
        if hasattr(error,key):result[key]=getattr(error,key)
    return result


class InventoryChanged(ValueError):
    """A listed network vanished; discard the snapshot and rescan."""


def checked_owner(owner):
    if not isinstance(owner,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',owner):
        raise ValueError('Owned Docker mode requires a validated CI owner')
    return owner


def candidates(excluded, identity):
    """Subtract all occupied prefixes, then choose /28s; no is_private shortcut."""
    seed=int(hashlib.sha256(identity.encode()).hexdigest(),16)
    for parent in RFC1918:
        free=[parent]
        for used in excluded:
            if used.version!=4:continue
            next_free=[]
            for part in free:
                if not part.overlaps(used):next_free.append(part)
                elif used.subnet_of(part):next_free.extend(part.address_exclude(used))
            free=next_free
        for part in sorted(free,key=lambda n:(int(n.network_address),n.prefixlen)):
            if part.prefixlen>28:continue
            count=part.num_addresses//16;offset=seed%count
            for step in range(min(count,128)):
                yield ip.ip_network((int(part.network_address)+16*((offset+step)%count),28))


def topology(graph, project):
    services=graph.get('services')or{}
    if not services or len(services)>13:raise ValueError('Default bridge requires1..13 endpoints')
    nets=graph.get('networks')or{}
    if set(nets)-{'default'}:raise ValueError('Additional task networks unsupported')
    default=nets.get('default')or{}
    if set(default)-{'name','driver','ipam'} or default.get('name',project+'_default')!=project+'_default' or default.get('driver','bridge')!='bridge' or default.get('ipam'):
        raise ValueError('Authored/external/static network policy unsupported')
    endpoints=0
    for service in services.values():
        if any(service.get(k) for k in ('ports','network_mode','container_name','privileged','devices','external_links')):
            raise ValueError('Task topology bypasses trial isolation')
        if any(str(cap).upper() in ('NET_ADMIN','SYS_ADMIN','ALL') for cap in (service.get('cap_add')or[])):
            raise ValueError('Privileged network capabilities unsupported')
        networks=service.get('networks') or {'default':None}
        if isinstance(networks,list):networks={key:None for key in networks}
        if set(networks)!={'default'} or any(value for value in networks.values()):
            raise ValueError('Static addresses, network options or extra networks unsupported')
        replicas=(service.get('deploy')or{}).get('replicas',1)
        if type(replicas)is not int or replicas<1:raise ValueError('Invalid service replicas')
        endpoints+=replicas
    if endpoints>13:raise ValueError('/28 endpoint capacity exceeded')
    return list(services)


def validate_external_network(graph, name):
    """Compose may synthesize empty IPAM for an otherwise external network."""
    networks=graph.get('networks')
    if not isinstance(networks,dict)or set(networks)!={'default'}:
        raise OwnershipError('Merged external network contains unsupported attributes')
    network=networks['default']
    if not isinstance(network,dict):raise OwnershipError('Invalid merged external network')
    normalized=dict(network)
    if normalized.get('ipam')=={}:normalized.pop('ipam')
    if normalized.get('external')is not True or normalized!={'name':name,'external':True}:
        raise OwnershipError('Merged external network contains unsupported attributes')


class Allocator:
    def __init__(self, labels, project, journal, endpoint_host, run=subprocess.run):
        self.labels=labels;self.project=project;self.journal=Path(journal);self.endpoint_host=endpoint_host;self.run=run
        self.record={'schema_version':1,'owner':labels[OWNER],'job':labels[JOB],'attempt':labels[ATTEMPT],
                     'project':project,'network_name':project+'-net','state':'intent','network_id':None}
        self.journal.parent.mkdir(parents=True,exist_ok=True)
        self.save()

    def save(self):
        temporary=self.journal.with_suffix('.tmp');temporary.write_text(json.dumps(self.record,indent=2)+'\n');temporary.replace(self.journal)

    def command(self,args,check=True):
        result=self.run(args,capture_output=True,text=True,timeout=30)
        if check and result.returncode:raise RuntimeError('Owned network command failed: '+args[0]+' '+args[1])
        return result

    def data(self,args):return json.loads(self.command(args).stdout)

    def local_daemon(self):
        if os.environ.get('DOCKER_HOST','unix:///var/run/docker.sock')!='unix:///var/run/docker.sock':raise ValueError('Only local rootful Docker socket supported')
        if Path('/.dockerenv').exists()or Path('/run/.containerenv').exists():raise ValueError('Allocator must run on daemon host, not in a container')
        # Route inventory is proven through a separately owned host-network helper.
        # The collector's PID1 namespace is not treated as Docker's namespace.
        context=self.data(['docker','context','inspect','--format','{{json .Endpoints.docker.Host}}'])
        if context!='unix:///var/run/docker.sock':raise ValueError('Remote/alternate Docker context unsupported')
        info=self.data(['docker','info','--format','{"id":{{json .ID}},"os":{{json .OSType}},"security":{{json .SecurityOptions}}}'])
        if info['os']!='linux' or any('rootless'in str(x).lower() for x in info['security']):raise ValueError('Rootful Linux Docker required')
        if self.record.get('daemon_id')not in(None,info['id']):raise OwnershipError('Docker daemon identity changed')
        self.record['daemon_id']=info['id'];self.save()

    def inspect(self,ident):
        result=self.command(['docker','network','inspect',ident],check=False)
        if result.returncode:
            if resource_absent(result,ident,'network'):return None
            raise inspection_failure(result,'docker.network.inspect')
        objects=json.loads(result.stdout)
        if len(objects)!=1:raise OwnershipError('Ambiguous network identity')
        return objects[0]

    def inventory(self, skip=None):
        for retry in range(3):
            try:return self._inventory_snapshot(skip)
            except InventoryChanged:
                if retry==2:raise
        raise AssertionError('Unreachable inventory retry state')

    def _inventory_snapshot(self, skip=None):
        exclusions=[]
        def add(value):
            if not value:return
            network=ip.ip_network(value,strict=False)
            if network.version==4 and network.prefixlen:exclusions.append(network)
        ids=self.command(['docker','network','ls','-q','--no-trunc']).stdout.split()
        for ident in ids:
            network=self.inspect(ident)
            if network is None:raise InventoryChanged('Listed network disappeared; discard whole inventory')
            if ident==skip:continue
            for config in network.get('IPAM',{}).get('Config')or[]:add(config.get('Subnet'))
        bridge='br-'+skip[:12]if skip else None
        from .host_inventory import collect
        provider=collect(self)
        local=self.data([sys.executable,str(Path(__file__).with_name('netlink_inventory.py')),self.endpoint_host])
        routes=local['routes']+provider['routes']
        addresses=local['addresses']+provider['addresses']
        for address in provider['dns']+provider['model_addresses']:add(address)
        if not isinstance(routes,list)or not routes or not isinstance(addresses,list)or not addresses:raise ValueError('Incomplete host IP inventory')
        for route in routes:
            if bridge is not None and route.get('dev')==bridge:continue
            if route.get('dst')!='default':add(route.get('dst'))
            add(route.get('gateway'))
            for hop in route.get('nexthops',[]):add(hop.get('gateway'))
        for interface in addresses:
            if bridge is not None and interface.get('ifname')==bridge:continue
            for addr in interface.get('addr_info',[]):
                if addr.get('family')=='inet':add(str(addr['local'])+'/'+str(addr['prefixlen']))
        for value in local['dns']+local['model_addresses']:add(value)
        return exclusions

    def verify(self,network):
        if not network or network['Id']!=self.record['network_id'] or network['Name']!=self.record['network_name'] or network.get('Driver')!='bridge' or network.get('Scope')!='local':raise OwnershipError('Network identity mismatch')
        if any((network.get('Labels')or{}).get(k)!=v for k,v in self.labels.items()):raise OwnershipError('Network ownership mismatch')
        config=network.get('IPAM',{}).get('Config')or[]
        if len(config)!=1 or config[0].get('Subnet')!=self.record['subnet'] or config[0].get('Gateway')!=self.record['gateway']:raise OwnershipError('Network CIDR mismatch')
        if network.get('Internal') or network.get('Options'):raise OwnershipError('Unexpected network isolation options')

    def reconcile_created_identity(self):
        """Recover a lost create response by exact identity, never by name alone.

        subprocess.run has already settled its owned CLI. A missing network is
        still uncertain: a daemon-side operation may become visible later.
        """
        self.local_daemon()
        for _ in range(3):
            network=self.inspect(self.record['network_name'])
            if network is None:continue
            ident=network.get('Id','')
            if not re.fullmatch(r'[0-9a-f]{64}',ident):
                raise OwnershipError('Reconciliation did not return a full network ID')
            # Validate every ownership/geometry field before publishing identity.
            prior=self.record.get('network_id')
            self.record['network_id']=ident
            try:self.verify(network)
            except BaseException:
                self.record['network_id']=prior
                raise
            self.record.update(state='created',identity_basis='Exact-name reconciliation after settled CLI; daemon,4labels,CIDR,gateway verified')
            self.save()
            return
        raise OwnershipError('Create identity unresolved; retain journal and fail closed')

    def acquire(self):
        self.local_daemon()
        if self.inspect(self.record['network_name']) is not None:raise OwnershipError('Network name collision; never adopt')
        deadline=time.monotonic()+120
        for attempt in range(64):
            if time.monotonic()>=deadline:raise ValueError('Network allocation deadline exceeded')
            occupied=self.inventory()
            subnet=next(candidates(occupied,self.project+str(attempt)),None)
            if subnet is None:raise ValueError('No collision-free RFC1918 /28 available')
            self.record.update(subnet=str(subnet),gateway=str(subnet.network_address+1),state='intent');self.save()
            args=['docker','network','create','--driver','bridge','--subnet',str(subnet),'--gateway',self.record['gateway']]
            for key,value in self.labels.items():args+=['--label',key+'='+value]
            try:created=self.command(args+[self.record['network_name']],check=False)
            except BaseException:
                self.record['state']='needs_reconciliation';self.save()
                try:self.reconcile_created_identity()
                except BaseException:pass
                raise
            if created.returncode:
                if 'overlap'in created.stderr.lower():continue
                raise RuntimeError('Explicit network creation failed; no implicit fallback')
            ident=created.stdout.strip()
            if not re.fullmatch(r'[0-9a-f]{64}',ident):
                self.record['state']='needs_reconciliation';self.save()
                self.reconcile_created_identity()
                raise OwnershipError('Create returned invalid ID; recovered identity retained for cleanup')
            self.record.update(network_id=ident,state='created');self.save()
            self.verify(self.inspect(ident))
            if any(subnet.overlaps(n)for n in self.inventory(skip=ident)):
                self.remove_empty();raise ValueError('Host routes changed during allocation')
            return self.record
        raise ValueError('Bounded network allocation retries exhausted')

    def remove_empty(self):
        ident=self.record.get('network_id')
        if not ident:
            if self.record.get('state')=='needs_reconciliation':
                self.reconcile_created_identity()
                ident=self.record['network_id']
            else:return
        self.local_daemon();network=self.inspect(ident)
        if network is None:self.record['state']='removed';self.save();return
        self.verify(network)
        if network.get('Containers'):raise OwnershipError('Network still has endpoints; no forced disconnect')
        self.command(['docker','network','rm',ident])
        if self.inspect(ident)is not None:raise RuntimeError('Network removal not confirmed')
        self.record['state']='removed';self.save()
