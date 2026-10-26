"""Bounded host-network inventory via an owned, unprivileged ephemeral helper."""
import json
from pathlib import Path
import re
import uuid

from .allocator import OWNER, MANAGED, ATTEMPT, JOB, OwnershipError

POLICY = 'host-inventory-v1'
# No dynamic shell or device imports. Docker start is bounded by command(timeout30).
COLLECT = Path(__file__).with_name('netlink_inventory.py').read_text()

FORMAT = ('{"id":{{json .Id}},"name":{{json .Name}},"image":{{json .Image}},'
          '"labels":{{json .Config.Labels}},"running":{{json .State.Running}},'
          '"network_mode":{{json .HostConfig.NetworkMode}},"privileged":{{json .HostConfig.Privileged}},'
          '"readonly":{{json .HostConfig.ReadonlyRootfs}},"cap_drop":{{json .HostConfig.CapDrop}},'
          '"security_opt":{{json .HostConfig.SecurityOpt}},"devices":{{json .HostConfig.Devices}},'
          '"device_requests":{{json .HostConfig.DeviceRequests}},"mounts":{{json .Mounts}}}')


def collect(allocator):
    """Return host route/address/DNS data only after verified helper cleanup."""
    allocator.local_daemon()
    daemon=allocator.record['daemon_id']
    ids=allocator.command(['docker','ps','-q','--no-trunc','--filter','label='+OWNER+'='+allocator.labels[OWNER],'--filter','name=tt-inference-server-']).stdout.split()
    servers=[]
    def inspect(ident):
        response=allocator.command(['docker','inspect','--format',FORMAT,ident],check=False)
        if response.returncode:
            if 'No such object'in response.stderr:return None
            raise OwnershipError('Inventory container inspection unavailable')
        return json.loads(response.stdout)
    for ident in ids:
        item=inspect(ident)
        if item is None:raise OwnershipError('Owned server inventory changed')
        if item['id']!=ident or (item.get('labels')or{}).get(OWNER)!=allocator.labels[OWNER]:
            raise OwnershipError('Owned server identity mismatch')
        if item['running'] and item['name'].startswith('/tt-inference-server-'):
            servers.append(item)
    if len(servers)!=1:raise OwnershipError('Host inventory requires exactly one running owned server image')
    server=servers[0];image=server['image']
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',image):raise OwnershipError('Immutable server image ID unavailable')
    volumes=allocator.data(['docker','image','inspect','--format','{{json .Config.Volumes}}',image])
    if volumes:raise OwnershipError('Inventory image declares implicit volumes')
    attempt=uuid.uuid4().hex;name='g12-inventory-'+attempt
    labels={OWNER:allocator.labels[OWNER],MANAGED:POLICY,ATTEMPT:attempt,JOB:allocator.labels[JOB]}
    directory=allocator.journal.parent/'inventory_helpers';directory.mkdir(exist_ok=True)
    path=directory/(attempt+'.json')
    record={'schema_version':1,'owner':labels[OWNER],'job':labels[JOB],'attempt':attempt,'daemon_id':daemon,
            'container_id':None,'name':name,'image_id':image,'server_id':server['id'],'state':'intent',
            'basis':'Owned unprivileged host-network helper; server networking unchanged'}
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
    save()
    def verify(item):
        if not item or not re.fullmatch(r'[0-9a-f]{64}',item.get('id','')) or item['name']!='/'+name or item['image']!=image:
            raise OwnershipError('Host inventory helper identity mismatch')
        if record['container_id']not in (None,item['id']):raise OwnershipError('Host inventory helper ID changed')
        if any((item.get('labels')or{}).get(k)!=v for k,v in labels.items()):raise OwnershipError('Host inventory helper ownership mismatch')
        if item['network_mode']!='host' or item['privileged'] or not item['readonly'] or item['devices'] or item['device_requests'] or item['mounts']:
            raise OwnershipError('Host inventory helper isolation changed')
        if set(item['cap_drop']or[])!={'ALL'} or not any(x in ('no-new-privileges','no-new-privileges:true')for x in item['security_opt']or[]):
            raise OwnershipError('Host inventory helper restrictions missing')
        allocator.local_daemon()
        if allocator.record['daemon_id']!=daemon:raise OwnershipError('Inventory Docker daemon changed')
    result=None;uncertain=False
    try:
        command=['docker','create','--pull','never','--name',name,'--network','host','--cap-drop','ALL',
                 '--security-opt','no-new-privileges','--read-only','--entrypoint','python3']
        for key,value in labels.items():command+=['--label',key+'='+value]
        try:
            created=allocator.command(command+[image,'-c',COLLECT,allocator.endpoint_host])
            ident=created.stdout.strip()
            if not re.fullmatch(r'[0-9a-f]{64}',ident):raise OwnershipError('Missing created helper ID')
            record.update(container_id=ident,state='created');save()
        except BaseException:
            uncertain=True;record['state']='needs_reconciliation';save()
            found=inspect(name)
            if found is not None:
                verify(found);record.update(container_id=found['id'],state='created');save();uncertain=False
            raise
        verify(inspect(ident))
        record['state']='running';save()
        response=allocator.command(['docker','start','--attach',ident])
        after=inspect(ident);verify(after)
        if after['running']:raise OwnershipError('Inventory helper did not settle')
        result=json.loads(response.stdout)
        if not all(isinstance(result.get(k),list)and result[k]for k in ('routes','addresses','model_addresses')) or not isinstance(result.get('dns'),list):
            raise ValueError('Incomplete daemon-host route/address inventory')
        current=inspect(server['id'])
        if not current or current['id']!=server['id']or current['image']!=image or not current['running']or (current.get('labels')or{}).get(OWNER)!=labels[OWNER]:
            raise OwnershipError('Owned server changed during inventory')
        record['counts']={k:len(result[k])for k in ('routes','addresses','dns','model_addresses')}
        record['network_namespace']=result.get('network_namespace');save()
    finally:
        ident=record['container_id']
        if ident:
            item=inspect(ident);verify(item)
            if item['running']:
                allocator.command(['docker','stop','--time','2',ident]);verify(inspect(ident))
            allocator.command(['docker','rm',ident])
            if inspect(ident)is not None:raise OwnershipError('Host inventory helper removal unconfirmed')
            record['state']='removed';save()
        elif uncertain:
            record['state']='needs_reconciliation';save()
        else:
            record['state']='not_created';save()
    allocator.record['inventory_provider']={'kind':'owned_host_network_helper','daemon_id':daemon,
        'server_id':server['id'],'image_id':image,'last_helper_journal':str(path),'cleanup_verified':True}
    allocator.save()
    return result
