"""Harbor0.23 extension: isolated explicit bridges and exact-ID owned teardown."""
import asyncio
import hashlib
import json
from pathlib import Path
import uuid

from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.base import ExecResult
from .owned_network import Allocator, OWNER, MANAGED, ATTEMPT, JOB, POLICY, OwnershipError, checked_owner, topology


async def settled(function, *args):
    """Do not abandon an in-flight Docker CLI mutation on coroutine cancellation."""
    task=asyncio.create_task(asyncio.to_thread(function,*args))
    try:return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:await task
        finally:raise


class OwnedDockerEnvironment(DockerEnvironment):
    def __init__(self,*args,owned_owner,owned_job,owned_journal_dir,owned_endpoint_host,**kwargs):
        owner=checked_owner(owned_owner)
        if not owned_endpoint_host:raise ValueError('Model endpoint hostname required')
        if kwargs.get('stream'):raise ValueError('Streaming import-path environments unsupported')
        policies=list(kwargs.get('phase_network_policies',()))
        if kwargs.get('network_policy') is not None:policies.append(kwargs['network_policy'])
        for policy in policies:
            if policy.network_mode.value!='public':raise ValueError('Dynamic/nonpublic network policy unsupported')
        if kwargs.get('keep_containers'):raise ValueError('Owned evaluation requires teardown, not retained containers')
        self._owned_overlay=None;self._owned_ready=False;self._owned_uncertain=False;self._owned_started=False
        self._owned_attempt=uuid.uuid4().hex
        self._owned_job=owned_job
        self._owned_labels={OWNER:owner,MANAGED:POLICY,ATTEMPT:self._owned_attempt,JOB:owned_job}
        # No allocation in construction; context_id arrives later from Harbor.
        self._owned_journal_dir=Path(owned_journal_dir).resolve()
        self._owned_endpoint_host=owned_endpoint_host
        self._owned_allocator=None
        super().__init__(*args,**kwargs)
        self._owned_logical_session=self.session_id
        self.session_id='g12-'+hashlib.sha256((owner+owned_job+self._owned_logical_session+self._owned_attempt).encode()).hexdigest()[:32]

    @property
    def _docker_compose_paths(self):
        paths=list(super()._docker_compose_paths)
        if self._owned_overlay is not None:paths.append(self._owned_overlay)
        return paths

    async def _prepare_owned(self):
        if self._owned_ready:return
        if self._is_windows_container or self._enable_egress_control or self.network_policy.network_mode.value!='public':
            raise ValueError('Only Linux public default-bridge tasks supported')
        # Called only from a Compose boundary after base start writes all overlays.
        rendered=await super()._run_docker_compose_command(['config','--format','json'])
        services=topology(json.loads(rendered.stdout),self.session_id)
        allocator=Allocator(self._owned_labels,self.session_id,
            self._owned_journal_dir/(self._owned_attempt+'.json'),self._owned_endpoint_host)
        self._owned_allocator=allocator
        allocator.record['logical_session']=self._owned_logical_session
        allocator.record['context_id']=str(getattr(self,'context_id',None));allocator.save()
        await self._assert_project_owned(empty=True)
        record=await settled(allocator.acquire)
        self._owned_overlay=self._owned_journal_dir/(self._owned_attempt+'.compose.json')
        self._owned_overlay.write_text(json.dumps({'services':{s:{'labels':self._owned_labels}for s in services},
            'networks':{'default':{'external':True,'name':record['network_name']}}},indent=2))
        final=await super()._run_docker_compose_command(['config','--format','json'])
        graph=json.loads(final.stdout)
        if graph.get('networks')!={'default':{'name':record['network_name'],'external':True}}:
            raise OwnershipError('Merged external network contains unsupported attributes')
        self._owned_ready=True

    async def _project_containers(self):
        allocator=self._owned_allocator
        if allocator is None:return []
        response=await settled(allocator.command,['docker','ps','-a','-q','--no-trunc','--filter','label=com.docker.compose.project='+self.session_id])
        return response.stdout.split()

    async def _container(self,ident):
        fmt='{"id":{{json .Id}},"labels":{{json .Config.Labels}}}'
        response=await settled(self._owned_allocator.command,['docker','inspect','--format',fmt,ident],False)
        if response.returncode:
            if 'No such object'in response.stderr:return None
            raise OwnershipError('Container inspection unavailable')
        item=json.loads(response.stdout)
        if item['id']!=ident or (item.get('labels')or{}).get('com.docker.compose.project')!=self.session_id or any((item.get('labels')or{}).get(k)!=v for k,v in self._owned_labels.items()):
            raise OwnershipError('Foreign container in attempt project; no teardown')
        return item

    async def _assert_project_owned(self,empty=False):
        ids=await self._project_containers()
        if empty and ids:raise OwnershipError('Unexpected existing attempt project')
        for ident in ids:await self._container(ident)
        return ids

    async def _teardown(self):
        allocator=self._owned_allocator
        if allocator is None:return
        if self._owned_uncertain or allocator.record.get('state')=='needs_reconciliation':
            allocator.record['state']='needs_reconciliation';allocator.save()
            raise OwnershipError('Unsettled daemon mutation; preserve journal for closed-job reconciliation')
        ids=await self._assert_project_owned()
        allocator.record['state']='stopping';allocator.save()
        for ident in ids:
            if await self._container(ident) is None:continue
            await settled(allocator.command,['docker','stop','--time','10',ident])
            if await self._container(ident) is None:continue
            await settled(allocator.command,['docker','rm',ident])
        if await self._project_containers():raise OwnershipError('Owned containers remain')
        await settled(allocator.remove_empty)

    async def _run_docker_compose_command(self,command,*args,**kwargs):
        if command[0]in('down','stop'):
            # Base start's stale-project down must not remove the new network.
            if not self._owned_started:
                await self._prepare_owned();await self._assert_project_owned(empty=True)
            else:await self._teardown()
            return ExecResult(stdout='',stderr='',return_code=0)
        await self._prepare_owned()
        await self._assert_project_owned()
        if command[0]=='up':
            await settled(self._owned_allocator.verify,await settled(self._owned_allocator.inspect,self._owned_allocator.record['network_id']))
            self._owned_started=True
        try:
            result=await super()._run_docker_compose_command(command,*args,**kwargs)
        except (asyncio.CancelledError,asyncio.TimeoutError):
            self._owned_uncertain=True
            self._owned_allocator.record['state']='needs_reconciliation';self._owned_allocator.save();raise
        if command[0]=='up':
            self._owned_allocator.record['state']='attached';self._owned_allocator.save()
        return result

    async def _collect_buffered_output(self,process,*,timeout_sec,stdin_data=None):
        try:return await DockerEnvironment._collect_buffered_output(process,timeout_sec=timeout_sec,stdin_data=stdin_data)
        except BaseException:
            self._owned_uncertain=True
            await DockerEnvironment._terminate_process(process)
            raise

    async def _collect_streamed_output(self,process,**kwargs):
        try:return await super()._collect_streamed_output(process,**kwargs)
        except BaseException:
            self._owned_uncertain=True
            raise

    async def start(self,*args,**kwargs):
        if kwargs.get('stream'):raise ValueError('Streaming import-path environments unsupported')
        try:return await super().start(*args,**kwargs)
        except BaseException:
            if self._owned_allocator is not None:
                try:await asyncio.shield(self._teardown())
                except BaseException:
                    self._owned_allocator.record['cleanup_error']='start_failure_requires_reconciliation';self._owned_allocator.save()
            raise

    async def stop(self,delete):
        await super().stop(delete)
        # Harbor logs teardown failures; explicit recheck must escape swallowing.
        await self._teardown()
