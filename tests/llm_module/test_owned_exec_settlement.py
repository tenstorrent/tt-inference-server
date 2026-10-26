"""Exercise actual adapter cancellation scopes; CPU only, no Docker."""
import asyncio
import json
import shlex
from types import SimpleNamespace
import pytest
from tests.llm_module.test_owned_docker_environment import adapter
from agentic_owned_networks.execution_scope import installed_agent_scope
from agentic_owned_networks.allocator import OwnershipError


async def ready(adapter, monkeypatch):
    module, daemon, make = adapter
    obj = make(); await obj.start(False)
    async def ids(*a, **k):return ['owned-full-id']
    async def container(*a):return {'fixture': 'ownership already checked'}
    monkeypatch.setattr(obj, '_assert_project_owned', ids)
    monkeypatch.setattr(obj, '_container', container)
    monkeypatch.setattr(module, 'validate_exec_authority', lambda item: None)
    return module, obj


@pytest.mark.asyncio
@pytest.mark.parametrize('operation,tainted', [('exec',False),('up',True),('build',True)])
async def test_actual_dispatch_collector_cancellation_scope(adapter,monkeypatch,operation,tainted):
    module,obj=await ready(adapter,monkeypatch)
    process=SimpleNamespace(terminated=False)
    async def boundary(self,command,*a,**k):
        return await self._collect_buffered_output(process,timeout_sec=1)
    monkeypatch.setattr(module.DockerEnvironment,'_run_docker_compose_command',boundary)
    with pytest.raises(asyncio.CancelledError):await obj._run_docker_compose_command([operation])
    assert process.terminated and obj._owned_uncertain is tainted
    if tainted:assert obj._owned_allocator.record['state']=='needs_reconciliation'


@pytest.mark.asyncio
async def test_verified_exec_never_clears_prior_taint(adapter,monkeypatch):
    module,obj=await ready(adapter,monkeypatch);obj._owned_uncertain=True
    obj._owned_allocator.record['state']='needs_reconciliation'
    async def boundary(self,*a,**k):raise asyncio.CancelledError()
    monkeypatch.setattr(module.DockerEnvironment,'_run_docker_compose_command',boundary)
    with pytest.raises(asyncio.CancelledError):await obj._run_docker_compose_command(['exec'])
    assert obj._owned_uncertain and obj._owned_allocator.record['state']=='needs_reconciliation'


def install_exec_fixture(module,monkeypatch, *, exitcode=0, bad=False, failure=False):
    calls=[]
    async def execute(self,command,**kwargs):
        args=shlex.split(command);calls.append(args)
        token=args[args.index('--token')+1]
        if 'settle' in args:
            receipt={'status':'settled','supervisor_exited':True,'token':token,
                     'supervisor':{'pid':100,'starttime':200},'cancelled':False}
            if bad:receipt['token']='wrong-token'
            return SimpleNamespace(return_code=125 if failure else 0,stdout=json.dumps(receipt))
        return SimpleNamespace(return_code=exitcode,stdout='normal output')
    monkeypatch.setattr(module.DockerEnvironment,'_compose_exec',execute,raising=False)
    return calls


async def invoke(obj):
    token=installed_agent_scope.set(True)
    try:return await obj._compose_exec('workload',service='main',cwd=None,env=None,timeout_sec=1,user=None)
    finally:installed_agent_scope.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize('exitcode',[0,7])
async def test_supervised_preserves_exit_after_settlement(adapter,monkeypatch,exitcode):
    module,obj=await ready(adapter,monkeypatch)
    calls=install_exec_fixture(module,monkeypatch,exitcode=exitcode)
    result=await invoke(obj)
    assert result.return_code==exitcode and result.stdout=='normal output'
    assert len(calls)==2 and obj._owned_allocator.record['workload_execs'][-1]['state']=='settled'


@pytest.mark.asyncio
@pytest.mark.parametrize('kind',['bad','failure'])
async def test_failed_or_wrong_receipt_blocks_verifier(adapter,monkeypatch,kind):
    module,obj=await ready(adapter,monkeypatch)
    install_exec_fixture(module,monkeypatch,bad=kind=='bad',failure=kind=='failure')
    with pytest.raises((OwnershipError,ValueError)):await invoke(obj)
    assert obj._owned_workload_unsettled
    with pytest.raises(OwnershipError):
        await obj._compose_exec('verifier',service='main',cwd=None,env=None,timeout_sec=1,user=None)


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_cleanup(adapter):
    module,_,_=adapter
    entered=asyncio.Event();release=asyncio.Event();finished=[]
    async def cleanup():
        entered.set();await release.wait();finished.append(True);return 'receipt'
    task=asyncio.create_task(module.completion_barrier(cleanup()))
    await entered.wait();task.cancel();await asyncio.sleep(0);task.cancel();await asyncio.sleep(0)
    assert not task.done() and not finished
    release.set();result,cancelled=await task
    assert result=='receipt' and cancelled and finished


@pytest.mark.asyncio
async def test_cancelled_agent_waits_settlement_before_error(adapter,monkeypatch):
    module,obj=await ready(adapter,monkeypatch)
    started=asyncio.Event();settling=asyncio.Event();release=asyncio.Event()
    async def execute(self,command,**kwargs):
        args=shlex.split(command);token=args[args.index('--token')+1]
        if 'run' in args:
            started.set();await asyncio.Event().wait()
        settling.set();await release.wait()
        return SimpleNamespace(return_code=0,stdout=json.dumps({'status':'settled','supervisor_exited':True,'token':token,'supervisor':{'pid':100,'starttime':200},'cancelled':True}))
    monkeypatch.setattr(module.DockerEnvironment,'_compose_exec',execute,raising=False)
    task=asyncio.create_task(invoke(obj));await started.wait();task.cancel();await settling.wait()
    task.cancel();await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert obj._owned_allocator.record['workload_execs'][-1]['state']=='settled'
    assert not obj._owned_workload_unsettled
