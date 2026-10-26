"""Adapter lifecycle fixtures mirror pinned Harbor boundaries; no Docker calls."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from tests.llm_module.test_owned_network import FakeDaemon, TestAllocator
from llm_module.agentic.owned_network import OwnershipError


@pytest.fixture
def adapter(monkeypatch,tmp_path):
    class Base:
        def __init__(self,**kwargs):
            self.session_id=kwargs['session_id'];self._is_windows_container=False;self._enable_egress_control=False
            self.network_policy=SimpleNamespace(network_mode=SimpleNamespace(value='public'))
            self.prepared=False;self.commands=[];self.bad_graph=False;self.fail_up=False
        @property
        def _docker_compose_paths(self):return [Path('base.json')]
        async def _run_docker_compose_command(self,command,*a,**k):
            assert self.prepared,'render must follow base overlay preparation'
            self.commands.append(command)
            graph={'services':{'main':{'ports':['80:80']}if self.bad_graph else {}},'networks':{'default':{'name':self.session_id+'_default'}}}
            if self._owned_overlay:graph=json.loads(self._owned_overlay.read_text())
            if command[0]=='up'and self.fail_up:raise RuntimeError('partial startup failed')
            return SimpleNamespace(stdout=json.dumps(graph),return_code=0)
        async def start(self,force_build):
            self.prepared=True
            await self._run_docker_compose_command(['build'])
            await self._run_docker_compose_command(['down','--remove-orphans'])
            await self._run_docker_compose_command(['up','--detach','--wait'])
        async def stop(self,delete):
            try:await self._run_docker_compose_command(['down','--volumes'])
            except Exception:pass
        @staticmethod
        async def _collect_buffered_output(process,**kwargs):raise asyncio.CancelledError()
        @staticmethod
        async def _terminate_process(process):process.terminated=True
    docker=ModuleType('harbor.environments.docker.docker');docker.DockerEnvironment=Base
    base=ModuleType('harbor.environments.base');base.ExecResult=lambda **k:SimpleNamespace(**k)
    monkeypatch.setitem(sys.modules,docker.__name__,docker);monkeypatch.setitem(sys.modules,base.__name__,base)
    path=Path('llm_module/agentic/owned_docker_environment.py')
    spec=importlib.util.spec_from_file_location('llm_module.agentic.fixture_owned_environment',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    daemon=FakeDaemon()
    original=daemon.run
    def run(args,**kwargs):
        if args[:2]==['docker','ps']:return SimpleNamespace(stdout='',returncode=0,stderr='')
        return original(args,**kwargs)
    class FixtureAllocator(TestAllocator):
        def inventory(self,skip=None):
            import ipaddress
            with daemon.lock:return [ipaddress.ip_network(n['IPAM']['Config'][0]['Subnet'])for key,n in daemon.networks.items()if key!=skip]
    monkeypatch.setattr(module,'Allocator',lambda labels,project,journal,host:FixtureAllocator(labels,project,journal,host,run))
    def make():return module.OwnedDockerEnvironment(session_id='logical__env',owned_owner='owner',owned_job='job',owned_journal_dir=str(tmp_path),owned_endpoint_host='localhost')
    return module,daemon,make


@pytest.mark.asyncio
async def test_prepared_merge_unique_attempts_and_no_broad_down(adapter):
    module,daemon,make=adapter;a=make();b=make()
    assert a.session_id!=b.session_id
    await a.start(False);await b.start(False)
    assert len(daemon.networks)==2
    assert all(command[0]!='down'for command in a.commands)
    graph=json.loads(a._owned_overlay.read_text())
    assert set(graph['networks']['default'])=={'external','name'}
    assert len(graph['services']['main']['labels'])==4
    await a.stop(True);await a.stop(True);await b.stop(True)
    assert not daemon.networks


@pytest.mark.asyncio
async def test_unsupported_graph_rejected_before_allocation(adapter):
    module,daemon,make=adapter;a=make();a.bad_graph=True
    with pytest.raises(ValueError):await a.start(False)
    assert daemon.created==0


@pytest.mark.asyncio
async def test_start_failure_cleans_only_its_empty_network(adapter):
    module,daemon,make=adapter;a=make();a.fail_up=True
    with pytest.raises(RuntimeError):await a.start(False)
    assert daemon.created==1 and not daemon.networks


@pytest.mark.asyncio
async def test_cancellation_settles_process_and_preserves_uncertain_lease(adapter):
    module,daemon,make=adapter;a=make();await a.start(False);proc=SimpleNamespace(terminated=False)
    with pytest.raises(asyncio.CancelledError):await a._collect_buffered_output(proc,timeout_sec=3)
    assert proc.terminated and a._owned_uncertain
    with pytest.raises(OwnershipError):await a.stop(True)
    assert len(daemon.networks)==1
    assert a._owned_allocator.record['state']=='needs_reconciliation'
