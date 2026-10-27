"""A deadline during egress restoration must settle cleanup and remain a deadline."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.llm_module.test_owned_docker_environment import adapter


def agent_class(adapter, monkeypatch, failure=False):
    class Parent:
        async def run(self, *args, **kwargs):
            if failure:raise RuntimeError("agent failed")
            return "agent-result"
    parent = ModuleType("harbor.agents.installed.mini_swe_agent")
    parent.MiniSweAgent = Parent
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.setitem(sys.modules, "agentic_owned_networks.docker_environment", adapter[0])
    spec = importlib.util.spec_from_file_location(
        "agentic_owned_networks.fixture_phase_mini", Path("agentic_owned_networks/mini_swe_agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OwnedMiniSweAgent


@pytest.mark.asyncio
async def test_deadline_during_egress_restore_is_not_returned_as_success(adapter, monkeypatch):
    agent = agent_class(adapter, monkeypatch)()
    entered, release = asyncio.Event(), asyncio.Event()
    async def enter():pass
    async def leave():
        entered.set()
        await release.wait()
    env = SimpleNamespace(_owned_agent_isolation=True,
                          enter_agent_isolation=enter, leave_agent_isolation=leave)
    task = asyncio.create_task(agent.run("task", env, None))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):await task


@pytest.mark.asyncio
async def test_agent_error_survives_successful_egress_restore(adapter, monkeypatch):
    agent = agent_class(adapter, monkeypatch, failure=True)()
    phases = []
    async def enter():phases.append("agent")
    async def leave():phases.append("verifier")
    env = SimpleNamespace(_owned_agent_isolation=True,
                          enter_agent_isolation=enter, leave_agent_isolation=leave)
    with pytest.raises(RuntimeError, match="agent failed"):
        await agent.run("task", env, None)
    assert phases == ["agent", "verifier"]


@pytest.mark.asyncio
async def test_unchanged_public_agent_does_not_require_phase_methods(adapter, monkeypatch):
    agent = agent_class(adapter, monkeypatch)()
    assert await agent.run("task", SimpleNamespace(), None) == "agent-result"
