"""Phase transitions must retain the exact internal bridge and settle before egress returns."""
import re
from types import SimpleNamespace

import pytest

from agentic_owned_networks.allocator import ATTEMPT, OwnershipError
from tests.llm_module.test_owned_docker_environment import adapter


async def prepared(adapter, monkeypatch):
    module, daemon, make = adapter
    env = make(owned_agent_isolation=True)
    env.prepared = True
    await env._prepare_owned()
    primary, setup = env._owned_allocator, env._owned_setup_allocator
    networks = {primary.record["network_name"]: {"NetworkID": primary.record["network_id"]}}
    item = dict(networks=networks, Mounts=[], HostConfig=dict(
        Privileged=False, PidMode="", IpcMode="", CapAdd=[], Devices=[],
        DeviceRequests=[], SecurityOpt=[], Binds=[]))

    async def ids(): return ["c" * 64]
    async def container(ident):
        assert ident == "c" * 64
        return item

    monkeypatch.setattr(env, "_assert_project_owned", ids)
    monkeypatch.setattr(env, "_container", container)
    original = setup.command
    actions = []

    def command(args, check=True):
        if args[:3] in (["docker", "network", "connect"], ["docker", "network", "disconnect"]):
            assert args[3:] == [setup.record["network_id"], "c" * 64]
            actions.append(args[2])
            if args[2] == "connect":
                networks[setup.record["network_name"]] = {"NetworkID": setup.record["network_id"]}
            else:
                del networks[setup.record["network_name"]]
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return original(args, check)

    monkeypatch.setattr(setup, "command", command)
    return env, networks, actions


@pytest.mark.asyncio
async def test_internal_primary_phase_transitions_and_recoverable_journals(adapter, monkeypatch):
    env, networks, actions = await prepared(adapter, monkeypatch)
    primary, setup = env._owned_allocator, env._owned_setup_allocator
    assert primary.inspect(primary.record["network_id"])["Internal"] is True
    assert setup.inspect(setup.record["network_id"])["Internal"] is False
    assert setup.labels[ATTEMPT] != primary.labels[ATTEMPT]
    assert re.fullmatch(r"[0-9a-f]{32}", setup.record["attempt"])
    assert setup.journal.name == setup.record["attempt"] + ".json"
    assert setup.record["parent_attempt"] == primary.record["attempt"]
    await env.leave_agent_isolation()  # setup has egress
    assert len(networks) == 2
    await env.enter_agent_isolation()
    assert set(networks) == {primary.record["network_name"]}
    await env.leave_agent_isolation()  # verifier may install dependencies
    assert len(networks) == 2 and actions == ["connect", "disconnect", "connect"]
    primary.verify(primary.inspect(primary.record["network_id"]))


@pytest.mark.asyncio
async def test_unsettled_workload_never_regains_public_egress(adapter, monkeypatch):
    env, networks, actions = await prepared(adapter, monkeypatch)
    env._owned_workload_unsettled = True
    with pytest.raises(OwnershipError, match="must remain detached"):
        await env.leave_agent_isolation()
    assert len(networks) == 1 and actions == []


@pytest.mark.asyncio
async def test_unexpected_attachment_fails_before_any_network_mutation(adapter, monkeypatch):
    env, networks, actions = await prepared(adapter, monkeypatch)
    networks["foreign"] = {"NetworkID": "f" * 64}
    with pytest.raises(OwnershipError, match="Unexpected agent network"):
        await env.enter_agent_isolation()
    assert actions == [] and env._owned_uncertain
    assert all(a.record["state"] == "needs_reconciliation"
               for a in (env._owned_allocator, env._owned_setup_allocator))


@pytest.mark.asyncio
async def test_changed_internal_flag_is_rejected(adapter, monkeypatch):
    env, networks, actions = await prepared(adapter, monkeypatch)
    primary = env._owned_allocator
    item = primary.inspect(primary.record["network_id"])
    item["Internal"] = False
    with pytest.raises(OwnershipError):
        primary.verify(item)
    assert actions == []


@pytest.mark.asyncio
async def test_failed_network_mutation_cannot_silently_retry_public_egress(adapter, monkeypatch):
    env, networks, actions = await prepared(adapter, monkeypatch)
    setup = env._owned_setup_allocator
    original = setup.command

    def fail(args, check=True):
        if args[:3] == ["docker", "network", "connect"]:
            raise TimeoutError("daemon response unknown")
        return original(args, check)

    monkeypatch.setattr(setup, "command", fail)
    with pytest.raises(TimeoutError):
        await env.leave_agent_isolation()
    assert env._owned_uncertain and len(networks) == 1
    with pytest.raises(OwnershipError, match="no further egress transitions"):
        await env.leave_agent_isolation()
