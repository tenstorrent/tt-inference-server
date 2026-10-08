# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import contextlib
import os
import pickle
import queue
import sys
import threading
import types
from unittest.mock import MagicMock, patch

import multihost
import pytest
from domain.image_generate_request import ImageGenerateRequest
from domain.video_generate_request import VideoGenerateRequest
from domain.video_i2v_generate_request import (
    ImagePromptEntry,
    VideoI2VGenerateRequest,
)
from domain.video_ref2va_generate_request import (
    MediaSource,
    MultimodalReferences,
    VideoRef2VAGenerateRequest,
)
from multihost import comm, is_follower_rank, is_multihost_launch, launch_rank
from multihost.comm import (
    CanaryMessage,
    RunMessage,
    ShutdownMessage,
    from_request_data,
    to_request_data,
)
from multihost.follower import MultiHostLockstepFollower
from multihost.lockstep_runner import MultiHostLockstepRunner, wrap_if_multihost

_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
_TIMEOUT_S = 5.0
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENV_DIR = os.path.join(_PROJECT_DIR, "python_env")


@contextlib.contextmanager
def _fresh_project_modules():
    """Import project modules afresh, then put ``sys.modules`` back.

    Other test files replace modules such as ``config.settings`` with Mocks in
    ``sys.modules`` and leave them there; anything imported after that is
    bound to the Mock.
    """
    saved = dict(sys.modules)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None) or ""
        is_project_module = (
            path.startswith(_PROJECT_DIR + os.sep)
            and not path.startswith(_VENV_DIR + os.sep)
            and not name.startswith("tests")
        )
        if not isinstance(module, types.ModuleType) or is_project_module:
            del sys.modules[name]
    try:
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


# --------------------------------------------------------------------------- fakes


class _FakeLink:
    """In-process stand-in for the MPI world: one inbox per follower, one barrier."""

    def __init__(self, size: int):
        self.size = size
        self.inboxes = {rank: queue.Queue() for rank in range(1, size)}
        self.barrier = threading.Barrier(size, timeout=_TIMEOUT_S)


class _FakeComm:
    def __init__(self, link: _FakeLink, rank: int):
        self._link = link
        self.rank = rank

    def broadcast(self, message):
        # Pickle like the real comm, so anything unpicklable fails here too.
        payload = pickle.dumps(message)
        for inbox in self._link.inboxes.values():
            inbox.put(payload)

    def receive(self):
        return pickle.loads(self._link.inboxes[self.rank].get(timeout=_TIMEOUT_S))

    def barrier(self):
        self._link.barrier.wait()


class _FakeRunner:
    def __init__(self, fail_prompts=(), warmup_ok=True):
        self.calls = []
        self.export_in_runner = True
        self.ttnn_device = None
        self.closed = False
        self.settings = "runner-settings"
        self._fail_prompts = set(fail_prompts)
        self._warmup_ok = warmup_ok

    def set_device(self):
        self.ttnn_device = "mesh"
        return self.ttnn_device

    async def warmup(self):
        return self._warmup_ok

    def run(self, requests):
        self.calls.append(requests)
        if requests[0].prompt in self._fail_prompts:
            raise ValueError(f"bad prompt {requests[0].prompt}")
        return [f"result-{requests[0].prompt}"]

    def _build_warmup_video_request(self):
        return VideoGenerateRequest.model_construct(
            prompt="warmup", negative_prompt="", num_inference_steps=2
        )

    def close_device(self):
        self.closed = True
        return True


class _Quad:
    """Rank 0's wrapped runner plus follower threads, wired through a fake link."""

    def __init__(self, size=2, follower_runner_kwargs=None):
        self.link = _FakeLink(size)
        self.leader_runner = _FakeRunner()
        self.follower_runners = {}
        self.exit_codes = {}
        self.threads = []
        for rank in range(1, size):
            runner = _FakeRunner(**(follower_runner_kwargs or {}))
            self.follower_runners[rank] = runner
            follower = MultiHostLockstepFollower(
                rank,
                runner_factory=lambda runner=runner: runner,
                comm=_FakeComm(self.link, rank),
            )
            thread = threading.Thread(
                target=lambda rank=rank, follower=follower: self.exit_codes.__setitem__(
                    rank, follower.run()
                ),
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)
        self.leader_runner.set_device()
        self.leader = MultiHostLockstepRunner(
            self.leader_runner, comm=_FakeComm(self.link, 0)
        )

    def start(self):
        assert asyncio.run(self.leader.warmup()) is True
        return self

    def shutdown(self):
        self.leader.close_device()
        for thread in self.threads:
            thread.join(timeout=_TIMEOUT_S)
            assert not thread.is_alive()


def _video_request(prompt="a cat", **kwargs):
    return VideoGenerateRequest(prompt=prompt, **kwargs)


# --------------------------------------------------------------------------- launch env


class TestLaunchEnv:
    def test_single_host(self, monkeypatch):
        monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
        monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
        assert launch_rank() is None
        assert not is_follower_rank()
        assert not is_multihost_launch()

    def test_rank_zero(self, monkeypatch):
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
        assert launch_rank() == 0
        assert not is_follower_rank()
        assert is_multihost_launch()

    def test_follower(self, monkeypatch):
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
        assert launch_rank() == 3
        assert is_follower_rank()

    def test_rank_zero_stops_when_launcher_exits(self, monkeypatch):
        # Parent is mpirun for two polls, then the process is reparented.
        parents = iter([100, 100, 100, 1])
        monkeypatch.setattr(multihost.os, "getppid", lambda: next(parents))
        monkeypatch.setattr(multihost, "_LAUNCHER_POLL_S", 0)
        kill = MagicMock()
        monkeypatch.setattr(multihost.os, "kill", kill)
        asyncio.run(asyncio.wait_for(multihost.stop_when_launcher_exits(), 5))
        kill.assert_called_once_with(os.getpid(), multihost.signal.SIGTERM)


# --------------------------------------------------------------------------- request data


def _assert_same_request(original, rebuilt):
    assert type(rebuilt) is type(original)
    assert rebuilt._task_id == original._task_id
    assert rebuilt.model_dump() == original.model_dump()
    assert rebuilt._start_event is None


class TestRequestData:
    def _round_trip(self, request):
        # Unpicklable, server-local.
        request._start_event = threading.Event()
        data = pickle.loads(pickle.dumps(to_request_data(request)))
        return from_request_data(data)

    def test_t2v(self):
        request = _video_request(
            prompt="sunrise", seed=7, aspect_ratio="16:9", duration=5
        )
        _assert_same_request(request, self._round_trip(request))

    @patch("domain.video_i2v_generate_request.get_settings")
    def test_fl2va_image_prompts(self, get_settings):
        get_settings.return_value = MagicMock(model_runner="tt-minimax-h3-fl2va")
        request = VideoI2VGenerateRequest(
            prompt="keyframes",
            image_prompts=[
                ImagePromptEntry(image=_TINY_PNG_BASE64, frame_pos=0),
                ImagePromptEntry(image=_TINY_PNG_BASE64, frame_pos=-1),
            ],
        )
        rebuilt = self._round_trip(request)
        _assert_same_request(request, rebuilt)
        assert [e.frame_pos for e in rebuilt.image_prompts] == [0, -1]

    def test_ref2va_references(self):
        request = VideoRef2VAGenerateRequest(
            prompt="a quiet room",
            references=MultimodalReferences(images=[MediaSource(b64=_TINY_PNG_BASE64)]),
        )
        rebuilt = self._round_trip(request)
        _assert_same_request(request, rebuilt)
        assert rebuilt.references.images[0].b64 == _TINY_PNG_BASE64

    def test_warmup_style_request_skips_validation(self):
        request = VideoGenerateRequest.model_construct(
            prompt="warmup", negative_prompt="", num_inference_steps=2
        )
        rebuilt = self._round_trip(request)
        assert rebuilt.num_inference_steps == 2
        assert rebuilt._task_id == request._task_id

    def test_other_private_attrs_travel(self):
        request = ImageGenerateRequest.model_construct(prompt="x")
        request._segments = [1, 2]
        rebuilt = self._round_trip(request)
        assert rebuilt._segments == [1, 2]


# --------------------------------------------------------------------------- comm


class _StrongInt:
    """Like ttnn's Rank and Size: converts with int() but is not an index."""

    def __init__(self, value):
        self._value = value

    def __int__(self):
        return self._value


def _fake_ttnn(rank, size, wire):
    module = types.SimpleNamespace()
    module.distributed_context_get_rank = lambda: _StrongInt(rank)
    module.distributed_context_get_size = lambda: _StrongInt(size)
    module.distributed_context_barrier = MagicMock()

    def send_bytes(data, dest, tag):
        wire.setdefault((dest, tag), queue.Queue()).put(bytes(data))

    def recv_bytes(n, source, tag):
        data = wire[(rank, tag)].get(timeout=_TIMEOUT_S)
        assert len(data) == n
        return data

    module.distributed_context_send_bytes = send_bytes
    module.distributed_context_recv_bytes = recv_bytes
    return module


class TestComm:
    def test_broadcast_reaches_every_other_rank(self, monkeypatch):
        wire = {}
        monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn(0, 3, wire))
        comm.broadcast(RunMessage([to_request_data(_video_request("hello"))]))
        for rank in (1, 2):
            monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn(rank, 3, wire))
            received = comm.receive()
            assert from_request_data(received.requests[0]).prompt == "hello"

    def test_broadcast_only_from_rank_zero(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn(1, 2, {}))
        with pytest.raises(RuntimeError):
            comm.broadcast(ShutdownMessage())


# --------------------------------------------------------------------------- lockstep


class TestLockstep:
    def test_run_is_mirrored_on_every_rank(self):
        quad = _Quad(size=4).start()
        request = _video_request("a dog", seed=3)
        assert quad.leader.run([request]) == ["result-a dog"]
        quad.shutdown()
        for runner in quad.follower_runners.values():
            (mirrored,) = runner.calls
            _assert_same_request(request, mirrored[0])
        assert quad.exit_codes == {1: 0, 2: 0, 3: 0}

    def test_follower_turns_off_export(self):
        quad = _Quad().start()
        quad.shutdown()
        assert quad.follower_runners[1].export_in_runner is False

    def test_attribute_passthrough(self):
        quad = _Quad().start()
        assert quad.leader.settings == "runner-settings"
        quad.leader.export_in_runner = False
        assert quad.leader_runner.export_in_runner is False
        quad.shutdown()

    def test_close_device_shuts_down_followers(self):
        quad = _Quad().start()
        quad.shutdown()
        assert quad.leader_runner.closed
        assert quad.follower_runners[1].closed
        assert quad.exit_codes == {1: 0}

    def test_request_error_does_not_stop_follower(self):
        quad = _Quad(follower_runner_kwargs={"fail_prompts": {"bad"}}).start()
        quad.leader_runner._fail_prompts = {"bad"}
        with pytest.raises(ValueError):
            quad.leader.run([_video_request("bad")])
        assert quad.leader.run([_video_request("good")]) == ["result-good"]
        quad.shutdown()
        prompts = [c[0].prompt for c in quad.follower_runners[1].calls]
        assert prompts == ["bad", "good"]
        assert quad.exit_codes == {1: 0}

    def test_follower_warmup_failure_is_fatal(self):
        link = _FakeLink(2)
        follower = MultiHostLockstepFollower(
            1,
            runner_factory=lambda: _FakeRunner(warmup_ok=False),
            comm=_FakeComm(link, 1),
        )
        assert follower.run() == 1

    def test_unknown_message_is_fatal(self):
        quad = _Quad().start()
        quad.leader._comm.broadcast("garbage")
        quad.threads[0].join(timeout=_TIMEOUT_S)
        assert quad.exit_codes == {1: 1}


class TestCanary:
    def test_shallow_probe_is_a_barrier_on_all_ranks(self):
        quad = _Quad(size=3).start()
        assert quad.leader.health_check(deep=False) is True
        quad.shutdown()
        assert all(not r.calls for r in quad.follower_runners.values())
        assert not quad.leader_runner.calls

    def test_deep_probe_replays_warmup_on_all_ranks(self):
        quad = _Quad(size=3).start()
        assert quad.leader.health_check(deep=True) is True
        quad.shutdown()
        for runner in [quad.leader_runner, *quad.follower_runners.values()]:
            (call,) = runner.calls
            assert call[0].prompt == "warmup"

    def test_probe_while_busy_does_not_touch_mesh(self):
        quad = _Quad().start()
        release = threading.Event()
        started = threading.Event()
        original_run = quad.leader_runner.run

        def slow_run(requests):
            started.set()
            release.wait(_TIMEOUT_S)
            return original_run(requests)

        quad.leader_runner.run = slow_run
        worker = threading.Thread(
            target=quad.leader.run, args=([_video_request("long")],)
        )
        worker.start()
        assert started.wait(_TIMEOUT_S)
        # Returns at once: no barrier (which would time out) and no broadcast.
        assert quad.leader.health_check(deep=True) is True
        release.set()
        worker.join(_TIMEOUT_S)
        quad.shutdown()
        prompts = [c[0].prompt for c in quad.follower_runners[1].calls]
        assert prompts == ["long"]

    def test_messages_have_no_server_state(self):
        for message in (CanaryMessage(deep=True), ShutdownMessage()):
            assert pickle.loads(pickle.dumps(message)) == message


# --------------------------------------------------------------------------- wrapping


class TestWrapIfMultihost:
    def _runner(self, mesh="mesh"):
        runner = _FakeRunner()
        runner.ttnn_device = mesh
        return runner

    def _with_ttnn(self, distributed: bool):
        fake = _fake_ttnn(0, 4, {})
        fake.using_distributed_env = lambda: distributed
        return patch.dict(sys.modules, {"ttnn": fake})

    def test_wraps_under_distributed_env(self):
        with self._with_ttnn(True):
            wrapped = wrap_if_multihost(self._runner())
        assert isinstance(wrapped, MultiHostLockstepRunner)

    def test_single_host_unchanged(self):
        runner = self._runner()
        with self._with_ttnn(False):
            assert wrap_if_multihost(runner) is runner

    def test_no_mesh_unchanged_and_ttnn_untouched(self):
        runner = self._runner(mesh=None)
        # Any ttnn call would AttributeError.
        broken = types.SimpleNamespace()
        with patch.dict(sys.modules, {"ttnn": broken}):
            assert wrap_if_multihost(runner) is runner


# --------------------------------------------------------------------------- settings


class TestGalaxyQuadDevice:
    """DEVICE=galaxy_quad selects the (4, 32) config, as DEVICE=galaxy selects (4, 8)."""

    def _settings(self, monkeypatch, device):
        monkeypatch.setenv("MODEL", "MiniMax-H3-FL2VA")
        monkeypatch.setenv("DEVICE", device)
        for var in (
            "MODEL_RUNNER",
            "DEVICE_MESH_SHAPE",
            "SP_MESH_4X32",
            "SD_3_5_FAST",
            "SD_3_5_BASE",
            "TP2",
        ):
            monkeypatch.delenv(var, raising=False)
        with _fresh_project_modules():
            from config.settings import Settings

            return Settings()

    def test_galaxy_quad_is_4x32(self, monkeypatch):
        settings = self._settings(monkeypatch, "galaxy_quad")
        assert settings.model_runner == "tt-minimax-h3-fl2va"
        assert settings.device_mesh_shape == (4, 32)
        assert settings.max_batch_size == 1
        assert settings.is_galaxy is False
        assert settings.request_processing_timeout_seconds == 5000

    def test_galaxy_is_still_4x8(self, monkeypatch):
        assert self._settings(monkeypatch, "galaxy").device_mesh_shape == (4, 8)

    def test_every_quad_config_mirrors_its_4x8_config(self):
        from config.constants import DeviceTypes, ModelConfigs

        quad = {
            runner: config
            for (runner, device), config in ModelConfigs.items()
            if device == DeviceTypes.GALAXY_QUAD
        }
        assert quad
        for runner, config in quad.items():
            single = ModelConfigs.get((runner, DeviceTypes.GALAXY)) or ModelConfigs.get(
                (runner, DeviceTypes.BLACKHOLE_GALAXY)
            )
            assert single is not None, runner
            assert config["device_mesh_shape"] == (4, 32)
            assert {**config, "device_mesh_shape": (4, 8)} == single, runner


# --------------------------------------------------------------------------- main.py

# Run in a fresh interpreter: importing main.py registers Prometheus metrics and
# needs the real config.settings, which other test files replace with Mocks.
_LIFESPAN_FOLLOWER = """
import asyncio
from unittest.mock import MagicMock, patch
import main

class Exited(BaseException):
    pass

def fake_exit(code):
    raise Exited(code)

follower = MagicMock()
follower.return_value.run.return_value = 3
with patch.object(main, "is_follower_rank", return_value=True), \\
        patch.object(main, "launch_rank", return_value=2), \\
        patch.object(main, "service_resolver") as resolver, \\
        patch("multihost.follower.MultiHostLockstepFollower", follower), \\
        patch("multihost.follower.install_exit_signal_handlers") as handlers, \\
        patch.object(main.os, "_exit", side_effect=fake_exit):
    async def enter():
        async with main.lifespan(main.app):
            raise AssertionError("follower rank reached the serving phase")
    try:
        asyncio.run(enter())
        raise AssertionError("follower rank did not exit")
    except Exited as exited:
        assert exited.args[0] == 3, exited.args
follower.assert_called_once_with(2)
handlers.assert_called_once()
resolver.assert_not_called()
print("LIFESPAN-OK")
"""

_LIFESPAN_RANK_ZERO = """
import asyncio
from unittest.mock import AsyncMock, patch
import main

with patch.object(main, "is_follower_rank", return_value=False), \\
        patch.object(main, "service_resolver") as resolver, \\
        patch.object(main, "get_job_manager") as job_manager:
    job_manager.return_value.shutdown = AsyncMock()
    async def enter():
        async with main.lifespan(main.app):
            resolver.return_value.start_workers.assert_called_once()
    asyncio.run(enter())
resolver.return_value.stop_workers.assert_called_once()
print("LIFESPAN-OK")
"""


# CI installs the server's requirements without the model stacks main imports
# transitively; mock whichever of them (and their submodules) are missing.
_MOCK_MISSING_MODEL_STACKS = """
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from unittest.mock import MagicMock

_MISSING = {
    name
    for name in ("torch", "transformers", "diffusers", "torchvision", "ttnn")
    if importlib.util.find_spec(name) is None
}


class _MockFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in _MISSING:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = MagicMock()
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _MockFinder())
"""


class TestLifespan:
    def _run(self, script):
        import subprocess

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE")
        }
        completed = subprocess.run(
            [sys.executable, "-c", _MOCK_MISSING_MODEL_STACKS + script],
            cwd=_PROJECT_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr[-3000:]
        assert "LIFESPAN-OK" in completed.stdout

    def test_follower_rank_runs_follower_and_exits_before_serving(self):
        self._run(_LIFESPAN_FOLLOWER)

    def test_rank_zero_starts_workers(self):
        self._run(_LIFESPAN_RANK_ZERO)
