# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from types import SimpleNamespace

from workflows.run_docker_server import run_docker_command


def test_run_docker_command_retains_foreground_process(monkeypatch, tmp_path):
    process = SimpleNamespace(poll=lambda: None)
    callbacks = []
    monkeypatch.setattr(
        "workflows.run_docker_server.subprocess.Popen", lambda *args, **kwargs: process
    )
    monkeypatch.setattr(
        "workflows.run_docker_server.subprocess.check_output",
        lambda *args, **kwargs: "container-id\n",
    )
    monkeypatch.setattr("workflows.run_docker_server.atexit.register", callbacks.append)

    payload = run_docker_command(
        ["docker", "run", "image"],
        "container-name",
        SimpleNamespace(workflow="server", service_port="8000"),
        SimpleNamespace(docker_image="image"),
        tmp_path / "docker.log",
    )

    assert payload["process"] is process
    callbacks[0]()


class FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout):
        return self.returncode


def _launch_fixture(monkeypatch, tmp_path, *, visible_at=None, returncode=None):
    import workflows.run_docker_server as launcher

    clock = [0.0]
    process = FakeProcess(returncode)
    callbacks, removals, queries, logs = [], [], [], []

    def popen(*args, **kwargs):
        logs.append(kwargs["stdout"])
        return process

    def query(command, **kwargs):
        queries.append(command)
        assert kwargs["timeout"] <= 10
        return "owned-id" if visible_at is not None and clock[0] >= visible_at else ""

    monkeypatch.setenv("TT_SERVER_READY_TIMEOUT_SECONDS", "60")
    monkeypatch.setattr(launcher.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(launcher.time, "time", lambda: clock[0])
    monkeypatch.setattr(
        launcher.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(launcher.subprocess, "check_output", query)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, **kwargs: (
            removals.append(command) or SimpleNamespace(returncode=0)
        ),
    )
    monkeypatch.setattr(launcher.atexit, "register", callbacks.append)

    def launch():
        return run_docker_command(
            ["docker", "run", "image"],
            "owned-name",
            SimpleNamespace(workflow="server", service_port="8000"),
            SimpleNamespace(docker_image="image"),
            tmp_path / "docker.log",
        )

    return launch, process, clock, callbacks, removals, queries, logs


def test_delayed_container_start_remains_supervised(monkeypatch, tmp_path):
    launch, process, clock, callbacks, removals, queries, logs = _launch_fixture(
        monkeypatch, tmp_path, visible_at=45
    )
    assert launch()["container_id"] == "owned-id"
    assert clock[0] == 45
    assert not process.terminated
    assert not removals
    assert all("name=^/owned-name$" in command for command in queries)
    callbacks[0]()
    assert logs[0].closed


def test_immediate_cli_failure_preserves_logs_and_closes_attempt(monkeypatch, tmp_path):
    import pytest
    from workflows.run_docker_server import DockerLaunchError

    launch, process, clock, callbacks, removals, queries, logs = _launch_fixture(
        monkeypatch, tmp_path, returncode=125
    )
    with pytest.raises(DockerLaunchError, match="status 125") as failure:
        launch()
    assert failure.value.retry_safe is False
    assert clock[0] == 0
    assert not queries
    assert not process.terminated
    assert removals == [["docker", "rm", "-f", "owned-name"]]
    assert not callbacks
    assert logs[0].closed
    assert (tmp_path / "docker.log").exists()


def test_timeout_cancels_only_owned_attempt_and_disallows_retry(monkeypatch, tmp_path):
    import pytest
    from workflows.run_docker_server import DockerLaunchError

    launch, process, clock, callbacks, removals, queries, logs = _launch_fixture(
        monkeypatch, tmp_path
    )
    with pytest.raises(DockerLaunchError, match="timed out") as failure:
        launch()
    assert failure.value.retry_safe is False
    assert clock[0] == 60
    assert process.terminated
    assert removals == [["docker", "rm", "-f", "owned-name"]]
    assert logs[0].closed
    assert not callbacks


def test_cleanup_failure_still_disallows_retry(monkeypatch, tmp_path):
    import pytest
    from workflows.run_docker_server import DockerLaunchError

    launch, process, clock, callbacks, removals, queries, logs = _launch_fixture(
        monkeypatch, tmp_path
    )

    def failed_cleanup(*args, **kwargs):
        raise TimeoutError("daemon unresponsive")

    monkeypatch.setattr("workflows.run_docker_server.subprocess.run", failed_cleanup)
    with pytest.raises(DockerLaunchError) as failure:
        launch()
    assert failure.value.retry_safe is False
    assert process.terminated
    assert logs[0].closed


def test_server_command_does_not_retry_unreconciled_launch(monkeypatch):
    from workflow_module.commands import ServerCommand, ServerLaunchSpec, ServerMode
    from workflows.run_docker_server import DockerLaunchError

    attempts = []

    def fail(self, spec):
        attempts.append(spec)
        raise DockerLaunchError("pending daemon creation")

    monkeypatch.setattr(ServerCommand, "_launch_once", fail)
    monkeypatch.setenv("TT_SERVER_BOOT_ATTEMPTS", "2")
    spec = ServerLaunchSpec(
        mode=ServerMode.DOCKER, model_spec=None, runtime_config=None, setup_config=None
    )
    result = ServerCommand(spec).execute()
    assert result.return_code == 1
    assert len(attempts) == 1
