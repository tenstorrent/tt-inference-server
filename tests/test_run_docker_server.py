# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from types import SimpleNamespace

import pytest

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


def test_slow_docker_start_does_not_trigger_a_second_launch(monkeypatch, tmp_path):
    now = [0.0]
    process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(
        "workflows.run_docker_server.subprocess.Popen", lambda *a, **k: process
    )
    monkeypatch.setattr("workflows.run_docker_server.time.time", lambda: now[0])
    monkeypatch.setattr(
        "workflows.run_docker_server.time.sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    monkeypatch.setattr(
        "workflows.run_docker_server.subprocess.check_output",
        lambda *a, **k: "container-id\n" if now[0] >= 44 else "",
    )
    monkeypatch.setattr(
        "workflows.run_docker_server.atexit.register", lambda callback: None
    )
    payload = run_docker_command(
        ["docker", "run", "image"],
        "owned-container",
        SimpleNamespace(workflow="server", service_port="8000"),
        SimpleNamespace(docker_image="image"),
        tmp_path / "docker.log",
    )
    assert payload["container_id"] == "container-id"
    assert now[0] == 44


@pytest.mark.parametrize("exited", [False, True])
def test_failed_docker_start_removes_only_its_own_container(
    monkeypatch, tmp_path, exited
):
    now = [0.0]
    events = []
    process = SimpleNamespace(
        poll=lambda: 1 if exited else None,
        terminate=lambda: events.append("terminate"),
        wait=lambda **kwargs: events.append("wait"),
    )
    monkeypatch.setattr(
        "workflows.run_docker_server.subprocess.Popen", lambda *a, **k: process
    )
    monkeypatch.setattr("workflows.run_docker_server.time.time", lambda: now[0])
    monkeypatch.setattr(
        "workflows.run_docker_server.time.sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    monkeypatch.setattr(
        "workflows.run_docker_server.subprocess.check_output", lambda *a, **k: ""
    )
    monkeypatch.setattr(
        "workflows.run_docker_server.subprocess.run",
        lambda args, **kwargs: events.append(args),
    )
    with pytest.raises(RuntimeError, match="Docker container failed to start"):
        run_docker_command(
            ["docker", "run", "image"],
            "owned-container",
            SimpleNamespace(workflow="server", service_port="8000"),
            SimpleNamespace(docker_image="image"),
            tmp_path / "docker.log",
        )
    assert events[0] == ["docker", "rm", "-f", "owned-container"]
    assert events[-1] == "wait"
    assert now[0] == (0 if exited else 180)
