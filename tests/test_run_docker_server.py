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
