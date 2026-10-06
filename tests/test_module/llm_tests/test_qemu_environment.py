# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor.environments.docker.docker")
from harbor.environments.docker.docker import DockerEnvironment
from llm_module.agentic.qemu_environment import QemuArchiveDockerEnvironment


def environment(name, image="alexgshaw/qemu-startup:20251031", return_code=0):
    env = object.__new__(QemuArchiveDockerEnvironment)
    env.environment_name = name
    env.task_env_config = SimpleNamespace(docker_image=image)
    env.exec = AsyncMock(return_value=SimpleNamespace(return_code=return_code, stderr="failure"))
    return env


@pytest.mark.parametrize("name", [
    "break-filter-js-from-html", "cobol-modernization", "compile-compcert",
    "feal-differential-cryptanalysis", "caffe-cifar-10", "password-recovery",
    "portfolio-optimization", "hf-model-inference", "financial-document-processor",
])
def test_other_tasks_do_not_execute_repair(monkeypatch, name):
    start = AsyncMock()
    monkeypatch.setattr(DockerEnvironment, "start", start)
    env = environment(name)
    asyncio.run(env.start(False))
    start.assert_awaited_once_with(False)
    env.exec.assert_not_called()


def test_exact_qemu_repairs_only_apt_sources(monkeypatch):
    start = AsyncMock()
    monkeypatch.setattr(DockerEnvironment, "start", start)
    env = environment("qemu-startup")
    asyncio.run(env.start(False))
    start.assert_awaited_once_with(False)
    env.exec.assert_awaited_once()
    assert env.exec.call_args.kwargs["user"] == "root"
    assert env.exec.call_args.kwargs["command"] == (
        "sed -i -e s,http://,https://,g "
        "-e s,deb.debian.org/debian-security,archive.debian.org/debian-security,g "
        "/etc/apt/sources.list"
    )


def test_unvalidated_image_fails_before_start(monkeypatch):
    start = AsyncMock()
    monkeypatch.setattr(DockerEnvironment, "start", start)
    env = environment("qemu-startup", image="different-image")
    with pytest.raises(RuntimeError, match="validated task image"):
        asyncio.run(env.start(False))
    start.assert_not_called()
    env.exec.assert_not_called()


def test_repair_failure_is_visible(monkeypatch):
    monkeypatch.setattr(DockerEnvironment, "start", AsyncMock())
    with pytest.raises(RuntimeError, match="apt-source repair failed"):
        asyncio.run(environment("qemu-startup", return_code=1).start(False))
