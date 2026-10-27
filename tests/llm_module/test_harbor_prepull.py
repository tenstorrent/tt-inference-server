# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""``harbor.prepull_images``: base images are pulled, with retries, before ``harbor run``.

On the Blackhole Galaxy runner every tau3-bench trial's environment build pulled
``python:3.12-slim`` itself, at ~11 KB/s with resets, and all four builds failed
together. One retried pull ahead of the run leaves the builds a local image.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_module.agentic import harbor


class FakeDocker:
    """``docker image inspect`` / ``docker pull`` with scripted outcomes."""

    def __init__(self, local=(), pull_results=()):
        self.local = set(local)
        self.pull_results = list(pull_results)  # per pull: returncode, or an exception
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0 if cmd[3] in self.local else 1)
        assert cmd[:2] == ["docker", "pull"], cmd
        outcome = self.pull_results.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == 0:
            self.local.add(cmd[2])
        return subprocess.CompletedProcess(cmd, outcome)

    def pulls(self):
        return [c[2] for c in self.calls if c[:2] == ["docker", "pull"]]


@pytest.fixture
def no_sleep():
    with patch.object(harbor.time, "sleep") as sleep:
        yield sleep


def _prepull(fake, images, attempts=3):
    with patch.object(harbor.subprocess, "run", fake):
        return harbor.prepull_images(images, attempts=attempts, timeout_sec=60)


def test_local_image_is_not_pulled(no_sleep):
    fake = FakeDocker(local={"python:3.12-slim"})
    assert _prepull(fake, ["python:3.12-slim"]) == []
    assert fake.pulls() == []


def test_failed_pulls_are_retried_until_one_succeeds(no_sleep):
    timeout = subprocess.TimeoutExpired(["docker", "pull"], 60)
    fake = FakeDocker(pull_results=[1, timeout, 0])
    assert _prepull(fake, ["python:3.12-slim"]) == []
    assert fake.pulls() == ["python:3.12-slim"] * 3
    assert no_sleep.call_count == 2


def test_an_image_that_never_pulls_is_reported_and_the_rest_still_pull(no_sleep):
    fake = FakeDocker(pull_results=[1, 1, 0])
    assert _prepull(fake, ["broken:1", "python:3.12-slim"], attempts=2) == ["broken:1"]
    assert fake.pulls() == ["broken:1", "broken:1", "python:3.12-slim"]


def test_missing_docker_cli_is_reported_without_retrying(no_sleep):
    def no_docker(cmd, **kwargs):
        raise FileNotFoundError("docker")

    with patch.object(harbor.subprocess, "run", no_docker):
        assert harbor.prepull_images(
            ["python:3.12-slim"], attempts=3, timeout_sec=60
        ) == ["python:3.12-slim"]
    no_sleep.assert_not_called()


def _config(tmp_path, environment_type, images):
    return harbor.HarborRunConfig(
        task_name="tau3",
        dataset="sierra-research/tau3-bench",
        agent="tau3_llm_agent",
        model_name="m",
        jobs_dir=tmp_path / "jobs",
        api_base="http://localhost:8000/v1",
        n_concurrent_trials=4,
        n_attempts=1,
        environment_type=environment_type,
        agent_kwargs={},
        n_tasks=None,
        override_cpus=None,
        override_memory_mb=None,
        timeout_multiplier=None,
        agent_timeout_sec=None,
        venv_python=Path("/opt/venv/bin/python"),
        prepull_images=images,
    )


@pytest.mark.parametrize(
    "environment_type, images, expected",
    [
        ("docker", ["python:3.12-slim"], [["python:3.12-slim"]]),
        ("docker", [], []),
        # Pods pull on their nodes, not through this host's daemon.
        ("kubernetes", ["python:3.12-slim"], []),
    ],
)
def test_run_prepulls_before_harbor_on_docker_only(
    tmp_path, environment_type, images, expected
):
    order = []
    with patch.object(
        harbor,
        "prepull_images",
        side_effect=lambda imgs, *a: order.append(list(imgs)) or [],
    ), patch.object(
        harbor,
        "run_with_progress",
        side_effect=lambda *a, **k: order.append("harbor") or 1,
    ):
        assert harbor.run(_config(tmp_path, environment_type, images)) == 1
    assert order == expected + ["harbor"]
