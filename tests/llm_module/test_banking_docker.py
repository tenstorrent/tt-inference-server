# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from llm_module.agentic import harbor
from llm_module.agentic.banking_docker import (
    TAU2_REVISION,
    docker_command,
    prepare_docker_path,
)

DOCKERFILE = (
    "FROM python:3.12-slim\nENV TAU2_BENCH_ROOT=/opt/tau2-bench\n"
    'RUN git clone --depth=1 "${TAU2_BENCH_REPO}" "${TAU2_BENCH_ROOT}"\n'
)


@pytest.fixture
def docker_adapter(tmp_path):
    # Exercise the actual executable wrapper, including shell quoting and exec.
    bin_dir = tmp_path / "real docker's bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        "print(json.dumps(sys.argv[1:]))\n"
        "sys.exit(19 if sys.argv[1:] == ['fail'] else 0)\n"
    )
    docker.chmod(0o755)
    directory = tmp_path / "job's adapter"
    path = prepare_docker_path(directory, Path(sys.executable), str(bin_dir))

    def run(args):
        return subprocess.run(
            ["docker", *args],
            env={**os.environ, "PATH": path},
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

    return run, directory


def task_compose(tmp_path, name, dockerfile=DOCKERFILE):
    task = tmp_path / name
    environment = task / "environment"
    environment.mkdir(parents=True)
    (task / "task.toml").write_text(f'[task]\nname = "{name}"\n')
    (environment / "Dockerfile").write_text(dockerfile)
    runtime = environment / "runtime-server"
    runtime.mkdir()
    (runtime / "Dockerfile").write_text(DOCKERFILE)
    return [
        "compose",
        "--project-directory",
        str(environment),
        "-f",
        "base.yaml",
        "-f",
        "resources.yaml",
    ]


def test_banking_build_overlay_and_concurrent_commands(tmp_path, docker_adapter):
    run, directory = docker_adapter
    original = DOCKERFILE + 'RUN echo "$HOME"\n'
    args = task_compose(
        tmp_path,
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-001",
        original,
    )
    commands = [
        ["build"],
        ["up", "--detach", "--wait"],
        ["exec", "main", "rm", "-f", "file"],
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda command: run([*args, *command]), commands * 2))
    overlay = None
    for result, command in zip(results, commands * 2):
        assert result.returncode == 0, result.stderr
        forwarded = json.loads(result.stdout)
        assert forwarded[: len(args)] == args
        assert forwarded[len(args)] == "-f"
        current = Path(forwarded[len(args) + 1])
        assert overlay is None or current == overlay
        overlay = current
        assert forwarded[len(args) + 2 :] == command
    contents = json.loads(overlay.read_text())
    assert set(contents["services"]) == {"main", "tau3-runtime"}
    dockerfile = contents["services"]["main"]["build"]["dockerfile_inline"]
    assert 'RUN echo "$$HOME"' in dockerfile
    for service in contents["services"].values():
        pinned = service["build"]["dockerfile_inline"]
        assert f'fetch --depth=1 "$${{TAU2_BENCH_REPO}}" {TAU2_REVISION}' in pinned
        assert "checkout --detach FETCH_HEAD" in pinned
        assert "git clone" not in pinned
        assert "websockets" not in pinned
        assert "RUN python3 -c 'import tau2.evaluator.evaluator'" in pinned
    assert (Path(args[2]) / "Dockerfile").read_text() == original
    assert (Path(args[2]) / "runtime-server/Dockerfile").read_text() == DOCKERFILE
    assert len(list((directory / "overlays").iterdir())) == 1


@pytest.mark.parametrize(
    "name", ["terminal-bench-task", "swebench-task", "tau3-retail-task"]
)
def test_other_tasks_pass_through(tmp_path, docker_adapter, name):
    run, directory = docker_adapter
    args = [*task_compose(tmp_path, name), "build"]
    result = run(args)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == args
    assert not (directory / "overlays").exists()


def test_plain_docker_and_exit_code_pass_through(docker_adapter):
    run, _ = docker_adapter
    result = run(["version"])
    assert result.returncode == 0
    assert json.loads(result.stdout) == ["version"]
    assert run(["fail"]).returncode == 19


def test_unknown_banking_image_fails_before_docker(tmp_path, docker_adapter):
    run, directory = docker_adapter
    args = task_compose(
        tmp_path,
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-001",
        "FROM changed\n",
    )
    result = run([*args, "build"])
    assert result.returncode != 0
    assert "Unrecognized Banking Dockerfile" in result.stderr
    assert not result.stdout
    assert not (directory / "overlays").exists()


@pytest.mark.skipif(
    shutil.which("docker") is None, reason="Docker CLI is not installed"
)
def test_compose_accepts_overlay_without_changing_task_configuration(tmp_path):
    original = DOCKERFILE + 'RUN echo "$HOME"\n'
    args = task_compose(
        tmp_path,
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-001",
        original,
    )
    environment = Path(args[2])
    base = environment / "compose.yaml"
    base.write_text(
        "services:\n"
        "  main:\n"
        "    build:\n"
        "      context: .\n"
        "    environment:\n"
        "      TAU2_USER_MODEL: example\n"
        "  tau3-runtime:\n"
        "    build:\n"
        "      context: ./runtime-server\n"
        "    environment:\n"
        "      TAU2_USER_MODEL: example\n"
        "    command: [python3, server.py]\n"
        "  unrelated:\n"
        "    image: example/unchanged\n"
    )
    # Compose config validates and merges files without starting Docker containers.
    command = docker_command(
        [*args[:3], "-f", str(base), "config", "--format", "json"],
        shutil.which("docker"),
        tmp_path / "overlays",
    )
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    services = json.loads(result.stdout)["services"]
    assert services["unrelated"]["image"] == "example/unchanged"
    assert services["main"]["environment"] == {"TAU2_USER_MODEL": "example"}
    assert services["tau3-runtime"]["environment"] == {"TAU2_USER_MODEL": "example"}
    assert services["tau3-runtime"]["command"] == ["python3", "server.py"]
    assert services["tau3-runtime"]["build"]["context"] == str(
        environment / "runtime-server"
    )
    assert services["main"]["build"]["context"] == str(environment)
    dockerfile = services["main"]["build"]["dockerfile_inline"]
    # Compose's serialized config retains dollar escaping for round trips.
    assert 'RUN echo "$$HOME"' in dockerfile
    assert TAU2_REVISION in dockerfile
    assert TAU2_REVISION in services["tau3-runtime"]["build"]["dockerfile_inline"]


def test_changed_runtime_dockerfile_fails_before_docker(tmp_path, docker_adapter):
    run, directory = docker_adapter
    args = task_compose(
        tmp_path, "sierra-research/tau3-bench__tau3-banking_knowledge-task-001"
    )
    (Path(args[2]) / "runtime-server/Dockerfile").write_text("FROM changed\n")
    result = run([*args, "build"])
    assert result.returncode != 0
    assert "Unrecognized Banking Dockerfile" in result.stderr
    assert not result.stdout
    assert not (directory / "overlays").exists()


@pytest.mark.parametrize(
    "dataset,environment,adapted",
    [
        ("sierra-research/tau3-bench", "docker", True),
        ("sierra-research/tau3-bench@1.0", "docker", True),
        ("sierra-research/tau3-bench", "kubernetes", False),
        ("terminal-bench/terminal-bench-2", "docker", False),
        ("swebench-verified", "docker", False),
    ],
)
def test_harbor_scopes_adapter_to_tau3_docker(
    tmp_path, monkeypatch, dataset, environment, adapted
):
    config = harbor.HarborRunConfig(
        task_name="test",
        dataset=dataset,
        agent="test",
        model_name="test",
        jobs_dir=tmp_path,
        api_base="http://localhost:8000/v1",
        n_concurrent_trials=1,
        n_attempts=1,
        environment_type=environment,
        agent_kwargs={},
        n_tasks=1,
        override_cpus=None,
        override_memory_mb=None,
        timeout_multiplier=None,
        agent_timeout_sec=None,
        venv_python=Path("/harbor-venv/bin/python"),
    )
    captured = {}

    def prepare(directory, interpreter, path):
        assert directory == tmp_path / "test_banking_docker"
        assert interpreter == config.venv_python
        assert path == os.environ["PATH"]
        return "adapted-path"

    def run(cmd, **kwargs):
        captured.update(kwargs)
        return 19

    original_path = os.environ["PATH"]
    monkeypatch.setattr(harbor, "prepare_docker_path", prepare)
    monkeypatch.setattr(harbor, "run_with_progress", run)
    assert harbor.run(config) == 19
    assert captured["env"]["PATH"] == ("adapted-path" if adapted else original_path)
    assert os.environ["PATH"] == original_path
    assert not (
        tmp_path / "test"
    ).exists()  # Harbor owns creation of its job directory.
