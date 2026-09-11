# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_single_docker.sh"
DOCKERFILE = REPO_ROOT / "vllm-tt-metal" / "vllm.tt-metal.src.dev.Dockerfile"


def _run_builder(
    cwd: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(BUILD_SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _builder_worktree(tmp_path: Path) -> Path:
    worktree = tmp_path / "tt-inference-server"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    (worktree / "VERSION").write_text("0.0.0\n")
    (worktree / "scripts").symlink_to(REPO_ROOT / "scripts", target_is_directory=True)
    (worktree / "workflows").symlink_to(
        REPO_ROOT / "workflows", target_is_directory=True
    )
    return worktree


@pytest.mark.parametrize("bad_ref", ["main", "A" * 40, "a" * 39])
def test_build_script_is_valid_bash_and_rejects_unpinned_quetzal_ref(tmp_path, bad_ref):
    subprocess.run(["bash", "-n", str(BUILD_SCRIPT)], check=True)

    result = _run_builder(
        _builder_worktree(tmp_path),
        "--quetzal-commit",
        bad_ref,
        "--quetzal-source-dir",
        str(REPO_ROOT),
    )

    assert result.returncode != 0
    assert "lowercase 40-hex commit" in result.stdout


def test_build_script_rejects_source_head_mismatch(tmp_path):
    result = _run_builder(
        _builder_worktree(tmp_path),
        "--quetzal-commit",
        "0" * 40,
        "--quetzal-source-dir",
        str(REPO_ROOT),
    )

    assert result.returncode != 0
    assert "source HEAD does not match" in result.stdout


def test_build_script_requires_external_source_with_pinned_commit(tmp_path):
    source_head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()

    result = _run_builder(_builder_worktree(tmp_path), "--quetzal-commit", source_head)

    assert result.returncode != 0
    assert "--quetzal-source-dir is required" in result.stdout


def test_quetzal_rebuild_pushes_when_remote_tag_already_exists(tmp_path):
    worktree = _builder_worktree(tmp_path)
    source_head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    docker_log = tmp_path / "docker.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        '#!/bin/bash\nprintf \'%s\\n\' "$*" >> "$DOCKER_LOG"\n'
        'if [ "$1 $2" = "build --help" ]; then echo --secret; fi\n'
    )
    fake_docker.chmod(0o755)
    env = {
        **os.environ,
        "DOCKER_LOG": str(docker_log),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }

    result = _run_builder(
        worktree,
        "--push",
        "--quetzal-commit",
        source_head,
        "--quetzal-source-dir",
        str(REPO_ROOT),
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    docker_commands = docker_log.read_text().splitlines()
    pushes = [line for line in docker_commands if line.startswith("push ")]
    builds = [
        line
        for line in docker_commands
        if line.startswith("build ") or line.startswith("buildx bake ")
    ]
    assert builds
    assert len(pushes) == 1
    assert docker_commands.index(builds[-1]) < docker_commands.index(pushes[0])
    assert f"-qz-{source_head[:12]}" in pushes[0]

    docker_log.write_text("")
    native = _run_builder(worktree, "--push", env=env)
    assert native.returncode == 0, native.stdout + native.stderr
    assert not any(
        line.startswith("push ") for line in docker_log.read_text().splitlines()
    )


def test_quetzal_image_build_contract_is_minimal_and_credential_free():
    build_script = BUILD_SCRIPT.read_text()
    dockerfile = DOCKERFILE.read_text()

    assert 'git -C "$TT_QUETZAL_SOURCE_DIR" archive --format=tar' in build_script
    assert '--secret "id=quetzal_source,src=${QUETZAL_SOURCE_ARCHIVE}"' in build_script
    assert 'QUETZAL_IMAGE_SUFFIX="-qz-${TT_QUETZAL_COMMIT_SHA:0:12}"' in build_script
    assert "DOCKER_BUILDKIT=1 docker build --help" in build_script
    assert "export DOCKER_BUILDKIT=1" in build_script
    assert 'if [[ -n "$TT_QUETZAL_COMMIT_SHA" ]]; then' in build_script
    assert "forcing an exact image rebuild" in build_script
    assert "build=true" in build_script
    assert "force_push=true" in build_script

    assert "RUN --mount=type=secret,id=quetzal_source,required=false" in dockerfile
    assert "uv build --wheel --out-dir /tmp/quetzal-wheel" in dockerfile
    assert 'uv pip install --no-cache-dir --no-deps "$1"' in dockerfile
    assert "import serving.artifact_discovery" in dockerfile
    assert "import tt_quetzalcoatlus.vllm_plugin" in dockerfile
    assert 'e.name == "quetzal_model_registry"' in dockerfile
    assert dockerfile.count("import serving.artifact_discovery") == 2
    assert (
        "org.opencontainers.image.quetzal.revision=${TT_QUETZAL_COMMIT_SHA}"
        in dockerfile
    )
    assert "TT_QUETZAL_COMMIT_SHA=${TT_QUETZAL_COMMIT_SHA}" in dockerfile

    combined = build_script + dockerfile
    assert "tt-quetzalcoatlus.git" not in combined
    assert "mesh_graph_descriptor" not in combined
    assert "PATCHSET" not in combined
