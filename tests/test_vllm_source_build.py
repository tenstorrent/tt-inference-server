# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Source-selection tests: only mocked Docker calls, never image builds."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_single_docker.sh"
SHA = "7f72b1c6e905f5137fe3377f2e7b42738d3f271d"
TT_METAL_SNAPSHOT_SHA = "919c110d3d4331b7753c1db78618e879905ae46d"


@pytest.mark.parametrize("repository", [None, "tenstorrent/vllm-tt-plugin", "tenstorrent/vllm"])
def test_source_repository_and_full_sha_reach_mocked_build(tmp_path, repository):
    worktree = tmp_path / "tt-inference-server"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    (worktree / "VERSION").write_text("0.0.0\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "docker.log"
    # Returning success for inspect avoids any tt-metal clone/build path.
    docker = fake_bin / "docker"
    docker.write_text('#!/bin/bash\nprintf \'%s\\n\' "$*" >> "$MOCK_DOCKER_LOG"\n')
    docker.chmod(0o755)
    # Catalog generation is unrelated to source forwarding; do not import it.
    python = fake_bin / "python3"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    args = ["bash", str(SCRIPT), "--force-build", "--vllm-commit", SHA]
    if repository is not None:
        args += ["--vllm-repository", repository]
    result = subprocess.run(
        args, cwd=worktree, capture_output=True, text=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "MOCK_DOCKER_LOG": str(log)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    builds = [line for line in log.read_text().splitlines() if line.startswith("build ")]
    assert len(builds) == 1
    assert f"TT_VLLM_REPOSITORY={repository or 'tenstorrent/vllm-tt-plugin'}" in builds[0]
    assert f"TT_VLLM_COMMIT_SHA_OR_TAG={SHA}" in builds[0]
    engine = "0.26.0+empty" if repository == "tenstorrent/vllm" else "installer-defined"
    assert f"TT_VLLM_ENGINE_VERSION={engine}" in builds[0]
    assert (f"monorepo-{SHA}" in builds[0]) == (repository == "tenstorrent/vllm")
    assert not any(line.startswith("push ") for line in log.read_text().splitlines())


@pytest.mark.parametrize("repository", ["other/vllm", "tenstorrent/vllm;false", ""])
def test_invalid_repository_fails_before_docker(tmp_path, repository):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--vllm-repository", repository],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Unsupported vLLM repository" in result.stderr


def test_dockerfile_preserves_both_source_installers_and_full_editable_tree():
    text = (ROOT / "vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile").read_text()
    assert "ARG TT_VLLM_REPOSITORY=tenstorrent/vllm-tt-plugin" in text
    assert "https://github.com/${TT_VLLM_REPOSITORY}.git" in text
    assert "bash /tmp/vllm-bundled-install/install_vllm_bundled_plugin.sh" in text
    assert "source plugins/vllm-tt-plugin/docs/install-vllm-tt.sh" not in text
    assert "else source docs/install-vllm-tt.sh" in text
    assert "google_gemma_4_26b_a4b_it/vllm_plugin_snapshot" in text
    assert "SOURCE_MANIFEST.json" in text
    assert "TT_VLLM_PLUGIN_SNAPSHOT_COMMIT" in text
    assert "${vllm_tt_plugin_dir} ${vllm_tt_plugin_dir}" in text
    assert "com.tenstorrent.vllm.repository=${TT_VLLM_REPOSITORY}" in text
    assert "com.tenstorrent.vllm.revision=${TT_VLLM_COMMIT_SHA_OR_TAG}" in text
    assert "git rev-parse HEAD" in text and "^[0-9a-f]{40}$" in text


def test_bundled_plugin_preserves_measured_installed_engine_recipe():
    import json

    helper = ROOT / "scripts/install_vllm_bundled_plugin.sh"
    subprocess.run(["bash", "-n", str(helper)], check=True)
    source = helper.read_text()
    manifest = json.loads((ROOT / "scripts/vllm_bundled_plugin_manifest.json").read_text())
    assert manifest["installer_revision"] == "c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42"
    assert manifest["engine_requirement"] == "vllm==0.26.0"
    assert manifest["engine_version"] == "0.26.0+empty"
    assert "sha256sum --check --status" in source
    assert 'VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-binary vllm "${provenance[4]}"' in source
    assert 'uv pip install --no-deps -e "$plugin_project_dir"' in source
    assert 'cd "$probe_tmp"' in source  # do not import an uninstalled engine from checkout cwd
    assert '"site-packages/vllm/" in engine.origin' in source
    assert "uv pip install -e ." not in source
    assert "--constraint" in source


def test_exact_tt_metal_publication_selects_plugin_snapshot(tmp_path):
    worktree = tmp_path / "tt-inference-server"
    worktree.mkdir()
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    (worktree / "VERSION").write_text("0.0.0\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "docker.log"
    docker = fake_bin / "docker"
    docker.write_text('#!/bin/bash\nprintf \'%s\\n\' "$*" >> "$MOCK_DOCKER_LOG"\n')
    docker.chmod(0o755)
    python = fake_bin / "python3"
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--force-build",
            "--tt-metal-commit",
            TT_METAL_SNAPSHOT_SHA,
            "--vllm-commit",
            "c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42",
        ],
        cwd=worktree,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "MOCK_DOCKER_LOG": str(log)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    build = next(line for line in log.read_text().splitlines() if line.startswith("build "))
    assert f"TT_VLLM_PLUGIN_SNAPSHOT_COMMIT={SHA}" in build
    assert "TT_VLLM_ENGINE_VERSION=0.26.0+empty" in build
    assert f"ttmetal-{SHA[:12]}" in build
