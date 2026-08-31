# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "vllm-tt-metal" / "vllm.tt-metal.src.quetzal.Dockerfile"
BUILD_SCRIPT = ROOT / "scripts" / "build_quetzal_dev_image.sh"
RUNNER = ROOT / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"


def test_quetzal_derivative_keeps_third_runtime_out_of_standard_image_identity():
    source = DOCKERFILE.read_text()
    assert "ARG TT_INFERENCE_SERVER_BASE_IMAGE" in source
    assert "FROM ${TT_INFERENCE_SERVER_BASE_IMAGE}" in source
    assert "ARG TT_QUETZAL_COMMIT_SHA" in source
    assert "ARG TT_INFERENCE_SERVER_COMMIT_SHA" in source
    assert "org.opencontainers.image.tt-inference-server.revision" in source
    assert "org.opencontainers.image.quetzal.revision" in source
    assert "@sha256:[0-9a-f]{64}" in source
    assert "^[0-9a-f]{40}$" in source


def test_quetzal_is_installed_non_editably_and_entry_point_is_verified():
    source = DOCKERFILE.read_text()
    runner = RUNNER.read_text()
    install_line = next(
        line
        for line in source.splitlines()
        if 'uv pip install --python "${PYTHON_ENV_DIR}/bin/python" --no-deps '
        "/tmp/quetzal-source" in line
    )
    assert " -e " not in install_line
    assert "> /tmp/packages.before" in source
    assert "> /tmp/packages.after" in source
    assert "cmp /tmp/packages.before /tmp/packages.after" in source
    assert "--requirements /tmp/quetzal-serve-requirements.txt" in source
    assert "--upgrade --no-deps --requirements" in source
    assert source.count('uv pip check --python "${PYTHON_ENV_DIR}/bin/python"') >= 3
    assert (
        'test -z "$(comm -13 /tmp/pip-check.base /tmp/pip-check.qualified)"' in source
    )
    assert "cmp /tmp/pip-check.qualified /tmp/pip-check.after" in source
    assert "cmp /tmp/pip-check.base /tmp/pip-check.qualified" not in source
    assert '&& /bin/bash -c "export VIRTUAL_ENV=' not in source
    assert "quetzal_model_registry" in source
    assert "tt_quetzalcoatlus.vllm_plugin:register" in source
    assert "import serving.artifact_bundle" in source
    assert "cat /tmp/quetzal-source/.tt-quetzal-commit" in source
    assert "COPY --from=quetzal_src" in source
    assert "COPY --from=ttis_src" in source
    assert "vllm-tt-metal/src/run_vllm_api_server.py" in source
    assert "/home/container_app_user/app/src/run_vllm_api_server.py" in source
    assert "model_spec.json" in source
    assert "/home/container_app_user/model_specs/model_spec.json" in source
    assert "grep -q '^def validate_quetzal_runtime('" in source
    assert "def validate_quetzal_runtime(" in runner
    assert "validate_quetzal_runtime_contract" not in source
    assert "d71abb2865d94511a1aaafbb02fabe1adfc5bd658ff9b876412f5f558111db4a" in source
    assert "e3ecc5557a84955bf0b95615e4b8e9fa83bcc431c9755e969ba5c441fc8d94cf" in source


def test_quetzal_runtime_is_rebuilt_in_base_abi_and_atomically_replaced():
    source = DOCKERFILE.read_text()
    revision = "b534549300fe2af11e6ee828675294bc0e359555"
    patchset_v1_sha = "22fb0bd2523b8a5c63fa20c3c8a1586dc9ead5150449d0eb02231fa8173a7edd"
    patchset_v2_sha = "e240fa3880ea0c2597dd7df8ab657a69aca9fe215de58220ae96e47a48a29910"

    assert "AS quetzal_ttmetal_builder" in source
    assert source.count("FROM ${TT_INFERENCE_SERVER_BASE_IMAGE}") == 2
    assert revision in source
    assert patchset_v1_sha in source
    assert patchset_v2_sha in source
    assert "unrecognized Quetzal TT-Metal patchset identity" in source
    assert "COPY --from=quetzal_src patches/tt-metal/" in source
    assert "COPY --from=quetzal_src tools/tt_metal_patchset.py" in source
    assert source.index('git -C "${TT_METAL_HOME}" fetch') < source.index(
        'git -C "${TT_METAL_HOME}" checkout --detach'
    )
    assert "--apply" in source
    assert source.count("tt_metal_patchset.py") >= 3
    assert (
        "--mount=type=cache,id=quetzal-tt-metal-cpm,"
        "target=/root/.cache/tt-metal-cpm,sharing=locked" in source
    )
    assert "CPM_SOURCE_CACHE=/root/.cache/tt-metal-cpm" in source
    assert "cmp /tmp/packages.before /tmp/packages.after" in source
    builder = source.split("FROM ${TT_INFERENCE_SERVER_BASE_IMAGE}", 2)[1]
    assert "> /tmp/pip-check.before" in builder
    assert "> /tmp/pip-check.after" in builder
    assert "cmp /tmp/pip-check.before /tmp/pip-check.after" in builder
    assert "LC_ALL=C sort > /tmp/pip-check" in builder
    assert "import ttnn, ttnn._ttnn, vllm" in source
    whiteout = "RUN rm -rf /home/container_app_user/tt-metal"
    runtime_copy = "COPY --from=quetzal_ttmetal_builder"
    assert source.index(whiteout) < source.index(runtime_copy)
    assert "/var/tmp/nkapre" not in source
    assert "org.opencontainers.image.tt-metal.revision" in source
    assert "org.opencontainers.image.tt-metal.patchset.sha256" in source
    assert "org.opencontainers.image.tt-metal.patchset.manifest.sha256" in source
    assert "ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_BASE_REVISION}" in source
    assert "ENV TT_METAL_PATCHSET_SHA256=${TT_METAL_PATCHSET_SHA256}" in source


def test_quetzal_runner_skips_native_registration_and_validates_package():
    source = RUNNER.read_text()
    assert "validate_quetzal_runtime(model_spec)" in source
    assert "Skipping native TT model registration for Quetzal artifact" in source
    assert "from vllm import ModelRegistry" in source
    assert source.index("validate_quetzal_runtime(model_spec)") < source.index(
        "register_tt_models(impl_id)"
    )


def test_build_wrapper_requires_digest_base_and_full_quetzal_commit():
    assert BUILD_SCRIPT.read_text().splitlines()[0] == "#!/usr/bin/env bash"
    subprocess.run(["bash", "-n", str(BUILD_SCRIPT)], check=True)
    help_result = subprocess.run(
        ["bash", str(BUILD_SCRIPT), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "IMAGE@sha256:DIGEST" in help_result.stdout
    assert "FULL_COMMIT_SHA" in help_result.stdout
    assert "PATH_TO_CLEAN_GIT_CHECKOUT" in help_result.stdout

    direct_result = subprocess.run(
        [str(BUILD_SCRIPT), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert direct_result.stdout == help_result.stdout


def test_build_wrapper_refuses_mutable_base_before_invoking_docker():
    result = subprocess.run(
        [
            "bash",
            str(BUILD_SCRIPT),
            "--base-image",
            "example.invalid/ttis:latest",
            "--quetzal-commit",
            "a" * 40,
            "--tag",
            "example.invalid/quetzal:test",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "pinned by an sha256 digest" in result.stderr


def _git_source(tmp_path):
    source = tmp_path / "quetzal"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n")
    serving_dir = source / "serving"
    serving_dir.mkdir()
    (serving_dir / "qualified_environments.json").write_text(
        json.dumps(
            {
                "schema": "quetzal.qualified-environments.v2",
                "base": {
                    "dependencies": {
                        "transformers": "5.15.0",
                        "numpy": "1.26.4",
                    }
                },
                "variants": {
                    "qwen36.serve": {
                        "model_ids": ["Qwen/Qwen3.6-27B"],
                        "lane": "serve",
                        "overrides": {},
                    }
                },
            }
        )
    )
    patch_dir = source / "patches" / "tt-metal"
    patch_dir.mkdir(parents=True)
    (patch_dir / "gdn-productization-v1.json").write_text("{}\n")
    (patch_dir / "gdn-productization-v2.json").write_text('{"version": 2}\n')
    tools_dir = source / "tools"
    tools_dir.mkdir()
    (tools_dir / "tt_metal_patchset.py").write_text("# fixture\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return source, commit


def test_build_wrapper_exports_clean_exact_commit_as_named_context(tmp_path):
    source, commit = _git_source(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "docker-args"
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$CAPTURE"\n')
    docker.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
    }
    subprocess.run(
        [
            "bash",
            str(BUILD_SCRIPT),
            "--base-image",
            f"example.invalid/ttis@sha256:{'1' * 64}",
            "--quetzal-source",
            str(source),
            "--quetzal-commit",
            commit,
            "--tt-metal-patchset",
            "v2",
            "--tag",
            "example.invalid/quetzal:test",
        ],
        check=True,
        env=env,
    )
    args = capture.read_text().splitlines()
    assert args[:3] == ["buildx", "build", "--load"]
    context = args[args.index("--build-context") + 1]
    assert context.startswith("quetzal_src=")
    contexts = [
        args[index + 1]
        for index, value in enumerate(args)
        if value == "--build-context"
    ]
    assert any(value.startswith("ttis_src=") for value in contexts)
    assert "TT_INFERENCE_SERVER_COMMIT_SHA=" in "\n".join(args)
    assert "TT_METAL_BASE_REVISION=b534549300fe2af11e6ee828675294bc0e359555" in args
    assert "TT_METAL_BASE_FETCH_REF=qz/mixtral-epd2-wait-min-20260827" in args
    expected_patchset = hashlib.sha256(b'{"version": 2}\n').hexdigest()
    assert f"TT_METAL_PATCHSET_SHA256={expected_patchset}" in args
    assert f"TT_METAL_PATCHSET_MANIFEST_SHA256={expected_patchset}" in args
    # The wrapper cleans the ephemeral export after the build command returns.
    assert not Path(context.split("=", 1)[1]).exists()


def test_build_wrapper_rejects_dirty_or_wrong_commit_source(tmp_path):
    source, commit = _git_source(tmp_path)
    common = [
        "bash",
        str(BUILD_SCRIPT),
        "--base-image",
        f"example.invalid/ttis@sha256:{'1' * 64}",
        "--quetzal-source",
        str(source),
        "--tag",
        "unused",
    ]
    (source / "untracked").write_text("dirty")
    result = subprocess.run(
        common + ["--quetzal-commit", commit], text=True, capture_output=True
    )
    assert result.returncode == 2
    assert "no tracked, staged, or untracked changes" in result.stderr
    (source / "untracked").unlink()
    result = subprocess.run(
        common + ["--quetzal-commit", "a" * 40], text=True, capture_output=True
    )
    assert result.returncode == 2
    assert "does not match" in result.stderr
