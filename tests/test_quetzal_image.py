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
    install_line = next(line for line in source.splitlines() if "uv pip install" in line)
    assert " -e " not in install_line
    assert "> /tmp/pip-check.before" in source
    assert "> /tmp/pip-check.after" in source
    assert "sed -E '/^Using Python /d;/^Checked [0-9]+ packages in /d'" in source
    assert "cmp /tmp/pip-check.before /tmp/pip-check.after" in source
    assert "quetzal_model_registry" in source
    assert "tt_quetzalcoatlus.vllm_plugin:register" in source
    assert "import serving.artifact_bundle" in source
    assert 'cat /tmp/quetzal-source/.tt-quetzal-commit' in source
    assert "COPY --from=quetzal_src" in source
    assert "COPY --from=ttis_src" in source
    assert "vllm-tt-metal/src/run_vllm_api_server.py" in source
    assert "/home/container_app_user/app/src/run_vllm_api_server.py" in source
    assert "model_spec.json" in source
    assert "/home/container_app_user/model_specs/model_spec.json" in source
    assert "validate_quetzal_runtime_contract" in source
    assert "git fetch" not in source
    assert "git clone" not in source


def test_quetzal_runner_skips_native_registration_and_validates_package():
    source = RUNNER.read_text()
    assert 'if impl_id == "quetzal":' in source
    assert 'Skipping native TT model registration for impl=quetzal' in source
    assert "validate_quetzal_runtime_contract(model_spec)" in source
    assert source.index('if impl_id == "quetzal":') < source.index(
        "validate_quetzal_runtime_contract(model_spec)"
    )


def test_build_wrapper_requires_digest_base_and_full_quetzal_commit():
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


def test_build_wrapper_refuses_mutable_base_before_invoking_docker():
    result = subprocess.run(
        [
            "bash", str(BUILD_SCRIPT),
            "--base-image", "example.invalid/ttis:latest",
            "--quetzal-commit", "a" * 40,
            "--tag", "example.invalid/quetzal:test",
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
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
                   check=True)
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            check=True, text=True, capture_output=True).stdout.strip()
    return source, commit


def test_build_wrapper_exports_clean_exact_commit_as_named_context(tmp_path):
    source, commit = _git_source(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "docker-args"
    docker = bin_dir / "docker"
    docker.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n")
    docker.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "CAPTURE": str(capture)}
    subprocess.run([
        "bash", str(BUILD_SCRIPT),
        "--base-image", f"example.invalid/ttis@sha256:{'1' * 64}",
        "--quetzal-source", str(source), "--quetzal-commit", commit,
        "--tag", "example.invalid/quetzal:test",
    ], check=True, env=env)
    args = capture.read_text().splitlines()
    assert args[:3] == ["buildx", "build", "--load"]
    context = args[args.index("--build-context") + 1]
    assert context.startswith("quetzal_src=")
    contexts = [
        args[index + 1] for index, value in enumerate(args)
        if value == "--build-context"
    ]
    assert any(value.startswith("ttis_src=") for value in contexts)
    assert "TT_INFERENCE_SERVER_COMMIT_SHA=" in "\n".join(args)
    # The wrapper cleans the ephemeral export after the build command returns.
    assert not Path(context.split("=", 1)[1]).exists()


def test_build_wrapper_rejects_dirty_or_wrong_commit_source(tmp_path):
    source, commit = _git_source(tmp_path)
    common = ["bash", str(BUILD_SCRIPT),
              "--base-image", f"example.invalid/ttis@sha256:{'1' * 64}",
              "--quetzal-source", str(source), "--tag", "unused"]
    (source / "untracked").write_text("dirty")
    result = subprocess.run(common + ["--quetzal-commit", commit], text=True,
                            capture_output=True)
    assert result.returncode == 2
    assert "no tracked, staged, or untracked changes" in result.stderr
    (source / "untracked").unlink()
    result = subprocess.run(common + ["--quetzal-commit", "a" * 40], text=True,
                            capture_output=True)
    assert result.returncode == 2
    assert "does not match" in result.stderr
