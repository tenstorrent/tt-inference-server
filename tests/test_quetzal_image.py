import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "vllm-tt-metal" / "vllm.tt-metal.src.quetzal.Dockerfile"
BUILD_SCRIPT = ROOT / "scripts" / "build_quetzal_dev_image.sh"


def test_quetzal_derivative_keeps_third_runtime_out_of_standard_image_identity():
    source = DOCKERFILE.read_text()
    assert "ARG TT_INFERENCE_SERVER_BASE_IMAGE" in source
    assert "FROM ${TT_INFERENCE_SERVER_BASE_IMAGE}" in source
    assert "ARG TT_QUETZAL_COMMIT_SHA" in source
    assert "org.opencontainers.image.quetzal.revision" in source
    assert "@sha256:[0-9a-f]{64}" in source
    assert "^[0-9a-f]{40}$" in source


def test_quetzal_is_installed_non_editably_and_entry_point_is_verified():
    source = DOCKERFILE.read_text()
    install_line = next(line for line in source.splitlines() if "uv pip install" in line)
    assert "--no-deps" in install_line
    assert " -e " not in install_line
    assert "quetzal_model_registry" in source
    assert "tt_quetzalcoatlus.vllm_plugin:register" in source
    assert "import serving.artifact_bundle" in source
    assert 'test "${resolved_commit}" = "${TT_QUETZAL_COMMIT_SHA}"' in source


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
