# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.validate_quetzal_serve_environment as environment_validator
from scripts.validate_quetzal_serve_environment import (
    LEGACY_QWEN_SOURCE_REVISION,
    ContractError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CURRENT_QWEN_SOURCE_REVISION = "374e94bfa4d742e0a4991683d7ccf4330b7cec3c"


def _source(
    tmp_path: Path,
    *,
    numpy: str,
    transformers: str = "5.15.0",
    installation_dependencies: dict[str, str] | None = None,
) -> Path:
    source = tmp_path / "quetzal"
    contract = source / "serving" / "qualified_environments.json"
    contract.parent.mkdir(parents=True)
    contract.write_text(
        json.dumps(
            {
                "schema": "quetzal.qualified-environments.v2",
                "base": {"dependencies": {"transformers": transformers}},
                "variants": {
                    "qwen36.serve": {
                        "model_ids": ["Qwen/Qwen3.6-27B"],
                        "lane": "serve",
                        "overrides": {"numpy": numpy},
                        "installation_dependencies": installation_dependencies or {},
                    }
                },
            }
        )
    )
    return source


def test_accepts_profile_inside_official_plugin_contract(tmp_path: Path) -> None:
    receipt = validate_contract(
        _source(tmp_path, numpy="1.26.4"),
        "a" * 40,
        plugin_project=ROOT / "tt-vllm-plugin" / "pyproject.toml",
    )
    assert receipt["status"] == "qualified"
    assert receipt["profile"] == "qwen36.serve"
    assert receipt["dependencies"] == {
        "numpy": "1.26.4",
        "transformers": "5.15.0",
    }


def test_accepts_multiple_model_profiles_with_one_exact_image_environment(
    tmp_path: Path,
) -> None:
    source = _source(
        tmp_path,
        numpy="1.26.4",
        installation_dependencies={"click": "8.4.2"},
    )
    contract_path = source / "serving" / "qualified_environments.json"
    contract = json.loads(contract_path.read_text())
    contract["variants"]["gpt_oss_120b.serve"] = {
        "model_ids": ["openai/gpt-oss-120b"],
        "lane": "serve",
        "overrides": {"numpy": "1.26.4"},
        "installation_dependencies": {"click": "8.4.2"},
    }
    contract_path.write_text(json.dumps(contract))

    receipt = validate_contract(
        source,
        "a" * 40,
        plugin_project=ROOT / "tt-vllm-plugin" / "pyproject.toml",
    )

    assert receipt["profiles"] == ["gpt_oss_120b.serve", "qwen36.serve"]
    assert receipt["model_ids"] == ["Qwen/Qwen3.6-27B", "openai/gpt-oss-120b"]
    assert receipt["dependencies"] == {
        "numpy": "1.26.4",
        "transformers": "5.15.0",
    }
    assert receipt["installation_dependencies"] == {"click": "8.4.2"}


def test_rejects_ambiguous_multi_profile_image_environment(tmp_path: Path) -> None:
    source = _source(tmp_path, numpy="1.26.4")
    contract_path = source / "serving" / "qualified_environments.json"
    contract = json.loads(contract_path.read_text())
    contract["variants"]["gpt_oss_120b.serve"] = {
        "model_ids": ["openai/gpt-oss-120b"],
        "lane": "serve",
        "overrides": {"numpy": "1.25.2"},
    }
    contract_path.write_text(json.dumps(contract))

    with pytest.raises(ContractError, match="one exact image environment"):
        validate_contract(
            source,
            "a" * 40,
            plugin_project=ROOT / "tt-vllm-plugin" / "pyproject.toml",
        )


def test_cli_emits_sorted_exact_requirements_after_contract_validation(
    tmp_path: Path,
) -> None:
    source = _source(
        tmp_path,
        numpy="1.26.4",
        installation_dependencies={"click": "8.4.2"},
    )
    requirements = tmp_path / "qualified.txt"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_quetzal_serve_environment.py"),
            "--source",
            str(source),
            "--source-revision",
            "a" * 40,
            "--plugin-project",
            str(ROOT / "tt-vllm-plugin" / "pyproject.toml"),
            "--requirements-output",
            str(requirements),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert requirements.read_text() == (
        "click==8.4.2\nnumpy==1.26.4\ntransformers==5.15.0\n"
    )


def test_installation_dependency_cannot_replace_qualified_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(ContractError, match="duplicates a qualified dependency"):
        validate_contract(
            _source(
                tmp_path,
                numpy="1.26.4",
                installation_dependencies={"numpy": "1.26.4"},
            ),
            "a" * 40,
            plugin_project=ROOT / "tt-vllm-plugin" / "pyproject.toml",
        )


def test_cli_refuses_to_install_unqualified_legacy_environment(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    source.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_quetzal_serve_environment.py"),
            "--source",
            str(source),
            "--source-revision",
            LEGACY_QWEN_SOURCE_REVISION,
            "--plugin-project",
            str(ROOT / "tt-vllm-plugin" / "pyproject.toml"),
            "--requirements-output",
            str(tmp_path / "must-not-exist.txt"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "no exact qualified requirements" in completed.stderr


def test_cli_needs_no_tomllib_or_tomli_on_builder_python(tmp_path: Path) -> None:
    source = tmp_path / "legacy-source"
    source.mkdir()
    block_toml_imports = """
import builtins
import sys

real_import = builtins.__import__
for module in ("tomllib", "tomli"):
    sys.modules.pop(module, None)

def blocked_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"tomllib", "tomli"}:
        raise ModuleNotFoundError(f"{name} blocked for builder test")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
"""

    unavailable = subprocess.run(
        [sys.executable, "-c", block_toml_imports + "\nimport tomllib"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert unavailable.returncode != 0
    assert "tomllib blocked for builder test" in unavailable.stderr

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            block_toml_imports
            + "\nimport runpy\nsys.argv = sys.argv[1:]\n"
            + "runpy.run_path(sys.argv[0], run_name='__main__')",
            str(ROOT / "scripts" / "validate_quetzal_serve_environment.py"),
            "--source",
            str(source),
            "--source-revision",
            LEGACY_QWEN_SOURCE_REVISION,
            "--plugin-project",
            str(ROOT / "tt-vllm-plugin" / "pyproject.toml"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "legacy-pinned"


def test_dependency_parser_fails_closed_on_missing_comma(tmp_path: Path) -> None:
    plugin_project = tmp_path / "pyproject.toml"
    plugin_project.write_text(
        """
[project]
dependencies = [
    "numpy>=1.24.4,<2"
    "transformers==5.15.0",
]
""".lstrip()
    )
    with pytest.raises(ContractError, match="missing a comma"):
        validate_contract(
            _source(tmp_path, numpy="1.26.4"),
            "d" * 40,
            plugin_project=plugin_project,
        )


def test_rejects_numpy_2_profile_before_image_build(tmp_path: Path) -> None:
    with pytest.raises(
        ContractError,
        match=r"qwen36\.serve pins numpy==2\.3\.5, outside tt-vllm-plugin",
    ):
        validate_contract(
            _source(tmp_path, numpy="2.3.5", transformers="5.15.0"),
            "b" * 40,
            plugin_project=ROOT / "tt-vllm-plugin" / "pyproject.toml",
        )


def test_only_exact_known_good_legacy_source_can_omit_profile(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    source.mkdir()
    receipt = validate_contract(
        source,
        LEGACY_QWEN_SOURCE_REVISION,
        plugin_project=ROOT / "tt-vllm-plugin" / "pyproject.toml",
    )
    assert receipt["status"] == "legacy-pinned"

    with pytest.raises(ContractError, match="no qualified serving environment"):
        validate_contract(
            source,
            "c" * 40,
            plugin_project=ROOT / "tt-vllm-plugin" / "pyproject.toml",
        )


def test_check_installed_uses_upstream_distribution_name_and_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "legacy"
    source.mkdir()
    python_prefix = tmp_path / "python_env"
    site_packages = python_prefix / "lib" / "python3.10" / "site-packages"
    site_packages.mkdir(parents=True)

    class InstalledPlugin:
        version = "0.1.0"

        @staticmethod
        def locate_file(_path: str) -> Path:
            return site_packages

    def distribution(name: str) -> InstalledPlugin:
        assert name == "vllm-tt-plugin"
        return InstalledPlugin()

    def version(name: str) -> str:
        assert name == "numpy"
        return "1.26.4"

    monkeypatch.setattr(environment_validator.sys, "prefix", str(python_prefix))
    monkeypatch.setattr(
        environment_validator.importlib.metadata, "distribution", distribution
    )
    monkeypatch.setattr(environment_validator.importlib.metadata, "version", version)

    receipt = validate_contract(
        source,
        LEGACY_QWEN_SOURCE_REVISION,
        plugin_project=ROOT / "tt-vllm-plugin" / "pyproject.toml",
        check_installed=True,
    )
    assert receipt["installed_plugin"] == {
        "version": "0.1.0",
        "python_prefix": str(python_prefix),
        "metadata_root": str(site_packages),
    }


def test_qwen_catalog_binds_candidate_v2_source_and_keeps_package_payload() -> None:
    catalog = (ROOT / "workflows" / "model_specs" / "dev" / "llm.yaml").read_text()
    assert (
        f"QUETZAL_REQUIRED_SOURCE_REVISION: {CURRENT_QWEN_SOURCE_REVISION}" in catalog
    )
    assert (
        "QUETZAL_PACKAGE_ID: sha256-"
        "f1d6cebaf6cd432c78721ec3b81101ab86493f387b37f63bc11aca2fc6f6d8d8-"
        "0a8efa103ee378c7cd0e2fa25b0426cbb82752e270f8927bdf44eb2cfe68ce66"
    ) in catalog


def test_docker_builds_validate_before_installing_quetzal() -> None:
    for relative in (
        "vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile",
        "vllm-tt-metal/vllm.tt-metal.src.quetzal.Dockerfile",
    ):
        dockerfile = (ROOT / relative).read_text()
        normalized = dockerfile.replace(
            '"${PYTHON_ENV_DIR}/bin/python"', "${PYTHON_ENV_DIR}/bin/python"
        )
        validator_command = (
            "${PYTHON_ENV_DIR}/bin/python /tmp/validate_quetzal_serve_environment.py"
        )
        validator = normalized.index(validator_command)
        install = (
            normalized.index("--no-deps /tmp/quetzal-source", validator)
            if relative.endswith("src.quetzal.Dockerfile")
            else normalized.index(
                '--no-cache-dir --no-deps "${quetzal_wheel}"', validator
            )
        )
        assert validator < install
        assert "--plugin-project /tmp/ttis-vllm-plugin-pyproject.toml" in dockerfile
        qualified_install = normalized.index(
            "--requirements /tmp/quetzal-serve-requirements.txt", validator
        )
        installed_validator = normalized.index("--check-installed", qualified_install)
        assert validator < qualified_install < installed_validator < install
        assert "uv pip check --python" in normalized[qualified_install:install]
        assert (
            'test -z "$(comm -13 /tmp/pip-check.base /tmp/pip-check.qualified)"'
            in normalized
        )
        assert "cmp /tmp/pip-check.qualified /tmp/pip-check.after" in normalized
        assert "--upgrade --no-deps --requirements" in normalized
    derivative = (
        ROOT / "vllm-tt-metal" / "vllm.tt-metal.src.quetzal.Dockerfile"
    ).read_text()
    assert (
        'uv pip install --python "${PYTHON_ENV_DIR}/bin/python" --no-deps '
        "/tmp/quetzal-source"
    ) in derivative


def test_dev_image_installer_and_validator_share_explicit_python_prefix() -> None:
    dockerfile = (
        ROOT / "vllm-tt-metal" / "vllm.tt-metal.src.dev.Dockerfile"
    ).read_text()
    assert "export VIRTUAL_ENV=${PYTHON_ENV_DIR}" in dockerfile
    assert "export PATH=${PYTHON_ENV_DIR}/bin:\\${PATH}" in dockerfile
    assert "export UV_PYTHON=${PYTHON_ENV_DIR}/bin/python" in dockerfile
    assert (
        "uv pip install --python ${PYTHON_ENV_DIR}/bin/python --upgrade pip"
        in dockerfile
    )
    assert (
        '${PYTHON_ENV_DIR}/bin/python -c \\"import importlib.metadata as m; '
        "assert m.distribution('vllm-tt-plugin').version\\\"" in dockerfile
    )


def test_official_builder_selects_dependency_qualified_dev_image() -> None:
    builder = (ROOT / "scripts" / "build_single_docker.sh").read_text()
    dockerfile_name = "vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile"
    assert f"-f {dockerfile_name}" in builder
    dockerfile = (ROOT / dockerfile_name).read_text()
    requirements = dockerfile.index(
        "--requirements-output /tmp/quetzal-serve-requirements.txt"
    )
    install = dockerfile.index(
        "--upgrade --no-deps --requirements /tmp/quetzal-serve-requirements.txt",
        requirements,
    )
    check_installed = dockerfile.index("--check-installed", install)
    wheel = dockerfile.index("uv build --wheel", check_installed)
    assert requirements < install < check_installed < wheel


def test_official_builder_uses_exact_tt_metal_tool_images() -> None:
    builder = (ROOT / "scripts" / "build_single_docker.sh").read_text()
    checkout = builder.index('git checkout "${RESOLVED_TT_METAL_COMMIT}"')
    tags = builder.index(
        ".github/scripts/compute-tool-tags.sh tenstorrent/tt-metal", checkout
    )
    tools = builder.index(".github/scripts/get-target-tools.sh ci-build", tags)
    validation = builder.index(
        "^ghcr\\.io/tenstorrent/tt-metal/tt-metalium/tools/${tool}:", tools
    )
    context = builder.index(
        "ci-build.contexts.${tool}-layer=docker-image://${tool_tag}", validation
    )
    bake = builder.index("docker buildx bake", context)
    assert checkout < tags < tools < validation < context < bake
    assert '"${TT_METAL_TOOL_CONTEXT_ARGS[@]}"' in builder[bake:]


def test_official_builder_does_not_bypass_runner_network_isolation() -> None:
    builder = (ROOT / "scripts" / "build_single_docker.sh").read_text()
    executable_lines = [
        line for line in builder.splitlines() if not line.lstrip().startswith("#")
    ]
    assert not any("network=host" in line for line in executable_lines)


def test_official_builder_adds_bounded_apt_retries() -> None:
    builder = (ROOT / "scripts" / "build_single_docker.sh").read_text()
    checkout = builder.index('git checkout "${RESOLVED_TT_METAL_COMMIT}"')
    patch = builder.index("patch_tt_metal_builder_apt_retries.py", checkout)
    bake = builder.index("docker buildx bake", patch)
    assert checkout < patch < bake


def test_tt_metal_builder_apt_retry_patch_is_fail_closed_and_idempotent(
    tmp_path: Path,
) -> None:
    from scripts.patch_tt_metal_builder_apt_retries import MARKER, patch

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        'FROM ubuntu:22.04\nENV UV_PYTHON_INSTALL_DIR="/usr/local/share/uv"\n'
        "RUN apt-get update\n"
    )
    assert patch(dockerfile) is True
    patched = dockerfile.read_text()
    assert patched.count(MARKER) == 1
    assert 'Acquire::Retries "10";' in patched
    assert 'Acquire::http::Timeout "30";' in patched
    assert patch(dockerfile) is False
    assert dockerfile.read_text() == patched

    unsupported = tmp_path / "UnsupportedDockerfile"
    unsupported.write_text("FROM ubuntu:22.04\n")
    with pytest.raises(ValueError, match="expected exactly one"):
        patch(unsupported)


def test_official_builder_retries_with_a_fresh_tt_metal_checkout() -> None:
    builder = (ROOT / "scripts" / "build_single_docker.sh").read_text()
    assert 'TT_METAL_BUILD_DIR=""' in builder
    assert (
        "TT_METAL_BUILD_DIR=$(mktemp -d "
        '"${repo_root}/temp_docker_build_dir.XXXXXX")' in builder
    )
    cleanup = builder.index('if [[ -n "$TT_METAL_BUILD_DIR" ]]')
    clone = builder.index(
        "git clone --depth 1 https://github.com/tenstorrent/tt-metal.git"
    )
    assert cleanup < clone
    assert 'rm -rf -- "$TT_METAL_BUILD_DIR"' in builder[cleanup:clone]
    assert 'tt_metal_build_dir="temp_docker_build_dir_' not in builder


def test_official_builder_reuses_an_existing_final_image_before_base_build() -> None:
    builder = (ROOT / "scripts" / "build_single_docker.sh").read_text()
    remote_probe = builder.index('image_exists_remote "${dev_image_tag}"')
    build_flag = builder.index("build_dev_image=false", remote_probe)
    base_build = builder.index(
        'if [ "$build_dev_image" = true ] && ! image_exists_local '
        '"${TT_METAL_DOCKERFILE_URL}"',
        build_flag,
    )
    assert remote_probe < build_flag < base_build
    assert "TTIS_IMAGE_RESULT dev_image_tag=${dev_image_tag}" in builder


def test_official_builder_rejects_an_invalid_container_uid() -> None:
    builder = (ROOT / "scripts" / "build_single_docker.sh").read_text()
    invalid_uid = builder.index(
        "Error: CONTAINER_APP_UID=${CONTAINER_APP_UID} is not a number"
    )
    assert "exit 1" in builder[invalid_uid : invalid_uid + 240]
