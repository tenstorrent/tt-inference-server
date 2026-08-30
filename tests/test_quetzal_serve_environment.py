# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_quetzal_serve_environment import (
    LEGACY_QWEN_SOURCE_REVISION,
    ContractError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _source(tmp_path: Path, *, numpy: str, transformers: str = "5.15.0") -> Path:
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


def test_qwen_catalog_keeps_known_good_source_and_package() -> None:
    catalog = (ROOT / "workflows" / "model_specs" / "dev" / "llm.yaml").read_text()
    assert f"QUETZAL_REQUIRED_SOURCE_REVISION: {LEGACY_QWEN_SOURCE_REVISION}" in catalog
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
        validator = dockerfile.index(
            "python /tmp/validate_quetzal_serve_environment.py"
        )
        install = dockerfile.index("pip install", validator)
        assert validator < install
    derivative = (
        ROOT / "vllm-tt-metal" / "vllm.tt-metal.src.quetzal.Dockerfile"
    ).read_text()
    assert "uv pip install --no-deps /tmp/quetzal-source" in derivative
