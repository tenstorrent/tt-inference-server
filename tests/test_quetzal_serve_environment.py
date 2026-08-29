# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
from pathlib import Path

import pytest

from scripts.validate_quetzal_serve_environment import (
    LEGACY_QWEN_SOURCE_REVISION,
    ContractError,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _source(tmp_path: Path, *, numpy: str, transformers: str = "4.55.0") -> Path:
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
        "transformers": "4.55.0",
    }


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
