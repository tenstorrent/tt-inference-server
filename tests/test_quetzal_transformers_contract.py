# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIN = "transformers==5.15.0"


def test_server_and_models_ci_clients_share_quetzal_transformers_pin() -> None:
    plugin = (ROOT / "tt-vllm-plugin" / "pyproject.toml").read_text()
    workflow = (ROOT / "requirements" / "workflow-run-script.txt").read_text()
    vllm_override = (ROOT / "requirements" / "llm-vllm-overrides.txt").read_text()
    assert PIN in plugin
    assert PIN in workflow
    assert PIN in vllm_override


def test_quetzal_image_rejects_frameworks_without_target_configs() -> None:
    for name in (
        "vllm.tt-metal.src.dev.Dockerfile",
        "vllm.tt-metal.src.quetzal.Dockerfile",
    ):
        dockerfile = (ROOT / "vllm-tt-metal" / name).read_text()
        assert "--requirements-output /tmp/quetzal-serve-requirements.txt" in dockerfile
        assert "--requirements /tmp/quetzal-serve-requirements.txt" in dockerfile
        assert "--upgrade --no-deps --requirements" in dockerfile
        assert (
            dockerfile.count(
                'uv pip check --python "${PYTHON_ENV_DIR}/bin/python"'
            )
            >= 3
        )
        assert (
            'test -z "$(comm -13 /tmp/pip-check.base /tmp/pip-check.qualified)"'
            in dockerfile
        )
        assert "cmp /tmp/pip-check.qualified /tmp/pip-check.after" in dockerfile
    derivative = (
        ROOT / "vllm-tt-metal" / "vllm.tt-metal.src.quetzal.Dockerfile"
    ).read_text()
    assert "m.version('transformers') == '5.15.0'" in derivative
    assert "{'gemma4', 'qwen3_5', 'qwen3_5_moe'}" in derivative
