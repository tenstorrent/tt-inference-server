# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path

import pytest

from workflows.model_spec import (
    DeviceModelSpec,
    ImplSpec,
    ModelSpec,
    ModelSpecTemplate,
)
from workflows.workflow_types import DeviceTypes, InferenceEngine


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def sample_impl() -> ImplSpec:
    return ImplSpec(
        impl_id="test_impl",
        impl_name="test-impl",
        repo_url="https://github.com/test/repo",
        code_path="models/test",
    )


def _make_template(
    *,
    impl: ImplSpec,
    weights,
    inference_engine: str,
    device: DeviceTypes,
    docker_image: str = "ghcr.io/test/image:1.0.0",
    template_env_vars=None,
    device_env_vars=None,
    min_ram_gb=None,
) -> ModelSpecTemplate:
    return ModelSpecTemplate(
        weights=list(weights),
        impl=impl,
        tt_metal_commit="abc1234",
        vllm_commit="def5678" if inference_engine == InferenceEngine.VLLM.value else None,
        inference_engine=inference_engine,
        device_model_specs=[
            DeviceModelSpec(
                device=device,
                max_concurrency=16,
                max_context=8192,
                default_impl=True,
                env_vars=dict(device_env_vars or {}),
            )
        ],
        env_vars=dict(template_env_vars or {}),
        docker_image=docker_image,
        min_ram_gb=min_ram_gb,
    )


@pytest.fixture
def vllm_spec(sample_impl) -> ModelSpec:
    template = _make_template(
        impl=sample_impl,
        weights=["acme/Llama-3.1-8B-Instruct"],
        inference_engine=InferenceEngine.VLLM.value,
        device=DeviceTypes.GALAXY,
        docker_image="ghcr.io/acme/vllm-server:0.11.1-bac8b34",
        template_env_vars={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
    )
    return template.expand_to_specs()[0]


@pytest.fixture
def media_spec(sample_impl) -> ModelSpec:
    template = _make_template(
        impl=sample_impl,
        weights=["openai/whisper-large-v3"],
        inference_engine=InferenceEngine.MEDIA.value,
        device=DeviceTypes.GALAXY,
        docker_image="ghcr.io/acme/media-server:0.11.1-bac8b34",
        min_ram_gb=6,
    )
    return template.expand_to_specs()[0]


@pytest.fixture
def forge_spec(sample_impl) -> ModelSpec:
    template = _make_template(
        impl=sample_impl,
        weights=["microsoft/resnet-50"],
        inference_engine=InferenceEngine.FORGE.value,
        device=DeviceTypes.T3K,
        docker_image="ghcr.io/acme/forge-server:0.11.1-bac8b34",
        min_ram_gb=6,
    )
    return template.expand_to_specs()[0]


@pytest.fixture
def spec_no_ram(sample_impl) -> ModelSpec:
    template = _make_template(
        impl=sample_impl,
        weights=["acme/no-ram-model"],
        inference_engine=InferenceEngine.VLLM.value,
        device=DeviceTypes.N150,
        docker_image="ghcr.io/acme/vllm-server:0.11.1",
    )
    spec = template.expand_to_specs()[0]
    object.__setattr__(spec, "min_ram_gb", None)
    return spec
